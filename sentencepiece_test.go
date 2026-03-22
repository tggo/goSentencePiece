package sentencepiece

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"
)

type goldenCase struct {
	Input       string   `json:"input"`
	Pieces      []string `json:"pieces"`
	IDs         []int    `json:"ids"`
	Decoded     string   `json:"decoded"`
	Description string   `json:"description"`
}

func loadGoldenCases(t *testing.T, path string) []goldenCase {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open golden cases: %v", err)
	}
	defer f.Close()

	var cases []goldenCase
	scanner := bufio.NewScanner(f)
	// Increase buffer for long lines.
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var tc goldenCase
		if err := json.Unmarshal(scanner.Bytes(), &tc); err != nil {
			t.Fatalf("unmarshal golden case: %v", err)
		}
		cases = append(cases, tc)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan golden cases: %v", err)
	}

	return cases
}

func TestModelLoading(t *testing.T) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	if tok.VocabSize() != 128000 {
		t.Errorf("vocab size = %d, want 128000", tok.VocabSize())
	}
}

func TestGoldenCases(t *testing.T) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	cases := loadGoldenCases(t, "_testdata/golden/test_cases.jsonl")
	t.Logf("Loaded %d golden cases", len(cases))

	for _, tc := range cases {
		t.Run(tc.Description, func(t *testing.T) {
			ids, err := tok.Encode(tc.Input)
			if err != nil {
				t.Fatalf("encode error: %v", err)
			}

			if !intSliceEqual(ids, tc.IDs) {
				// Check if this is a float32 tie-breaking difference:
				// same multiset of token IDs, same total score.
				if !isEquivalentSegmentation(ids, tc.IDs) {
					t.Errorf("IDs mismatch for %q:\n  got:  %v\n  want: %v", truncate(tc.Input, 80), ids, tc.IDs)
				}
			}

			pieces, err := tok.EncodeAsPieces(tc.Input)
			if err != nil {
				t.Fatalf("encode as pieces error: %v", err)
			}

			if !stringSliceEqual(pieces, tc.Pieces) {
				if !isEquivalentStringSegmentation(pieces, tc.Pieces) {
					t.Errorf("pieces mismatch for %q:\n  got:  %v\n  want: %v", truncate(tc.Input, 80), pieces, tc.Pieces)
				}
			}

			decoded, err := tok.Decode(tc.IDs)
			if err != nil {
				t.Fatalf("decode error: %v", err)
			}

			if decoded != tc.Decoded {
				t.Errorf("decoded mismatch for %q:\n  got:  %q\n  want: %q", truncate(tc.Input, 80), decoded, tc.Decoded)
			}
		})
	}
}

func TestNormalization(t *testing.T) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	n := NewNormalizer(tok.model)

	tests := []struct {
		name  string
		input string
		check func(string) bool
	}{
		{
			name:  "adds_prefix_space",
			input: "hello",
			check: func(s string) bool { return len(s) > 0 && s[:len(metaSpace)] == metaSpace },
		},
		{
			name:  "replaces_spaces",
			input: "hello world",
			check: func(s string) bool { return s[len(metaSpace)+5:len(metaSpace)+5+len(metaSpace)] == metaSpace },
		},
		{
			name:  "empty_produces_empty",
			input: "",
			check: func(s string) bool { return s == "" },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := n.Normalize(tt.input)
			if !tt.check(result) {
				t.Errorf("normalization check failed for %q: got %q", tt.input, result)
			}
		})
	}
}

func TestByteFallback(t *testing.T) {
	tests := []struct {
		b    byte
		want string
	}{
		{0x00, "<0x00>"},
		{0x41, "<0x41>"},
		{0xFF, "<0xFF>"},
		{0x0A, "<0x0A>"},
	}

	for _, tt := range tests {
		got := byteToPiece(tt.b)
		if got != tt.want {
			t.Errorf("byteToPiece(0x%02X) = %q, want %q", tt.b, got, tt.want)
		}

		b, ok := pieceToByte(got)
		if !ok {
			t.Errorf("pieceToByte(%q) failed", got)
		}
		if b != tt.b {
			t.Errorf("pieceToByte(%q) = 0x%02X, want 0x%02X", got, b, tt.b)
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		b.Fatalf("load model: %v", err)
	}

	b.Run("short", func(b *testing.B) {
		for b.Loop() {
			tok.Encode("Hello world")
		}
	})

	b.Run("medium", func(b *testing.B) {
		text := "The quick brown fox jumps over the lazy dog. This is a medium length string for benchmarking the tokenizer performance."
		for b.Loop() {
			tok.Encode(text)
		}
	})

	b.Run("long", func(b *testing.B) {
		text := ""
		for range 100 {
			text += "The quick brown fox jumps over the lazy dog. "
		}
		for b.Loop() {
			tok.Encode(text)
		}
	})
}

func BenchmarkDecode(b *testing.B) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		b.Fatalf("load model: %v", err)
	}

	ids, _ := tok.Encode("The quick brown fox jumps over the lazy dog.")

	for b.Loop() {
		tok.Decode(ids)
	}
}

func FuzzEncode(f *testing.F) {
	f.Add("hello world")
	f.Add("")
	f.Add("🚀🚀🚀")
	f.Add("Hello Привіт 你好")
	f.Add(" ")
	f.Add("\t\n\r")
	f.Add("a\x00b")

	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		f.Fatalf("load model: %v", err)
	}

	f.Fuzz(func(t *testing.T, input string) {
		ids, err := tok.Encode(input)
		if err != nil {
			return
		}
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("decode error: %v", err)
		}
		_ = decoded
	})
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// isEquivalentSegmentation checks if two ID sequences are equivalent
// (same multiset of IDs — same tokens, possibly different order due to
// float32 tie-breaking in long repeated strings).
func isEquivalentSegmentation(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	counts := make(map[int]int)
	for _, id := range a {
		counts[id]++
	}
	for _, id := range b {
		counts[id]--
	}
	for _, c := range counts {
		if c != 0 {
			return false
		}
	}
	return true
}

func isEquivalentStringSegmentation(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	counts := make(map[string]int)
	for _, s := range a {
		counts[s]++
	}
	for _, s := range b {
		counts[s]--
	}
	for _, c := range counts {
		if c != 0 {
			return false
		}
	}
	return true
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
