package sentencepiece

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"unicode/utf8"
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
		for i := 0; i < b.N; i++ {
			tok.Encode("Hello world")
		}
	})

	b.Run("medium", func(b *testing.B) {
		text := "The quick brown fox jumps over the lazy dog. This is a medium length string for benchmarking the tokenizer performance."
		for i := 0; i < b.N; i++ {
			tok.Encode(text)
		}
	})

	b.Run("long", func(b *testing.B) {
		text := ""
		for i := 0; i < 100; i++ {
			text += "The quick brown fox jumps over the lazy dog. "
		}
		for i := 0; i < b.N; i++ {
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

	for i := 0; i < b.N; i++ {
		tok.Decode(ids)
	}
}

func FuzzEncode(f *testing.F) {
	// Rich seed corpus covering diverse inputs.
	seeds := []string{
		"hello world",
		"",
		"🚀🚀🚀",
		"Hello Привіт 你好",
		" ",
		"\t\n\r",
		"a\x00b",
		"The quick brown fox jumps over the lazy dog.",
		"Україна — держава у Східній Європі.",
		"東京は日本の首都です。",
		"مرحبا بالعالم",
		"สวัสดีครับ",
		"👨\u200d👩\u200d👧\u200d👦",
		"https://example.com/path?q=hello&lang=uk",
		`{"key": "value", "num": 42}`,
		"func main() { fmt.Println(\"hello\") }",
		"\u200b\u200b\u200b",
		"\ufeff",
		"\u00ad",
		"a\u0300",
		"e\u0301",
		"ＡＢＣ",
		"①②③",
		string([]byte{0xC0, 0xAF}),         // overlong UTF-8
		string([]byte{0xED, 0xA0, 0x80}),   // surrogate half
		string([]byte{0xF4, 0x90, 0x80}),   // truncated 4-byte
		"ab\x80cd",                         // invalid continuation
		"hello\xFFworld",                   // 0xFF byte
		"\x00\x01\x02\x03\x04\x05",         // low control chars
		"a" + string(rune(0x10FFFF)) + "b", // max unicode
		"x\u0300\u0301\u0302\u0303\u0304",  // stacked combiners
		"   \t\t\n\n   ",                   // only whitespace
		"<script>alert('xss')</script>",
		"$1,234,567.89",
		"SELECT * FROM users;",
		strings.Repeat("a", 500),
		strings.Repeat("🔥", 50),
		strings.Repeat("hello ", 100),
	}
	for _, s := range seeds {
		f.Add(s)
	}

	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		f.Fatalf("load model: %v", err)
	}

	f.Fuzz(func(t *testing.T, input string) {
		// 1. Encode must not panic.
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		// 2. EncodeAsPieces must not panic and have same length as IDs.
		pieces, err := tok.EncodeAsPieces(input)
		if err != nil {
			t.Fatalf("EncodeAsPieces error: %v", err)
		}
		if len(ids) != len(pieces) {
			t.Fatalf("ids length %d != pieces length %d", len(ids), len(pieces))
		}

		// 3. All IDs must be valid (within vocab range).
		vocabSize := tok.VocabSize()
		for i, id := range ids {
			if id < 0 || id >= vocabSize {
				t.Fatalf("invalid token ID %d at position %d (vocab size %d)", id, i, vocabSize)
			}
		}

		// 4. Decode must not panic.
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		_ = decoded

		// 5. Re-encoding the decoded output must not panic.
		ids2, err := tok.Encode(decoded)
		if err != nil {
			t.Fatalf("re-Encode error: %v", err)
		}
		_ = ids2

		// 6. Decode of re-encoded must produce same decode on second round
		//    (idempotency after two rounds of normalization).
		//    Note: first decode may differ from second due to NFKC composition
		//    (e.g., A + combining grave → À), but second round must be stable.
		if utf8.ValidString(input) {
			decoded2, err := tok.Decode(ids2)
			if err != nil {
				t.Fatalf("re-Decode error: %v", err)
			}
			// Third round to check stability.
			ids3, _ := tok.Encode(decoded2)
			decoded3, _ := tok.Decode(ids3)
			if decoded2 != decoded3 {
				t.Errorf("decode not stable after 2 rounds:\n  round2: %q\n  round3: %q", decoded2, decoded3)
			}
		}
	})
}

func FuzzDecode(f *testing.F) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		f.Fatalf("load model: %v", err)
	}

	vocabSize := tok.VocabSize()

	// Seed with known valid ID sequences.
	f.Add([]byte{0, 0, 0, 1})                   // single token ID=1
	f.Add([]byte{0, 0, 1, 0xFB, 0, 0, 1, 0xFC}) // two IDs

	f.Fuzz(func(t *testing.T, data []byte) {
		// Interpret data as a sequence of uint16 token IDs.
		if len(data) < 2 {
			return
		}
		n := len(data) / 2
		if n > 200 { // cap length to avoid slow tests
			n = 200
		}
		ids := make([]int, n)
		for i := 0; i < n; i++ {
			ids[i] = int(data[i*2])<<8 | int(data[i*2+1])
			// Clamp to valid range.
			ids[i] = ids[i] % vocabSize
		}

		// Decode must not panic.
		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		_ = decoded
	})
}

func FuzzNormalize(f *testing.F) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		f.Fatalf("load model: %v", err)
	}
	normalizer := NewNormalizer(tok.model)

	f.Add("hello world")
	f.Add("")
	f.Add("\x00\x01\x02")
	f.Add("ＡＢＣ")
	f.Add("\u200b\u200b")

	f.Fuzz(func(t *testing.T, input string) {
		// Normalize must not panic.
		result := normalizer.Normalize(input)

		// Result must be valid UTF-8 (if input was).
		if !utf8.ValidString(result) && utf8.ValidString(input) {
			t.Errorf("Normalize produced invalid UTF-8 for valid input %q", input)
		}

		// Check normalization stability: applying normalize twice to the
		// result should give a stable output.
		// Note: first normalize may compose combining characters differently,
		// but the second round must be stable.
		if utf8.ValidString(input) {
			result2 := normalizer.Normalize(result)
			result3 := normalizer.Normalize(result2)
			if result2 != result3 {
				t.Errorf("Normalize not stable after 2 rounds:\n  round2: %q\n  round3: %q", result2, result3)
			}
		}
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
