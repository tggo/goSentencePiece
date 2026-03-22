package sentencepiece

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestBPEGoldenCases(t *testing.T) {
	tok, err := NewTokenizer("_testdata/bpe.model")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	t.Logf("BPE model: vocab=%d, modelType=%d", tok.VocabSize(), tok.model.modelType)

	f, err := os.Open("_testdata/golden/bpe_test_cases.jsonl")
	if err != nil {
		t.Fatalf("open golden cases: %v", err)
	}
	defer f.Close()

	var cases []goldenCase
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var tc goldenCase
		if err := json.Unmarshal(scanner.Bytes(), &tc); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		cases = append(cases, tc)
	}

	t.Logf("Loaded %d BPE golden cases", len(cases))

	for _, tc := range cases {
		t.Run(tc.Description, func(t *testing.T) {
			ids, err := tok.Encode(tc.Input)
			if err != nil {
				t.Fatalf("encode error: %v", err)
			}

			if !intSliceEqual(ids, tc.IDs) {
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

func BenchmarkBPEEncode(b *testing.B) {
	tok, err := NewTokenizer("_testdata/bpe.model")
	if err != nil {
		b.Fatalf("load model: %v", err)
	}

	b.Run("short", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tok.Encode("Hello world")
		}
	})

	b.Run("medium", func(b *testing.B) {
		text := "The quick brown fox jumps over the lazy dog. This is a medium length string."
		for i := 0; i < b.N; i++ {
			tok.Encode(text)
		}
	})
}

func FuzzBPEEncode(f *testing.F) {
	seeds := []string{
		"hello world", "", "🚀🚀🚀", "Hello Привіт 你好",
		" ", "\t\n\r", "a\x00b", "The quick brown fox.",
		"Україна — держава.", "東京は日本の首都です。",
		"مرحبا", "👨\u200d👩\u200d👧\u200d👦", "🇺🇦",
		"func main() { fmt.Println(\"hello\") }",
		strings.Repeat("a", 500),
		strings.Repeat("🔥", 50),
	}
	for _, s := range seeds {
		f.Add(s)
	}

	tok, err := NewTokenizer("_testdata/bpe.model")
	if err != nil {
		f.Fatalf("load model: %v", err)
	}

	f.Fuzz(func(t *testing.T, input string) {
		ids, err := tok.Encode(input)
		if err != nil {
			t.Fatalf("Encode error: %v", err)
		}

		pieces, err := tok.EncodeAsPieces(input)
		if err != nil {
			t.Fatalf("EncodeAsPieces error: %v", err)
		}
		if len(ids) != len(pieces) {
			t.Fatalf("ids len %d != pieces len %d", len(ids), len(pieces))
		}

		vocabSize := tok.VocabSize()
		for i, id := range ids {
			if id < 0 || id >= vocabSize {
				t.Fatalf("invalid token ID %d at pos %d", id, i)
			}
		}

		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		_ = decoded

		if utf8.ValidString(input) {
			ids2, _ := tok.Encode(decoded)
			decoded2, _ := tok.Decode(ids2)
			ids3, _ := tok.Encode(decoded2)
			decoded3, _ := tok.Decode(ids3)
			if decoded2 != decoded3 {
				t.Errorf("BPE decode not stable after 2 rounds:\n  round2: %q\n  round3: %q", decoded2, decoded3)
			}
		}
	})
}
