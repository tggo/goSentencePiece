package sentencepiece

import (
	"os"
	"testing"
)

const hfTokenizerPath = "_testdata/tokenizer.json"

func skipIfNoHFTokenizer(t testing.TB) {
	t.Helper()
	if _, err := os.Stat(hfTokenizerPath); os.IsNotExist(err) {
		t.Skip("tokenizer.json not found (run: python _testdata/download_hf_tokenizer.py)")
	}
}

func TestTokenizerJSONLoad(t *testing.T) {
	skipIfNoHFTokenizer(t)
	tok, err := NewTokenizerFromJSON(hfTokenizerPath)
	if err != nil {
		t.Fatalf("load tokenizer.json: %v", err)
	}

	if tok.VocabSize() != 256000 {
		t.Errorf("vocab size = %d, want 256000", tok.VocabSize())
	}

	m := tok.Model()
	if m.Type() != ModelTypeBPE {
		t.Errorf("model type = %d, want BPE (%d)", m.Type(), ModelTypeBPE)
	}

	// Verify special token IDs from mmBERT tokenizer.json.
	if m.PadID() != 0 {
		t.Errorf("padID = %d, want 0", m.PadID())
	}
	if m.EosID() != 1 {
		t.Errorf("eosID = %d, want 1", m.EosID())
	}
	if m.BosID() != 2 {
		t.Errorf("bosID = %d, want 2", m.BosID())
	}
	if m.UnkID() != 3 {
		t.Errorf("unkID = %d, want 3", m.UnkID())
	}
}

func TestTokenizerJSONAutoDetect(t *testing.T) {
	skipIfNoHFTokenizer(t)
	// NewTokenizer should auto-detect JSON format.
	tok, err := NewTokenizer(hfTokenizerPath)
	if err != nil {
		t.Fatalf("auto-detect load: %v", err)
	}

	if tok.VocabSize() != 256000 {
		t.Errorf("vocab size = %d, want 256000", tok.VocabSize())
	}

	// Existing protobuf path should still work.
	tokPb, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		t.Fatalf("protobuf load: %v", err)
	}
	if tokPb.VocabSize() != 128000 {
		t.Errorf("protobuf vocab size = %d, want 128000", tokPb.VocabSize())
	}
}

func TestTokenizerJSONFromReader(t *testing.T) {
	skipIfNoHFTokenizer(t)
	f, err := os.Open(hfTokenizerPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()

	tok, err := NewTokenizerFromJSONReader(f)
	if err != nil {
		t.Fatalf("load from reader: %v", err)
	}
	if tok.VocabSize() != 256000 {
		t.Errorf("vocab size = %d, want 256000", tok.VocabSize())
	}
}

func TestTokenizerJSONGoldenCases(t *testing.T) {
	skipIfNoHFTokenizer(t)
	tok, err := NewTokenizerFromJSON(hfTokenizerPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	cases := loadGoldenCases(t, "_testdata/golden/hf_test_cases.jsonl")
	t.Logf("Loaded %d HF golden cases", len(cases))

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

			decoded, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("decode error: %v", err)
			}
			if decoded != tc.Decoded {
				t.Errorf("decoded mismatch for %q:\n  got:  %q\n  want: %q", truncate(tc.Input, 80), decoded, tc.Decoded)
			}
		})
	}
}

func TestTokenizerJSONPostProcessor(t *testing.T) {
	skipIfNoHFTokenizer(t)
	tok, err := NewTokenizerFromJSON(hfTokenizerPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	enc := tok.EncodeWithOptions("Hello", true)
	if enc == nil {
		t.Fatal("EncodeWithOptions returned nil")
	}

	// mmBERT post-processor: [BOS] $A [EOS]
	// BOS=2, EOS=1
	if len(enc.IDs) < 3 {
		t.Fatalf("expected at least 3 tokens, got %d: %v", len(enc.IDs), enc.IDs)
	}
	if enc.IDs[0] != 2 {
		t.Errorf("first token = %d, want BOS (2)", enc.IDs[0])
	}
	if enc.IDs[len(enc.IDs)-1] != 1 {
		t.Errorf("last token = %d, want EOS (1)", enc.IDs[len(enc.IDs)-1])
	}
}

func BenchmarkTokenizerJSONLoad(b *testing.B) {
	skipIfNoHFTokenizer(b)
	data, err := os.ReadFile(hfTokenizerPath)
	if err != nil {
		b.Fatalf("read: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loadFromJSON(data)
		if err != nil {
			b.Fatalf("load: %v", err)
		}
	}
}

func BenchmarkTokenizerJSONEncode(b *testing.B) {
	skipIfNoHFTokenizer(b)
	tok, err := NewTokenizerFromJSON(hfTokenizerPath)
	if err != nil {
		b.Fatalf("load: %v", err)
	}

	b.Run("short", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tok.Encode("Hello world")
		}
	})
	b.Run("medium", func(b *testing.B) {
		text := "The transformer architecture revolutionized natural language processing when it was introduced in the landmark paper Attention Is All You Need."
		for i := 0; i < b.N; i++ {
			tok.Encode(text)
		}
	})
}
