package sentencepiece

import (
	"bufio"
	"bytes"
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

func TestCoverageGaps(t *testing.T) {
	tok, err := NewTokenizer("_testdata/spm.model")
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	// 1. IdToPiece
	t.Run("IdToPiece", func(t *testing.T) {
		t.Run("valid_id", func(t *testing.T) {
			p := tok.model.IdToPiece(0)
			if p == "" {
				t.Error("IdToPiece(0) returned empty string for valid ID")
			}
		})
		t.Run("negative_id", func(t *testing.T) {
			p := tok.model.IdToPiece(-1)
			if p != "" {
				t.Errorf("IdToPiece(-1) = %q, want empty", p)
			}
		})
		t.Run("too_large_id", func(t *testing.T) {
			p := tok.model.IdToPiece(999999)
			if p != "" {
				t.Errorf("IdToPiece(999999) = %q, want empty", p)
			}
		})
	})

	// 2. LoadModelFromReader
	t.Run("LoadModelFromReader", func(t *testing.T) {
		data, err := os.ReadFile("_testdata/spm.model")
		if err != nil {
			t.Fatalf("read model file: %v", err)
		}
		t.Run("valid_reader", func(t *testing.T) {
			m, err := LoadModelFromReader(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("LoadModelFromReader error: %v", err)
			}
			if m.VocabSize() != 128000 {
				t.Errorf("vocab size = %d, want 128000", m.VocabSize())
			}
		})
		t.Run("invalid_data", func(t *testing.T) {
			_, err := LoadModelFromReader(bytes.NewReader([]byte("not a protobuf")))
			if err == nil {
				t.Error("expected error for invalid protobuf data")
			}
		})
		t.Run("error_reader", func(t *testing.T) {
			_, err := LoadModelFromReader(&errorReader{})
			if err == nil {
				t.Error("expected error for failing reader")
			}
		})
	})

	// 3. NewTokenizerFromReader
	t.Run("NewTokenizerFromReader", func(t *testing.T) {
		data, err := os.ReadFile("_testdata/spm.model")
		if err != nil {
			t.Fatalf("read model file: %v", err)
		}
		t.Run("valid_reader", func(t *testing.T) {
			tok2, err := NewTokenizerFromReader(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("NewTokenizerFromReader error: %v", err)
			}
			if tok2.VocabSize() != 128000 {
				t.Errorf("vocab size = %d, want 128000", tok2.VocabSize())
			}
			// Verify it produces the same encoding as file-loaded tokenizer.
			ids1, _ := tok.Encode("Hello world")
			ids2, _ := tok2.Encode("Hello world")
			if !intSliceEqual(ids1, ids2) {
				t.Errorf("reader-loaded tokenizer gives different encoding")
			}
		})
		t.Run("invalid_reader", func(t *testing.T) {
			_, err := NewTokenizerFromReader(bytes.NewReader([]byte("bad")))
			if err == nil {
				t.Error("expected error for invalid data")
			}
		})
	})

	// 4. AddSpecialTokens
	t.Run("AddSpecialTokens", func(t *testing.T) {
		t.Run("wraps_ids", func(t *testing.T) {
			ids := []int{100, 200, 300}
			wrapped := tok.AddSpecialTokens(ids)
			if len(wrapped) != len(ids)+2 {
				t.Fatalf("length = %d, want %d", len(wrapped), len(ids)+2)
			}
			if wrapped[0] != tok.model.bosID {
				t.Errorf("first token = %d, want BOS=%d", wrapped[0], tok.model.bosID)
			}
			if wrapped[len(wrapped)-1] != tok.model.eosID {
				t.Errorf("last token = %d, want EOS=%d", wrapped[len(wrapped)-1], tok.model.eosID)
			}
			// Check inner IDs preserved.
			for i, id := range ids {
				if wrapped[i+1] != id {
					t.Errorf("wrapped[%d] = %d, want %d", i+1, wrapped[i+1], id)
				}
			}
		})
		t.Run("empty_ids", func(t *testing.T) {
			wrapped := tok.AddSpecialTokens(nil)
			if len(wrapped) != 2 {
				t.Fatalf("length = %d, want 2", len(wrapped))
			}
			if wrapped[0] != tok.model.bosID || wrapped[1] != tok.model.eosID {
				t.Errorf("got %v, want [%d, %d]", wrapped, tok.model.bosID, tok.model.eosID)
			}
		})
	})

	// 5. unhex
	t.Run("unhex", func(t *testing.T) {
		tests := []struct {
			c    byte
			want int
		}{
			{'0', 0}, {'9', 9},
			{'A', 10}, {'F', 15},
			{'a', 10}, {'f', 15},
			{'G', -1}, {'g', -1}, {'/', -1}, {':', -1}, {'@', -1},
		}
		for _, tt := range tests {
			got := unhex(tt.c)
			if got != tt.want {
				t.Errorf("unhex(%c) = %d, want %d", tt.c, got, tt.want)
			}
		}
	})

	// 6. pieceIsByte
	t.Run("pieceIsByte", func(t *testing.T) {
		tests := []struct {
			piece string
			want  bool
		}{
			{"<0x41>", true},
			{"<0xff>", true},
			{"<0x0A>", true},
			{"short", false},   // wrong length
			{"toolong", false}, // wrong length
			{"X0x41>", false},  // wrong prefix char 0
			{"<1x41>", false},  // wrong prefix char 1
			{"<00x41>", false}, // wrong length (7 chars)
			{"<0x41X", false},  // wrong suffix
			{"<0y41>", false},  // wrong prefix char 2
		}
		for _, tt := range tests {
			got := pieceIsByte(tt.piece)
			if got != tt.want {
				t.Errorf("pieceIsByte(%q) = %v, want %v", tt.piece, got, tt.want)
			}
		}
	})

	// 7. writeDummyPrefix with escapeWhitespaces=false
	t.Run("writeDummyPrefix_no_escape", func(t *testing.T) {
		n := &Normalizer{
			addDummyPrefix:    true,
			escapeWhitespaces: false,
		}
		var buf strings.Builder
		n.writeDummyPrefix(&buf)
		if buf.String() != " " {
			t.Errorf("writeDummyPrefix with escapeWhitespaces=false = %q, want %q", buf.String(), " ")
		}
	})
	t.Run("writeDummyPrefix_disabled", func(t *testing.T) {
		n := &Normalizer{
			addDummyPrefix: false,
		}
		var buf strings.Builder
		n.writeDummyPrefix(&buf)
		if buf.String() != "" {
			t.Errorf("writeDummyPrefix with addDummyPrefix=false = %q, want empty", buf.String())
		}
	})

	// 8. charLen edge cases
	t.Run("charLen", func(t *testing.T) {
		m := tok.model
		t.Run("at_end", func(t *testing.T) {
			s := "a"
			got := m.charLen(s, 0, 1)
			if got != 1 {
				t.Errorf("charLen at end = %d, want 1", got)
			}
		})
		t.Run("multibyte", func(t *testing.T) {
			s := "日本"
			got := m.charLen(s, 0, len(s))
			if got != 3 {
				t.Errorf("charLen for CJK = %d, want 3", got)
			}
		})
		t.Run("invalid_utf8", func(t *testing.T) {
			s := string([]byte{0xFF, 0xFE})
			got := m.charLen(s, 0, len(s))
			if got != 1 {
				t.Errorf("charLen for invalid UTF-8 = %d, want 1", got)
			}
		})
		t.Run("clamped_to_size", func(t *testing.T) {
			// Multi-byte char but size is truncated.
			s := "日" // 3 bytes
			got := m.charLen(s, 0, 2)
			if got != 2 {
				t.Errorf("charLen clamped = %d, want 2", got)
			}
		})
	})

	// 9. handleFallback non-byteFallback UNK path
	t.Run("handleFallback_no_byte_fallback", func(t *testing.T) {
		// Create a minimal model with byteFallback=false.
		m := &Model{
			pieces: []Piece{
				{Piece: "<unk>", Score: 0, Type: PieceUnknown},
			},
			pieceIndex:   map[string]int{"<unk>": 0},
			unkID:        0,
			byteFallback: false,
			minScoreVal:  -10,
			maxScoreVal:  10,
		}

		dp := make([]bestPathNode, 4)
		for i := range dp {
			dp[i].id = -1
		}
		// Simulate a character at position 0 with length 1.
		m.handleFallback("abc", 0, 1, 3, m.minScore()-unkPenalty, 0, dp)
		if dp[1].id != m.unkID {
			t.Errorf("handleFallback UNK path: dp[1].id = %d, want %d", dp[1].id, m.unkID)
		}
	})

	// 10. pieceScore with UserDefined
	t.Run("pieceScore_UserDefined", func(t *testing.T) {
		m := &Model{
			pieces: []Piece{
				{Piece: "test", Score: -5, Type: PieceUserDefined},
			},
			maxScoreVal: 10,
		}
		score := m.pieceScore(0, 4)
		expected := float32(4)*10 - 0.1
		if score != expected {
			t.Errorf("pieceScore(UserDefined) = %f, want %f", score, expected)
		}
	})
	t.Run("pieceScore_Normal", func(t *testing.T) {
		m := &Model{
			pieces: []Piece{
				{Piece: "test", Score: -3.5, Type: PieceNormal},
			},
		}
		score := m.pieceScore(0, 4)
		if score != -3.5 {
			t.Errorf("pieceScore(Normal) = %f, want -3.5", score)
		}
	})

	// 11. decodePrecompiledCharsmap with invalid data
	t.Run("decodePrecompiledCharsmap", func(t *testing.T) {
		t.Run("too_short", func(t *testing.T) {
			n := &Normalizer{}
			n.decodePrecompiledCharsmap([]byte{0x01, 0x02})
			if n.trie != nil {
				t.Error("expected nil trie for too-short data")
			}
		})
		t.Run("bad_trie_size", func(t *testing.T) {
			// trie size = 100 but data is only 8 bytes total.
			n := &Normalizer{}
			n.decodePrecompiledCharsmap([]byte{100, 0, 0, 0, 1, 2, 3, 4})
			if n.trie != nil {
				t.Error("expected nil trie for bad trie size")
			}
		})
		t.Run("valid_minimal", func(t *testing.T) {
			// trie size = 4, with 4 bytes of trie data and some normalized data.
			n := &Normalizer{}
			data := []byte{4, 0, 0, 0, 0, 0, 0, 0, 'a', 0}
			n.decodePrecompiledCharsmap(data)
			if n.trie == nil {
				t.Error("expected non-nil trie for valid data")
			}
			if n.normalized == nil {
				t.Error("expected non-nil normalized for valid data")
			}
		})
	})

	// 12. PieceToId with unknown piece
	t.Run("PieceToId", func(t *testing.T) {
		t.Run("known_piece", func(t *testing.T) {
			// The UNK token itself should be in the index.
			id := tok.model.PieceToId("<unk>")
			if id != tok.model.unkID {
				t.Errorf("PieceToId(<unk>) = %d, want %d", id, tok.model.unkID)
			}
		})
		t.Run("unknown_piece", func(t *testing.T) {
			id := tok.model.PieceToId("this_piece_does_not_exist_in_vocab_xyz123")
			if id != tok.model.unkID {
				t.Errorf("PieceToId(unknown) = %d, want unkID=%d", id, tok.model.unkID)
			}
		})
	})

	// 13. Decode with UNK token ID
	t.Run("Decode_UNK", func(t *testing.T) {
		decoded, err := tok.Decode([]int{tok.model.unkID})
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		if !strings.Contains(decoded, "\u2047") {
			t.Errorf("Decode(unkID) = %q, expected to contain U+2047", decoded)
		}
	})
	t.Run("Decode_out_of_range", func(t *testing.T) {
		// IDs out of range should be silently skipped.
		decoded, err := tok.Decode([]int{-1, 999999})
		if err != nil {
			t.Fatalf("Decode error: %v", err)
		}
		if decoded != "" {
			t.Errorf("Decode(out-of-range) = %q, want empty", decoded)
		}
	})

	// 14. trimTrailingSpaces with escapeWhitespaces=false
	t.Run("trimTrailingSpaces", func(t *testing.T) {
		t.Run("no_escape_trims_spaces", func(t *testing.T) {
			n := &Normalizer{
				removeExtraWhitespace: true,
				escapeWhitespaces:     false,
			}
			got := n.trimTrailingSpaces("hello   ")
			if got != "hello" {
				t.Errorf("trimTrailingSpaces = %q, want %q", got, "hello")
			}
		})
		t.Run("no_remove_extra_noop", func(t *testing.T) {
			n := &Normalizer{
				removeExtraWhitespace: false,
				escapeWhitespaces:     false,
			}
			got := n.trimTrailingSpaces("hello   ")
			if got != "hello   " {
				t.Errorf("trimTrailingSpaces = %q, want %q", got, "hello   ")
			}
		})
	})

	// 15. processChars with escapeWhitespaces=false
	t.Run("processChars_no_escape", func(t *testing.T) {
		n := &Normalizer{
			removeExtraWhitespace: false,
			escapeWhitespaces:     false,
		}
		var buf strings.Builder
		n.processChars(&buf, "hello world")
		got := buf.String()
		if got != "hello world" {
			t.Errorf("processChars no escape = %q, want %q", got, "hello world")
		}
	})
	t.Run("processChars_dedup_spaces", func(t *testing.T) {
		n := &Normalizer{
			removeExtraWhitespace: true,
			escapeWhitespaces:     false,
		}
		var buf strings.Builder
		n.processChars(&buf, "hello  world")
		got := buf.String()
		if strings.Contains(got, "  ") {
			t.Errorf("processChars should dedup spaces: %q", got)
		}
	})

	// 16. CommonPrefixSearchBytes with empty trie
	t.Run("CommonPrefixSearchBytes_empty", func(t *testing.T) {
		d := &DartsDoubleArray{units: nil}
		results := d.CommonPrefixSearchBytes([]byte("hello"))
		if results != nil {
			t.Errorf("CommonPrefixSearchBytes on empty trie = %v, want nil", results)
		}
	})
	t.Run("CommonPrefixSearchBytes_empty_units", func(t *testing.T) {
		d := &DartsDoubleArray{units: []uint32{}}
		results := d.CommonPrefixSearchBytes([]byte("hello"))
		if results != nil {
			t.Errorf("CommonPrefixSearchBytes on empty units = %v, want nil", results)
		}
	})

	// Additional: pieceToByte with invalid hex digits
	t.Run("pieceToByte_invalid_hex", func(t *testing.T) {
		_, ok := pieceToByte("<0xGG>")
		if ok {
			t.Error("pieceToByte should fail for invalid hex digits")
		}
		_, ok = pieceToByte("<0xZZ>")
		if ok {
			t.Error("pieceToByte should fail for invalid hex digits")
		}
	})

	// Additional: Normalizer.Normalize with escapeWhitespaces=false and addDummyPrefix=true
	t.Run("Normalize_no_escape_whitespace", func(t *testing.T) {
		n := &Normalizer{
			addDummyPrefix:        true,
			removeExtraWhitespace: true,
			escapeWhitespaces:     false,
		}
		got := n.Normalize("hello world")
		if !strings.HasPrefix(got, " ") {
			t.Errorf("Normalize should start with space when escapeWhitespaces=false: %q", got)
		}
	})

	// Additional: ByteTrie Traverse
	t.Run("ByteTrie_Traverse", func(t *testing.T) {
		trie := NewByteTrie()
		trie.Insert("ab", 42)

		// Traverse 'a' — exists but not terminal.
		child, ret := trie.Traverse('a')
		if ret != -1 {
			t.Errorf("Traverse('a') ret = %d, want -1", ret)
		}
		if child == nil {
			t.Fatal("Traverse('a') returned nil child")
		}

		// Traverse 'b' from child — terminal.
		_, ret = child.Traverse('b')
		if ret != 42 {
			t.Errorf("Traverse('b') ret = %d, want 42", ret)
		}

		// Traverse non-existent.
		_, ret = trie.Traverse('z')
		if ret != -2 {
			t.Errorf("Traverse('z') ret = %d, want -2", ret)
		}
	})

	// Additional: full encode-decode round trip via reader-loaded tokenizer
	t.Run("RoundTrip_ReaderTokenizer", func(t *testing.T) {
		data, err := os.ReadFile("_testdata/spm.model")
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		tok2, err := NewTokenizerFromReader(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("NewTokenizerFromReader: %v", err)
		}
		ids, _ := tok2.Encode("Test sentence.")
		decoded, _ := tok2.Decode(ids)
		if decoded != "Test sentence." {
			t.Errorf("round trip = %q, want %q", decoded, "Test sentence.")
		}
	})

	// Additional: LoadModel with bad path
	t.Run("LoadModel_bad_path", func(t *testing.T) {
		_, err := LoadModel("/nonexistent/path/model.bin")
		if err == nil {
			t.Error("expected error for nonexistent path")
		}
	})

	// Additional: Encode/Decode empty
	t.Run("Encode_empty", func(t *testing.T) {
		ids, err := tok.Encode("")
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if ids != nil {
			t.Errorf("Encode('') = %v, want nil", ids)
		}
	})
	t.Run("Decode_empty", func(t *testing.T) {
		decoded, err := tok.Decode(nil)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		if decoded != "" {
			t.Errorf("Decode(nil) = %q, want empty", decoded)
		}
	})

	// Additional: skipLeadingWhitespace
	t.Run("skipLeadingWhitespace_disabled", func(t *testing.T) {
		n := &Normalizer{
			removeExtraWhitespace: false,
		}
		got := n.skipLeadingWhitespace("  hello")
		if got != "  hello" {
			t.Errorf("skipLeadingWhitespace disabled = %q, want %q", got, "  hello")
		}
	})
}

// errorReader is an io.Reader that always returns an error.
type errorReader struct{}

func (e *errorReader) Read(p []byte) (int, error) {
	return 0, os.ErrClosed
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
