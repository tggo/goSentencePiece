package sentencepiece

import (
	"encoding/json"
	"os"
	"strings"
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

// --- Unit tests for JSON parsing functions (no tokenizer.json needed) ---

func TestLoadFromJSON_InvalidJSON(t *testing.T) {
	_, err := loadFromJSON([]byte(`not json`))
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestLoadFromJSON_UnsupportedModelType(t *testing.T) {
	j := `{"model":{"type":"WordPiece","vocab":{},"merges":[]},"added_tokens":[]}`
	_, err := loadFromJSON([]byte(j))
	if err == nil {
		t.Fatal("expected error for WordPiece")
	}
}

func TestLoadFromJSON_MinimalBPE(t *testing.T) {
	j := `{
		"added_tokens": [
			{"id":0,"content":"<unk>","special":true},
			{"id":1,"content":"<pad>","special":true}
		],
		"model": {
			"type": "BPE",
			"vocab": {"<unk>":0, "<pad>":1, "a":2, "b":3, "ab":4},
			"merges": [["a","b"]],
			"unk_token": "<unk>",
			"byte_fallback": false
		},
		"pre_tokenizer": null,
		"post_processor": null
	}`
	tok, err := loadFromJSON([]byte(j))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.VocabSize() != 5 {
		t.Errorf("vocab = %d, want 5", tok.VocabSize())
	}
	if tok.Model().Type() != ModelTypeBPE {
		t.Errorf("type = %d, want BPE", tok.Model().Type())
	}
	if tok.Model().UnkID() != 0 {
		t.Errorf("unkID = %d, want 0", tok.Model().UnkID())
	}
}

func TestLoadFromJSON_MinimalUnigram(t *testing.T) {
	j := `{
		"added_tokens": [
			{"id":0,"content":"<unk>","special":true}
		],
		"model": {
			"type": "Unigram",
			"unk_id": 0,
			"vocab": [
				["<unk>", 0.0],
				["a", -1.0],
				["b", -1.5],
				["ab", -0.5],
				["▁", -2.0]
			]
		},
		"pre_tokenizer": null,
		"post_processor": null
	}`
	tok, err := loadFromJSON([]byte(j))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.VocabSize() != 5 {
		t.Errorf("vocab = %d, want 5", tok.VocabSize())
	}
	if tok.Model().Type() != ModelTypeUnigram {
		t.Errorf("type = %d, want Unigram", tok.Model().Type())
	}
}

func TestAssignBPEScores_StringFormat(t *testing.T) {
	// Test the "A B" string merge format.
	pieces := []Piece{
		{Piece: "a", Type: PieceNormal},
		{Piece: "b", Type: PieceNormal},
		{Piece: "ab", Type: PieceNormal},
	}
	pieceIndex := map[string]int{"a": 0, "b": 1, "ab": 2}
	raw := json.RawMessage(`["a b"]`)

	if err := assignBPEScores(pieces, pieceIndex, raw); err != nil {
		t.Fatalf("assignBPEScores: %v", err)
	}
	if pieces[2].Score != 1.0 {
		t.Errorf("score of 'ab' = %f, want 1.0", pieces[2].Score)
	}
}

func TestAssignBPEScores_NestedFormat(t *testing.T) {
	pieces := []Piece{
		{Piece: "a", Type: PieceNormal},
		{Piece: "b", Type: PieceNormal},
		{Piece: "ab", Type: PieceNormal},
		{Piece: "c", Type: PieceNormal},
		{Piece: "abc", Type: PieceNormal},
	}
	pieceIndex := map[string]int{"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4}
	raw := json.RawMessage(`[["a","b"],["ab","c"]]`)

	if err := assignBPEScores(pieces, pieceIndex, raw); err != nil {
		t.Fatalf("assignBPEScores: %v", err)
	}
	// merge[0] → "ab" score = 2, merge[1] → "abc" score = 1
	if pieces[2].Score != 2.0 {
		t.Errorf("score of 'ab' = %f, want 2.0", pieces[2].Score)
	}
	if pieces[4].Score != 1.0 {
		t.Errorf("score of 'abc' = %f, want 1.0", pieces[4].Score)
	}
}

func TestParseUnigramVocab(t *testing.T) {
	raw := json.RawMessage(`[["<unk>",0.0],["hello",-1.5],["▁world",-2.3]]`)
	pieces, idx, err := parseUnigramVocab(raw)
	if err != nil {
		t.Fatalf("parseUnigramVocab: %v", err)
	}
	if len(pieces) != 3 {
		t.Fatalf("len = %d, want 3", len(pieces))
	}
	if pieces[1].Piece != "hello" || pieces[1].Score != -1.5 {
		t.Errorf("piece[1] = %+v", pieces[1])
	}
	if idx["▁world"] != 2 {
		t.Errorf("index['▁world'] = %d, want 2", idx["▁world"])
	}
}

func TestClassifyJSONPieces(t *testing.T) {
	pieces := []Piece{
		{Piece: "<pad>", Type: PieceNormal},
		{Piece: "<unk>", Type: PieceNormal},
		{Piece: "<0x41>", Type: PieceNormal},
		{Piece: "<unused0>", Type: PieceNormal},
		{Piece: "hello", Type: PieceNormal},
		{Piece: "[@BOS@]", Type: PieceNormal},
	}
	addedTokens := []addedTokenJSON{
		{ID: 0, Content: "<pad>", Special: true},
		{ID: 1, Content: "<unk>", Special: true},
		{ID: 3, Content: "<unused0>", Special: false},
		{ID: 5, Content: "[@BOS@]", Special: false},
	}
	classifyJSONPieces(pieces, addedTokens)

	if pieces[0].Type != PieceControl {
		t.Errorf("pad type = %d, want Control", pieces[0].Type)
	}
	if pieces[1].Type != PieceUnknown {
		t.Errorf("unk type = %d, want Unknown", pieces[1].Type)
	}
	if pieces[2].Type != PieceByte {
		t.Errorf("byte type = %d, want Byte", pieces[2].Type)
	}
	if pieces[3].Type != PieceUnused {
		t.Errorf("unused type = %d, want Unused", pieces[3].Type)
	}
	if pieces[4].Type != PieceNormal {
		t.Errorf("hello type = %d, want Normal", pieces[4].Type)
	}
	if pieces[5].Type != PieceUserDefined {
		t.Errorf("[@BOS@] type = %d, want UserDefined", pieces[5].Type)
	}
}

func TestMetaspacePreTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello world", []string{"▁Hello", "▁world"}},
		{" ", []string{"▁"}},
		{"  ", []string{"▁", "▁"}},
		{"", nil},
		{"a", []string{"▁a"}},
		{" hello ", []string{"▁hello", "▁"}},
	}
	for _, tc := range tests {
		got := metaspacePreTokenize(tc.input)
		if !stringSliceEqual(got, tc.want) {
			t.Errorf("metaspacePreTokenize(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestSplitKeepDelimiter(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"▁Hello▁world", []string{"▁Hello", "▁world"}},
		{"▁", []string{"▁"}},
		{"▁▁", []string{"▁", "▁"}},
		{"abc", []string{"abc"}},
		{"▁a▁b▁c", []string{"▁a", "▁b", "▁c"}},
	}
	for _, tc := range tests {
		got := splitKeepDelimiter(tc.input, '▁')
		if !stringSliceEqual(got, tc.want) {
			t.Errorf("splitKeepDelimiter(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestResolveTokenID(t *testing.T) {
	added := []addedTokenJSON{
		{ID: 5, Content: "<bos>"},
		{ID: 10, Content: "<eos>"},
	}
	idx := map[string]int{"<bos>": 5, "fallback": 99}

	if id := resolveTokenID(added, idx, "<bos>"); id != 5 {
		t.Errorf("bos = %d, want 5", id)
	}
	if id := resolveTokenID(added, idx, "<eos>"); id != 10 {
		t.Errorf("eos = %d, want 10", id)
	}
	if id := resolveTokenID(added, idx, "fallback"); id != 99 {
		t.Errorf("fallback = %d, want 99", id)
	}
	if id := resolveTokenID(added, idx, "missing"); id != -1 {
		t.Errorf("missing = %d, want -1", id)
	}
}

func TestResolveUnkID(t *testing.T) {
	// With unk_id pointer.
	unkID := 7
	tj := &tokenizerJSON{Model: modelJSON{UnkID: &unkID}}
	if id := resolveUnkID(tj, nil); id != 7 {
		t.Errorf("unk_id pointer = %d, want 7", id)
	}

	// With unk_token.
	tj2 := &tokenizerJSON{Model: modelJSON{UnkToken: "<unk>"}}
	idx := map[string]int{"<unk>": 3}
	if id := resolveUnkID(tj2, idx); id != 3 {
		t.Errorf("unk_token = %d, want 3", id)
	}

	// Default.
	tj3 := &tokenizerJSON{}
	if id := resolveUnkID(tj3, nil); id != 0 {
		t.Errorf("default = %d, want 0", id)
	}
}

func TestParseJSONNormConfig(t *testing.T) {
	// Metaspace with always.
	raw := json.RawMessage(`{"type":"Metaspace","prepend_scheme":"always"}`)
	addDummy, escapeSp := parseJSONNormConfig(raw)
	if !addDummy || !escapeSp {
		t.Errorf("Metaspace always: addDummy=%v, escape=%v", addDummy, escapeSp)
	}

	// Metaspace with never.
	raw = json.RawMessage(`{"type":"Metaspace","prepend_scheme":"never"}`)
	addDummy, escapeSp = parseJSONNormConfig(raw)
	if addDummy || !escapeSp {
		t.Errorf("Metaspace never: addDummy=%v, escape=%v", addDummy, escapeSp)
	}

	// Null.
	addDummy, escapeSp = parseJSONNormConfig(nil)
	if addDummy || escapeSp {
		t.Errorf("nil: addDummy=%v, escape=%v", addDummy, escapeSp)
	}

	// Non-Metaspace.
	raw = json.RawMessage(`{"type":"ByteLevel"}`)
	addDummy, escapeSp = parseJSONNormConfig(raw)
	if addDummy || escapeSp {
		t.Errorf("ByteLevel: addDummy=%v, escape=%v", addDummy, escapeSp)
	}
}

func TestHasMetaspacePreTokenizer(t *testing.T) {
	if !hasMetaspacePreTokenizer(json.RawMessage(`{"type":"Metaspace"}`)) {
		t.Error("should detect Metaspace")
	}
	if hasMetaspacePreTokenizer(json.RawMessage(`{"type":"ByteLevel"}`)) {
		t.Error("should not detect ByteLevel as Metaspace")
	}
	if hasMetaspacePreTokenizer(nil) {
		t.Error("should not detect nil")
	}
	if hasMetaspacePreTokenizer(json.RawMessage(`null`)) {
		t.Error("should not detect null")
	}
}

func TestParseJSONPostProcessor(t *testing.T) {
	// Null.
	if pp := parseJSONPostProcessor(nil); pp != nil {
		t.Error("expected nil for nil input")
	}
	if pp := parseJSONPostProcessor(json.RawMessage(`null`)); pp != nil {
		t.Error("expected nil for null")
	}

	// Non-TemplateProcessing type.
	if pp := parseJSONPostProcessor(json.RawMessage(`{"type":"ByteLevel"}`)); pp != nil {
		t.Error("expected nil for non-TemplateProcessing")
	}

	// Valid TemplateProcessing.
	raw := json.RawMessage(`{
		"type": "TemplateProcessing",
		"single": [
			{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
			{"Sequence": {"id": "A", "type_id": 0}},
			{"SpecialToken": {"id": "[SEP]", "type_id": 0}}
		],
		"pair": [
			{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
			{"Sequence": {"id": "A", "type_id": 0}},
			{"SpecialToken": {"id": "[SEP]", "type_id": 0}},
			{"Sequence": {"id": "B", "type_id": 1}}
		],
		"special_tokens": {
			"[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
			"[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
		}
	}`)
	pp := parseJSONPostProcessor(raw)
	if pp == nil {
		t.Fatal("expected non-nil PostProcessor")
	}
}

func TestBuildAddedTokensTrie(t *testing.T) {
	tokens := []addedTokenJSON{
		{ID: 0, Content: "<pad>", Special: true},      // excluded: special
		{ID: 1, Content: "<0x41>", Special: false},     // excluded: byte token
		{ID: 2, Content: "\n", Special: false},         // included
		{ID: 3, Content: "\n\n", Special: false},       // included
		{ID: 4, Content: "<unused0>", Special: false},  // included
		{ID: 5, Content: "[@BOS@]", Special: false},    // included
	}
	trie := buildAddedTokensTrie(tokens)
	if trie == nil {
		t.Fatal("expected non-nil trie")
	}
}

func TestParseUnigramVocab_Errors(t *testing.T) {
	// Bad JSON.
	_, _, err := parseUnigramVocab(json.RawMessage(`not json`))
	if err == nil {
		t.Error("expected error for bad JSON")
	}

	// Bad token entry.
	_, _, err = parseUnigramVocab(json.RawMessage(`[[123, -1.0]]`))
	if err == nil {
		t.Error("expected error for non-string token")
	}

	// Bad score entry.
	_, _, err = parseUnigramVocab(json.RawMessage(`[["hello", "not_a_number"]]`))
	if err == nil {
		t.Error("expected error for non-numeric score")
	}
}

func TestParseBPEVocab_Error(t *testing.T) {
	_, _, err := parseBPEVocab(json.RawMessage(`not json`))
	if err == nil {
		t.Error("expected error for bad JSON")
	}
}

func TestNewTokenizerFromJSON_BadPath(t *testing.T) {
	_, err := NewTokenizerFromJSON("/nonexistent/file.json")
	if err == nil {
		t.Fatal("expected error for bad path")
	}
}

func TestNewTokenizerFromJSONReader_BadData(t *testing.T) {
	_, err := NewTokenizerFromJSONReader(strings.NewReader("not json"))
	if err == nil {
		t.Fatal("expected error for bad data")
	}
}

func TestNewTokenizerFromReader_JSON(t *testing.T) {
	// NewTokenizerFromReader should auto-detect JSON.
	j := `{
		"added_tokens": [],
		"model": {
			"type": "BPE",
			"vocab": {"a":0, "b":1},
			"merges": [],
			"unk_token": "a",
			"byte_fallback": false
		},
		"pre_tokenizer": null,
		"post_processor": null
	}`
	tok, err := NewTokenizerFromReader(strings.NewReader(j))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.VocabSize() != 2 {
		t.Errorf("vocab = %d, want 2", tok.VocabSize())
	}
}

func TestNewTokenizer_JSON_AutoDetect(t *testing.T) {
	// Test auto-detect with a JSON file created in a temp dir.
	dir := t.TempDir()
	path := dir + "/test.json"
	j := `{
		"added_tokens": [],
		"model": {
			"type": "BPE",
			"vocab": {"x":0},
			"merges": [],
			"unk_token": "x",
			"byte_fallback": false
		},
		"pre_tokenizer": null,
		"post_processor": null
	}`
	if err := os.WriteFile(path, []byte(j), 0644); err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.Model().Type() != ModelTypeBPE {
		t.Errorf("type = %d, want BPE", tok.Model().Type())
	}
}

func TestEncodeToEncoding_Empty(t *testing.T) {
	j := `{
		"added_tokens": [],
		"model": {
			"type": "BPE",
			"vocab": {"a":0},
			"merges": [],
			"unk_token": "a",
			"byte_fallback": false
		},
		"pre_tokenizer": {"type":"Metaspace","prepend_scheme":"always"},
		"post_processor": null
	}`
	tok, err := loadFromJSON([]byte(j))
	if err != nil {
		t.Fatal(err)
	}
	// Empty input through encodeToEncoding.
	enc := tok.EncodeWithOptions("", false)
	if enc.Len() != 0 {
		t.Errorf("empty encode len = %d, want 0", enc.Len())
	}
}

func TestIsJSON(t *testing.T) {
	tests := []struct {
		data []byte
		want bool
	}{
		{[]byte(`{"a":1}`), true},
		{[]byte(`  { }`), true},
		{[]byte{0x0a, 0x7b}, true},  // \n{
		{[]byte(`protobuf`), false},
		{[]byte{}, false},
		{[]byte("  "), false},
	}
	for _, tc := range tests {
		got := isJSON(tc.data)
		if got != tc.want {
			t.Errorf("isJSON(%q) = %v, want %v", tc.data, got, tc.want)
		}
	}
}

// --- Benchmarks ---

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
