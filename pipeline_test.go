package sentencepiece

import (
	"testing"
)

const testModelPath = "_testdata/spm.model"

func TestEncoding(t *testing.T) {
	enc := newEncoding([]int{1, 2, 3}, []string{"a", "b", "c"})

	if enc.Len() != 3 {
		t.Fatalf("expected Len()=3, got %d", enc.Len())
	}

	// Verify all masks are initialized correctly.
	for i := 0; i < enc.Len(); i++ {
		if enc.AttentionMask[i] != 1 {
			t.Errorf("AttentionMask[%d] = %d, want 1", i, enc.AttentionMask[i])
		}
		if enc.TypeIDs[i] != 0 {
			t.Errorf("TypeIDs[%d] = %d, want 0", i, enc.TypeIDs[i])
		}
		if enc.SpecialTokensMask[i] != 0 {
			t.Errorf("SpecialTokensMask[%d] = %d, want 0", i, enc.SpecialTokensMask[i])
		}
	}

	// Empty encoding.
	empty := newEncoding(nil, nil)
	if empty.Len() != 0 {
		t.Fatalf("expected Len()=0 for empty encoding, got %d", empty.Len())
	}
}

func TestTemplateProcessing(t *testing.T) {
	clsID := 101
	sepID := 102

	tp := NewTemplateProcessing(
		[]TemplatePiece{
			{SpecialToken: "[CLS]", TokenID: clsID, TypeID: 0},
			{IsSequence: true, SequenceID: 0, TypeID: 0},
			{SpecialToken: "[SEP]", TokenID: sepID, TypeID: 0},
		},
		nil,
	)

	enc := newEncoding([]int{10, 20, 30}, []string{"hello", "world", "!"})

	result := tp.Process(enc, true)
	if result.Len() != 5 {
		t.Fatalf("expected 5 tokens after processing, got %d", result.Len())
	}

	// Check IDs: [CLS]=101, 10, 20, 30, [SEP]=102
	expectedIDs := []int{101, 10, 20, 30, 102}
	for i, id := range expectedIDs {
		if result.IDs[i] != id {
			t.Errorf("IDs[%d] = %d, want %d", i, result.IDs[i], id)
		}
	}

	// Check special tokens mask.
	expectedSpecial := []int{1, 0, 0, 0, 1}
	for i, s := range expectedSpecial {
		if result.SpecialTokensMask[i] != s {
			t.Errorf("SpecialTokensMask[%d] = %d, want %d", i, result.SpecialTokensMask[i], s)
		}
	}

	// When addSpecialTokens=false, encoding should be unchanged.
	noSpecial := tp.Process(enc, false)
	if noSpecial.Len() != 3 {
		t.Fatalf("expected 3 tokens without special tokens, got %d", noSpecial.Len())
	}
}

func TestBertStylePostProcessor(t *testing.T) {
	clsID := 1
	sepID := 2

	pp := BertStylePostProcessor(clsID, sepID)

	enc := newEncoding([]int{10, 20}, []string{"a", "b"})
	result := pp.Process(enc, true)

	if result.Len() != 4 {
		t.Fatalf("expected 4 tokens, got %d", result.Len())
	}

	expectedIDs := []int{1, 10, 20, 2}
	for i, id := range expectedIDs {
		if result.IDs[i] != id {
			t.Errorf("IDs[%d] = %d, want %d", i, result.IDs[i], id)
		}
	}

	expectedTokens := []string{"[CLS]", "a", "b", "[SEP]"}
	for i, tok := range expectedTokens {
		if result.Tokens[i] != tok {
			t.Errorf("Tokens[%d] = %q, want %q", i, result.Tokens[i], tok)
		}
	}
}

func TestPadding(t *testing.T) {
	t.Run("PadRight", func(t *testing.T) {
		enc := newEncoding([]int{1, 2, 3}, []string{"a", "b", "c"})
		params := &PaddingParams{
			Direction: PadRight,
			PadID:     0,
			PadToken:  "[PAD]",
		}

		padded := PadEncoding(enc, 5, params)
		if padded.Len() != 5 {
			t.Fatalf("expected 5 tokens, got %d", padded.Len())
		}

		expectedIDs := []int{1, 2, 3, 0, 0}
		expectedAttn := []int{1, 1, 1, 0, 0}
		for i := range expectedIDs {
			if padded.IDs[i] != expectedIDs[i] {
				t.Errorf("IDs[%d] = %d, want %d", i, padded.IDs[i], expectedIDs[i])
			}
			if padded.AttentionMask[i] != expectedAttn[i] {
				t.Errorf("AttentionMask[%d] = %d, want %d", i, padded.AttentionMask[i], expectedAttn[i])
			}
		}
	})

	t.Run("PadLeft", func(t *testing.T) {
		enc := newEncoding([]int{1, 2}, []string{"a", "b"})
		params := &PaddingParams{
			Direction: PadLeft,
			PadID:     0,
			PadToken:  "[PAD]",
		}

		padded := PadEncoding(enc, 4, params)
		if padded.Len() != 4 {
			t.Fatalf("expected 4 tokens, got %d", padded.Len())
		}

		expectedIDs := []int{0, 0, 1, 2}
		expectedAttn := []int{0, 0, 1, 1}
		for i := range expectedIDs {
			if padded.IDs[i] != expectedIDs[i] {
				t.Errorf("IDs[%d] = %d, want %d", i, padded.IDs[i], expectedIDs[i])
			}
			if padded.AttentionMask[i] != expectedAttn[i] {
				t.Errorf("AttentionMask[%d] = %d, want %d", i, padded.AttentionMask[i], expectedAttn[i])
			}
		}
	})

	t.Run("BatchPadding", func(t *testing.T) {
		encs := []*Encoding{
			newEncoding([]int{1, 2, 3}, []string{"a", "b", "c"}),
			newEncoding([]int{4, 5}, []string{"d", "e"}),
			newEncoding([]int{6}, []string{"f"}),
		}
		params := &PaddingParams{
			Strategy:  PadToLongest,
			Direction: PadRight,
			PadID:     0,
			PadToken:  "[PAD]",
		}

		padded := PadEncodings(encs, params)
		for i, enc := range padded {
			if enc.Len() != 3 {
				t.Errorf("encoding[%d] length = %d, want 3", i, enc.Len())
			}
		}

		// Second encoding should have one pad token.
		if padded[1].IDs[2] != 0 {
			t.Errorf("expected pad ID at position 2, got %d", padded[1].IDs[2])
		}
		if padded[1].AttentionMask[2] != 0 {
			t.Errorf("expected attention mask 0 at pad position, got %d", padded[1].AttentionMask[2])
		}
	})
}

func TestTruncation(t *testing.T) {
	enc := newEncoding([]int{1, 2, 3, 4, 5}, []string{"a", "b", "c", "d", "e"})

	params := &TruncationParams{MaxLength: 3}
	truncated := TruncateEncoding(enc, params)

	if truncated.Len() != 3 {
		t.Fatalf("expected 3 tokens, got %d", truncated.Len())
	}

	expectedIDs := []int{1, 2, 3}
	for i, id := range expectedIDs {
		if truncated.IDs[i] != id {
			t.Errorf("IDs[%d] = %d, want %d", i, truncated.IDs[i], id)
		}
	}

	// Truncation should be a no-op when encoding is shorter than max.
	short := newEncoding([]int{1, 2}, []string{"a", "b"})
	result := TruncateEncoding(short, params)
	if result.Len() != 2 {
		t.Fatalf("expected 2 tokens (no truncation), got %d", result.Len())
	}
}

func TestEncodeWithOptions(t *testing.T) {
	tok, err := NewTokenizer(testModelPath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// Set up full pipeline: post-processor + truncation.
	bosID := tok.Model().BosID()
	eosID := tok.Model().EosID()
	tok.WithPostProcessor(BertStylePostProcessor(bosID, eosID)).
		WithTruncation(&TruncationParams{MaxLength: 10})

	enc := tok.EncodeWithOptions("Hello world", true)
	if enc == nil {
		t.Fatal("EncodeWithOptions returned nil")
	}
	if enc.Len() == 0 {
		t.Fatal("EncodeWithOptions returned empty encoding")
	}

	// First token should be BOS (from CLS position).
	if enc.IDs[0] != bosID {
		t.Errorf("first token ID = %d, want BOS %d", enc.IDs[0], bosID)
	}

	// Last token should be EOS (from SEP position).
	if enc.IDs[enc.Len()-1] != eosID {
		t.Errorf("last token ID = %d, want EOS %d", enc.IDs[enc.Len()-1], eosID)
	}

	// Should not exceed max length.
	if enc.Len() > 10 {
		t.Errorf("encoding length %d exceeds max 10", enc.Len())
	}

	// All attention mask values should be 1 (no padding applied).
	for i, v := range enc.AttentionMask {
		if v != 1 {
			t.Errorf("AttentionMask[%d] = %d, want 1", i, v)
		}
	}

	// Special tokens mask: first and last should be 1.
	if enc.SpecialTokensMask[0] != 1 {
		t.Errorf("SpecialTokensMask[0] = %d, want 1", enc.SpecialTokensMask[0])
	}
	if enc.SpecialTokensMask[enc.Len()-1] != 1 {
		t.Errorf("SpecialTokensMask[last] = %d, want 1", enc.SpecialTokensMask[enc.Len()-1])
	}

	// Without special tokens — should have no CLS/SEP.
	encNoSpecial := tok.EncodeWithOptions("Hello world", false)
	if encNoSpecial.IDs[0] == bosID {
		t.Error("expected no BOS token when addSpecialTokens=false")
	}
}

func TestEncodeBatchWithOptions(t *testing.T) {
	tok, err := NewTokenizer(testModelPath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	tok.WithPadding(&PaddingParams{
		Strategy:  PadToLongest,
		Direction: PadRight,
		PadID:     0,
		PadToken:  "[PAD]",
	})

	texts := []string{
		"Hello world",
		"Hi",
		"This is a longer sentence for testing",
	}

	encodings := tok.EncodeBatchWithOptions(texts, false)
	if len(encodings) != 3 {
		t.Fatalf("expected 3 encodings, got %d", len(encodings))
	}

	// All encodings should have the same length (padded to longest).
	maxLen := 0
	for _, enc := range encodings {
		if enc.Len() > maxLen {
			maxLen = enc.Len()
		}
	}

	for i, enc := range encodings {
		if enc.Len() != maxLen {
			t.Errorf("encoding[%d] length = %d, want %d", i, enc.Len(), maxLen)
		}
	}

	// The shortest encoding should have padding (attention mask = 0) at the end.
	shortest := encodings[1] // "Hi" should be shortest
	foundPad := false
	for _, v := range shortest.AttentionMask {
		if v == 0 {
			foundPad = true
			break
		}
	}
	if !foundPad {
		t.Error("expected padding in shortest encoding but found none")
	}
}
