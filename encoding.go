package sentencepiece

// Encoding holds the complete result of tokenizing a text, including token IDs,
// string pieces, character offsets, and auxiliary masks needed for ML inference.
type Encoding struct {
	IDs               []int    // Token IDs
	Tokens            []string // String pieces
	AttentionMask     []int    // 1 for real tokens, 0 for padding
	TypeIDs           []int    // Segment IDs (0 for first sequence, 1 for second)
	SpecialTokensMask []int    // 1 for special tokens, 0 for normal tokens
}

// Len returns the number of tokens in the encoding.
func (e *Encoding) Len() int { return len(e.IDs) }

// newEncoding creates an Encoding from raw encoded pieces with all masks
// initialized to default values (attention=1, typeID=0, specialToken=0).
func newEncoding(ids []int, tokens []string) *Encoding {
	n := len(ids)
	enc := &Encoding{
		IDs:               ids,
		Tokens:            tokens,
		AttentionMask:     make([]int, n),
		TypeIDs:           make([]int, n),
		SpecialTokensMask: make([]int, n),
	}
	for i := range enc.AttentionMask {
		enc.AttentionMask[i] = 1
	}
	return enc
}
