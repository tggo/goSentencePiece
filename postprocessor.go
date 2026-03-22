package sentencepiece

// PostProcessor modifies an Encoding after tokenization, typically to add
// special tokens like [CLS] and [SEP].
type PostProcessor interface {
	Process(encoding *Encoding, addSpecialTokens bool) *Encoding
}

// TemplateProcessing adds special tokens based on a template.
// For example, DeBERTa uses: [CLS] $A [SEP] for single, [CLS] $A [SEP] $B [SEP] for pairs.
type TemplateProcessing struct {
	SingleTemplate []TemplatePiece
	PairTemplate   []TemplatePiece
}

// TemplatePiece describes one element of a post-processing template. It is
// either a special token (with a pre-resolved ID) or a placeholder for a
// tokenized sequence ($A or $B).
type TemplatePiece struct {
	SpecialToken string // e.g., "[CLS]", "[SEP]" — looked up in vocab
	TokenID      int    // Pre-resolved ID
	TypeID       int    // Segment type ID
	IsSequence   bool   // true if this is $A or $B placeholder
	SequenceID   int    // 0 for $A, 1 for $B
}

// NewTemplateProcessing creates a post-processor from template definitions.
// The single template is used when encoding a single sequence, and the pair
// template is used when encoding two sequences (not yet supported by the
// current pipeline, but the template is stored for future use).
func NewTemplateProcessing(single, pair []TemplatePiece) *TemplateProcessing {
	return &TemplateProcessing{
		SingleTemplate: single,
		PairTemplate:   pair,
	}
}

// Process applies the template to the encoding. When addSpecialTokens is false,
// the encoding is returned unchanged.
func (tp *TemplateProcessing) Process(enc *Encoding, addSpecialTokens bool) *Encoding {
	if !addSpecialTokens {
		return enc
	}

	tmpl := tp.SingleTemplate

	// Count final length.
	totalLen := 0
	for _, piece := range tmpl {
		if piece.IsSequence {
			totalLen += enc.Len()
		} else {
			totalLen++
		}
	}

	ids := make([]int, 0, totalLen)
	tokens := make([]string, 0, totalLen)
	typeIDs := make([]int, 0, totalLen)
	specialMask := make([]int, 0, totalLen)
	attentionMask := make([]int, 0, totalLen)

	for _, piece := range tmpl {
		if piece.IsSequence {
			ids = append(ids, enc.IDs...)
			tokens = append(tokens, enc.Tokens...)
			for range enc.IDs {
				typeIDs = append(typeIDs, piece.TypeID)
				specialMask = append(specialMask, 0)
				attentionMask = append(attentionMask, 1)
			}
		} else {
			ids = append(ids, piece.TokenID)
			tokens = append(tokens, piece.SpecialToken)
			typeIDs = append(typeIDs, piece.TypeID)
			specialMask = append(specialMask, 1)
			attentionMask = append(attentionMask, 1)
		}
	}

	return &Encoding{
		IDs:               ids,
		Tokens:            tokens,
		AttentionMask:     attentionMask,
		TypeIDs:           typeIDs,
		SpecialTokensMask: specialMask,
	}
}

// BertStylePostProcessor returns a TemplateProcessing configured for the
// classic BERT pattern: [CLS] $A [SEP] for single sequences.
func BertStylePostProcessor(clsID, sepID int) *TemplateProcessing {
	single := []TemplatePiece{
		{SpecialToken: "[CLS]", TokenID: clsID, TypeID: 0},
		{IsSequence: true, SequenceID: 0, TypeID: 0},
		{SpecialToken: "[SEP]", TokenID: sepID, TypeID: 0},
	}
	pair := []TemplatePiece{
		{SpecialToken: "[CLS]", TokenID: clsID, TypeID: 0},
		{IsSequence: true, SequenceID: 0, TypeID: 0},
		{SpecialToken: "[SEP]", TokenID: sepID, TypeID: 0},
		{IsSequence: true, SequenceID: 1, TypeID: 1},
		{SpecialToken: "[SEP]", TokenID: sepID, TypeID: 1},
	}
	return NewTemplateProcessing(single, pair)
}
