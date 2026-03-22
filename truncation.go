package sentencepiece

// TruncationParams configures truncation behavior.
type TruncationParams struct {
	MaxLength int // Maximum number of tokens to keep.
	Stride    int // Number of tokens to keep for overflowing (0 = no overflow).
}

// TruncateEncoding truncates an Encoding to the specified maximum length. If
// the encoding is already within the limit, it is returned unchanged.
func TruncateEncoding(enc *Encoding, params *TruncationParams) *Encoding {
	if enc.Len() <= params.MaxLength {
		return enc
	}

	n := params.MaxLength
	return &Encoding{
		IDs:               enc.IDs[:n:n],
		Tokens:            enc.Tokens[:n:n],
		AttentionMask:     enc.AttentionMask[:n:n],
		TypeIDs:           enc.TypeIDs[:n:n],
		SpecialTokensMask: enc.SpecialTokensMask[:n:n],
	}
}
