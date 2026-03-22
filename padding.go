package sentencepiece

// PaddingStrategy determines how sequences are padded.
type PaddingStrategy int

const (
	// PadToLongest pads all sequences in a batch to the length of the longest.
	PadToLongest PaddingStrategy = iota
	// PadToMaxLength pads all sequences to a fixed maximum length.
	PadToMaxLength
)

// PaddingDirection determines which side to pad.
type PaddingDirection int

const (
	// PadRight appends padding tokens to the end of the sequence.
	PadRight PaddingDirection = iota
	// PadLeft prepends padding tokens to the beginning of the sequence.
	PadLeft
)

// PaddingParams configures padding behavior.
type PaddingParams struct {
	Strategy  PaddingStrategy
	Direction PaddingDirection
	MaxLength int    // Used with PadToMaxLength
	PadID     int    // Token ID used for padding (usually 0)
	PadToken  string // Token string for padding
}

// PadEncodings pads a slice of Encodings to the same length. When using
// PadToLongest, the target length is the maximum length among all encodings.
// When using PadToMaxLength, the target length is params.MaxLength.
func PadEncodings(encodings []*Encoding, params *PaddingParams) []*Encoding {
	if len(encodings) == 0 {
		return encodings
	}

	targetLen := 0
	switch params.Strategy {
	case PadToLongest:
		for _, enc := range encodings {
			if enc.Len() > targetLen {
				targetLen = enc.Len()
			}
		}
	case PadToMaxLength:
		targetLen = params.MaxLength
	}

	result := make([]*Encoding, len(encodings))
	for i, enc := range encodings {
		result[i] = PadEncoding(enc, targetLen, params)
	}
	return result
}

// PadEncoding pads a single Encoding to the specified target length. If the
// encoding is already at or above the target length, it is returned unchanged.
func PadEncoding(enc *Encoding, targetLen int, params *PaddingParams) *Encoding {
	if enc.Len() >= targetLen {
		return enc
	}

	padCount := targetLen - enc.Len()

	padIDs := make([]int, padCount)
	padTokens := make([]string, padCount)
	padAttention := make([]int, padCount)
	padTypeIDs := make([]int, padCount)
	padSpecial := make([]int, padCount)

	for i := 0; i < padCount; i++ {
		padIDs[i] = params.PadID
		padTokens[i] = params.PadToken
		// padAttention, padTypeIDs, padSpecial remain 0
	}

	switch params.Direction {
	case PadLeft:
		return &Encoding{
			IDs:               append(padIDs, enc.IDs...),
			Tokens:            append(padTokens, enc.Tokens...),
			AttentionMask:     append(padAttention, enc.AttentionMask...),
			TypeIDs:           append(padTypeIDs, enc.TypeIDs...),
			SpecialTokensMask: append(padSpecial, enc.SpecialTokensMask...),
		}
	default: // PadRight
		return &Encoding{
			IDs:               append(cloneInts(enc.IDs), padIDs...),
			Tokens:            append(cloneStrings(enc.Tokens), padTokens...),
			AttentionMask:     append(cloneInts(enc.AttentionMask), padAttention...),
			TypeIDs:           append(cloneInts(enc.TypeIDs), padTypeIDs...),
			SpecialTokensMask: append(cloneInts(enc.SpecialTokensMask), padSpecial...),
		}
	}
}

func cloneInts(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

func cloneStrings(s []string) []string {
	c := make([]string, len(s))
	copy(c, s)
	return c
}
