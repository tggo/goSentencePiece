package sentencepiece

import (
	"strings"
)

// Encoder handles the full encoding and decoding pipeline, combining
// normalization and Unigram Viterbi tokenization. It is the core engine
// used by Tokenizer.
type Encoder struct {
	model      *Model
	normalizer *Normalizer
}

// NewEncoder creates a new Encoder for the given model, initializing the
// normalizer from the model's NormalizerSpec configuration.
func NewEncoder(model *Model) *Encoder {
	return &Encoder{
		model:      model,
		normalizer: NewNormalizer(model),
	}
}

// Encode normalizes and tokenizes the input text, returning the resulting
// token IDs. It returns nil for empty input.
func (e *Encoder) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	normalized := e.normalizer.Normalize(text)
	pieces := e.model.encodeUnigram(normalized)

	ids := make([]int, len(pieces))
	for i, p := range pieces {
		ids[i] = p.id
	}

	return ids
}

// EncodeAsPieces normalizes and tokenizes the input text, returning the string
// representation of each token piece. It returns nil for empty input.
func (e *Encoder) EncodeAsPieces(text string) []string {
	if len(text) == 0 {
		return nil
	}

	normalized := e.normalizer.Normalize(text)
	pieces := e.model.encodeUnigram(normalized)

	result := make([]string, len(pieces))
	for i, p := range pieces {
		result[i] = p.piece
	}

	return result
}

// Decode converts a sequence of token IDs back to the original text string.
// Control tokens are skipped, byte-fallback tokens are reassembled into UTF-8,
// and the SentencePiece meta-space character is converted back to a regular space.
func (e *Encoder) Decode(ids []int) string {
	if len(ids) == 0 {
		return ""
	}

	var buf strings.Builder
	var byteBuf []byte

	for _, id := range ids {
		if id < 0 || id >= len(e.model.pieces) {
			continue
		}

		p := e.model.pieces[id]

		// Skip control tokens.
		if p.Type == PieceControl {
			continue
		}

		// Handle byte tokens: accumulate and flush as UTF-8.
		if p.Type == PieceByte {
			b, ok := pieceToByte(p.Piece)
			if ok {
				byteBuf = append(byteBuf, b)
				continue
			}
		}

		// Flush any accumulated bytes.
		if len(byteBuf) > 0 {
			buf.Write(byteBuf)
			byteBuf = byteBuf[:0]
		}

		piece := p.Piece

		// Handle UNK token.
		if p.Type == PieceUnknown {
			// SentencePiece uses unk_surface for display, but Decode typically
			// outputs the unk surface character.
			buf.WriteString(" \u2047 ")
			continue
		}

		// Replace meta-space ▁ with regular space.
		piece = strings.ReplaceAll(piece, metaSpace, " ")

		buf.WriteString(piece)
	}

	// Flush remaining bytes.
	if len(byteBuf) > 0 {
		buf.Write(byteBuf)
	}

	result := buf.String()

	// Remove leading space that was added by the normalizer.
	if e.model.addDummyPrefix && len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}

	return result
}
