package sentencepiece

import (
	"strings"
)

// Encoder handles the encoding and decoding of text.
type Encoder struct {
	model      *Model
	normalizer *Normalizer
}

// NewEncoder creates a new encoder for the given model.
func NewEncoder(model *Model) *Encoder {
	return &Encoder{
		model:      model,
		normalizer: NewNormalizer(model),
	}
}

// Encode tokenizes the input text and returns token IDs.
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

// EncodeAsPieces tokenizes the input text and returns piece strings.
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

// Decode converts token IDs back to a string.
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
