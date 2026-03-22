package sentencepiece

import (
	"strings"
)

// Encoder handles the full encoding and decoding pipeline, combining
// normalization with either Unigram Viterbi or BPE merge tokenization.
// It is the core engine used by Tokenizer.
//
// Encoder is safe for concurrent use by multiple goroutines.
type Encoder struct {
	model      *Model
	normalizer *Normalizer

	// preTokenize, when set, replaces the default normalization pipeline.
	// It receives raw input text and returns pre-tokenized segments that
	// are each encoded independently. Used by HuggingFace tokenizer.json
	// loaders for Metaspace pre-tokenization.
	preTokenize func(string) []string

	// addedTokens, when set, is a trie of added tokens that are matched
	// and extracted from input text before normalization/BPE. This
	// implements HuggingFace's added_tokens matching behavior.
	addedTokens *ByteTrie

	// stripLeadingSpace controls whether Decode strips the leading space
	// added by the normalizer. True for SentencePiece models (default),
	// false for HuggingFace tokenizer.json models.
	stripLeadingSpace bool
}

// NewEncoder creates a new Encoder for the given model, initializing the
// normalizer from the model's NormalizerSpec configuration.
func NewEncoder(model *Model) *Encoder {
	return &Encoder{
		model:             model,
		normalizer:        NewNormalizer(model),
		stripLeadingSpace: model.addDummyPrefix,
	}
}

// Encode normalizes and tokenizes the input text, returning the resulting
// token IDs. It returns nil for empty input.
func (e *Encoder) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	pieces := e.mergeConsecutiveUnk(e.encodePieces(text))

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

	pieces := e.mergeConsecutiveUnk(e.encodePieces(text))

	result := make([]string, len(pieces))
	for i, p := range pieces {
		result[i] = p.piece
	}

	return result
}

// encodePieces runs the normalization/pre-tokenization and encoding pipeline.
func (e *Encoder) encodePieces(text string) []encodedPiece {
	// If added tokens are configured, match and split them out first.
	if e.addedTokens != nil {
		return e.encodeWithAddedTokens(text)
	}
	return e.encodeSegment(text)
}

// encodeSegment encodes a single text segment (no added token matching).
// It always normalizes first, then optionally pre-tokenizes into segments.
func (e *Encoder) encodeSegment(text string) []encodedPiece {
	normalized := e.normalizer.Normalize(text)

	if e.preTokenize != nil {
		segments := e.preTokenize(normalized)
		var all []encodedPiece
		for _, seg := range segments {
			all = append(all, e.model.encode(seg)...)
		}
		return all
	}

	return e.model.encode(normalized)
}

// encodeWithAddedTokens splits text on added tokens, encodes each non-added
// segment normally, and interleaves the results.
func (e *Encoder) encodeWithAddedTokens(text string) []encodedPiece {
	// Split the text into segments: alternating regular text and added tokens.
	segments := e.splitOnAddedTokens(text)

	var result []encodedPiece
	for _, seg := range segments {
		if seg.id >= 0 {
			result = append(result, encodedPiece{id: seg.id, piece: seg.text})
		} else {
			result = append(result, e.encodeSegment(seg.text)...)
		}
	}
	return result
}

type textSegment struct {
	text string
	id   int // >= 0 for added tokens, -1 for regular text
}

// splitOnAddedTokens scans text left-to-right, greedily matching added tokens
// (longest match wins). Returns segments of regular text (id=-1) and matched
// added tokens (id=token ID).
func (e *Encoder) splitOnAddedTokens(text string) []textSegment {
	var segments []textSegment
	regularStart := 0

	for i := 0; i < len(text); {
		matchLen, matchID := e.longestAddedToken(text[i:])
		if matchLen > 0 {
			// Flush any regular text before this match.
			if i > regularStart {
				segments = append(segments, textSegment{text: text[regularStart:i], id: -1})
			}
			segments = append(segments, textSegment{text: text[i : i+matchLen], id: matchID})
			i += matchLen
			regularStart = i
		} else {
			i++
		}
	}

	// Flush remaining regular text.
	if regularStart < len(text) {
		segments = append(segments, textSegment{text: text[regularStart:], id: -1})
	}

	return segments
}

// longestAddedToken finds the longest added token match at the start of text.
// Returns (matchLength, tokenID) or (0, 0) if no match.
func (e *Encoder) longestAddedToken(text string) (int, int) {
	node := e.addedTokens
	bestLen := 0
	bestID := 0

	for i := 0; i < len(text); i++ {
		var ret int
		node, ret = node.Traverse(text[i])
		if ret == -2 {
			break
		}
		if ret >= 0 {
			bestLen = i + 1
			bestID = ret
		}
	}

	return bestLen, bestID
}

// mergeConsecutiveUnk merges consecutive UNK pieces into a single piece.
// This matches the C++ reference behavior (sentencepiece_processor.cc:609-625)
// which merges "continuous run of unknown pieces" when byte_fallback is disabled.
func (e *Encoder) mergeConsecutiveUnk(pieces []encodedPiece) []encodedPiece {
	if e.model.byteFallback || len(pieces) == 0 {
		return pieces
	}

	merged := make([]encodedPiece, 0, len(pieces))
	for _, p := range pieces {
		isUnk := p.id == e.model.unkID
		if isUnk && len(merged) > 0 && merged[len(merged)-1].id == e.model.unkID {
			// Merge with previous UNK.
			merged[len(merged)-1].piece += p.piece
		} else {
			merged = append(merged, p)
		}
	}
	return merged
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
	if e.stripLeadingSpace && len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}

	return result
}
