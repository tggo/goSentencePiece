package sentencepiece

import (
	"unicode/utf8"
)

const unkPenalty = 10.0

// encodedPiece represents a single piece in the encoding result.
type encodedPiece struct {
	id    int
	piece string
}

// bestPathNode stores Viterbi DP state for each byte position.
type bestPathNode struct {
	id            int     // vocab piece ID (-1 = uninitialized)
	bestPathScore float32 // cumulative score to reach this position
	startsAt      int     // byte position where the best token starts
}

// encodeUnigram performs Viterbi-based Unigram tokenization.
// This follows the exact SentencePiece Viterbi implementation from unigram_model.cc.
func (m *Model) encodeUnigram(normalized string) []encodedPiece {
	if len(normalized) == 0 {
		return nil
	}

	size := len(normalized)
	unkScore := m.minScore() - unkPenalty

	// DP array indexed by byte position.
	bestPathEndsAt := make([]bestPathNode, size+1)
	for i := range bestPathEndsAt {
		bestPathEndsAt[i].id = -1
	}

	// Forward Viterbi pass.
	startsAt := 0
	for startsAt < size {
		bestScoreHere := bestPathEndsAt[startsAt].bestPathScore

		// Length of one UTF-8 character at this position.
		_, mblen := utf8.DecodeRuneInString(normalized[startsAt:])
		if mblen == 0 {
			mblen = 1
		}
		if startsAt+mblen > size {
			mblen = size - startsAt
		}

		hasSingleNode := false

		// Traverse trie byte-by-byte, exactly like the reference.
		node := m.vocabTrie
		keyPos := startsAt

		for keyPos < size {
			// Traverse one byte.
			var ret int
			node, ret = node.Traverse(normalized[keyPos])
			keyPos++

			if ret == -2 {
				break // No transition.
			}
			if ret >= 0 {
				// Found a piece.
				if m.pieces[ret].Type == PieceUnused {
					continue
				}

				length := keyPos - startsAt

				var score float32
				if m.pieces[ret].Type == PieceUserDefined {
					score = float32(length)*m.maxScoreVal - 0.1
				} else {
					score = m.pieces[ret].Score
				}

				candidateScore := score + bestScoreHere
				target := &bestPathEndsAt[keyPos]

				if target.id == -1 || candidateScore > target.bestPathScore {
					target.bestPathScore = candidateScore
					target.startsAt = startsAt
					target.id = ret
				}

				if !hasSingleNode && length == mblen {
					hasSingleNode = true
				}
			}
		}

		// UNK / byte fallback for characters with no single-char piece.
		if !hasSingleNode {
			endPos := startsAt + mblen
			if endPos <= size {
				if m.byteFallback {
					m.insertByteFallback(normalized, startsAt, endPos, bestScoreHere, bestPathEndsAt)
				} else {
					candidateScore := unkScore + bestScoreHere
					target := &bestPathEndsAt[endPos]
					if target.id == -1 || candidateScore > target.bestPathScore {
						target.bestPathScore = candidateScore
						target.startsAt = startsAt
						target.id = m.unkID
					}
				}
			}
		}

		startsAt += mblen
	}

	// Backtrack.
	if bestPathEndsAt[size].id == -1 {
		return []encodedPiece{{id: m.unkID, piece: normalized}}
	}

	var result []encodedPiece
	endsAt := size
	for endsAt > 0 {
		node := bestPathEndsAt[endsAt]
		startPos := node.startsAt
		pieceID := node.id

		if m.byteFallback && m.pieces[pieceID].Type == PieceByte {
			for i := endsAt - 1; i >= startPos; i-- {
				byteToken := byteToPiece(normalized[i])
				byteID := m.PieceToId(byteToken)
				result = append(result, encodedPiece{id: byteID, piece: byteToken})
			}
		} else {
			pieceStr := normalized[startPos:endsAt]
			result = append(result, encodedPiece{id: pieceID, piece: pieceStr})
		}

		endsAt = startPos
	}

	// Reverse.
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result
}

// insertByteFallback inserts byte-level tokens for a character span.
func (m *Model) insertByteFallback(normalized string, startPos, endPos int, scoreHere float32, dp []bestPathNode) {
	var totalByteScore float32
	for i := startPos; i < endPos; i++ {
		byteToken := byteToPiece(normalized[i])
		if byteID, ok := m.pieceIndex[byteToken]; ok {
			totalByteScore += m.pieces[byteID].Score
		}
	}

	candidateScore := totalByteScore + scoreHere
	target := &dp[endPos]
	if target.id == -1 || candidateScore > target.bestPathScore {
		target.bestPathScore = candidateScore
		target.startsAt = startPos
		byteToken := byteToPiece(normalized[startPos])
		if byteID, ok := m.pieceIndex[byteToken]; ok {
			target.id = byteID
		} else {
			target.id = m.unkID
		}
	}
}

// minScore returns the minimum score across all normal pieces.
func (m *Model) minScore() float32 {
	return m.minScoreVal
}
