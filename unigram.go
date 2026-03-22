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

	dp := m.viterbiForward(normalized)
	return m.viterbiBacktrack(normalized, dp)
}

// viterbiForward runs the forward Viterbi DP pass over the normalized string,
// returning the best-path DP array indexed by byte position.
func (m *Model) viterbiForward(normalized string) []bestPathNode {
	size := len(normalized)
	unkScore := m.minScore() - unkPenalty

	dp := make([]bestPathNode, size+1)
	for i := range dp {
		dp[i].id = -1
	}

	startsAt := 0
	for startsAt < size {
		bestScoreHere := dp[startsAt].bestPathScore
		mblen := m.charLen(normalized, startsAt, size)

		hasSingleNode := m.trieSearch(normalized, startsAt, size, mblen, bestScoreHere, dp)

		if !hasSingleNode {
			m.handleFallback(normalized, startsAt, mblen, size, unkScore, bestScoreHere, dp)
		}

		startsAt += mblen
	}

	return dp
}

// charLen returns the UTF-8 byte length of the character at position pos,
// clamped to the remaining input size.
func (m *Model) charLen(normalized string, pos, size int) int {
	_, mblen := utf8.DecodeRuneInString(normalized[pos:])
	if mblen == 0 {
		mblen = 1
	}
	if pos+mblen > size {
		mblen = size - pos
	}
	return mblen
}

// trieSearch traverses the vocab trie byte-by-byte from startsAt, updating
// the DP array for each matching piece. Returns true if any match covers
// exactly one Unicode character (hasSingleNode).
func (m *Model) trieSearch(normalized string, startsAt, size, mblen int, bestScoreHere float32, dp []bestPathNode) bool {
	hasSingleNode := false
	node := m.vocabTrie
	keyPos := startsAt

	for keyPos < size {
		var ret int
		node, ret = node.Traverse(normalized[keyPos])
		keyPos++

		if ret == -2 {
			break
		}
		if ret < 0 || m.pieces[ret].Type == PieceUnused {
			continue
		}

		length := keyPos - startsAt
		score := m.pieceScore(ret, length)
		candidateScore := score + bestScoreHere
		target := &dp[keyPos]

		if target.id == -1 || candidateScore > target.bestPathScore {
			target.bestPathScore = candidateScore
			target.startsAt = startsAt
			target.id = ret
		}

		if !hasSingleNode && length == mblen {
			hasSingleNode = true
		}
	}

	return hasSingleNode
}

// pieceScore returns the score for a piece, with user-defined pieces receiving
// a bonus proportional to their length.
func (m *Model) pieceScore(pieceID, length int) float32 {
	if m.pieces[pieceID].Type == PieceUserDefined {
		return float32(length)*m.maxScoreVal - 0.1
	}
	return m.pieces[pieceID].Score
}

// handleFallback handles the case where no vocab piece covers a single
// Unicode character at the current position — either via byte fallback
// or UNK token.
func (m *Model) handleFallback(normalized string, startsAt, mblen, size int, unkScore, bestScoreHere float32, dp []bestPathNode) {
	endPos := startsAt + mblen
	if endPos > size {
		return
	}

	if m.byteFallback {
		m.insertByteFallback(normalized, startsAt, endPos, bestScoreHere, dp)
	} else {
		candidateScore := unkScore + bestScoreHere
		target := &dp[endPos]
		if target.id == -1 || candidateScore > target.bestPathScore {
			target.bestPathScore = candidateScore
			target.startsAt = startsAt
			target.id = m.unkID
		}
	}
}

// viterbiBacktrack recovers the optimal segmentation by walking backwards
// through the DP array.
func (m *Model) viterbiBacktrack(normalized string, dp []bestPathNode) []encodedPiece {
	size := len(normalized)
	if dp[size].id == -1 {
		return []encodedPiece{{id: m.unkID, piece: normalized}}
	}

	var result []encodedPiece
	endsAt := size
	for endsAt > 0 {
		node := dp[endsAt]
		startPos := node.startsAt
		pieceID := node.id

		if m.byteFallback && m.pieces[pieceID].Type == PieceByte {
			// Emit byte tokens in reverse (will be reversed later).
			for i := endsAt - 1; i >= startPos; i-- {
				byteToken := byteToPiece(normalized[i])
				byteID := m.PieceToId(byteToken)
				result = append(result, encodedPiece{id: byteID, piece: byteToken})
			}
		} else {
			result = append(result, encodedPiece{id: pieceID, piece: normalized[startPos:endsAt]})
		}

		endsAt = startPos
	}

	// Reverse to get forward order.
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
