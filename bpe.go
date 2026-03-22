package sentencepiece

import (
	"container/heap"
	"unicode/utf8"
)

// bpeSymbol represents a symbol in the BPE doubly-linked list.
// start/end are byte offsets into the normalized string.
type bpeSymbol struct {
	start  int
	end    int
	prev   int // -1 if head
	next   int // -1 if tail
	freeze bool
}

func (s bpeSymbol) len() int              { return s.end - s.start }
func (s bpeSymbol) empty() bool           { return s.start == s.end }
func (s bpeSymbol) slice(n string) string { return n[s.start:s.end] }

// bpeSymbolPair is a candidate merge of two adjacent symbols.
type bpeSymbolPair struct {
	left  int
	right int
	score float32
	size  int // combined byte length at creation time
}

// bpePairHeap is a max-heap ordered by (score DESC, left ASC).
type bpePairHeap []*bpeSymbolPair

func (h bpePairHeap) Len() int { return len(h) }
func (h bpePairHeap) Less(i, j int) bool {
	if h[i].score != h[j].score {
		return h[i].score > h[j].score
	}
	return h[i].left < h[j].left
}
func (h bpePairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *bpePairHeap) Push(x any)   { *h = append(*h, x.(*bpeSymbolPair)) }
func (h *bpePairHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

// encodeBPE performs BPE encoding using greedy best-first merging.
func (m *Model) encodeBPE(normalized string) []encodedPiece {
	if len(normalized) == 0 {
		return nil
	}

	symbols := m.bpeInitSymbols(normalized)
	if len(symbols) == 0 {
		return nil
	}

	revMerge := make(map[string][2]string)
	agenda := &bpePairHeap{}
	heap.Init(agenda)

	for i := 1; i < len(symbols); i++ {
		m.bpeMaybeAddPair(normalized, symbols, i-1, i, agenda, revMerge)
	}

	// Main merge loop.
	for agenda.Len() > 0 {
		top := heap.Pop(agenda).(*bpeSymbolPair)

		left := &symbols[top.left]
		right := &symbols[top.right]

		// Skip stale entries.
		if left.empty() || right.empty() {
			continue
		}
		if left.len()+right.len() != top.size {
			continue
		}
		if left.freeze || right.freeze {
			continue
		}

		// Merge: extend left to cover right.
		left.end = right.end
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].prev = top.left
		}
		right.start = right.end // mark deleted

		// Add new potential merges.
		if left.prev >= 0 {
			m.bpeMaybeAddPair(normalized, symbols, left.prev, top.left, agenda, revMerge)
		}
		if left.next >= 0 {
			m.bpeMaybeAddPair(normalized, symbols, top.left, left.next, agenda, revMerge)
		}
	}

	return m.bpeCollectResults(normalized, symbols, revMerge)
}

// bpeInitSymbols splits normalized text into initial symbols.
func (m *Model) bpeInitSymbols(normalized string) []bpeSymbol {
	var symbols []bpeSymbol
	pos := 0

	for pos < len(normalized) {
		// Try user-defined symbols (longest match via trie).
		bestLen := 0
		node := m.vocabTrie
		for i := pos; i < len(normalized); i++ {
			var ret int
			node, ret = node.Traverse(normalized[i])
			if ret == -2 {
				break
			}
			if ret >= 0 && m.pieces[ret].Type == PieceUserDefined {
				bestLen = i - pos + 1
			}
		}

		var sym bpeSymbol
		sym.start = pos
		if bestLen > 0 {
			sym.end = pos + bestLen
			sym.freeze = true
			pos += bestLen
		} else {
			_, runeLen := utf8.DecodeRuneInString(normalized[pos:])
			if runeLen == 0 {
				runeLen = 1
			}
			sym.end = pos + runeLen
			pos += runeLen
		}

		idx := len(symbols)
		sym.prev = idx - 1
		sym.next = -1
		if idx > 0 {
			symbols[idx-1].next = idx
		}
		symbols = append(symbols, sym)
	}

	return symbols
}

// bpeMaybeAddPair checks if merging two adjacent symbols produces a known
// vocab piece, and if so adds it to the agenda.
func (m *Model) bpeMaybeAddPair(normalized string, symbols []bpeSymbol, left, right int, agenda *bpePairHeap, revMerge map[string][2]string) {
	if left < 0 || right < 0 {
		return
	}
	sl := symbols[left]
	sr := symbols[right]
	if sl.empty() || sr.empty() || sl.freeze || sr.freeze {
		return
	}

	merged := normalized[sl.start:sr.end]
	id, ok := m.pieceIndex[merged]
	if !ok {
		return
	}

	if m.pieces[id].Type == PieceUnused {
		revMerge[merged] = [2]string{sl.slice(normalized), sr.slice(normalized)}
	}

	heap.Push(agenda, &bpeSymbolPair{
		left:  left,
		right: right,
		score: m.pieces[id].Score,
		size:  len(merged),
	})
}

// bpeCollectResults collects final pieces, resegmenting UNUSED ones.
func (m *Model) bpeCollectResults(normalized string, symbols []bpeSymbol, revMerge map[string][2]string) []encodedPiece {
	var result []encodedPiece

	// Find head of linked list.
	head := -1
	for i, s := range symbols {
		if !s.empty() {
			head = i
			break
		}
	}
	if head < 0 {
		return nil
	}
	for symbols[head].prev >= 0 {
		head = symbols[head].prev
	}

	for idx := head; idx >= 0; idx = symbols[idx].next {
		sym := symbols[idx]
		if sym.empty() {
			continue
		}
		piece := sym.slice(normalized)
		m.bpeResegment(piece, revMerge, &result)
	}

	return result
}

// bpeResegment recursively splits UNUSED pieces into usable pieces.
func (m *Model) bpeResegment(piece string, revMerge map[string][2]string, result *[]encodedPiece) {
	id, ok := m.pieceIndex[piece]
	if !ok {
		// Unknown piece: byte fallback or UNK.
		if m.byteFallback {
			for i := 0; i < len(piece); i++ {
				bt := byteToPiece(piece[i])
				*result = append(*result, encodedPiece{id: m.PieceToId(bt), piece: bt})
			}
		} else {
			*result = append(*result, encodedPiece{id: m.unkID, piece: piece})
		}
		return
	}

	if m.pieces[id].Type != PieceUnused {
		*result = append(*result, encodedPiece{id: id, piece: piece})
		return
	}

	// UNUSED: split recursively.
	split, hasSplit := revMerge[piece]
	if !hasSplit {
		*result = append(*result, encodedPiece{id: m.unkID, piece: piece})
		return
	}
	m.bpeResegment(split[0], revMerge, result)
	m.bpeResegment(split[1], revMerge, result)
}
