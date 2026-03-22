package sentencepiece

import "encoding/binary"

// ByteTrie is a byte-level trie used for vocabulary prefix matching during
// Viterbi tokenization. Each node branches on a single byte value (0-255),
// and terminal nodes store the vocabulary piece ID. The trie supports
// incremental byte-by-byte traversal via Traverse, matching the Darts-clone
// traverse() behavior used by the reference SentencePiece C++ implementation.
type ByteTrie struct {
	children [256]*ByteTrie
	pieceID  int  // -1 if not a terminal node
	hasValue bool // true if this node represents a piece
}

// NewByteTrie creates a new empty ByteTrie with no children and pieceID -1
// (indicating a non-terminal node).
func NewByteTrie() *ByteTrie {
	return &ByteTrie{
		pieceID: -1,
	}
}

// Insert adds a piece (as byte sequence) to the trie.
func (t *ByteTrie) Insert(piece string, id int) {
	node := t
	for i := 0; i < len(piece); i++ {
		b := piece[i]
		child := node.children[b]
		if child == nil {
			child = NewByteTrie()
			node.children[b] = child
		}
		node = child
	}
	node.pieceID = id
	node.hasValue = true
}

// Traverse advances one byte through the trie.
// nodePos should point to a *ByteTrie (stored externally as interface).
// Returns:
//
//	-2 if no transition exists
//	-1 if transition exists but not terminal
//	>= 0 the piece ID at this node
func (t *ByteTrie) Traverse(b byte) (*ByteTrie, int) {
	child := t.children[b]
	if child == nil {
		return nil, -2
	}
	if child.hasValue {
		return child, child.pieceID
	}
	return child, -1
}

// DartsDoubleArray implements the Darts-clone double-array trie data structure
// used by SentencePiece's precompiled charsmap for Unicode normalization. It
// stores NFKC and custom character mapping rules in a compact array of uint32
// units, enabling efficient CommonPrefixSearch over byte sequences.
type DartsDoubleArray struct {
	units []uint32
}

// NewDartsDoubleArray creates a new DartsDoubleArray from the raw binary trie
// data extracted from the precompiled charsmap. The data is interpreted as a
// sequence of little-endian uint32 values representing the double-array units.
func NewDartsDoubleArray(data []byte) *DartsDoubleArray {
	numUnits := len(data) / 4
	units := make([]uint32, numUnits)
	for i := 0; i < numUnits; i++ {
		units[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	return &DartsDoubleArray{units: units}
}

// dartsOffset extracts the offset from a Darts unit value.
// offset = (unit >> 10) << ((unit & 0x200) >> 6)
func dartsOffset(unit uint32) uint32 {
	return (unit >> 10) << ((unit & 0x200) >> 6)
}

// dartsHasLeaf returns whether this node has a leaf (terminal value).
func dartsHasLeaf(unit uint32) bool {
	return ((unit >> 8) & 1) == 1
}

// dartsValue extracts the value from a leaf unit.
func dartsValue(unit uint32) int {
	return int(unit & 0x7FFFFFFF)
}

// dartsLabel extracts the label (check byte) from a unit.
func dartsLabel(unit uint32) uint32 {
	return unit & ((1 << 31) | 0xFF)
}

// DartsMatch holds a single match result from CommonPrefixSearchBytes. Value
// is the offset into the normalized replacement table, and Length is the number
// of input bytes consumed by this match.
type DartsMatch struct {
	Value  int
	Length int
}

// CommonPrefixSearchBytes finds all prefix matches in the double-array trie for
// the given byte slice. It returns matches in order of increasing length,
// following the exact Darts-clone commonPrefixSearch algorithm. Each match
// contains the replacement table offset (Value) and the number of matched
// input bytes (Length). Returns nil if the trie is empty.
func (d *DartsDoubleArray) CommonPrefixSearchBytes(key []byte) []DartsMatch {
	var results []DartsMatch
	n := len(d.units)
	if n == 0 {
		return nil
	}

	// Start at root (position 0).
	nodePos := 0
	unit := d.units[nodePos]
	// XOR with root's offset.
	nodePos ^= int(dartsOffset(unit))

	for i := 0; i < len(key); i++ {
		b := uint32(key[i])

		// XOR with key byte.
		p := nodePos ^ int(b)
		if p < 0 || p >= n {
			break
		}

		unit = d.units[p]
		// Check label.
		if dartsLabel(unit) != b {
			break
		}

		// XOR with this unit's offset.
		nodePos = p ^ int(dartsOffset(unit))

		// Check for leaf.
		if dartsHasLeaf(unit) {
			if nodePos >= 0 && nodePos < n {
				value := dartsValue(d.units[nodePos])
				results = append(results, DartsMatch{Value: value, Length: i + 1})
			}
		}
	}

	return results
}
