package sentencepiece

import "encoding/binary"

// ByteTrie is a byte-level trie for vocabulary matching.
// It supports incremental byte-by-byte traversal matching the
// Darts-clone traverse() behavior used by the reference.
type ByteTrie struct {
	children [256]*ByteTrie
	pieceID  int  // -1 if not a terminal node
	hasValue bool // true if this node represents a piece
}

// NewByteTrie creates a new empty byte trie.
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

// DartsDoubleArray implements the Darts-clone double-array trie
// used by SentencePiece's precompiled charsmap.
type DartsDoubleArray struct {
	units []uint32
}

// NewDartsDoubleArray creates a new Darts double-array trie from raw data.
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

// DartsMatch holds a match result from CommonPrefixSearchBytes.
type DartsMatch struct {
	Value  int
	Length int
}

// CommonPrefixSearchBytes finds all prefix matches in the trie for the given bytes.
// This follows the exact Darts-clone commonPrefixSearch algorithm.
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
