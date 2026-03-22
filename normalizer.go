package sentencepiece

import (
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const metaSpaceRune = '\u2581' // ▁ LOWER ONE EIGHTH BLOCK
const metaSpace = "▁"

// Normalizer handles text normalization according to the model's NormalizerSpec.
type Normalizer struct {
	addDummyPrefix        bool
	removeExtraWhitespace bool
	escapeWhitespaces     bool
	trie                  *DartsDoubleArray
	normalized            []byte // Null-terminated strings concatenated.
}

// NewNormalizer creates a normalizer from model config.
func NewNormalizer(m *Model) *Normalizer {
	n := &Normalizer{
		addDummyPrefix:        m.addDummyPrefix,
		removeExtraWhitespace: m.removeExtraWhitespace,
		escapeWhitespaces:     m.escapeWhitespaces,
	}

	if len(m.precompiledCharsmap) > 0 {
		n.decodePrecompiledCharsmap(m.precompiledCharsmap)
	}

	return n
}

// decodePrecompiledCharsmap decodes the binary precompiled charsmap.
func (n *Normalizer) decodePrecompiledCharsmap(data []byte) {
	if len(data) < 4 {
		return
	}

	trieSize := int(uint32(data[0]) | uint32(data[1])<<8 | uint32(data[2])<<16 | uint32(data[3])<<24)
	if trieSize < 0 || 4+trieSize > len(data) {
		return
	}

	n.trie = NewDartsDoubleArray(data[4 : 4+trieSize])
	n.normalized = data[4+trieSize:]
}

// Normalize applies the full SentencePiece normalization pipeline.
// This follows the exact algorithm from normalizer.cc.
func (n *Normalizer) Normalize(input string) string {
	if len(input) == 0 {
		return ""
	}

	remaining := input

	// Step 1: Skip leading whitespace (if removeExtraWhitespace).
	if n.removeExtraWhitespace {
		for len(remaining) > 0 {
			consumed, replacement := n.normalizePrefix([]byte(remaining))
			if consumed == 0 {
				break
			}
			if replacement != " " {
				break
			}
			remaining = remaining[consumed:]
		}
	}

	// All chars are whitespace.
	if len(remaining) == 0 {
		return ""
	}

	var buf strings.Builder
	buf.Grow(len(input) + 3)

	// Step 2: Add dummy prefix.
	if n.addDummyPrefix {
		if n.escapeWhitespaces {
			buf.WriteString(metaSpace)
		} else {
			buf.WriteByte(' ')
		}
	}

	// Step 3: Process remaining characters.
	isPrevSpace := n.removeExtraWhitespace

	for len(remaining) > 0 {
		consumed, sp := n.normalizePrefix([]byte(remaining))
		if consumed == 0 {
			break
		}

		// Remove heading spaces from the replacement if previous was space.
		if isPrevSpace {
			for strings.HasPrefix(sp, " ") {
				sp = sp[1:]
			}
		}

		if len(sp) > 0 {
			// Write each byte, replacing ' ' with ▁ if escapeWhitespaces.
			for i := 0; i < len(sp); i++ {
				if n.escapeWhitespaces && sp[i] == ' ' {
					buf.WriteString(metaSpace)
				} else {
					buf.WriteByte(sp[i])
				}
			}
			// Check if the replacement ends with space.
			isPrevSpace = sp[len(sp)-1] == ' '
		}

		remaining = remaining[consumed:]
		if !n.removeExtraWhitespace {
			isPrevSpace = false
		}
	}

	result := buf.String()

	// Step 4: Remove trailing escaped spaces.
	if n.removeExtraWhitespace {
		space := metaSpace
		if !n.escapeWhitespaces {
			space = " "
		}
		for strings.HasSuffix(result, space) {
			result = result[:len(result)-len(space)]
		}
	}

	return result
}

// normalizePrefix finds the longest matching prefix in the charsmap.
// Returns (bytes consumed, replacement string).
func (n *Normalizer) normalizePrefix(input []byte) (int, string) {
	if len(input) == 0 {
		return 0, ""
	}

	if n.trie != nil {
		// Find all prefix matches.
		matches := n.trie.CommonPrefixSearchBytes(input)

		if len(matches) > 0 {
			// Use the longest match.
			best := matches[len(matches)-1]
			consumed := best.Length

			// Extract the replacement string from normalized data.
			offset := best.Value
			if offset < len(n.normalized) {
				end := offset
				for end < len(n.normalized) && n.normalized[end] != 0 {
					end++
				}
				replacement := string(n.normalized[offset:end])
				return consumed, replacement
			}
		}
	}

	// No match: consume one Unicode character, NFKC normalize it.
	r, size := utf8.DecodeRune(input)
	if r == utf8.RuneError && size <= 1 {
		// Invalid UTF-8: replace with U+FFFD.
		return 1, "\uFFFD"
	}

	normalized := norm.NFKC.String(string(input[:size]))
	return size, normalized
}
