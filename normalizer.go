package sentencepiece

import (
	"strings"
	"unicode/utf8"
)

const metaSpaceRune = '\u2581' // ▁ LOWER ONE EIGHTH BLOCK
const metaSpace = "▁"

// Normalizer handles text normalization according to the model's NormalizerSpec.
// It applies the precompiled character map (Darts double-array trie encoding
// NFKC and custom rules), whitespace deduplication, dummy prefix insertion,
// and space-to-metaspace escaping.
type Normalizer struct {
	addDummyPrefix        bool
	removeExtraWhitespace bool
	escapeWhitespaces     bool
	trie                  *DartsDoubleArray
	normalized            []byte // Null-terminated strings concatenated.
}

// NewNormalizer creates a Normalizer from the model's normalization settings,
// decoding the precompiled charsmap if present.
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

// Normalize applies the full SentencePiece normalization pipeline to the input
// string. It follows the exact algorithm from normalizer.cc: skip leading
// whitespace, add dummy prefix, apply charsmap replacements, deduplicate
// whitespace, escape spaces to metaspace, and trim trailing whitespace.
func (n *Normalizer) Normalize(input string) string {
	if len(input) == 0 {
		return ""
	}

	remaining := n.skipLeadingWhitespace(input)
	if len(remaining) == 0 {
		return ""
	}

	var buf strings.Builder
	buf.Grow(len(input) + 3)

	n.writeDummyPrefix(&buf)
	n.processChars(&buf, remaining)

	return n.trimTrailingSpaces(buf.String())
}

// skipLeadingWhitespace consumes leading chars that normalize to space.
func (n *Normalizer) skipLeadingWhitespace(input string) string {
	if !n.removeExtraWhitespace {
		return input
	}
	remaining := input
	for len(remaining) > 0 {
		consumed, replacement := n.normalizePrefix([]byte(remaining))
		if consumed == 0 || replacement != " " {
			break
		}
		remaining = remaining[consumed:]
	}
	return remaining
}

// writeDummyPrefix writes the prefix space/metaspace if configured.
func (n *Normalizer) writeDummyPrefix(buf *strings.Builder) {
	if !n.addDummyPrefix {
		return
	}
	if n.escapeWhitespaces {
		buf.WriteString(metaSpace)
	} else {
		buf.WriteByte(' ')
	}
}

// processChars processes the remaining input character by character through the
// charsmap, deduplicating spaces and escaping whitespace as configured.
func (n *Normalizer) processChars(buf *strings.Builder, remaining string) {
	isPrevSpace := n.removeExtraWhitespace

	for len(remaining) > 0 {
		consumed, sp := n.normalizePrefix([]byte(remaining))
		if consumed == 0 {
			break
		}

		// Remove heading spaces if previous output was space.
		if isPrevSpace {
			for strings.HasPrefix(sp, " ") {
				sp = sp[1:]
			}
		}

		if len(sp) > 0 {
			n.writeReplacement(buf, sp)
			isPrevSpace = sp[len(sp)-1] == ' '
		}

		remaining = remaining[consumed:]
		if !n.removeExtraWhitespace {
			isPrevSpace = false
		}
	}
}

// writeReplacement writes a charsmap replacement to buf, escaping spaces to
// metaspace if configured.
func (n *Normalizer) writeReplacement(buf *strings.Builder, sp string) {
	for i := 0; i < len(sp); i++ {
		if n.escapeWhitespaces && sp[i] == ' ' {
			buf.WriteString(metaSpace)
		} else {
			buf.WriteByte(sp[i])
		}
	}
}

// trimTrailingSpaces removes trailing escaped/raw spaces from the result.
func (n *Normalizer) trimTrailingSpaces(result string) string {
	if !n.removeExtraWhitespace {
		return result
	}
	space := metaSpace
	if !n.escapeWhitespaces {
		space = " "
	}
	for strings.HasSuffix(result, space) {
		result = result[:len(result)-len(space)]
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
		matches := n.trie.CommonPrefixSearchBytes(input)
		if len(matches) > 0 {
			best := matches[len(matches)-1]
			consumed := best.Length

			offset := best.Value
			if offset < len(n.normalized) {
				end := offset
				for end < len(n.normalized) && n.normalized[end] != 0 {
					end++
				}
				return consumed, string(n.normalized[offset:end])
			}
		}
	}

	// No match: consume one Unicode character, pass through as-is.
	// The charsmap already contains all NFKC mappings, so unmatched
	// chars should not be NFKC-normalized again.
	r, size := utf8.DecodeRune(input)
	if r == utf8.RuneError && size <= 1 {
		return 1, "\uFFFD"
	}

	return size, string(input[:size])
}
