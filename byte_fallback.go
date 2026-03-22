package sentencepiece

import "fmt"

// byteToPiece converts a byte value to a SentencePiece byte token string.
// E.g., 0x41 → "<0x41>"
func byteToPiece(b byte) string {
	return fmt.Sprintf("<0x%02X>", b)
}

// pieceIsByte checks if a piece string is a byte token.
func pieceIsByte(piece string) bool {
	if len(piece) != 6 {
		return false
	}
	return piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>'
}

// pieceToByte extracts the byte value from a byte token piece.
// Returns the byte and true if the piece is a byte token, false otherwise.
func pieceToByte(piece string) (byte, bool) {
	if !pieceIsByte(piece) {
		return 0, false
	}
	hi := unhex(piece[3])
	lo := unhex(piece[4])
	if hi < 0 || lo < 0 {
		return 0, false
	}
	return byte(hi<<4 | lo), true
}

func unhex(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'A' && c <= 'F':
		return int(c - 'A' + 10)
	case c >= 'a' && c <= 'f':
		return int(c - 'a' + 10)
	default:
		return -1
	}
}
