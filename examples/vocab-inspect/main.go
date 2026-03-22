// Inspect a SentencePiece model vocabulary.
//
// Usage:
//
//	go run ./examples/vocab-inspect _testdata/spm.model [search-term]
package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	sp "github.com/tggo/goSentencePiece"
)

func typeName(t sp.PieceType) string {
	switch t {
	case sp.PieceNormal:
		return "NORMAL"
	case sp.PieceUnknown:
		return "UNKNOWN"
	case sp.PieceControl:
		return "CONTROL"
	case sp.PieceUserDefined:
		return "USER"
	case sp.PieceUnused:
		return "UNUSED"
	case sp.PieceByte:
		return "BYTE"
	default:
		return fmt.Sprintf("TYPE(%d)", t)
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model> [search-term]\n", os.Args[0])
		os.Exit(1)
	}

	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	m := tok.Model()
	vs := m.VocabSize()

	mt := "Unigram"
	if m.Type() == sp.ModelTypeBPE {
		mt = "BPE"
	}
	fmt.Printf("Model type: %s\nVocab size: %d\n", mt, vs)
	fmt.Printf("Special IDs — UNK: %d, BOS: %d, EOS: %d, PAD: %d\n\n", m.UnkID(), m.BosID(), m.EosID(), m.PadID())

	show := func(label string, start, end int) {
		fmt.Printf("=== %s ===\n", label)
		for i := start; i < end && i < vs; i++ {
			p := m.GetPiece(i)
			fmt.Printf("  [%6d] %-20q  score=%8.4f  type=%s\n", i, p.Piece, p.Score, typeName(p.Type))
		}
	}
	show("First 10 pieces", 0, 10)
	fmt.Println()
	show("Last 10 pieces", vs-10, vs)

	var byteCount, controlCount int
	var controls []string
	for i := 0; i < vs; i++ {
		p := m.GetPiece(i)
		if p.Type == sp.PieceByte {
			byteCount++
		}
		if p.Type == sp.PieceControl {
			controlCount++
			controls = append(controls, fmt.Sprintf("[%d] %q", i, p.Piece))
		}
	}
	fmt.Printf("\nByte tokens:    %d\n", byteCount)
	fmt.Printf("Control tokens: %d  %s\n", controlCount, strings.Join(controls, ", "))

	if len(os.Args) >= 3 {
		term := os.Args[2]
		fmt.Printf("\n=== Search: %q ===\n", term)
		found := 0
		for i := 0; i < vs; i++ {
			p := m.GetPiece(i)
			if strings.Contains(p.Piece, term) {
				fmt.Printf("  [%6d] %-20q  score=%8.4f  type=%s\n", i, p.Piece, p.Score, typeName(p.Type))
				found++
				if found >= 20 {
					fmt.Println("  ... (truncated)")
					break
				}
			}
		}
		if found == 0 {
			fmt.Println("  (no matches)")
		}
	}
}
