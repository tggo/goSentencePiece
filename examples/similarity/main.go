// Encode two texts and compute Jaccard similarity of their token ID sets.
//
// Usage:
//
//	go run ./examples/similarity _testdata/spm.model ["text1" "text2"]
package main

import (
	"fmt"
	"log"
	"os"

	sp "github.com/tggo/goSentencePiece"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model> [text1 text2]\n", os.Args[0])
		os.Exit(1)
	}

	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	text1 := "The cat sat on the mat"
	text2 := "The dog sat on the rug"
	if len(os.Args) >= 4 {
		text1, text2 = os.Args[2], os.Args[3]
	}

	ids1, _ := tok.Encode(text1)
	ids2, _ := tok.Encode(text2)
	pieces1, _ := tok.EncodeAsPieces(text1)
	pieces2, _ := tok.EncodeAsPieces(text2)

	set1 := make(map[int]bool, len(ids1))
	for _, id := range ids1 {
		set1[id] = true
	}
	set2 := make(map[int]bool, len(ids2))
	for _, id := range ids2 {
		set2[id] = true
	}

	var shared []int
	for id := range set1 {
		if set2[id] {
			shared = append(shared, id)
		}
	}
	union := len(set1) + len(set2) - len(shared)
	jaccard := float64(len(shared)) / float64(union)

	fmt.Printf("Text 1: %q\n  Pieces: %v\n  IDs:    %v\n\n", text1, pieces1, ids1)
	fmt.Printf("Text 2: %q\n  Pieces: %v\n  IDs:    %v\n\n", text2, pieces2, ids2)
	fmt.Printf("Shared token IDs: %v (%d unique)\n", shared, len(shared))
	fmt.Printf("Union size:       %d\n", union)
	fmt.Printf("Jaccard similarity: %.4f\n", jaccard)
}
