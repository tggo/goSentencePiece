// Basic usage of the goSentencePiece tokenizer.
//
// Run:
//
//	go run ./examples/basic _testdata/spm.model
package main

import (
	"fmt"
	"log"
	"os"

	sp "github.com/tggo/goSentencePiece"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model.model>\n", os.Args[0])
		os.Exit(1)
	}

	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Vocab size:", tok.VocabSize())
	fmt.Println("BOS ID:", tok.Model().BosID())
	fmt.Println("EOS ID:", tok.Model().EosID())
	fmt.Println()

	texts := []string{
		"Hello world",
		"Привіт світ",
		"The quick brown fox jumps over the lazy dog.",
		"func main() { fmt.Println(\"hello\") }",
	}

	for _, text := range texts {
		ids, _ := tok.Encode(text)
		pieces, _ := tok.EncodeAsPieces(text)
		decoded, _ := tok.Decode(ids)

		fmt.Printf("Input:   %q\n", text)
		fmt.Printf("IDs:     %v\n", ids)
		fmt.Printf("Pieces:  %v\n", pieces)
		fmt.Printf("Decoded: %q\n", decoded)
		fmt.Println()
	}

	// Batch encoding
	results, _ := tok.EncodeBatch(texts)
	fmt.Printf("Batch encoded %d texts, total tokens: ", len(results))
	total := 0
	for _, ids := range results {
		total += len(ids)
	}
	fmt.Println(total)
}
