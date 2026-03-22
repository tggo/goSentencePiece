//go:build ignore

// Embedding a SentencePiece model in the binary using go:embed.
//
// Before running, copy your model file here:
//
//	cp _testdata/spm.model examples/embed/spm.model
//
// Then run:
//
//	go run ./examples/embed/main.go
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"log"

	sp "github.com/tggo/goSentencePiece"
)

//go:embed spm.model
var modelData []byte

func main() {
	tok, err := sp.NewTokenizerFromReader(bytes.NewReader(modelData))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Loaded model from embedded data (%d bytes)\n", len(modelData))
	fmt.Println("Vocab size:", tok.VocabSize())
	fmt.Println()

	text := "Embedding models in Go binaries is convenient!"
	ids, _ := tok.Encode(text)
	pieces, _ := tok.EncodeAsPieces(text)

	fmt.Printf("Input:  %q\n", text)
	fmt.Printf("IDs:    %v\n", ids)
	fmt.Printf("Pieces: %v\n", pieces)
}
