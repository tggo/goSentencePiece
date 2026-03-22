package sentencepiece_test

import (
	"fmt"
	"log"

	sp "github.com/tggo/goSentencePiece"
)

func ExampleNewTokenizer() {
	tok, err := sp.NewTokenizer("_testdata/spm.model")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Vocab size:", tok.VocabSize())
	// Output:
	// Vocab size: 128000
}

func ExampleTokenizer_Encode() {
	tok, err := sp.NewTokenizer("_testdata/spm.model")
	if err != nil {
		log.Fatal(err)
	}

	ids, err := tok.Encode("Hello world")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("IDs:", ids)
	// Output:
	// IDs: [5365 447]
}

func ExampleTokenizer_Decode() {
	tok, err := sp.NewTokenizer("_testdata/spm.model")
	if err != nil {
		log.Fatal(err)
	}

	ids, _ := tok.Encode("Hello world")
	text, err := tok.Decode(ids)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Decoded:", text)
	// Output:
	// Decoded: Hello world
}

func ExampleTokenizer_EncodeAsPieces() {
	tok, err := sp.NewTokenizer("_testdata/spm.model")
	if err != nil {
		log.Fatal(err)
	}

	pieces, err := tok.EncodeAsPieces("Hello world")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Pieces:", pieces)
	// Output:
	// Pieces: [▁Hello ▁world]
}
