package sentencepiece_test

import (
	"bytes"
	"embed"
	"fmt"
	"log"

	sp "github.com/tggo/goSentencePiece"
)

//go:embed _testdata/spm.model
var modelData []byte

func Example_embed() {
	_ = embed.FS{} // ensure embed import is used

	tok, err := sp.NewTokenizerFromReader(bytes.NewReader(modelData))
	if err != nil {
		log.Fatal(err)
	}

	ids, _ := tok.Encode("Hello world")
	fmt.Println("IDs:", ids)
	// Output:
	// IDs: [5365 447]
}
