// Compare tokenization between two models (e.g. Unigram vs BPE).
//
// Usage:
//
//	go run ./examples/compare _testdata/spm.model _testdata/bpe.model
package main

import (
	"fmt"
	"log"
	"os"

	sp "github.com/tggo/goSentencePiece"
)

func modelLabel(tok *sp.Tokenizer) string {
	switch tok.Model().Type() {
	case sp.ModelTypeBPE:
		return "BPE"
	default:
		return "Unigram"
	}
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: %s <model1> <model2>\n", os.Args[0])
		os.Exit(1)
	}

	tok1, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatalf("model1: %v", err)
	}
	tok2, err := sp.NewTokenizer(os.Args[2])
	if err != nil {
		log.Fatalf("model2: %v", err)
	}

	label1 := fmt.Sprintf("%s (%dK)", modelLabel(tok1), tok1.VocabSize()/1000)
	label2 := fmt.Sprintf("%s (%dK)", modelLabel(tok2), tok2.VocabSize()/1000)
	fmt.Printf("Comparing: %s vs %s\n\n", label1, label2)

	texts := []struct {
		tag, text string
	}{
		{"English", "The quick brown fox jumps over the lazy dog."},
		{"Ukrainian", "Привіт, як справи? Все добре, дякую!"},
		{"Code", "func main() { fmt.Println(\"hello world\") }"},
		{"Emoji", "Hello! 🌍🔥🚀 How are you? 😊"},
		{"Long word", "Supercalifragilisticexpialidocious and antidisestablishmentarianism"},
	}

	for _, tc := range texts {
		p1, _ := tok1.EncodeAsPieces(tc.text)
		p2, _ := tok2.EncodeAsPieces(tc.text)
		ids1, _ := tok1.Encode(tc.text)
		ids2, _ := tok2.Encode(tc.text)

		fmt.Printf("--- %s ---\n", tc.tag)
		fmt.Printf("  Input: %q\n", tc.text)
		fmt.Printf("  %-16s %3d tokens: %v\n", label1, len(ids1), p1)
		fmt.Printf("  %-16s %3d tokens: %v\n", label2, len(ids2), p2)

		diff := len(ids1) - len(ids2)
		if diff > 0 {
			fmt.Printf("  %s uses %d fewer tokens\n", label2, diff)
		} else if diff < 0 {
			fmt.Printf("  %s uses %d fewer tokens\n", label1, -diff)
		} else {
			fmt.Printf("  Same token count\n")
		}
		fmt.Println()
	}
}
