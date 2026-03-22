// Preparing inputs for mmBERT-small (jhu-clsp/mmBERT-small) ONNX inference
// using the Gemma 2 tokenizer.
//
// mmBERT-small is a ModernBERT-based multilingual encoder that uses the
// Gemma 2 tokenizer (256K vocab, BPE). This example shows how to tokenize
// text and prepare the three tensors needed for ONNX inference:
//   - input_ids:      token IDs with [CLS] and [SEP] special tokens
//   - attention_mask:  1 for real tokens, 0 for padding
//   - token_type_ids:  0 for all tokens (single-sequence input)
//
// Download the tokenizer model:
//
//	wget -O tmp/gemma-2-2b-tokenizer.model \
//	  https://huggingface.co/google/gemma-2-2b/resolve/main/tokenizer.model
//
// Run:
//
//	go run ./examples/mmbert tmp/gemma-2-2b-tokenizer.model
package main

import (
	"fmt"
	"log"
	"os"

	sp "github.com/tggo/goSentencePiece"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <tokenizer.model>\n", os.Args[0])
		os.Exit(1)
	}

	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	m := tok.Model()
	fmt.Println("Vocab size:", tok.VocabSize())
	fmt.Println("BOS (CLS):", m.BosID(), " →", m.IdToPiece(m.BosID()))
	fmt.Println("EOS (SEP):", m.EosID(), " →", m.IdToPiece(m.EosID()))
	fmt.Println("PAD:      ", m.PadID(), " →", m.IdToPiece(m.PadID()))
	fmt.Println("UNK:      ", m.UnkID(), " →", m.IdToPiece(m.UnkID()))
	fmt.Println("MASK:     ", m.PieceToId("<mask>"))
	fmt.Println()

	// --- Single text encoding ---
	text := "The quick brown fox jumps over the lazy dog."
	inputIDs, attentionMask, tokenTypeIDs := encodeForBERT(tok, text, 32)

	fmt.Printf("Text:           %q\n", text)
	fmt.Printf("input_ids:      %v\n", inputIDs)
	fmt.Printf("attention_mask: %v\n", attentionMask)
	fmt.Printf("token_type_ids: %v\n", tokenTypeIDs)
	fmt.Println()

	// --- Decode back (strips control tokens automatically) ---
	decoded, _ := tok.Decode(inputIDs)
	fmt.Printf("Decoded:        %q\n", decoded)
	fmt.Println()

	// --- Multilingual examples ---
	texts := []string{
		"Привіт, світе!",
		"这是一个测试",
		"Bonjour le monde!",
		"مرحبا بالعالم",
	}
	maxLen := 16
	fmt.Printf("Batch encoding (maxLen=%d):\n\n", maxLen)
	for _, t := range texts {
		ids, mask, _ := encodeForBERT(tok, t, maxLen)
		fmt.Printf("  %q\n    ids:  %v\n    mask: %v\n\n", t, ids, mask)
	}
}

// encodeForBERT tokenizes text and returns padded tensors ready for ONNX inference.
//
// Returns:
//   - input_ids:      [CLS] + tokens + [SEP] + [PAD...]  (length = maxLen)
//   - attention_mask:  1 for real tokens, 0 for padding    (length = maxLen)
//   - token_type_ids:  all zeros                           (length = maxLen)
//
// Tokens are truncated to maxLen-2 to leave room for CLS and SEP.
func encodeForBERT(tok *sp.Tokenizer, text string, maxLen int) ([]int, []int, []int) {
	ids, _ := tok.Encode(text)

	// Truncate if needed (reserve 2 slots for CLS + SEP).
	if len(ids) > maxLen-2 {
		ids = ids[:maxLen-2]
	}

	// Wrap with special tokens: [CLS] tokens [SEP]
	ids = tok.AddSpecialTokens(ids)

	realLen := len(ids)
	padID := tok.Model().PadID()

	inputIDs := make([]int, maxLen)
	attentionMask := make([]int, maxLen)
	tokenTypeIDs := make([]int, maxLen)

	copy(inputIDs, ids)
	for i := realLen; i < maxLen; i++ {
		inputIDs[i] = padID
	}
	for i := 0; i < realLen; i++ {
		attentionMask[i] = 1
	}
	// tokenTypeIDs stays all zeros.

	return inputIDs, attentionMask, tokenTypeIDs
}
