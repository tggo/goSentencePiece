// Preparing inputs for mmBERT-small (jhu-clsp/mmBERT-small) ONNX inference
// using the Gemma tokenizer.
//
// mmBERT-small is a ModernBERT-based multilingual encoder that uses the
// Gemma tokenizer (256K vocab, BPE). This example shows how to:
//
//  1. Load the SentencePiece tokenizer model
//  2. Configure the pipeline (post-processing, truncation, padding)
//  3. Tokenize text into ONNX-ready tensors (input_ids, attention_mask, token_type_ids)
//
// The tokenizer model is the Gemma SentencePiece model. Download it from:
//
//	# Requires HuggingFace auth (Gemma license agreement)
//	huggingface-cli download google/gemma-3-1b-it tokenizer.model --local-dir tmp
//
// Or use the test model shipped with goSentencePiece:
//
//	go run ./examples/mmbert _testdata/bpe.model
package main

import (
	"fmt"
	"log"
	"os"

	sp "github.com/tggo/goSentencePiece"
)

const maxLen = 32

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <tokenizer.model>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Download the Gemma tokenizer model:\n")
		fmt.Fprintf(os.Stderr, "  huggingface-cli download google/gemma-3-1b-it tokenizer.model --local-dir tmp\n\n")
		fmt.Fprintf(os.Stderr, "Or use the test model:\n")
		fmt.Fprintf(os.Stderr, "  go run ./examples/mmbert _testdata/bpe.model\n")
		os.Exit(1)
	}

	// ── Step 1: Load tokenizer ─────────────────────────────────────
	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	m := tok.Model()

	// Resolve pad ID: Gemma has pad_id=-1 (not set), mmBERT uses 0 (<pad>).
	padID := m.PadID()
	if padID < 0 {
		padID = m.PieceToId("<pad>")
	}

	fmt.Println("=== mmBERT-small tokenizer (Gemma) ===")
	fmt.Printf("Vocab: %d | BOS(CLS): %d | EOS(SEP): %d | PAD: %d\n\n",
		tok.VocabSize(), m.BosID(), m.EosID(), padID)

	// ── Step 2: Configure the pipeline ─────────────────────────────
	// mmBERT uses BERT-style inputs: [BOS] tokens [EOS] + padding
	tok.
		WithPostProcessor(sp.BertStylePostProcessor(m.BosID(), m.EosID())).
		WithTruncation(&sp.TruncationParams{MaxLength: maxLen}).
		WithPadding(&sp.PaddingParams{
			Strategy:  sp.PadToMaxLength,
			Direction: sp.PadRight,
			MaxLength: maxLen,
			PadID:     padID,
		})

	// ── Step 3: Single text → ONNX tensors ─────────────────────────
	text := "The quick brown fox jumps over the lazy dog."
	enc := tok.EncodeWithOptions(text, true)

	fmt.Printf("Text:           %q\n", text)
	fmt.Printf("Tokens:         %v\n", enc.Tokens)
	fmt.Printf("input_ids:      %v\n", enc.IDs)
	fmt.Printf("attention_mask: %v\n", enc.AttentionMask)
	fmt.Printf("token_type_ids: %v\n", enc.TypeIDs)
	fmt.Println()

	// Decode back (strips control tokens).
	decoded, _ := tok.Decode(enc.IDs)
	fmt.Printf("Decoded:        %q\n\n", decoded)

	// ── Step 4: Multilingual batch → ONNX tensors ──────────────────
	texts := []string{
		"Hello, world!",
		"Привіт, світе!",
		"这是一个测试",
		"Bonjour le monde!",
		"مرحبا بالعالم",
	}

	fmt.Printf("=== Batch encoding (maxLen=%d) ===\n\n", maxLen)
	encodings := tok.EncodeBatchWithOptions(texts, true)

	for i, e := range encodings {
		fmt.Printf("  %q\n", texts[i])
		fmt.Printf("    input_ids:      %v\n", e.IDs)
		fmt.Printf("    attention_mask: %v\n", e.AttentionMask)
		fmt.Println()
	}

	// ── These tensors are ready for ONNX Runtime ───────────────────
	// Pass to onnxruntime-go:
	//   session.Run(map[string]*ort.Tensor{
	//       "input_ids":      ort.NewTensor(enc.IDs),
	//       "attention_mask": ort.NewTensor(enc.AttentionMask),
	//       "token_type_ids": ort.NewTensor(enc.TypeIDs),
	//   })
	fmt.Println("✓ All tensors ready for ONNX Runtime inference")
}
