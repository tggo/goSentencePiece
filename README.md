# goSentencePiece

Pure Go implementation of the [SentencePiece](https://github.com/google/sentencepiece) Unigram tokenizer. Produces **byte-identical output** to the C++ / Python `sentencepiece` library.

Built for running DeBERTa v3 (and other Unigram-based models) in Go services without CGo or Python dependencies.

## Features

- **Pure Go** — no CGo, no Rust FFI, no external C libraries
- **Byte-identical** to the reference C++ implementation (validated against 505 golden test cases)
- **Unigram model** with Viterbi decoding
- **Byte fallback** (`<0xHH>` tokens) for characters not in vocabulary
- **Precompiled charsmap** normalization via Darts double-array trie (NFKC + custom rules)
- **Fast**: short strings ~330ns, medium ~3.5μs, long (4500 chars) ~100μs
- Zero runtime dependencies beyond stdlib + `google.golang.org/protobuf` + `golang.org/x/text`

## Installation

```bash
go get github.com/promova/sentencepiece
```

Requires Go 1.22+.

## Usage

```go
package main

import (
    "fmt"
    "log"

    sp "github.com/promova/sentencepiece"
)

func main() {
    tok, err := sp.NewTokenizer("spm.model")
    if err != nil {
        log.Fatal(err)
    }

    // Encode text to token IDs
    ids, _ := tok.Encode("Hello world")
    fmt.Println("IDs:", ids)

    // Encode text to string pieces
    pieces, _ := tok.EncodeAsPieces("Hello world")
    fmt.Println("Pieces:", pieces)

    // Decode token IDs back to text
    text, _ := tok.Decode(ids)
    fmt.Println("Decoded:", text)

    // Wrap with special tokens (BOS/EOS)
    wrapped := tok.AddSpecialTokens(ids)
    fmt.Println("With special tokens:", wrapped)

    fmt.Println("Vocab size:", tok.VocabSize())
}
```

## API

```go
// Create a tokenizer from a .model file
func NewTokenizer(modelPath string) (*Tokenizer, error)

// Encode text to token IDs
func (t *Tokenizer) Encode(text string) ([]int, error)

// Encode text to string pieces
func (t *Tokenizer) EncodeAsPieces(text string) ([]string, error)

// Decode token IDs back to text
func (t *Tokenizer) Decode(ids []int) (string, error)

// Wrap with BOS/EOS tokens
func (t *Tokenizer) AddSpecialTokens(ids []int) []int

// Get vocabulary size
func (t *Tokenizer) VocabSize() int
```

## Supported models

Any SentencePiece `.model` file that uses the **Unigram** model type. Tested with:

- `microsoft/deberta-v3-small`
- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`

Other Unigram models (XLNet, ALBERT, T5, etc.) should work but are not yet tested.

**Note:** BPE models are not supported.

## Running tests

The test suite requires the SentencePiece model file. Download it first:

```bash
# Set up Python venv and download model
make venv
make golden

# Run tests
make test

# Run benchmarks
make bench

# Run fuzz tests (60s)
make fuzz
```

## Benchmarks

Measured on Apple M1 Pro:

| Input | Time | Allocs |
|-------|------|--------|
| Short (11 chars) | 331 ns | 5 |
| Medium (120 chars) | 3.5 μs | 10 |
| Long (4500 chars) | 102 μs | 15 |

## Architecture

```
sentencepiece.go    — public Tokenizer type and constructor
model.go            — protobuf loading, vocab index, ByteTrie
normalizer.go       — precompiled charsmap (Darts trie), NFKC, whitespace
unigram.go          — Viterbi decoding (forward DP + backtrack)
encoder.go          — Encode/Decode with byte-token handling
byte_fallback.go    — <0xHH> token encoding/decoding
trie.go             — ByteTrie (vocab), DartsDoubleArray (charsmap)
proto/              — generated protobuf code
```

## How it works

1. **Normalization**: Input text is normalized using the model's precompiled character map (a Darts double-array trie that encodes NFKC and custom rules). Whitespace is deduplicated, a prefix space is added, and spaces are replaced with `▁` (U+2581).

2. **Viterbi tokenization**: The normalized text is segmented into pieces using dynamic programming. A byte-level trie is traversed to find all vocabulary pieces starting at each position. The algorithm finds the segmentation that maximizes total log-probability.

3. **Byte fallback**: Characters not covered by any vocabulary piece are encoded as individual UTF-8 bytes using `<0xHH>` tokens.

4. **Decoding**: Token IDs are mapped back to piece strings. Byte tokens are accumulated and flushed as UTF-8. The `▁` prefix is converted back to spaces.

## License

MIT — see [LICENSE](LICENSE).
