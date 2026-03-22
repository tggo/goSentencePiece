# goSentencePiece

[![CI](https://github.com/tggo/goSentencePiece/actions/workflows/ci.yml/badge.svg)](https://github.com/tggo/goSentencePiece/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/tggo/goSentencePiece.svg)](https://pkg.go.dev/github.com/tggo/goSentencePiece)
[![Go Report Card](https://goreportcard.com/badge/github.com/tggo/goSentencePiece)](https://goreportcard.com/report/github.com/tggo/goSentencePiece)
[![Coverage](https://img.shields.io/badge/coverage-84.9%25-brightgreen)](https://github.com/tggo/goSentencePiece)

Pure Go implementation of the [SentencePiece](https://github.com/google/sentencepiece) Unigram tokenizer. Produces **byte-identical output** to the C++ / Python `sentencepiece` library -- no CGo, no Rust FFI, no external C libraries.

Built for running DeBERTa v3 (and other Unigram-based models) in Go services.

## Installation

```bash
go get github.com/tggo/goSentencePiece
```

Requires Go 1.22+.

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    sp "github.com/tggo/goSentencePiece"
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

## Features

- **Pure Go** -- zero CGo, zero Rust FFI, zero external C libraries
- **Byte-identical** to the reference C++ implementation (validated against 505 golden test cases)
- **Unigram model** with Viterbi decoding
- **Byte fallback** (`<0xHH>` tokens) for characters not in vocabulary
- **Precompiled charsmap** normalization via Darts double-array trie (NFKC + custom rules)
- **io.Reader support** -- load models from embedded files, HTTP responses, or any reader
- **Fast** -- see benchmarks below
- Zero runtime dependencies beyond stdlib + `google.golang.org/protobuf`

## API

### Tokenizer

```go
// Create from file path
func NewTokenizer(modelPath string) (*Tokenizer, error)

// Create from io.Reader (embedded files, HTTP responses, etc.)
func NewTokenizerFromReader(r io.Reader) (*Tokenizer, error)

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

### Model

```go
// Load model from file
func LoadModel(path string) (*Model, error)

// Load model from reader
func LoadModelFromReader(r io.Reader) (*Model, error)

// Vocabulary lookup
func (m *Model) VocabSize() int
func (m *Model) IdToPiece(id int) string
func (m *Model) PieceToId(piece string) int
```

## Supported Models

Any SentencePiece `.model` file that uses the **Unigram** model type. Tested with:

- `microsoft/deberta-v3-small`
- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`

Other Unigram models (XLNet, ALBERT, T5, etc.) should work but are not yet tested.

**Note:** BPE models are not supported.

## Benchmarks

Measured on Apple M1 Pro:

| Input | Go | Allocs |
|-------|-----|--------|
| Short (11 chars) | ~330 ns | 5 |
| Medium (120 chars) | ~3.5 us | 10 |
| Long (4500 chars) | ~102 us | 15 |

Run benchmarks yourself:

```bash
make bench
```

To compare with Python:

```bash
make venv
.venv/bin/python _testdata/bench_python.py
```

## Project Structure

```
sentencepiece.go    -- public Tokenizer type and constructors
model.go            -- protobuf loading, vocab index, ByteTrie
normalizer.go       -- precompiled charsmap (Darts trie), NFKC, whitespace
unigram.go          -- Viterbi decoding (forward DP + backtrack)
encoder.go          -- Encode/Decode with byte-token handling
byte_fallback.go    -- <0xHH> token encoding/decoding
trie.go             -- ByteTrie (vocab), DartsDoubleArray (charsmap)
proto/              -- generated protobuf code
_testdata/          -- test model and golden test cases
```

## How It Works

1. **Normalization**: Input text is normalized using the model's precompiled character map (a Darts double-array trie that encodes NFKC and custom rules). Whitespace is deduplicated, a prefix space is added, and spaces are replaced with the metaspace character.

2. **Viterbi tokenization**: The normalized text is segmented into pieces using dynamic programming. A byte-level trie is traversed to find all vocabulary pieces starting at each position. The algorithm finds the segmentation that maximizes total log-probability.

3. **Byte fallback**: Characters not covered by any vocabulary piece are encoded as individual UTF-8 bytes using `<0xHH>` tokens.

4. **Decoding**: Token IDs are mapped back to piece strings. Byte tokens are accumulated and flushed as UTF-8. The metaspace prefix is converted back to spaces.

## Running Tests

```bash
# Set up Python venv and download model + golden data
make venv
make golden

# Run tests
make test

# Run benchmarks
make bench

# Run fuzz tests (60s)
make fuzz

# Run linters
make lint

# Run tests with coverage
make cover
```

## License

MIT -- see [LICENSE](LICENSE).
