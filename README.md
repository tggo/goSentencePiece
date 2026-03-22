# goSentencePiece

[![CI](https://github.com/tggo/goSentencePiece/actions/workflows/ci.yml/badge.svg)](https://github.com/tggo/goSentencePiece/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/tggo/goSentencePiece.svg)](https://pkg.go.dev/github.com/tggo/goSentencePiece)
[![Go Report Card](https://goreportcard.com/badge/github.com/tggo/goSentencePiece)](https://goreportcard.com/report/github.com/tggo/goSentencePiece)
[![Coverage](https://img.shields.io/badge/coverage-97.4%25-brightgreen)](https://github.com/tggo/goSentencePiece)
[![Golden Tests](https://img.shields.io/badge/golden_tests-5155-brightgreen)](https://github.com/tggo/goSentencePiece)

Pure Go implementation of the [SentencePiece](https://github.com/google/sentencepiece) tokenizer (Unigram and BPE). Produces **byte-identical output** to the C++ / Python `sentencepiece` library -- no CGo, no Rust FFI, no external C libraries.

Built for running DeBERTa v3, Gemma, LLaMA, and other SentencePiece-based models in Go services.

## Installation

```bash
go get github.com/tggo/goSentencePiece
```

Requires Go 1.23+.

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
- **Byte-identical** to the reference C++ implementation (validated against 5155 golden test cases)
- **Dual format** -- loads both SentencePiece `.model` (protobuf) and HuggingFace `tokenizer.json`
- **Unigram model** with Viterbi decoding
- **BPE model** with greedy best-first merging
- **Byte fallback** (`<0xHH>` tokens) for characters not in vocabulary
- **Precompiled charsmap** normalization via Darts double-array trie (NFKC + custom rules)
- **Metaspace pre-tokenization** for HuggingFace tokenizer.json models
- **ML pipeline** -- post-processing, padding, truncation, attention masks
- **Batch encoding** -- encode multiple texts with automatic padding
- **ONNX-ready** -- produces `input_ids`, `attention_mask`, `token_type_ids` tensors
- **go:embed support** -- load models from embedded files with `NewTokenizerFromReader`
- **Typed errors** -- sentinel errors for invalid or unsupported models
- **Fast** -- see benchmarks below
- Zero runtime dependencies beyond stdlib + `google.golang.org/protobuf`

## API

### Tokenizer

```go
// Create from file path (auto-detects .model or tokenizer.json format)
func NewTokenizer(path string) (*Tokenizer, error)

// Create from io.Reader (auto-detects format)
func NewTokenizerFromReader(r io.Reader) (*Tokenizer, error)

// Create from HuggingFace tokenizer.json explicitly
func NewTokenizerFromJSON(path string) (*Tokenizer, error)
func NewTokenizerFromJSONReader(r io.Reader) (*Tokenizer, error)

// Encode text to token IDs
func (t *Tokenizer) Encode(text string) ([]int, error)

// Encode text to string pieces
func (t *Tokenizer) EncodeAsPieces(text string) ([]string, error)

// Decode token IDs back to text
func (t *Tokenizer) Decode(ids []int) (string, error)

// Encode multiple texts at once
func (t *Tokenizer) EncodeBatch(texts []string) ([][]int, error)

// Wrap with BOS/EOS tokens
func (t *Tokenizer) AddSpecialTokens(ids []int) []int

// Get vocabulary size
func (t *Tokenizer) VocabSize() int

// Access the underlying model
func (t *Tokenizer) Model() *Model

// Pipeline configuration (builder pattern)
func (t *Tokenizer) WithPostProcessor(pp PostProcessor) *Tokenizer
func (t *Tokenizer) WithTruncation(params *TruncationParams) *Tokenizer
func (t *Tokenizer) WithPadding(params *PaddingParams) *Tokenizer

// Full encoding with metadata (post-processing + truncation + padding)
func (t *Tokenizer) EncodeWithOptions(text string, addSpecialTokens bool) *Encoding
func (t *Tokenizer) EncodeBatchWithOptions(texts []string, addSpecialTokens bool) []*Encoding
```

### Encoding

```go
type Encoding struct {
    IDs               []int    // Token IDs
    Tokens            []string // String pieces
    AttentionMask     []int    // 1 for real tokens, 0 for padding
    TypeIDs           []int    // Segment IDs (0 for first, 1 for second)
    SpecialTokensMask []int    // 1 for special tokens, 0 for normal
}
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

// Special token IDs
func (m *Model) UnkID() int
func (m *Model) BosID() int
func (m *Model) EosID() int
func (m *Model) PadID() int
```

## Supported Models

Any SentencePiece `.model` file (protobuf) or HuggingFace `tokenizer.json` that uses **Unigram** or **BPE** model type.

**SentencePiece .model (protobuf):**
- `microsoft/deberta-v3-small` / `base` / `large` (Unigram)
- `google/gemma-3-1b-it` (BPE, 256K vocab)
- Other Unigram/BPE SentencePiece models (XLNet, ALBERT, T5, LLaMA, Mistral, etc.)

**HuggingFace tokenizer.json:**
- `onnx-community/mmBERT-small-ONNX` (BPE, 256K vocab)
- Other BPE/Unigram tokenizer.json models with Metaspace pre-tokenizer

**Note:** WORD and CHAR model types are not supported. ByteLevel pre-tokenizer (GPT-2/RoBERTa) is not yet supported.

## Benchmarks

Measured on Apple M4 Max. Python `sentencepiece` is a C++ library with Python SWIG bindings.

| Operation | Input | Go | Python (C++) | Speedup |
|-----------|-------|---:|-------------:|--------:|
| Encode | short (11 chars) | 301 ns | 1.1 μs | **3.7x** |
| Encode | medium (120 chars) | 3.3 μs | 5.6 μs | **1.7x** |
| Encode | long (4500 chars) | 94 μs | 183 μs | **1.9x** |
| Decode | short (10 tokens) | 343 ns | 801 ns | **2.3x** |

Go beats the C++ reference on short inputs (Python FFI overhead dominates).
On long inputs, pure Go Viterbi is ~2x faster than C++ via Python bindings.

```bash
# Go benchmarks
make bench

# Python benchmarks (requires make venv)
.venv/bin/python _testdata/bench_python.py
```

## Examples

The [examples/](examples/) directory contains runnable programs:

| Example | Description |
|---------|-------------|
| `basic` | Encode, decode, batch encode, vocab metadata |
| `embed` | Load model from `go:embed` binary data |
| `mmbert` | mmBERT-small ONNX inference prep (tokenizer.json or .model) |
| `similarity` | Jaccard similarity between two texts at token level |
| `streaming` | Memory-efficient line-by-line tokenization from stdin |
| `vocab-inspect` | Inspect vocab: special tokens, byte tokens, piece search |
| `benchmark` | CLI throughput benchmark (tokens/sec) |
| `compare` | Side-by-side Unigram vs BPE tokenization comparison |

```bash
go run ./examples/basic _testdata/spm.model
go run ./examples/mmbert _testdata/tokenizer.json   # HuggingFace format
go run ./examples/mmbert _testdata/bpe.model         # SentencePiece format
go run ./examples/compare _testdata/spm.model _testdata/bpe.model
go run ./examples/vocab-inspect _testdata/spm.model
cat file.txt | go run ./examples/benchmark _testdata/spm.model
```

## Project Structure

```
sentencepiece.go    -- public Tokenizer type, pipeline, constructors
model.go            -- protobuf loading, vocab index, ByteTrie
tokenizer_json.go   -- HuggingFace tokenizer.json loading
normalizer.go       -- precompiled charsmap (Darts trie), NFKC, whitespace
unigram.go          -- Viterbi decoding (forward DP + backtrack)
bpe.go              -- BPE greedy merge with priority queue
encoder.go          -- Encode/Decode with byte-token handling
encoding.go         -- Encoding struct (IDs, masks, offsets)
postprocessor.go    -- PostProcessor interface, template processing
padding.go          -- Padding (right/left, fixed/batch-longest)
truncation.go       -- Truncation to max length
byte_fallback.go    -- <0xHH> token encoding/decoding
errors.go           -- sentinel error types
trie.go             -- ByteTrie (vocab), DartsDoubleArray (charsmap)
proto/              -- generated protobuf code
examples/           -- runnable examples (basic, embed, mmbert)
_testdata/          -- test models and golden test cases
```

## How It Works

1. **Normalization**: Input text is normalized using the model's precompiled character map (a Darts double-array trie that encodes NFKC and custom rules). Whitespace is deduplicated, a prefix space is added, and spaces are replaced with the metaspace character.

2. **Viterbi tokenization**: The normalized text is segmented into pieces using dynamic programming. A byte-level trie is traversed to find all vocabulary pieces starting at each position. The algorithm finds the segmentation that maximizes total log-probability.

3. **Byte fallback**: Characters not covered by any vocabulary piece are encoded as individual UTF-8 bytes using `<0xHH>` tokens.

4. **Decoding**: Token IDs are mapped back to piece strings. Byte tokens are accumulated and flushed as UTF-8. The metaspace prefix is converted back to spaces.

## Thread Safety

`Tokenizer` is safe for concurrent use by multiple goroutines after creation. The model and normalizer are read-only after initialization, so no locking is needed.

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
