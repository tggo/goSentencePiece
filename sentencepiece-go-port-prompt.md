# Task: Port SentencePiece Unigram Tokenizer to Pure Go

## Goal

Create a pure Go implementation of the SentencePiece Unigram tokenizer that produces **byte-identical output** to the Python `sentencepiece` library. The implementation must support DeBERTa v3 models (Unigram model type with byte-fallback).

## Context

- Source reference: https://github.com/google/sentencepiece
- Target model: `microsoft/deberta-v3-small` (uses Unigram, NOT BPE)
- The `.model` file is a protobuf (`sentencepiece_model.proto`)
- This will be used for ONNX inference in a Go service — no Python dependencies allowed in production

## Constraints

- **Pure Go only** — no CGo, no Rust FFI, no external C libraries
- Go module name: `github.com/promova/sentencepiece`
- Min Go version: 1.22
- Zero runtime dependencies beyond stdlib + `google.golang.org/protobuf` + `golang.org/x/text`

---

## Phase 1: Golden Test Dataset (Python)

### Step 1.1: Setup

```bash
mkdir -p _testdata/golden
pip install sentencepiece transformers protobuf
```

### Step 1.2: Download the model

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
# Save the sentencepiece model file
# It's usually at tokenizer.vocab_file or inside the tokenizer dir
```

Save the `.model` file to `_testdata/spm.model`.

### Step 1.3: Generate golden test cases

Create `_testdata/generate_golden.py` that produces `_testdata/golden/test_cases.jsonl`.

Each line is a JSON object:

```json
{
  "input": "raw input string",
  "pieces": ["▁Hello", "▁world"],
  "ids": [123, 456],
  "decoded": "Hello world",
  "description": "basic_english"
}
```

**CRITICAL**: Generate at least 500 test cases covering ALL of these categories:

**Basic:**
- Empty string
- Single character (ASCII, Cyrillic, CJK, emoji)
- Single word, multiple words
- Numbers, floats, negative numbers
- Punctuation only

**Unicode & Encoding:**
- Latin, Cyrillic (Ukrainian: "Привіт світ"), CJK, Arabic, Thai, Devanagari
- Mixed scripts in one string: "Hello Привіт 你好 🚀"
- Combining characters, diacritics (é vs e + ́)
- Emoji: single, ZWJ sequences (👨‍👩‍👧‍👦), skin tones, flags
- Unicode normalization edge cases (NFKC)
- Surrogate-adjacent codepoints
- Right-to-left text
- Zero-width spaces, BOM, soft hyphens

**Whitespace:**
- Leading/trailing spaces
- Multiple consecutive spaces
- Tabs, newlines, \r\n, vertical tab
- Only whitespace

**Length:**
- Very long strings (10K+ chars)
- Single very long word (no spaces, 1000+ chars)

**Special patterns:**
- URLs: `https://example.com/path?q=hello&lang=uk`
- Email addresses
- Code snippets: `func main() { fmt.Println("hello") }`
- JSON strings
- Markdown
- HTML tags
- Numbers with separators: `1,234,567.89`
- Dates: `2024-01-15`, `15/01/2024`

**Byte-fallback:**
- Strings that force `<0xHH>` token usage
- Invalid UTF-8 sequences (if applicable to sentencepiece behavior)
- Rare Unicode characters not in vocabulary

**Model-specific:**
- Strings that produce `[UNK]` tokens
- Max length boundary (128 tokens, 256, 512)
- Repeated characters: "aaaaaaa", "абабабаб"

### Step 1.4: Generate protobuf reference data

Also dump the raw model metadata for validation:

```python
import sentencepiece_model_pb2  # compile from proto
# or use: from sentencepiece import SentencePieceProcessor
```

Create `_testdata/golden/model_info.json`:

```json
{
  "vocab_size": 128100,
  "model_type": "unigram",
  "bos_id": 1,
  "eos_id": 2,
  "unk_id": 0,
  "pad_id": -1,
  "byte_fallback": true,
  "normalization_rule": "nfkc",
  "sample_vocab_entries": [
    {"piece": "▁the", "score": -3.2, "type": "NORMAL"},
    {"piece": "<0x41>", "score": -9.0, "type": "BYTE"}
  ]
}
```

### Step 1.5: Validate golden data

Run a self-check: load the golden JSONL back and verify every case against `sentencepiece` to ensure the file is correct.

---

## Phase 2: Go Implementation

### Step 2.1: Project structure

```
github.com/promova/sentencepiece/
├── go.mod
├── go.sum
├── model.go              # protobuf loading, model struct
├── normalizer.go         # NFKC + sentencepiece preprocess
├── unigram.go            # Viterbi tokenization core
├── encoder.go            # public Encode/Decode API
├── byte_fallback.go      # <0xHH> handling
├── sentencepiece.go      # top-level Tokenizer type + constructor
├── sentencepiece_test.go # golden tests
├── proto/
│   └── sentencepiece_model.pb.go  # generated protobuf
├── _testdata/
│   ├── spm.model
│   └── golden/
│       ├── test_cases.jsonl
│       └── model_info.json
└── _testdata/
    └── generate_golden.py
```

### Step 2.2: Protobuf

Get the proto definition from https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto

Generate Go code:

```bash
protoc --go_out=proto/ --go_opt=paths=source_relative sentencepiece_model.proto
```

### Step 2.3: Model loading (`model.go`)

```go
type Model struct {
    pieces    []Piece      // vocab: piece string → id, score, type
    pieceIndex map[string]int // fast lookup: piece string → vocab index
    unkID     int
    bosID     int
    eosID     int
    byteFallback bool
    // normalizer config
}

func LoadModel(path string) (*Model, error)
```

- Read file, unmarshal protobuf
- Build the piece index (map for O(1) lookup)
- Validate: check model_type == UNIGRAM, byte_fallback flag

### Step 2.4: Normalizer (`normalizer.go`)

SentencePiece normalization for DeBERTa v3:

1. **NFKC normalization** — use `golang.org/x/text/unicode/norm`
2. **Whitespace normalization** — replace all whitespace variants with regular space
3. **Prepend space** — add `▁` (U+2581) at the beginning
4. **Space replacement** — replace spaces with `▁` in the output

The normalizer spec is embedded in the `.model` protobuf under `normalizer_spec`. Parse and respect it.

### Step 2.5: Unigram Tokenizer (`unigram.go`)

This is the core algorithm. Implement **Viterbi decoding**:

```
Input: normalized string S of length N
Output: optimal tokenization (sequence of pieces that maximizes total log-probability)

Algorithm:
1. Build a DAG: for each position i in S, find all pieces in vocab 
   that match S[i:i+len(piece)]
2. Use dynamic programming (backward or forward) to find the 
   highest-scoring path through the DAG
3. Backtrack to recover the optimal segmentation
```

**Performance considerations:**
- Use a **trie** (or double-array trie) for prefix matching instead of brute-force
- Pre-sort vocab by piece length for efficient matching
- This is the hot path — profile and optimize

### Step 2.6: Byte Fallback (`byte_fallback.go`)

When a character has no matching piece in vocab:
1. Encode the character as UTF-8 bytes
2. Map each byte to `<0xHH>` token (e.g., byte 0x41 → `<0x41>`)
3. These tokens must exist in the vocabulary

### Step 2.7: Public API (`sentencepiece.go`)

```go
type Tokenizer struct {
    model *Model
}

func NewTokenizer(modelPath string) (*Tokenizer, error)

// Encode returns token IDs for the input string
func (t *Tokenizer) Encode(text string) ([]int, error)

// EncodeAsPieces returns string pieces
func (t *Tokenizer) EncodeAsPieces(text string) ([]string, error)

// Decode converts token IDs back to string
func (t *Tokenizer) Decode(ids []int) (string, error)

// AddSpecialTokens wraps with BOS/EOS if configured
func (t *Tokenizer) AddSpecialTokens(ids []int) []int

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int
```

### Step 2.8: Tests (`sentencepiece_test.go`)

```go
func TestGoldenCases(t *testing.T) {
    tok, err := NewTokenizer("_testdata/spm.model")
    require.NoError(t, err)

    cases := loadGoldenCases(t, "_testdata/golden/test_cases.jsonl")
    
    for _, tc := range cases {
        t.Run(tc.Description, func(t *testing.T) {
            ids, err := tok.Encode(tc.Input)
            require.NoError(t, err)
            assert.Equal(t, tc.IDs, ids, "token IDs mismatch for: %q", tc.Input)

            pieces, err := tok.EncodeAsPieces(tc.Input)
            require.NoError(t, err)
            assert.Equal(t, tc.Pieces, pieces, "pieces mismatch for: %q", tc.Input)

            decoded, err := tok.Decode(tc.IDs)
            require.NoError(t, err)
            assert.Equal(t, tc.Decoded, decoded, "decode mismatch for: %q", tc.Input)
        })
    }
}
```

Also add:
- `TestModelLoading` — verify vocab size, special token IDs
- `TestNormalization` — test NFKC independently
- `TestByteFallback` — test byte encoding edge cases
- Benchmarks: `BenchmarkEncode`, `BenchmarkEncodeLong`, `BenchmarkDecode`

---

## Phase 3: Validation & Hardening

### Step 3.1: Run all golden tests — must be 100% pass rate

If any test fails:
1. Print the failing input, expected output, actual output
2. Debug the specific algorithm step that diverges
3. Fix and re-run

**Do NOT skip or mark tests as "known failures".** Every single golden test must pass.

### Step 3.2: Benchmarks

Target performance:
- Short string (< 50 chars): < 10μs
- Medium string (200 chars): < 50μs
- Long string (10K chars): < 5ms

### Step 3.3: Fuzz testing

Add `FuzzEncode` — feed random strings, ensure no panics, no out-of-bounds.

```go
func FuzzEncode(f *testing.F) {
    f.Add("hello world")
    f.Add("")
    f.Add("🚀🚀🚀")
    f.Fuzz(func(t *testing.T, input string) {
        tok, _ := NewTokenizer("_testdata/spm.model")
        ids, err := tok.Encode(input)
        if err != nil {
            return
        }
        decoded, err := tok.Decode(ids)
        // decoded may not equal input exactly (normalization), but must not panic
        _ = decoded
    })
}
```

---

## Execution Order

1. **Phase 1 first, completely.** Do not start Go code until golden dataset exists and is self-validated.
2. **Phase 2: implement incrementally.** After each component (normalizer, unigram, byte_fallback), run the relevant subset of tests.
3. **Phase 3: only after all golden tests pass.**

## Important Notes

- The SentencePiece `▁` character is U+2581 (LOWER ONE EIGHTH BLOCK), not an underscore
- DeBERTa v3 uses `add_prefix_space=True` by default — the normalizer must prepend `▁`
- Token ID 0 is typically `[UNK]`, 1 is `[CLS]`/`<s>`, 2 is `[SEP]`/`</s>`
- The Viterbi algorithm must handle the case where NO piece matches at a position → byte fallback
- Protobuf model file may be large (~1MB). Don't load it per-request; load once and reuse
- All string operations must be Unicode-safe (rune-based iteration, not byte-based)
- The trie for prefix matching should be built once at model load time
