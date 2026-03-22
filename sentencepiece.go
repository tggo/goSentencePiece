// Package sentencepiece provides a pure Go implementation of the SentencePiece
// tokenizer, supporting both Unigram (Viterbi) and BPE model types. It produces
// byte-identical output to the reference C++/Python sentencepiece library without
// requiring CGo or external dependencies.
package sentencepiece

import (
	"fmt"
	"io"
	"os"
)

// Tokenizer is the top-level SentencePiece tokenizer. It wraps a loaded model
// and encoder, providing methods for encoding text to token IDs, decoding IDs
// back to text, and managing special tokens (BOS/EOS).
//
// Tokenizer is safe for concurrent use by multiple goroutines after creation.
type Tokenizer struct {
	model      *Model
	encoder    *Encoder
	postProc   PostProcessor
	padding    *PaddingParams
	truncation *TruncationParams
}

// NewTokenizer creates a new Tokenizer by loading a model file from the given
// path. It auto-detects the format: HuggingFace tokenizer.json (if the file
// starts with '{') or SentencePiece protobuf .model (otherwise).
func NewTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model file: %w", err)
	}

	if isJSON(data) {
		return loadFromJSON(data)
	}

	model, err := loadModelFromBytes(data)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{
		model:   model,
		encoder: NewEncoder(model),
	}, nil
}

// NewTokenizerFromReader creates a new Tokenizer by reading a model from the
// provided io.Reader. It auto-detects the format: HuggingFace tokenizer.json
// or SentencePiece protobuf .model.
func NewTokenizerFromReader(r io.Reader) (*Tokenizer, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read model data: %w", err)
	}

	if isJSON(data) {
		return loadFromJSON(data)
	}

	model, err := loadModelFromBytes(data)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{
		model:   model,
		encoder: NewEncoder(model),
	}, nil
}

// isJSON reports whether data starts with a JSON object ('{').
func isJSON(data []byte) bool {
	for _, b := range data {
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		case '{':
			return true
		default:
			return false
		}
	}
	return false
}

// Encode tokenizes the input text using Unigram Viterbi decoding and returns
// the corresponding token IDs. An empty input returns a nil slice.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	return t.encoder.Encode(text), nil
}

// EncodeAsPieces tokenizes the input text and returns the string representation
// of each token piece (e.g., ["▁Hello", "▁world"]).
func (t *Tokenizer) EncodeAsPieces(text string) ([]string, error) {
	return t.encoder.EncodeAsPieces(text), nil
}

// Decode converts a sequence of token IDs back into the original text string.
// Control tokens are skipped and byte-fallback tokens are reassembled into UTF-8.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	return t.encoder.Decode(ids), nil
}

// AddSpecialTokens wraps the given token IDs with the model's BOS (beginning of
// sentence) and EOS (end of sentence) token IDs.
func (t *Tokenizer) AddSpecialTokens(ids []int) []int {
	result := make([]int, 0, len(ids)+2)
	result = append(result, t.model.bosID)
	result = append(result, ids...)
	result = append(result, t.model.eosID)
	return result
}

// VocabSize returns the total number of pieces in the model's vocabulary.
func (t *Tokenizer) VocabSize() int {
	return t.model.VocabSize()
}

// Model returns the underlying loaded Model, providing access to vocabulary
// metadata such as piece lookup, vocab size, and special token IDs.
func (t *Tokenizer) Model() *Model {
	return t.model
}

// EncodeBatch tokenizes multiple input strings in a single call, returning
// a slice of token ID slices. This is a convenience method equivalent to
// calling Encode on each input individually.
func (t *Tokenizer) EncodeBatch(texts []string) ([][]int, error) {
	results := make([][]int, len(texts))
	for i, text := range texts {
		ids, err := t.Encode(text)
		if err != nil {
			return nil, err
		}
		results[i] = ids
	}
	return results, nil
}

// WithPostProcessor sets the post-processor for the tokenizer and returns the
// tokenizer for method chaining.
func (t *Tokenizer) WithPostProcessor(pp PostProcessor) *Tokenizer {
	t.postProc = pp
	return t
}

// WithPadding sets the padding parameters and returns the tokenizer for method
// chaining.
func (t *Tokenizer) WithPadding(params *PaddingParams) *Tokenizer {
	t.padding = params
	return t
}

// WithTruncation sets the truncation parameters and returns the tokenizer for
// method chaining.
func (t *Tokenizer) WithTruncation(params *TruncationParams) *Tokenizer {
	t.truncation = params
	return t
}

// EncodeWithOptions tokenizes the input text and returns a full Encoding with
// all metadata. It applies post-processing, truncation, and padding (if
// configured) in that order.
func (t *Tokenizer) EncodeWithOptions(text string, addSpecialTokens bool) *Encoding {
	// Step 1: Normalize + tokenize using the existing encoder.
	enc := t.encodeToEncoding(text)

	// Step 2: Apply post-processor if set.
	if t.postProc != nil && addSpecialTokens {
		enc = t.postProc.Process(enc, true)
	}

	// Step 3: Apply truncation if set.
	if t.truncation != nil {
		enc = TruncateEncoding(enc, t.truncation)
	}

	// Step 4: Ensure attention mask is set to 1 for all real tokens.
	for i := range enc.AttentionMask {
		enc.AttentionMask[i] = 1
	}

	// Step 5: Apply padding if set.
	if t.padding != nil && t.padding.Strategy == PadToMaxLength {
		enc = PadEncoding(enc, t.padding.MaxLength, t.padding)
	}

	return enc
}

// EncodeBatchWithOptions tokenizes multiple texts and returns padded Encodings.
// Post-processing and truncation are applied to each encoding individually,
// then the batch is padded together.
func (t *Tokenizer) EncodeBatchWithOptions(texts []string, addSpecialTokens bool) []*Encoding {
	encodings := make([]*Encoding, len(texts))
	for i, text := range texts {
		// Encode with post-processing and truncation, but defer batch padding.
		enc := t.encodeToEncoding(text)

		if t.postProc != nil && addSpecialTokens {
			enc = t.postProc.Process(enc, true)
		}

		if t.truncation != nil {
			enc = TruncateEncoding(enc, t.truncation)
		}

		// Ensure attention mask is all 1s before padding.
		for j := range enc.AttentionMask {
			enc.AttentionMask[j] = 1
		}

		encodings[i] = enc
	}

	// Apply batch padding if configured.
	if t.padding != nil {
		encodings = PadEncodings(encodings, t.padding)
	}

	return encodings
}

// encodeToEncoding runs the core tokenization and builds an Encoding struct.
func (t *Tokenizer) encodeToEncoding(text string) *Encoding {
	if len(text) == 0 {
		return newEncoding(nil, nil)
	}

	pieces := t.encoder.mergeConsecutiveUnk(t.encoder.encodePieces(text))

	ids := make([]int, len(pieces))
	tokens := make([]string, len(pieces))
	for i, p := range pieces {
		ids[i] = p.id
		tokens[i] = p.piece
	}

	return newEncoding(ids, tokens)
}
