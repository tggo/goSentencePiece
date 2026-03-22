// Package sentencepiece provides a pure Go implementation of the SentencePiece
// Unigram tokenizer. It produces byte-identical output to the reference C++/Python
// sentencepiece library without requiring CGo or external dependencies.
package sentencepiece

import "io"

// Tokenizer is the top-level SentencePiece tokenizer. It wraps a loaded model
// and encoder, providing methods for encoding text to token IDs, decoding IDs
// back to text, and managing special tokens (BOS/EOS).
type Tokenizer struct {
	model   *Model
	encoder *Encoder
}

// NewTokenizer creates a new Tokenizer by loading a SentencePiece .model file
// from the given file path. It returns an error if the file cannot be read or
// the protobuf data cannot be parsed.
func NewTokenizer(modelPath string) (*Tokenizer, error) {
	model, err := LoadModel(modelPath)
	if err != nil {
		return nil, err
	}

	return &Tokenizer{
		model:   model,
		encoder: NewEncoder(model),
	}, nil
}

// NewTokenizerFromReader creates a new Tokenizer by reading a SentencePiece
// .model from the provided io.Reader. This is useful for loading models from
// embedded files, HTTP responses, or any other non-file source.
func NewTokenizerFromReader(r io.Reader) (*Tokenizer, error) {
	model, err := LoadModelFromReader(r)
	if err != nil {
		return nil, err
	}

	return &Tokenizer{
		model:   model,
		encoder: NewEncoder(model),
	}, nil
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
