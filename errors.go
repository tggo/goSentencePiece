package sentencepiece

import "errors"

// Sentinel errors returned by the tokenizer.
var (
	// ErrInvalidModel is returned when the model file cannot be parsed as a
	// valid SentencePiece protobuf.
	ErrInvalidModel = errors.New("sentencepiece: invalid model")

	// ErrUnsupportedModel is returned when the model uses an unsupported
	// model type (e.g., BPE instead of Unigram).
	ErrUnsupportedModel = errors.New("sentencepiece: unsupported model type")
)
