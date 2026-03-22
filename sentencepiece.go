package sentencepiece

// Tokenizer is the top-level SentencePiece tokenizer.
type Tokenizer struct {
	model   *Model
	encoder *Encoder
}

// NewTokenizer creates a new tokenizer from a .model file path.
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

// Encode returns token IDs for the input string.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	return t.encoder.Encode(text), nil
}

// EncodeAsPieces returns string pieces for the input string.
func (t *Tokenizer) EncodeAsPieces(text string) ([]string, error) {
	return t.encoder.EncodeAsPieces(text), nil
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	return t.encoder.Decode(ids), nil
}

// AddSpecialTokens wraps token IDs with BOS/EOS tokens.
func (t *Tokenizer) AddSpecialTokens(ids []int) []int {
	result := make([]int, 0, len(ids)+2)
	result = append(result, t.model.bosID)
	result = append(result, ids...)
	result = append(result, t.model.eosID)
	return result
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return t.model.VocabSize()
}
