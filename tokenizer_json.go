package sentencepiece

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// tokenizerJSON represents the top-level HuggingFace tokenizer.json structure.
type tokenizerJSON struct {
	AddedTokens   []addedTokenJSON `json:"added_tokens"`
	PreTokenizer  json.RawMessage  `json:"pre_tokenizer"`
	Model         modelJSON        `json:"model"`
	PostProcessor json.RawMessage  `json:"post_processor"`
}

type addedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type modelJSON struct {
	Type         string          `json:"type"`
	Vocab        json.RawMessage `json:"vocab"`
	Merges       json.RawMessage `json:"merges"`
	UnkToken     string          `json:"unk_token"`
	UnkID        *int            `json:"unk_id"`
	ByteFallback bool            `json:"byte_fallback"`
}

// NewTokenizerFromJSON loads a tokenizer from a HuggingFace tokenizer.json file.
func NewTokenizerFromJSON(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer file: %w", err)
	}
	return loadFromJSON(data)
}

// NewTokenizerFromJSONReader loads a tokenizer from a reader containing
// HuggingFace tokenizer.json data.
func NewTokenizerFromJSONReader(r io.Reader) (*Tokenizer, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer data: %w", err)
	}
	return loadFromJSON(data)
}

func loadFromJSON(data []byte) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidModel, err)
	}

	model, err := buildModelFromJSON(&tj)
	if err != nil {
		return nil, err
	}

	encoder := NewEncoder(model)

	// For JSON tokenizers with Metaspace pre-tokenizer, use custom
	// pre-tokenization instead of SentencePiece normalization.
	if hasMetaspacePreTokenizer(tj.PreTokenizer) {
		encoder.preTokenize = metaspacePreTokenize
		encoder.stripLeadingSpace = false
	}

	// Build added tokens trie for pre-matching.
	if at := buildAddedTokensTrie(tj.AddedTokens); at != nil {
		encoder.addedTokens = at
	}

	tok := &Tokenizer{
		model:   model,
		encoder: encoder,
	}

	if pp := parseJSONPostProcessor(tj.PostProcessor); pp != nil {
		tok.postProc = pp
	}

	return tok, nil
}

func buildModelFromJSON(tj *tokenizerJSON) (*Model, error) {
	var modelType ModelType
	switch tj.Model.Type {
	case "BPE":
		modelType = ModelTypeBPE
	case "Unigram":
		modelType = ModelTypeUnigram
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedModel, tj.Model.Type)
	}

	pieces, pieceIndex, err := parseJSONVocab(modelType, tj.Model.Vocab)
	if err != nil {
		return nil, err
	}

	classifyJSONPieces(pieces, tj.AddedTokens)

	if modelType == ModelTypeBPE && len(tj.Model.Merges) > 0 {
		if err := assignBPEScores(pieces, pieceIndex, tj.Model.Merges); err != nil {
			return nil, err
		}
	}

	// Normalizer config is only used as fallback when the Metaspace
	// pre-tokenizer is not active (the Metaspace path bypasses the
	// SentencePiece normalizer entirely via encoder.preTokenize).
	addDummy, escapeSp := parseJSONNormConfig(tj.PreTokenizer)

	m := &Model{
		modelType:             modelType,
		pieces:                pieces,
		pieceIndex:            pieceIndex,
		unkID:                 resolveUnkID(tj, pieceIndex),
		bosID:                 resolveTokenID(tj.AddedTokens, pieceIndex, "<bos>"),
		eosID:                 resolveTokenID(tj.AddedTokens, pieceIndex, "<eos>"),
		padID:                 resolveTokenID(tj.AddedTokens, pieceIndex, "<pad>"),
		byteFallback:          tj.Model.ByteFallback,
		addDummyPrefix:        addDummy,
		escapeWhitespaces:     escapeSp,
		removeExtraWhitespace: false,
	}

	buildModelTrie(m)
	computeMinMaxScores(m)

	return m, nil
}

func parseJSONVocab(modelType ModelType, raw json.RawMessage) ([]Piece, map[string]int, error) {
	switch modelType {
	case ModelTypeBPE:
		return parseBPEVocab(raw)
	default:
		return parseUnigramVocab(raw)
	}
}

func buildModelTrie(m *Model) {
	m.vocabTrie = NewByteTrie()
	for i, p := range m.pieces {
		if p.Type == PieceNormal || p.Type == PieceUserDefined {
			m.vocabTrie.Insert(p.Piece, i)
		}
	}
}

func computeMinMaxScores(m *Model) {
	m.minScoreVal = float32(math.MaxFloat32)
	m.maxScoreVal = -float32(math.MaxFloat32)
	for _, p := range m.pieces {
		if p.Type == PieceNormal && p.Score != 0 {
			if p.Score < m.minScoreVal {
				m.minScoreVal = p.Score
			}
			if p.Score > m.maxScoreVal {
				m.maxScoreVal = p.Score
			}
		}
	}
}

func parseBPEVocab(raw json.RawMessage) ([]Piece, map[string]int, error) {
	var vocab map[string]int
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, nil, fmt.Errorf("parse BPE vocab: %w", err)
	}

	maxID := 0
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}

	pieces := make([]Piece, maxID+1)
	pieceIndex := make(map[string]int, len(vocab))
	for token, id := range vocab {
		pieces[id] = Piece{Piece: token, Type: PieceNormal}
		pieceIndex[token] = id
	}

	return pieces, pieceIndex, nil
}

func parseUnigramVocab(raw json.RawMessage) ([]Piece, map[string]int, error) {
	var vocab [][2]json.RawMessage
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, nil, fmt.Errorf("parse Unigram vocab: %w", err)
	}

	pieces := make([]Piece, len(vocab))
	pieceIndex := make(map[string]int, len(vocab))
	for i, entry := range vocab {
		var token string
		var score float64
		if err := json.Unmarshal(entry[0], &token); err != nil {
			return nil, nil, fmt.Errorf("parse vocab token: %w", err)
		}
		if err := json.Unmarshal(entry[1], &score); err != nil {
			return nil, nil, fmt.Errorf("parse vocab score: %w", err)
		}
		pieces[i] = Piece{Piece: token, Score: float32(score), Type: PieceNormal}
		pieceIndex[token] = i
	}

	return pieces, pieceIndex, nil
}

func assignBPEScores(pieces []Piece, pieceIndex map[string]int, raw json.RawMessage) error {
	// Try nested array format: [["A", "B"], ...]
	var nested [][]string
	if json.Unmarshal(raw, &nested) == nil {
		total := len(nested)
		for i, m := range nested {
			if len(m) >= 2 {
				if id, ok := pieceIndex[m[0]+m[1]]; ok {
					pieces[id].Score = float32(total - i)
				}
			}
		}
		return nil
	}

	// Try string format: ["A B", ...]
	var flat []string
	if err := json.Unmarshal(raw, &flat); err != nil {
		return fmt.Errorf("parse merges: %w", err)
	}
	total := len(flat)
	for i, s := range flat {
		parts := strings.SplitN(s, " ", 2)
		if len(parts) == 2 {
			if id, ok := pieceIndex[parts[0]+parts[1]]; ok {
				pieces[id].Score = float32(total - i)
			}
		}
	}
	return nil
}

func classifyJSONPieces(pieces []Piece, addedTokens []addedTokenJSON) {
	addedByID := make(map[int]*addedTokenJSON, len(addedTokens))
	for i := range addedTokens {
		addedByID[addedTokens[i].ID] = &addedTokens[i]
	}

	for i := range pieces {
		at, isAdded := addedByID[i]
		switch {
		case isAdded && at.Content == "<unk>":
			pieces[i].Type = PieceUnknown
		case isAdded && at.Special:
			pieces[i].Type = PieceControl
		case isAdded && strings.HasPrefix(at.Content, "<unused"):
			pieces[i].Type = PieceUnused
		case isAdded && !at.Special:
			pieces[i].Type = PieceUserDefined
		case pieceIsByte(pieces[i].Piece):
			pieces[i].Type = PieceByte
		}
	}
}

func resolveUnkID(tj *tokenizerJSON, pieceIndex map[string]int) int {
	if tj.Model.UnkID != nil {
		return *tj.Model.UnkID
	}
	if tj.Model.UnkToken != "" {
		if id, ok := pieceIndex[tj.Model.UnkToken]; ok {
			return id
		}
	}
	return 0
}

func resolveTokenID(addedTokens []addedTokenJSON, pieceIndex map[string]int, content string) int {
	for _, at := range addedTokens {
		if at.Content == content {
			return at.ID
		}
	}
	if id, ok := pieceIndex[content]; ok {
		return id
	}
	return -1
}

func parseJSONNormConfig(preTokData json.RawMessage) (addDummyPrefix, escapeWhitespaces bool) {
	if len(preTokData) == 0 || string(preTokData) == "null" {
		return false, false
	}
	var pt struct {
		Type          string `json:"type"`
		PrependScheme string `json:"prepend_scheme"`
	}
	if json.Unmarshal(preTokData, &pt) != nil {
		return false, false
	}
	if pt.Type == "Metaspace" {
		escapeWhitespaces = true
		addDummyPrefix = pt.PrependScheme == "always" || pt.PrependScheme == "first"
	}
	return
}

// buildAddedTokensTrie creates a ByteTrie from non-special added tokens for
// pre-matching in the encoding pipeline. Special tokens (like <pad>, <bos>)
// are excluded since they should not appear in regular input text.
// Byte-fallback tokens (<0xHH>) are excluded since they are handled by BPE.
func buildAddedTokensTrie(addedTokens []addedTokenJSON) *ByteTrie {
	trie := NewByteTrie()
	count := 0

	for _, at := range addedTokens {
		if at.Special {
			continue
		}
		if pieceIsByte(at.Content) {
			continue
		}
		trie.Insert(at.Content, at.ID)
		count++
	}

	if count == 0 {
		return nil
	}
	return trie
}

func hasMetaspacePreTokenizer(preTokData json.RawMessage) bool {
	if len(preTokData) == 0 || string(preTokData) == "null" {
		return false
	}
	var pt struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(preTokData, &pt) == nil && pt.Type == "Metaspace"
}

// metaspacePreTokenize implements the HuggingFace Metaspace pre-tokenizer:
// 1. Replace spaces with ▁
// 2. Prepend ▁ if text doesn't already start with ▁
// 3. Split on ▁ boundaries, keeping ▁ with the following segment
func metaspacePreTokenize(text string) []string {
	if len(text) == 0 {
		return nil
	}

	// Replace spaces with ▁.
	text = strings.ReplaceAll(text, " ", metaSpace)

	// Prepend ▁ if text doesn't start with ▁.
	if !strings.HasPrefix(text, metaSpace) {
		text = metaSpace + text
	}

	// Split on ▁, keeping ▁ attached to the right segment.
	return splitKeepDelimiter(text, '▁')
}

// splitKeepDelimiter splits text on a delimiter rune, keeping the delimiter
// attached to the following segment. Empty segments are dropped.
func splitKeepDelimiter(text string, delim rune) []string {
	var segments []string
	start := 0

	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == delim && i > start {
			segments = append(segments, text[start:i])
			start = i
		}
		i += size
	}

	if start < len(text) {
		segments = append(segments, text[start:])
	}

	return segments
}

// --- Post-processor parsing ---

type postProcJSON struct {
	Type          string                         `json:"type"`
	Single        []templateEntryJSON            `json:"single"`
	Pair          []templateEntryJSON            `json:"pair"`
	SpecialTokens map[string]specialTokenRefJSON `json:"special_tokens"`
}

type templateEntryJSON struct {
	SpecialToken *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"SpecialToken"`
	Sequence *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"Sequence"`
}

type specialTokenRefJSON struct {
	IDs []int `json:"ids"`
}

func parseJSONPostProcessor(data json.RawMessage) PostProcessor {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	var pp postProcJSON
	if json.Unmarshal(data, &pp) != nil || pp.Type != "TemplateProcessing" {
		return nil
	}

	single := convertJSONTemplate(pp.Single, pp.SpecialTokens)
	pair := convertJSONTemplate(pp.Pair, pp.SpecialTokens)
	return NewTemplateProcessing(single, pair)
}

func convertJSONTemplate(entries []templateEntryJSON, specials map[string]specialTokenRefJSON) []TemplatePiece {
	result := make([]TemplatePiece, len(entries))
	for i, e := range entries {
		if e.SpecialToken != nil {
			tokenID := 0
			if ref, ok := specials[e.SpecialToken.ID]; ok && len(ref.IDs) > 0 {
				tokenID = ref.IDs[0]
			}
			result[i] = TemplatePiece{
				SpecialToken: e.SpecialToken.ID,
				TokenID:      tokenID,
				TypeID:       e.SpecialToken.TypeID,
			}
		} else if e.Sequence != nil {
			seqID := 0
			if e.Sequence.ID == "B" {
				seqID = 1
			}
			result[i] = TemplatePiece{
				IsSequence: true,
				SequenceID: seqID,
				TypeID:     e.Sequence.TypeID,
			}
		}
	}
	return result
}
