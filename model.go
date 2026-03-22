package sentencepiece

//go:generate protoc --go_out=. --go_opt=paths=source_relative proto/sentencepiece_model.proto

import (
	"fmt"
	"io"
	"math"
	"os"

	pb "github.com/tggo/goSentencePiece/proto"
	"google.golang.org/protobuf/proto"
)

// PieceType represents the type of a vocabulary piece as defined by the
// SentencePiece model specification. Each piece in the vocabulary has a type
// that determines how it is handled during encoding and decoding.
type PieceType int32

const (
	// PieceNormal is a regular vocabulary piece learned during training.
	PieceNormal PieceType = 1
	// PieceUnknown represents the unknown token used for out-of-vocabulary inputs.
	PieceUnknown PieceType = 2
	// PieceControl represents control tokens such as BOS and EOS.
	PieceControl PieceType = 3
	// PieceUserDefined represents user-defined tokens that always take priority.
	PieceUserDefined PieceType = 4
	// PieceUnused represents unused/reserved vocabulary slots.
	PieceUnused PieceType = 5
	// PieceByte represents a byte-level fallback token (e.g., "<0x41>").
	PieceByte PieceType = 6
)

// Piece represents a single entry in the SentencePiece vocabulary, including
// its string representation, log-probability score, and type.
type Piece struct {
	Piece string
	Score float32
	Type  PieceType
}

// ModelType represents the tokenization algorithm used by the model.
type ModelType int32

const (
	// ModelTypeUnigram uses Viterbi decoding to find the optimal segmentation.
	ModelTypeUnigram ModelType = 1
	// ModelTypeBPE uses greedy byte-pair encoding merges.
	ModelTypeBPE ModelType = 2
)

// Model holds the loaded SentencePiece model data, including the vocabulary,
// piece index, normalizer configuration, and a byte trie for efficient prefix
// matching during tokenization.
type Model struct {
	modelType    ModelType
	pieces       []Piece
	pieceIndex   map[string]int
	unkID        int
	bosID        int
	eosID        int
	padID        int
	byteFallback bool

	// Normalizer config from the model.
	addDummyPrefix        bool
	removeExtraWhitespace bool
	escapeWhitespaces     bool
	precompiledCharsmap   []byte

	// Byte-level trie for vocab prefix matching (matches reference traverse behavior).
	vocabTrie *ByteTrie
	// Precomputed scores.
	maxScoreVal float32
	minScoreVal float32
}

// LoadModel loads and parses a SentencePiece model from a .model file at the
// given path. It reads the protobuf data, builds the vocabulary index and byte
// trie, and extracts normalizer and trainer configuration.
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model file: %w", err)
	}
	return loadModelFromBytes(data)
}

// LoadModelFromReader loads and parses a SentencePiece model from the provided
// io.Reader. This allows loading models from any source, such as embedded
// files, network streams, or in-memory buffers.
func LoadModelFromReader(r io.Reader) (*Model, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read model data: %w", err)
	}
	return loadModelFromBytes(data)
}

func loadModelFromBytes(data []byte) (*Model, error) {
	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidModel, err)
	}

	// Check model type — support Unigram and BPE.
	modelType := ModelTypeUnigram
	if ts := modelProto.TrainerSpec; ts != nil {
		switch ts.GetModelType() {
		case pb.TrainerSpec_UNIGRAM:
			modelType = ModelTypeUnigram
		case pb.TrainerSpec_BPE:
			modelType = ModelTypeBPE
		default:
			return nil, fmt.Errorf("%w: got %v", ErrUnsupportedModel, ts.GetModelType())
		}
	}

	m := &Model{
		modelType:  modelType,
		pieces:     make([]Piece, len(modelProto.Pieces)),
		pieceIndex: make(map[string]int, len(modelProto.Pieces)),
	}

	// Load pieces.
	for i, sp := range modelProto.Pieces {
		m.pieces[i] = Piece{
			Piece: sp.GetPiece(),
			Score: sp.GetScore(),
			Type:  PieceType(sp.GetType()),
		}
		m.pieceIndex[sp.GetPiece()] = i
	}

	// Load trainer spec for special token IDs and byte_fallback.
	if ts := modelProto.TrainerSpec; ts != nil {
		m.unkID = int(ts.GetUnkId())
		m.bosID = int(ts.GetBosId())
		m.eosID = int(ts.GetEosId())
		m.padID = int(ts.GetPadId())
		m.byteFallback = ts.GetByteFallback()
	}

	// Load normalizer spec.
	if ns := modelProto.NormalizerSpec; ns != nil {
		m.addDummyPrefix = ns.GetAddDummyPrefix()
		m.removeExtraWhitespace = ns.GetRemoveExtraWhitespaces()
		m.escapeWhitespaces = ns.GetEscapeWhitespaces()
		m.precompiledCharsmap = ns.GetPrecompiledCharsmap()
	}

	// Build byte-level trie for prefix matching.
	m.vocabTrie = NewByteTrie()
	for i, p := range m.pieces {
		if p.Type == PieceNormal || p.Type == PieceUserDefined {
			m.vocabTrie.Insert(p.Piece, i)
		}
	}

	// Precompute min/max scores.
	m.minScoreVal = float32(math.MaxFloat32)
	m.maxScoreVal = -float32(math.MaxFloat32)
	for _, p := range m.pieces {
		if p.Type == PieceNormal {
			if p.Score < m.minScoreVal {
				m.minScoreVal = p.Score
			}
			if p.Score > m.maxScoreVal {
				m.maxScoreVal = p.Score
			}
		}
	}

	return m, nil
}

// encode dispatches to the appropriate encoding algorithm based on model type.
func (m *Model) encode(normalized string) []encodedPiece {
	switch m.modelType {
	case ModelTypeBPE:
		return m.encodeBPE(normalized)
	default:
		return m.encodeUnigram(normalized)
	}
}

// VocabSize returns the total number of pieces in the vocabulary, including
// normal, control, byte, and special tokens.
func (m *Model) VocabSize() int {
	return len(m.pieces)
}

// IdToPiece returns the string representation of the piece with the given ID.
// It returns an empty string if the ID is out of range.
func (m *Model) IdToPiece(id int) string {
	if id < 0 || id >= len(m.pieces) {
		return ""
	}
	return m.pieces[id].Piece
}

// PieceToId returns the vocabulary ID for the given piece string.
// It returns the unknown token ID if the piece is not found in the vocabulary.
func (m *Model) PieceToId(piece string) int {
	if id, ok := m.pieceIndex[piece]; ok {
		return id
	}
	return m.unkID
}

// UnkID returns the vocabulary ID of the unknown token.
func (m *Model) UnkID() int { return m.unkID }

// BosID returns the vocabulary ID of the beginning-of-sentence token.
func (m *Model) BosID() int { return m.bosID }

// EosID returns the vocabulary ID of the end-of-sentence token.
func (m *Model) EosID() int { return m.eosID }

// PadID returns the vocabulary ID of the padding token (-1 if not set).
func (m *Model) PadID() int { return m.padID }
