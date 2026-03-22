package sentencepiece

import (
	"fmt"
	"math"
	"os"

	pb "github.com/promova/sentencepiece/proto"
	"google.golang.org/protobuf/proto"
)

// PieceType represents the type of a vocabulary piece.
type PieceType int32

const (
	PieceNormal      PieceType = 1
	PieceUnknown     PieceType = 2
	PieceControl     PieceType = 3
	PieceUserDefined PieceType = 4
	PieceUnused      PieceType = 5
	PieceByte        PieceType = 6
)

// Piece represents a single vocabulary entry.
type Piece struct {
	Piece string
	Score float32
	Type  PieceType
}

// Model holds the loaded SentencePiece model data.
type Model struct {
	pieces       []Piece
	pieceIndex   map[string]int
	unkID        int
	bosID        int
	eosID        int
	padID        int
	byteFallback bool

	// Normalizer config from the model.
	addDummyPrefix       bool
	removeExtraWhitespace bool
	escapeWhitespaces    bool
	precompiledCharsmap  []byte

	// Byte-level trie for vocab prefix matching (matches reference traverse behavior).
	vocabTrie *ByteTrie
	// Precomputed scores.
	maxScoreVal float32
	minScoreVal float32
}

// LoadModel loads a SentencePiece model from a .model file.
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model file: %w", err)
	}

	var modelProto pb.ModelProto
	if err := proto.Unmarshal(data, &modelProto); err != nil {
		return nil, fmt.Errorf("unmarshal protobuf: %w", err)
	}

	m := &Model{
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

// VocabSize returns the number of pieces in the vocabulary.
func (m *Model) VocabSize() int {
	return len(m.pieces)
}

// IdToPiece returns the piece string for a given ID.
func (m *Model) IdToPiece(id int) string {
	if id < 0 || id >= len(m.pieces) {
		return ""
	}
	return m.pieces[id].Piece
}

// PieceToId returns the ID for a given piece string.
// Returns unkID if not found.
func (m *Model) PieceToId(piece string) int {
	if id, ok := m.pieceIndex[piece]; ok {
		return id
	}
	return m.unkID
}
