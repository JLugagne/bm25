package bm25

import (
	"errors"
	"log"
)

// BM25L is an implementation of the BM25L variant.
type BM25L struct {
	*bm25Base
	k1 float64
	b  float64
}

// NewBM25L creates a new instance of the BM25L struct.
func NewBM25L(corpus []string, tokenizer func(string) []string, k1 float64, b float64, logger *log.Logger) (*BM25L, error) {
	if k1 < 0 {
		return nil, errors.New("k1 must be non-negative")
	}
	if b < 0 || b > 1 {
		return nil, errors.New("b must be between 0 and 1")
	}

	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25L{bm25Base: base, k1: k1, b: b}, nil
}

func (l *BM25L) GetScores(query []string) ([]float64, error) {
	return l.getScores(query, l.k1, l.b, okapiScore, scoreBatchOkapi)
}

func (l *BM25L) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return l.getBatchScores(query, docIDs, l.k1, l.b, okapiScore)
}

func (l *BM25L) GetTopN(query []string, n int) ([]string, error) {
	return l.getTopN(query, n, l.k1, l.b, okapiScore, scoreBatchOkapi)
}
