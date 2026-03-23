package bm25

import (
	"errors"
	"log"
)

// BM25Okapi is an implementation of the Okapi BM25 variant.
type BM25Okapi struct {
	*bm25Base
	k1 float64
	b  float64
}

// NewBM25Okapi creates a new instance of the BM25Okapi struct.
func NewBM25Okapi(corpus []string, tokenizer func(string) []string, k1 float64, b float64, logger *log.Logger) (*BM25Okapi, error) {
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

	return &BM25Okapi{bm25Base: base, k1: k1, b: b}, nil
}

func okapiScore(qFreq, k float64) float64 {
	return qFreq / (qFreq + k)
}

func (o *BM25Okapi) GetScores(query []string) ([]float64, error) {
	return o.getScores(query, o.k1, o.b, okapiScore, scoreBatchOkapi)
}

func (o *BM25Okapi) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return o.getBatchScores(query, docIDs, o.k1, o.b, okapiScore)
}

func (o *BM25Okapi) GetTopN(query []string, n int) ([]string, error) {
	return o.getTopN(query, n, o.k1, o.b, okapiScore, scoreBatchOkapi)
}
