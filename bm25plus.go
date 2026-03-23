package bm25

import (
	"errors"
	"log"
)

// BM25Plus is an implementation of the BM25Plus variant.
type BM25Plus struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	epsilon float64
	batchFn scoreBatchFunc
}

// NewBM25Plus creates a new instance of the BM25Plus struct.
func NewBM25Plus(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, epsilon float64, logger *log.Logger) (*BM25Plus, error) {
	if k1 < 0 {
		return nil, errors.New("k1 must be non-negative")
	}
	if b < 0 || b > 1 {
		return nil, errors.New("b must be between 0 and 1")
	}
	if delta < 0 {
		return nil, errors.New("delta must be non-negative")
	}
	if epsilon < 0 {
		return nil, errors.New("epsilon must be non-negative")
	}

	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25Plus{
		bm25Base: base, k1: k1, b: b, delta: delta, epsilon: epsilon,
		batchFn: makeBatchPlus(delta),
	}, nil
}

func (p *BM25Plus) scoreFn(qFreq, k float64) float64 {
	return p.delta + (qFreq / (qFreq + k))
}

func (p *BM25Plus) GetScores(query []string) ([]float64, error) {
	return p.getScores(query, p.k1, p.b, p.scoreFn, p.batchFn)
}

func (p *BM25Plus) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return p.getBatchScores(query, docIDs, p.k1, p.b, p.scoreFn)
}

func (p *BM25Plus) GetTopN(query []string, n int) ([]string, error) {
	return p.getTopN(query, n, p.k1, p.b, p.scoreFn, p.batchFn)
}
