package bm25

import (
	"errors"
	"log"
)

// BM25T is an implementation of the BM25T variant.
type BM25T struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	batchFn scoreBatchFunc
}

// NewBM25T creates a new instance of the BM25T struct.
func NewBM25T(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25T, error) {
	if k1 < 0 {
		return nil, errors.New("k1 must be non-negative")
	}
	if b < 0 || b > 1 {
		return nil, errors.New("b must be between 0 and 1")
	}
	if delta < 0 {
		return nil, errors.New("delta must be non-negative")
	}

	base, err := NewBM25Base(corpus, tokenizer, logger)
	if err != nil {
		return nil, err
	}

	return &BM25T{
		bm25Base: base, k1: k1, b: b, delta: delta,
		batchFn: makeBatchTF(delta),
	}, nil
}

func (t *BM25T) scoreFn(qFreq, k float64) float64 {
	return t.delta + (qFreq*(1+k))/(qFreq+k)
}

func (t *BM25T) GetScores(query []string) ([]float64, error) {
	return t.getScores(query, t.k1, t.b, t.scoreFn, t.batchFn)
}

func (t *BM25T) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return t.getBatchScores(query, docIDs, t.k1, t.b, t.scoreFn)
}

func (t *BM25T) GetTopN(query []string, n int) ([]string, error) {
	return t.getTopN(query, n, t.k1, t.b, t.scoreFn, t.batchFn)
}
