package bm25

import (
	"errors"
	"log"
)

// BM25Adpt is an implementation of the BM25Adpt variant.
type BM25Adpt struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	batchFn scoreBatchFunc
}

// NewBM25Adpt creates a new instance of the BM25Adpt struct.
func NewBM25Adpt(corpus []string, tokenizer func(string) []string, k1 float64, b float64, delta float64, logger *log.Logger) (*BM25Adpt, error) {
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

	return &BM25Adpt{
		bm25Base: base, k1: k1, b: b, delta: delta,
		batchFn: makeBatchTF(delta),
	}, nil
}

func (a *BM25Adpt) scoreFn(qFreq, k float64) float64 {
	return a.delta + (qFreq*(1+k))/(qFreq+k)
}

func (a *BM25Adpt) GetScores(query []string) ([]float64, error) {
	return a.getScores(query, a.k1, a.b, a.scoreFn, a.batchFn)
}

func (a *BM25Adpt) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return a.getBatchScores(query, docIDs, a.k1, a.b, a.scoreFn)
}

func (a *BM25Adpt) GetTopN(query []string, n int) ([]string, error) {
	return a.getTopN(query, n, a.k1, a.b, a.scoreFn, a.batchFn)
}
