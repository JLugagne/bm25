package bm25

import (
	"errors"
	"log"
)

// BM25Adpt implements the adaptive BM25 ranking variant. It shares the same
// scoring formula as [BM25T]:
//
//	score(q, d) = IDF(q) * (delta + tf(q,d) * (1+k) / (tf(q,d) + k))
//
// where k = k1 * ((1 - b) + b * |d| / avgdl).
//
// The adaptive variant is intended for use cases where parameters may be
// tuned per query or per collection.
type BM25Adpt struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	batchFn scoreBatchFunc
}

// NewBM25Adpt creates a new adaptive BM25 index over the given corpus.
//
// Parameters:
//   - corpus: slice of raw document strings to index
//   - tokenizer: splits a document string into tokens; must not return empty slices
//   - k1: term-frequency saturation parameter (must be >= 0; typical value: 1.5)
//   - b: document-length normalization parameter (must be in [0, 1]; typical value: 0.75)
//   - delta: lower-bound offset added to each term score (must be >= 0; typical value: 1.0)
//   - logger: optional logger for diagnostic messages; may be nil
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

// GetScores returns adaptive BM25 scores for every document in the corpus
// with respect to the given query tokens.
func (a *BM25Adpt) GetScores(query []string) ([]float64, error) {
	return a.getScores(query, a.k1, a.b, a.scoreFn, a.batchFn)
}

// GetBatchScores returns adaptive BM25 scores for the subset of documents
// identified by docIDs.
func (a *BM25Adpt) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return a.getBatchScores(query, docIDs, a.k1, a.b, a.scoreFn)
}

// GetTopN returns the top n highest-scoring documents for the given query
// using the adaptive BM25 ranking function.
func (a *BM25Adpt) GetTopN(query []string, n int) ([]string, error) {
	return a.getTopN(query, n, a.k1, a.b, a.scoreFn, a.batchFn)
}
