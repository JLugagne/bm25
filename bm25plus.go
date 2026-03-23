package bm25

import (
	"errors"
	"log"
)

// BM25Plus implements the BM25+ ranking variant (Lv & Zhai, 2011), which adds
// a lower-bound delta term to the scoring formula to avoid over-penalizing
// long documents.
//
// The scoring formula for a single query term q and document d is:
//
//	score(q, d) = IDF(q) * (delta + tf(q,d) / (tf(q,d) + k))
//
// where k = k1 * ((1 - b) + b * |d| / avgdl).
type BM25Plus struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	epsilon float64
	batchFn scoreBatchFunc
}

// NewBM25Plus creates a new BM25+ index over the given corpus.
//
// Parameters:
//   - corpus: slice of raw document strings to index
//   - tokenizer: splits a document string into tokens; must not return empty slices
//   - k1: term-frequency saturation parameter (must be >= 0; typical value: 1.5)
//   - b: document-length normalization parameter (must be in [0, 1]; typical value: 0.75)
//   - delta: lower-bound offset added to each term score (must be >= 0; typical value: 1.0)
//   - epsilon: smoothing parameter (must be >= 0)
//   - logger: optional logger for diagnostic messages; may be nil
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

// GetScores returns BM25+ scores for every document in the corpus with
// respect to the given query tokens.
func (p *BM25Plus) GetScores(query []string) ([]float64, error) {
	return p.getScores(query, p.k1, p.b, p.scoreFn, p.batchFn)
}

// GetBatchScores returns BM25+ scores for the subset of documents identified
// by docIDs.
func (p *BM25Plus) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return p.getBatchScores(query, docIDs, p.k1, p.b, p.scoreFn)
}

// GetTopN returns the top n highest-scoring documents for the given query
// using the BM25+ ranking function.
func (p *BM25Plus) GetTopN(query []string, n int) ([]string, error) {
	return p.getTopN(query, n, p.k1, p.b, p.scoreFn, p.batchFn)
}
