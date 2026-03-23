package bm25

import (
	"errors"
	"log"
)

// BM25T implements the BM25T ranking variant, which uses an alternative
// term-frequency component combined with a delta offset.
//
// The scoring formula for a single query term q and document d is:
//
//	score(q, d) = IDF(q) * (delta + tf(q,d) * (1+k) / (tf(q,d) + k))
//
// where k = k1 * ((1 - b) + b * |d| / avgdl).
type BM25T struct {
	*bm25Base
	k1      float64
	b       float64
	delta   float64
	batchFn scoreBatchFunc
}

// NewBM25T creates a new BM25T index over the given corpus.
//
// Parameters:
//   - corpus: slice of raw document strings to index
//   - tokenizer: splits a document string into tokens; must not return empty slices
//   - k1: term-frequency saturation parameter (must be >= 0; typical value: 1.5)
//   - b: document-length normalization parameter (must be in [0, 1]; typical value: 0.75)
//   - delta: lower-bound offset added to each term score (must be >= 0; typical value: 1.0)
//   - logger: optional logger for diagnostic messages; may be nil
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

// GetScores returns BM25T scores for every document in the corpus with
// respect to the given query tokens.
func (t *BM25T) GetScores(query []string) ([]float64, error) {
	return t.getScores(query, t.k1, t.b, t.scoreFn, t.batchFn)
}

// GetBatchScores returns BM25T scores for the subset of documents identified
// by docIDs.
func (t *BM25T) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return t.getBatchScores(query, docIDs, t.k1, t.b, t.scoreFn)
}

// GetTopN returns the top n highest-scoring documents for the given query
// using the BM25T ranking function.
func (t *BM25T) GetTopN(query []string, n int) ([]string, error) {
	return t.getTopN(query, n, t.k1, t.b, t.scoreFn, t.batchFn)
}
