package bm25

import (
	"errors"
	"log"
)

// BM25L implements the BM25L ranking variant, which addresses the
// document-length normalization bias present in the standard Okapi BM25.
//
// BM25L uses the same scoring formula as [BM25Okapi] but applies different
// IDF weighting to reduce the penalty on longer documents. Parameters k1 and b
// have the same semantics as in Okapi BM25.
type BM25L struct {
	*bm25Base
	k1 float64
	b  float64
}

// NewBM25L creates a new BM25L index over the given corpus.
//
// Parameters:
//   - corpus: slice of raw document strings to index
//   - tokenizer: splits a document string into tokens; must not return empty slices
//   - k1: term-frequency saturation parameter (must be >= 0; typical value: 1.5)
//   - b: document-length normalization parameter (must be in [0, 1]; typical value: 0.75)
//   - logger: optional logger for diagnostic messages; may be nil
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

// GetScores returns BM25L scores for every document in the corpus with
// respect to the given query tokens.
func (l *BM25L) GetScores(query []string) ([]float64, error) {
	return l.getScores(query, l.k1, l.b, okapiScore, scoreBatchOkapi)
}

// GetBatchScores returns BM25L scores for the subset of documents identified
// by docIDs.
func (l *BM25L) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return l.getBatchScores(query, docIDs, l.k1, l.b, okapiScore)
}

// GetTopN returns the top n highest-scoring documents for the given query
// using the BM25L ranking function.
func (l *BM25L) GetTopN(query []string, n int) ([]string, error) {
	return l.getTopN(query, n, l.k1, l.b, okapiScore, scoreBatchOkapi)
}
