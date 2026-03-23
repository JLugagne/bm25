package bm25

import (
	"errors"
	"log"
)

// BM25Okapi implements the classic Okapi BM25 ranking function
// (Robertson & Walker, 1994).
//
// The scoring formula for a single query term q and document d is:
//
//	score(q, d) = IDF(q) * tf(q,d) / (tf(q,d) + k)
//
// where k = k1 * ((1 - b) + b * |d| / avgdl).
//
// Parameters k1 and b control term-frequency saturation and document-length
// normalization respectively. Typical defaults are k1 = 1.5 and b = 0.75.
type BM25Okapi struct {
	*bm25Base
	k1 float64
	b  float64
}

// NewBM25Okapi creates a new Okapi BM25 index over the given corpus.
//
// Parameters:
//   - corpus: slice of raw document strings to index
//   - tokenizer: splits a document string into tokens; must not return empty slices
//   - k1: term-frequency saturation parameter (must be >= 0; typical value: 1.5)
//   - b: document-length normalization parameter (must be in [0, 1]; typical value: 0.75)
//   - logger: optional logger for diagnostic messages; may be nil
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

// GetScores returns BM25 Okapi scores for every document in the corpus with
// respect to the given query tokens.
func (o *BM25Okapi) GetScores(query []string) ([]float64, error) {
	return o.getScores(query, o.k1, o.b, okapiScore, scoreBatchOkapi)
}

// GetBatchScores returns BM25 Okapi scores for the subset of documents
// identified by docIDs.
func (o *BM25Okapi) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return o.getBatchScores(query, docIDs, o.k1, o.b, okapiScore)
}

// GetTopN returns the top n highest-scoring documents for the given query
// using the Okapi BM25 ranking function.
func (o *BM25Okapi) GetTopN(query []string, n int) ([]string, error) {
	return o.getTopN(query, n, o.k1, o.b, okapiScore, scoreBatchOkapi)
}
