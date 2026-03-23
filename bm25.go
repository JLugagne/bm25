package bm25

import (
	"errors"
	"fmt"
	"log"
	"math"
)

// BM25 is the common interface implemented by all BM25 ranking variants.
//
// Every implementation precomputes IDF values and term-frequency vectors at
// construction time, so all methods are safe for concurrent use.
type BM25 interface {
	// CorpusSize returns the number of documents in the indexed corpus.
	CorpusSize() int

	// AvgDocLen returns the average document length (in tokens) across the corpus.
	AvgDocLen() float64

	// DocLengths returns a slice containing the token count of each document,
	// indexed by document position in the original corpus.
	DocLengths() []int

	// IDF returns the precomputed inverse document frequency for term.
	// Terms not present in the corpus return 0 with a nil error.
	// An error is returned only if term is the empty string.
	IDF(term string) (float64, error)

	// GetScores returns BM25 scores for every document in the corpus with
	// respect to the given query tokens. The returned slice is indexed by
	// document position.
	GetScores(query []string) ([]float64, error)

	// GetBatchScores returns BM25 scores for the subset of documents
	// identified by docIDs. The returned slice is parallel to docIDs.
	GetBatchScores(query []string, docIDs []int) ([]float64, error)

	// GetTopN returns the top n highest-scoring documents for the given query,
	// returned as their joined token strings.
	GetTopN(query []string, n int) ([]string, error)
}

// scoreFunc computes the BM25 score component for a single document given
// the term frequency in that document and the precomputed k value.
// idf * scoreFunc(qFreq, k) produces the final per-term score contribution.
type scoreFunc func(qFreq, k float64) float64

// scoreBatchFunc is the SIMD-friendly batch scoring interface.
// It processes contiguous slices: for each i in [0, len(tf)),
// scores[i] += idf * f(tf[i], kVals[i]).
// Implementations may use AVX2 or scalar code depending on CPU features.
type scoreBatchFunc func(scores, tf, kVals []float64, idf float64)

// bm25Base is a base struct that holds common fields and methods for all BM25 variants.
// All fields are immutable after construction, making it safe for concurrent use
// without any synchronization.
type bm25Base struct {
	corpus       [][]string
	corpusSize   int
	avgDocLen    float64
	docLengths   []int
	docLensF64   []float64            // docLengths as float64 for vectorized scoring
	idf          map[string]float64   // precomputed IDF for every term in the corpus
	termFreqVecs map[string][]float64 // term -> per-document frequency vector
	tokenizer    func(string) []string
	logger       *log.Logger
}

// NewBM25Base creates a new bm25Base by tokenizing the corpus and precomputing
// all IDF values and per-term frequency vectors. The resulting struct is fully
// immutable and safe for concurrent reads without locks.
//
// The tokenizer function splits each document string into a slice of tokens.
// It must not return an empty slice for any document. An optional logger
// receives diagnostic messages during construction.
func NewBM25Base(corpus []string, tokenizer func(string) []string, logger *log.Logger) (*bm25Base, error) {
	if len(corpus) == 0 {
		return nil, errors.New("corpus cannot be empty")
	}

	if tokenizer == nil {
		return nil, errors.New("tokenizer function cannot be nil")
	}

	corpusSize := len(corpus)
	tokenized := make([][]string, corpusSize)
	docLengths := make([]int, corpusSize)
	docLensF64 := make([]float64, corpusSize)
	docFreqs := make(map[string]int)

	// termCounts[term][docIdx] = count of term in doc. Built during tokenization.
	termCounts := make(map[string][]float64)

	var totalDocLen int
	for i, doc := range corpus {
		tokens := tokenizer(doc)
		if len(tokens) == 0 {
			return nil, errors.New("tokenizer function returned an empty slice for document at index " + fmt.Sprintf("%d", i))
		}
		tokenized[i] = tokens
		docLengths[i] = len(tokens)
		docLensF64[i] = float64(len(tokens))
		totalDocLen += len(tokens)

		localFreq := make(map[string]float64)
		for _, token := range tokens {
			localFreq[token]++
		}
		for term, freq := range localFreq {
			vec, ok := termCounts[term]
			if !ok {
				vec = make([]float64, corpusSize)
				termCounts[term] = vec
				docFreqs[term] = 0
			}
			vec[i] = freq
			docFreqs[term]++
		}
	}

	avgDocLen := float64(totalDocLen) / float64(corpusSize)

	// Precompute IDF for every term. Only keep terms with positive IDF.
	idfMap := make(map[string]float64, len(docFreqs))
	termFreqVecs := make(map[string][]float64, len(docFreqs))
	fCorpusSize := float64(corpusSize)
	for term, df := range docFreqs {
		v := math.Log(fCorpusSize / float64(df))
		if v > 0 {
			idfMap[term] = v
			termFreqVecs[term] = termCounts[term]
		}
	}

	if logger != nil {
		logger.Printf("Corpus size: %d, Average document length: %.2f", corpusSize, avgDocLen)
	}

	return &bm25Base{
		corpus:       tokenized,
		corpusSize:   corpusSize,
		avgDocLen:    avgDocLen,
		docLengths:   docLengths,
		docLensF64:   docLensF64,
		idf:          idfMap,
		termFreqVecs: termFreqVecs,
		tokenizer:    tokenizer,
		logger:       logger,
	}, nil
}

// CorpusSize returns the number of documents in the indexed corpus.
func (b *bm25Base) CorpusSize() int {
	return b.corpusSize
}

// AvgDocLen returns the average document length (in tokens) across the corpus.
func (b *bm25Base) AvgDocLen() float64 {
	return b.avgDocLen
}

// DocLengths returns the token count of each document, indexed by document
// position in the original corpus.
func (b *bm25Base) DocLengths() []int {
	return b.docLengths
}

// IDF returns the precomputed inverse document frequency for term.
// Terms not present in the corpus return 0 with a nil error.
// An error is returned only if term is the empty string.
func (b *bm25Base) IDF(term string) (float64, error) {
	if term == "" {
		return 0, errors.New("term cannot be empty")
	}
	return b.idf[term], nil
}

// termIDF pairs a term with its precomputed IDF value and frequency vector.
type termIDF struct {
	term string
	idf  float64
	tf   []float64 // precomputed per-document term frequencies
}

func (b *bm25Base) idfTerms(query []string) []termIDF {
	terms := make([]termIDF, 0, len(query))
	for _, q := range query {
		if v := b.idf[q]; v > 0 {
			terms = append(terms, termIDF{q, v, b.termFreqVecs[q]})
		}
	}
	return terms
}

// scoreRange scores documents in [start, end) using scalar code with the
// precomputed term frequency vectors.
func (b *bm25Base) scoreRange(scores []float64, start, end int, terms []termIDF, k1, bParam float64, fn scoreFunc) {
	invAvg := bParam / b.avgDocLen
	k1Base := k1 * (1 - bParam)
	for _, t := range terms {
		tf := t.tf
		for i := start; i < end; i++ {
			k := k1Base + k1*invAvg*b.docLensF64[i]
			scores[i] += t.idf * fn(tf[i], k)
		}
	}
}

// scoreRangeSIMD scores documents in [start, end) using the batch scoring
// function which may use SIMD instructions on supported CPUs.
func (b *bm25Base) scoreRangeSIMD(scores []float64, start, end int, terms []termIDF, k1, bParam float64, fn scoreBatchFunc) {
	n := end - start
	kVals := make([]float64, n)
	k1InvAvg := k1 * bParam / b.avgDocLen
	k1Base := k1 * (1 - bParam)
	computeKVals(kVals, b.docLensF64[start:end], k1Base, k1InvAvg)

	for _, t := range terms {
		fn(scores[start:end], t.tf[start:end], kVals, t.idf)
	}
}

// getScores is the shared implementation for GetScores across all variants.
func (b *bm25Base) getScores(query []string, k1, bParam float64, fn scoreFunc, batchFn scoreBatchFunc) ([]float64, error) {
	if len(query) == 0 {
		return nil, errors.New("query cannot be empty")
	}

	terms := b.idfTerms(query)
	scores := make([]float64, b.corpusSize)

	if len(terms) == 0 {
		return scores, nil
	}

	if batchFn != nil {
		b.scoreRangeSIMD(scores, 0, b.corpusSize, terms, k1, bParam, batchFn)
	} else {
		b.scoreRange(scores, 0, b.corpusSize, terms, k1, bParam, fn)
	}

	return scores, nil
}

// scoreRangeByIDs scores a subset of documents identified by docIDs.
func (b *bm25Base) scoreRangeByIDs(scores []float64, docIDs []int, terms []termIDF, k1, bParam float64, fn scoreFunc) {
	invAvg := bParam / b.avgDocLen
	k1Base := k1 * (1 - bParam)
	for _, t := range terms {
		tf := t.tf
		for i, docID := range docIDs {
			k := k1Base + k1*invAvg*b.docLensF64[docID]
			scores[i] += t.idf * fn(tf[docID], k)
		}
	}
}

// getBatchScores is the shared implementation for GetBatchScores across all variants.
func (b *bm25Base) getBatchScores(query []string, docIDs []int, k1, bParam float64, fn scoreFunc) ([]float64, error) {
	if len(query) == 0 {
		return nil, errors.New("query cannot be empty")
	}
	if len(docIDs) == 0 {
		return nil, errors.New("document IDs cannot be empty")
	}
	if err := b.validateDocIDs(docIDs); err != nil {
		return nil, err
	}

	terms := b.idfTerms(query)
	scores := make([]float64, len(docIDs))

	if len(terms) == 0 {
		return scores, nil
	}

	b.scoreRangeByIDs(scores, docIDs, terms, k1, bParam, fn)
	return scores, nil
}

// getTopN is the shared implementation for GetTopN across all variants.
func (b *bm25Base) getTopN(query []string, n int, k1, bParam float64, fn scoreFunc, batchFn scoreBatchFunc) ([]string, error) {
	if len(query) == 0 {
		return nil, errors.New("query cannot be empty")
	}
	if n <= 0 {
		return nil, errors.New("n must be a positive integer")
	}

	scores, err := b.getScores(query, k1, bParam, fn, batchFn)
	if err != nil {
		return nil, err
	}

	topNIndices, err := TopNIndices(scores, n)
	if err != nil {
		return nil, err
	}

	topDocs := make([]string, len(topNIndices))
	for i, idx := range topNIndices {
		topDocs[i] = JoinTokens(b.corpus[idx], " ")
	}

	return topDocs, nil
}

// GetScores returns BM25 scores for every document in the corpus.
// This base implementation always returns an error; use a concrete variant instead.
func (b *bm25Base) GetScores(query []string) ([]float64, error) {
	return nil, errors.New("not implemented")
}

// GetBatchScores returns BM25 scores for the documents identified by docIDs.
// This base implementation always returns an error; use a concrete variant instead.
func (b *bm25Base) GetBatchScores(query []string, docIDs []int) ([]float64, error) {
	return nil, errors.New("not implemented")
}

// GetTopN returns the top n highest-scoring documents for the given query.
// This base implementation always returns an error; use a concrete variant instead.
func (b *bm25Base) GetTopN(query []string, n int) ([]string, error) {
	return nil, errors.New("not implemented")
}

// validateDocIDs checks that all document IDs are within valid range.
func (b *bm25Base) validateDocIDs(docIDs []int) error {
	for _, docID := range docIDs {
		if docID < 0 || docID >= b.corpusSize {
			return fmt.Errorf("invalid document ID: %d", docID)
		}
	}
	return nil
}
