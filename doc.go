// Package bm25 provides efficient implementations of the BM25 family of
// ranking functions for full-text search and information retrieval.
//
// BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
// documents based on the query terms appearing in each document. This package
// offers five variants:
//
//   - [BM25Okapi] — the classic Okapi BM25 formulation (Robertson & Walker, 1994)
//   - [BM25L] — addresses the document-length normalization bias of Okapi BM25
//     by using the same scoring formula with different IDF weighting
//   - [BM25Plus] — adds a lower-bound delta to penalize long documents less
//     aggressively (Lv & Zhai, 2011)
//   - [BM25T] — a variant that uses an alternative term-frequency component
//     tf*(1+k) / (tf+k) plus a delta offset
//   - [BM25Adpt] — an adaptive variant sharing the BM25T scoring formula
//
// All variants implement the [BM25] interface. The index is fully precomputed
// at construction time (IDF values and per-term frequency vectors), making the
// resulting structs immutable and safe for concurrent reads without locks.
//
// # Quick Start
//
//	corpus := []string{
//	    "the cat sat on the mat",
//	    "the dog chased the cat",
//	    "the bird flew over the mat",
//	}
//	tokenizer := func(s string) []string { return strings.Fields(s) }
//
//	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	scores, _ := bm.GetScores([]string{"cat", "mat"})
//	topDocs, _ := bm.GetTopN([]string{"cat", "mat"}, 2)
//
// # SIMD Acceleration
//
// On amd64 platforms with AVX2 support, scoring is automatically accelerated
// using SIMD instructions. A scalar fallback is used on all other platforms.
//
// # Serialization
//
// Precomputed indexes can be serialized to and deserialized from the SafeTensors
// binary format using the [bm25Base.Serialize] method and the Load* functions
// ([LoadBM25Okapi], [LoadBM25L], etc.). This allows building an index once and
// reloading it without re-processing the corpus.
package bm25
