// Variants example: compare all five BM25 variants on the same corpus and query.
package main

import (
	"fmt"
	"strings"

	"github.com/JLugagne/bm25"
)

func main() {
	corpus := []string{
		"information retrieval is the activity of obtaining resources relevant to an information need",
		"search engines use information retrieval techniques to index and rank web pages",
		"machine learning can improve information retrieval by learning relevance patterns",
		"natural language processing helps understand user queries for better retrieval",
		"the BM25 algorithm is a bag of words retrieval function used by search engines",
		"deep learning models can capture semantic meaning beyond keyword matching",
		"traditional retrieval methods rely on term frequency and inverse document frequency",
		"hybrid search combines keyword matching with semantic vector search",
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	query := tokenizer("information retrieval search engine")

	// BM25 Okapi — the classic variant.
	okapi, _ := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)

	// BM25L — reduces long-document penalty.
	l, _ := bm25.NewBM25L(corpus, tokenizer, 1.5, 0.75, nil)

	// BM25+ — adds a lower-bound term frequency component.
	plus, _ := bm25.NewBM25Plus(corpus, tokenizer, 1.5, 0.75, 1.0, 0.25, nil)

	// BM25-Adpt — adaptive variant.
	adpt, _ := bm25.NewBM25Adpt(corpus, tokenizer, 1.5, 0.75, 1.0, nil)

	// BM25T — term-frequency saturation variant.
	t, _ := bm25.NewBM25T(corpus, tokenizer, 1.5, 0.75, 1.0, nil)

	variants := []struct {
		name string
		impl bm25.BM25
	}{
		{"Okapi", okapi},
		{"BM25L", l},
		{"BM25+", plus},
		{"BM25-Adpt", adpt},
		{"BM25T", t},
	}

	for _, v := range variants {
		scores, err := v.impl.GetScores(query)
		if err != nil {
			panic(err)
		}

		fmt.Printf("=== %s ===\n", v.name)
		for i, score := range scores {
			if score > 0 {
				fmt.Printf("  [%d] %.4f  %s\n", i, score, corpus[i])
			}
		}
		fmt.Println()
	}
}
