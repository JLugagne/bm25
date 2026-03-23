// Basic example: index a small corpus and rank documents against a query.
package main

import (
	"fmt"
	"strings"

	"github.com/JLugagne/bm25"
)

func main() {
	corpus := []string{
		"The cat sat on the mat",
		"The dog chased the cat",
		"The bird flew over the mat",
		"A fish swam in the pond",
		"The cat and the dog played together",
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	// Create a BM25 Okapi index with default parameters (k1=1.5, b=0.75).
	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Corpus size: %d\n", bm.CorpusSize())
	fmt.Printf("Avg doc length: %.2f\n\n", bm.AvgDocLen())

	// Score all documents for a query.
	query := tokenizer("cat mat")
	scores, err := bm.GetScores(query)
	if err != nil {
		panic(err)
	}

	fmt.Println("Scores for query \"cat mat\":")
	for i, score := range scores {
		fmt.Printf("  [%d] %.4f  %s\n", i, score, corpus[i])
	}

	// Get the top 3 most relevant documents.
	fmt.Println("\nTop 3 documents:")
	topDocs, err := bm.GetTopN(query, 3)
	if err != nil {
		panic(err)
	}
	for i, doc := range topDocs {
		fmt.Printf("  %d. %s\n", i+1, doc)
	}
}
