// Serialize example: build a BM25 index, save it to disk, then load it back
// and verify that scores are identical.
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/JLugagne/bm25"
)

func main() {
	corpus := []string{
		"Go is a statically typed compiled language designed at Google",
		"Python is a high level interpreted language known for readability",
		"Rust is a systems programming language focused on safety and performance",
		"JavaScript is the language of the web used for both frontend and backend",
		"TypeScript adds static typing to JavaScript for larger codebases",
		"C is a low level language that provides direct memory access",
		"Java is a popular object oriented language that runs on the JVM",
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	// --- Build and save ---

	okapi, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		panic(err)
	}

	f, err := os.Create("index.safetensors")
	if err != nil {
		panic(err)
	}
	if err := okapi.Serialize(f); err != nil {
		f.Close()
		panic(err)
	}
	f.Close()

	stat, _ := os.Stat("index.safetensors")
	fmt.Printf("Saved index to index.safetensors (%d bytes)\n", stat.Size())

	// --- Load and query ---

	f, err = os.Open("index.safetensors")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	loaded, err := bm25.LoadBM25Okapi(f, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded index: %d docs, avg length %.2f\n\n", loaded.CorpusSize(), loaded.AvgDocLen())

	// Score a query with the loaded index.
	query := tokenizer("systems language performance")
	scores, err := loaded.GetScores(query)
	if err != nil {
		panic(err)
	}

	fmt.Println("Scores for query \"systems language performance\":")
	for i, score := range scores {
		fmt.Printf("  [%d] %.4f  %s\n", i, score, corpus[i])
	}

	// Verify scores match the original index.
	origScores, _ := okapi.GetScores(query)
	match := true
	for i := range scores {
		if scores[i] != origScores[i] {
			match = false
			break
		}
	}
	fmt.Printf("\nScores match original: %v\n", match)

	// Clean up.
	os.Remove("index.safetensors")
}
