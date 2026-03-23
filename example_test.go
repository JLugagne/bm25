package bm25_test

import (
	"bytes"
	"fmt"
	"log"
	"strings"

	"github.com/JLugagne/bm25"
)

func tokenizer(s string) []string {
	return strings.Fields(strings.ToLower(s))
}

func Example() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat", "mat"}, 2)
	if err != nil {
		log.Fatal(err)
	}

	for _, doc := range topDocs {
		fmt.Println(doc)
	}
	// Output:
	// the cat sat on the mat
	// the dog chased the cat
}

func ExampleNewBM25Okapi() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Corpus size:", bm.CorpusSize())
	fmt.Printf("Avg doc length: %.2f\n", bm.AvgDocLen())
	// Output:
	// Corpus size: 3
	// Avg doc length: 5.67
}

func ExampleBM25Okapi_GetScores() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	scores, err := bm.GetScores([]string{"cat"})
	if err != nil {
		log.Fatal(err)
	}

	for i, score := range scores {
		fmt.Printf("doc %d: %.4f\n", i, score)
	}
	// Output:
	// doc 0: 0.1580
	// doc 1: 0.1713
	// doc 2: 0.0000
}

func ExampleBM25Okapi_GetBatchScores() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Score only documents 0 and 2.
	scores, err := bm.GetBatchScores([]string{"mat"}, []int{0, 2})
	if err != nil {
		log.Fatal(err)
	}

	for i, score := range scores {
		fmt.Printf("result %d: %.4f\n", i, score)
	}
	// Output:
	// result 0: 0.1580
	// result 1: 0.1580
}

func ExampleBM25Okapi_GetTopN() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat", "mat"}, 2)
	if err != nil {
		log.Fatal(err)
	}

	for _, doc := range topDocs {
		fmt.Println(doc)
	}
	// Output:
	// the cat sat on the mat
	// the dog chased the cat
}

func ExampleNewBM25L() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25L(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat"}, 1)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(topDocs[0])
	// Output:
	// the dog chased the cat
}

func ExampleNewBM25Plus() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Plus(corpus, tokenizer, 1.5, 0.75, 1.0, 0.25, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat"}, 1)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(topDocs[0])
	// Output:
	// the dog chased the cat
}

func ExampleNewBM25T() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25T(corpus, tokenizer, 1.5, 0.75, 1.0, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat"}, 1)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(topDocs[0])
	// Output:
	// the cat sat on the mat
}

func ExampleNewBM25Adpt() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	bm, err := bm25.NewBM25Adpt(corpus, tokenizer, 1.5, 0.75, 1.0, nil)
	if err != nil {
		log.Fatal(err)
	}

	topDocs, err := bm.GetTopN([]string{"cat"}, 1)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(topDocs[0])
	// Output:
	// the cat sat on the mat
}

func ExampleCountTermFreq() {
	freq, err := bm25.CountTermFreq("cat", "the cat sat on the cat mat", tokenizer)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("cat appears", freq, "times")
	// Output:
	// cat appears 2 times
}

func ExampleTopNIndices() {
	scores := []float64{0.1, 0.9, 0.5, 0.3}

	indices, err := bm25.TopNIndices(scores, 2)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(indices)
	// Output:
	// [1 2]
}

func ExampleLoadBM25Okapi() {
	corpus := []string{
		"the cat sat on the mat",
		"the dog chased the cat",
		"the bird flew over the mat",
	}

	// Build and serialize an index.
	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	var buf bytes.Buffer
	if err := bm.Serialize(&buf); err != nil {
		log.Fatal(err)
	}

	// Load the index back.
	loaded, err := bm25.LoadBM25Okapi(&buf, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		log.Fatal(err)
	}

	scores, err := loaded.GetScores([]string{"cat"})
	if err != nil {
		log.Fatal(err)
	}

	for i, score := range scores {
		fmt.Printf("doc %d: %.4f\n", i, score)
	}
	// Output:
	// doc 0: 0.1580
	// doc 1: 0.1713
	// doc 2: 0.0000
}
