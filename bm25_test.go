package bm25

import (
	"strings"
	"testing"
)

func TestNewBM25Base(t *testing.T) {
	_, err := NewBM25Base([]string{}, func(s string) []string { return []string{} }, nil)
	if err == nil {
		t.Errorf("Expected an error for an empty corpus, but got nil")
	}

	_, err = NewBM25Base([]string{"hello", "world"}, nil, nil)
	if err == nil {
		t.Errorf("Expected an error for a nil tokenizer, but got nil")
	}

	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	_, err = NewBM25Base(corpus, tokenizer, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestCorpusSize(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	base, _ := NewBM25Base(corpus, tokenizer, nil)

	if base.CorpusSize() != 2 {
		t.Errorf("Expected corpus size 2, but got %d", base.CorpusSize())
	}
}

func TestAvgDocLen(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	base, _ := NewBM25Base(corpus, tokenizer, nil)

	if base.AvgDocLen() != 3.0 {
		t.Errorf("Expected average document length 3.0, but got %.2f", base.AvgDocLen())
	}
}

func TestDocLengths(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	base, _ := NewBM25Base(corpus, tokenizer, nil)

	expected := []int{2, 4}
	docLengths := base.DocLengths()
	if len(docLengths) != len(expected) {
		t.Errorf("Expected %d document lengths, but got %d", len(expected), len(docLengths))
	}
	for i, length := range docLengths {
		if length != expected[i] {
			t.Errorf("Expected document length %d at index %d, but got %d", expected[i], i, length)
		}
	}
}

func TestIDF(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	base, _ := NewBM25Base(corpus, tokenizer, nil)

	// Empty term
	_, err := base.IDF("")
	if err == nil {
		t.Errorf("Expected an error for an empty term, but got nil")
	}

	// Term not in corpus
	idf, err := base.IDF("nonexistent")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if idf != 0.0 {
		t.Errorf("Expected IDF 0.0 for a term not present in the corpus, but got %.2f", idf)
	}

	// Term present in all documents — with log(N/df), if df=N then idf=log(1)=0
	// Note: "is" only appears in doc1, so df=1, idf=log(2)=0.693...
	// We need a term in ALL docs. No such term in this corpus, so skip that test.

	// Term present in 1 of 2 documents: idf = log(2/1) = 0.693...
	idf, err = base.IDF("hello")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !almostEqual(idf, 0.6931471805599453) {
		t.Errorf("Expected IDF 0.6931471805599453 for the term 'hello', but got %.16f", idf)
	}

	// Test IDF caching — second call should return same value
	idf2, err := base.IDF("hello")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if idf != idf2 {
		t.Errorf("Expected cached IDF to match, got %.16f vs %.16f", idf, idf2)
	}
}
