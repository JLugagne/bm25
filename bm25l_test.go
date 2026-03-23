package bm25

import (
	"strings"
	"testing"
)

func TestNewBM25L(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }

	_, err := NewBM25L(corpus, tokenizer, -1.0, 0.75, nil)
	if err == nil {
		t.Errorf("Expected an error for negative k1, but got nil")
	}

	_, err = NewBM25L(corpus, tokenizer, 1.2, 1.5, nil)
	if err == nil {
		t.Errorf("Expected an error for b outside the range [0, 1], but got nil")
	}

	_, err = NewBM25L(corpus, tokenizer, 1.2, 0.75, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestBM25LGetScores(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	bm25, _ := NewBM25L(corpus, tokenizer, 1.2, 0.75, nil)

	_, err := bm25.GetScores([]string{})
	if err == nil {
		t.Errorf("Expected an error for an empty query, but got nil")
	}

	scores, err := bm25.GetScores([]string{"hello"})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expected := []float64{0.3648143055578659, 0.0}
	for i, score := range scores {
		if !almostEqual(score, expected[i]) {
			t.Errorf("Expected score %.16f at index %d, but got %.16f", expected[i], i, score)
		}
	}

	scores, err = bm25.GetScores([]string{"this", "test"})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expected = []float64{0.0, 0.5545177444479562}
	for i, score := range scores {
		if !almostEqual(score, expected[i]) {
			t.Errorf("Expected score %.16f at index %d, but got %.16f", expected[i], i, score)
		}
	}
}

func TestBM25LGetBatchScores(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	bm25, _ := NewBM25L(corpus, tokenizer, 1.2, 0.75, nil)

	_, err := bm25.GetBatchScores([]string{}, []int{0, 1})
	if err == nil {
		t.Errorf("Expected an error for an empty query, but got nil")
	}

	_, err = bm25.GetBatchScores([]string{"hello"}, []int{})
	if err == nil {
		t.Errorf("Expected an error for an empty document IDs slice, but got nil")
	}

	_, err = bm25.GetBatchScores([]string{"hello"}, []int{-1, 2})
	if err == nil {
		t.Errorf("Expected an error for invalid document IDs, but got nil")
	}

	scores, err := bm25.GetBatchScores([]string{"hello"}, []int{0})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expected := []float64{0.3648143055578659}
	for i, score := range scores {
		if !almostEqual(score, expected[i]) {
			t.Errorf("Expected score %.16f at index %d, but got %.16f", expected[i], i, score)
		}
	}

	scores, err = bm25.GetBatchScores([]string{"this", "test"}, []int{1})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	expected = []float64{0.5545177444479562}
	for i, score := range scores {
		if !almostEqual(score, expected[i]) {
			t.Errorf("Expected score %.16f at index %d, but got %.16f", expected[i], i, score)
		}
	}
}

func TestBM25LGetTopN(t *testing.T) {
	corpus := []string{"hello world", "this is a test"}
	tokenizer := func(s string) []string { return strings.Split(s, " ") }
	bm25, _ := NewBM25L(corpus, tokenizer, 1.2, 0.75, nil)

	_, err := bm25.GetTopN([]string{}, 2)
	if err == nil {
		t.Errorf("Expected an error for an empty query, but got nil")
	}

	_, err = bm25.GetTopN([]string{"hello"}, 0)
	if err == nil {
		t.Errorf("Expected an error for n <= 0, but got nil")
	}

	topDocs, err := bm25.GetTopN([]string{"hello"}, 1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(topDocs) != 1 || topDocs[0] != "hello world" {
		t.Errorf("Expected ['hello world'], but got %v", topDocs)
	}

	topDocs, err = bm25.GetTopN([]string{"this", "test"}, 1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(topDocs) != 1 || topDocs[0] != "this is a test" {
		t.Errorf("Expected ['this is a test'], but got %v", topDocs)
	}
}
