package bm25

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func tokenize(s string) []string { return strings.Fields(strings.ToLower(s)) }

var testCorpus = []string{
	"the cat sat on the mat",
	"the dog chased the cat",
	"the bird flew over the dog",
}

func TestSerializeRoundTripOkapi(t *testing.T) {
	orig, err := NewBM25Okapi(testCorpus, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25Okapi(&buf, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Compare scoring.
	query := []string{"cat", "dog"}
	origScores, err := orig.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}

	if len(origScores) != len(loadedScores) {
		t.Fatalf("score length mismatch: %d vs %d", len(origScores), len(loadedScores))
	}
	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestSerializeRoundTripPlus(t *testing.T) {
	orig, err := NewBM25Plus(testCorpus, tokenize, 1.5, 0.75, 1.0, 0.25, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25Plus(&buf, tokenize, 1.5, 0.75, 1.0, 0.25, nil)
	if err != nil {
		t.Fatal(err)
	}

	query := []string{"bird", "flew"}
	origScores, err := orig.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestSerializeRoundTripBatchScores(t *testing.T) {
	orig, err := NewBM25Okapi(testCorpus, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25Okapi(&buf, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	query := []string{"cat"}
	docIDs := []int{0, 2}
	origScores, err := orig.GetBatchScores(query, docIDs)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetBatchScores(query, docIDs)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestSerializeRoundTripL(t *testing.T) {
	orig, err := NewBM25L(testCorpus, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25L(&buf, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	query := []string{"cat", "dog"}
	origScores, err := orig.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestSerializeRoundTripT(t *testing.T) {
	orig, err := NewBM25T(testCorpus, tokenize, 1.5, 0.75, 1.0, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25T(&buf, tokenize, 1.5, 0.75, 1.0, nil)
	if err != nil {
		t.Fatal(err)
	}

	query := []string{"bird", "flew"}
	origScores, err := orig.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestSerializeRoundTripAdpt(t *testing.T) {
	orig, err := NewBM25Adpt(testCorpus, tokenize, 1.5, 0.75, 1.0, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25Adpt(&buf, tokenize, 1.5, 0.75, 1.0, nil)
	if err != nil {
		t.Fatal(err)
	}

	query := []string{"cat", "bird"}
	origScores, err := orig.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}
	loadedScores, err := loaded.GetScores(query)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origScores {
		if math.Abs(origScores[i]-loadedScores[i]) > 1e-12 {
			t.Errorf("score[%d]: orig=%v loaded=%v", i, origScores[i], loadedScores[i])
		}
	}
}

func TestLoadedMetadata(t *testing.T) {
	orig, err := NewBM25Okapi(testCorpus, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := orig.Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBM25Okapi(&buf, tokenize, 1.5, 0.75, nil)
	if err != nil {
		t.Fatal(err)
	}

	if loaded.CorpusSize() != orig.CorpusSize() {
		t.Errorf("corpus size: %d vs %d", loaded.CorpusSize(), orig.CorpusSize())
	}
	if math.Abs(loaded.AvgDocLen()-orig.AvgDocLen()) > 1e-12 {
		t.Errorf("avg doc len: %v vs %v", loaded.AvgDocLen(), orig.AvgDocLen())
	}

	origLens := orig.DocLengths()
	loadedLens := loaded.DocLengths()
	for i := range origLens {
		if origLens[i] != loadedLens[i] {
			t.Errorf("doc len[%d]: %d vs %d", i, origLens[i], loadedLens[i])
		}
	}
}
