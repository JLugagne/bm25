package bm25

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

// TestCrossValidation loads the shared testdata corpus and verifies that
// Go's BM25 scoring matches the expected values computed by the Python
// reference implementation. Both use IDF = log(N/df) and Okapi BM25 scoring.
func TestCrossValidation(t *testing.T) {
	corpusData, err := os.ReadFile("testdata/corpus.json")
	if err != nil {
		t.Fatalf("Failed to read corpus.json: %v", err)
	}
	queriesData, err := os.ReadFile("testdata/queries.json")
	if err != nil {
		t.Fatalf("Failed to read queries.json: %v", err)
	}
	expectedData, err := os.ReadFile("testdata/expected_okapi.json")
	if err != nil {
		t.Fatalf("Failed to read expected_okapi.json: %v", err)
	}

	var corpus []string
	if err := json.Unmarshal(corpusData, &corpus); err != nil {
		t.Fatalf("Failed to parse corpus.json: %v", err)
	}

	var queries [][]string
	if err := json.Unmarshal(queriesData, &queries); err != nil {
		t.Fatalf("Failed to parse queries.json: %v", err)
	}

	var expected struct {
		K1     float64     `json:"k1"`
		B      float64     `json:"b"`
		Scores [][]float64 `json:"scores"`
		Top3   [][]int     `json:"top3"`
	}
	if err := json.Unmarshal(expectedData, &expected); err != nil {
		t.Fatalf("Failed to parse expected_okapi.json: %v", err)
	}

	tokenizer := func(s string) []string { return strings.Split(s, " ") }

	bm25, err := NewBM25Okapi(corpus, tokenizer, expected.K1, expected.B, nil)
	if err != nil {
		t.Fatalf("Failed to create BM25Okapi: %v", err)
	}

	const tol = 1e-10

	for qi, query := range queries {
		t.Run("query_"+strings.Join(query, "_"), func(t *testing.T) {
			scores, err := bm25.GetScores(query)
			if err != nil {
				t.Fatalf("GetScores failed: %v", err)
			}

			if len(scores) != len(expected.Scores[qi]) {
				t.Fatalf("Expected %d scores, got %d", len(expected.Scores[qi]), len(scores))
			}

			for di, exp := range expected.Scores[qi] {
				if math.Abs(scores[di]-exp) > tol {
					t.Errorf("doc %d: expected %.16f, got %.16f (diff=%.2e)",
						di, exp, scores[di], math.Abs(scores[di]-exp))
				}
			}

			// Verify TopN
			topDocs, err := bm25.GetTopN(query, 3)
			if err != nil {
				t.Fatalf("GetTopN failed: %v", err)
			}

			for i, expectedIdx := range expected.Top3[qi] {
				expectedDoc := corpus[expectedIdx]
				if i < len(topDocs) && topDocs[i] != expectedDoc {
					t.Errorf("top[%d]: expected doc %d (%q), got %q",
						i, expectedIdx, expectedDoc[:40], topDocs[i][:40])
				}
			}
		})
	}
}
