package bm25

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// --- Shared testdata loaders ---

var tokenizer = func(s string) []string { return strings.Split(s, " ") }

func loadCorpus(b *testing.B, size int) []string {
	b.Helper()
	data, err := os.ReadFile(fmt.Sprintf("testdata/bench_corpus_%d.json", size))
	if err != nil {
		b.Fatalf("failed to load corpus: %v", err)
	}
	var corpus []string
	if err := json.Unmarshal(data, &corpus); err != nil {
		b.Fatalf("failed to parse corpus: %v", err)
	}
	return corpus
}

func loadQuery(b *testing.B, n int) []string {
	b.Helper()
	data, err := os.ReadFile(fmt.Sprintf("testdata/bench_query_%d.json", n))
	if err != nil {
		b.Fatalf("failed to load query: %v", err)
	}
	var query []string
	if err := json.Unmarshal(data, &query); err != nil {
		b.Fatalf("failed to parse query: %v", err)
	}
	return query
}

// --- Helpers to build each variant ---

func newOkapi(b *testing.B, corpus []string) *BM25Okapi {
	b.Helper()
	o, err := NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		b.Fatalf("NewBM25Okapi: %v", err)
	}
	return o
}

func newL(b *testing.B, corpus []string) *BM25L {
	b.Helper()
	l, err := NewBM25L(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		b.Fatalf("NewBM25L: %v", err)
	}
	return l
}

func newPlus(b *testing.B, corpus []string) *BM25Plus {
	b.Helper()
	p, err := NewBM25Plus(corpus, tokenizer, 1.5, 0.75, 1.0, 0.25, nil)
	if err != nil {
		b.Fatalf("NewBM25Plus: %v", err)
	}
	return p
}

func newT(b *testing.B, corpus []string) *BM25T {
	b.Helper()
	t, err := NewBM25T(corpus, tokenizer, 1.5, 0.75, 1.0, nil)
	if err != nil {
		b.Fatalf("NewBM25T: %v", err)
	}
	return t
}

func newAdpt(b *testing.B, corpus []string) *BM25Adpt {
	b.Helper()
	a, err := NewBM25Adpt(corpus, tokenizer, 1.5, 0.75, 1.0, nil)
	if err != nil {
		b.Fatalf("NewBM25Adpt: %v", err)
	}
	return a
}

// ============================================================
// Benchmark: Construction
// ============================================================

func BenchmarkConstruction(b *testing.B) {
	for _, size := range []int{50, 100, 500, 1000} {
		corpus := loadCorpus(b, size)
		b.Run(fmt.Sprintf("BM25Okapi/%d_docs", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
			}
		})
	}
}

// ============================================================
// Benchmark: GetScores — variant comparison (50 docs, 3 terms)
// ============================================================

func BenchmarkGetScores_50docs_3terms(b *testing.B) {
	corpus := loadCorpus(b, 50)
	query := loadQuery(b, 3)

	b.Run("BM25Okapi", func(b *testing.B) {
		o := newOkapi(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			o.GetScores(query)
		}
	})
	b.Run("BM25L", func(b *testing.B) {
		l := newL(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			l.GetScores(query)
		}
	})
	b.Run("BM25Plus", func(b *testing.B) {
		p := newPlus(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.GetScores(query)
		}
	})
	b.Run("BM25T", func(b *testing.B) {
		t := newT(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			t.GetScores(query)
		}
	})
	b.Run("BM25Adpt", func(b *testing.B) {
		a := newAdpt(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			a.GetScores(query)
		}
	})
}

// ============================================================
// Benchmark: GetScores — 500 docs, 3 terms
// ============================================================

func BenchmarkGetScores_500docs_3terms(b *testing.B) {
	corpus := loadCorpus(b, 500)
	query := loadQuery(b, 3)
	o := newOkapi(b, corpus)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o.GetScores(query)
	}
}

func BenchmarkGetTopN_500docs_3terms(b *testing.B) {
	corpus := loadCorpus(b, 500)
	query := loadQuery(b, 3)
	o := newOkapi(b, corpus)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o.GetTopN(query, 10)
	}
}

// ============================================================
// Benchmark: GetScores — 1000 docs, 5 terms
// ============================================================

func BenchmarkGetScores_1000docs_5terms(b *testing.B) {
	corpus := loadCorpus(b, 1000)
	query := loadQuery(b, 5)

	b.Run("BM25Okapi", func(b *testing.B) {
		o := newOkapi(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			o.GetScores(query)
		}
	})
	b.Run("BM25L", func(b *testing.B) {
		l := newL(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			l.GetScores(query)
		}
	})
	b.Run("BM25Plus", func(b *testing.B) {
		p := newPlus(b, corpus)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.GetScores(query)
		}
	})
}

func BenchmarkGetTopN_1000docs_5terms(b *testing.B) {
	corpus := loadCorpus(b, 1000)
	query := loadQuery(b, 5)
	o := newOkapi(b, corpus)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o.GetTopN(query, 10)
	}
}

// ============================================================
// Benchmark: Corpus scaling (BM25Okapi, 3-term query)
// ============================================================

func BenchmarkCorpusScaling(b *testing.B) {
	query := loadQuery(b, 3)
	for _, size := range []int{50, 100, 500, 1000} {
		corpus := loadCorpus(b, size)
		o := newOkapi(b, corpus)
		b.Run(fmt.Sprintf("%d_docs", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				o.GetScores(query)
			}
		})
	}
}

// ============================================================
// Benchmark: Query scaling (BM25Okapi, 500 docs)
// ============================================================

func BenchmarkQueryScaling(b *testing.B) {
	corpus := loadCorpus(b, 500)
	o := newOkapi(b, corpus)
	for _, n := range []int{1, 3, 5, 10} {
		query := loadQuery(b, n)
		b.Run(fmt.Sprintf("%d_terms", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				o.GetScores(query)
			}
		})
	}
}

// ============================================================
// Benchmark: Memory allocation tracking
// ============================================================

func BenchmarkAllocs_GetScores(b *testing.B) {
	corpus := loadCorpus(b, 500)
	query := loadQuery(b, 3)
	o := newOkapi(b, corpus)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o.GetScores(query)
	}
}

func BenchmarkAllocs_Construction(b *testing.B) {
	corpus := loadCorpus(b, 500)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	}
}
