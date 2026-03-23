# BM25 Golang Implementation

This is a fork of [crawlab-team/bm25](https://github.com/crawlab-team/bm25) — a comprehensive implementation of BM25 ranking variants in Go. BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query.

This fork includes significant correctness fixes, performance optimizations, and a complete rewrite of the scoring internals.

## Improvements Over Original

### Correctness Fixes
- **Fixed BM25 scoring math** — the original implementation had incorrect score calculations across all variants
- **Cross-validated against Python `rank_bm25`** — test suite verifies Go output matches the reference Python implementation on shared corpora

### Performance
- **AVX2 SIMD scoring** — hand-written x86-64 assembly for the inner scoring loops (Okapi, BM25L, BM25+, BM25T/Adpt), with automatic scalar fallback on non-AVX2 hardware
- **Precomputed TF vectors** — term frequencies are computed once at construction time and stored as dense `[]float64` vectors, eliminating per-query map lookups
- **Precomputed IDF map** — all IDF values computed at construction, not on every query
- **Immutable structs after construction** — all mutexes removed; structs are safe for concurrent reads with zero synchronization overhead
- **Vectorized k-value computation** — `k1*(1-b) + k1*b*docLen/avgDocLen` computed in bulk via SIMD
- **SafeTensors serialization** — save/load precomputed indexes using the [SafeTensors](https://huggingface.co/docs/safetensors/) binary format; skip corpus reprocessing on startup

### Code Quality
- **Flat package layout** — removed the nested `bm25/bm25` package; import directly as `github.com/crawlab-team/bm25`
- **Complete test suite** — unit tests for all 5 variants, utility functions, and cross-validation against Python
- **Shared benchmark corpus** — deterministic JSON test data used by both Go and Python benchmarks for fair comparison

## BM25 Variants

- **Okapi BM25** — the classic variant
- **BM25L** — addresses long-document bias
- **BM25+** — adds a lower-bound term frequency component
- **BM25-Adpt** — adaptive variant
- **BM25T** — term-frequency saturation variant

Based on ["A Study of Efficient and Robust IR Metrics"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.8440&rep=rep1&type=pdf).

## Benchmark Results: Go vs Python

Both implementations use the same shared test corpus (generated from `testdata/gen_bench_corpus.py`).

**Hardware:** Intel Core i7-1185G7 @ 3.00GHz (4C/8T, Tiger Lake), 32GB RAM, AVX2+FMA
**Go:** 1.25.8 linux/amd64
**Python:** 3.14.3 with `rank_bm25`

### GetScores (Okapi BM25)

| Benchmark | Go (ns/op) | Python (ns/op) | Speedup |
|---|---:|---:|---:|
| 50 docs, 3-term query | 361 | 33,573 | **93x** |
| 500 docs, 3-term query | 2,297 | 169,292 | **74x** |
| 1000 docs, 5-term query | 5,214 | 625,686 | **120x** |

### Construction (Okapi BM25)

| Corpus Size | Go (ns/op) | Python (ns/op) | Speedup |
|---|---:|---:|---:|
| 50 docs | 1,273,908 | 2,110,851 | **1.7x** |
| 100 docs | 2,592,636 | 4,183,753 | **1.6x** |
| 500 docs | 13,180,155 | 21,318,684 | **1.6x** |
| 1000 docs | 25,757,198 | 43,178,645 | **1.7x** |

### Corpus Scaling (Okapi BM25, 3-term query)

| Corpus Size | Go (ns/op) | Python (ns/op) | Speedup |
|---|---:|---:|---:|
| 50 docs | 362 | 33,484 | **93x** |
| 100 docs | 2,401 | 48,032 | **20x** |
| 500 docs | 7,963 | 166,155 | **21x** |
| 1000 docs | 14,335 | 384,981 | **27x** |

### Query Scaling (Okapi BM25, 500 docs)

| Query Terms | Go (ns/op) | Python (ns/op) | Speedup |
|---|---:|---:|---:|
| 1 term | 6,447 | 67,051 | **10x** |
| 3 terms | 7,508 | 169,361 | **23x** |
| 5 terms | 8,617 | 285,953 | **33x** |
| 10 terms | 12,186 | 593,126 | **49x** |

### Running Benchmarks

```bash
# Run the comparison script
./benchmark/run_benchmarks.sh

# Or run individually:
go test -bench=. -benchmem -benchtime=2s -run='^$'
python3 benchmark/bench_python.py
```

## Installation

```bash
go get github.com/crawlab-team/bm25
```

## Usage

### Initializing

```go
import (
    "strings"
    "github.com/crawlab-team/bm25"
)

corpus := []string{
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
}

tokenizer := func(s string) []string {
    return strings.Split(s, " ")
}

bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
if err != nil {
    // Handle error
}
```

### Ranking Documents

```go
query := tokenizer("windy London")

scores, err := bm.GetScores(query)
if err != nil {
    // Handle error
}
```

### Top-N Retrieval

```go
topDocs, err := bm.GetTopN(query, 10)
if err != nil {
    // Handle error
}
```

### Batch Scoring

Score a subset of documents by ID:

```go
scores, err := bm.GetBatchScores(query, []int{0, 2, 5})
if err != nil {
    // Handle error
}
```

### Serialization

Save a computed index to disk using [SafeTensors](https://huggingface.co/docs/safetensors/) format, then load it back without reprocessing the corpus:

```go
// Save the index.
f, _ := os.Create("index.safetensors")
bm.Serialize(f)
f.Close()

// Load it back (no corpus needed — only the tokenizer and parameters).
f, _ = os.Open("index.safetensors")
loaded, err := bm25.LoadBM25Okapi(f, tokenizer, 1.5, 0.75, nil)
f.Close()

scores, _ := loaded.GetScores(query)
```

All variants have a corresponding load function: `LoadBM25Okapi`, `LoadBM25L`, `LoadBM25Plus`, `LoadBM25T`, `LoadBM25Adpt`.

The SafeTensors format stores IDF values, term-frequency vectors, and document lengths as raw float64 tensors — no text-to-float parsing overhead on load. The `safetensors` sub-package is also usable standalone for any typed tensor storage.

## Contributing

Contributions welcome. Please open an issue or submit a pull request with appropriate tests.

## License

[MIT License](LICENSE)

## Acknowledgments

- Original Go implementation by [crawlab-team](https://github.com/crawlab-team/bm25)
- Python reference implementation by [Dorian Brown](https://github.com/dorianbrown/rank_bm25)
- BM25 variant research: Luca Pinto, Diego Ceccarelli, and Claudio Lucchese
