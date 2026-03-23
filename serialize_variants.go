package bm25

import (
	"io"
	"log"
)

// LoadBM25Okapi deserializes a BM25 Okapi index previously written with
// Serialize. The caller must supply the same tokenizer used during original
// construction, along with the k1 and b parameters.
func LoadBM25Okapi(r io.Reader, tokenizer func(string) []string, k1, b float64, logger *log.Logger) (*BM25Okapi, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25Okapi{bm25Base: base, k1: k1, b: b}, nil
}

// LoadBM25L deserializes a BM25L index previously written with Serialize.
// The caller must supply the same tokenizer used during original construction,
// along with the k1 and b parameters.
func LoadBM25L(r io.Reader, tokenizer func(string) []string, k1, b float64, logger *log.Logger) (*BM25L, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25L{bm25Base: base, k1: k1, b: b}, nil
}

// LoadBM25Plus deserializes a BM25+ index previously written with Serialize.
// The caller must supply the same tokenizer used during original construction,
// along with the k1, b, delta, and epsilon parameters.
func LoadBM25Plus(r io.Reader, tokenizer func(string) []string, k1, b, delta, epsilon float64, logger *log.Logger) (*BM25Plus, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25Plus{
		bm25Base: base, k1: k1, b: b, delta: delta, epsilon: epsilon,
		batchFn: makeBatchPlus(delta),
	}, nil
}

// LoadBM25T deserializes a BM25T index previously written with Serialize.
// The caller must supply the same tokenizer used during original construction,
// along with the k1, b, and delta parameters.
func LoadBM25T(r io.Reader, tokenizer func(string) []string, k1, b, delta float64, logger *log.Logger) (*BM25T, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25T{
		bm25Base: base, k1: k1, b: b, delta: delta,
		batchFn: makeBatchTF(delta),
	}, nil
}

// LoadBM25Adpt deserializes an adaptive BM25 index previously written with
// Serialize. The caller must supply the same tokenizer used during original
// construction, along with the k1, b, and delta parameters.
func LoadBM25Adpt(r io.Reader, tokenizer func(string) []string, k1, b, delta float64, logger *log.Logger) (*BM25Adpt, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25Adpt{
		bm25Base: base, k1: k1, b: b, delta: delta,
		batchFn: makeBatchTF(delta),
	}, nil
}
