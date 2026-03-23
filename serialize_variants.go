package bm25

import (
	"io"
	"log"
)

// LoadBM25Okapi loads a serialized BM25Okapi index from r.
func LoadBM25Okapi(r io.Reader, tokenizer func(string) []string, k1, b float64, logger *log.Logger) (*BM25Okapi, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25Okapi{bm25Base: base, k1: k1, b: b}, nil
}

// LoadBM25L loads a serialized BM25L index from r.
func LoadBM25L(r io.Reader, tokenizer func(string) []string, k1, b float64, logger *log.Logger) (*BM25L, error) {
	base, err := LoadBase(r, tokenizer, logger)
	if err != nil {
		return nil, err
	}
	return &BM25L{bm25Base: base, k1: k1, b: b}, nil
}

// LoadBM25Plus loads a serialized BM25Plus index from r.
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

// LoadBM25T loads a serialized BM25T index from r.
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

// LoadBM25Adpt loads a serialized BM25Adpt index from r.
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
