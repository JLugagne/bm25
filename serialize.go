package bm25

import (
	"fmt"
	"io"
	"log"
	"sort"
	"strings"

	"github.com/JLugagne/bm25/safetensors"
)

const (
	tensorDocLens   = "doc_lens"
	tensorAvgDocLen = "avg_doc_len"
	prefixIDF       = "idf:"
	prefixTF        = "tf:"
	metaCorpusSize  = "corpus_size"
	metaTerms       = "terms" // comma-separated sorted term list for stable ordering
)

// Serialize writes the precomputed vectors from a bm25Base (IDF map,
// term-frequency vectors, doc lengths, avg doc length) to w in safetensors
// format. The original corpus text is NOT stored — only the numeric vectors
// needed for scoring.
func (b *bm25Base) Serialize(w io.Writer) error {
	// Collect terms in sorted order for deterministic output.
	terms := make([]string, 0, len(b.idf))
	for t := range b.idf {
		terms = append(terms, t)
	}
	sort.Strings(terms)

	// 2 fixed tensors + 2 per term (idf scalar + tf vector).
	tensors := make([]safetensors.Tensor, 0, 2+2*len(terms))

	tensors = append(tensors, safetensors.Float64Tensor(tensorDocLens, b.docLensF64))
	tensors = append(tensors, safetensors.Float64ScalarTensor(tensorAvgDocLen, b.avgDocLen))

	for _, t := range terms {
		tensors = append(tensors, safetensors.Float64ScalarTensor(prefixIDF+t, b.idf[t]))
		tensors = append(tensors, safetensors.Float64Tensor(prefixTF+t, b.termFreqVecs[t]))
	}

	f := &safetensors.File{
		Tensors: tensors,
		Metadata: map[string]string{
			metaCorpusSize: fmt.Sprintf("%d", b.corpusSize),
			metaTerms:      strings.Join(terms, ","),
		},
	}

	return safetensors.Serialize(w, f)
}

// LoadBase reads precomputed BM25 vectors from r (safetensors format) and
// reconstructs a bm25Base. The caller must supply the same tokenizer used
// during construction (needed for future queries) and an optional logger.
//
// The original corpus text is NOT restored — only the numeric vectors. Methods
// that rely on the tokenized corpus (e.g. GetTopN returning document text)
// will not work unless the corpus is re-attached separately.
func LoadBase(r io.Reader, tokenizer func(string) []string, logger *log.Logger) (*bm25Base, error) {
	f, err := safetensors.Deserialize(r)
	if err != nil {
		return nil, fmt.Errorf("bm25: deserialize: %w", err)
	}

	// Index tensors by name.
	byName := make(map[string]*safetensors.Tensor, len(f.Tensors))
	for i := range f.Tensors {
		byName[f.Tensors[i].Name] = &f.Tensors[i]
	}

	// Parse corpus size from metadata.
	csStr, ok := f.Metadata[metaCorpusSize]
	if !ok {
		return nil, fmt.Errorf("bm25: missing metadata key %q", metaCorpusSize)
	}
	var corpusSize int
	if _, err := fmt.Sscanf(csStr, "%d", &corpusSize); err != nil {
		return nil, fmt.Errorf("bm25: parse corpus_size: %w", err)
	}

	// Doc lengths.
	dlTensor, ok := byName[tensorDocLens]
	if !ok {
		return nil, fmt.Errorf("bm25: missing tensor %q", tensorDocLens)
	}
	docLensF64, err := safetensors.BytesToFloat64(dlTensor.Data)
	if err != nil {
		return nil, fmt.Errorf("bm25: decode doc_lens: %w", err)
	}
	if len(docLensF64) != corpusSize {
		return nil, fmt.Errorf("bm25: doc_lens length %d != corpus_size %d", len(docLensF64), corpusSize)
	}

	docLengths := make([]int, corpusSize)
	for i, v := range docLensF64 {
		docLengths[i] = int(v)
	}

	// Avg doc length.
	adlTensor, ok := byName[tensorAvgDocLen]
	if !ok {
		return nil, fmt.Errorf("bm25: missing tensor %q", tensorAvgDocLen)
	}
	adlVals, err := safetensors.BytesToFloat64(adlTensor.Data)
	if err != nil || len(adlVals) != 1 {
		return nil, fmt.Errorf("bm25: decode avg_doc_len: %w", err)
	}
	avgDocLen := adlVals[0]

	// Reconstruct IDF and term frequency vectors.
	idfMap := make(map[string]float64)
	termFreqVecs := make(map[string][]float64)

	for name, tensor := range byName {
		if !strings.HasPrefix(name, prefixIDF) {
			continue
		}
		term := strings.TrimPrefix(name, prefixIDF)

		idfVals, err := safetensors.BytesToFloat64(tensor.Data)
		if err != nil || len(idfVals) != 1 {
			return nil, fmt.Errorf("bm25: decode idf for %q: %w", term, err)
		}
		idfMap[term] = idfVals[0]

		tfTensor, ok := byName[prefixTF+term]
		if !ok {
			return nil, fmt.Errorf("bm25: missing tf vector for term %q", term)
		}
		tfVals, err := safetensors.BytesToFloat64(tfTensor.Data)
		if err != nil {
			return nil, fmt.Errorf("bm25: decode tf for %q: %w", term, err)
		}
		if len(tfVals) != corpusSize {
			return nil, fmt.Errorf("bm25: tf vector for %q has length %d, expected %d", term, len(tfVals), corpusSize)
		}
		termFreqVecs[term] = tfVals
	}

	if logger != nil {
		logger.Printf("Loaded BM25 index: corpus_size=%d, avg_doc_len=%.2f, terms=%d", corpusSize, avgDocLen, len(idfMap))
	}

	return &bm25Base{
		corpusSize:   corpusSize,
		avgDocLen:    avgDocLen,
		docLengths:   docLengths,
		docLensF64:   docLensF64,
		idf:          idfMap,
		termFreqVecs: termFreqVecs,
		tokenizer:    tokenizer,
		logger:       logger,
	}, nil
}
