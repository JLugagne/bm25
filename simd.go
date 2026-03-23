package bm25

// computeKVals computes k values: kVals[i] = k1Base + k1InvAvg * docLens[i]
// This is the first step in scoring, producing the per-document k parameter.
func computeKVals(kVals, docLens []float64, k1Base, k1InvAvg float64) {
	computeKValsArch(kVals, docLens, k1Base, k1InvAvg)
}

// scoreBatchOkapi computes: scores[i] += idf * (tf[i] / (tf[i] + kVals[i]))
// This is the Okapi BM25 / BM25L scoring formula.
func scoreBatchOkapi(scores, tf, kVals []float64, idf float64) {
	scoreBatchOkapiArch(scores, tf, kVals, idf)
}

// scoreBatchPlus computes: scores[i] += idf * (delta + tf[i] / (tf[i] + kVals[i]))
// This is the BM25Plus scoring formula.
func makeBatchPlus(delta float64) scoreBatchFunc {
	return func(scores, tf, kVals []float64, idf float64) {
		scoreBatchPlusArch(scores, tf, kVals, idf, delta)
	}
}

// scoreBatchTF computes: scores[i] += idf * (delta + (tf[i]*(1+kVals[i]))/(tf[i]+kVals[i]))
// This is the BM25T / BM25Adpt scoring formula.
func makeBatchTF(delta float64) scoreBatchFunc {
	return func(scores, tf, kVals []float64, idf float64) {
		scoreBatchTFArch(scores, tf, kVals, idf, delta)
	}
}
