package bm25

// Scalar fallback implementations used on all architectures and as the
// tail loop for SIMD paths.

func computeKValsScalar(kVals, docLens []float64, k1Base, k1InvAvg float64) {
	for i := range kVals {
		kVals[i] = k1Base + k1InvAvg*docLens[i]
	}
}

func scoreBatchOkapiScalar(scores, tf, kVals []float64, idf float64) {
	for i := range scores {
		scores[i] += idf * (tf[i] / (tf[i] + kVals[i]))
	}
}

func scoreBatchPlusScalar(scores, tf, kVals []float64, idf, delta float64) {
	for i := range scores {
		scores[i] += idf * (delta + tf[i]/(tf[i]+kVals[i]))
	}
}

func scoreBatchTFScalar(scores, tf, kVals []float64, idf, delta float64) {
	for i := range scores {
		scores[i] += idf * (delta + (tf[i]*(1+kVals[i]))/(tf[i]+kVals[i]))
	}
}
