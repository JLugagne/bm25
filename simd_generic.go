//go:build !amd64

package bm25

func computeKValsArch(kVals, docLens []float64, k1Base, k1InvAvg float64) {
	computeKValsScalar(kVals, docLens, k1Base, k1InvAvg)
}

func scoreBatchOkapiArch(scores, tf, kVals []float64, idf float64) {
	scoreBatchOkapiScalar(scores, tf, kVals, idf)
}

func scoreBatchPlusArch(scores, tf, kVals []float64, idf, delta float64) {
	scoreBatchPlusScalar(scores, tf, kVals, idf, delta)
}

func scoreBatchTFArch(scores, tf, kVals []float64, idf, delta float64) {
	scoreBatchTFScalar(scores, tf, kVals, idf, delta)
}
