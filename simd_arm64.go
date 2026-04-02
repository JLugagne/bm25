//go:build arm64

package bm25

func computeKValsArch(kVals, docLens []float64, k1Base, k1InvAvg float64) {
	if len(kVals) >= 2 {
		computeKValsNEON(kVals, docLens, k1Base, k1InvAvg)
		return
	}
	computeKValsScalar(kVals, docLens, k1Base, k1InvAvg)
}

func scoreBatchOkapiArch(scores, tf, kVals []float64, idf float64) {
	if len(scores) >= 2 {
		scoreBatchOkapiNEON(scores, tf, kVals, idf)
		return
	}
	scoreBatchOkapiScalar(scores, tf, kVals, idf)
}

func scoreBatchPlusArch(scores, tf, kVals []float64, idf, delta float64) {
	if len(scores) >= 2 {
		scoreBatchPlusNEON(scores, tf, kVals, idf, delta)
		return
	}
	scoreBatchPlusScalar(scores, tf, kVals, idf, delta)
}

func scoreBatchTFArch(scores, tf, kVals []float64, idf, delta float64) {
	if len(scores) >= 2 {
		scoreBatchTFNEON(scores, tf, kVals, idf, delta)
		return
	}
	scoreBatchTFScalar(scores, tf, kVals, idf, delta)
}

// Assembly function declarations — implemented in simd_arm64.s

//go:noescape
func computeKValsNEON(kVals, docLens []float64, k1Base, k1InvAvg float64)

//go:noescape
func scoreBatchOkapiNEON(scores, tf, kVals []float64, idf float64)

//go:noescape
func scoreBatchPlusNEON(scores, tf, kVals []float64, idf, delta float64)

//go:noescape
func scoreBatchTFNEON(scores, tf, kVals []float64, idf, delta float64)
