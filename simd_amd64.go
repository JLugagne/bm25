//go:build amd64

package bm25

import "golang.org/x/sys/cpu"

var hasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA

func computeKValsArch(kVals, docLens []float64, k1Base, k1InvAvg float64) {
	if hasAVX2 && len(kVals) >= 4 {
		computeKValsAVX2(kVals, docLens, k1Base, k1InvAvg)
		return
	}
	computeKValsScalar(kVals, docLens, k1Base, k1InvAvg)
}

func scoreBatchOkapiArch(scores, tf, kVals []float64, idf float64) {
	if hasAVX2 && len(scores) >= 4 {
		scoreBatchOkapiAVX2(scores, tf, kVals, idf)
		return
	}
	scoreBatchOkapiScalar(scores, tf, kVals, idf)
}

func scoreBatchPlusArch(scores, tf, kVals []float64, idf, delta float64) {
	if hasAVX2 && len(scores) >= 4 {
		scoreBatchPlusAVX2(scores, tf, kVals, idf, delta)
		return
	}
	scoreBatchPlusScalar(scores, tf, kVals, idf, delta)
}

func scoreBatchTFArch(scores, tf, kVals []float64, idf, delta float64) {
	if hasAVX2 && len(scores) >= 4 {
		scoreBatchTFAVX2(scores, tf, kVals, idf, delta)
		return
	}
	scoreBatchTFScalar(scores, tf, kVals, idf, delta)
}

// Assembly function declarations — implemented in simd_amd64.s

//go:noescape
func computeKValsAVX2(kVals, docLens []float64, k1Base, k1InvAvg float64)

//go:noescape
func scoreBatchOkapiAVX2(scores, tf, kVals []float64, idf float64)

//go:noescape
func scoreBatchPlusAVX2(scores, tf, kVals []float64, idf, delta float64)

//go:noescape
func scoreBatchTFAVX2(scores, tf, kVals []float64, idf, delta float64)
