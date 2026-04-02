#include "textflag.h"

// ARM64 NEON (AdvSIMD) implementations for BM25 batch scoring.
// Processes 2 float64s per iteration using 128-bit V registers.
//
// Go's arm64 assembler lacks NEON floating-point arithmetic instructions
// (fadd, fmul, fdiv for vectors), so we encode them as raw WORD values.
// Macro naming: VFADD/VFMUL/VFDIV Vd.2D, Vn.2D, Vm.2D
//
// Encoding reference (Advanced SIMD three-same, float64x2):
//   fadd: 0x4E60D400 | Rm<<16 | Rn<<5 | Rd
//   fmul: 0x6E60DC00 | Rm<<16 | Rn<<5 | Rd
//   fdiv: 0x6E60FC00 | Rm<<16 | Rn<<5 | Rd

// func computeKValsNEON(kVals, docLens []float64, k1Base, k1InvAvg float64)
// kVals[i] = k1Base + k1InvAvg * docLens[i]
TEXT ·computeKValsNEON(SB), NOSPLIT, $0-64
	MOVD kVals_base+0(FP), R0      // kVals data ptr
	MOVD kVals_len+8(FP), R1       // length
	MOVD docLens_base+24(FP), R2   // docLens data ptr
	FMOVD k1Base+48(FP), F0        // k1Base scalar -> V0.D[0]
	FMOVD k1InvAvg+56(FP), F1      // k1InvAvg scalar -> V1.D[0]
	VDUP V0.D[0], V0.D2            // V0 = [k1Base, k1Base]
	VDUP V1.D[0], V1.D2            // V1 = [k1InvAvg, k1InvAvg]

	// Process 2 elements per iteration
	LSR $1, R1, R3                  // R3 = len / 2
	CBZ R3, kvals_tail

kvals_loop:
	VLD1 (R2), [V2.D2]             // V2 = docLens[i:i+2]
	VMOV V0.B16, V3.B16            // V3 = k1Base (copy for accumulator)
	VFMLA V2.D2, V1.D2, V3.D2     // V3 += k1InvAvg * docLens = k1Base + k1InvAvg*docLens
	VST1 [V3.D2], (R0)             // store to kVals
	ADD $16, R0
	ADD $16, R2
	SUB $1, R3
	CBNZ R3, kvals_loop

kvals_tail:
	// Handle remaining element (0-1) with scalar
	TST $1, R1
	BEQ kvals_done
	FMOVD (R2), F2
	FMADDD F1, F2, F0, F3          // F3 = F0 + F1*F2 = k1Base + k1InvAvg*docLens[i]
	FMOVD F3, (R0)

kvals_done:
	RET

// func scoreBatchOkapiNEON(scores, tf, kVals []float64, idf float64)
// scores[i] += idf * (tf[i] / (tf[i] + kVals[i]))
TEXT ·scoreBatchOkapiNEON(SB), NOSPLIT, $0-80
	MOVD scores_base+0(FP), R0     // scores ptr
	MOVD scores_len+8(FP), R1      // length
	MOVD tf_base+24(FP), R2        // tf ptr
	MOVD kVals_base+48(FP), R3     // kVals ptr
	FMOVD idf+72(FP), F0           // idf scalar -> V0.D[0]
	VDUP V0.D[0], V0.D2            // V0 = [idf, idf]

	LSR $1, R1, R4                  // R4 = len / 2
	CBZ R4, okapi_tail

okapi_loop:
	VLD1 (R2), [V1.D2]             // V1 = tf[i:i+2]
	VLD1 (R3), [V2.D2]             // V2 = kVals[i:i+2]
	WORD $0x4E62D423                // fadd V3.2D, V1.2D, V2.2D  (V3 = tf + kVals)
	WORD $0x6E63FC24                // fdiv V4.2D, V1.2D, V3.2D  (V4 = tf / (tf + kVals))
	WORD $0x6E64DC04                // fmul V4.2D, V0.2D, V4.2D  (V4 = idf * result)
	VLD1 (R0), [V5.D2]             // V5 = scores[i:i+2]
	WORD $0x4E65D485                // fadd V5.2D, V4.2D, V5.2D  (V5 += contribution)
	VST1 [V5.D2], (R0)
	ADD $16, R0
	ADD $16, R2
	ADD $16, R3
	SUB $1, R4
	CBNZ R4, okapi_loop

okapi_tail:
	TST $1, R1
	BEQ okapi_done

	FMOVD (R2), F1                  // tf
	FMOVD (R3), F2                  // kVal
	FADDD F1, F2, F3                // tf + kVal
	FDIVD F3, F1, F4                // tf / (tf + kVal)
	FMULD F0, F4, F4                // idf * result
	FMOVD (R0), F5                  // scores[i]
	FADDD F4, F5, F5                // scores[i] += ...
	FMOVD F5, (R0)

okapi_done:
	RET

// func scoreBatchPlusNEON(scores, tf, kVals []float64, idf, delta float64)
// scores[i] += idf * (delta + tf[i] / (tf[i] + kVals[i]))
TEXT ·scoreBatchPlusNEON(SB), NOSPLIT, $0-88
	MOVD scores_base+0(FP), R0
	MOVD scores_len+8(FP), R1
	MOVD tf_base+24(FP), R2
	MOVD kVals_base+48(FP), R3
	FMOVD idf+72(FP), F0
	FMOVD delta+80(FP), F6
	VDUP V0.D[0], V0.D2            // V0 = idf
	VDUP V6.D[0], V6.D2            // V6 = delta

	LSR $1, R1, R4
	CBZ R4, plus_tail

plus_loop:
	VLD1 (R2), [V1.D2]             // tf
	VLD1 (R3), [V2.D2]             // kVals
	WORD $0x4E62D423                // fadd V3.2D, V1.2D, V2.2D  (tf + kVals)
	WORD $0x6E63FC24                // fdiv V4.2D, V1.2D, V3.2D  (tf / (tf + kVals))
	WORD $0x4E64D4C4                // fadd V4.2D, V6.2D, V4.2D  (delta + tf/(tf+kVals))
	WORD $0x6E64DC04                // fmul V4.2D, V0.2D, V4.2D  (idf * (...))
	VLD1 (R0), [V5.D2]
	WORD $0x4E65D485                // fadd V5.2D, V4.2D, V5.2D
	VST1 [V5.D2], (R0)
	ADD $16, R0
	ADD $16, R2
	ADD $16, R3
	SUB $1, R4
	CBNZ R4, plus_loop

plus_tail:
	TST $1, R1
	BEQ plus_done

	FMOVD (R2), F1
	FMOVD (R3), F2
	FADDD F1, F2, F3
	FDIVD F3, F1, F4
	FADDD F6, F4, F4
	FMULD F0, F4, F4
	FMOVD (R0), F5
	FADDD F4, F5, F5
	FMOVD F5, (R0)

plus_done:
	RET

// func scoreBatchTFNEON(scores, tf, kVals []float64, idf, delta float64)
// scores[i] += idf * (delta + (tf[i]*(1+kVals[i])) / (tf[i]+kVals[i]))
TEXT ·scoreBatchTFNEON(SB), NOSPLIT, $0-88
	MOVD scores_base+0(FP), R0
	MOVD scores_len+8(FP), R1
	MOVD tf_base+24(FP), R2
	MOVD kVals_base+48(FP), R3
	FMOVD idf+72(FP), F0
	FMOVD delta+80(FP), F6
	VDUP V0.D[0], V0.D2            // V0 = idf
	VDUP V6.D[0], V6.D2            // V6 = delta

	// V7 = [1.0, 1.0]
	FMOVD $1.0, F7
	VDUP V7.D[0], V7.D2

	LSR $1, R1, R4
	CBZ R4, tf_tail

tf_loop:
	VLD1 (R2), [V1.D2]             // V1 = tf
	VLD1 (R3), [V2.D2]             // V2 = kVals
	WORD $0x4E62D4E3                // fadd V3.2D, V7.2D, V2.2D  (V3 = 1 + kVals)
	WORD $0x6E63DC23                // fmul V3.2D, V1.2D, V3.2D  (V3 = tf * (1 + kVals))
	WORD $0x4E62D424                // fadd V4.2D, V1.2D, V2.2D  (V4 = tf + kVals)
	WORD $0x6E64FC63                // fdiv V3.2D, V3.2D, V4.2D  (V3 = tf*(1+kVals) / (tf+kVals))
	WORD $0x4E63D4C3                // fadd V3.2D, V6.2D, V3.2D  (V3 = delta + ...)
	WORD $0x6E63DC03                // fmul V3.2D, V0.2D, V3.2D  (V3 = idf * (...))
	VLD1 (R0), [V5.D2]
	WORD $0x4E65D465                // fadd V5.2D, V3.2D, V5.2D
	VST1 [V5.D2], (R0)
	ADD $16, R0
	ADD $16, R2
	ADD $16, R3
	SUB $1, R4
	CBNZ R4, tf_loop

tf_tail:
	TST $1, R1
	BEQ tf_done

	FMOVD (R2), F1                  // tf
	FMOVD (R3), F2                  // kVal
	FADDD F7, F2, F3                // 1 + kVal
	FMULD F1, F3, F3                // tf * (1 + kVal)
	FADDD F1, F2, F4                // tf + kVal
	FDIVD F4, F3, F3                // tf*(1+kVal) / (tf+kVal)
	FADDD F6, F3, F3                // delta + ...
	FMULD F0, F3, F3                // idf * (...)
	FMOVD (R0), F5
	FADDD F3, F5, F5
	FMOVD F5, (R0)

tf_done:
	RET
