#include "textflag.h"

// func computeKValsAVX2(kVals, docLens []float64, k1Base, k1InvAvg float64)
// kVals[i] = k1Base + k1InvAvg * docLens[i]
TEXT ·computeKValsAVX2(SB), NOSPLIT, $0-64
	MOVQ kVals_base+0(FP), DI      // kVals data ptr
	MOVQ kVals_len+8(FP), SI       // length
	MOVQ docLens_base+24(FP), DX   // docLens data ptr
	VBROADCASTSD k1Base+48(FP), Y0  // Y0 = [k1Base, k1Base, k1Base, k1Base]
	VBROADCASTSD k1InvAvg+56(FP), Y1 // Y1 = [k1InvAvg, ...]

	// Process 4 elements per iteration
	MOVQ SI, CX
	SHRQ $2, CX    // CX = len / 4
	JZ   kvals_tail

kvals_loop:
	VMOVUPD (DX), Y2           // Y2 = docLens[i:i+4]
	VFMADD213PD Y0, Y1, Y2    // Y2 = Y1*Y2 + Y0 = k1InvAvg*docLens + k1Base
	VMOVUPD Y2, (DI)           // store to kVals
	ADDQ $32, DI
	ADDQ $32, DX
	DECQ CX
	JNZ  kvals_loop

kvals_tail:
	// Handle remaining elements (0-3) with scalar
	ANDQ $3, SI
	JZ   kvals_done
	VMOVSD k1Base+48(FP), X3
	VMOVSD k1InvAvg+56(FP), X4

kvals_tail_loop:
	VMOVSD (DX), X5
	VFMADD213SD X3, X4, X5    // X5 = k1InvAvg * docLens[i] + k1Base
	VMOVSD X5, (DI)
	ADDQ $8, DI
	ADDQ $8, DX
	DECQ SI
	JNZ  kvals_tail_loop

kvals_done:
	VZEROUPPER
	RET

// func scoreBatchOkapiAVX2(scores, tf, kVals []float64, idf float64)
// scores[i] += idf * (tf[i] / (tf[i] + kVals[i]))
TEXT ·scoreBatchOkapiAVX2(SB), NOSPLIT, $0-80
	MOVQ scores_base+0(FP), DI     // scores ptr
	MOVQ scores_len+8(FP), SI      // length
	MOVQ tf_base+24(FP), DX        // tf ptr
	MOVQ kVals_base+48(FP), CX     // kVals ptr
	VBROADCASTSD idf+72(FP), Y0    // Y0 = [idf, idf, idf, idf]

	MOVQ SI, R8
	SHRQ $2, R8     // R8 = len / 4
	JZ   okapi_tail

okapi_loop:
	VMOVUPD (DX), Y1          // Y1 = tf[i:i+4]
	VMOVUPD (CX), Y2          // Y2 = kVals[i:i+4]
	VADDPD  Y1, Y2, Y3        // Y3 = tf + kVals
	VDIVPD  Y3, Y1, Y4        // Y4 = tf / (tf + kVals)
	VMULPD  Y0, Y4, Y4        // Y4 = idf * (tf / (tf + kVals))
	VMOVUPD (DI), Y5          // Y5 = scores[i:i+4]
	VADDPD  Y4, Y5, Y5        // Y5 += contribution
	VMOVUPD Y5, (DI)          // store
	ADDQ $32, DI
	ADDQ $32, DX
	ADDQ $32, CX
	DECQ R8
	JNZ  okapi_loop

okapi_tail:
	ANDQ $3, SI
	JZ   okapi_done
	VMOVSD idf+72(FP), X0

okapi_tail_loop:
	VMOVSD (DX), X1            // tf
	VMOVSD (CX), X2            // kVal
	VADDSD X1, X2, X3          // tf + kVal
	VDIVSD X3, X1, X4          // tf / (tf + kVal)
	VMULSD X0, X4, X4          // idf * result
	VMOVSD (DI), X5            // scores[i]
	VADDSD X4, X5, X5          // scores[i] += ...
	VMOVSD X5, (DI)
	ADDQ $8, DI
	ADDQ $8, DX
	ADDQ $8, CX
	DECQ SI
	JNZ  okapi_tail_loop

okapi_done:
	VZEROUPPER
	RET

// func scoreBatchPlusAVX2(scores, tf, kVals []float64, idf, delta float64)
// scores[i] += idf * (delta + tf[i] / (tf[i] + kVals[i]))
TEXT ·scoreBatchPlusAVX2(SB), NOSPLIT, $0-88
	MOVQ scores_base+0(FP), DI
	MOVQ scores_len+8(FP), SI
	MOVQ tf_base+24(FP), DX
	MOVQ kVals_base+48(FP), CX
	VBROADCASTSD idf+72(FP), Y0     // Y0 = idf
	VBROADCASTSD delta+80(FP), Y6   // Y6 = delta

	MOVQ SI, R8
	SHRQ $2, R8
	JZ   plus_tail

plus_loop:
	VMOVUPD (DX), Y1          // tf
	VMOVUPD (CX), Y2          // kVals
	VADDPD  Y1, Y2, Y3        // tf + kVals
	VDIVPD  Y3, Y1, Y4        // tf / (tf + kVals)
	VADDPD  Y6, Y4, Y4        // delta + tf/(tf+kVals)
	VMULPD  Y0, Y4, Y4        // idf * (...)
	VMOVUPD (DI), Y5
	VADDPD  Y4, Y5, Y5
	VMOVUPD Y5, (DI)
	ADDQ $32, DI
	ADDQ $32, DX
	ADDQ $32, CX
	DECQ R8
	JNZ  plus_loop

plus_tail:
	ANDQ $3, SI
	JZ   plus_done
	VMOVSD idf+72(FP), X0
	VMOVSD delta+80(FP), X6

plus_tail_loop:
	VMOVSD (DX), X1
	VMOVSD (CX), X2
	VADDSD X1, X2, X3
	VDIVSD X3, X1, X4
	VADDSD X6, X4, X4
	VMULSD X0, X4, X4
	VMOVSD (DI), X5
	VADDSD X4, X5, X5
	VMOVSD X5, (DI)
	ADDQ $8, DI
	ADDQ $8, DX
	ADDQ $8, CX
	DECQ SI
	JNZ  plus_tail_loop

plus_done:
	VZEROUPPER
	RET

// func scoreBatchTFAVX2(scores, tf, kVals []float64, idf, delta float64)
// scores[i] += idf * (delta + (tf[i]*(1+kVals[i])) / (tf[i]+kVals[i]))
TEXT ·scoreBatchTFAVX2(SB), NOSPLIT, $0-88
	MOVQ scores_base+0(FP), DI
	MOVQ scores_len+8(FP), SI
	MOVQ tf_base+24(FP), DX
	MOVQ kVals_base+48(FP), CX
	VBROADCASTSD idf+72(FP), Y0     // Y0 = idf
	VBROADCASTSD delta+80(FP), Y6   // Y6 = delta

	// Y7 = [1.0, 1.0, 1.0, 1.0]
	MOVQ $0x3FF0000000000000, R9    // 1.0 in IEEE 754
	MOVQ R9, X7
	VBROADCASTSD X7, Y7

	MOVQ SI, R8
	SHRQ $2, R8
	JZ   tf_tail

tf_loop:
	VMOVUPD (DX), Y1          // Y1 = tf
	VMOVUPD (CX), Y2          // Y2 = kVals
	VADDPD  Y7, Y2, Y3        // Y3 = 1 + kVals
	VMULPD  Y1, Y3, Y3        // Y3 = tf * (1 + kVals)
	VADDPD  Y1, Y2, Y4        // Y4 = tf + kVals
	VDIVPD  Y4, Y3, Y3        // Y3 = tf*(1+kVals) / (tf+kVals)
	VADDPD  Y6, Y3, Y3        // Y3 = delta + ...
	VMULPD  Y0, Y3, Y3        // Y3 = idf * (...)
	VMOVUPD (DI), Y5
	VADDPD  Y3, Y5, Y5
	VMOVUPD Y5, (DI)
	ADDQ $32, DI
	ADDQ $32, DX
	ADDQ $32, CX
	DECQ R8
	JNZ  tf_loop

tf_tail:
	ANDQ $3, SI
	JZ   tf_done
	VMOVSD idf+72(FP), X0
	VMOVSD delta+80(FP), X6
	MOVQ $0x3FF0000000000000, R9
	MOVQ R9, X7

tf_tail_loop:
	VMOVSD (DX), X1            // tf
	VMOVSD (CX), X2            // kVal
	VADDSD X7, X2, X3          // 1 + kVal
	VMULSD X1, X3, X3          // tf * (1 + kVal)
	VADDSD X1, X2, X4          // tf + kVal
	VDIVSD X4, X3, X3          // tf*(1+kVal) / (tf+kVal)
	VADDSD X6, X3, X3          // delta + ...
	VMULSD X0, X3, X3          // idf * (...)
	VMOVSD (DI), X5
	VADDSD X3, X5, X5
	VMOVSD X5, (DI)
	ADDQ $8, DI
	ADDQ $8, DX
	ADDQ $8, CX
	DECQ SI
	JNZ  tf_tail_loop

tf_done:
	VZEROUPPER
	RET
