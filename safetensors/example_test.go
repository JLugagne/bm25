package safetensors_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/JLugagne/bm25/safetensors"
)

func ExampleSerialize() {
	f := &safetensors.File{
		Tensors: []safetensors.Tensor{
			safetensors.Float64Tensor("weights", []float64{1.0, 2.0, 3.0}),
			safetensors.Float64ScalarTensor("bias", 0.5),
		},
		Metadata: map[string]string{
			"version": "1",
		},
	}

	var buf bytes.Buffer
	if err := safetensors.Serialize(&buf, f); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Wrote %d bytes\n", buf.Len())
	// Output:
	// Wrote 178 bytes
}

func ExampleDeserialize() {
	// First, write a file.
	original := &safetensors.File{
		Tensors: []safetensors.Tensor{
			safetensors.Float64Tensor("data", []float64{1.5, 2.5, 3.5}),
		},
	}

	var buf bytes.Buffer
	if err := safetensors.Serialize(&buf, original); err != nil {
		log.Fatal(err)
	}

	// Now read it back.
	f, err := safetensors.Deserialize(&buf)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Tensors: %d\n", len(f.Tensors))
	fmt.Printf("Name: %s\n", f.Tensors[0].Name)
	fmt.Printf("DType: %s\n", f.Tensors[0].DType)

	vals, err := safetensors.BytesToFloat64(f.Tensors[0].Data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Values: %v\n", vals)
	// Output:
	// Tensors: 1
	// Name: data
	// DType: F64
	// Values: [1.5 2.5 3.5]
}

func ExampleFloat64Tensor() {
	t := safetensors.Float64Tensor("weights", []float64{1.0, 2.0, 3.0})
	fmt.Printf("Name: %s, DType: %s, Shape: %v\n", t.Name, t.DType, t.Shape)
	// Output:
	// Name: weights, DType: F64, Shape: [3]
}

func ExampleFloat64ScalarTensor() {
	t := safetensors.Float64ScalarTensor("learning_rate", 0.001)
	fmt.Printf("Name: %s, DType: %s, Shape: %v\n", t.Name, t.DType, t.Shape)
	// Output:
	// Name: learning_rate, DType: F64, Shape: []
}

func ExampleBytesToFloat64() {
	// Create a tensor and extract its values.
	t := safetensors.Float64Tensor("data", []float64{42.0, 3.14})
	vals, err := safetensors.BytesToFloat64(t.Data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%.2f, %.2f\n", vals[0], vals[1])
	// Output:
	// 42.00, 3.14
}

func ExampleDType_ByteSize() {
	size, err := safetensors.DTypeF64.ByteSize()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("F64 element size: %d bytes\n", size)
	// Output:
	// F64 element size: 8 bytes
}
