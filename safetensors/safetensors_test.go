package safetensors

import (
	"bytes"
	"math"
	"testing"
)

func TestRoundTripEmpty(t *testing.T) {
	f := &File{Tensors: nil}
	var buf bytes.Buffer
	if err := Serialize(&buf, f); err != nil {
		t.Fatal(err)
	}
	got, err := Deserialize(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Tensors) != 0 {
		t.Fatalf("expected 0 tensors, got %d", len(got.Tensors))
	}
}

func TestRoundTripFloat64(t *testing.T) {
	want := []float64{1.0, 2.5, -3.14, 0, math.Inf(1), math.SmallestNonzeroFloat64}
	f := &File{
		Tensors: []Tensor{Float64Tensor("weights", want)},
		Metadata: map[string]string{
			"format": "bm25",
		},
	}

	var buf bytes.Buffer
	if err := Serialize(&buf, f); err != nil {
		t.Fatal(err)
	}

	got, err := Deserialize(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if len(got.Tensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(got.Tensors))
	}
	if got.Tensors[0].Name != "weights" {
		t.Fatalf("expected name %q, got %q", "weights", got.Tensors[0].Name)
	}
	if got.Tensors[0].DType != DTypeF64 {
		t.Fatalf("expected dtype %q, got %q", DTypeF64, got.Tensors[0].DType)
	}

	vals, err := BytesToFloat64(got.Tensors[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != len(want) {
		t.Fatalf("expected %d elements, got %d", len(want), len(vals))
	}
	for i, v := range vals {
		if v != want[i] && !(math.IsNaN(v) && math.IsNaN(want[i])) {
			t.Errorf("[%d] expected %v, got %v", i, want[i], v)
		}
	}

	if got.Metadata["format"] != "bm25" {
		t.Errorf("expected metadata format=bm25, got %q", got.Metadata["format"])
	}
}

func TestRoundTripScalar(t *testing.T) {
	want := 42.5
	f := &File{
		Tensors: []Tensor{Float64ScalarTensor("avg", want)},
	}
	var buf bytes.Buffer
	if err := Serialize(&buf, f); err != nil {
		t.Fatal(err)
	}
	got, err := Deserialize(&buf)
	if err != nil {
		t.Fatal(err)
	}
	vals, err := BytesToFloat64(got.Tensors[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != 1 || vals[0] != want {
		t.Fatalf("expected [%v], got %v", want, vals)
	}
}

func TestRoundTripMultipleTensors(t *testing.T) {
	f := &File{
		Tensors: []Tensor{
			Float64Tensor("a", []float64{1, 2, 3}),
			Float64Tensor("b", []float64{4, 5}),
			Float64ScalarTensor("c", 99),
		},
	}
	var buf bytes.Buffer
	if err := Serialize(&buf, f); err != nil {
		t.Fatal(err)
	}
	got, err := Deserialize(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Tensors) != 3 {
		t.Fatalf("expected 3 tensors, got %d", len(got.Tensors))
	}

	// Tensors come back sorted by data offset, which matches insertion order.
	check := func(idx int, name string, want []float64) {
		t.Helper()
		tensor := got.Tensors[idx]
		if tensor.Name != name {
			t.Errorf("[%d] expected name %q, got %q", idx, name, tensor.Name)
		}
		vals, err := BytesToFloat64(tensor.Data)
		if err != nil {
			t.Fatal(err)
		}
		if len(vals) != len(want) {
			t.Fatalf("[%d] expected %d elements, got %d", idx, len(want), len(vals))
		}
		for i, v := range vals {
			if v != want[i] {
				t.Errorf("[%d][%d] expected %v, got %v", idx, i, want[i], v)
			}
		}
	}
	check(0, "a", []float64{1, 2, 3})
	check(1, "b", []float64{4, 5})
	check(2, "c", []float64{99})
}

func TestBytesToFloat64BadLength(t *testing.T) {
	_, err := BytesToFloat64([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for non-8-divisible length")
	}
}

func TestDTypeByteSize(t *testing.T) {
	cases := []struct {
		d    DType
		want int
	}{
		{DTypeBool, 1}, {DTypeU8, 1}, {DTypeI8, 1},
		{DTypeU16, 2}, {DTypeI16, 2}, {DTypeF16, 2}, {DTypeBF16, 2},
		{DTypeU32, 4}, {DTypeI32, 4}, {DTypeF32, 4},
		{DTypeU64, 8}, {DTypeI64, 8}, {DTypeF64, 8},
	}
	for _, tc := range cases {
		got, err := tc.d.ByteSize()
		if err != nil {
			t.Errorf("dtype %q: unexpected error: %v", tc.d, err)
		}
		if got != tc.want {
			t.Errorf("dtype %q: expected %d, got %d", tc.d, tc.want, got)
		}
	}
	_, err := DType("INVALID").ByteSize()
	if err == nil {
		t.Error("expected error for unknown dtype")
	}
}
