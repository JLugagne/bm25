// Package safetensors implements reading and writing the SafeTensors binary
// format for storing typed, named tensors with zero-copy-friendly layout.
//
// The file layout is:
//
//	[8 bytes]            header_size (little-endian uint64)
//	[header_size bytes]  UTF-8 JSON header
//	[remaining bytes]    raw tensor data (contiguous)
//
// The JSON header maps tensor names to {dtype, shape, data_offsets} and may
// contain a "__metadata__" key with arbitrary string key-value pairs.
//
// This package is used internally by the bm25 package for index serialization,
// but can also be used independently for any SafeTensors-compatible workflow.
//
// # Writing
//
//	f := &safetensors.File{
//	    Tensors: []safetensors.Tensor{
//	        safetensors.Float64Tensor("weights", []float64{1.0, 2.0, 3.0}),
//	    },
//	    Metadata: map[string]string{"version": "1"},
//	}
//	err := safetensors.Serialize(w, f)
//
// # Reading
//
//	f, err := safetensors.Deserialize(r)
//	vals, err := safetensors.BytesToFloat64(f.Tensors[0].Data)
package safetensors
