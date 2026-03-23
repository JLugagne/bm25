// Package safetensors implements the SafeTensors binary format for storing
// typed tensors with zero-copy-friendly layout.
//
// File layout:
//
//	[8 bytes] header_size (little-endian uint64)
//	[header_size bytes] UTF-8 JSON header
//	[remaining bytes] raw tensor data (contiguous)
//
// The JSON header maps tensor names to {dtype, shape, data_offsets} and may
// contain a "__metadata__" key with string→string pairs.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"unsafe"
)

// DType identifies the element type of a tensor.
type DType string

const (
	DTypeBool DType = "BOOL"
	DTypeU8   DType = "U8"
	DTypeI8   DType = "I8"
	DTypeU16  DType = "U16"
	DTypeI16  DType = "I16"
	DTypeF16  DType = "F16"
	DTypeBF16 DType = "BF16"
	DTypeU32  DType = "U32"
	DTypeI32  DType = "I32"
	DTypeF32  DType = "F32"
	DTypeU64  DType = "U64"
	DTypeI64  DType = "I64"
	DTypeF64  DType = "F64"
)

// ByteSize returns the number of bytes per element for this dtype.
func (d DType) ByteSize() (int, error) {
	switch d {
	case DTypeBool, DTypeU8, DTypeI8:
		return 1, nil
	case DTypeU16, DTypeI16, DTypeF16, DTypeBF16:
		return 2, nil
	case DTypeU32, DTypeI32, DTypeF32:
		return 4, nil
	case DTypeU64, DTypeI64, DTypeF64:
		return 8, nil
	default:
		return 0, fmt.Errorf("safetensors: unknown dtype %q", d)
	}
}

const maxHeaderSize = 100 * 1024 * 1024 // 100 MB

// TensorInfo describes a tensor in the header.
type TensorInfo struct {
	DType       DType    `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// NumElements returns the total number of elements described by the shape.
func (ti *TensorInfo) NumElements() uint64 {
	if len(ti.Shape) == 0 {
		return 0
	}
	n := uint64(1)
	for _, s := range ti.Shape {
		n *= s
	}
	return n
}

// Tensor holds a named tensor's metadata and raw data.
type Tensor struct {
	Name  string
	DType DType
	Shape []uint64
	Data  []byte
}

// File represents a parsed safetensors file.
type File struct {
	Tensors  []Tensor
	Metadata map[string]string
}

// headerEntry is used for JSON round-tripping; it handles the union of
// TensorInfo and __metadata__.
type headerEntry struct {
	DType       DType    `json:"dtype,omitempty"`
	Shape       []uint64 `json:"shape,omitempty"`
	DataOffsets [2]int64 `json:"data_offsets,omitempty"`
}

// Serialize writes a File to w in safetensors format.
func Serialize(w io.Writer, f *File) error {
	if f == nil {
		return errors.New("safetensors: nil file")
	}

	// Build header and compute data offsets.
	header := make(map[string]json.RawMessage, len(f.Tensors)+1)

	if len(f.Metadata) > 0 {
		b, err := json.Marshal(f.Metadata)
		if err != nil {
			return fmt.Errorf("safetensors: marshal metadata: %w", err)
		}
		header["__metadata__"] = b
	}

	var offset int64
	for i := range f.Tensors {
		t := &f.Tensors[i]
		begin := offset
		end := begin + int64(len(t.Data))
		entry := headerEntry{
			DType:       t.DType,
			Shape:       t.Shape,
			DataOffsets: [2]int64{begin, end},
		}
		b, err := json.Marshal(entry)
		if err != nil {
			return fmt.Errorf("safetensors: marshal tensor %q: %w", t.Name, err)
		}
		header[t.Name] = b
		offset = end
	}

	headerBytes, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("safetensors: marshal header: %w", err)
	}

	// Write header size (8 bytes LE).
	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(headerBytes)))
	if _, err := w.Write(sizeBuf[:]); err != nil {
		return err
	}

	// Write header JSON.
	if _, err := w.Write(headerBytes); err != nil {
		return err
	}

	// Write tensor data in order.
	for i := range f.Tensors {
		if _, err := w.Write(f.Tensors[i].Data); err != nil {
			return err
		}
	}

	return nil
}

// Deserialize reads a safetensors file from r.
func Deserialize(r io.Reader) (*File, error) {
	// Read header size.
	var sizeBuf [8]byte
	if _, err := io.ReadFull(r, sizeBuf[:]); err != nil {
		return nil, fmt.Errorf("safetensors: read header size: %w", err)
	}
	headerSize := binary.LittleEndian.Uint64(sizeBuf[:])
	if headerSize > maxHeaderSize {
		return nil, fmt.Errorf("safetensors: header size %d exceeds maximum %d", headerSize, maxHeaderSize)
	}

	// Read header JSON.
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	f := &File{}

	// Extract metadata.
	if metaRaw, ok := rawHeader["__metadata__"]; ok {
		f.Metadata = make(map[string]string)
		if err := json.Unmarshal(metaRaw, &f.Metadata); err != nil {
			return nil, fmt.Errorf("safetensors: parse metadata: %w", err)
		}
		delete(rawHeader, "__metadata__")
	}

	// Parse tensor entries and sort by data offset for sequential reading.
	type namedEntry struct {
		name string
		info TensorInfo
	}
	entries := make([]namedEntry, 0, len(rawHeader))
	for name, raw := range rawHeader {
		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		entries = append(entries, namedEntry{name, info})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].info.DataOffsets[0] < entries[j].info.DataOffsets[0]
	})

	// Read tensor data sequentially.
	f.Tensors = make([]Tensor, len(entries))
	for i, e := range entries {
		size := e.info.DataOffsets[1] - e.info.DataOffsets[0]
		if size < 0 {
			return nil, fmt.Errorf("safetensors: tensor %q has negative size", e.name)
		}
		data := make([]byte, size)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, fmt.Errorf("safetensors: read tensor %q data: %w", e.name, err)
		}
		f.Tensors[i] = Tensor{
			Name:  e.name,
			DType: e.info.DType,
			Shape: e.info.Shape,
			Data:  data,
		}
	}

	return f, nil
}

// --- Typed helpers for float64 tensors ---

// Float64ToBytes converts a []float64 to raw little-endian bytes via
// zero-copy reinterpretation (safe on little-endian platforms, which
// covers all Go targets in practice).
func Float64ToBytes(v []float64) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*8)
}

// BytesToFloat64 converts raw little-endian bytes back to []float64.
// It makes a copy to avoid aliasing the original byte slice.
func BytesToFloat64(b []byte) ([]float64, error) {
	if len(b)%8 != 0 {
		return nil, fmt.Errorf("safetensors: byte slice length %d not divisible by 8", len(b))
	}
	n := len(b) / 8
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[i*8 : (i+1)*8]))
	}
	return out, nil
}

// Float64Tensor creates a Tensor from a named float64 slice.
func Float64Tensor(name string, v []float64) Tensor {
	// Copy bytes so the tensor owns its data.
	raw := Float64ToBytes(v)
	data := make([]byte, len(raw))
	copy(data, raw)
	return Tensor{
		Name:  name,
		DType: DTypeF64,
		Shape: []uint64{uint64(len(v))},
		Data:  data,
	}
}

// Float64ScalarTensor creates a 0-d tensor from a single float64.
func Float64ScalarTensor(name string, v float64) Tensor {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
	return Tensor{
		Name:  name,
		DType: DTypeF64,
		Shape: []uint64{},
		Data:  buf[:],
	}
}
