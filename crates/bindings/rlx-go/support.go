// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Package rlx is Go for rlx over its C ABI (crates/bindings/rlx-capi): every C
// function is a Go function or method (generated.go, from rlx-bind's tables).
//
// cgo finds the library in the workspace's target/debug by default; point
// CGO_LDFLAGS (and the runtime library path) at another build to use it.
package rlx

// #cgo CFLAGS: -I${SRCDIR}/../rlx-capi/include
// #cgo LDFLAGS: -L${SRCDIR}/../../../target/debug -lrlx_capi -Wl,-rpath,${SRCDIR}/../../../target/debug
// #include <stdlib.h>
// #include "rlx.h"
import "C"

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"
	"unsafe"
)

// Node is a node of a graph — the id every builder returns.
type Node = uint32

// None is "no node", for an optional node argument.
const None Node = math.MaxUint32

// ErrorKind is the RLX_ERR_* status of a failed call.
type ErrorKind int

const (
	ErrType     ErrorKind = -1 // a wrong argument: unknown name, bad shape, wrong arity
	ErrRange    ErrorKind = -2 // an index or value out of range
	ErrRuntime  ErrorKind = -3 // well-formed but not carried out: I/O, a backend not built
	ErrPanic    ErrorKind = -4 // a Rust assertion inside rlx, caught at the boundary
	ErrNotFound ErrorKind = -5 // a file that does not exist
)

// Error is a failed rlx call.
type Error struct {
	Kind    ErrorKind
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("rlx (%d): %s", e.Kind, e.Message) }

func check(rc C.int32_t) error {
	if rc == 0 {
		return nil
	}
	return &Error{Kind: ErrorKind(rc), Message: C.GoString(C.rlx_last_error())}
}

// Shape is a node's static dims (-1 for a dynamic axis) and dtype.
type Shape struct {
	Dims  []int64 `json:"dims"`
	Dtype string  `json:"dtype"`
}

// TypedTensor is a tensor of any dtype as raw bytes.
type TypedTensor struct {
	Bytes []byte
	Dtype string
}

func ptrTo[T any](v T) *T { return &v }

// Ptr returns a pointer to v, for the optional fields of an …Opts struct:
// rlx.Ptr("metal"), rlx.Ptr(int32(-1)).
func Ptr[T any](v T) *T { return &v }

// ── arguments ───────────────────────────────────────────────

func cStringOrNil(s *string) *C.char {
	if s == nil {
		return nil
	}
	return C.CString(*s)
}

func freeC(p *C.char) {
	if p != nil {
		C.free(unsafe.Pointer(p))
	}
}

func cJSON(v any) (*C.char, error) {
	if v == nil {
		return nil, nil
	}
	switch m := v.(type) {
	case map[string]any:
		if m == nil {
			return nil, nil
		}
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, &Error{Kind: ErrType, Message: "options are not JSON-encodable: " + err.Error()}
	}
	return C.CString(string(b)), nil
}

func nodeOrNone(n *Node) C.uint32_t {
	if n == nil {
		return C.uint32_t(None)
	}
	return C.uint32_t(*n)
}

func optU64(v *uint64) *C.uint64_t {
	if v == nil {
		return nil
	}
	x := C.uint64_t(*v)
	return &x
}

func optSize(v *int) *C.size_t {
	if v == nil {
		return nil
	}
	x := C.size_t(*v)
	return &x
}

func optF32(v *float32) *C.float {
	if v == nil {
		return nil
	}
	x := C.float(*v)
	return &x
}

func cF32s(v []float32) (*C.float, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	return (*C.float)(unsafe.Pointer(&v[0])), C.size_t(len(v))
}

func cBytes(v []byte) (*C.uint8_t, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	return (*C.uint8_t)(unsafe.Pointer(&v[0])), C.size_t(len(v))
}

func cU32s[T ~uint32](v []T) (*C.uint32_t, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	return (*C.uint32_t)(unsafe.Pointer(&v[0])), C.size_t(len(v))
}

func cSizes(v []int) (*C.size_t, C.size_t) {
	if v == nil {
		return nil, 0
	}
	out := make([]C.size_t, len(v)+1) // never empty, so &out[0] is valid
	for i, x := range v {
		out[i] = C.size_t(x)
	}
	return &out[0], C.size_t(len(v))
}

func cI64s(v []int64) (*C.int64_t, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	return (*C.int64_t)(unsafe.Pointer(&v[0])), C.size_t(len(v))
}

// cRows flattens rows of width w; with splat a one-value row stands for w copies.
func cRows(rows [][]int, w int, splat bool) (*C.size_t, C.size_t, error) {
	flat := []int{}
	for i, r := range rows {
		switch {
		case len(r) == w:
			flat = append(flat, r...)
		case splat && len(r) == 1:
			for j := 0; j < w; j++ {
				flat = append(flat, r[0])
			}
		default:
			return nil, 0, &Error{Kind: ErrType, Message: fmt.Sprintf("row %d needs %d values, got %d", i, w, len(r))}
		}
	}
	p, _ := cSizes(flat)
	return p, C.size_t(len(rows)), nil
}

func cFRows(rows [][]float32, w int) (*C.float, C.size_t, error) {
	flat := make([]float32, 0, len(rows)*w+1)
	for i, r := range rows {
		if len(r) != w {
			return nil, 0, &Error{Kind: ErrType, Message: fmt.Sprintf("row %d needs %d values, got %d", i, w, len(r))}
		}
		flat = append(flat, r...)
	}
	flat = append(flat, 0)
	return (*C.float)(unsafe.Pointer(&flat[0])), C.size_t(len(rows)), nil
}

func graphOrNil(g *Graph) *C.RlxGraph {
	if g == nil {
		return nil
	}
	return g.h
}

// cStrings is a list of C strings in C memory (cgo forbids passing Go memory
// that holds pointers).
type cStrings struct {
	ptr  **C.char
	n    C.size_t
	strs []*C.char
}

func newCStrings(list []string) *cStrings {
	if list == nil {
		return &cStrings{}
	}
	cs := &cStrings{n: C.size_t(len(list))}
	cs.ptr = (**C.char)(C.malloc(C.size_t(len(list)+1) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	arr := unsafe.Slice(cs.ptr, len(list)+1)
	for i, s := range list {
		p := C.CString(s)
		cs.strs = append(cs.strs, p)
		arr[i] = p
	}
	return cs
}

func (cs *cStrings) free() {
	for _, p := range cs.strs {
		C.free(unsafe.Pointer(p))
	}
	if cs.ptr != nil {
		C.free(unsafe.Pointer(cs.ptr))
	}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// cInputs is named tensors as the parallel C arrays a run takes; the data stay
// in Go memory, pinned for the call.
type cInputs struct {
	names  **C.char
	data   **C.float
	lens   *C.size_t
	dtypes **C.char
	n      C.size_t
	keep   []*cStrings
	extra  []unsafe.Pointer
	pin    runtime.Pinner
}

func cArray(n int) unsafe.Pointer {
	return C.malloc(C.size_t(n+1) * C.size_t(unsafe.Sizeof(uintptr(0))))
}

func newCInputs(m map[string][]float32) *cInputs {
	keys := sortedKeys(m)
	ci := &cInputs{n: C.size_t(len(keys))}
	names := newCStrings(keys)
	ci.keep = append(ci.keep, names)
	ci.names = names.ptr
	data := cArray(len(keys))
	lens := cArray(len(keys))
	ci.extra = append(ci.extra, data, lens)
	d := unsafe.Slice((**C.float)(data), len(keys)+1)
	l := unsafe.Slice((*C.size_t)(lens), len(keys)+1)
	for i, k := range keys {
		v := m[k]
		if len(v) > 0 {
			ci.pin.Pin(&v[0])
			d[i] = (*C.float)(unsafe.Pointer(&v[0]))
		} else {
			d[i] = nil
		}
		l[i] = C.size_t(len(v))
	}
	ci.data = (**C.float)(data)
	ci.lens = (*C.size_t)(lens)
	return ci
}

// cTypedInputs is the typed-run form of cInputs.
type cTypedInputs struct {
	cInputs
	data8 **C.uint8_t
}

func newCTypedInputs(m map[string]TypedTensor) *cTypedInputs {
	keys := sortedKeys(m)
	ci := &cTypedInputs{}
	ci.n = C.size_t(len(keys))
	names := newCStrings(keys)
	dtypes := make([]string, len(keys))
	for i, k := range keys {
		dtypes[i] = m[k].Dtype
	}
	dt := newCStrings(dtypes)
	ci.keep = append(ci.keep, names, dt)
	ci.names, ci.dtypes = names.ptr, dt.ptr
	data := cArray(len(keys))
	lens := cArray(len(keys))
	ci.extra = append(ci.extra, data, lens)
	d := unsafe.Slice((**C.uint8_t)(data), len(keys)+1)
	l := unsafe.Slice((*C.size_t)(lens), len(keys)+1)
	for i, k := range keys {
		b := m[k].Bytes
		if len(b) > 0 {
			ci.pin.Pin(&b[0])
			d[i] = (*C.uint8_t)(unsafe.Pointer(&b[0]))
		}
		l[i] = C.size_t(len(b))
	}
	ci.data8 = (**C.uint8_t)(data)
	ci.lens = (*C.size_t)(lens)
	return ci
}

func (ci *cInputs) free() {
	ci.pin.Unpin()
	for _, k := range ci.keep {
		k.free()
	}
	for _, p := range ci.extra {
		C.free(p)
	}
}

// ── results ─────────────────────────────────────────────────

func takeString(p *C.char) string {
	if p == nil {
		return ""
	}
	defer C.rlx_string_free(p)
	return C.GoString(p)
}

func takeStringOrNil(p *C.char) *string {
	if p == nil {
		return nil
	}
	s := takeString(p)
	return &s
}

func takeJSON[T any](p *C.char) (T, error) {
	var v T
	text := takeString(p)
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		return v, &Error{Kind: ErrRuntime, Message: "decoding rlx JSON: " + err.Error()}
	}
	return v, nil
}

func takeF32s(v C.RlxF32s) []float32 {
	defer C.rlx_f32s_free(v)
	if v.ptr == nil {
		return []float32{}
	}
	return append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(v.ptr)), int(v.len))...)
}

func takeBytes(v C.RlxBytes) []byte {
	defer C.rlx_bytes_free(v)
	if v.ptr == nil {
		return []byte{}
	}
	return C.GoBytes(unsafe.Pointer(v.ptr), C.int(v.len))
}

func takeU32s(v C.RlxU32s) []uint32 {
	defer C.rlx_u32s_free(v)
	if v.ptr == nil {
		return []uint32{}
	}
	return append([]uint32(nil), unsafe.Slice((*uint32)(unsafe.Pointer(v.ptr)), int(v.len))...)
}

func takeNodes(v C.RlxU32s) []Node { return takeU32s(v) }

func takeOutputs(o *C.RlxOutputs) ([][]float32, error) {
	defer C.rlx_outputs_free(o)
	n := int(C.rlx_outputs_len(o))
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		var p *C.float
		var l C.size_t
		if err := check(C.rlx_outputs_get(o, C.size_t(i), &p, &l)); err != nil {
			return nil, err
		}
		out[i] = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(p)), int(l))...)
	}
	return out, nil
}

func takeTypedOutputs(o *C.RlxTypedOutputs) ([]TypedTensor, error) {
	defer C.rlx_typed_outputs_free(o)
	n := int(C.rlx_typed_outputs_len(o))
	out := make([]TypedTensor, n)
	for i := 0; i < n; i++ {
		var p *C.uint8_t
		var l C.size_t
		var dt *C.char
		if err := check(C.rlx_typed_outputs_get(o, C.size_t(i), &p, &l, &dt)); err != nil {
			return nil, err
		}
		out[i] = TypedTensor{Bytes: C.GoBytes(unsafe.Pointer(p), C.int(l)), Dtype: C.GoString(dt)}
	}
	return out, nil
}

func takeTensors(m *C.RlxTensorMap) (map[string][]float32, error) {
	defer C.rlx_tensor_map_free(m)
	n := int(C.rlx_tensor_map_len(m))
	out := make(map[string][]float32, n)
	for i := 0; i < n; i++ {
		var name *C.char
		var p *C.float
		var l C.size_t
		if err := check(C.rlx_tensor_map_get(m, C.size_t(i), &name, &p, &l)); err != nil {
			return nil, err
		}
		out[C.GoString(name)] = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(p)), int(l))...)
	}
	return out, nil
}

func optF32Result(v C.float, known C.bool) *float32 {
	if !bool(known) {
		return nil
	}
	x := float32(v)
	return &x
}

// ── the manual rows ─────────────────────────────────────────

// Step updates each params[name] in place from grads[name].
func (self *Optimizer) Step(params map[string][]float32, grads map[string][]float32) error {
	keys := sortedKeys(params)
	names := newCStrings(keys)
	defer names.free()
	var pin runtime.Pinner
	defer pin.Unpin()
	ps := cArray(len(keys))
	gs := cArray(len(keys))
	ls := cArray(len(keys))
	defer C.free(ps)
	defer C.free(gs)
	defer C.free(ls)
	p := unsafe.Slice((**C.float)(ps), len(keys)+1)
	g := unsafe.Slice((**C.float)(gs), len(keys)+1)
	l := unsafe.Slice((*C.size_t)(ls), len(keys)+1)
	for i, k := range keys {
		grad, ok := grads[k]
		if !ok || len(grad) != len(params[k]) || len(grad) == 0 {
			return &Error{Kind: ErrType, Message: fmt.Sprintf("Optimizer.Step: no matching gradient for '%s'", k)}
		}
		pin.Pin(&params[k][0])
		pin.Pin(&grad[0])
		p[i] = (*C.float)(unsafe.Pointer(&params[k][0]))
		g[i] = (*C.float)(unsafe.Pointer(&grad[0]))
		l[i] = C.size_t(len(grad))
	}
	return check(C.rlx_optimizer_step(self.h, names.ptr, (**C.float)(ps), (**C.float)(gs), (*C.size_t)(ls), nil, nil, C.size_t(len(keys))))
}

// RunOpts drive Trainer.Run.
type RunOpts struct {
	LR     func(step uint64) float32            // the learning rate per step (nil: unchanged)
	OnStep func(step uint64, loss float32) bool // false stops the run
	Every  int                                  // call OnStep every this many steps (default 1)
}

// Run trains steps steps, pulling inputs from batch (nil when the data is
// exhausted). It returns the run summary.
func (self *Trainer) Run(steps int, batch func(step uint64) map[string][]float32, opts *RunOpts) (map[string]any, error) {
	if opts == nil {
		opts = &RunOpts{}
	}
	every := opts.Every
	if every <= 0 {
		every = 1
	}
	start, err := self.Steps()
	if err != nil {
		return nil, err
	}
	var losses []float32
	stopped := false
	t0 := time.Now()
	for i := 0; i < steps; i++ {
		s := start + uint64(i)
		b := batch(s)
		if b == nil {
			break
		}
		if opts.LR != nil {
			if err := self.SetLr(opts.LR(s)); err != nil {
				return nil, err
			}
		}
		loss, err := self.Step(b)
		if err != nil {
			return nil, err
		}
		losses = append(losses, loss)
		if opts.OnStep != nil && (i+1)%every == 0 && !opts.OnStep(s, loss) {
			stopped = true
			break
		}
	}
	summary := map[string]any{"steps": len(losses), "stopped": stopped,
		"steps_per_second": float64(len(losses)) / math.Max(time.Since(t0).Seconds(), 1e-9)}
	if len(losses) > 0 {
		sum := 0.0
		for _, l := range losses {
			sum += float64(l)
		}
		summary["first_loss"], summary["last_loss"] = losses[0], losses[len(losses)-1]
		summary["mean_loss"] = sum / float64(len(losses))
	}
	if gn, err := self.GradNorm(); err == nil && gn != nil {
		summary["grad_norm"] = *gn
	}
	return summary, nil
}
