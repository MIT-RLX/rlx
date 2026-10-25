// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated generated.odin stands on: the library, errors, and
// moving values to and from C. Results are allocated with context.allocator.
package rlx


import "core:encoding/json"
import "core:fmt"
import "core:slice"
import "core:strings"
import "core:time"

foreign import lib "system:rlx_capi"

// A node of a graph: the id every builder returns.
Node :: distinct u32
// "No node", for an optional node argument.
NONE :: Node(0xFFFFFFFF)

// A failed call's status (rlx-capi's RLX_ERR_*); its message is `last_error()`.
Error :: enum i32 {
	None      = 0,
	Type      = -1,
	Range     = -2,
	Runtime   = -3,
	Panic     = -4,
	Not_Found = -5,
}

// A tensor of any dtype: its little-endian bytes and dtype name.
Typed_Tensor :: struct {
	data:  []u8,
	dtype: string,
}

Slice :: struct {
	ptr: rawptr,
	len: uint,
}

@(default_calling_convention = "c")
foreign lib {
	rlx_last_error :: proc() -> cstring ---
	rlx_string_free :: proc(s: rawptr) ---
	rlx_f32s_release :: proc(v: ^Slice) ---
	rlx_bytes_release :: proc(v: ^Slice) ---
	rlx_u32s_release :: proc(v: ^Slice) ---
	rlx_outputs_len :: proc(o: rawptr) -> uint ---
	rlx_outputs_get :: proc(o: rawptr, i: uint, d: ^[^]f32, n: ^uint) -> i32 ---
	rlx_outputs_free :: proc(o: rawptr) ---
	rlx_typed_outputs_len :: proc(o: rawptr) -> uint ---
	rlx_typed_outputs_get :: proc(o: rawptr, i: uint, d: ^[^]u8, n: ^uint, dt: ^cstring) -> i32 ---
	rlx_typed_outputs_free :: proc(o: rawptr) ---
	rlx_tensor_map_len :: proc(m: rawptr) -> uint ---
	rlx_tensor_map_get :: proc(m: rawptr, i: uint, name: ^cstring, d: ^[^]f32, n: ^uint) -> i32 ---
	rlx_tensor_map_free :: proc(m: rawptr) ---
	rlx_optimizer_step :: proc(o: rawptr, names: [^]cstring, params: [^][^]f32, grads: [^][^]f32, lens: [^]uint, shapes: rawptr, ndims: rawptr, n: uint) -> i32 ---
}

// The message of the last failed call on this thread.
last_error :: proc() -> string {
	return string(rlx_last_error())
}

// The value, or a panic with the last error's message.
must :: proc(v: $T, err: Error, loc := #caller_location) -> T {
	if err != .None {
		panic(fmt.tprintf("rlx %v: %s", err, last_error()), loc)
	}
	return v
}

// ── arguments ──────────────────────────────────────────────────

opt_cstring :: proc(s: string) -> cstring {
	return len(s) == 0 ? nil : strings.clone_to_cstring(s)
}

maybe_cstring :: proc(s: Maybe(string)) -> cstring {
	v, ok := s.?
	return ok ? strings.clone_to_cstring(v) : nil
}

to_u64s :: proc(v: []int) -> []u64 {
	out := make([]u64, len(v))
	for x, i in v do out[i] = u64(x)
	return out
}

to_i64s :: proc(v: []int) -> []i64 {
	out := make([]i64, len(v))
	for x, i in v do out[i] = i64(x)
	return out
}

flat_rows :: proc(v: [][]int, w: int, splat: bool) -> []u64 {
	out := make([dynamic]u64)
	for r in v {
		if len(r) == w {
			for x in r do append(&out, u64(x))
		} else if splat && len(r) == 1 {
			for _ in 0 ..< w do append(&out, u64(r[0]))
		} else {
			panic(fmt.tprintf("rlx: a row needs %d values, got %d", w, len(r)))
		}
	}
	return out[:]
}

flat_frows :: proc(v: [][]f32, w: int) -> []f32 {
	out := make([dynamic]f32)
	for r in v {
		if len(r) != w do panic(fmt.tprintf("rlx: a row needs %d values, got %d", w, len(r)))
		append(&out, ..r)
	}
	return out[:]
}

cstrings :: proc(v: []string) -> []cstring {
	out := make([]cstring, len(v))
	for s, i in v do out[i] = strings.clone_to_cstring(s)
	return out
}

free_cstrings :: proc(v: []cstring) {
	for s in v do delete(s)
	delete(v)
}

Inputs :: struct {
	names:  []cstring,
	data:   []rawptr,
	lens:   []uint,
	dtypes: []cstring,
}

make_inputs :: proc(m: map[string][]f32) -> Inputs {
	keys := slice.map_keys(m) or_else nil
	defer delete(keys)
	slice.sort(keys)
	in_ := Inputs{names = make([]cstring, len(keys)), data = make([]rawptr, len(keys)), lens = make([]uint, len(keys))}
	for k, i in keys {
		in_.names[i] = strings.clone_to_cstring(k)
		in_.data[i] = raw_data(m[k])
		in_.lens[i] = uint(len(m[k]))
	}
	return in_
}

make_typed_inputs :: proc(m: map[string]Typed_Tensor) -> Inputs {
	keys := slice.map_keys(m) or_else nil
	defer delete(keys)
	slice.sort(keys)
	n := len(keys)
	in_ := Inputs{names = make([]cstring, n), data = make([]rawptr, n), lens = make([]uint, n), dtypes = make([]cstring, n)}
	for k, i in keys {
		t := m[k]
		in_.names[i] = strings.clone_to_cstring(k)
		in_.data[i] = raw_data(t.data)
		in_.lens[i] = uint(len(t.data))
		in_.dtypes[i] = strings.clone_to_cstring(t.dtype)
	}
	return in_
}

free_inputs :: proc(in_: Inputs) {
	free_cstrings(in_.names)
	if in_.dtypes != nil do free_cstrings(in_.dtypes)
	delete(in_.data)
	delete(in_.lens)
}

// ── results ────────────────────────────────────────────────────

take_maybe_string :: proc(p: rawptr) -> Maybe(string) {
	if p == nil do return nil
	s := strings.clone_from_cstring(cstring(p))
	rlx_string_free(p)
	return s
}

take_string :: proc(p: rawptr) -> string {
	return take_maybe_string(p).? or_else ""
}

take_json :: proc(p: rawptr) -> json.Value {
	s := take_string(p)
	defer delete(s)
	v, _ := json.parse_string(s)
	return v
}

take_strings :: proc(p: rawptr) -> []string {
	v := take_json(p)
	defer json.destroy_value(v)
	arr := v.(json.Array) or_else nil
	out := make([]string, len(arr))
	for e, i in arr do out[i] = strings.clone(e.(json.String) or_else "")
	return out
}

take_ints :: proc(p: rawptr) -> []int {
	v := take_json(p)
	defer json.destroy_value(v)
	return ints_of(v)
}

@(private)
ints_of :: proc(v: json.Value) -> []int {
	arr := v.(json.Array) or_else nil
	out := make([]int, len(arr))
	for e, i in arr {
		#partial switch n in e {
		case json.Integer:
			out[i] = int(n)
		case json.Float:
			out[i] = int(n)
		}
	}
	return out
}

take_int_rows :: proc(p: rawptr) -> [][]int {
	v := take_json(p)
	defer json.destroy_value(v)
	arr := v.(json.Array) or_else nil
	out := make([][]int, len(arr))
	for e, i in arr do out[i] = ints_of(e)
	return out
}

take_f32s :: proc(o: ^Slice) -> []f32 {
	v := slice.clone(([^]f32)(o.ptr)[:o.len])
	rlx_f32s_release(o)
	return v
}

take_bytes :: proc(o: ^Slice) -> []u8 {
	v := slice.clone(([^]u8)(o.ptr)[:o.len])
	rlx_bytes_release(o)
	return v
}

take_u32s :: proc(o: ^Slice) -> []Node {
	v := slice.clone(([^]Node)(o.ptr)[:o.len])
	rlx_u32s_release(o)
	return v
}

take_outputs :: proc(o: rawptr) -> [][]f32 {
	n := rlx_outputs_len(o)
	out := make([][]f32, n)
	for i in 0 ..< n {
		d: [^]f32
		k: uint
		rlx_outputs_get(o, i, &d, &k)
		out[i] = slice.clone(d[:k])
	}
	rlx_outputs_free(o)
	return out
}

take_typed_outputs :: proc(o: rawptr) -> []Typed_Tensor {
	n := rlx_typed_outputs_len(o)
	out := make([]Typed_Tensor, n)
	for i in 0 ..< n {
		d: [^]u8
		k: uint
		dt: cstring
		rlx_typed_outputs_get(o, i, &d, &k, &dt)
		out[i] = Typed_Tensor{slice.clone(d[:k]), strings.clone_from_cstring(dt)}
	}
	rlx_typed_outputs_free(o)
	return out
}

take_tensors :: proc(m: rawptr) -> map[string][]f32 {
	out := make(map[string][]f32)
	for i in 0 ..< rlx_tensor_map_len(m) {
		name: cstring
		d: [^]f32
		k: uint
		rlx_tensor_map_get(m, i, &name, &d, &k)
		out[strings.clone_from_cstring(name)] = slice.clone(d[:k])
	}
	rlx_tensor_map_free(m)
	return out
}

// ── the manual rows ────────────────────────────────────────────

// Update each `params[name]` in place from `grads[name]`.
optimizer_step :: proc(self: Optimizer, params: map[string][]f32, grads: map[string][]f32) -> Error {
	keys := slice.map_keys(params) or_else nil
	defer delete(keys)
	slice.sort(keys)
	n := len(keys)
	names := cstrings(keys)
	defer free_cstrings(names)
	ps := make([][^]f32, n)
	gs := make([][^]f32, n)
	lens := make([]uint, n)
	defer {delete(ps); delete(gs); delete(lens)}
	for k, i in keys {
		g, ok := grads[k]
		if !ok do return .Type
		ps[i] = raw_data(params[k])
		gs[i] = raw_data(g)
		lens[i] = uint(len(params[k]))
	}
	return Error(rlx_optimizer_step(rawptr(self), raw_data(names), raw_data(ps), raw_data(gs), raw_data(lens), nil, nil, uint(n)))
}

// What `trainer_run` did.
Train_Summary :: struct {
	steps:            int,
	stopped:          bool,
	first_loss:       Maybe(f32),
	last_loss:        Maybe(f32),
	mean_loss:        Maybe(f32),
	steps_per_second: f64,
	grad_norm:        Maybe(f32),
}

// Train `steps` steps, pulling inputs from `batch(step)` (`ok = false`: data
// exhausted). `lr(step)` schedules the rate; `on_step(step, loss)` returning
// false stops early (called every `every` steps).
trainer_run :: proc(
	self: Trainer,
	steps: int,
	batch: proc(step: int) -> (map[string][]f32, bool),
	lr: proc(step: int) -> f32 = nil,
	on_step: proc(step: int, loss: f32) -> bool = nil,
	every := 1,
) -> (r: Train_Summary, err: Error) {
	stride := max(every, 1)
	start := trainer_steps(self) or_return
	losses := make([dynamic]f32)
	defer delete(losses)
	stopped := false
	t0 := time.tick_now()
	for i in 0 ..< steps {
		s := start + i
		b, ok := batch(s)
		if !ok do break
		if lr != nil do trainer_set_lr(self, lr(s)) or_return
		loss := trainer_step(self, b) or_return
		append(&losses, loss)
		if on_step != nil && (i + 1) % stride == 0 && !on_step(s, loss) {
			stopped = true
			break
		}
	}
	secs := max(time.duration_seconds(time.tick_since(t0)), 1e-9)
	r = Train_Summary{steps = len(losses), stopped = stopped, steps_per_second = f64(len(losses)) / secs}
	if len(losses) > 0 {
		r.first_loss = losses[0]
		r.last_loss = losses[len(losses) - 1]
		sum: f32
		for l in losses do sum += l
		r.mean_loss = sum / f32(len(losses))
	}
	r.grad_norm = trainer_grad_norm(self) or_return
	return r, .None
}

