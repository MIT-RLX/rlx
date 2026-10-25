// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated generated.v stands on: linking, errors, and moving
// values to and from C.
module rlx

#flag -lrlx_capi
#flag -I @VMODROOT
#include "rlx.h"

// none_node is "no node", for an optional node argument.
pub const none_node = u32(0xFFFFFFFF)

// RlxError is a failed rlx call: kind is type, range, runtime, panic or not_found.
pub struct RlxError {
	Error
pub:
	kind    string
	message string
}

pub fn (e RlxError) msg() string {
	return e.message
}

// TypedTensor is a tensor of any dtype: its little-endian bytes and dtype name.
pub struct TypedTensor {
pub:
	data  []u8
	dtype string
}

struct Slice {
	ptr voidptr
	len usize
}

fn C.rlx_last_error() &char
fn C.rlx_string_free(s voidptr)
fn C.rlx_f32s_release(v voidptr)
fn C.rlx_bytes_release(v voidptr)
fn C.rlx_u32s_release(v voidptr)
fn C.rlx_outputs_len(o voidptr) usize
fn C.rlx_outputs_get(o voidptr, i usize, d voidptr, n voidptr) i32
fn C.rlx_outputs_free(o voidptr)
fn C.rlx_typed_outputs_len(o voidptr) usize
fn C.rlx_typed_outputs_get(o voidptr, i usize, d voidptr, n voidptr, dt voidptr) i32
fn C.rlx_typed_outputs_free(o voidptr)
fn C.rlx_tensor_map_len(m voidptr) usize
fn C.rlx_tensor_map_get(m voidptr, i usize, name voidptr, d voidptr, n voidptr) i32
fn C.rlx_tensor_map_free(m voidptr)
fn C.rlx_optimizer_step(o voidptr, names voidptr, params voidptr, grads voidptr, lens voidptr, shapes voidptr, ndims voidptr, n usize) i32

const kinds = {
	-1: 'type'
	-2: 'range'
	-3: 'runtime'
	-4: 'panic'
	-5: 'not_found'
}

fn check_rc(rc i32) ! {
	if rc != 0 {
		return RlxError{
			kind:    kinds[int(rc)] or { 'other' }
			message: unsafe { cstring_to_vstring(C.rlx_last_error()) }
		}
	}
}

// ── arguments ─────────────────────────────────────────────────

fn ptr_of[T](a []T) voidptr {
	if a.len == 0 {
		return unsafe { nil }
	}
	return a.data
}

fn to_f32s(v []f64) []f32 {
	return v.map(f32(it))
}

fn to_u32s(v []int) []u32 {
	return v.map(u32(it))
}

fn to_u64s(v []int) []u64 {
	return v.map(u64(it))
}

fn to_i64s(v []int) []i64 {
	return v.map(i64(it))
}

fn cstrs(v []string) []&char {
	return v.map(&char(it.str))
}

fn rows(v [][]int, w int, splat bool) ![]int {
	mut out := []int{}
	for i, r in v {
		if r.len == w {
			out << r
		} else if splat && r.len == 1 {
			for _ in 0 .. w {
				out << r[0]
			}
		} else {
			return RlxError{
				kind:    'type'
				message: 'row ${i} needs ${w} values, got ${r.len}'
			}
		}
	}
	return out
}

fn frows(v [][]f64, w int) ![]f64 {
	mut out := []f64{}
	for i, r in v {
		if r.len != w {
			return RlxError{
				kind:    'type'
				message: 'row ${i} needs ${w} values, got ${r.len}'
			}
		}
		out << r
	}
	return out
}

struct Inputs {
	names  []&char
	data   []voidptr
	lens   []usize
	dtypes []&char
	keep   [][]f32
}

fn new_inputs(m map[string][]f64) Inputs {
	mut keys := m.keys()
	keys.sort()
	keep := keys.map(to_f32s(m[it]))
	return Inputs{
		names: cstrs(keys)
		data:  keep.map(ptr_of(it))
		lens:  keep.map(usize(it.len))
		keep:  keep
	}
}

fn new_typed_inputs(m map[string]TypedTensor) Inputs {
	mut keys := m.keys()
	keys.sort()
	return Inputs{
		names:  cstrs(keys)
		data:   keys.map(ptr_of(m[it].data))
		lens:   keys.map(usize(m[it].data.len))
		dtypes: keys.map(&char(m[it].dtype.str))
	}
}

// ── results ───────────────────────────────────────────────────

fn take_string_opt(p voidptr) ?string {
	if p == unsafe { nil } {
		return none
	}
	s := unsafe { cstring_to_vstring(&char(p)) }
	C.rlx_string_free(p)
	return s
}

fn take_string(p voidptr) string {
	return take_string_opt(p) or { '' }
}

fn floats_at(p voidptr, n usize) []f64 {
	mut out := []f64{cap: int(n)}
	a := unsafe { &f32(p) }
	for i in 0 .. int(n) {
		out << f64(unsafe { a[i] })
	}
	return out
}

fn take_f32s(mut o Slice) []f64 {
	v := floats_at(o.ptr, o.len)
	C.rlx_f32s_release(voidptr(&o))
	return v
}

fn take_bytes(mut o Slice) []u8 {
	v := if o.len == 0 { []u8{} } else { unsafe { (&u8(o.ptr)).vbytes(int(o.len)) }.clone() }
	C.rlx_bytes_release(voidptr(&o))
	return v
}

fn take_u32s(mut o Slice) []int {
	mut out := []int{}
	a := unsafe { &u32(o.ptr) }
	for i in 0 .. int(o.len) {
		out << int(unsafe { a[i] })
	}
	C.rlx_u32s_release(voidptr(&o))
	return out
}

fn take_outputs(o voidptr) [][]f64 {
	mut out := [][]f64{}
	for i in 0 .. C.rlx_outputs_len(o) {
		mut d := voidptr(unsafe { nil })
		mut n := usize(0)
		C.rlx_outputs_get(o, i, voidptr(&d), voidptr(&n))
		out << floats_at(d, n)
	}
	C.rlx_outputs_free(o)
	return out
}

fn take_typed_outputs(o voidptr) []TypedTensor {
	mut out := []TypedTensor{}
	for i in 0 .. C.rlx_typed_outputs_len(o) {
		mut d := voidptr(unsafe { nil })
		mut n := usize(0)
		mut dt := &char(unsafe { nil })
		C.rlx_typed_outputs_get(o, i, voidptr(&d), voidptr(&n), voidptr(&dt))
		out << TypedTensor{
			data:  if n == 0 { []u8{} } else { unsafe { (&u8(d)).vbytes(int(n)) }.clone() }
			dtype: unsafe { cstring_to_vstring(dt) }
		}
	}
	C.rlx_typed_outputs_free(o)
	return out
}

fn take_tensors(m voidptr) map[string][]f64 {
	mut out := map[string][]f64{}
	for i in 0 .. C.rlx_tensor_map_len(m) {
		mut name := &char(unsafe { nil })
		mut d := voidptr(unsafe { nil })
		mut n := usize(0)
		C.rlx_tensor_map_get(m, i, voidptr(&name), voidptr(&d), voidptr(&n))
		out[unsafe { cstring_to_vstring(name) }] = floats_at(d, n)
	}
	C.rlx_tensor_map_free(m)
	return out
}
