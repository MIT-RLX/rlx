// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JavaScript `Value` ↔ Rust conversions.
//!
//! Tensors cross the boundary as `Float32Array`. Reading one is a bulk
//! `copy_from_slice` out of the backing `ArrayBuffer` rather than an
//! element-at-a-time `Value` round trip: a `[1,4096]` hidden state is 4096
//! property reads on the slow path and one memcpy on this one. Any other
//! typed array, or a plain `Array` of numbers, still works — it just takes
//! the per-element path.

use quickrs_core::builtins::typedarray::ElementType;
use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, ObjectData, PropFlags};
use quickrs_core::value::{JsResult, PropKey, Value};

// ── arguments ───────────────────────────────────────────────

/// `args[i]`, or `undefined` past the end — JS calling convention.
pub fn arg(args: &[Value], i: usize) -> &Value {
    args.get(i).unwrap_or(&Value::Undefined)
}

pub fn is_nullish(v: &Value) -> bool {
    matches!(v, Value::Undefined | Value::Null)
}

pub fn to_f64(ctx: &mut Context, v: &Value) -> JsResult<f64> {
    ctx.to_number(v)
}

pub fn to_f32(ctx: &mut Context, v: &Value) -> JsResult<f32> {
    Ok(ctx.to_number(v)? as f32)
}

pub fn to_usize(ctx: &mut Context, v: &Value, what: &str) -> JsResult<usize> {
    let n = ctx.to_number(v)?;
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return ctx.throw_type(&format!("{what}: expected a non-negative integer, got {n}"));
    }
    Ok(n as usize)
}

pub fn to_i32(ctx: &mut Context, v: &Value) -> JsResult<i32> {
    ctx.to_int32(v)
}

pub fn to_u64(ctx: &mut Context, v: &Value, what: &str) -> JsResult<u64> {
    Ok(to_usize(ctx, v, what)? as u64)
}

pub fn to_bool(v: &Value) -> bool {
    v.to_boolean()
}

/// `undefined` → the default, anything else → its string form.
pub fn to_string_or(ctx: &mut Context, v: &Value, default: &str) -> JsResult<String> {
    if is_nullish(v) {
        return Ok(default.to_string());
    }
    ctx.to_rust_string(v)
}

// ── arrays ──────────────────────────────────────────────────

/// Length of anything array-like (`Array`, typed array, `{length: n}`).
fn length_of(ctx: &mut Context, v: &Value) -> JsResult<usize> {
    Ok(ctx.length_of(v)? as usize)
}

fn element(ctx: &mut Context, v: &Value, i: usize) -> JsResult<Value> {
    ctx.get_property(v, &PropKey::Index(i as u32))
}

pub fn to_usize_vec(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<usize>> {
    if is_nullish(v) {
        return Ok(Vec::new());
    }
    let n = length_of(ctx, v)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = element(ctx, v, i)?;
        out.push(to_usize(ctx, &e, what)?);
    }
    Ok(out)
}

pub fn to_i64_vec(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<i64>> {
    if is_nullish(v) {
        return Ok(Vec::new());
    }
    let n = length_of(ctx, v)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = element(ctx, v, i)?;
        let x = ctx.to_number(&e)?;
        if !x.is_finite() || x.fract() != 0.0 {
            return ctx.throw_type(&format!("{what}: expected integers, got {x}"));
        }
        out.push(x as i64);
    }
    Ok(out)
}

pub fn to_u32_vec(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<u32>> {
    Ok(to_usize_vec(ctx, v, what)?
        .into_iter()
        .map(|n| n as u32)
        .collect())
}

pub fn to_string_vec(ctx: &mut Context, v: &Value) -> JsResult<Vec<String>> {
    if is_nullish(v) {
        return Ok(Vec::new());
    }
    let n = length_of(ctx, v)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = element(ctx, v, i)?;
        out.push(ctx.to_rust_string(&e)?);
    }
    Ok(out)
}

// ── typed arrays ────────────────────────────────────────────

/// The `(buffer, byte_offset, byte_length)` a view covers, resolving a
/// length-tracking view against its buffer's current size.
fn view_window(obj: &Gc<JsObject>) -> Option<(Gc<JsObject>, usize, usize)> {
    let borrowed = obj.borrow();
    let ObjectData::View(view) = &borrowed.data else {
        return None;
    };
    let buffer = view.buffer.clone();
    let offset = view.byte_offset;
    let len = match view.byte_length {
        Some(n) => n,
        None => {
            let b = buffer.borrow();
            let ObjectData::Buffer(data) = &b.data else {
                return None;
            };
            data.bytes.len().saturating_sub(offset)
        }
    };
    Some((buffer, offset, len))
}

/// Copy a view's window out of its `ArrayBuffer`. `None` if the buffer is
/// detached or the window no longer fits (both are observable from script by
/// transferring or shrinking the buffer mid-call).
fn view_bytes(obj: &Gc<JsObject>) -> Option<Vec<u8>> {
    let (buffer, offset, len) = view_window(obj)?;
    let b = buffer.borrow();
    let ObjectData::Buffer(data) = &b.data else {
        return None;
    };
    if data.detached || offset + len > data.bytes.len() {
        return None;
    }
    Some(data.bytes[offset..offset + len].to_vec())
}

/// Copy a `Float32Array`'s window straight into `out` as one `memcpy`.
///
/// `Vec<f32>` is 4-aligned by construction, so writing bytes through it is
/// sound; `copy_from_slice` on `u8` has no alignment requirement of its own.
/// The element-at-a-time `from_ne_bytes` version this replaced measured
/// 2.7 GiB/s against memcpy's ~50 — the bounds check per 4-byte chunk did not
/// vectorize.
fn copy_f32_window(obj: &Gc<JsObject>, out: &mut Vec<f32>) -> Option<()> {
    let (buffer, offset, len) = view_window(obj)?;
    let borrowed = buffer.borrow();
    let ObjectData::Buffer(data) = &borrowed.data else {
        return None;
    };
    if data.detached || offset + len > data.bytes.len() {
        return None;
    }
    let src = &data.bytes[offset..offset + len];
    let n = src.len() / 4;
    out.clear();
    out.resize(n, 0.0);
    // SAFETY: `out` holds `n` initialized `f32`s, so `n * 4` bytes of its
    // allocation are valid to write, and `src` is a disjoint borrow of the
    // JS-side buffer.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    dst.copy_from_slice(&src[..n * 4]);
    Some(())
}

/// Run `f` over a typed array's bytes **without copying them**.
///
/// The point is asymmetry of scale: pulling 64 rows out of a 47 M-element
/// dataset should cost the 64 rows, not the dataset. Returns `None` when `v` is
/// not a typed array or its buffer is detached.
///
/// `f` must not touch the `Context`: the buffer is borrowed for its duration, so
/// compute a result (or an error to raise afterwards) and return it.
pub fn with_view_bytes<T>(v: &Value, f: impl FnOnce(&[u8]) -> T) -> Option<T> {
    let obj = v.as_object()?;
    let (buffer, offset, len) = view_window(obj)?;
    let borrowed = buffer.borrow();
    let ObjectData::Buffer(data) = &borrowed.data else {
        return None;
    };
    if data.detached || offset + len > data.bytes.len() {
        return None;
    }
    Some(f(&data.bytes[offset..offset + len]))
}

/// As [`with_view_bytes`], reinterpreting the window as `f32`.
///
/// `None` when the array is not a `Float32Array` — a silent reinterpretation of
/// someone else's element type is exactly the bug class worth refusing.
pub fn with_f32_view<T>(v: &Value, f: impl FnOnce(&[f32]) -> T) -> Option<T> {
    if !matches!(Context::typed_array_type(v), Some(ElementType::Float32)) {
        return None;
    }
    with_view_bytes(v, |bytes| {
        let n = bytes.len() / 4;
        // A `Vec<f32>`-backed buffer is 4-aligned, but an `ArrayBuffer` sliced
        // at an odd byte offset need not be, so read through `from_ne_bytes`
        // rather than casting the pointer.
        let mut scratch: Vec<f32> = Vec::new();
        let aligned = (bytes.as_ptr() as usize).is_multiple_of(align_of::<f32>());
        if aligned {
            // SAFETY: checked alignment, and `n * 4 <= bytes.len()`.
            let view = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) };
            f(view)
        } else {
            scratch.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])),
            );
            f(&scratch)
        }
    })
}

/// Read any array-like into `out`, reusing its allocation.
///
/// `Float32Array` takes the memcpy above; `Float64Array` converts in bulk;
/// every other typed array and plain `Array` goes element by element. The
/// `out`-parameter shape is what lets a training loop run allocation-free
/// after the first step.
pub fn read_f32_into(ctx: &mut Context, v: &Value, what: &str, out: &mut Vec<f32>) -> JsResult<()> {
    if is_nullish(v) {
        return ctx.throw_type(&format!("{what}: expected a Float32Array, got undefined"));
    }
    if let (Some(obj), Some(ty)) = (v.as_object(), Context::typed_array_type(v)) {
        match ty {
            ElementType::Float32 => {
                return match copy_f32_window(obj, out) {
                    Some(()) => Ok(()),
                    None => ctx.throw_type(&format!("{what}: the backing ArrayBuffer is detached")),
                };
            }
            ElementType::Float64 => {
                let Some(bytes) = view_bytes(obj) else {
                    return ctx.throw_type(&format!("{what}: the backing ArrayBuffer is detached"));
                };
                out.clear();
                out.extend(bytes.chunks_exact(8).map(|c| {
                    f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                }));
                return Ok(());
            }
            _ => {
                let n = ctx.typed_array_length(obj);
                let obj = obj.clone();
                out.clear();
                out.reserve(n);
                for i in 0..n {
                    let e = ctx.typed_array_get(&obj, i).unwrap_or(Value::Undefined);
                    let x = ctx.to_number(&e)? as f32;
                    out.push(x);
                }
                return Ok(());
            }
        }
    }
    // Plain Array / array-like.
    let n = length_of(ctx, v)?;
    out.clear();
    out.reserve(n);
    for i in 0..n {
        let e = element(ctx, v, i)?;
        let x = ctx.to_number(&e)? as f32;
        out.push(x);
    }
    Ok(())
}

/// Allocating wrapper over [`read_f32_into`], for one-shot reads.
pub fn to_f32_vec(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<f32>> {
    let mut out = Vec::new();
    read_f32_into(ctx, v, what, &mut out)?;
    Ok(out)
}

/// Named `{name: Float32Array}` inputs, read into buffers that persist across
/// calls.
///
/// A step loop passes the same keys every time, so after the first call this
/// allocates nothing: each slot's `Vec<f32>` is refilled in place.
#[derive(Default)]
pub struct InputScratch {
    slots: Vec<(String, Vec<f32>)>,
    /// How many slots the current call filled; the tail is kept for its
    /// allocation but not handed to the runtime.
    live: usize,
}

impl InputScratch {
    /// Refill from an options bag. Every conversion happens here, before the
    /// caller borrows any handle — see [`crate::handle`] on aliasing.
    pub fn load(&mut self, ctx: &mut Context, bag: &Value) -> JsResult<()> {
        let names = own_keys(ctx, bag)?;
        self.live = 0;
        for name in names {
            let value = field(ctx, bag, &name)?;
            // Reuse the slot that already holds this name when the key order
            // is stable, which it is for a loop over one object literal.
            let at = match self.slots.iter().position(|(n, _)| *n == name) {
                Some(found) => {
                    self.slots.swap(self.live, found);
                    self.live
                }
                None => {
                    if self.live < self.slots.len() {
                        self.slots[self.live].0 = name.clone();
                        self.live
                    } else {
                        self.slots.push((name.clone(), Vec::new()));
                        self.slots.len() - 1
                    }
                }
            };
            let mut buffer = std::mem::take(&mut self.slots[at].1);
            let label = format!("input '{name}'");
            let result = read_f32_into(ctx, &value, &label, &mut buffer);
            self.slots[at].1 = buffer;
            result?;
            self.live += 1;
        }
        Ok(())
    }

    /// Append one extra `(name, values)` the caller supplies itself — the
    /// `d_output` gradient seed a training step adds.
    pub fn push(&mut self, name: &str, values: &[f32]) {
        if self.slots.iter().take(self.live).any(|(n, _)| n == name) {
            return;
        }
        if self.live < self.slots.len() {
            self.slots[self.live].0 = name.to_string();
            self.slots[self.live].1.clear();
            self.slots[self.live].1.extend_from_slice(values);
        } else {
            self.slots.push((name.to_string(), values.to_vec()));
        }
        self.live += 1;
    }

    pub fn pairs(&self) -> Vec<(&str, &[f32])> {
        self.slots[..self.live]
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect()
    }
}

/// Raw bytes of any typed array or `ArrayBuffer`, for the dtype-explicit
/// `runTyped` / `setParamTyped` path where the caller owns the encoding.
pub fn to_bytes(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<u8>> {
    let Some(obj) = v.as_object() else {
        return ctx.throw_type(&format!("{what}: expected a TypedArray or ArrayBuffer"));
    };
    {
        let b = obj.borrow();
        if let ObjectData::Buffer(data) = &b.data {
            if data.detached {
                return ctx.throw_type(&format!("{what}: the ArrayBuffer is detached"));
            }
            return Ok(data.bytes.to_vec());
        }
    }
    match view_bytes(obj) {
        Some(bytes) => Ok(bytes),
        None => ctx.throw_type(&format!("{what}: expected a TypedArray or ArrayBuffer")),
    }
}

/// Build a `Float32Array` over a fresh buffer holding `data`.
pub fn new_f32_array(ctx: &mut Context, data: &[f32]) -> Value {
    let byte_len = std::mem::size_of_val(data);
    let buffer = ctx.new_array_buffer(byte_len, false);
    {
        let mut borrowed = buffer.borrow_mut();
        if let ObjectData::Buffer(buf) = &mut borrowed.data {
            // SAFETY: `data` holds initialized `f32`s, so its `len * 4` bytes
            // are valid to read; the destination is a distinct allocation.
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
            buf.bytes[..byte_len].copy_from_slice(src);
        }
    }
    let view = ctx.new_typed_array(ElementType::Float32, buffer, 0, Some(byte_len));
    Value::Object(view)
}

/// Build a `Uint8Array` over a fresh buffer holding `data`.
pub fn new_u8_array(ctx: &mut Context, data: &[u8]) -> Value {
    let buffer = ctx.new_array_buffer(data.len(), false);
    {
        let mut b = buffer.borrow_mut();
        if let ObjectData::Buffer(buf) = &mut b.data {
            buf.bytes.copy_from_slice(data);
        }
    }
    let view = ctx.new_typed_array(ElementType::Uint8, buffer, 0, Some(data.len()));
    Value::Object(view)
}

// ── constructing JS values ──────────────────────────────────

pub fn new_string(ctx: &mut Context, s: &str) -> Value {
    Value::Str(ctx.intern(s))
}

pub fn new_string_array<S: AsRef<str>>(ctx: &mut Context, items: &[S]) -> Value {
    let values: Vec<Value> = items
        .iter()
        .map(|s| Value::Str(ctx.intern(s.as_ref())))
        .collect();
    Value::Object(ctx.new_array_from(values))
}

pub fn new_array(ctx: &mut Context, values: Vec<Value>) -> Value {
    Value::Object(ctx.new_array_from(values))
}

pub fn new_usize_array(ctx: &mut Context, dims: &[usize]) -> Value {
    let values = dims.iter().map(|d| Value::number(*d as f64)).collect();
    Value::Object(ctx.new_array_from(values))
}

/// A plain object built from `(key, value)` pairs, all enumerable.
pub fn new_object(ctx: &mut Context, fields: Vec<(&str, Value)>) -> Value {
    let obj = ctx.new_object();
    for (k, v) in fields {
        ctx.define_value(&obj, k, v, PropFlags::C_W_E);
    }
    Value::Object(obj)
}

/// Read a named property off an options object; `undefined` when absent or
/// when the options argument itself was omitted.
pub fn field(ctx: &mut Context, options: &Value, name: &str) -> JsResult<Value> {
    if is_nullish(options) {
        return Ok(Value::Undefined);
    }
    let key = PropKey::from_string(&ctx.intern(name));
    ctx.get_property(options, &key)
}

/// The own enumerable string keys of a plain object, in insertion order.
///
/// Used for the `{name: Float32Array}` bag that `run` / `setParams` take, so
/// iteration order is the order the script wrote the literal in.
pub fn own_keys(ctx: &mut Context, v: &Value) -> JsResult<Vec<String>> {
    let Some(obj) = v.as_object().cloned() else {
        return Ok(Vec::new());
    };
    let keys = ctx.own_property_keys(&obj)?;
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        // Non-enumerable slots are skipped so an object with inherited or
        // hidden machinery can't smuggle extra entries in.
        let enumerable = {
            let borrowed = obj.borrow();
            borrowed.get_own(&key).map(|p| p.flags.enumerable())
        };
        if enumerable != Some(true) {
            continue;
        }
        match key {
            PropKey::Str(s) => out.push(s.to_string_lossy()),
            PropKey::Index(i) => out.push(i.to_string()),
            PropKey::Sym(_) => {}
        }
    }
    Ok(out)
}
