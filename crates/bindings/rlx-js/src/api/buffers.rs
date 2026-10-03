// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed-array utilities that would otherwise be interpreted loops.
//!
//! Dataset preparation is the one place a script does per-element work over tens
//! of millions of values, and an interpreted `for` loop there dominates
//! everything else: normalizing MNIST's 47 M pixels in JS took 19.7 s against
//! 0.9 s for `Float32Array.prototype.set` and less again for one native pass
//! that widens *and* scales.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};

use crate::convert::*;

/// `toFloat32(source, {scale, bias})` → `Float32Array` of `v * scale + bias`.
///
/// Defaults to a plain conversion (`scale: 1, bias: 0`). Folding the affine in
/// means a `(pixel / 255 - mean) / std` pass costs one traversal:
///
/// ```js
/// const std = 0.3081, mean = 0.1307;
/// const x = rlx.toFloat32(raw, { scale: 1 / (255 * std), bias: -mean / std });
/// ```
///
/// Dispatches on `source`'s **element type**, per element, not per byte. The
/// first version read every input as raw bytes, so
/// `toFloat32(new Float32Array([1, 2, 3]))` returned twelve values
/// (`0, 0, 128, 63, …` — the IEEE bytes) instead of three. A conversion helper
/// that silently reinterprets its input is worse than one that refuses.
fn f_to_f32(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use quickrs_core::builtins::typedarray::ElementType;

    let source = arg(args, 0).clone();
    let options = arg(args, 1).clone();
    let scale_v = field(ctx, &options, "scale")?;
    let scale = if is_nullish(&scale_v) {
        1.0
    } else {
        to_f32(ctx, &scale_v)?
    };
    let bias_v = field(ctx, &options, "bias")?;
    let bias = if is_nullish(&bias_v) {
        0.0
    } else {
        to_f32(ctx, &bias_v)?
    };
    let affine = |v: f32| {
        if scale == 1.0 && bias == 0.0 {
            v
        } else {
            v * scale + bias
        }
    };

    // Borrowed, not copied: converting a 47 MB dataset should be one pass.
    let converted = match Context::typed_array_type(&source) {
        Some(ElementType::Uint8) | Some(ElementType::Uint8Clamped) => {
            with_view_bytes(&source, |b| {
                b.iter().map(|x| affine(*x as f32)).collect::<Vec<_>>()
            })
        }
        Some(ElementType::Int8) => with_view_bytes(&source, |b| {
            b.iter()
                .map(|x| affine(*x as i8 as f32))
                .collect::<Vec<_>>()
        }),
        Some(ElementType::Int16) => with_view_bytes(&source, |b| {
            b.chunks_exact(2)
                .map(|c| affine(i16::from_ne_bytes([c[0], c[1]]) as f32))
                .collect::<Vec<_>>()
        }),
        Some(ElementType::Uint16) => with_view_bytes(&source, |b| {
            b.chunks_exact(2)
                .map(|c| affine(u16::from_ne_bytes([c[0], c[1]]) as f32))
                .collect::<Vec<_>>()
        }),
        Some(ElementType::Int32) => with_view_bytes(&source, |b| {
            b.chunks_exact(4)
                .map(|c| affine(i32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f32))
                .collect::<Vec<_>>()
        }),
        Some(ElementType::Uint32) => with_view_bytes(&source, |b| {
            b.chunks_exact(4)
                .map(|c| affine(u32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f32))
                .collect::<Vec<_>>()
        }),
        Some(ElementType::Float32) => with_f32_view(&source, |v| {
            v.iter().map(|x| affine(*x)).collect::<Vec<_>>()
        }),
        Some(ElementType::Float64) => with_view_bytes(&source, |b| {
            b.chunks_exact(8)
                .map(|c| {
                    affine(
                        f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
                    )
                })
                .collect::<Vec<_>>()
        }),
        // Float16 / BigInt64 / BigUint64, or not a typed array at all.
        _ => None,
    };
    if let Some(out) = converted {
        return Ok(new_f32_array(ctx, &out));
    }

    // A plain `Array` (or an element type without a fast path) goes through the
    // generic numeric reader, which is per-element but correct.
    let values = to_f32_vec(ctx, &source, "toFloat32 input")?;
    let out: Vec<f32> = values.into_iter().map(affine).collect();
    Ok(new_f32_array(ctx, &out))
}

/// `oneHot(labels, classes, {batch})` → `Float32Array` of `labels.length x classes`.
///
/// The inner loop of every classification batch, and the one place a stray
/// out-of-range label silently writes into the previous row rather than erroring.
fn f_one_hot(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let labels = to_usize_vec(ctx, arg(args, 0), "oneHot labels")?;
    let classes = to_usize(ctx, arg(args, 1), "oneHot classes")?;
    if classes == 0 {
        return ctx.throw_type("oneHot: `classes` must be at least 1");
    }
    let mut out = vec![0.0f32; labels.len() * classes];
    for (row, label) in labels.iter().enumerate() {
        if *label >= classes {
            return ctx.throw_range(&format!(
                "oneHot: label {label} at index {row} is outside 0..{classes}"
            ));
        }
        out[row * classes + label] = 1.0;
    }
    Ok(new_f32_array(ctx, &out))
}

/// `gatherRows(source, indices, rowLength)` → the selected rows, concatenated.
///
/// This is batch assembly: `source` is the flat dataset, `indices` the shuffled
/// row ids. Doing it with `subarray` + `set` per row costs one JS call per row;
/// here it is one call per batch.
fn f_gather_rows(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let source = arg(args, 0).clone();
    let indices = to_usize_vec(ctx, arg(args, 1), "gatherRows indices")?;
    let row = to_usize(ctx, arg(args, 2), "gatherRows rowLength")?;
    if row == 0 {
        return ctx.throw_type("gatherRows: `rowLength` must be at least 1");
    }

    // Borrowed. The first version read the source with `to_f32_vec`, which
    // copied the whole dataset on every call and made gathering 64 rows out of
    // 47 M elements *slower* than `subarray` + `set` in JS by 4x.
    let gathered = with_f32_view(&source, |src| {
        let rows = src.len() / row;
        let mut out = vec![0.0f32; indices.len() * row];
        for (slot, index) in indices.iter().enumerate() {
            if *index >= rows {
                return Err((*index, slot, rows));
            }
            let from = index * row;
            out[slot * row..(slot + 1) * row].copy_from_slice(&src[from..from + row]);
        }
        Ok(out)
    });
    match gathered {
        Some(Ok(out)) => Ok(new_f32_array(ctx, &out)),
        Some(Err((index, slot, rows))) => ctx.throw_range(&format!(
            "gatherRows: index {index} at position {slot} is outside 0..{rows}"
        )),
        None => ctx.throw_type("gatherRows: `source` must be a Float32Array"),
    }
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "toFloat32" => f_to_f32, 2;
        "oneHot" => f_one_hot, 3;
        "gatherRows" => f_gather_rows, 3;
    };
}
