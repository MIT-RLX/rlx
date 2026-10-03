// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rest of the IR surface, hung off the same `rlx.Graph` prototype:
//! linalg factorizations, quantization, DSP/FFT, shape manipulation, losses
//! and the fused matmul family.
//!
//! Split from [`crate::api::graph`] only for readability — to a script these
//! are ordinary `Graph` methods. The same argument-conversion-before-borrow
//! rule applies to every one of them.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, NativeFn};
use quickrs_core::value::{JsResult, PropKey, Value};
use rlx_ir::fft::FftNorm;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{PadMode, QrPart, SvdPart};
use rlx_ir::{DType, Dim, Graph, NodeId, Shape};

use crate::api::graph::{check_node, check_nodes, graph_of, id, node, parse_activation, shape_arg};
use crate::convert::*;

// ── shared parsers ──────────────────────────────────────────

pub(crate) fn parse_fft_norm(ctx: &mut Context, v: &Value) -> JsResult<FftNorm> {
    let label = to_string_or(ctx, v, "backward")?;
    Ok(match label.trim().to_ascii_lowercase().as_str() {
        "backward" | "none" => FftNorm::Backward,
        "ortho" | "orthonormal" => FftNorm::Ortho,
        "forward" => FftNorm::Forward,
        other => {
            return ctx.throw_type(&format!(
                "unknown FFT norm '{other}' (backward, ortho, forward)"
            ));
        }
    })
}

// ── the declarative table ───────────────────────────────────
//
// Each line is: JS name, typed parameters (with JS-style defaults), optional
// `-> pair` for two-output ops, and the IR call. `graph_ops!` reads every
// argument before it borrows the graph, so a getter in an argument position
// cannot alias a live handle.

fn install_ops(ctx: &mut Context, proto: &Gc<JsObject>) {
    graph_ops! {
        ctx, proto;

        // ── creation ──
        /// `zeros(dims, dtype)`
        "zeros"(dims: usizes, dtype: dtype = DType::F32) => |g| g.full(&dims, 0.0, dtype);
        /// `full(dims, value, dtype)`
        "full"(dims: usizes, value: f32, dtype: dtype = DType::F32) => |g| g.full(&dims, value, dtype);

        // ── elementwise ──
        "sigmoid"(x: node) => |g| g.sigmoid(x);
        "log"(x: node) => |g| g.log(x);
        "abs"(x: node) => |g| g.abs(x);
        "recip"(x: node) => |g| g.recip(x);
        "rsqrt"(x: node) => |g| g.rsqrt(x);
        /// Complex conjugate — the Wirtinger-AD primitive, not a no-op on C64.
        "conjugate"(z: node) => |g| g.conjugate(z);
        "clamp"(x: node, min: f32, max: f32) => |g| g.clamp_(x, min, max);

        // ── shape manipulation ──
        /// `slice(x, axis, start, length, step)`; `step` may be negative.
        "slice"(x: node, axis: usize, start: usize, len: usize, step: i64 = 1)
            => |g| g.slice_(x, axis, start, len, step);
        "tile"(x: node, reps: usizes) => |g| g.tile_(x, reps);
        /// Keep one triangle, zero the other.
        "trilu"(x: node, upper: bool, diagonal: i64 = 0) => |g| g.trilu_(x, upper, diagonal);
        "roll"(x: node, shifts: i64s, dims: usizes) => |g| g.roll_(x, shifts, dims);
        "reverse"(x: node, axes: usizes) => |g| g.reverse(x, axes);
        "gatherElements"(data: node, indices: node, axis: i32) => |g| g.gather_elements(data, indices, axis);
        "cumprod"(x: node, axis: i32, exclusive: bool = false) => |g| g.cumprod_(x, axis, exclusive);
        "cummax"(x: node, axis: i32, exclusive: bool = false) => |g| g.cummax_(x, axis, exclusive);

        // ── ordering (shape == input, or reduced along one axis) ──
        "argmax"(x: node, axis: usize, keepDim: bool = false) => |g, ctx| {
            let shape = reduced(ctx, g, x, axis, keepDim)?;
            g.argmax(x, axis, keepDim, shape)
        };
        "argmin"(x: node, axis: usize, keepDim: bool = false) => |g, ctx| {
            let shape = reduced(ctx, g, x, axis, keepDim)?;
            g.argmin(x, axis, keepDim, shape)
        };
        "sort"(x: node, axis: usize, descending: bool = false) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(x));
            g.sort(x, axis, descending, shape)
        };
        "argsort"(x: node, axis: usize, descending: bool = false) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(x));
            g.argsort(x, axis, descending, shape)
        };

        // ── losses ──
        /// Fused softmax + cross-entropy against a dense target distribution.
        /// The unfused softmax→log→gather chain loses precision on confident
        /// logits and costs three extra kernels.
        "softmaxCrossEntropy"(logits: node, targets: node) => |g| g.softmax_cross_entropy(logits, targets);

        // ── fused / low-precision matmul ──
        "linearBias"(x: node, w: node) => |g| g.linear_bias(x, w, None);
        "quantize"(x: node, scale: f32, zeroPoint: i32 = 0) => |g| g.quantize(x, scale, zeroPoint);
        "dequantize"(x: node, scale: f32, zeroPoint: i32 = 0) => |g| g.dequantize(x, scale, zeroPoint);

        // ── normalization ──
        /// LayerNorm over the channel axis of an NCHW feature map.
        "layerNorm2d"(x: node, gamma: node, beta: node, eps: f32 = 1e-5)
            => |g| g.layer_norm2d(x, gamma, beta, eps);

        // ── linear algebra ──
        "batchedDenseSolve"(a: node, b: node) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(b));
            g.batched_dense_solve(a, b, shape)
        };
        "cholesky"(a: node) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(a));
            g.cholesky(a, shape)
        };
        "det"(a: node) => |g| {
            let shape = Shape::scalar(g.shape(a).dtype());
            g.det(a, shape)
        };
        "logdet"(a: node) => |g| {
            let shape = Shape::scalar(g.shape(a).dtype());
            g.logdet(a, shape)
        };
        /// Symmetric eigendecomposition → `[eigenvalues, eigenvectors]`.
        "eigh"(a: node) -> pair => |g| g.eigh(a);

        // ── statistics ──
        "histogram"(x: node, bins: usize, min: f32, max: f32) => |g| g.histogram(x, bins, min, max);

        // ── DSP / spectral ──
        "fftNorm"(x: node, inverse: bool = false, norm: fft_norm = FftNorm::Backward)
            => |g| g.fft_norm(x, inverse, norm);
        /// Real signal → full spectrum `[re, im]`.
        "fftReal"(x: node, norm: fft_norm = FftNorm::Backward) -> pair => |g| g.fft_real(x, norm);
        /// Non-power-of-two real FFT → half spectrum `[re, im]`.
        "rfftExact"(x: node, n: usize, norm: fft_norm = FftNorm::Backward) -> pair
            => |g| g.rfft_exact(x, n, norm);
        "psd"(re: node, im: node) => |g| g.psd(re, im);
        /// Real signal straight to PSD — no spectrum materialized in between.
        "psdReal"(x: node, norm: fft_norm = FftNorm::Backward) => |g| g.psd_real(x, norm);
        "stft"(x: node, frameLen: usize, hop: usize, norm: fft_norm = FftNorm::Backward)
            => |g| g.stft(x, frameLen, hop, norm);
        "fftfreq"(n: usize) => |g| g.fftfreq_tensor(n);
        "rfftfreq"(n: usize) => |g| g.rfftfreq_tensor(n);
    }
}

/// Output shape of a one-axis reduction. Takes `ctx` so a bad axis becomes a
/// `TypeError` the script can catch rather than a panic.
fn reduced(
    ctx: &mut Context,
    g: &Graph,
    x: NodeId,
    axis: usize,
    keep_dim: bool,
) -> JsResult<Shape> {
    match rlx_ir::shape::reduce_shape(g.shape(x), &[axis], keep_dim) {
        Ok(shape) => Ok(shape),
        Err(e) => ctx.throw_type(&format!("reduce shape: {e}")),
    }
}

// ── ops the table deliberately does not model ───────────────
//
// Nested arrays, a string that selects which factor to return, and explicit
// output shapes are each one-off enough that a macro arm for them would be
// harder to read than the function.

/// `"constant"` (with an optional fill value), `"reflect"`, `"replicate"`,
/// `"circular"`.
fn parse_pad_mode(ctx: &mut Context, v: &Value, value: f32) -> JsResult<PadMode> {
    let label = to_string_or(ctx, v, "constant")?;
    Ok(match label.trim().to_ascii_lowercase().as_str() {
        "constant" | "zero" | "zeros" => PadMode::Constant(value),
        "reflect" => PadMode::Reflect,
        "replicate" | "edge" => PadMode::Replicate,
        "circular" | "wrap" => PadMode::Circular,
        other => {
            return ctx.throw_type(&format!(
                "unknown pad mode '{other}' (constant, reflect, replicate, circular)"
            ));
        }
    })
}

/// `pad(x, [[before, after], …], mode?, value?)` — one pair per axis, or a
/// single number per axis for a symmetric width.
fn m_pad(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let x = node(ctx, args, 0, "pad")?;
    let raw = arg(args, 1).clone();
    let axes = ctx.length_of(&raw)? as usize;
    let mut pads = Vec::with_capacity(axes);
    for i in 0..axes {
        let row = ctx.get_property(&raw, &PropKey::Index(i as u32))?;
        let pair = to_usize_vec(ctx, &row, "pad widths")?;
        match pair.len() {
            1 => pads.push([pair[0], pair[0]]),
            2 => pads.push([pair[0], pair[1]]),
            got => {
                return ctx.throw_type(&format!(
                    "pad: axis {i} needs [before, after], got {got} values"
                ));
            }
        }
    }
    let value = if is_nullish(arg(args, 3)) {
        0.0
    } else {
        to_f32(ctx, arg(args, 3))?
    };
    let mode = parse_pad_mode(ctx, arg(args, 2), value)?;
    let g = graph_of(ctx, this)?;
    check_node(ctx, g, x, "pad")?;
    Ok(id(g.pad_(x, pads, mode)))
}

/// Static dims of a node, or a `TypeError` — the thin factorizations need
/// `min(m, n)` at build time, so a dynamic axis cannot be deferred.
fn static_dims(ctx: &mut Context, g: &Graph, n: NodeId, what: &str) -> JsResult<Vec<usize>> {
    check_node(ctx, g, n, what)?;
    let shape = g.shape(n);
    let mut dims = Vec::with_capacity(shape.rank());
    for d in shape.dims() {
        match d {
            Dim::Static(v) => dims.push(*v),
            Dim::Dynamic(name) => {
                return ctx.throw_type(&format!(
                    "{what}: needs a static shape, but axis '{name}' is dynamic"
                ));
            }
        }
    }
    if dims.len() != 2 {
        return ctx.throw_type(&format!(
            "{what}: expected a 2-D matrix, got rank {}",
            dims.len()
        ));
    }
    Ok(dims)
}

/// `qr(a, "q" | "r")`. `a` is `[m, n]`, `k = min(m, n)`; `Q` is `[m, k]`,
/// `R` is `[k, n]`.
fn m_qr(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let a = node(ctx, args, 0, "qr")?;
    let part_label = to_string_or(ctx, arg(args, 1), "q")?;
    let g = graph_of(ctx, this)?;
    let dims = static_dims(ctx, g, a, "qr")?;
    let (m, n) = (dims[0], dims[1]);
    let k = m.min(n);
    let dtype = g.shape(a).dtype();
    let (part, out) = match part_label.trim().to_ascii_lowercase().as_str() {
        "q" => (QrPart::Q, Shape::new(&[m, k], dtype)),
        "r" => (QrPart::R, Shape::new(&[k, n], dtype)),
        other => return ctx.throw_type(&format!("qr part must be 'q' or 'r', got '{other}'")),
    };
    Ok(id(g.qr(a, part, out)))
}

/// `svd(a, "u" | "s" | "vt")` — one factor of the thin SVD.
fn m_svd(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let a = node(ctx, args, 0, "svd")?;
    let part_label = to_string_or(ctx, arg(args, 1), "s")?;
    let g = graph_of(ctx, this)?;
    let dims = static_dims(ctx, g, a, "svd")?;
    let (m, n) = (dims[0], dims[1]);
    let k = m.min(n);
    let dtype = g.shape(a).dtype();
    let (part, out) = match part_label.trim().to_ascii_lowercase().as_str() {
        "u" => (SvdPart::U, Shape::new(&[m, k], dtype)),
        "s" => (SvdPart::S, Shape::new(&[k], dtype)),
        "vt" | "v" => (SvdPart::Vt, Shape::new(&[k, n], dtype)),
        other => {
            return ctx.throw_type(&format!("svd part must be 'u', 's' or 'vt', got '{other}'"));
        }
    };
    Ok(id(g.svd(a, part, out)))
}

/// `scaledMatmul(lhs, rhs, format?, layout?)` — native FP8/FP6/FP4 GEMM.
/// `rhs` must already be K-last (`[n, k]`), the TN layout tensor cores want.
fn m_scaled_matmul(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let lhs = node(ctx, args, 0, "scaledMatmul")?;
    let rhs = node(ctx, args, 1, "scaledMatmul")?;
    let format_label = to_string_or(ctx, arg(args, 2), "f8e4m3")?;
    let format: rlx_ir::ScaledFormat = match format_label.parse() {
        Ok(f) => f,
        Err(e) => return ctx.throw_type(&format!("scaledMatmul format: {e}")),
    };
    let layout_label = to_string_or(ctx, arg(args, 3), "per_tensor")?;
    {
        let g = graph_of(ctx, this)?;
        check_nodes(ctx, g, &[lhs, rhs], "scaledMatmul")?;
    }
    let layout = match layout_label.trim().to_ascii_lowercase().as_str() {
        "per_tensor" | "pertensor" => rlx_ir::ScaleLayout::PerTensor,
        "mx" => rlx_ir::ScaleLayout::mx(),
        "nvfp4" => rlx_ir::ScaleLayout::nvfp4(),
        other => {
            return ctx.throw_type(&format!(
                "unknown scale layout '{other}' (per_tensor, mx, nvfp4)"
            ));
        }
    };
    Ok(id(
        graph_of(ctx, this)?.scaled_matmul(lhs, rhs, format, layout)
    ))
}

/// `loraMatmul(x, w, a, b, scale, outDims, dtype)` — `x·W + scale·(x·A)·B`.
fn m_lora_matmul(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let x = node(ctx, args, 0, "loraMatmul")?;
    let w = node(ctx, args, 1, "loraMatmul")?;
    let a = node(ctx, args, 2, "loraMatmul")?;
    let b = node(ctx, args, 3, "loraMatmul")?;
    let scale = to_f32(ctx, arg(args, 4))?;
    let shape = shape_arg(ctx, args, 5, 6)?;
    Ok(id(
        graph_of(ctx, this)?.lora_matmul(x, w, a, b, scale, shape)
    ))
}

/// `fusedMatmulBiasAct(x, w, bias, outDims, dtype, activation?)`.
fn m_fused_matmul_bias_act(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let x = node(ctx, args, 0, "fusedMatmulBiasAct")?;
    let w = node(ctx, args, 1, "fusedMatmulBiasAct")?;
    let bias = node(ctx, args, 2, "fusedMatmulBiasAct")?;
    let shape = shape_arg(ctx, args, 3, 4)?;
    let activation = if is_nullish(arg(args, 5)) {
        None
    } else {
        let label = ctx.to_rust_string(arg(args, 5))?;
        Some(parse_activation(ctx, &label)?)
    };
    Ok(id(
        graph_of(ctx, this)?.fused_matmul_bias_act(x, w, bias, activation, shape)
    ))
}

/// `customFn(inputs, fwdGraph, vjpGraph?, jvpGraph?)` — a sub-graph with
/// optional hand-written AD rules (JAX's `custom_vjp` / `custom_jvp`).
///
/// Every body graph passed in is **consumed**, like `Session.compile`.
fn m_custom_fn(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use crate::api::graph::take_graph;

    let inputs = to_u32_vec(ctx, arg(args, 0), "customFn inputs")?;
    let fwd = take_graph(ctx, arg(args, 1))?;
    let vjp = if is_nullish(arg(args, 2)) {
        None
    } else {
        Some(take_graph(ctx, arg(args, 2))?)
    };
    let jvp = if is_nullish(arg(args, 3)) {
        None
    } else {
        Some(take_graph(ctx, arg(args, 3))?)
    };
    let ids: Vec<NodeId> = inputs.into_iter().map(NodeId).collect();
    let g = graph_of(ctx, this)?;
    check_nodes(ctx, g, &ids, "customFn inputs")?;
    Ok(id(g.custom_fn(ids, fwd, vjp, jvp)))
}

// ── graph introspection ─────────────────────────────────────

/// Every `Param` name in the graph — what `setParams` has to cover.
fn m_param_names(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = rlx_runtime::graph_param_names(graph_of(ctx, this)?);
    Ok(new_string_array(ctx, &names))
}

/// Distinct op kinds present, sorted — a one-line answer to "why won't this
/// backend take my graph".
fn m_op_kinds(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let mut kinds: Vec<String> = graph_of(ctx, this)?
        .nodes()
        .iter()
        .map(|n| format!("{:?}", n.op.kind()))
        .collect();
    kinds.sort();
    kinds.dedup();
    Ok(new_string_array(ctx, &kinds))
}

/// Serialize to the graph-JSON that `rlx-check` and the Rust tools read.
///
/// Behind the `graph-json` feature: it needs `rlx-ir/serialize`, and turning
/// that on here would switch it on for every crate in a `--workspace` build
/// through feature unification.
#[cfg(feature = "graph-json")]
fn m_to_json(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let json = match serde_json::to_string(&*graph_of(ctx, this)?) {
        Ok(text) => text,
        Err(e) => return ctx.throw_internal(&format!("toJSON: {e}")),
    };
    Ok(new_string(ctx, &json))
}

// ── installation ────────────────────────────────────────────

pub fn install(ctx: &mut Context, proto: &Gc<JsObject>) {
    install_ops(ctx, proto);

    let methods: &[(&str, NativeFn, u32)] = &[
        ("pad", m_pad, 4),
        ("qr", m_qr, 2),
        ("svd", m_svd, 2),
        ("scaledMatmul", m_scaled_matmul, 4),
        ("loraMatmul", m_lora_matmul, 7),
        ("fusedMatmulBiasAct", m_fused_matmul_bias_act, 6),
        ("customFn", m_custom_fn, 4),
        ("paramNames", m_param_names, 0),
        ("opKinds", m_op_kinds, 0),
    ];
    for (name, func, arity) in methods {
        ctx.define_method(proto, name, *func, *arity, 0);
    }
    #[cfg(feature = "graph-json")]
    ctx.define_method(proto, "toJSON", m_to_json, 0, 0);
}
