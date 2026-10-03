// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.Graph` — the symbolic IR builder.
//!
//! Every method returns an integer node id, so JavaScript composes graphs the
//! same way the Rust builder does:
//!
//! ```js
//! const g = new rlx.Graph("mlp");
//! const x = g.input("x", [1, 4], "f32");
//! const w = g.param("w", [4, 2], "f32");
//! g.setOutputs([g.relu(g.matmul(x, w))]);
//! ```
//!
//! Argument conversion always happens *before* the graph handle is borrowed:
//! converting can run script (a getter on an array-like), and script could
//! re-enter this same graph. See the aliasing note on [`crate::handle`].

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_ir::fft::FftNorm;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::convert::*;
use crate::handle::{self, CLASS_GRAPH};

/// A graph, or the hole left after `Session.compile` moved it out.
pub struct GraphSlot {
    pub inner: Option<Graph>,
}

impl GraphSlot {
    pub fn new(graph: Graph) -> Self {
        Self { inner: Some(graph) }
    }
}

/// `&mut Graph` behind `this`, or a `TypeError`.
pub(crate) fn graph_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, this, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.as_mut() {
        Some(g) => Ok(g),
        None => ctx.throw_type(
            "this Graph was consumed by Session.compile() — build a new one, or call \
             Session.compile on a fresh graph",
        ),
    }
}

/// Read-only view, for `grad` / `deviceReport` which borrow rather than consume.
pub fn graph_ref<'a>(ctx: &mut Context, v: &Value) -> JsResult<&'a Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, v, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.as_ref() {
        Some(g) => Ok(g),
        None => ctx.throw_type("Graph has already been consumed by Session.compile()"),
    }
}

/// Move the graph out — `Session.compile` takes ownership, as rlx-runtime does.
pub fn take_graph(ctx: &mut Context, v: &Value) -> JsResult<Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, v, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.take() {
        Some(g) => Ok(g),
        None => ctx.throw_type("Graph has already been consumed by Session.compile()"),
    }
}

pub fn wrap_graph(ctx: &mut Context, graph: Graph) -> Value {
    handle::wrap(ctx, CLASS_GRAPH, GraphSlot::new(graph))
}

// ── string → enum ───────────────────────────────────────────

pub fn parse_dtype(ctx: &mut Context, s: &str) -> JsResult<DType> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f64" | "float64" | "double" => DType::F64,
        "i8" | "int8" => DType::I8,
        "u8" | "uint8" => DType::U8,
        "i16" | "int16" => DType::I16,
        "i32" | "int32" => DType::I32,
        "u32" | "uint32" => DType::U32,
        "i64" | "int64" => DType::I64,
        "bool" => DType::Bool,
        "c64" | "complex64" => DType::C64,
        "c128" | "complex128" => DType::C128,
        other => return ctx.throw_type(&format!("unknown dtype '{other}' (f32, f16, i32, …)")),
    })
}

pub fn dtype_label(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F64 => "f64",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C64 => "c64",
        DType::C128 => "c128",
    }
}

pub(crate) fn parse_activation(ctx: &mut Context, s: &str) -> JsResult<Activation> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "gelu" => Activation::Gelu,
        "gelu_approx" | "geluapprox" => Activation::GeluApprox,
        "silu" => Activation::Silu,
        "relu" => Activation::Relu,
        "sigmoid" => Activation::Sigmoid,
        "tanh" => Activation::Tanh,
        "exp" => Activation::Exp,
        "log" => Activation::Log,
        "sqrt" => Activation::Sqrt,
        "rsqrt" => Activation::Rsqrt,
        "neg" => Activation::Neg,
        "abs" => Activation::Abs,
        "round" => Activation::Round,
        other => return ctx.throw_type(&format!("unknown activation '{other}'")),
    })
}

pub(crate) fn parse_binop(ctx: &mut Context, s: &str) -> JsResult<BinaryOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "add" | "+" => BinaryOp::Add,
        "sub" | "-" => BinaryOp::Sub,
        "mul" | "*" => BinaryOp::Mul,
        "div" | "/" => BinaryOp::Div,
        "max" => BinaryOp::Max,
        "min" => BinaryOp::Min,
        "pow" => BinaryOp::Pow,
        other => return ctx.throw_type(&format!("unknown binary op '{other}'")),
    })
}

pub(crate) fn parse_cmp(ctx: &mut Context, s: &str) -> JsResult<CmpOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "eq" | "==" => CmpOp::Eq,
        "ne" | "!=" => CmpOp::Ne,
        "lt" | "<" => CmpOp::Lt,
        "le" | "<=" => CmpOp::Le,
        "gt" | ">" => CmpOp::Gt,
        "ge" | ">=" => CmpOp::Ge,
        other => return ctx.throw_type(&format!("unknown comparison '{other}'")),
    })
}

pub(crate) fn parse_reduce(ctx: &mut Context, s: &str) -> JsResult<ReduceOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "sum" => ReduceOp::Sum,
        "mean" => ReduceOp::Mean,
        "max" => ReduceOp::Max,
        "min" => ReduceOp::Min,
        "prod" => ReduceOp::Prod,
        other => return ctx.throw_type(&format!("unknown reduction '{other}'")),
    })
}

pub(crate) fn parse_mask_kind(ctx: &mut Context, s: &str) -> JsResult<MaskKind> {
    let lower = s.trim().to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("sliding:") {
        return match rest.parse::<usize>() {
            Ok(n) => Ok(MaskKind::SlidingWindow(n)),
            Err(_) => ctx.throw_type(&format!(
                "sliding mask needs a window: 'sliding:4096', got '{s}'"
            )),
        };
    }
    Ok(match lower.as_str() {
        "none" => MaskKind::None,
        "causal" => MaskKind::Causal,
        other => {
            return ctx.throw_type(&format!(
                "unknown mask kind '{other}' (none, causal, sliding:<N>)"
            ));
        }
    })
}

// ── argument readers ────────────────────────────────────────

pub(crate) fn node(ctx: &mut Context, args: &[Value], i: usize, what: &str) -> JsResult<NodeId> {
    let n = ctx.to_number(arg(args, i))?;
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return ctx.throw_type(&format!(
            "{what}: expected a node id (a number returned by another Graph method), got {n}"
        ));
    }
    Ok(NodeId(n as u32))
}

pub(crate) fn shape_arg(
    ctx: &mut Context,
    args: &[Value],
    dims_at: usize,
    dtype_at: usize,
) -> JsResult<Shape> {
    let dims = to_usize_vec(ctx, arg(args, dims_at), "shape")?;
    let label = to_string_or(ctx, arg(args, dtype_at), "f32")?;
    let dtype = parse_dtype(ctx, &label)?;
    Ok(Shape::new(&dims, dtype))
}

/// Reject a node id the graph does not have.
///
/// Without this, `x.add(2)` on a two-node graph **panicked the host process**:
/// the id went straight into `Graph::shape`, which indexes. An embedded engine
/// must not let a script abort its host, and a `RangeError` is the right answer
/// for an out-of-range index.
pub(crate) fn check_node(ctx: &mut Context, graph: &Graph, id: NodeId, what: &str) -> JsResult<()> {
    let count = graph.nodes().len();
    if (id.0 as usize) < count {
        return Ok(());
    }
    ctx.throw_range(&format!(
        "{what}: node id {} does not exist in this graph (it has {count} node(s)).          Scalars are not node ids — wrap one with `constant(v)` first.",
        id.0
    ))
}

/// [`check_node`] over several ids at once.
pub(crate) fn check_nodes(
    ctx: &mut Context,
    graph: &Graph,
    ids: &[NodeId],
    what: &str,
) -> JsResult<()> {
    for id in ids {
        check_node(ctx, graph, *id, what)?;
    }
    Ok(())
}

pub(crate) fn id(n: NodeId) -> Value {
    Value::number(n.0 as f64)
}

// ── methods ─────────────────────────────────────────────────

fn m_input(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let shape = shape_arg(ctx, args, 1, 2)?;
    Ok(id(graph_of(ctx, this)?.input(name, shape)))
}

fn m_param(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let shape = shape_arg(ctx, args, 1, 2)?;
    Ok(id(graph_of(ctx, this)?.param(name, shape)))
}

fn m_constant(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let value = to_f64(ctx, arg(args, 0))?;
    let label = to_string_or(ctx, arg(args, 1), "f32")?;
    let dtype = parse_dtype(ctx, &label)?;
    let g = graph_of(ctx, this)?;
    match g.try_constant(value, dtype) {
        Ok(n) => Ok(id(n)),
        Err(e) => ctx.throw_range(&e),
    }
}

fn m_set_outputs(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids: Vec<NodeId> = to_u32_vec(ctx, arg(args, 0), "setOutputs")?
        .into_iter()
        .map(NodeId)
        .collect();
    let g = graph_of(ctx, this)?;
    check_nodes(ctx, g, &ids, "setOutputs")?;
    g.set_outputs(ids);
    Ok(Value::Undefined)
}

fn m_outputs(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let outs: Vec<Value> = graph_of(ctx, this)?
        .outputs
        .iter()
        .map(|n| Value::number(n.0 as f64))
        .collect();
    Ok(new_array(ctx, outs))
}

/// `{dims: [...], dtype: "f32"}`. A dynamic dim reads back as `-1`.
fn m_shape_of(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let n = node(ctx, args, 0, "shapeOf")?;
    let (dims, label) = {
        let g = graph_of(ctx, this)?;
        check_node(ctx, g, n, "shapeOf")?;
        let shape = g.shape(n);
        let dims: Vec<Value> = shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(v) => Value::number(*v as f64),
                rlx_ir::Dim::Dynamic(_) => Value::number(-1.0),
            })
            .collect();
        (dims, dtype_label(shape.dtype()))
    };
    let dims = new_array(ctx, dims);
    let dtype = new_string(ctx, label);
    Ok(new_object(ctx, vec![("dims", dims), ("dtype", dtype)]))
}

fn m_node_count(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(Value::number(graph_of(ctx, this)?.nodes().len() as f64))
}

fn m_name(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let name = graph_of(ctx, this)?.name.clone();
    Ok(new_string(ctx, &name))
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let g = graph_of(ctx, this)?;
        format!("[rlx.Graph {} nodes={}]", g.name, g.nodes().len())
    };
    Ok(new_string(ctx, &text))
}

// ── the declarative table ───────────────────────────────────
//
// One line per IR op: JS name, typed parameters with JS-style defaults, and
// the builder call. `graph_ops!` reads every argument before borrowing the
// graph and derives `Function.length` from the signature, so neither can
// drift from the other.

fn install_ops(ctx: &mut Context, proto: &Gc<JsObject>) {
    graph_ops! {
        ctx, proto;

        // ── linear algebra ──
        /// Shape-inferred matmul. Prefer this over `matmulWithShape`.
        "matmul"(a: node, b: node) => |g| g.mm(a, b);
        /// Explicit output shape — for a layout inference would not pick.
        "matmulWithShape"(a: node, b: node, dims: usizes, dtype: dtype = DType::F32)
            => |g| g.matmul(a, b, Shape::new(&dims, dtype));
        /// `x = A⁻¹·b`. `A` is `[N, N]`; `b` is `[N]` or `[N, K]`, and `x`
        /// takes `b`'s shape.
        "denseSolve"(a: node, b: node) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(b));
            g.dense_solve(a, b, shape)
        };

        // ── elementwise ──
        "add"(a: node, b: node) => |g| g.add(a, b);
        "sub"(a: node, b: node) => |g| g.sub(a, b);
        "mul"(a: node, b: node) => |g| g.mul(a, b);
        "div"(a: node, b: node) => |g| g.div(a, b);
        /// `binary("pow" | "max" | "min", a, b)` — the ops without a shorthand.
        "binary"(op: binop, a: node, b: node) => |g, ctx| {
            match op {
                BinaryOp::Add => g.add(a, b),
                BinaryOp::Sub => g.sub(a, b),
                BinaryOp::Mul => g.mul(a, b),
                BinaryOp::Div => g.div(a, b),
                other => {
                    let shape = match rlx_ir::shape::binary_shape(g.shape(a), g.shape(b)) {
                        Ok(s) => s,
                        Err(e) => return ctx.throw_type(&format!("binary({other:?}) shape: {e}")),
                    };
                    g.binary(other, a, b, shape)
                }
            }
        };

        /// `activation("sigmoid" | "rsqrt" | …, x)` — the full set by name.
        "activation"(kind: activation, x: node) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(x));
            g.activation(kind, x, shape)
        };
        "gelu"(x: node) => |g| g.gelu(x);
        /// Tanh-approximation GELU — what most ViT and GPT-2 checkpoints used.
        "geluApprox"(x: node) => |g| g.gelu_approx(x);
        "silu"(x: node) => |g| g.silu(x);
        "relu"(x: node) => |g| g.relu(x);
        "exp"(x: node) => |g| g.exp(x);
        "sqrt"(x: node) => |g| g.sqrt(x);
        "neg"(x: node) => |g| g.neg(x);
        "tanh"(x: node) => |g| g.tanh(x);
        /// Identity forward, zero backward (`jax.lax.stop_gradient` / `detach`).
        "stopGradient"(x: node) => |g| g.stop_gradient(x);

        /// `compare("lt" | "<" | …, a, b)` → a Bool tensor.
        "compare"(op: cmp_op, a: node, b: node) => |g, ctx| {
            let shape = match rlx_ir::shape::compare_shape(g.shape(a), g.shape(b)) {
                Ok(s) => s,
                Err(e) => return ctx.throw_type(&format!("compare shape: {e}")),
            };
            g.add_node(Op::Compare(op), vec![a, b], shape)
        };
        "where"(cond: node, a: node, b: node) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(a));
            g.add_node(Op::Where, vec![cond, a, b], shape)
        };

        // ── reduction ──
        "reduce"(x: node, op: reduce_op, axes: usizes, keepDim: bool = false) => |g, ctx| {
            let shape = match rlx_ir::shape::reduce_shape(g.shape(x), &axes, keepDim) {
                Ok(s) => s,
                Err(e) => return ctx.throw_type(&format!("reduce shape: {e}")),
            };
            g.reduce(x, op, axes, keepDim, shape)
        };
        "sum"(x: node, axes: usizes, keepDim: bool = false) => |g| g.sum(x, axes, keepDim);
        "mean"(x: node, axes: usizes, keepDim: bool = false) => |g| g.mean(x, axes, keepDim);
        "softmax"(x: node, axis: i32 = -1) => |g| g.sm(x, axis);
        "cumsum"(x: node, axis: i32, exclusive: bool = false) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(x));
            g.cumsum(x, axis, exclusive, shape)
        };

        // ── shape ──
        "reshape"(x: node, dims: i64s) => |g| g.reshape_(x, dims);
        "transpose"(x: node, perm: usizes) => |g| g.transpose_(x, perm);
        "narrow"(x: node, axis: usize, start: usize, len: usize) => |g| g.narrow_(x, axis, start, len);
        "concat"(inputs: nodes, axis: usize) => |g| g.concat_(inputs, axis);
        "gather"(table: node, indices: node, axis: usize) => |g| g.gather_(table, indices, axis);
        "cast"(x: node, to: dtype) => |g| g.cast(x, to);

        // ── normalization ──
        "layerNorm"(x: node, gamma: node, beta: node, axis: i32 = -1, eps: f32 = 1e-5) => |g| {
            let shape = rlx_ir::shape::unary_shape(g.shape(x));
            g.layer_norm(x, gamma, beta, axis, eps, shape)
        };
        "rmsNorm"(x: node, gamma: node, beta: node, eps: f32 = 1e-5)
            => |g| g.rms_norm(x, gamma, beta, eps);
        "groupNorm"(x: node, gamma: node, beta: node, numGroups: usize, eps: f32 = 1e-5)
            => |g| g.group_norm(x, gamma, beta, numGroups, eps);

        // ── attention ──
        /// Explicit mask tensor. `MaskKind::Custom` is `[batch, keyLen]` ONLY —
        /// a per-query mask is a `Bias` tensor, not this.
        "attention"(q: node, k: node, v: node, mask: node, numHeads: usize, headDim: usize)
            => |g| g.attention_(q, k, v, mask, numHeads, headDim);
        /// Kernel-synthesized mask: `"none"`, `"causal"`, `"sliding:<N>"`. No
        /// mask tensor is allocated.
        "attentionKind"(q: node, k: node, v: node, numHeads: usize, headDim: usize, maskKind: mask = MaskKind::Causal)
            => |g| {
                let shape = rlx_ir::shape::attention_shape(g.shape(q));
                g.attention_kind(q, k, v, numHeads, headDim, maskKind, shape)
            };

        // ── FFT ──
        "fft"(x: node) => |g| g.fft(x, false);
        "ifft"(x: node) => |g| g.fft(x, true);
        /// Real-input FFT → `[re, im]` half spectrum, which `irfft` takes back.
        "rfft"(x: node, norm: fft_norm = FftNorm::Backward) -> pair => |g| g.rfft(x, norm);
        "irfft"(re: node, im: node, n: usize, norm: fft_norm = FftNorm::Backward)
            => |g| g.irfft(re, im, n, norm);
    }
}

// ── ops with a shape the table does not model ───────────────

/// `rope(x, cos, sin, headDim, nRot?)`.
///
/// `nRot` defaults to the full head. Partial rotary is opt-in because its
/// table stride is `nRot/2`, not `headDim/2` — getting that wrong is a bug
/// three backends once agreed on.
fn m_rope(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let x = node(ctx, args, 0, "rope")?;
    let cos = node(ctx, args, 1, "rope")?;
    let sin = node(ctx, args, 2, "rope")?;
    let head_dim = to_usize(ctx, arg(args, 3), "rope headDim")?;
    {
        let g = graph_of(ctx, this)?;
        check_nodes(ctx, g, &[x, cos, sin], "rope")?;
    }
    if is_nullish(arg(args, 4)) {
        return Ok(id(graph_of(ctx, this)?.rope(x, cos, sin, head_dim)));
    }
    let n_rot = to_usize(ctx, arg(args, 4), "rope nRot")?;
    Ok(id(graph_of(ctx, this)?.rope_n(x, cos, sin, head_dim, n_rot)))
}

/// A `[h, w]` option, defaulted, also accepting a scalar for a square value.
fn pair_option(
    ctx: &mut Context,
    opts: &Value,
    name: &str,
    default: [usize; 2],
) -> JsResult<[usize; 2]> {
    let v = field(ctx, opts, name)?;
    if is_nullish(&v) {
        return Ok(default);
    }
    if let Value::Int(_) | Value::Float(_) = v {
        let n = to_usize(ctx, &v, name)?;
        return Ok([n, n]);
    }
    let dims = to_usize_vec(ctx, &v, name)?;
    match dims.len() {
        1 => Ok([dims[0], dims[0]]),
        2 => Ok([dims[0], dims[1]]),
        n => ctx.throw_type(&format!("conv2d {name}: expected 1 or 2 values, got {n}")),
    }
}

/// `conv2d(x, w, {kernelSize, stride, padding, dilation, groups})` — NCHW,
/// output shape inferred.
fn m_conv2d(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let x = node(ctx, args, 0, "conv2d")?;
    let w = node(ctx, args, 1, "conv2d")?;
    let opts = arg(args, 2).clone();
    let kernel = pair_option(ctx, &opts, "kernelSize", [1, 1])?;
    let stride = pair_option(ctx, &opts, "stride", [1, 1])?;
    let padding = pair_option(ctx, &opts, "padding", [0, 0])?;
    let dilation = pair_option(ctx, &opts, "dilation", [1, 1])?;
    let groups_v = field(ctx, &opts, "groups")?;
    let groups = if is_nullish(&groups_v) {
        1
    } else {
        to_usize(ctx, &groups_v, "conv2d groups")?
    };
    Ok(id(
        graph_of(ctx, this)?.conv2d(x, w, kernel, stride, padding, dilation, groups)
    ))
}

/// `sample(logits, {topK, topP, temperature, seed, shape, dtype})` — fused
/// logits → token id, so sampling never leaves the device.
fn m_sample(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = node(ctx, args, 0, "sample")?;
    let opts = arg(args, 1).clone();
    let top_k_v = field(ctx, &opts, "topK")?;
    let top_k = if is_nullish(&top_k_v) {
        0
    } else {
        to_usize(ctx, &top_k_v, "sample topK")?
    };
    let top_p_v = field(ctx, &opts, "topP")?;
    let top_p = if is_nullish(&top_p_v) {
        1.0
    } else {
        to_f32(ctx, &top_p_v)?
    };
    let temp_v = field(ctx, &opts, "temperature")?;
    let temperature = if is_nullish(&temp_v) {
        1.0
    } else {
        to_f32(ctx, &temp_v)?
    };
    let seed_v = field(ctx, &opts, "seed")?;
    let seed = if is_nullish(&seed_v) {
        0
    } else {
        to_u64(ctx, &seed_v, "sample seed")?
    };
    let shape_v = field(ctx, &opts, "shape")?;
    let dims = to_usize_vec(ctx, &shape_v, "sample shape")?;
    let dtype_v = field(ctx, &opts, "dtype")?;
    let label = to_string_or(ctx, &dtype_v, "i32")?;
    let dtype = parse_dtype(ctx, &label)?;
    let shape = Shape::new(&dims, dtype);
    Ok(id(graph_of(ctx, this)?.sample(
        logits,
        top_k,
        top_p,
        temperature,
        seed,
        shape,
    )))
}

// ── installation ────────────────────────────────────────────

/// Build the `Graph` prototype and the `rlx.Graph` constructor.
pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    let proto = js_class! {
        ctx, namespace;
        name: "Graph",
        class: CLASS_GRAPH,
        ctor: construct,
        methods: {
            "input" => m_input, 3;
            "param" => m_param, 3;
            "constant" => m_constant, 2;
            "setOutputs" => m_set_outputs, 1;
            "outputs" => m_outputs, 0;
            "shapeOf" => m_shape_of, 1;
            "nodeCount" => m_node_count, 0;
            "name" => m_name, 0;
            "toString" => m_to_string, 0;
            "rope" => m_rope, 5;
            "conv2d" => m_conv2d, 3;
            "sample" => m_sample, 2;
        }
    };
    install_ops(ctx, &proto);
    // The rest of the IR surface — linalg, DSP, quantization, shape
    // manipulation — hangs off the same prototype from its own module.
    crate::api::graph_ext::install(ctx, &proto);
}

/// `new rlx.Graph("name")` — also callable without `new`.
fn construct(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = to_string_or(ctx, arg(args, 0), "graph")?;
    Ok(wrap_graph(ctx, Graph::new(name)))
}
