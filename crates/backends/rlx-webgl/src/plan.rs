// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph → [`Plan`]: a flat, slot-indexed op list with all index arithmetic
//! (gather / reduce groups) precomputed in plain Rust so the CPU and WebGL
//! executors run byte-for-byte the same lowering.

use crate::{Result, WebglError};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Dim, Graph, NodeId, Op, OpKind};
use std::collections::{HashMap, HashSet};

/// Pointwise activation kinds rlx-webgl can evaluate (forward + backward).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Relu,
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Sigmoid,
    Tanh,
    Abs,
    Sin,
    Cos,
    Silu,
    Recip,
    /// `x·Φ(x)`, with `erf` from Abramowitz–Stegun 7.1.26 (|err| ≤ 1.5e-7;
    /// GLSL has no `erf`).
    Gelu,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
    GeluApprox,
    /// `ln(1 + eˣ)`, as `max(x, 0) + ln(1 + e^-|x|)` so it never overflows.
    Softplus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Pow,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Atan2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Red {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

#[derive(Clone, Debug)]
pub enum LeafSource {
    Input(String),
    Param(String),
    Const(Vec<f32>),
}

/// One executable step. `out`/`a`/`b`/… are *slot* indices into the value
/// table (one slot per graph node).
#[derive(Clone, Debug)]
pub enum Step {
    Leaf {
        out: usize,
        src: LeafSource,
    },
    /// Pointwise activation `f(a)`.
    Unary {
        out: usize,
        a: usize,
        act: Act,
    },
    /// Pointwise activation backward `dy * f'(x)` (inputs `[x, dy]`).
    ActBack {
        out: usize,
        x: usize,
        dy: usize,
        act: Act,
    },
    /// Element-wise binary; inputs share `out`'s element count.
    Binary {
        out: usize,
        a: usize,
        b: usize,
        op: Bin,
    },
    /// Element-wise comparison → 1.0 / 0.0.
    Compare {
        out: usize,
        a: usize,
        b: usize,
        cmp: Cmp,
    },
    /// `cond != 0 ? a : b` (inputs `[cond, a, b]`).
    Where {
        out: usize,
        cond: usize,
        a: usize,
        b: usize,
    },
    /// `(m×k) · (k×n)` row-major, `batch` times: block `i` of `a` (and of
    /// `b` when `b_batched`, else the one `b`) gives block `i` of `out`.
    MatMul {
        out: usize,
        a: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
        b_batched: bool,
    },
    /// `out[i] = src[idx[i]]` — Reshape / Transpose / Expand / Narrow / Reverse / Cast(f32).
    Gather {
        out: usize,
        src: usize,
        idx: Vec<u32>,
    },
    /// `out[i] = reduce_op_j src[groups[i*fanin + j]]` (padding = `u32::MAX`).
    Reduce {
        out: usize,
        src: usize,
        groups: Vec<u32>,
        fanin: usize,
        op: Red,
    },
    /// Softmax over the last axis (`cols`).
    Softmax {
        out: usize,
        a: usize,
        rows: usize,
        cols: usize,
    },
    /// LayerNorm over the last axis: `(x − mean)/√(var+eps) · gamma + beta`.
    LayerNorm {
        out: usize,
        x: usize,
        gamma: usize,
        beta: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    },
    /// RmsNorm over the last axis: `x/√(mean(x²)+eps) · gamma + beta`.
    RmsNorm {
        out: usize,
        x: usize,
        gamma: usize,
        beta: usize,
        rows: usize,
        cols: usize,
        eps: f32,
    },
    /// ArgMax/ArgMin over one axis → f32-encoded index of the extreme element.
    /// `groups[oi*fanin + j]` = src linear index of axis position `j`.
    ArgReduce {
        out: usize,
        src: usize,
        groups: Vec<u32>,
        fanin: usize,
        is_max: bool,
    },
    /// Embedding-style gather with a runtime index tensor:
    /// `out[o] = table[base[o] + round(indices[which[o]]) * axis_stride]`.
    GatherRuntime {
        out: usize,
        table: usize,
        indices: usize,
        which: Vec<u32>,
        base: Vec<u32>,
        axis_stride: u32,
    },
    /// Standalone complex `Op::Cast` lane-move on the f32-uniform arena
    /// (real↔C64, real↔C128, C64↔C128). Representation:
    ///   C64  = 2 f32 lanes `[re, im]`;
    ///   C128 = 4 f32 lanes df64 `[re_hi, re_lo, im_hi, im_lo]`.
    /// Every real→complex source is an f32 real (lo=0), so all six directions
    /// are pure lane MOVES — no compensated df64 arithmetic. `mode` 0..5 matches
    /// the wgpu/cuda/vulkan `complex_cast` table (see the executors). `n` is the
    /// complex-element count (== the real-element count on the real side). A
    /// same-dtype complex cast is a plain lane-preserving [`Step::Gather`]
    /// identity, not this step.
    ComplexCast {
        out: usize,
        src: usize,
        mode: u32,
        n: usize,
    },
    /// C64 element-wise binary (Add/Sub/Mul/Div). Each output complex element
    /// reads BOTH `[re, im]` lanes of its operands — the reason it cannot ride
    /// the scalar-per-lane `Binary` path. Broadcast is per-operand complex-
    /// element modulo (`k % n_a`, `k % n_b`), matching the rlx-cpu modulo
    /// fallback. Max/Min/Pow and all C128 arithmetic are rejected at lowering
    /// (identical to rlx-cpu). `n` = output complex-element count.
    BinaryC64 {
        out: usize,
        a: usize,
        b: usize,
        op: Bin,
        n: usize,
        n_a: usize,
        n_b: usize,
    },
    /// Host/transport custom op (`collective.*`): the single f32 `input` is
    /// staged to a registered CPU kernel via
    /// `rlx_cpu::op_registry::run_f32_custom_op_host`. There is no GPU/GLSL
    /// mirror — a fragment shader can't drive a process group — so this step
    /// only runs on the *native* CPU executor. On wasm (no TCP transport) it
    /// errors. See [`crate::exec_cpu`].
    Custom {
        out: usize,
        input: usize,
        name: String,
        attrs: Vec<u8>,
    },
}

/// A lowered graph ready for either executor.
#[derive(Clone)]
pub struct Plan {
    pub steps: Vec<Step>,
    /// `(rows, cols)` per slot — the WebGL texture dimensions.
    pub slot_dims: Vec<(usize, usize)>,
    /// Elements per slot: `rows * cols`, except for a slot laid out wide
    /// (`widen_tall_slots`), whose last texture row is padding.
    pub slot_len: Vec<usize>,
    /// Output slots, in `graph.outputs` order.
    pub outputs: Vec<usize>,
    /// The IR op (kind and shape) each slot was allocated for — for errors.
    pub slot_origin: Vec<String>,
}

pub(crate) const PAD: u32 = u32::MAX;

/// The IR ops this backend can lower (the WebGL2 render-to-texture surface).
/// Mirrors the `Backend::supported_ops` convention used by the native backends.
pub fn supported_ops() -> &'static [OpKind] {
    &[
        OpKind::Input,
        OpKind::Param,
        OpKind::Constant,
        OpKind::Cast,
        OpKind::Activation,
        OpKind::ActivationBackward,
        OpKind::ReluBackward,
        OpKind::Binary,
        OpKind::Compare,
        OpKind::Where,
        OpKind::MatMul,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Narrow,
        OpKind::Expand,
        OpKind::Reverse,
        OpKind::Reduce,
        OpKind::Softmax,
        OpKind::LayerNorm,
        OpKind::RmsNorm,
        // DiT modulation — claimed for fusion; expanded by
        // `unfuse_dit_modulation` before `lower`.
        OpKind::AdaLayerNorm,
        OpKind::GatedResidual,
        OpKind::StopGradient,
        OpKind::Cumsum,
        OpKind::Concat,
        OpKind::Pool,
        OpKind::Conv,
        OpKind::Im2Col,
        OpKind::ArgMax,
        OpKind::ArgMin,
        OpKind::Gather,
        OpKind::Rope,
        // Host/transport custom ops (`collective.*`). Kept in the supported set
        // so legalization preserves the node; `lower` restricts to the
        // collective names and the CPU executor host-delegates them (native
        // only — see `exec_cpu`).
        OpKind::Custom,
    ]
}

/// `collective.*` op names this backend host-delegates (mirrors
/// `rlx_collectives::{ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, COPY_TO_PARALLEL,
/// REDUCE_FROM_PARALLEL}`, which rlx-webgl cannot depend on — later publish
/// tier — so keep them in sync). Every other `Op::Custom` name is rejected
/// (rlx-webgl is f32-only and has no other host kernels wired).
pub(crate) const COLLECTIVE_OPS: &[&str] = &[
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    "collective.copy_to_parallel",
    "collective.reduce_from_parallel",
];

/// Per-output groups for ArgMax/ArgMin: `groups[o*axsz + p]` = src linear index
/// of axis position `p` for output element `o` (axis-ordered, so the winning `p`
/// is the index to emit).
fn arg_groups(in_dims: &[usize], out_dims: &[usize], axis: usize, keep_dim: bool) -> Vec<u32> {
    let axsz = in_dims[axis];
    let in_strides = strides(in_dims);
    let n_out = total(out_dims);
    let mut groups = vec![0u32; n_out * axsz];
    for o in 0..n_out {
        let oc = unravel(o, out_dims);
        let mut ic = vec![0usize; in_dims.len()];
        if keep_dim {
            for (k, &c) in oc.iter().enumerate() {
                ic[k] = if k == axis { 0 } else { c };
            }
        } else {
            let mut j = 0;
            for (k, slot) in ic.iter_mut().enumerate() {
                if k != axis {
                    *slot = oc[j];
                    j += 1;
                }
            }
        }
        for p in 0..axsz {
            ic[axis] = p;
            groups[o * axsz + p] = ravel(&ic, &in_strides) as u32;
        }
    }
    groups
}

/// Normalize a possibly-negative axis and require it to be the last one.
fn require_last_axis(axis: i32, rank: usize, op: &str) -> Result<()> {
    let ax = if axis < 0 { rank as i32 + axis } else { axis };
    if rank == 0 || ax != rank as i32 - 1 {
        return Err(WebglError(format!(
            "{op} supported only over the last axis (got axis {axis} of rank {rank})"
        )));
    }
    Ok(())
}

fn map_act(a: Activation) -> Result<Act> {
    Ok(match a {
        Activation::Relu => Act::Relu,
        Activation::Neg => Act::Neg,
        Activation::Exp => Act::Exp,
        Activation::Log => Act::Log,
        Activation::Sqrt => Act::Sqrt,
        Activation::Rsqrt => Act::Rsqrt,
        Activation::Sigmoid => Act::Sigmoid,
        Activation::Tanh => Act::Tanh,
        Activation::Abs => Act::Abs,
        Activation::Sin => Act::Sin,
        Activation::Cos => Act::Cos,
        Activation::Silu => Act::Silu,
        Activation::Recip => Act::Recip,
        Activation::Gelu => Act::Gelu,
        Activation::GeluApprox => Act::GeluApprox,
        Activation::Softplus => Act::Softplus,
        other => return Err(WebglError(format!("activation {other:?} not supported"))),
    })
}

fn dims_of(shape: &rlx_ir::Shape) -> Result<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Ok(*n),
            Dim::Dynamic(s) => Err(WebglError(format!("dynamic dim ?{s} unsupported on WebGL"))),
        })
        .collect()
}

/// f32 lanes per logical element for a dtype on the f32-uniform WebGL arena.
/// Real / int / bool are widened to exactly ONE f32 lane; complex is simulated
/// with multiple lanes — C64 = 2 `[re, im]`, C128 = 4 df64
/// `[re_hi, re_lo, im_hi, im_lo]`. This is a lane COUNT, deliberately NOT
/// `size_bytes()/4` (which would wrongly give I64 two lanes and Bool zero).
pub(crate) fn lanes(dt: DType) -> usize {
    match dt {
        DType::C64 => 2,
        DType::C128 => 4,
        _ => 1,
    }
}

/// Flatten a logical shape to a 2D texture footprint `(rows, cols)`, preserving
/// row-major element order (cols = last dim).
impl Plan {
    /// The largest texture side this plan needs: every tensor is one
    /// `rows × cols` texture, and WebGL caps each side at `MAX_TEXTURE_SIZE`
    /// (16384 on most desktop browsers). Index tables are laid out wide and do
    /// not count.
    pub fn max_texture_side(&self) -> usize {
        self.largest_texture().map_or(0, |(_, r, c)| r.max(c))
    }

    /// `(origin, rows, cols)` of the slot with the longest texture side.
    pub fn largest_texture(&self) -> Option<(&str, usize, usize)> {
        let (i, &(r, c)) = self
            .slot_dims
            .iter()
            .enumerate()
            .max_by_key(|(_, (r, c))| (*r).max(*c))?;
        let origin = self.slot_origin.get(i).map_or("?", String::as_str);
        Some((origin, r, c))
    }
}

pub(crate) fn rows_cols(dims: &[usize]) -> (usize, usize) {
    if dims.is_empty() {
        return (1, 1);
    }
    let cols = (*dims.last().unwrap()).max(1);
    let total: usize = dims.iter().product();
    let rows = (total / cols).max(1);
    (rows, cols)
}

fn total(dims: &[usize]) -> usize {
    if dims.is_empty() {
        1
    } else {
        dims.iter().product()
    }
}

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

fn unravel(mut i: usize, dims: &[usize]) -> Vec<usize> {
    let st = strides(dims);
    let mut c = vec![0usize; dims.len()];
    for ax in 0..dims.len() {
        c[ax] = i / st[ax];
        i %= st[ax];
    }
    c
}

fn ravel(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides).map(|(c, s)| c * s).sum()
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Conv/pool spatial output size along one axis.
fn conv_out(sz: usize, k: usize, s: usize, p: usize, d: usize) -> usize {
    (sz + 2 * p - (d * (k - 1) + 1)) / s + 1
}

/// im2col gather indices: `[N·Hout·Wout, Cin·kH·kW]` (row-major), `PAD` where the
/// receptive-field window falls in the padding. Shared by `Conv` and `Im2Col`.
#[allow(clippy::too_many_arguments)]
fn im2col_idx(
    in_dims: &[usize],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    hout: usize,
    wout: usize,
) -> Vec<u32> {
    let (cin, h, w) = (in_dims[1], in_dims[2], in_dims[3]);
    let in_strides = strides(in_dims);
    let patch = cin * kh * kw;
    let npos = in_dims[0] * hout * wout;
    let mut idx = vec![PAD; npos * patch];
    for pos in 0..npos {
        let nn = pos / (hout * wout);
        let r2 = pos % (hout * wout);
        let (ho, wo) = (r2 / wout, r2 % wout);
        for ci in 0..cin {
            for ki in 0..kh {
                for kj in 0..kw {
                    let ih = ho * sh + ki * dh;
                    let iw = wo * sw + kj * dw;
                    if ih >= ph && iw >= pw {
                        let (rih, riw) = (ih - ph, iw - pw);
                        if rih < h && riw < w {
                            let col = ci * kh * kw + ki * kw + kj;
                            idx[pos * patch + col] = (nn * in_strides[0]
                                + ci * in_strides[1]
                                + rih * in_strides[2]
                                + riw * in_strides[3])
                                as u32;
                        }
                    }
                }
            }
        }
    }
    idx
}

/// Backward ops this backend lowers directly (kept by the decomposition pass).
const NATIVE_BACKWARD: &[OpKind] = &[OpKind::ReluBackward, OpKind::ActivationBackward];

/// Lower a graph into a [`Plan`].
///
/// First **legalizes** the graph against [`supported_ops`]: dedicated
/// `*Backward` ops are decomposed into primitives
/// ([`rlx_autodiff::decompose_backward_ops_except`]) and unsupported forward
/// composites are rewritten into primitives
/// ([`rlx_compile::rewrite_for_backend`]) — this lowers GroupNorm,
/// BatchNormInference, SoftmaxCrossEntropy, DotGeneral, control flow, fused /
/// RNN ops and every `*Backward` op down to the small kernel set the WebGL
/// executor implements. Then each remaining node is lowered to a [`Step`].
pub fn build_plan(graph: &Graph) -> Result<Plan> {
    let g = graph.clone();
    let g = rlx_autodiff::decompose_backward_ops_except(g, NATIVE_BACKWARD);
    let g = rlx_fusion::unfuse_dit_modulation(g);
    let g = rlx_compile::rewrite_for_backend(g, supported_ops());
    lower(&g)
}

/// Row-major broadcast index map: `out_idx → in_idx` for broadcasting `in_dims`
/// to `out_dims` (numpy rules, right-aligned, size-1 dims broadcast).
fn broadcast_idx(in_dims: &[usize], out_dims: &[usize]) -> Vec<u32> {
    let in_strides = strides(in_dims);
    let offset = out_dims.len().saturating_sub(in_dims.len());
    let n = total(out_dims);
    let mut idx = Vec::with_capacity(n);
    for o in 0..n {
        let oc = unravel(o, out_dims);
        let mut in_lin = 0usize;
        for k in 0..in_dims.len() {
            let coord = if in_dims[k] == 1 { 0 } else { oc[k + offset] };
            in_lin += coord * in_strides[k];
        }
        idx.push(in_lin as u32);
    }
    idx
}

/// Accumulates [`Step`]s and slots. Supports synthetic intermediate slots so a
/// node can lower to several steps (e.g. an implicit broadcast before a binary).
struct Builder {
    slot_dims: Vec<(usize, usize)>,
    slot_len: Vec<usize>,
    slot_origin: Vec<String>,
    /// The node being lowered, recorded as each slot's origin.
    cur: String,
    steps: Vec<Step>,
}

impl Builder {
    fn new(cap: usize) -> Self {
        Self {
            slot_dims: Vec::with_capacity(cap),
            slot_len: Vec::with_capacity(cap),
            slot_origin: Vec::with_capacity(cap),
            cur: String::new(),
            steps: Vec::with_capacity(cap),
        }
    }

    fn alloc(&mut self, dims: &[usize]) -> usize {
        self.alloc_lanes(dims, 1)
    }

    /// Allocate a slot holding `lane` f32 lanes per logical element (see
    /// [`lanes`]). `lane == 1` for real/int/bool (the historical footprint);
    /// complex slots widen the texture's column count by the lane factor so the
    /// whole `[re, im]` (C64) / df64 (C128) footprint is stored, read back, and
    /// addressable by the complex kernels. The extra lanes ride the last
    /// (fastest) axis, so complex element `e` occupies flat lanes
    /// `e*lane .. e*lane+lane`.
    fn alloc_lanes(&mut self, dims: &[usize], lane: usize) -> usize {
        let (r, c) = rows_cols(dims);
        let i = self.slot_dims.len();
        self.slot_dims.push((r, c * lane));
        self.slot_len.push(r * c * lane);
        self.slot_origin.push(self.cur.clone());
        i
    }

    fn emit(&mut self, s: Step) {
        self.steps.push(s);
    }

    /// Broadcast `src` (shape `in_dims`) to `out_dims`, returning a slot holding
    /// the expanded value — or `src` unchanged when the shapes already match.
    /// Implicit broadcasting (the legalization passes rely on it) is realized as
    /// a precomputed gather, reusing the verified gather kernel.
    fn expand_to(&mut self, src: usize, in_dims: &[usize], out_dims: &[usize]) -> usize {
        if in_dims == out_dims {
            return src;
        }
        let idx = broadcast_idx(in_dims, out_dims);
        let out = self.alloc(out_dims);
        self.emit(Step::Gather { out, src, idx });
        out
    }
}

/// Lower an already-legalized graph (only [`supported_ops`] kinds) into a
/// [`Plan`]. Nodes are assumed topologically ordered (RLX builds bottom-up); a
/// non-topo input reference is reported as an error rather than mis-executed.
fn lower(graph: &Graph) -> Result<Plan> {
    let nodes = graph.nodes();
    let mut b = Builder::new(nodes.len());
    // NodeId → (output slot, logical dims, dtype). Keyed by id because synthetic
    // intermediate slots make slot indices diverge from node positions. The
    // dtype is carried so complex casts can classify src/dst and the complex
    // binary can require C64 operands.
    let mut id_info: HashMap<NodeId, (usize, Vec<usize>, DType)> =
        HashMap::with_capacity(nodes.len());

    for node in nodes {
        let dims = dims_of(&node.shape)?;
        b.cur = format!("{:?} {dims:?}", node.op.kind());
        let node_dt = node.shape.dtype();
        let node_lane = lanes(node_dt);
        let (r, c) = rows_cols(&dims);
        let out = b.alloc_lanes(&dims, node_lane);
        id_info.insert(node.id, (out, dims.clone(), node_dt));

        let input_slot = |i: usize| -> Result<usize> {
            node.inputs
                .get(i)
                .and_then(|id| id_info.get(id))
                .map(|(s, _, _)| *s)
                .ok_or_else(|| {
                    WebglError(format!(
                        "input {i} of {:?} unresolved (graph not topo?)",
                        node.op
                    ))
                })
        };
        let input_dims = |i: usize| -> Result<Vec<usize>> {
            node.inputs
                .get(i)
                .and_then(|id| id_info.get(id))
                .map(|(_, d, _)| d.clone())
                .ok_or_else(|| WebglError(format!("input {i} dims unresolved")))
        };
        let input_dt = |i: usize| -> Result<DType> {
            node.inputs
                .get(i)
                .and_then(|id| id_info.get(id))
                .map(|(_, _, dt)| *dt)
                .ok_or_else(|| WebglError(format!("input {i} dtype unresolved")))
        };
        // Lane-aware identity: for complex tensors this copies every `[re, im]`
        // (df64) lane, so Reshape / StopGradient / same-dtype Cast stay correct.
        let identity = || (0..(total(&dims) * node_lane) as u32).collect::<Vec<_>>();

        match &node.op {
            Op::Input { name } => b.emit(Step::Leaf {
                out,
                src: LeafSource::Input(name.clone()),
            }),
            Op::Param { name } => b.emit(Step::Leaf {
                out,
                src: LeafSource::Param(name.clone()),
            }),
            Op::Constant { data } => b.emit(Step::Leaf {
                out,
                src: LeafSource::Const(decode_f32_le(data)),
            }),
            Op::Cast { to } => {
                let dst = *to;
                let src_dt = input_dt(0)?;
                // Real F64 has no storage on the f32-uniform arena and this
                // backend performs no f64→f32 demotion — reject rather than
                // silently corrupt. (C128's df64 *components* ARE carried, 4 f32
                // lanes; those fall through to the complex path below.)
                if dst == DType::F64 || src_dt == DType::F64 {
                    return Err(WebglError(format!(
                        "cast {src_dt:?}->{dst:?} unsupported: the f32 WebGL arena has no \
                         F64 storage (C128 df64 is the complex path)"
                    )));
                }
                if src_dt.is_complex() || dst.is_complex() {
                    // Complex simulation: pure lane moves on the f32 arena
                    // (real↔C64, real↔C128, C64↔C128). Same-dtype complex is a
                    // full-lane identity copy (identity() is lane-aware). All
                    // other directions route to the standalone complex_cast
                    // kernel; modes mirror wgpu/cuda/vulkan.
                    if src_dt == dst {
                        b.emit(Step::Gather {
                            out,
                            src: input_slot(0)?,
                            idx: identity(),
                        });
                    } else {
                        // Element count N is shape-preserving across the cast,
                        // so total(dims) is the complex-element count either way.
                        let n = total(&dims);
                        let mode: u32 = match (src_dt, dst) {
                            (s, DType::C64) if !s.is_complex() => 0,  // real → C64
                            (DType::C64, d) if !d.is_complex() => 1,  // C64 → real
                            (s, DType::C128) if !s.is_complex() => 2, // real → C128
                            (DType::C128, d) if !d.is_complex() => 3, // C128 → real
                            (DType::C64, DType::C128) => 4,
                            (DType::C128, DType::C64) => 5,
                            _ => {
                                return Err(WebglError(format!(
                                    "complex cast {src_dt:?}->{dst:?} unclassified"
                                )));
                            }
                        };
                        b.emit(Step::ComplexCast {
                            out,
                            src: input_slot(0)?,
                            mode,
                            n,
                        });
                    }
                } else if dst == DType::F32 {
                    // Value-preserving copy on the f32-uniform arena (every value
                    // is already f32-encoded). Handles int→f32 / f32→f32.
                    b.emit(Step::Gather {
                        out,
                        src: input_slot(0)?,
                        idx: identity(),
                    });
                } else {
                    return Err(WebglError(format!(
                        "cast to {dst:?} unsupported (f32-only; C64/C128 complex simulated)"
                    )));
                }
            }
            Op::StopGradient => {
                // Forward identity (the AD pass already applied the reverse rule).
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx: identity(),
                });
            }
            Op::Activation(a) => b.emit(Step::Unary {
                out,
                a: input_slot(0)?,
                act: map_act(*a)?,
            }),
            Op::ReluBackward => b.emit(Step::ActBack {
                out,
                x: input_slot(0)?,
                dy: input_slot(1)?,
                act: Act::Relu,
            }),
            Op::ActivationBackward { kind } => b.emit(Step::ActBack {
                out,
                x: input_slot(0)?,
                dy: input_slot(1)?,
                act: map_act(*kind)?,
            }),
            Op::Binary(bop) => {
                // Complex binary: each output element reads BOTH `[re, im]`
                // lanes of its operands, so it cannot ride the scalar-per-lane
                // Binary path (nor the pre-gather broadcast, which is element-
                // unit). Route BEFORE `expand_to` to a standalone binary_c64
                // step whose broadcast is per-operand complex-element modulo.
                // Rejections mirror rlx-cpu exactly: C128 arithmetic entirely,
                // and C64 Max/Min/Pow.
                if node_dt.is_complex() {
                    if node_dt == DType::C128 {
                        return Err(WebglError(
                            "Binary on C128 unsupported: complex-f64 arithmetic is not wired \
                             (rlx-cpu rejects it too); only C64 Add/Sub/Mul/Div"
                                .into(),
                        ));
                    }
                    let op = match bop {
                        BinaryOp::Add => Bin::Add,
                        BinaryOp::Sub => Bin::Sub,
                        BinaryOp::Mul => Bin::Mul,
                        BinaryOp::Div => Bin::Div,
                        BinaryOp::Max
                        | BinaryOp::Min
                        | BinaryOp::Pow
                        | BinaryOp::Mod
                        | BinaryOp::BitAnd
                        | BinaryOp::BitOr
                        | BinaryOp::BitXor
                        | BinaryOp::Shl
                        | BinaryOp::Shr
                        | BinaryOp::Atan2 => {
                            return Err(WebglError(format!(
                                "C64 Binary {bop:?} is undefined for complex \
                                 (only Add/Sub/Mul/Div); matches rlx-cpu rejection"
                            )));
                        }
                    };
                    // Both operands must be C64-laid-out (2 f32 lanes/elem) — the
                    // kernel reads 2*k lanes. A real operand (1 lane) would be
                    // misread, so require an explicit upstream Cast (rlx-cpu's
                    // complex binary assumes the same 2-lane operand layout).
                    let (a_dt, b_dt) = (input_dt(0)?, input_dt(1)?);
                    if a_dt != DType::C64 || b_dt != DType::C64 {
                        return Err(WebglError(format!(
                            "C64 Binary requires both operands C64 (got {a_dt:?}, {b_dt:?}); \
                             insert an explicit Cast to C64 upstream"
                        )));
                    }
                    let (ad, bd) = (input_dims(0)?, input_dims(1)?);
                    b.emit(Step::BinaryC64 {
                        out,
                        a: input_slot(0)?,
                        b: input_slot(1)?,
                        op,
                        n: total(&dims),
                        n_a: total(&ad).max(1),
                        n_b: total(&bd).max(1),
                    });
                    continue;
                }
                let op = match bop {
                    BinaryOp::Add => Bin::Add,
                    BinaryOp::Sub => Bin::Sub,
                    BinaryOp::Mul => Bin::Mul,
                    BinaryOp::Div => Bin::Div,
                    BinaryOp::Max => Bin::Max,
                    BinaryOp::Min => Bin::Min,
                    BinaryOp::Pow => Bin::Pow,
                    BinaryOp::Mod => Bin::Mod,
                    BinaryOp::BitAnd => Bin::BitAnd,
                    BinaryOp::BitOr => Bin::BitOr,
                    BinaryOp::BitXor => Bin::BitXor,
                    BinaryOp::Shl => Bin::Shl,
                    BinaryOp::Shr => Bin::Shr,
                    BinaryOp::Atan2 => Bin::Atan2,
                };
                let (ad, bd) = (input_dims(0)?, input_dims(1)?);
                let a = b.expand_to(input_slot(0)?, &ad, &dims);
                let rhs = b.expand_to(input_slot(1)?, &bd, &dims);
                b.emit(Step::Binary { out, a, b: rhs, op });
            }
            Op::Compare(c) => {
                let cmp = match c {
                    CmpOp::Eq => Cmp::Eq,
                    CmpOp::Ne => Cmp::Ne,
                    CmpOp::Lt => Cmp::Lt,
                    CmpOp::Le => Cmp::Le,
                    CmpOp::Gt => Cmp::Gt,
                    CmpOp::Ge => Cmp::Ge,
                };
                let (ad, bd) = (input_dims(0)?, input_dims(1)?);
                let a = b.expand_to(input_slot(0)?, &ad, &dims);
                let rhs = b.expand_to(input_slot(1)?, &bd, &dims);
                b.emit(Step::Compare {
                    out,
                    a,
                    b: rhs,
                    cmp,
                });
            }
            Op::Where => {
                let (cd, ad, bd) = (input_dims(0)?, input_dims(1)?, input_dims(2)?);
                let cond = b.expand_to(input_slot(0)?, &cd, &dims);
                let a = b.expand_to(input_slot(1)?, &ad, &dims);
                let rhs = b.expand_to(input_slot(2)?, &bd, &dims);
                b.emit(Step::Where {
                    out,
                    cond,
                    a,
                    b: rhs,
                });
            }
            Op::MatMul => {
                // `(M, K) · (K, N)` row-major. K comes from the lhs' last dim;
                // N = rhs_total / K so a flattened rhs (e.g. a concat'd `[K·N]`,
                // as the conv-backward decomposition produces) is handled too.
                let a_dims = input_dims(0)?;
                let b_dims = input_dims(1)?;
                let (m, k) = rows_cols(&a_dims);
                let tb = total(&b_dims);
                if k == 0 || !tb.is_multiple_of(k) {
                    return Err(WebglError(format!(
                        "matmul shape mismatch: {a_dims:?}·{b_dims:?}"
                    )));
                }
                // `[..batch, M, K] · [..batch, K, N]` (attention's per-head
                // products): one (M×K)·(K×N) per batch block. A rank-2 rhs, or
                // one whose batch dims are all 1, is shared by every block.
                if b_dims.len() >= 3 && a_dims.len() >= 3 {
                    let bm = a_dims[a_dims.len() - 2];
                    let a_batch: usize = a_dims[..a_dims.len() - 2].iter().product();
                    let b_batch: usize = b_dims[..b_dims.len() - 2].iter().product();
                    let bn = *b_dims.last().unwrap();
                    if b_dims[b_dims.len() - 2] != k || (b_batch != a_batch && b_batch != 1) {
                        return Err(WebglError(format!(
                            "batched matmul shape mismatch: {a_dims:?}·{b_dims:?}"
                        )));
                    }
                    b.emit(Step::MatMul {
                        out,
                        a: input_slot(0)?,
                        b: input_slot(1)?,
                        m: bm,
                        k,
                        n: bn,
                        batch: a_batch,
                        b_batched: b_batch > 1,
                    });
                } else {
                    let n = tb / k;
                    b.emit(Step::MatMul {
                        out,
                        a: input_slot(0)?,
                        b: input_slot(1)?,
                        m,
                        k,
                        n,
                        batch: 1,
                        b_batched: false,
                    });
                }
            }
            Op::Reshape { .. } => {
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx: identity(),
                });
            }
            Op::Transpose { perm } => {
                // Lane-aware for complex: a complex element spans `node_lane`
                // contiguous f32 lanes (C64 = 2 `[re, im]`, C128 = 4 df64) and
                // slots are lane-based. The naive element-indexed reindex copies
                // ONE lane per output ELEMENT from the wrong source lane,
                // shattering the `[re, im]` pairing. Append a contiguous
                // innermost lane axis that maps to ITSELF (never permuted) so a
                // whole complex element moves as a group; `ravel`/`unravel` then
                // yield N*lane lane indices matching the lane-based slots.
                // `node_lane == 1` (real/int/bool) ⇒ the extra axis is a strict
                // no-op. Mirrors the wgpu Transpose fix.
                let in_dims0 = input_dims(0)?;
                let (in_dims, out_dims, perm_l) = if node_lane > 1 {
                    let mut ind = in_dims0.clone();
                    ind.push(node_lane);
                    let mut outd = dims.clone();
                    outd.push(node_lane);
                    let mut pl = perm.clone();
                    pl.push(perm.len()); // new lane axis (index == rank) → itself
                    (ind, outd, pl)
                } else {
                    (in_dims0, dims.clone(), perm.clone())
                };
                let in_strides = strides(&in_dims);
                let n = total(&out_dims);
                let mut idx = Vec::with_capacity(n);
                for o in 0..n {
                    let oc = unravel(o, &out_dims);
                    let mut ic = vec![0usize; in_dims.len()];
                    for (i, &p) in perm_l.iter().enumerate() {
                        ic[p] = oc[i];
                    }
                    idx.push(ravel(&ic, &in_strides) as u32);
                }
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx,
                });
            }
            Op::Expand { .. } => {
                let in_dims = input_dims(0)?;
                if dims.len() < in_dims.len() {
                    return Err(WebglError("expand cannot reduce rank".into()));
                }
                // Lane-aware broadcast for complex dtypes. A complex element
                // spans `node_lane` contiguous f32 lanes (C64 = 2 `[re, im]`,
                // C128 = 4 df64) and slots are lane-based, so a naive element-
                // indexed broadcast copies ONE lane per output ELEMENT — it
                // writes only N of the N*lane lanes, from the wrong source
                // lanes, shattering the `[re, im]` pairing. Append a contiguous
                // innermost lane axis (size = node_lane, `in == out`, NEVER
                // broadcast) so whole complex elements copy as a group; the
                // existing `broadcast_idx` then yields N*lane lane indices that
                // match the lane-based slot layout. `node_lane == 1` for
                // real/int/bool ⇒ the extra axis is a strict no-op. Mirrors the
                // cuda/wgpu Expand fix, which appends the same lane axis to the
                // expand dims/strides.
                let idx = if node_lane > 1 {
                    let mut in_lane = in_dims.clone();
                    let mut out_lane = dims.clone();
                    in_lane.push(node_lane);
                    out_lane.push(node_lane);
                    broadcast_idx(&in_lane, &out_lane)
                } else {
                    broadcast_idx(&in_dims, &dims)
                };
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx,
                });
            }
            Op::Narrow { axis, start, .. } => {
                // Lane-aware for complex (see Transpose). `axis`/`start` stay in
                // ELEMENT units; the appended innermost lane axis carries the
                // `[re, im]` (df64) group so a narrowed complex element keeps all
                // its lanes. `node_lane == 1` ⇒ strict no-op.
                let in_dims0 = input_dims(0)?;
                let (in_dims, out_dims) = if node_lane > 1 {
                    let mut ind = in_dims0.clone();
                    ind.push(node_lane);
                    let mut outd = dims.clone();
                    outd.push(node_lane);
                    (ind, outd)
                } else {
                    (in_dims0, dims.clone())
                };
                let in_strides = strides(&in_dims);
                let n = total(&out_dims);
                let mut idx = Vec::with_capacity(n);
                for o in 0..n {
                    let mut ic = unravel(o, &out_dims);
                    ic[*axis] += *start;
                    idx.push(ravel(&ic, &in_strides) as u32);
                }
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx,
                });
            }
            Op::Reverse { axes } => {
                let in_dims = input_dims(0)?;
                let in_strides = strides(&in_dims);
                let axset: HashSet<usize> = axes.iter().copied().collect();
                let n = total(&dims);
                let mut idx = Vec::with_capacity(n);
                for o in 0..n {
                    let mut ic = unravel(o, &dims);
                    for (ax, coord) in ic.iter_mut().enumerate() {
                        if axset.contains(&ax) {
                            *coord = in_dims[ax] - 1 - *coord;
                        }
                    }
                    idx.push(ravel(&ic, &in_strides) as u32);
                }
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx,
                });
            }
            Op::Reduce { op, axes, keep_dim } => {
                let red = match op {
                    ReduceOp::Sum => Red::Sum,
                    ReduceOp::Mean => Red::Mean,
                    ReduceOp::Max => Red::Max,
                    ReduceOp::Min => Red::Min,
                    ReduceOp::Prod => Red::Prod,
                };
                let in_dims = input_dims(0)?;
                let out_strides = strides(&dims);
                let axes_set: HashSet<usize> = axes.iter().copied().collect();
                let n_out = total(&dims);
                let in_n = total(&in_dims);
                let mut groups: Vec<Vec<u32>> = vec![Vec::new(); n_out];
                for i in 0..in_n {
                    let ic = unravel(i, &in_dims);
                    let oc: Vec<usize> = if *keep_dim {
                        ic.iter()
                            .enumerate()
                            .map(|(ax, &cc)| if axes_set.contains(&ax) { 0 } else { cc })
                            .collect()
                    } else {
                        ic.iter()
                            .enumerate()
                            .filter(|(ax, _)| !axes_set.contains(ax))
                            .map(|(_, &cc)| cc)
                            .collect()
                    };
                    groups[ravel(&oc, &out_strides)].push(i as u32);
                }
                let fanin = groups.iter().map(|g| g.len()).max().unwrap_or(0).max(1);
                let mut flat = vec![PAD; n_out * fanin];
                for (oi, g) in groups.iter().enumerate() {
                    for (j, &v) in g.iter().enumerate() {
                        flat[oi * fanin + j] = v;
                    }
                }
                b.emit(Step::Reduce {
                    out,
                    src: input_slot(0)?,
                    groups: flat,
                    fanin,
                    op: red,
                });
            }
            Op::Softmax { axis } => {
                require_last_axis(*axis, dims.len(), "softmax")?;
                b.emit(Step::Softmax {
                    out,
                    a: input_slot(0)?,
                    rows: r,
                    cols: c,
                });
            }
            Op::LayerNorm { axis, eps } => {
                require_last_axis(*axis, dims.len(), "layer_norm")?;
                b.emit(Step::LayerNorm {
                    out,
                    x: input_slot(0)?,
                    gamma: input_slot(1)?,
                    beta: input_slot(2)?,
                    rows: r,
                    cols: c,
                    eps: *eps,
                });
            }
            Op::RmsNorm { axis, eps } => {
                require_last_axis(*axis, dims.len(), "rms_norm")?;
                b.emit(Step::RmsNorm {
                    out,
                    x: input_slot(0)?,
                    gamma: input_slot(1)?,
                    beta: input_slot(2)?,
                    rows: r,
                    cols: c,
                    eps: *eps,
                });
            }
            Op::Cumsum { axis, exclusive } => {
                // Cumsum over the last axis == matmul by a triangular ones matrix:
                // out[r, j] = Σ_{i ≤ j (or i < j)} x[r, i].
                require_last_axis(*axis, dims.len(), "cumsum")?;
                let cols = c;
                let mut u = vec![0f32; cols * cols];
                for i in 0..cols {
                    for j in 0..cols {
                        let include = if *exclusive { i < j } else { i <= j };
                        if include {
                            u[i * cols + j] = 1.0;
                        }
                    }
                }
                let u_slot = b.alloc(&[cols, cols]);
                b.emit(Step::Leaf {
                    out: u_slot,
                    src: LeafSource::Const(u),
                });
                b.emit(Step::MatMul {
                    out,
                    a: input_slot(0)?,
                    b: u_slot,
                    m: r,
                    k: cols,
                    n: cols,
                    batch: 1,
                    b_batched: false,
                });
            }
            Op::Concat { axis } => {
                // Place each input into its slice of the output via a masked
                // gather, then sum: out = Σ_k (mask_k ⊙ gather_k). Reuses the
                // gather + binary kernels — no concat-specific shader.
                //
                // Lane-aware for complex: the per-input gather indices, the 0/1
                // mask, and EVERY intermediate slot are widened by `node_lane`
                // (append an innermost lane axis to the dim lists) so whole
                // `[re, im]` (df64) complex elements move/mask/accumulate as a
                // group. `axis`/`span` stay ELEMENT units — the lane axis is
                // never the concat axis. The mask is uniform across an element's
                // lanes (it depends on the element-axis coord only), so the
                // masked-sum stays disjoint and lane-exact. The final
                // `identity()` gather is already lane-aware. `node_lane == 1` ⇒
                // every widening is a strict no-op (identical to the old path).
                let mut out_dims = dims.clone();
                if node_lane > 1 {
                    out_dims.push(node_lane);
                }
                let n = total(&out_dims);
                let mut axis_off = 0usize;
                let mut acc: Option<usize> = None;
                for ii in 0..node.inputs.len() {
                    let in_dims0 = input_dims(ii)?;
                    let span = in_dims0[*axis]; // element units
                    let mut in_dims = in_dims0;
                    if node_lane > 1 {
                        in_dims.push(node_lane);
                    }
                    let in_strides = strides(&in_dims);
                    let mut idx = vec![0u32; n];
                    let mut mask = vec![0f32; n];
                    for o in 0..n {
                        let mut oc = unravel(o, &out_dims);
                        let a = oc[*axis];
                        if a >= axis_off && a < axis_off + span {
                            oc[*axis] = a - axis_off;
                            idx[o] = ravel(&oc, &in_strides) as u32;
                            mask[o] = 1.0;
                        }
                    }
                    axis_off += span;
                    let gathered = b.alloc_lanes(&dims, node_lane);
                    b.emit(Step::Gather {
                        out: gathered,
                        src: input_slot(ii)?,
                        idx,
                    });
                    let mask_slot = b.alloc_lanes(&dims, node_lane);
                    b.emit(Step::Leaf {
                        out: mask_slot,
                        src: LeafSource::Const(mask),
                    });
                    let masked = b.alloc_lanes(&dims, node_lane);
                    b.emit(Step::Binary {
                        out: masked,
                        a: gathered,
                        b: mask_slot,
                        op: Bin::Mul,
                    });
                    acc = Some(match acc {
                        None => masked,
                        Some(prev) => {
                            let s = b.alloc_lanes(&dims, node_lane);
                            b.emit(Step::Binary {
                                out: s,
                                a: prev,
                                b: masked,
                                op: Bin::Add,
                            });
                            s
                        }
                    });
                }
                let src = acc.ok_or_else(|| WebglError("concat with no inputs".into()))?;
                b.emit(Step::Gather {
                    out,
                    src,
                    idx: identity(),
                });
            }
            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                // NCHW pooling == a Reduce whose groups are each output's window
                // (padding → skipped). Reuses the reduce kernel.
                let in_dims = input_dims(0)?;
                if in_dims.len() != 4 || dims.len() != 4 {
                    return Err(WebglError("pool requires 4D NCHW".into()));
                }
                let (h, w) = (in_dims[2], in_dims[3]);
                let (kh, kw) = (kernel_size[0], kernel_size[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let (ph, pw) = (padding[0], padding[1]);
                let in_strides = strides(&in_dims);
                let n_out = total(&dims);
                let fanin = kh * kw;
                let mut groups = vec![PAD; n_out * fanin];
                for o in 0..n_out {
                    let oc = unravel(o, &dims); // [n, c, ho, wo]
                    let mut j = 0;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oc[2] * sh + ki;
                            let iw = oc[3] * sw + kj;
                            if ih >= ph && iw >= pw {
                                let (rih, riw) = (ih - ph, iw - pw);
                                if rih < h && riw < w {
                                    let lin = oc[0] * in_strides[0]
                                        + oc[1] * in_strides[1]
                                        + rih * in_strides[2]
                                        + riw * in_strides[3];
                                    groups[o * fanin + j] = lin as u32;
                                }
                            }
                            j += 1;
                        }
                    }
                }
                let red = match kind {
                    ReduceOp::Max => Red::Max,
                    ReduceOp::Mean => Red::Mean,
                    ReduceOp::Sum => Red::Sum,
                    ReduceOp::Min => Red::Min,
                    other => return Err(WebglError(format!("pool kind {other:?} not supported"))),
                };
                b.emit(Step::Reduce {
                    out,
                    src: input_slot(0)?,
                    groups,
                    fanin,
                    op: red,
                });
            }
            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // Forward conv as im2col-gather + matmul + output transpose-gather.
                // input [N,Cin,H,W], weight [Cout,Cin,kH,kW], out [N,Cout,Hout,Wout].
                let in_dims = input_dims(0)?;
                let w_dims = input_dims(1)?;
                if in_dims.len() != 4 || w_dims.len() != 4 || dims.len() != 4 {
                    return Err(WebglError("conv requires 4D NCHW".into()));
                }
                if *groups != 1 {
                    // Grouped / depthwise: each output reads only its group's
                    // channels, so a shared [patch, cout] matmul does not
                    // apply. Gather every output's window and its weights side
                    // by side, multiply, and sum the window:
                    //   out[n,co,ho,wo] = Σ_{ci,ki,kj} x[n, g·cin_g + ci, ·, ·] · w[co, ci, ki, kj]
                    // with g = co / cout_g. (Depthwise k-tap convs — LFM2's
                    // ShortConv — are `red = k`, so this stays small.)
                    let g = *groups;
                    let (cin, cout) = (in_dims[1], w_dims[0]);
                    if cin % g != 0 || cout % g != 0 || w_dims[1] != cin / g {
                        return Err(WebglError(format!(
                            "conv groups={g}: cin {cin} / cout {cout} / weight {w_dims:?} do not divide"
                        )));
                    }
                    let (cin_g, cout_g) = (cin / g, cout / g);
                    let (kh, kw) = (kernel_size[0], kernel_size[1]);
                    let (sh, sw) = (stride[0], stride[1]);
                    let (ph, pw) = (padding[0], padding[1]);
                    let (dh, dw) = (dilation[0], dilation[1]);
                    let (h, w) = (in_dims[2], in_dims[3]);
                    let red = cin_g * kh * kw;
                    let out_n = total(&dims);
                    let (xs, ws) = (strides(&in_dims), strides(&w_dims));
                    let mut x_idx = vec![PAD; out_n * red];
                    let mut w_idx = vec![0u32; out_n * red];
                    for o in 0..out_n {
                        let oc = unravel(o, &dims); // [n, co, ho, wo]
                        let grp = oc[1] / cout_g;
                        for ci in 0..cin_g {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let r = (ci * kh + ki) * kw + kj;
                                    w_idx[o * red + r] =
                                        (oc[1] * ws[0] + ci * ws[1] + ki * ws[2] + kj * ws[3])
                                            as u32;
                                    let ih = oc[2] * sh + ki * dh;
                                    let iw = oc[3] * sw + kj * dw;
                                    if ih >= ph && iw >= pw && ih - ph < h && iw - pw < w {
                                        x_idx[o * red + r] = (oc[0] * xs[0]
                                            + (grp * cin_g + ci) * xs[1]
                                            + (ih - ph) * xs[2]
                                            + (iw - pw) * xs[3])
                                            as u32;
                                    }
                                }
                            }
                        }
                    }
                    let p_slot = b.alloc(&[out_n, red]);
                    b.emit(Step::Gather {
                        out: p_slot,
                        src: input_slot(0)?,
                        idx: x_idx,
                    });
                    let w_slot = b.alloc(&[out_n, red]);
                    b.emit(Step::Gather {
                        out: w_slot,
                        src: input_slot(1)?,
                        idx: w_idx,
                    });
                    let m_slot = b.alloc(&[out_n, red]);
                    b.emit(Step::Binary {
                        out: m_slot,
                        a: p_slot,
                        b: w_slot,
                        op: Bin::Mul,
                    });
                    b.emit(Step::Reduce {
                        out,
                        src: m_slot,
                        groups: (0..(out_n * red) as u32).collect(),
                        fanin: red,
                        op: Red::Sum,
                    });
                } else {
                    let cin = in_dims[1];
                    let cout = w_dims[0];
                    let (kh, kw) = (kernel_size[0], kernel_size[1]);
                    let (sh, sw) = (stride[0], stride[1]);
                    let (ph, pw) = (padding[0], padding[1]);
                    let (dh, dw) = (dilation[0], dilation[1]);
                    let (hout, wout) = (dims[2], dims[3]);
                    let patch = cin * kh * kw;
                    let npos = dims[0] * hout * wout;

                    // im2col patches P[pos, col] (PAD where the window hits padding).
                    let p_idx = im2col_idx(&in_dims, kh, kw, sh, sw, ph, pw, dh, dw, hout, wout);
                    let p_slot = b.alloc(&[npos, patch]);
                    b.emit(Step::Gather {
                        out: p_slot,
                        src: input_slot(0)?,
                        idx: p_idx,
                    });

                    // Weight → [patch, cout]: W2[col, co] = weight[co, ci, ki, kj].
                    let w_strides = strides(&w_dims);
                    let mut w_idx = vec![0u32; patch * cout];
                    for col in 0..patch {
                        let ci = col / (kh * kw);
                        let r2 = col % (kh * kw);
                        let (ki, kj) = (r2 / kw, r2 % kw);
                        for co in 0..cout {
                            w_idx[col * cout + co] = (co * w_strides[0]
                                + ci * w_strides[1]
                                + ki * w_strides[2]
                                + kj * w_strides[3])
                                as u32;
                        }
                    }
                    let w2_slot = b.alloc(&[patch, cout]);
                    b.emit(Step::Gather {
                        out: w2_slot,
                        src: input_slot(1)?,
                        idx: w_idx,
                    });

                    // M[pos, co] = P @ W2.
                    let m_slot = b.alloc(&[npos, cout]);
                    b.emit(Step::MatMul {
                        out: m_slot,
                        a: p_slot,
                        b: w2_slot,
                        m: npos,
                        k: patch,
                        n: cout,
                        batch: 1,
                        b_batched: false,
                    });

                    // Scatter to NCHW: out[n,co,ho,wo] = M[(n·Hout·Wout + ho·Wout + wo), co].
                    let out_n = total(&dims);
                    let mut o_idx = vec![0u32; out_n];
                    for o in 0..out_n {
                        let oc = unravel(o, &dims); // [n, co, ho, wo]
                        let pos = oc[0] * (hout * wout) + oc[2] * wout + oc[3];
                        o_idx[o] = (pos * cout + oc[1]) as u32;
                    }
                    b.emit(Step::Gather {
                        out,
                        src: m_slot,
                        idx: o_idx,
                    });
                }
            }
            Op::Im2Col {
                kernel_size,
                stride,
                padding,
                dilation,
            } => {
                // Output [N·Hout·Wout, Cin·kH·kW] — exactly the conv patch matrix.
                let in_dims = input_dims(0)?;
                if in_dims.len() != 4 {
                    return Err(WebglError("im2col requires 4D NCHW".into()));
                }
                let (h, w) = (in_dims[2], in_dims[3]);
                let (kh, kw) = (kernel_size[0], kernel_size[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let (ph, pw) = (padding[0], padding[1]);
                let (dh, dw) = (dilation[0], dilation[1]);
                let hout = conv_out(h, kh, sh, ph, dh);
                let wout = conv_out(w, kw, sw, pw, dw);
                let idx = im2col_idx(&in_dims, kh, kw, sh, sw, ph, pw, dh, dw, hout, wout);
                b.emit(Step::Gather {
                    out,
                    src: input_slot(0)?,
                    idx,
                });
            }
            Op::ArgMax { axis, keep_dim } => {
                let groups = arg_groups(&input_dims(0)?, &dims, *axis, *keep_dim);
                let fanin = input_dims(0)?[*axis];
                b.emit(Step::ArgReduce {
                    out,
                    src: input_slot(0)?,
                    groups,
                    fanin,
                    is_max: true,
                });
            }
            Op::ArgMin { axis, keep_dim } => {
                let groups = arg_groups(&input_dims(0)?, &dims, *axis, *keep_dim);
                let fanin = input_dims(0)?[*axis];
                b.emit(Step::ArgReduce {
                    out,
                    src: input_slot(0)?,
                    groups,
                    fanin,
                    is_max: false,
                });
            }
            Op::Gather { axis } => {
                // Embedding-style gather: replace `axis` of `table` with the
                // (runtime) `indices` tensor — out[pre, idx_coords, post] =
                // table[pre, indices[idx_coords], post].
                let table_dims = input_dims(0)?;
                let idx_dims = input_dims(1)?;
                let axis = *axis;
                let t_strides = strides(&table_dims);
                let post = t_strides[axis]; // product of dims after `axis`
                let a = table_dims[axis];
                let idx_n = total(&idx_dims).max(1);
                let n_out = total(&dims);
                // Lane-aware for a complex TABLE: each table/output element spans
                // `node_lane` f32 lanes. `base`/`axis_stride` become TABLE-LANE
                // offsets (element offset × node_lane, plus the intra-element lane
                // `l`), so `base + idx_val*axis_stride == table_element*lane + l`.
                // Every output lane reuses its element's runtime index (`indices`
                // is real → 1 lane/elem, so `which` stays an element index).
                // `node_lane == 1` ⇒ the historical element-unit path.
                let lane = node_lane;
                let mut which = vec![0u32; n_out * lane];
                let mut base = vec![0u32; n_out * lane];
                for o in 0..n_out {
                    // output flattened row-major over [pre, idx_n, post].
                    let p_post = o % post;
                    let rest = o / post;
                    let ix = rest % idx_n;
                    let p_pre = rest / idx_n;
                    let base_elem = p_pre * (a * post) + p_post;
                    for l in 0..lane {
                        which[o * lane + l] = ix as u32;
                        base[o * lane + l] = (base_elem * lane + l) as u32;
                    }
                }
                b.emit(Step::GatherRuntime {
                    out,
                    table: input_slot(0)?,
                    indices: input_slot(1)?,
                    which,
                    base,
                    axis_stride: (post * lane) as u32,
                });
            }
            Op::Rope {
                head_dim,
                n_rot,
                style,
                ..
            } => {
                // Decomposed: out = mask ? (x·cos + rot(x)·sin) : x — cos/sin
                // gathered from the runtime caches by position. NeoX rotates
                // the pair (i, i + n_rot/2), GPT-J the adjacent pair
                // (2j, 2j + 1) by angle j — as rlx-cpu does.
                let interleaved = matches!(style, rlx_ir::op::RopeStyle::GptJ);
                let head_dim = *head_dim;
                let n_rot = *n_rot;
                let x_dims = input_dims(0)?;
                let cos_dims = input_dims(1)?;
                let n = total(&dims);
                let rank = x_dims.len();
                let last = *x_dims.last().unwrap_or(&head_dim);
                let heads_per_seq = if last > head_dim && last % head_dim == 0 {
                    last / head_dim
                } else {
                    1
                };
                let s = if rank >= 2 { x_dims[rank - 2] } else { 1 }.max(1);
                let rot_half = n_rot / 2;
                // Row stride is the cos/sin table's OWN last dimension — not
                // `n_rot/2`, and not `head_dim/2`. The two differ under partial
                // rotation and the layout is a per-model choice: Qwen3.5
                // allocates `[max_pos, head_dim/2]` and uses only the leading
                // `n_rot/2` columns of each row, while DeepSeek-V4 MLA packs
                // `[.., n_rot/2]` with no slack. Deriving it either way reads
                // some other position's angles for every row past the first on
                // the other layout, so take it from the shape.
                let tab_half = if cos_dims.len() >= 2 {
                    cos_dims.last().copied().unwrap_or(rot_half).max(1)
                } else {
                    rot_half.max(1)
                };
                let cos_rows = cos_dims.first().copied().unwrap_or(1).max(1);

                let mut cos_idx = vec![0u32; n];
                let mut sin_idx = vec![0u32; n];
                let mut rot_idx = vec![0u32; n];
                let mut sign = vec![0f32; n];
                let mut mask = vec![0f32; n];
                for e in 0..n {
                    let chunk = e / head_dim;
                    let i = e % head_dim;
                    let token = chunk / heads_per_seq;
                    let pos = if cos_rows == 1 {
                        0
                    } else {
                        (token % s).min(cos_rows - 1)
                    };
                    rot_idx[e] = e as u32;
                    if i < n_rot {
                        mask[e] = 1.0;
                        let col = if interleaved {
                            i / 2
                        } else {
                            i % rot_half.max(1)
                        };
                        cos_idx[e] = (pos * tab_half + col) as u32;
                        sin_idx[e] = (pos * tab_half + col) as u32;
                        let off = chunk * head_dim;
                        if interleaved {
                            if i % 2 == 0 {
                                rot_idx[e] = (off + i + 1) as u32;
                                sign[e] = -1.0;
                            } else {
                                rot_idx[e] = (off + i - 1) as u32;
                                sign[e] = 1.0;
                            }
                        } else if i < rot_half {
                            rot_idx[e] = (off + rot_half + i) as u32;
                            sign[e] = -1.0;
                        } else {
                            rot_idx[e] = (off + i - rot_half) as u32;
                            sign[e] = 1.0;
                        }
                    }
                }
                let (x, cos, sin) = (input_slot(0)?, input_slot(1)?, input_slot(2)?);
                let cos_g = b.alloc(&dims);
                b.emit(Step::Gather {
                    out: cos_g,
                    src: cos,
                    idx: cos_idx,
                });
                let sin_g = b.alloc(&dims);
                b.emit(Step::Gather {
                    out: sin_g,
                    src: sin,
                    idx: sin_idx,
                });
                let xrot = b.alloc(&dims);
                b.emit(Step::Gather {
                    out: xrot,
                    src: x,
                    idx: rot_idx,
                });
                let sign_c = b.alloc(&dims);
                b.emit(Step::Leaf {
                    out: sign_c,
                    src: LeafSource::Const(sign),
                });
                let xrot_s = b.alloc(&dims);
                b.emit(Step::Binary {
                    out: xrot_s,
                    a: xrot,
                    b: sign_c,
                    op: Bin::Mul,
                });
                let t1 = b.alloc(&dims);
                b.emit(Step::Binary {
                    out: t1,
                    a: x,
                    b: cos_g,
                    op: Bin::Mul,
                });
                let t2 = b.alloc(&dims);
                b.emit(Step::Binary {
                    out: t2,
                    a: xrot_s,
                    b: sin_g,
                    op: Bin::Mul,
                });
                let rotated = b.alloc(&dims);
                b.emit(Step::Binary {
                    out: rotated,
                    a: t1,
                    b: t2,
                    op: Bin::Add,
                });
                let mask_c = b.alloc(&dims);
                b.emit(Step::Leaf {
                    out: mask_c,
                    src: LeafSource::Const(mask),
                });
                b.emit(Step::Where {
                    out,
                    cond: mask_c,
                    a: rotated,
                    b: x,
                });
            }
            Op::Custom {
                name,
                num_inputs,
                attrs,
            } => {
                // Only the host/transport `collective.*` ops are wired (single
                // f32 input, host-delegated). Any other custom op name is
                // rejected — rlx-webgl has no other host kernels and is f32-only.
                if !COLLECTIVE_OPS.contains(&name.as_str()) {
                    return Err(WebglError(format!(
                        "custom op {name:?} not supported on WebGL (only collective.* ops)"
                    )));
                }
                if *num_inputs != 1 {
                    return Err(WebglError(format!(
                        "collective op {name:?} expects 1 input, graph has {num_inputs}"
                    )));
                }
                b.emit(Step::Custom {
                    out,
                    input: input_slot(0)?,
                    name: name.clone(),
                    attrs: attrs.clone(),
                });
            }
            other => {
                return Err(WebglError(format!(
                    "op {:?} not supported on WebGL",
                    other.kind()
                )));
            }
        }
    }

    let outputs = graph
        .outputs
        .iter()
        .map(|id| {
            id_info
                .get(id)
                .map(|(s, _, _)| *s)
                .ok_or_else(|| WebglError("graph output not found".into()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut plan = Plan {
        steps: b.steps,
        slot_dims: b.slot_dims,
        slot_len: b.slot_len,
        slot_origin: b.slot_origin,
        outputs,
    };
    widen_tall_slots(&mut plan);
    Ok(plan)
}

/// A slot taller than this is laid out wide instead, when it can be.
const TALL_ROWS: usize = 8192;
/// Column count of a widened slot.
const WIDE_COLS: usize = 8192;

/// Lay out tall slots (`rows > TALL_ROWS`) as `ceil(n / WIDE_COLS) × WIDE_COLS`.
///
/// A slot is `rest × last_dim` texels, so a tensor with a short last axis —
/// a conv weight `[cout, cin, 1, k]` is `cout·cin × k` — overflows WebGL's
/// `MAX_TEXTURE_SIZE` (16384 on most desktops) long before it is large.
///
/// Shape matters only to kernels that work on rows: matmul, softmax, the
/// norms (and custom ops, unknown). Gather / reduce / leaf uploads address
/// texels by linear index, and element-wise kernels need only that every
/// operand shares the output's layout. So slots joined by element-wise steps
/// form a group, and a tall group is widened together unless a member is
/// produced or read by a row kernel.
fn widen_tall_slots(plan: &mut Plan) {
    let n = plan.slot_dims.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let union = |p: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra] = rb;
        }
    };
    let mut pinned = vec![false; n];
    for step in &plan.steps {
        match step {
            Step::Leaf { .. }
            | Step::Gather { .. }
            | Step::Reduce { .. }
            | Step::ArgReduce { .. }
            | Step::GatherRuntime { .. } => {}
            Step::Unary { out, a, .. } => union(&mut parent, *out, *a),
            Step::ActBack { out, x, dy, .. } => {
                union(&mut parent, *out, *x);
                union(&mut parent, *out, *dy);
            }
            Step::Binary { out, a, b, .. } | Step::Compare { out, a, b, .. } => {
                union(&mut parent, *out, *a);
                union(&mut parent, *out, *b);
            }
            Step::Where { out, cond, a, b } => {
                union(&mut parent, *out, *cond);
                union(&mut parent, *out, *a);
                union(&mut parent, *out, *b);
            }
            Step::MatMul { out, a, b, .. } => {
                for s in [*out, *a, *b] {
                    pinned[s] = true;
                }
            }
            Step::Softmax { out, a, .. } => {
                pinned[*out] = true;
                pinned[*a] = true;
            }
            Step::LayerNorm {
                out,
                x,
                gamma,
                beta,
                ..
            }
            | Step::RmsNorm {
                out,
                x,
                gamma,
                beta,
                ..
            } => {
                for s in [*out, *x, *gamma, *beta] {
                    pinned[s] = true;
                }
            }
            // Lane-packed complex kernels: keep their natural layout.
            Step::ComplexCast { out, .. } | Step::BinaryC64 { out, .. } => pinned[*out] = true,
            Step::Custom { out, input, .. } => {
                pinned[*out] = true;
                pinned[*input] = true;
            }
        }
    }
    // A group widens only as a whole: every member tall-enough-to-matter or
    // not, same element count, nothing pinned.
    let mut group_ok: HashMap<usize, bool> = HashMap::new();
    let mut group_tall: HashMap<usize, bool> = HashMap::new();
    let mut group_len: HashMap<usize, usize> = HashMap::new();
    for s in 0..n {
        let r = find(&mut parent, s);
        let len = plan.slot_len[s];
        let ok = group_ok.entry(r).or_insert(true);
        *ok &= !pinned[s] && *group_len.entry(r).or_insert(len) == len;
        *group_tall.entry(r).or_insert(false) |= plan.slot_dims[s].0 > TALL_ROWS;
    }
    for s in 0..n {
        let r = find(&mut parent, s);
        if group_ok[&r] && group_tall[&r] {
            let len = plan.slot_len[s];
            let cols = len.clamp(1, WIDE_COLS);
            plan.slot_dims[s] = (len.div_ceil(cols), cols);
        }
    }
}
