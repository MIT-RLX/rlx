// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph → [`Plan`]: a flat, slot-indexed op list with all index arithmetic
//! (gather / reduce groups) precomputed in plain Rust so the CPU and WebGL
//! executors run byte-for-byte the same lowering.

use crate::host::HostOp;
use crate::linalg::LinalgOp;
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
    Tan,
    Atan,
    /// Round half to even; straight-through gradient (`1`).
    Round,
    Floor,
    Ceil,
    /// `sign(0) = 0`.
    Sign,
    /// `x > 0 ? x : eˣ − 1` (α = 1).
    Elu,
    /// Abramowitz–Stegun 7.1.26, as `Gelu` uses.
    Erf,
    /// `x · clamp(x + 3, 0, 6) / 6`.
    HardSwish,
    /// `clamp(x / 6 + 0.5, 0, 1)`.
    HardSigmoid,
    /// `x · tanh(softplus(x))`.
    Mish,
    /// `x / (1 + |x|)`.
    Softsign,
    /// `min(x, 0) − ln(1 + e^-|x|)`.
    LogSigmoid,
}

/// How [`Step::ScatterRuntime`] folds an update into its output element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterRed {
    /// Overwrite (the last update in index order wins).
    Replace,
    Add,
    Mul,
    Max,
    Min,
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
    /// With `wrap > 0` a negative index counts from the end (`+ wrap`) and the
    /// result is clamped to `[0, wrap)` (ONNX `GatherElements`, as rlx-cpu).
    GatherRuntime {
        out: usize,
        table: usize,
        indices: usize,
        which: Vec<u32>,
        base: Vec<u32>,
        axis_stride: u32,
        wrap: u32,
    },
    /// Scatter with a runtime index tensor, one output element per fragment
    /// (WebGL has no atomics, so each output gathers the updates that land on
    /// it): `out[o]` starts at `data[o]` (or 0 without `data`), then for
    /// `j in 0..count`, every `updates[ubase[o] + j·ustride]` whose index
    /// `round(indices[ibase[o] + j·istride])` (negative `+ wrap`) equals
    /// `key[o]` is folded in by `red`, in `j` order. `ibase[o] = PAD` = none.
    ScatterRuntime {
        out: usize,
        data: Option<usize>,
        updates: usize,
        indices: usize,
        ibase: Vec<u32>,
        ubase: Vec<u32>,
        key: Vec<u32>,
        count: usize,
        istride: usize,
        ustride: usize,
        wrap: u32,
        red: ScatterRed,
    },
    /// Stable sort along one axis of `src` viewed as `[outer, len, inner]`,
    /// keeping the first `k` positions: output `[outer, k, inner]` holds the
    /// values (or f32 indices) of rank `0..k`. Rank = number of elements that
    /// sort before (ties: smaller index first) — `Sort`/`ArgSort`/`TopK`.
    SortAxis {
        out: usize,
        src: usize,
        outer: usize,
        len: usize,
        inner: usize,
        k: usize,
        descending: bool,
        indices: bool,
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
    /// One radix-2 Stockham pass of a power-of-two FFT of length `n` over
    /// every row (one fragment per output lane). Output element `k`, with
    /// `j = (k / 2ns)·ns + k mod ns`: `x[j] ± w · x[j + n/2]`, `+` in the first
    /// `ns` of each `2ns` block, `w = tw[ns − 1 + k mod ns]` (complex, from the
    /// host-computed table). Passes run `ns = 1, 2, …, n/2` and leave the result
    /// in natural order. Rows are 2N-block (`[re… | im…]`) when `*_block`,
    /// else interleaved; `scale` multiplies the output.
    FftStage {
        out: usize,
        src: usize,
        tw: usize,
        n: usize,
        ns: usize,
        in_block: bool,
        out_block: bool,
        scale: f32,
    },
    /// An op with no WebGL lowering, run on rlx-cpu (see [`crate::host`]):
    /// operand slots in, output slot out, read and written linearly.
    Host {
        out: usize,
        inputs: Vec<usize>,
        op: std::sync::Arc<HostOp>,
    },
    /// Dense linear algebra on `batch` `[n, n]` matrices (and `[n, nrhs]`
    /// right-hand sides): run on the host by both executors — see
    /// [`crate::linalg`].
    Linalg {
        out: usize,
        a: usize,
        b: Option<usize>,
        n: usize,
        nrhs: usize,
        batch: usize,
        op: LinalgOp,
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
    /// NCDHW 3-D convolution computed directly, one output element per
    /// fragment summing its `cin/groups · kD·kH·kW` taps, plus `bias[co]`
    /// when given. Weight `[cout, cin/groups, kD, kH, kW]`.
    Conv3d {
        out: usize,
        x: usize,
        w: usize,
        bias: Option<usize>,
        n: usize,
        cin: usize,
        din: [usize; 3],
        cout: usize,
        dout: [usize; 3],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        dil: [usize; 3],
        groups: usize,
        /// applied to each output element as it is written (`fold_activations`)
        act: Option<Act>,
    },
    /// ArgMax / ArgMin along the middle axis of `src` viewed as
    /// `[outer, len, inner]` → `[outer, inner]` f32 indices (first extreme
    /// wins), positions computed rather than read from a table.
    ArgAxis {
        out: usize,
        src: usize,
        outer: usize,
        len: usize,
        inner: usize,
        is_max: bool,
    },
    /// A 1×1×1 convolution and the argmax over its output channels in one
    /// pass: `out[n, p] = argmax_co (bias[co] + Σ_ci w[co, ci] · x[n, ci, p])`.
    /// The per-class scores — classes × voxels — are never stored
    /// (`fuse_conv1_argmax`).
    Conv1ArgMax {
        out: usize,
        x: usize,
        w: usize,
        bias: Option<usize>,
        n: usize,
        cin: usize,
        cout: usize,
        spatial: usize,
    },
    /// GroupNorm on `[n, c, spatial]`: each group's `c/groups` channels are
    /// one contiguous run; `(x − mean)/√(var + eps) · gamma[c] + beta[c]`.
    GroupNorm {
        out: usize,
        x: usize,
        gamma: usize,
        beta: usize,
        n: usize,
        c: usize,
        groups: usize,
        spatial: usize,
        eps: f32,
        /// applied to each output element as it is written (`fold_activations`)
        act: Option<Act>,
    },
    /// Numpy broadcasting by arithmetic: output element at coordinates `c`
    /// (in `out_dims`) reads `src[Σ c[k] · in_strides[k]]`, a stride 0 on a
    /// broadcast axis. At most [`BROADCAST_RANK`] axes.
    Broadcast {
        out: usize,
        src: usize,
        out_dims: Vec<usize>,
        in_strides: Vec<usize>,
    },
    /// Every element of `out` = the one element of `src` (a scalar
    /// broadcast: a table would hold one index per output element).
    Splat {
        out: usize,
        src: usize,
    },
    /// `out = src`, element for element (a Reshape whose texture shape
    /// changes).
    Copy {
        out: usize,
        src: usize,
    },
    /// Two tensors joined along one axis, each viewed as `[outer, len]` with
    /// `len` the axis times everything after it: output row `o` is `a`'s row
    /// then `b`'s. (More inputs chain.)
    Concat {
        out: usize,
        a: usize,
        b: usize,
        outer: usize,
        a_len: usize,
        b_len: usize,
    },
    /// NCDHW max / average pooling (`nc` = batch × channels); padding is
    /// skipped by a max and counts as zero in an average (rlx-cpu's
    /// `count_include_pad`).
    Pool3d {
        out: usize,
        src: usize,
        nc: usize,
        din: [usize; 3],
        dout: [usize; 3],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        is_max: bool,
    },
    /// NCDHW nearest-neighbour resize: `src = min(dst · in / out, in − 1)`.
    Resize3d {
        out: usize,
        src: usize,
        nc: usize,
        din: [usize; 3],
        dout: [usize; 3],
    },
}

impl Step {
    /// The slot this step writes.
    pub fn out(&self) -> usize {
        match self {
            Step::Leaf { out, .. }
            | Step::Unary { out, .. }
            | Step::ActBack { out, .. }
            | Step::Binary { out, .. }
            | Step::Compare { out, .. }
            | Step::Where { out, .. }
            | Step::MatMul { out, .. }
            | Step::Gather { out, .. }
            | Step::Reduce { out, .. }
            | Step::Softmax { out, .. }
            | Step::LayerNorm { out, .. }
            | Step::RmsNorm { out, .. }
            | Step::ArgReduce { out, .. }
            | Step::GatherRuntime { out, .. }
            | Step::ScatterRuntime { out, .. }
            | Step::SortAxis { out, .. }
            | Step::ComplexCast { out, .. }
            | Step::BinaryC64 { out, .. }
            | Step::FftStage { out, .. }
            | Step::Host { out, .. }
            | Step::Linalg { out, .. }
            | Step::Custom { out, .. }
            | Step::Conv3d { out, .. }
            | Step::ArgAxis { out, .. }
            | Step::Conv1ArgMax { out, .. }
            | Step::GroupNorm { out, .. }
            | Step::Copy { out, .. }
            | Step::Splat { out, .. }
            | Step::Broadcast { out, .. }
            | Step::Concat { out, .. }
            | Step::Pool3d { out, .. }
            | Step::Resize3d { out, .. } => *out,
        }
    }

    /// The slots this step reads — what the GL executor frees a texture
    /// after the last of.
    pub fn reads(&self) -> Vec<usize> {
        match self {
            Step::Leaf { .. } => vec![],
            Step::Unary { a, .. } | Step::Softmax { a, .. } => vec![*a],
            Step::ActBack { x, dy, .. } => vec![*x, *dy],
            Step::Binary { a, b, .. }
            | Step::Compare { a, b, .. }
            | Step::BinaryC64 { a, b, .. } => {
                vec![*a, *b]
            }
            Step::MatMul { a, b, .. } => vec![*a, *b],
            Step::Where { cond, a, b, .. } => vec![*cond, *a, *b],
            Step::Gather { src, .. }
            | Step::Reduce { src, .. }
            | Step::ArgReduce { src, .. }
            | Step::SortAxis { src, .. }
            | Step::ComplexCast { src, .. }
            | Step::ArgAxis { src, .. }
            | Step::Copy { src, .. }
            | Step::Splat { src, .. }
            | Step::Broadcast { src, .. }
            | Step::Pool3d { src, .. }
            | Step::Resize3d { src, .. } => vec![*src],
            Step::Concat { a, b, .. } => vec![*a, *b],
            Step::LayerNorm { x, gamma, beta, .. }
            | Step::RmsNorm { x, gamma, beta, .. }
            | Step::GroupNorm { x, gamma, beta, .. } => vec![*x, *gamma, *beta],
            Step::GatherRuntime { table, indices, .. } => vec![*table, *indices],
            Step::ScatterRuntime {
                data,
                updates,
                indices,
                ..
            } => {
                let mut v = vec![*updates, *indices];
                v.extend(data);
                v
            }
            Step::FftStage { src, tw, .. } => vec![*src, *tw],
            Step::Host { inputs, .. } => inputs.clone(),
            Step::Linalg { a, b, .. } => {
                let mut v = vec![*a];
                v.extend(b);
                v
            }
            Step::Custom { input, .. } => vec![*input],
            Step::Conv3d { x, w, bias, .. } | Step::Conv1ArgMax { x, w, bias, .. } => {
                let mut v = vec![*x, *w];
                v.extend(bias);
                v
            }
        }
    }

    /// Rewrite every slot this step reads ([`Step::reads`]) through `f`.
    pub fn remap_reads(&mut self, f: impl Fn(usize) -> usize) {
        let g = |s: &mut usize| *s = f(*s);
        match self {
            Step::Leaf { .. } => {}
            Step::Unary { a, .. } | Step::Softmax { a, .. } => g(a),
            Step::ActBack { x, dy, .. } => {
                g(x);
                g(dy);
            }
            Step::Binary { a, b, .. }
            | Step::Compare { a, b, .. }
            | Step::BinaryC64 { a, b, .. }
            | Step::MatMul { a, b, .. }
            | Step::Concat { a, b, .. } => {
                g(a);
                g(b);
            }
            Step::Where { cond, a, b, .. } => {
                g(cond);
                g(a);
                g(b);
            }
            Step::Gather { src, .. }
            | Step::Reduce { src, .. }
            | Step::ArgReduce { src, .. }
            | Step::SortAxis { src, .. }
            | Step::ComplexCast { src, .. }
            | Step::ArgAxis { src, .. }
            | Step::Copy { src, .. }
            | Step::Splat { src, .. }
            | Step::Broadcast { src, .. }
            | Step::Pool3d { src, .. }
            | Step::Resize3d { src, .. } => g(src),
            Step::LayerNorm { x, gamma, beta, .. }
            | Step::RmsNorm { x, gamma, beta, .. }
            | Step::GroupNorm { x, gamma, beta, .. } => {
                g(x);
                g(gamma);
                g(beta);
            }
            Step::GatherRuntime { table, indices, .. } => {
                g(table);
                g(indices);
            }
            Step::ScatterRuntime {
                data,
                updates,
                indices,
                ..
            } => {
                g(updates);
                g(indices);
                if let Some(d) = data {
                    g(d);
                }
            }
            Step::FftStage { src, tw, .. } => {
                g(src);
                g(tw);
            }
            Step::Host { inputs, .. } => inputs.iter_mut().for_each(g),
            Step::Linalg { a, b, .. } => {
                g(a);
                if let Some(b) = b {
                    g(b);
                }
            }
            Step::Custom { input, .. } => g(input),
            Step::Conv3d { x, w, bias, .. } | Step::Conv1ArgMax { x, w, bias, .. } => {
                g(x);
                g(w);
                if let Some(b) = bias {
                    g(b);
                }
            }
        }
    }
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
        OpKind::GroupNorm,
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
        OpKind::Conv3d,
        OpKind::ConvTranspose3d,
        OpKind::Im2Col,
        OpKind::ArgMax,
        OpKind::ArgMin,
        OpKind::Gather,
        OpKind::GatherElements,
        OpKind::GatherNd,
        OpKind::ScatterAdd,
        OpKind::ScatterNd,
        OpKind::ScatterElements,
        OpKind::Sort,
        OpKind::ArgSort,
        OpKind::TopK,
        OpKind::LayerNorm2d,
        OpKind::Interpolate3d,
        OpKind::Fft,
        OpKind::RngUniform,
        OpKind::RngNormal,
        OpKind::Cholesky,
        OpKind::TriangularSolve,
        OpKind::DenseSolve,
        OpKind::BatchedDenseSolve,
        OpKind::Det,
        OpKind::LogDet,
        OpKind::Rope,
        // Host/transport custom ops (`collective.*`). Kept in the supported set
        // so legalization preserves the node; `lower` restricts to the
        // collective names and the CPU executor host-delegates them (native
        // only — see `exec_cpu`).
        OpKind::Custom,
    ]
}

/// Every op a WebGL plan can run: the native set ([`supported_ops`]) plus
/// what rlx-cpu runs, which the planner hands to the host fallback
/// ([`crate::host`]). What a runtime backend should claim. Legalization still
/// lowers against [`supported_ops`] alone, so an op with a native lowering
/// (`GroupNorm`, `Pad`, …) keeps it even though rlx-cpu also claims it.
pub fn runnable_ops() -> &'static [OpKind] {
    static OPS: std::sync::OnceLock<Vec<OpKind>> = std::sync::OnceLock::new();
    OPS.get_or_init(|| {
        let mut v = supported_ops().to_vec();
        v.extend(
            rlx_cpu::SUPPORTED_OPS
                .iter()
                .filter(|k| !supported_ops().contains(k)),
        );
        v
    })
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

pub(crate) fn map_act(a: Activation) -> Result<Act> {
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
        Activation::Tan => Act::Tan,
        Activation::Atan => Act::Atan,
        Activation::Round => Act::Round,
        Activation::Floor => Act::Floor,
        Activation::Ceil => Act::Ceil,
        Activation::Sign => Act::Sign,
        Activation::Elu => Act::Elu,
        Activation::Erf => Act::Erf,
        Activation::HardSwish => Act::HardSwish,
        Activation::HardSigmoid => Act::HardSigmoid,
        Activation::Mish => Act::Mish,
        Activation::Softsign => Act::Softsign,
        Activation::LogSigmoid => Act::LogSigmoid,
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

/// FFT of length `n` along each of `rows` rows of `src` into `out`, output
/// scaled by `scale`. `block` = 2N-block layout (else interleaved), used for
/// both the input and the output.
///
/// A power-of-two `n` is `log₂ n` [`Step::FftStage`] passes. Any other `n` is
/// Bluestein: `X_k = c_k · Σ_m (x_m c_m) · b_{k−m}` with the chirp
/// `c_m = e^{σπi m²/n}`, `b_m = c̄_m` — a length-`M` (power of two ≥ 2n − 1)
/// circular convolution done as FFT · (host-precomputed FFT of `b`) · IFFT.
/// Every constant (twiddles, chirp, filter spectrum) is computed in f64 on the
/// host, so the shaders evaluate no `sin`/`cos`. O(n log n) per row.
#[allow(clippy::too_many_arguments)]
fn emit_fft(
    b: &mut Builder,
    src: usize,
    out: usize,
    rows: usize,
    n: usize,
    block: bool,
    inverse: bool,
    scale: f64,
) {
    let sign = if inverse { 1.0 } else { -1.0 };
    if n <= 1 {
        // Length-1 transforms are the identity (every norm scales by 1).
        b.emit(Step::Gather {
            out,
            src,
            idx: (0..(rows * 2 * n) as u32).collect(),
        });
        return;
    }
    if n.is_power_of_two() {
        fft_passes(b, src, out, rows, n, (block, block), sign, scale as f32);
        return;
    }
    let m = (2 * n - 1).next_power_of_two();
    let lane = |k: usize, c: usize| if block { c * n + k } else { 2 * k + c };
    // x zero-padded to M, interleaved.
    let x = fft_slot(b, rows * m);
    b.emit(Step::Gather {
        out: x,
        src,
        idx: (0..rows * 2 * m)
            .map(|i| {
                let (r, k, c) = (i / (2 * m), i / 2 % m, i % 2);
                if k < n {
                    (r * 2 * n + lane(k, c)) as u32
                } else {
                    PAD
                }
            })
            .collect(),
    });
    // e^{σπi k²/n}, with k² mod 2n keeping the angle small.
    let chirp = |k: usize| {
        let th = sign * std::f64::consts::PI * ((k * k) % (2 * n)) as f64 / n as f64;
        (th.cos(), th.sin())
    };
    let cmul = |b: &mut Builder, a: usize, c: Vec<f32>| {
        let cs = fft_slot(b, m);
        b.emit(Step::Leaf {
            out: cs,
            src: LeafSource::Const(c),
        });
        let o = fft_slot(b, rows * m);
        b.emit(Step::BinaryC64 {
            out: o,
            a,
            b: cs,
            op: Bin::Mul,
            n: rows * m,
            n_a: rows * m,
            n_b: m,
        });
        o
    };
    let pre = (0..m)
        .flat_map(|k| {
            let (c, s) = if k < n { chirp(k) } else { (0.0, 0.0) };
            [c as f32, s as f32]
        })
        .collect();
    let xc = cmul(b, x, pre);
    let xf = fft_slot(b, rows * m);
    fft_passes(b, xc, xf, rows, m, (false, false), -1.0, 1.0);
    // FFT of b (b_k = c̄_k for |k| < n, circularly), pre-divided by M for the
    // unnormalized inverse below.
    let mut filt = vec![(0f64, 0f64); m];
    for k in 0..n {
        let (c, s) = chirp(k);
        filt[k] = (c, -s);
        filt[(m - k) % m] = (c, -s);
    }
    host_fft(&mut filt, -1.0);
    let spec = filt
        .iter()
        .flat_map(|&(re, im)| [(re / m as f64) as f32, (im / m as f64) as f32])
        .collect();
    let y = cmul(b, xf, spec);
    let conv = fft_slot(b, rows * m);
    fft_passes(b, y, conv, rows, m, (false, false), 1.0, 1.0);
    let post = (0..m)
        .flat_map(|k| {
            let (c, s) = if k < n { chirp(k) } else { (0.0, 0.0) };
            [(c * scale) as f32, (s * scale) as f32]
        })
        .collect();
    let z = cmul(b, conv, post);
    // First n of each row, back to the op's layout.
    b.emit(Step::Gather {
        out,
        src: z,
        idx: (0..rows * 2 * n)
            .map(|i| {
                let (r, l) = (i / (2 * n), i % (2 * n));
                let (k, c) = if block {
                    (l % n, l / n)
                } else {
                    (l / 2, l % 2)
                };
                (r * 2 * m + 2 * k + c) as u32
            })
            .collect(),
    });
}

/// The `log₂ n` Stockham passes of a power-of-two FFT (`sign` −1 forward,
/// +1 inverse), reading `layout.0`, writing `layout.1` (true = 2N-block);
/// intermediates are interleaved.
#[allow(clippy::too_many_arguments)]
fn fft_passes(
    b: &mut Builder,
    src: usize,
    out: usize,
    rows: usize,
    n: usize,
    layout: (bool, bool),
    sign: f64,
    scale: f32,
) {
    // Twiddles of every pass: entry `ns − 1 + j` = e^{σπi j/ns}, j < ns.
    let mut tw = Vec::with_capacity(2 * n);
    let mut ns = 1;
    while ns < n {
        for j in 0..ns {
            let th = sign * std::f64::consts::PI * j as f64 / ns as f64;
            tw.extend([th.cos() as f32, th.sin() as f32]);
        }
        ns *= 2;
    }
    tw.resize(2 * n, 0.0);
    let tw_slot = fft_slot(b, n);
    b.emit(Step::Leaf {
        out: tw_slot,
        src: LeafSource::Const(tw),
    });
    let (mut cur, mut ns) = (src, 1);
    while ns < n {
        let last = 2 * ns == n;
        let dst = if last { out } else { fft_slot(b, rows * n) };
        b.emit(Step::FftStage {
            out: dst,
            src: cur,
            tw: tw_slot,
            n,
            ns,
            in_block: ns == 1 && layout.0,
            out_block: last && layout.1,
            scale: if last { scale } else { 1.0 },
        });
        cur = dst;
        ns *= 2;
    }
}

/// Lower `node` to a [`Step::Host`] (rlx-cpu runs it), if rlx-cpu can.
fn emit_host(
    b: &mut Builder,
    graph: &Graph,
    node: &rlx_ir::Node,
    out: usize,
    inputs: Vec<usize>,
) -> Result<()> {
    let kind = node.op.kind();
    if !rlx_cpu::SUPPORTED_OPS.contains(&kind) {
        return Err(WebglError(format!("op {kind:?} not supported on WebGL")));
    }
    let shapes: Vec<rlx_ir::Shape> = node
        .inputs
        .iter()
        .map(|&i| graph.node(i).shape.clone())
        .collect();
    b.emit(Step::Host {
        out,
        inputs,
        op: std::sync::Arc::new(HostOp::new(&node.op, &shapes, &node.shape)),
    });
    Ok(())
}

/// A slot of `count` complex values (interleaved f32 lanes) for FFT
/// intermediates. Every FFT step addresses them linearly, so rows are capped at
/// 4096 lanes rather than one transform wide: a long transform's rows would
/// otherwise reach `MAX_TEXTURE_SIZE` (Bluestein pads to 2× the length).
fn fft_slot(b: &mut Builder, count: usize) -> usize {
    // Largest power of two ≤ 2048 dividing `count` (every count here is a
    // multiple of the power-of-two transform length).
    let cols = 1usize << count.trailing_zeros().min(11);
    b.alloc_lanes(&[count / cols, cols], 2)
}

/// In-place iterative radix-2 FFT in f64 (power-of-two length), for the
/// plan-time Bluestein filter spectrum.
fn host_fft(a: &mut [(f64, f64)], sign: f64) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let th = sign * std::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((th * k as f64).cos(), (th * k as f64).sin());
                let (ur, ui) = a[start + k];
                let (vr, vi) = a[start + k + len / 2];
                let (tr, ti) = (vr * wr - vi * wi, vr * wi + vi * wr);
                a[start + k] = (ur + tr, ui + ti);
                a[start + k + len / 2] = (ur - tr, ui - ti);
            }
        }
        len *= 2;
    }
}

/// Axis `a` of a rank-`rank` tensor, negative counting from the end.
fn norm_axis(a: i32, rank: usize) -> Result<usize> {
    let r = rank as i32;
    let ax = if a < 0 { a + r } else { a };
    if ax < 0 || ax >= r {
        return Err(WebglError(format!("axis {a} out of range for rank {rank}")));
    }
    Ok(ax as usize)
}

fn scatter_red(r: rlx_ir::op::ScatterNdReduction) -> ScatterRed {
    use rlx_ir::op::ScatterNdReduction as R;
    match r {
        R::None => ScatterRed::Replace,
        R::Add => ScatterRed::Add,
        R::Mul => ScatterRed::Mul,
        R::Max => ScatterRed::Max,
        R::Min => ScatterRed::Min,
    }
}

/// `nq` index tuples `idx [nq, k]` (k = `dims.len()`) → `[nq]` linear row
/// numbers `Σ_m wrap(idx[q, m]) · Π dims[m+1..]`, a negative component counting
/// from the end of its axis (ONNX GatherND / ScatterND, as rlx-cpu).
fn linearize_tuples(b: &mut Builder, idx: usize, nq: usize, dims: &[usize]) -> usize {
    let k = dims.len();
    let n = nq * k;
    let row_strides = strides(dims);
    let leaf = |b: &mut Builder, f: &dyn Fn(usize) -> f32| {
        let s = b.alloc(&[nq, k]);
        b.emit(Step::Leaf {
            out: s,
            src: LeafSource::Const((0..n).map(f).collect()),
        });
        s
    };
    let zero = leaf(b, &|_| 0.0);
    let axis_len = leaf(b, &|i| dims[i % k] as f32);
    let stride = leaf(b, &|i| row_strides[i % k] as f32);
    let bin = |b: &mut Builder, a: usize, c: usize, op: Bin| {
        let s = b.alloc(&[nq, k]);
        b.emit(Step::Binary {
            out: s,
            a,
            b: c,
            op,
        });
        s
    };
    let neg = b.alloc(&[nq, k]);
    b.emit(Step::Compare {
        out: neg,
        a: idx,
        b: zero,
        cmp: Cmp::Lt,
    });
    let shift = bin(b, neg, axis_len, Bin::Mul);
    let wrapped = bin(b, idx, shift, Bin::Add);
    let scaled = bin(b, wrapped, stride, Bin::Mul);
    let lin = b.alloc(&[nq]);
    b.emit(Step::Reduce {
        out: lin,
        src: scaled,
        groups: (0..n as u32).collect(),
        fanin: k,
        op: Red::Sum,
    });
    lin
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

/// A constant's bytes as the f32 values every slot holds: integer / bool /
/// half / f64 constants (index tables built by legalization, e.g. `LowerSlice`'s
/// I64 indices) are widened by value. C64 is already f32 lanes; C128 is
/// re-laid as df64 lanes `[hi, lo]` per component.
pub(crate) fn decode_const(bytes: &[u8], dt: DType) -> Vec<f32> {
    fn le<const N: usize>(bytes: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
        bytes.as_chunks::<N>().0.iter().copied()
    }
    let half = |h: u16| -> f32 {
        let (sign, exp, man) = (
            (h >> 15) as u32,
            ((h >> 10) & 0x1f) as i32,
            (h & 0x3ff) as f32,
        );
        let v = match exp {
            0 => man * 2f32.powi(-24),
            31 if man == 0.0 => f32::INFINITY,
            31 => f32::NAN,
            e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
        };
        if sign == 1 { -v } else { v }
    };
    match dt {
        DType::F64 => le::<8>(bytes)
            .map(|c| f64::from_le_bytes(c) as f32)
            .collect(),
        DType::C128 => le::<8>(bytes)
            .flat_map(|c| {
                let v = f64::from_le_bytes(c);
                let hi = v as f32;
                [hi, (v - hi as f64) as f32]
            })
            .collect(),
        DType::F16 => le::<2>(bytes)
            .map(|c| half(u16::from_le_bytes(c)))
            .collect(),
        DType::BF16 => le::<2>(bytes)
            .map(|c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16))
            .collect(),
        DType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => bytes.iter().map(|&b| b as f32).collect(),
        DType::I16 => le::<2>(bytes)
            .map(|c| i16::from_le_bytes(c) as f32)
            .collect(),
        DType::I32 => le::<4>(bytes)
            .map(|c| i32::from_le_bytes(c) as f32)
            .collect(),
        DType::U32 => le::<4>(bytes)
            .map(|c| u32::from_le_bytes(c) as f32)
            .collect(),
        DType::I64 => le::<8>(bytes)
            .map(|c| i64::from_le_bytes(c) as f32)
            .collect(),
        _ => decode_f32_le(bytes),
    }
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `out[o] = Σ_r x[x_idx[o·red + r]] · w[w_idx[o·red + r]]` (`PAD` in `x_idx`
/// reads zero): gather each output's window and its weights side by side,
/// multiply, and sum the window. Grouped / transposed convs, where outputs do
/// not share one `[patch, cout]` matmul.
fn window_dot(
    b: &mut Builder,
    x: usize,
    w: usize,
    out: usize,
    x_idx: Vec<u32>,
    w_idx: Vec<u32>,
    red: usize,
) {
    let len = x_idx.len();
    let p_slot = b.alloc(&[len / red, red]);
    b.emit(Step::Gather {
        out: p_slot,
        src: x,
        idx: x_idx,
    });
    let w_slot = b.alloc(&[len / red, red]);
    b.emit(Step::Gather {
        out: w_slot,
        src: w,
        idx: w_idx,
    });
    let m_slot = b.alloc(&[len / red, red]);
    b.emit(Step::Binary {
        out: m_slot,
        a: p_slot,
        b: w_slot,
        op: Bin::Mul,
    });
    b.emit(Step::Reduce {
        out,
        src: m_slot,
        groups: (0..len as u32).collect(),
        fanin: red,
        op: Red::Sum,
    });
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
/// ([`rlx_compile::rewrite_for_backend`]) — this lowers
/// BatchNormInference, SoftmaxCrossEntropy, DotGeneral, control flow, fused /
/// RNN ops and every `*Backward` op down to the small kernel set the WebGL
/// executor implements. Then each remaining node is lowered to a [`Step`].
pub fn build_plan(graph: &Graph) -> Result<Plan> {
    let g = fold_conv3d_bias(graph.clone());
    let g = rlx_autodiff::decompose_backward_ops_except(g, NATIVE_BACKWARD);
    let g = rlx_fusion::unfuse_dit_modulation(g);
    let g = rlx_compile::rewrite_for_backend(g, supported_ops());
    lower(&g)
}

/// Most axes a [`Step::Broadcast`] takes (its shader's uniform arrays).
pub const BROADCAST_RANK: usize = 8;

/// `in_dims` broadcast to `out_dims` as per-output-axis strides into the
/// input (numpy rules, right-aligned; 0 where the input is 1 or absent).
fn broadcast_strides(in_dims: &[usize], out_dims: &[usize]) -> Vec<usize> {
    let st = strides(in_dims);
    let offset = out_dims.len() - in_dims.len();
    (0..out_dims.len())
        .map(|k| {
            if k < offset || in_dims[k - offset] == 1 {
                0
            } else {
                st[k - offset]
            }
        })
        .collect()
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
        if total(in_dims) == 1 {
            let out = self.alloc(out_dims);
            self.emit(Step::Splat { out, src });
            return out;
        }
        if out_dims.len() <= BROADCAST_RANK && in_dims.len() <= out_dims.len() {
            let out = self.alloc(out_dims);
            self.emit(Step::Broadcast {
                out,
                src,
                out_dims: out_dims.to_vec(),
                in_strides: broadcast_strides(in_dims, out_dims),
            });
            return out;
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
                src: LeafSource::Const(decode_const(data, node_dt)),
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
                let src = input_slot(0)?;
                if node_lane > 1 {
                    b.emit(Step::Gather {
                        out,
                        src,
                        idx: identity(),
                    });
                } else if rows_cols(&input_dims(0)?) == (r, c) {
                    // same texture shape: the node *is* its input (the slot
                    // allocated for it stays empty)
                    id_info.insert(node.id, (src, dims.clone(), node_dt));
                } else {
                    // same elements, another shape: a linear copy (an
                    // identity gather would upload an index per element)
                    b.emit(Step::Copy { out, src });
                }
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
                let (mut in_l, mut out_l) = (in_dims.clone(), dims.clone());
                if node_lane > 1 {
                    in_l.push(node_lane);
                    out_l.push(node_lane);
                }
                if out_l.len() <= BROADCAST_RANK {
                    // by arithmetic: a table is one index per output element
                    b.emit(Step::Broadcast {
                        out,
                        src: input_slot(0)?,
                        in_strides: broadcast_strides(&in_l, &out_l),
                        out_dims: out_l,
                    });
                } else {
                    b.emit(Step::Gather {
                        out,
                        src: input_slot(0)?,
                        idx: broadcast_idx(&in_l, &out_l),
                    });
                }
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
            Op::Concat { axis } if node_lane == 1 => {
                // Pairwise, by arithmetic: no index table or mask the size
                // of the output (two of each per input otherwise).
                let outer: usize = dims[..*axis].iter().product();
                let row = |d: &[usize]| d[*axis..].iter().product::<usize>();
                let (mut acc, mut acc_len) = (input_slot(0)?, row(&input_dims(0)?));
                for ii in 1..node.inputs.len() {
                    let b_len = row(&input_dims(ii)?);
                    let last = ii + 1 == node.inputs.len();
                    let dst = if last {
                        out
                    } else {
                        let mut d = dims.clone();
                        d[*axis] =
                            (acc_len + b_len) / dims[*axis + 1..].iter().product::<usize>().max(1);
                        b.alloc(&d)
                    };
                    b.emit(Step::Concat {
                        out: dst,
                        a: acc,
                        b: input_slot(ii)?,
                        outer,
                        a_len: acc_len,
                        b_len,
                    });
                    (acc, acc_len) = (dst, acc_len + b_len);
                }
                if node.inputs.len() == 1 {
                    b.emit(Step::Copy { out, src: acc });
                }
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
            } if kernel_size.len() == 3 => {
                let in_dims = input_dims(0)?;
                if in_dims.len() != 5 || dims.len() != 5 {
                    return Err(WebglError("3-D pool requires 5D NCDHW".into()));
                }
                let is_max = match kind {
                    ReduceOp::Max => true,
                    ReduceOp::Mean => false,
                    other => return Err(WebglError(format!("3-D pool {other:?} not supported"))),
                };
                let a3 = |v: &Vec<usize>| [v[0], v[1], v[2]];
                b.emit(Step::Pool3d {
                    out,
                    src: input_slot(0)?,
                    nc: in_dims[0] * in_dims[1],
                    din: [in_dims[2], in_dims[3], in_dims[4]],
                    dout: [dims[2], dims[3], dims[4]],
                    k: a3(kernel_size),
                    stride: a3(stride),
                    pad: a3(padding),
                    is_max,
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
            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                // Direct: one output element per fragment summing its taps.
                // (An im2col matrix for a volume is `cin·k³` times the volume:
                // a 30-channel 3×3×3 layer over a 164×164×224 brain crop
                // would be 4.9e9 entries.) input [N,Cin,D,H,W], weight
                // [Cout,Cin/g,kD,kH,kW], out [N,Cout,Do,Ho,Wo]; an optional
                // third input is a per-channel bias (`fold_conv3d_bias`).
                let in_dims = input_dims(0)?;
                let w_dims = input_dims(1)?;
                if in_dims.len() != 5 || w_dims.len() != 5 || dims.len() != 5 {
                    return Err(WebglError("conv3d requires 5D NCDHW".into()));
                }
                let g = *groups;
                let (cin, cout) = (in_dims[1], w_dims[0]);
                if g == 0 || cin % g != 0 || cout % g != 0 || w_dims[1] != cin / g {
                    return Err(WebglError(format!(
                        "conv3d groups={g}: cin {cin} / cout {cout} / weight {w_dims:?} do not divide"
                    )));
                }
                let bias = if node.inputs.len() > 2 {
                    Some(input_slot(2)?)
                } else {
                    None
                };
                b.emit(Step::Conv3d {
                    out,
                    x: input_slot(0)?,
                    w: input_slot(1)?,
                    bias,
                    n: in_dims[0],
                    cin,
                    din: [in_dims[2], in_dims[3], in_dims[4]],
                    cout,
                    dout: [dims[2], dims[3], dims[4]],
                    k: [w_dims[2], w_dims[3], w_dims[4]],
                    stride: *stride,
                    pad: *padding,
                    dil: *dilation,
                    groups: g,
                    act: None,
                });
            }
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                // Gather form of the CPU scatter kernel: input [N,Cin,D,H,W],
                // weight [Cin,Cout/g,kD,kH,kW], out [N,Cout,Do,Ho,Wo] (output
                // padding is already folded into the output shape).
                //   out[n,oc,o] = Σ_{ic∈grp(oc), t} x[n,ic,i] · w[ic, oc mod Cout/g, t]
                // over taps with o + p − t·dil = i·s, 0 ≤ i < in.
                let in_dims = input_dims(0)?;
                let w_dims = input_dims(1)?;
                if in_dims.len() != 5 || w_dims.len() != 5 || dims.len() != 5 {
                    return Err(WebglError("conv_transpose3d requires 5D NCDHW".into()));
                }
                let g = *groups;
                let (cin, cout) = (in_dims[1], dims[1]);
                if g == 0
                    || cin % g != 0
                    || cout % g != 0
                    || w_dims[0] != cin
                    || w_dims[1] != cout / g
                {
                    return Err(WebglError(format!(
                        "conv_transpose3d groups={g}: cin {cin} / cout {cout} / weight {w_dims:?} do not divide"
                    )));
                }
                let (cin_g, cout_g) = (cin / g, cout / g);
                let ks = [w_dims[2], w_dims[3], w_dims[4]];
                let kvol = ks[0] * ks[1] * ks[2];
                let red = cin_g * kvol;
                let osp = [dims[2], dims[3], dims[4]];
                let npos_n = osp[0] * osp[1] * osp[2];
                let xs = strides(&in_dims);
                let unravel3 =
                    |p: usize, e: [usize; 3]| [p / (e[1] * e[2]), p / e[2] % e[1], p % e[2]];
                let out_n = total(&dims);
                let mut x_idx = vec![PAD; out_n * red];
                let mut w_idx = vec![0u32; out_n * red];
                for o in 0..out_n {
                    let (n, oc, p) = (o / (cout * npos_n), o / npos_n % cout, o % npos_n);
                    let op = unravel3(p, osp);
                    let (grp, oc_off) = (oc / cout_g, oc % cout_g);
                    for r in 0..red {
                        let (ic, tl) = (grp * cin_g + r / kvol, r % kvol);
                        let t = unravel3(tl, ks);
                        w_idx[o * red + r] = ((ic * cout_g + oc_off) * kvol + tl) as u32;
                        let mut lin = Some(n * xs[0] + ic * xs[1]);
                        for a in 0..3 {
                            let num = (op[a] + padding[a]).checked_sub(t[a] * dilation[a]);
                            lin = match (lin, num) {
                                (Some(l), Some(v))
                                    if v % stride[a] == 0 && v / stride[a] < in_dims[a + 2] =>
                                {
                                    Some(l + v / stride[a] * xs[a + 2])
                                }
                                _ => None,
                            };
                        }
                        if let Some(l) = lin {
                            x_idx[o * red + r] = l as u32;
                        }
                    }
                }
                window_dot(
                    &mut b,
                    input_slot(0)?,
                    input_slot(1)?,
                    out,
                    x_idx,
                    w_idx,
                    red,
                );
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
            Op::ArgMax { axis, .. } | Op::ArgMin { axis, .. } => {
                // The input viewed as [outer, len, inner]: positions are
                // arithmetic, so no per-output index table (which, over a
                // class axis of a volume, is classes × voxels entries).
                let in_dims = input_dims(0)?;
                b.emit(Step::ArgAxis {
                    out,
                    src: input_slot(0)?,
                    outer: in_dims[..*axis].iter().product(),
                    len: in_dims[*axis],
                    inner: in_dims[*axis + 1..].iter().product(),
                    is_max: matches!(node.op, Op::ArgMax { .. }),
                });
            }
            Op::GroupNorm { num_groups, eps } => {
                // [N, C, …] with the trailing axes as one contiguous run per
                // channel: per (n, group) mean and variance, then the affine.
                let in_dims = input_dims(0)?;
                if in_dims.len() < 2 || *num_groups == 0 || in_dims[1] % num_groups != 0 {
                    return Err(WebglError(format!(
                        "group_norm: {num_groups} groups over {in_dims:?}"
                    )));
                }
                b.emit(Step::GroupNorm {
                    out,
                    x: input_slot(0)?,
                    gamma: input_slot(1)?,
                    beta: input_slot(2)?,
                    n: in_dims[0],
                    c: in_dims[1],
                    groups: *num_groups,
                    spatial: in_dims[2..].iter().product(),
                    eps: *eps,
                    act: None,
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
                    wrap: 0,
                });
            }
            Op::GatherElements { axis } => {
                // out[o] = data[o with the axis coordinate replaced by idx[o]].
                let d = input_dims(0)?;
                let ax = norm_axis(*axis, d.len())?;
                let ds = strides(&d);
                let n = total(&dims);
                let base = (0..n)
                    .map(|o| {
                        let c = unravel(o, &dims);
                        (0..d.len())
                            .filter(|&k| k != ax)
                            .map(|k| c[k].min(d[k] - 1) * ds[k])
                            .sum::<usize>() as u32
                    })
                    .collect();
                b.emit(Step::GatherRuntime {
                    out,
                    table: input_slot(0)?,
                    indices: input_slot(1)?,
                    which: (0..n as u32).collect(),
                    base,
                    axis_stride: ds[ax] as u32,
                    wrap: d[ax] as u32,
                });
            }
            Op::GatherNd { batch_dims } => {
                // data [B.., D_0..D_{k-1}, rest..], indices [B.., Q.., k]: each
                // index tuple picks one `rest`-sized row.
                let d = input_dims(0)?;
                let id = input_dims(1)?;
                let bd = (*batch_dims).max(0) as usize;
                let k = *id
                    .last()
                    .ok_or_else(|| WebglError("gather_nd: scalar indices".into()))?;
                let tuples = total(&id[..id.len() - 1]);
                let rest = total(&d[bd + k..]);
                let per_batch = tuples / total(&d[..bd]).max(1);
                let batch_stride = total(&d[bd..]);
                let lin = linearize_tuples(&mut b, input_slot(1)?, tuples, &d[bd..bd + k]);
                let n = total(&dims);
                b.emit(Step::GatherRuntime {
                    out,
                    table: input_slot(0)?,
                    indices: lin,
                    which: (0..n).map(|o| (o / rest) as u32).collect(),
                    base: (0..n)
                        .map(|o| ((o / rest / per_batch) * batch_stride + o % rest) as u32)
                        .collect(),
                    axis_stride: rest as u32,
                    wrap: 0,
                });
            }
            Op::ScatterAdd { axis } => {
                // inputs [updates, indices]: out[.., idx[j], ..] += updates[.., j, ..].
                let u = input_dims(0)?;
                let ax = *axis;
                let (a_len, j) = (dims[ax], u[ax]);
                let post = total(&dims[ax + 1..]);
                let n = total(&dims);
                b.emit(Step::ScatterRuntime {
                    out,
                    data: None,
                    updates: input_slot(0)?,
                    indices: input_slot(1)?,
                    ibase: vec![0; n],
                    ubase: (0..n)
                        .map(|o| ((o / (a_len * post)) * j * post + o % post) as u32)
                        .collect(),
                    key: (0..n).map(|o| ((o / post) % a_len) as u32).collect(),
                    count: j,
                    istride: 1,
                    ustride: post,
                    wrap: 0,
                    red: ScatterRed::Add,
                });
            }
            Op::ScatterNd { reduction } => {
                // inputs [data, indices [Q.., k], updates [Q.., rest..]].
                let id = input_dims(1)?;
                let k = *id
                    .last()
                    .ok_or_else(|| WebglError("scatter_nd: scalar indices".into()))?;
                let nq = total(&id[..id.len() - 1]);
                let rest = total(&dims[k..]);
                let lin = linearize_tuples(&mut b, input_slot(1)?, nq, &dims[..k]);
                let n = total(&dims);
                b.emit(Step::ScatterRuntime {
                    out,
                    data: Some(input_slot(0)?),
                    updates: input_slot(2)?,
                    indices: lin,
                    ibase: vec![0; n],
                    ubase: (0..n).map(|o| (o % rest) as u32).collect(),
                    key: (0..n).map(|o| (o / rest) as u32).collect(),
                    count: nq,
                    istride: 1,
                    ustride: rest,
                    wrap: 0,
                    red: scatter_red(*reduction),
                });
            }
            Op::ScatterElements { axis, reduction } => {
                // inputs [data, indices, updates]; indices/updates share shape `u`.
                // out[c] gathers every update in its axis line whose index is c[axis].
                let u = input_dims(1)?;
                let ax = norm_axis(*axis, dims.len())?;
                let us = strides(&u);
                let n = total(&dims);
                let mut ibase = vec![PAD; n];
                let mut key = vec![0u32; n];
                for o in 0..n {
                    let c = unravel(o, &dims);
                    key[o] = c[ax] as u32;
                    if (0..dims.len()).all(|k| k == ax || c[k] < u[k]) {
                        ibase[o] = (0..dims.len())
                            .filter(|&k| k != ax)
                            .map(|k| c[k] * us[k])
                            .sum::<usize>() as u32;
                    }
                }
                b.emit(Step::ScatterRuntime {
                    out,
                    data: Some(input_slot(0)?),
                    updates: input_slot(2)?,
                    indices: input_slot(1)?,
                    ubase: ibase.clone(),
                    ibase,
                    key,
                    count: u[ax],
                    istride: us[ax],
                    ustride: us[ax],
                    wrap: dims[ax] as u32,
                    red: scatter_red(*reduction),
                });
            }
            Op::Sort { axis, descending } | Op::ArgSort { axis, descending } => {
                let d = input_dims(0)?;
                let ax = *axis;
                b.emit(Step::SortAxis {
                    out,
                    src: input_slot(0)?,
                    outer: total(&d[..ax]),
                    len: d[ax],
                    inner: total(&d[ax + 1..]),
                    k: d[ax],
                    descending: *descending,
                    indices: matches!(node.op, Op::ArgSort { .. }),
                });
            }
            Op::LayerNorm2d { eps } => {
                // Normalize over channels at each (n, h, w): move C last, run
                // the last-axis LayerNorm, move it back.
                let (n, c) = (dims[0], dims[1]);
                let sp = total(&dims[2..]);
                let rows = n * sp;
                let to_last = (0..rows * c)
                    .map(|i| {
                        let (r, ci) = (i / c, i % c);
                        ((r / sp * c + ci) * sp + r % sp) as u32
                    })
                    .collect();
                let x_t = b.alloc(&[rows, c]);
                b.emit(Step::Gather {
                    out: x_t,
                    src: input_slot(0)?,
                    idx: to_last,
                });
                let y_t = b.alloc(&[rows, c]);
                b.emit(Step::LayerNorm {
                    out: y_t,
                    x: x_t,
                    gamma: input_slot(1)?,
                    beta: input_slot(2)?,
                    rows,
                    cols: c,
                    eps: *eps,
                });
                let back = (0..rows * c)
                    .map(|o| {
                        let (ni, ci, si) = (o / (c * sp), o / sp % c, o % sp);
                        ((ni * sp + si) * c + ci) as u32
                    })
                    .collect();
                b.emit(Step::Gather {
                    out,
                    src: y_t,
                    idx: back,
                });
            }
            Op::Interpolate3d { .. } => {
                // Nearest: src = min(floor(dst · in / out), in − 1) per axis,
                // computed per fragment (a table would be one index per
                // output voxel).
                let d = input_dims(0)?;
                b.emit(Step::Resize3d {
                    out,
                    src: input_slot(0)?,
                    nc: d[0] * d[1],
                    din: [d[2], d[3], d[4]],
                    dout: [dims[2], dims[3], dims[4]],
                });
            }
            Op::Fft { inverse, norm, .. } => {
                // Rows are `[re… | im…]` (F32/F64 2N-block) or interleaved
                // `[re, im]` pairs (C64); see `emit_fft`.
                let (n, block) = match input_dt(0)? {
                    DType::C64 => (*dims.last().unwrap_or(&0), false),
                    DType::F32 | DType::F64 => (dims.last().unwrap_or(&0) / 2, true),
                    other => return Err(WebglError(format!("fft of {other:?} not supported"))),
                };
                let rows = total(&dims) * node_lane / (2 * n).max(1);
                let scale = norm.output_scale(n, *inverse);
                emit_fft(&mut b, input_slot(0)?, out, rows, n, block, *inverse, scale);
            }
            Op::RngUniform {
                low,
                high,
                key,
                op_seed,
            } => {
                // Counter-based and keyed by the node, so the values are fixed
                // by the graph: computed here with the default RNG policy (the
                // one an rlx-cpu session uses) and uploaded as a constant.
                let mut v = vec![0f32; total(&dims)];
                rlx_ir::fill_uniform_like(
                    &mut v,
                    *low,
                    *high,
                    rlx_ir::RngOptions::default(),
                    *key,
                    *op_seed,
                );
                b.emit(Step::Leaf {
                    out,
                    src: LeafSource::Const(v),
                });
            }
            Op::RngNormal {
                mean,
                scale,
                key,
                op_seed,
            } => {
                let mut v = vec![0f32; total(&dims)];
                rlx_ir::fill_normal_like(
                    &mut v,
                    *mean,
                    *scale,
                    rlx_ir::RngOptions::default(),
                    *key,
                    *op_seed,
                );
                b.emit(Step::Leaf {
                    out,
                    src: LeafSource::Const(v),
                });
            }
            Op::Cholesky
            | Op::Det
            | Op::LogDet
            | Op::TriangularSolve { .. }
            | Op::DenseSolve
            | Op::BatchedDenseSolve => {
                let a_dims = input_dims(0)?;
                let n = *a_dims.last().unwrap_or(&0);
                let batch = total(&a_dims) / (n * n).max(1);
                let rhs = matches!(
                    node.op,
                    Op::TriangularSolve { .. } | Op::DenseSolve | Op::BatchedDenseSolve
                );
                let op = match node.op {
                    Op::Cholesky => LinalgOp::Cholesky,
                    Op::Det => LinalgOp::Det,
                    Op::LogDet => LinalgOp::LogDet,
                    Op::TriangularSolve { lower, transpose } => {
                        LinalgOp::TriSolve { lower, transpose }
                    }
                    _ => LinalgOp::Solve,
                };
                b.emit(Step::Linalg {
                    out,
                    a: input_slot(0)?,
                    b: if rhs { Some(input_slot(1)?) } else { None },
                    n,
                    nrhs: if rhs {
                        total(&dims) / (batch * n).max(1)
                    } else {
                        0
                    },
                    batch,
                    op,
                });
            }
            Op::TopK { k } => {
                // Indices of the k largest along the last axis, sorted.
                let d = input_dims(0)?;
                let len = *d
                    .last()
                    .ok_or_else(|| WebglError("topk: scalar input".into()))?;
                b.emit(Step::SortAxis {
                    out,
                    src: input_slot(0)?,
                    outer: total(&d) / len.max(1),
                    len,
                    inner: 1,
                    k: *k,
                    descending: true,
                    indices: true,
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
                    // Registered (non-collective) custom ops run their rlx-cpu kernel.
                    let ins = (0..node.inputs.len())
                        .map(input_slot)
                        .collect::<Result<_>>()?;
                    emit_host(&mut b, graph, node, out, ins)?;
                } else if *num_inputs != 1 {
                    return Err(WebglError(format!(
                        "collective op {name:?} expects 1 input, graph has {num_inputs}"
                    )));
                } else {
                    b.emit(Step::Custom {
                        out,
                        input: input_slot(0)?,
                        name: name.clone(),
                        attrs: attrs.clone(),
                    });
                }
            }
            // No WebGL lowering: run the op on rlx-cpu.
            _ => {
                let ins = (0..node.inputs.len())
                    .map(input_slot)
                    .collect::<Result<_>>()?;
                emit_host(&mut b, graph, node, out, ins)?;
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
    fuse_conv1_argmax(&mut plan);
    drop_unwritten_slots(&mut plan);
    widen_tall_slots(&mut plan);
    alias_copies(&mut plan);
    fold_activations(&mut plan);
    drop_unwritten_slots(&mut plan);
    Ok(plan)
}

/// A 3-D `FusedConvBiasAct` (no residual) as a three-input `Op::Conv3d` —
/// the bias added by the conv kernel — then its activation, if any.
///
/// Legalized the generic way it becomes `conv → Reshape(bias) → Expand →
/// Add`, and the `Expand` is a gather with one index per output element: as
/// large as the activation itself, per layer. Backend-private like rlx-cpu's
/// and wgpu's folds: `Op::Conv3d` is two inputs in the IR contract, and this
/// graph only goes on to the planner.
fn fold_conv3d_bias(g: Graph) -> Graph {
    let is_target = |n: &rlx_ir::Node| {
        matches!(&n.op, Op::FusedConvBiasAct { kernel_size, has_residual: false, .. } if kernel_size.len() == 3)
            && n.inputs.len() == 3
    };
    if !g.nodes().iter().any(is_target) {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let id = match &node.op {
            Op::FusedConvBiasAct {
                stride,
                padding,
                dilation,
                groups,
                activation,
                ..
            } if is_target(node) => {
                let a3 = |v: &Vec<usize>| [v[0], v[1], v[2]];
                let conv = out.add_node(
                    Op::Conv3d {
                        stride: a3(stride),
                        padding: a3(padding),
                        dilation: a3(dilation),
                        groups: *groups,
                    },
                    ins,
                    node.shape.clone(),
                );
                match activation {
                    Some(a) => out.add_node(Op::Activation(*a), vec![conv], node.shape.clone()),
                    None => conv,
                }
            }
            _ => out.add_node(node.op.clone(), ins, node.shape.clone()),
        };
        id_map.insert(node.id, id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// A 1×1×1, unpadded, ungrouped [`Step::Conv3d`] read only by an argmax over
/// its channels becomes one [`Step::Conv1ArgMax`]: a segmentation head's
/// class scores (104 classes over a 6-million-voxel crop is 6.2e8 values,
/// past any texture) are never stored.
fn fuse_conv1_argmax(plan: &mut Plan) {
    let mut readers = vec![0usize; plan.slot_dims.len()];
    for step in &plan.steps {
        for s in step.reads() {
            readers[s] += 1;
        }
    }
    // the conv's place takes the fused step; the argmax's is dropped
    let mut fused: Vec<(usize, Step)> = Vec::new();
    let mut drop = vec![false; plan.steps.len()];
    for (ai, step) in plan.steps.iter().enumerate() {
        let Step::ArgAxis {
            out,
            src,
            outer,
            len,
            inner,
            is_max: true,
        } = *step
        else {
            continue;
        };
        if readers[src] != 1 || plan.outputs.contains(&src) {
            continue;
        }
        let Some(ci) = plan.steps[..ai].iter().rposition(|s| s.out() == src) else {
            continue;
        };
        if let Step::Conv3d {
            x,
            w,
            bias,
            n,
            cin,
            din,
            cout,
            dout,
            k: [1, 1, 1],
            stride: [1, 1, 1],
            pad: [0, 0, 0],
            groups: 1,
            ..
        } = plan.steps[ci]
            && din == dout
            && (outer, len, inner) == (n, cout, din.iter().product())
        {
            fused.push((
                ci,
                Step::Conv1ArgMax {
                    out,
                    x,
                    w,
                    bias,
                    n,
                    cin,
                    cout,
                    spatial: inner,
                },
            ));
            drop[ai] = true;
        }
    }
    for (i, s) in fused {
        plan.steps[i] = s;
    }
    let mut i = 0;
    plan.steps.retain(|_| {
        i += 1;
        !drop[i - 1]
    });
}

/// Drop a [`Step::Copy`] whose source is laid out as its output — a Reshape
/// between shapes whose rows differ but which widening gave one layout (a
/// group norm's `[n, groups, …]` view of a volume) — and read the source
/// instead. Every slot is written once, so the source holds the same values
/// wherever the copy was read. MindGrab copied each activation twice a layer
/// (50 copies of a 1 GB texture, 6% of a run).
fn alias_copies(plan: &mut Plan) {
    let mut to: Vec<usize> = (0..plan.slot_dims.len()).collect();
    plan.steps.retain(|step| match step {
        Step::Copy { out, src }
            if plan.slot_dims[*out] == plan.slot_dims[*src]
                && plan.slot_len[*out] == plan.slot_len[*src]
                && !plan.outputs.contains(out) =>
        {
            to[*out] = to[*src];
            false
        }
        _ => true,
    });
    for step in &mut plan.steps {
        step.remap_reads(|s| to[s]);
    }
}

/// Fold a `Unary` into the `Conv3d` or `GroupNorm` that writes its input,
/// when it is that slot's only reader, the slot is not an output and the two
/// slots are laid out alike: the producer applies the activation as it
/// writes (`act`) straight into the `Unary`'s slot, and a full-size pass per
/// layer is gone — brainchop's MeshNets have one after every conv or norm.
/// `RLX_WEBGL_NO_FOLD_ACT=1` keeps them separate.
fn fold_activations(plan: &mut Plan) {
    if rlx_ir::env::flag("RLX_WEBGL_NO_FOLD_ACT") {
        return;
    }
    let mut reads = vec![0usize; plan.slot_dims.len()];
    for step in &plan.steps {
        for r in step.reads() {
            reads[r] += 1;
        }
    }
    let mut writer = vec![usize::MAX; plan.slot_dims.len()];
    for (i, step) in plan.steps.iter().enumerate() {
        writer[step.out()] = i;
    }
    let mut gone = vec![false; plan.steps.len()];
    for i in 0..plan.steps.len() {
        let Step::Unary { out: u, a, act } = plan.steps[i] else {
            continue;
        };
        let p = writer[a];
        if p == usize::MAX
            || reads[a] != 1
            || plan.outputs.contains(&a)
            || plan.slot_dims[a] != plan.slot_dims[u]
            || plan.slot_len[a] != plan.slot_len[u]
        {
            continue;
        }
        match &mut plan.steps[p] {
            Step::Conv3d {
                out, act: f @ None, ..
            }
            | Step::GroupNorm {
                out, act: f @ None, ..
            } => {
                *out = u;
                *f = Some(act);
                gone[i] = true;
            }
            _ => {}
        }
    }
    let mut i = 0;
    plan.steps.retain(|_| {
        i += 1;
        !gone[i - 1]
    });
}

/// Size zero for a slot no step writes — a Reshape's alias, a fused conv's
/// scores — so it neither counts toward [`Plan::largest_texture`] nor gets
/// a buffer in the CPU executor.
fn drop_unwritten_slots(plan: &mut Plan) {
    let mut written = vec![false; plan.slot_dims.len()];
    for step in &plan.steps {
        written[step.out()] = true;
    }
    for (s, w) in written.into_iter().enumerate() {
        if !w {
            plan.slot_dims[s] = (0, 0);
            plan.slot_len[s] = 0;
        }
    }
}

/// A slot taller than this is laid out wide instead, when it can be.
const TALL_ROWS: usize = 8192;
/// Column count of a widened slot.
const WIDE_COLS: usize = 8192;
/// Widest a widened slot gets (desktop `MAX_TEXTURE_SIZE`).
const MAX_COLS: usize = 16384;

/// The `MAX_TEXTURE_SIZE` plans are laid out for: `RLX_WEBGL_MAX_TEXTURE`,
/// else a desktop's 16384. Below 16384 (4096–8192 on some phones) every
/// width above shrinks to it, so a tensor that fits `side²` texels gets a
/// texture of at most `side` a side.
pub fn max_texture() -> usize {
    rlx_ir::env::parse_or("RLX_WEBGL_MAX_TEXTURE", MAX_COLS).max(1)
}

/// Lay out tall slots (`rows > TALL_ROWS`), and slots wider than a texture
/// (`cols > MAX_COLS`, a long 1-D tensor), as `ceil(n / WIDE_COLS) × WIDE_COLS`.
///
/// A slot is `rest × last_dim` texels, so a tensor with a short last axis —
/// a conv weight `[cout, cin, 1, k]` is `cout·cin × k` — overflows WebGL's
/// `MAX_TEXTURE_SIZE` (16384 on most desktops) long before it is large.
///
/// Matmul, softmax, the norms, gather, reduce and leaf uploads address texels
/// by linear index (with each slot's own width), and element-wise kernels
/// need only that every operand shares the output's layout (softmax and the
/// norms read their input at the output's texel too). So slots joined that
/// way form a group, and a tall group is widened together unless a member is
/// produced or read by a kernel that needs its natural layout (the
/// lane-packed complex ones, custom ops).
fn widen_tall_slots(plan: &mut Plan) {
    let side = max_texture();
    let (tall_rows, wide_cols, max_cols) =
        (TALL_ROWS.min(side), WIDE_COLS.min(side), MAX_COLS.min(side));
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
            // read and written by linear index
            Step::Leaf { .. }
            | Step::Conv3d { .. }
            | Step::ArgAxis { .. }
            | Step::Conv1ArgMax { .. }
            | Step::GroupNorm { .. }
            | Step::Copy { .. }
            | Step::Splat { .. }
            | Step::Broadcast { .. }
            | Step::Concat { .. }
            | Step::Pool3d { .. }
            | Step::Resize3d { .. }
            | Step::Gather { .. }
            | Step::Reduce { .. }
            | Step::ArgReduce { .. }
            | Step::GatherRuntime { .. }
            | Step::SortAxis { .. }
            | Step::FftStage { .. }
            | Step::Host { .. }
            | Step::Linalg { .. } => {}
            // `data` is read at the output's own texel.
            Step::ScatterRuntime { out, data, .. } => {
                if let Some(d) = data {
                    union(&mut parent, *out, *d);
                }
            }
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
            // read by linear index (each operand's own texture width)
            Step::MatMul { .. } => {}
            // the input is read at the output's texel, the row by index
            Step::Softmax { out, a, .. } => union(&mut parent, *out, *a),
            // likewise, and gamma / beta share one width
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
                union(&mut parent, *out, *x);
                union(&mut parent, *gamma, *beta);
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
        // tall, or a single row wider than a texture may be (a training
        // graph's flat gradient vector is one `1 x n` row of millions)
        let (rows, cols) = plan.slot_dims[s];
        *group_tall.entry(r).or_insert(false) |= rows > tall_rows || cols > max_cols;
    }
    for s in 0..n {
        let r = find(&mut parent, s);
        if group_ok[&r] && group_tall[&r] {
            let len = plan.slot_len[s];
            // past WIDE_COLS² elements (a 30-channel activation of a brain
            // crop is 1.8e8) the rows would overflow instead: square up to
            // MAX_COLS wide
            let cols = len.clamp(1, wide_cols.max(len.div_ceil(wide_cols).min(max_cols)));
            plan.slot_dims[s] = (len.div_ceil(cols), cols);
        }
    }
}
