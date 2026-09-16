// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Finite-difference backward gate** — every case, on every backend, against
//! that backend's *own* forward.
//!
//! rlx has finite-difference gradient tests, and they are good ones. They are
//! also almost all pinned to `Device::Cpu`: `rope_vjp_fd`, `rms_norm_rank_vjp_fd`
//! and friends hardcode the CPU session. That leaves the exact gap the tree's
//! own history keeps falling into — the bugs were in *backend kernels*, and the
//! test that would have caught them only ran on the backend that was already
//! right.
//!
//! Three real defects motivate this file, and each names a property the gate
//! must have:
//!
//! * **RoPE table stride.** The table stride is `n_rot/2`, not `head_dim/2`.
//!   Three backends used the wrong one — and *agreed with each other*, so
//!   cross-backend parity was green. Finite differences caught it. Property:
//!   **the oracle must not be another backend.** Each backend is checked against
//!   its own forward, so N backends agreeing wrongly proves nothing here.
//! * **`rms_norm_backward` cross term.** An extra `1/r` on the input gradient,
//!   wrong in *all seven* implementations at once. Property: **a case must run
//!   everywhere**, because a defect shared by every backend has no majority to
//!   vote against it.
//! * **RmsNorm `beta`.** Dropped entirely by wgpu, CUDA and ROCm while CPU and
//!   Metal honoured it. Property: **cases must exercise every input**, not just
//!   the obvious one — hence separate `wrt` entries for gamma and beta.
//!
//! A fourth, `SoftmaxCrossEntropyBackward` broadcasting `d_loss[N]` across the
//! class axis, only reproduces at `N > 1`; the case below uses `N = 4` for that
//! reason. And per the RoPE post-mortem, this gate reports **every** failing
//! `(case × device)` pair before failing, rather than stopping at the first —
//! one failure tells you something broke, the whole matrix tells you what.
//!
//! Cost note: central differences need `2·|wrt|` forward passes per case, so
//! every shape here is deliberately tiny. This is a correctness gate, not a
//! benchmark.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::op::{Activation, MaskKind, PadMode, ReduceOp, RopeStyle};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

// ── Case description ────────────────────────────────────────────────────────

/// One differentiable graph, the input its gradient is taken with respect to,
/// and the numbers to feed it.
struct FdCase {
    /// Case name, as it appears in the failure matrix.
    name: &'static str,
    graph: fn() -> Graph,
    /// Name of the graph input the gradient is taken with respect to.
    wrt: &'static str,
    /// Values for `wrt`. Its length also sets how many FD probes run.
    wrt_values: fn() -> Vec<f32>,
    /// Other graph inputs, fed verbatim to every run.
    inputs: fn() -> Vec<(&'static str, Vec<f32>)>,
    /// Parameters, uploaded once per compile.
    params: fn() -> Vec<(&'static str, Vec<f32>)>,
    /// Central-difference step. Larger is less noisy but more truncation error;
    /// linear ops (RoPE) tolerate a big step, curved ones need a small one.
    eps: f32,
    /// Max allowed `|analytic - fd| / (1 + |fd|)`.
    tol: f64,
}

// ── Graph builders ──────────────────────────────────────────────────────────

const B: usize = 1;
const SEQ: usize = 4;
const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const D: usize = HEADS * HEAD_DIM;
const N: usize = B * SEQ * D;
const MAX_POS: usize = 8;

fn rope_graph(n_rot: usize, style: RopeStyle) -> Graph {
    let shape = Shape::new(&[B, SEQ, D], F);
    let table = Shape::new(&[MAX_POS, n_rot / 2], F);
    let mut g = Graph::new("rope");
    let x = g.input("x", shape.clone());
    let cos = g.input("cos", table.clone());
    let sin = g.input("sin", table);
    let y = g.add_node(
        Op::Rope {
            head_dim: HEAD_DIM,
            n_rot,
            style,
        },
        vec![x, cos, sin],
        shape,
    );
    g.set_outputs(vec![y]);
    g
}

/// RoPE cos/sin tables for `n_rot`. Note the stride: `n_rot/2` per position,
/// **not** `head_dim/2` — the distinction that three backends got wrong.
fn rope_tables(n_rot: usize) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos = vec![0.0f32; MAX_POS * half];
    let mut sin = vec![0.0f32; MAX_POS * half];
    for p in 0..MAX_POS {
        for i in 0..half {
            let theta = 10000f64.powf(-2.0 * i as f64 / n_rot as f64);
            let angle = p as f64 * theta;
            cos[p * half + i] = angle.cos() as f32;
            sin[p * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

fn norm_graph(op: Op, name: &str) -> Graph {
    let shape = Shape::new(&[B, SEQ, D], F);
    let mut g = Graph::new(name);
    let x = g.input("x", shape.clone());
    let gamma = g.param("gamma", Shape::new(&[D], F));
    let beta = g.param("beta", Shape::new(&[D], F));
    let y = g.add_node(op, vec![x, gamma, beta], shape);
    g.set_outputs(vec![y]);
    g
}

fn rms_norm_graph() -> Graph {
    norm_graph(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        "rmsnorm",
    )
}

fn layer_norm_graph() -> Graph {
    norm_graph(
        Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        },
        "layernorm",
    )
}

/// RmsNorm differentiated w.r.t. `gamma` / `beta` instead of `x`. Those are
/// `param` nodes in [`rms_norm_graph`]; a gradient w.r.t. a leaf needs it to be
/// an `input`, so this variant promotes them.
fn rms_norm_scale_graph() -> Graph {
    let shape = Shape::new(&[B, SEQ, D], F);
    let mut g = Graph::new("rmsnorm_scale");
    let x = g.param("x", shape.clone());
    let gamma = g.input("gamma", Shape::new(&[D], F));
    let beta = g.input("beta", Shape::new(&[D], F));
    let y = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gamma, beta],
        shape,
    );
    g.set_outputs(vec![y]);
    g
}

const SCE_ROWS: usize = 4;
const SCE_CLASSES: usize = 5;

/// Softmax cross-entropy over `[SCE_ROWS, SCE_CLASSES]` logits.
///
/// `SCE_ROWS > 1` is load-bearing: the backward defect this guards against
/// broadcast the per-row `d_loss[N]` across the class axis, which is invisible
/// at `N == 1` because the broadcast is then the identity.
fn softmax_ce_graph() -> Graph {
    let logits = Shape::new(&[SCE_ROWS, SCE_CLASSES], F);
    let mut g = Graph::new("sce");
    let x = g.input("x", logits.clone());
    let target = g.param("target", logits);
    let loss = g.add_node(
        Op::SoftmaxCrossEntropy,
        vec![x, target],
        Shape::new(&[SCE_ROWS], F),
    );
    g.set_outputs(vec![loss]);
    g
}

/// `[batch, heads, seq, head_dim]` for the attention cases.
///
/// The element count is *derived* from this rather than restated, so editing a
/// dimension cannot leave `spread(ATTN_N, ..)` feeding a differently-sized
/// tensor — a mismatch that would surface as an opaque shape panic rather than
/// as the gradient check the case exists to run.
const ATTN_SHAPE: [usize; 4] = [1, 2, 4, 4];
const ATTN_HEADS: usize = ATTN_SHAPE[1];
const ATTN_HEAD_DIM: usize = ATTN_SHAPE[3];

/// SDPA. `mask_kind` is a *field* of `Op::Attention`, and a backward that drops
/// it computes the unmasked gradient for a causal forward — right shape, right
/// magnitude, wrong values above the diagonal. Exactly the RoPE `style` failure
/// mode, on the op that matters most.
fn attention_graph(mask: MaskKind) -> Graph {
    let qkv = Shape::new(&ATTN_SHAPE, F);
    let mut g = Graph::new("attn");
    let q = g.input("x", qkv.clone());
    let k = g.param("k", qkv.clone());
    let v = g.param("v", qkv.clone());
    let y = g.add_node(
        Op::Attention {
            num_heads: ATTN_HEADS,
            head_dim: ATTN_HEAD_DIM,
            v_head_dim: None,
            mask_kind: mask,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        qkv,
    );
    g.set_outputs(vec![y]);
    g
}

const ATTN_N: usize = ATTN_SHAPE[0] * ATTN_SHAPE[1] * ATTN_SHAPE[2] * ATTN_SHAPE[3];

/// `Op::Pad`'s `mode` decides where each padded position's gradient flows back
/// to: nowhere (Constant), a mirrored interior element (Reflect), the edge
/// (Replicate), or the opposite edge (Circular). A backward that ignores `mode`
/// still produces a correctly-shaped gradient — with mass on the wrong elements.
fn pad_graph(mode: PadMode) -> Graph {
    let inp = Shape::new(&[1, 6], F);
    let out = Shape::new(&[1, 10], F);
    let mut g = Graph::new("pad");
    let x = g.input("x", inp);
    let y = g.add_node(
        Op::Pad {
            pads: vec![[0, 0], [2, 2]],
            mode,
        },
        vec![x],
        out,
    );
    g.set_outputs(vec![y]);
    g
}

/// `Op::Slice`'s `step` (negative here) reverses which input element each output
/// element came from. A backward that scatters forward-order gradients into a
/// reverse-order slice is wrong in a way no shape check sees.
fn slice_graph(step: i64) -> Graph {
    let inp = Shape::new(&[8], F);
    let out = Shape::new(&[4], F);
    let mut g = Graph::new("slice");
    let x = g.input("x", inp);
    let y = g.add_node(
        Op::Slice {
            axis: 0,
            start: if step < 0 { 7 } else { 0 },
            len: 4,
            step,
        },
        vec![x],
        out,
    );
    g.set_outputs(vec![y]);
    g
}

/// `num_groups` partitions the channel axis; a backward that normalizes over the
/// wrong partition gets plausible magnitudes and wrong per-channel values.
fn group_norm_graph(num_groups: usize) -> Graph {
    let nchw = Shape::new(&GN_SHAPE, F);
    let mut g = Graph::new("groupnorm");
    let x = g.input("x", nchw.clone());
    let gamma = g.param("gamma", Shape::new(&[GC], F));
    let beta = g.param("beta", Shape::new(&[GC], F));
    let y = g.add_node(
        Op::GroupNorm {
            num_groups,
            eps: 1e-5,
        },
        vec![x, gamma, beta],
        nchw,
    );
    g.set_outputs(vec![y]);
    g
}

/// `[batch, channels, h, w]` for the group-norm case, with the element count
/// derived rather than restated for the same reason as `ATTN_SHAPE`.
const GN_SHAPE: [usize; 4] = [1, 4, 2, 2];
const GC: usize = GN_SHAPE[1];
const GN_N: usize = GN_SHAPE[0] * GN_SHAPE[1] * GN_SHAPE[2] * GN_SHAPE[3];

/// `axes` + `keep_dim` decide the broadcast shape the upstream cotangent is
/// expanded back through. Reducing the wrong axis is the classic silent error —
/// and `Mean` additionally scales by the reduced extent, so a wrong axis is
/// wrong by a factor, not just a permutation.
fn reduce_graph(op: ReduceOp, axes: Vec<usize>, keep_dim: bool) -> Graph {
    let inp = Shape::new(&[2, 3, 4], F);
    let mut g = Graph::new("reduce");
    let x = g.input("x", inp.clone());
    let out = {
        let mut dims: Vec<usize> = vec![2, 3, 4];
        if keep_dim {
            for a in &axes {
                dims[*a] = 1;
            }
        } else {
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            for a in sorted.iter().rev() {
                dims.remove(*a);
            }
        }
        Shape::new(&dims, F)
    };
    let y = g.add_node(Op::Reduce { op, axes, keep_dim }, vec![x], out);
    g.set_outputs(vec![y]);
    g
}

/// `upper` and `diagonal` select which half is kept; the backward must mask the
/// gradient with the *same* triangle.
fn trilu_graph(upper: bool, diagonal: i64) -> Graph {
    let sh = Shape::new(&[4, 4], F);
    let mut g = Graph::new("trilu");
    let x = g.input("x", sh.clone());
    let y = g.add_node(Op::Trilu { upper, diagonal }, vec![x], sh);
    g.set_outputs(vec![y]);
    g
}

fn activation_graph(kind: Activation, name: &str) -> Graph {
    let shape = Shape::new(&[N], F);
    let mut g = Graph::new(name);
    let x = g.input("x", shape.clone());
    let y = g.activation(kind, x, shape);
    g.set_outputs(vec![y]);
    g
}

// ── Deterministic, well-conditioned sample values ───────────────────────────

fn spread(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| scale * (((i * 7) % 11) as f32 - 5.0) / 5.0)
        .collect()
}

fn cases() -> Vec<FdCase> {
    vec![
        // RoPE is linear in x, so a large step is exact and low-noise.
        FdCase {
            name: "rope_full_neox",
            graph: || rope_graph(HEAD_DIM, RopeStyle::NeoX),
            wrt: "x",
            wrt_values: || spread(N, 0.3),
            inputs: || {
                let (cos, sin) = rope_tables(HEAD_DIM);
                vec![("cos", cos), ("sin", sin)]
            },
            params: Vec::new,
            eps: 1e-2,
            tol: 2e-3,
        },
        // The partial case Qwen3.5/3.6 uses: n_rot < head_dim, trailing dims
        // copied through. This is the shape the stride bug lived in.
        FdCase {
            name: "rope_partial_neox",
            graph: || rope_graph(HEAD_DIM / 2, RopeStyle::NeoX),
            wrt: "x",
            wrt_values: || spread(N, 0.3),
            inputs: || {
                let (cos, sin) = rope_tables(HEAD_DIM / 2);
                vec![("cos", cos), ("sin", sin)]
            },
            params: Vec::new,
            eps: 1e-2,
            tol: 2e-3,
        },
        // GptJ pairing (the GGUF convention) exercises a different index map.
        FdCase {
            name: "rope_partial_gptj",
            graph: || rope_graph(HEAD_DIM / 2, RopeStyle::GptJ),
            wrt: "x",
            wrt_values: || spread(N, 0.3),
            inputs: || {
                let (cos, sin) = rope_tables(HEAD_DIM / 2);
                vec![("cos", cos), ("sin", sin)]
            },
            params: Vec::new,
            eps: 1e-2,
            tol: 2e-3,
        },
        // The 1/r cross-term case: wrong in all seven implementations at once.
        FdCase {
            name: "rms_norm_wrt_x",
            graph: rms_norm_graph,
            wrt: "x",
            wrt_values: || spread(N, 0.35),
            inputs: Vec::new,
            params: || {
                vec![
                    ("gamma", (0..D).map(|i| 0.8 + 0.02 * i as f32).collect()),
                    ("beta", (0..D).map(|i| 0.05 * i as f32).collect()),
                ]
            },
            eps: 1e-3,
            tol: 5e-3,
        },
        // gamma and beta separately: the dropped-beta defect is invisible from
        // the x gradient alone.
        FdCase {
            name: "rms_norm_wrt_gamma",
            graph: rms_norm_scale_graph,
            wrt: "gamma",
            wrt_values: || (0..D).map(|i| 0.8 + 0.02 * i as f32).collect(),
            inputs: || vec![("beta", (0..D).map(|i| 0.05 * i as f32).collect())],
            params: || vec![("x", spread(N, 0.35))],
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "rms_norm_wrt_beta",
            graph: rms_norm_scale_graph,
            wrt: "beta",
            wrt_values: || (0..D).map(|i| 0.05 * i as f32).collect(),
            inputs: || vec![("gamma", (0..D).map(|i| 0.8 + 0.02 * i as f32).collect())],
            params: || vec![("x", spread(N, 0.35))],
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "layer_norm_wrt_x",
            graph: layer_norm_graph,
            wrt: "x",
            wrt_values: || spread(N, 0.35),
            inputs: Vec::new,
            params: || {
                vec![
                    ("gamma", (0..D).map(|i| 0.8 + 0.02 * i as f32).collect()),
                    ("beta", (0..D).map(|i| 0.05 * i as f32).collect()),
                ]
            },
            eps: 1e-3,
            tol: 5e-3,
        },
        // N > 1 on purpose — see `softmax_ce_graph`.
        FdCase {
            name: "softmax_cross_entropy",
            graph: softmax_ce_graph,
            wrt: "x",
            wrt_values: || spread(SCE_ROWS * SCE_CLASSES, 0.8),
            inputs: Vec::new,
            params: || {
                // One-hot targets, a different class per row.
                let mut t = vec![0.0f32; SCE_ROWS * SCE_CLASSES];
                for r in 0..SCE_ROWS {
                    t[r * SCE_CLASSES + (r % SCE_CLASSES)] = 1.0;
                }
                vec![("target", t)]
            },
            eps: 2e-3,
            tol: 5e-3,
        },
        // Activations whose backward DECOMPOSES at AD on most backends: a wrong
        // decomposition corrupts training gradients only where it is used.
        FdCase {
            name: "gelu",
            graph: || activation_graph(Activation::Gelu, "gelu"),
            wrt: "x",
            wrt_values: || spread(N, 1.5),
            inputs: Vec::new,
            params: Vec::new,
            eps: 2e-3,
            tol: 5e-3,
        },
        // ── Ops whose VJP carries a droppable FIELD ─────────────────────────
        //
        // The RoPE bug was structural: `Op::Rope` gained `style`, `vjp_rope` kept
        // `..`, and the gradient went wrong on every backend while the forward
        // stayed right. The cases below are the same risk on other ops — each one
        // varies a *field* that a field-blind backward would silently ignore.
        FdCase {
            name: "attention_mask_none",
            graph: || attention_graph(MaskKind::None),
            wrt: "x",
            wrt_values: || spread(ATTN_N, 0.5),
            inputs: Vec::new,
            params: || vec![("k", spread(ATTN_N, 0.4)), ("v", spread(ATTN_N, 0.6))],
            eps: 2e-3,
            tol: 2e-2,
        },
        // Causal is the one that matters: a mask-blind backward returns the
        // unmasked gradient, which differs only above the diagonal.
        FdCase {
            name: "attention_mask_causal",
            graph: || attention_graph(MaskKind::Causal),
            wrt: "x",
            wrt_values: || spread(ATTN_N, 0.5),
            inputs: Vec::new,
            params: || vec![("k", spread(ATTN_N, 0.4)), ("v", spread(ATTN_N, 0.6))],
            eps: 2e-3,
            tol: 2e-2,
        },
        // All four pad modes: each routes a padded position's gradient to a
        // different input element (or to nothing).
        FdCase {
            name: "pad_constant",
            graph: || pad_graph(PadMode::Constant(0.0)),
            wrt: "x",
            wrt_values: || spread(6, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "pad_reflect",
            graph: || pad_graph(PadMode::Reflect),
            wrt: "x",
            wrt_values: || spread(6, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "pad_replicate",
            graph: || pad_graph(PadMode::Replicate),
            wrt: "x",
            wrt_values: || spread(6, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "pad_circular",
            graph: || pad_graph(PadMode::Circular),
            wrt: "x",
            wrt_values: || spread(6, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        // Negative step: the gradient must scatter back in reverse order.
        FdCase {
            name: "slice_negative_step",
            graph: || slice_graph(-1),
            wrt: "x",
            wrt_values: || spread(8, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "slice_positive_step",
            graph: || slice_graph(2),
            wrt: "x",
            wrt_values: || spread(8, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        // num_groups > 1 is the case a group-blind backward gets wrong; the
        // num_groups == 1 control shares its code path with plain LayerNorm.
        FdCase {
            name: "group_norm_groups2",
            graph: || group_norm_graph(2),
            wrt: "x",
            wrt_values: || spread(GN_N, 0.4),
            inputs: Vec::new,
            params: || {
                vec![
                    ("gamma", vec![0.9, 1.1, 0.8, 1.2]),
                    ("beta", vec![0.0, 0.05, -0.05, 0.1]),
                ]
            },
            eps: 1e-3,
            tol: 1e-2,
        },
        // Reduce: `axes` + `keep_dim` set the broadcast the cotangent expands
        // through, and Mean additionally scales by the reduced extent.
        FdCase {
            name: "reduce_sum_middle_axis",
            graph: || reduce_graph(ReduceOp::Sum, vec![1], false),
            wrt: "x",
            wrt_values: || spread(2 * 3 * 4, 0.6),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "reduce_mean_keepdim",
            graph: || reduce_graph(ReduceOp::Mean, vec![2], true),
            wrt: "x",
            wrt_values: || spread(2 * 3 * 4, 0.6),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        // upper / diagonal must mask the gradient with the same triangle.
        FdCase {
            name: "trilu_lower_diag0",
            graph: || trilu_graph(false, 0),
            wrt: "x",
            wrt_values: || spread(16, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "trilu_upper_diag1",
            graph: || trilu_graph(true, 1),
            wrt: "x",
            wrt_values: || spread(16, 0.7),
            inputs: Vec::new,
            params: Vec::new,
            eps: 1e-3,
            tol: 5e-3,
        },
        FdCase {
            name: "silu",
            graph: || activation_graph(Activation::Silu, "silu"),
            wrt: "x",
            wrt_values: || spread(N, 1.5),
            inputs: Vec::new,
            params: Vec::new,
            eps: 2e-3,
            tol: 5e-3,
        },
    ]
}

// ── The gate ────────────────────────────────────────────────────────────────

struct Failure {
    case: &'static str,
    device: &'static str,
    detail: String,
}

/// Analytic gradient from the backend's own backward, versus central
/// differences of the backend's own forward. Returns `Err` with a description
/// when they disagree past `tol`.
fn check_case(case: &FdCase, dev: Device) -> Result<(), String> {
    let fwd_graph = (case.graph)();
    let x0 = (case.wrt_values)();
    let inputs = (case.inputs)();
    let params = (case.params)();

    // The cotangent. Varying per element (rather than all-ones) means an error
    // that cancels under a uniform weighting still shows up.
    let out_len = {
        let mut probe = Session::new(dev).compile(fwd_graph.clone());
        for (n, v) in &params {
            probe.set_param(n, v);
        }
        let mut feed: Vec<(&str, &[f32])> = vec![(case.wrt, &x0[..])];
        feed.extend(inputs.iter().map(|(n, v)| (*n, &v[..])));
        probe.run(&feed)[0].len()
    };
    let cot: Vec<f32> = (0..out_len)
        .map(|i| 0.5 + 0.25 * ((i % 7) as f32))
        .collect();

    // Analytic.
    let bwd = grad_with_loss_wrt(
        &fwd_graph,
        &[Wrt::Leaf(case.wrt.into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = Session::new(dev).compile(bwd);
    for (n, v) in &params {
        compiled.set_param(n, v);
    }
    let analytic = {
        let mut feed: Vec<(&str, &[f32])> = vec![(case.wrt, &x0[..])];
        feed.extend(inputs.iter().map(|(n, v)| (*n, &v[..])));
        feed.push(("d_output", &cot[..]));
        let outs = compiled.run(&feed);
        outs.last()
            .ok_or_else(|| "backward graph produced no outputs".to_string())?
            .clone()
    };
    if analytic.len() != x0.len() {
        return Err(format!(
            "gradient has {} elements, wrt input has {}",
            analytic.len(),
            x0.len()
        ));
    }

    // Central differences of the same device's forward.
    let mut fwd = Session::new(dev).compile(fwd_graph);
    for (n, v) in &params {
        fwd.set_param(n, v);
    }
    let weighted = |fwd: &mut rlx_runtime::CompiledGraph, x: &[f32]| -> f64 {
        let mut feed: Vec<(&str, &[f32])> = vec![(case.wrt, x)];
        feed.extend(inputs.iter().map(|(n, v)| (*n, &v[..])));
        fwd.run(&feed)[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    let mut probe = x0.clone();
    for i in 0..x0.len() {
        probe[i] = x0[i] + case.eps;
        let lp = weighted(&mut fwd, &probe);
        probe[i] = x0[i] - case.eps;
        let lm = weighted(&mut fwd, &probe);
        probe[i] = x0[i];

        let fd = (lp - lm) / (2.0 * case.eps as f64);
        let rel = (analytic[i] as f64 - fd).abs() / (1.0 + fd.abs());
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
    }

    if worst > case.tol {
        let i = worst_at;
        probe[i] = x0[i] + case.eps;
        let lp = weighted(&mut fwd, &probe);
        probe[i] = x0[i] - case.eps;
        let lm = weighted(&mut fwd, &probe);
        let fd = (lp - lm) / (2.0 * case.eps as f64);
        return Err(format!(
            "worst rel err {worst:.3e} > tol {:.1e} at element {i}: analytic {:.6} vs fd {fd:.6}",
            case.tol, analytic[i]
        ));
    }
    Ok(())
}

/// Run every case on `dev`, collecting all failures rather than stopping at the
/// first — the whole matrix is the diagnostic.
fn gate(dev: Device, label: &'static str) {
    if common::skip_unless(dev) {
        eprintln!("fd_backward_gate: {label} unavailable — skipping");
        return;
    }
    let _guard = common::GpuTestGuard::acquire(dev);

    // Panics below are caught per case and reported as rows; silence the default
    // hook so the log shows the matrix rather than a wall of backtraces. Restored
    // before the final assert so a genuine test failure still prints normally.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut failures: Vec<Failure> = Vec::new();
    let mut passed = 0usize;
    for case in cases() {
        // A backend that PANICS on one op must not take the whole device's report
        // with it. rlx-mlx did exactly that — its rank-1 `ScatterAdd` lowering
        // aborted the run, so the 22 other verdicts for MLX were never printed
        // and the failure looked like "MLX is broken" rather than "MLX cannot
        // lower this one op". Per-case isolation turns an abort into one row of
        // the matrix, which is the whole point of reporting a matrix.
        let name = case.name;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_case(&case, dev)));
        match outcome {
            Ok(Ok(())) => passed += 1,
            Ok(Err(detail)) => failures.push(Failure {
                case: name,
                device: label,
                detail,
            }),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                failures.push(Failure {
                    case: name,
                    device: label,
                    detail: format!("PANICKED: {}", msg.lines().next().unwrap_or(&msg)),
                });
            }
        }
    }

    std::panic::set_hook(prev_hook);
    eprintln!(
        "fd_backward_gate [{label}]: {passed} passed, {} failed",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "finite-difference backward gate failed on {label}:\n{}",
        failures
            .iter()
            .map(|f| format!("  {} / {}: {}", f.device, f.case, f.detail))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ── Per-device entry points ─────────────────────────────────────────────────
//
// One test per backend rather than one loop over backends: a failure names the
// backend in the test name, and `--no-fail-fast` then reports every backend's
// verdict instead of stopping at the first bad one.

#[test]
fn fd_backward_gate_cpu() {
    gate(Device::Cpu, "cpu");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn fd_backward_gate_metal() {
    gate(Device::Metal, "metal");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn fd_backward_gate_mlx() {
    gate(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn fd_backward_gate_wgpu() {
    gate(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn fd_backward_gate_cuda() {
    gate(Device::Cuda, "cuda");
}

#[test]
#[cfg(feature = "rocm")]
fn fd_backward_gate_rocm() {
    gate(Device::Rocm, "rocm");
}

/// The gate is only as good as its coverage, and coverage is easy to lose
/// silently — a case removed during a refactor leaves a green suite. Pin the
/// defects that motivated the file by name.
#[test]
fn gate_covers_the_defects_it_was_built_for() {
    let names: Vec<&str> = cases().iter().map(|c| c.name).collect();
    for required in [
        "rope_partial_neox",     // table stride n_rot/2 vs head_dim/2
        "rope_partial_gptj",     // the GGUF pairing convention
        "rms_norm_wrt_x",        // the 1/r cross term
        "rms_norm_wrt_beta",     // beta dropped by 3 backends
        "softmax_cross_entropy", // d_loss[N] broadcast across classes
        // Field-sensitive cases: each varies an Op field a field-blind backward
        // would ignore, which is the RoPE `style` defect's structural class.
        "attention_mask_causal",
        "pad_reflect",
        "pad_circular",
        "slice_negative_step",
        "group_norm_groups2",
        "reduce_mean_keepdim",
        "trilu_upper_diag1",
    ] {
        assert!(names.contains(&required), "gate lost its `{required}` case");
    }
}

/// The per-case panic isolation must actually isolate. The defect that motivated
/// it (MLX's rank-1 `ScatterAdd`) is fixed, so nothing in `cases()` panics any
/// more — which is exactly why the mechanism needs its own test rather than
/// relying on a live failure to keep exercising it.
#[test]
fn a_panicking_case_becomes_one_row_not_an_abort() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut reported: Vec<String> = Vec::new();
    let mut passed = 0usize;
    for i in 0..3 {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if i == 1 {
                panic!("backend cannot lower this op");
            }
            Ok::<(), String>(())
        }));
        match outcome {
            Ok(Ok(())) => passed += 1,
            Ok(Err(d)) => reported.push(d),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                reported.push(format!("PANICKED: {}", msg.lines().next().unwrap_or(&msg)));
            }
        }
    }
    std::panic::set_hook(prev);
    // The two healthy cases still reported, and the panic became a row.
    assert_eq!(passed, 2, "a panic in one case swallowed the others");
    assert_eq!(reported.len(), 1);
    assert!(
        reported[0].starts_with("PANICKED: backend cannot lower this op"),
        "panic payload lost: {}",
        reported[0]
    );
}

/// The softmax-CE case must keep more than one row, or the broadcast defect it
/// exists to catch becomes unobservable.
#[test]
fn softmax_ce_case_has_more_than_one_row() {
    // A const block, so a shrink to SCE_ROWS = 1 fails to *compile* rather than
    // waiting for someone to run this test.
    const {
        assert!(
            SCE_ROWS > 1,
            "SoftmaxCrossEntropyBackward's d_loss broadcast is the identity at N == 1"
        )
    };
}
