// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Reverse-mode automatic differentiation (VJP transform).
//!
//! Takes a forward graph that produces a single scalar output (the
//! loss) and returns a new graph whose outputs are
//! `[loss, grad_w_param0, grad_w_param1, ...]` for the parameters
//! listed in `wrt`. Running the returned graph through any backend
//! gives the loss and all parameter gradients in one pass.
//!
//! ## Implementation
//!
//! Standard reverse-mode AD: walk the forward graph in reverse topo
//! order; for each visited node, emit gradient nodes that contribute
//! to the gradients of its inputs. Multiple consumers' contributions
//! are summed via `BinaryOp::Add`.
//!
//! For ops with a closed-form gradient kernel (`ReluBackward`,
//! `MaxPool2dBackward`, `Conv2dBackwardInput/Weight`,
//! `AttentionBackward`, `SoftmaxCrossEntropyBackward` — added in the
//! rlx-ir backward-op
//! family), the VJP emits the dedicated kernel rather than composing
//! the gradient from primitives.
//!
//! ## Broadcast handling
//!
//! Forward broadcasts (e.g. `[N, C] + [C]` → `[N, C]`) require the
//! reverse pass to *un-broadcast* the gradient back to the broadcast
//! input's smaller shape via a `Reduce::Sum` over the inserted /
//! tiled axes. `unbroadcast` does this; both `Op::Binary` and
//! `Op::Expand` VJPs use it.
//!
//! ## Coverage
//!
//! Element-wise: `Binary(Add/Sub/Mul/Div/Min/Max/Pow)`,
//! `Activation(*)` (Relu via dedicated `ReluBackward`, others via
//! generic `ActivationBackward`), `Compare` (zero gradient),
//! `Where`, `Cast`, `Quantize/Dequantize` (straight-through).
//!
//! Linear / reductions: `MatMul`, `Conv`, `Pool{Max,Mean}`,
//! `Reduce{Sum,Mean,Min,Max,Prod}`, `Softmax`, `LayerNorm`
//! (dedicated kernels), `RmsNorm` (composed), `Rope` (composed
//! via negated sin), `SoftmaxCrossEntropyWithLogits`.
//!
//! Shape: `Reshape`, `Transpose`, `Expand`, `Concat`, `Narrow`,
//! `Gather` (axis=0), `ScatterAdd`.
//!
//! Attention: `Op::Attention` → three [`Op::AttentionBackward`]
//! nodes (`dQ` / `dK` / `dV`) for all mask kinds. Causal /
//! SlidingWindow masks are applied inside the backward kernel (no
//! mask tensor). `Custom` / `Bias` pass the forward mask input.
//! before softmax; Custom uses the user-provided mask tensor;
//! None is the no-mask path.
//!
//! Pre-pass: `UnfuseElementwiseRegions` runs automatically before
//! the gradient walk so `Op::ElementwiseRegion` decomposes into
//! its primitive chain (covered op-by-op above).
//!
//! Sampling-style (`TopK`, `Sample`): non-differentiable — emit no
//! gradient (forward is a discrete selector / draw).
//!
//! Pre-pass: [`crate::prepare_ad::prepare_graph_for_ad`] runs before
//! the gradient walk (also exposed as [`PrepareForAutodiff`] pass).
//! It unfuses elementwise regions and tier-2 fused ops
//! (`FusedMatMulBiasAct`, `FusedResidualLN`, `FusedResidualRmsNorm`,
//! `FusedSwiGLU`, `FusedAttentionBlock`, `FusedTransformerLayer`,
//! `GatedDeltaNet`, `SelectiveScan`, …), lowers `DotGeneral`, inlines
//! `If` / unrolls `While`, inlines `CustomFn` without `vjp_body`, and
//! rewrites scans for trajectory AD.
//!
//! For HIR builders, [`rlx_ir::hir::FusionPolicy::for_autodiff`] lowers
//! to primitive MIR; [`grad_with_loss_module`] accepts HIR or MIR
//! [`GraphModule`] stages (not LIR).
//!
//! Cumsum: backward composed via matmul with a constant
//! upper-triangular ones matrix (avoids needing a new `Op::Flip`
//! primitive across all backends). Fine for typical sequence
//! lengths; an L×L matmul where L is the sequence size.
//!
//! Quantized / MoE: `Op::DequantMatMul` (QAT straight-through),
//! `Op::QMatMul`, `Op::QConv2d`, and `Op::GroupedMatMul` are all
//! supported via composed straight-through VJPs. Plain
//! `Op::Quantize/Dequantize` straight-through covers the typical
//! fake-quant fp32 training path.
//!
//! Coverage today: every op in the IR has a VJP rule or a
//! pre-pass that rewrites it into ones that do. SelectiveScan
//! (Mamba SSM step) and GatedDeltaNet (Qwen3.5 linear-attention
//! scan) decompose by unrolling the time loop into Mul / Add /
//! MatMul / Activation::Exp / Concat / Narrow / Reshape / Expand
//! primitives — same shape as the rlx-mlx lowering, just emitted
//! as IR nodes instead of MLX arrays.
//! FusedTransformerLayer / FusedAttentionBlock / FusedSwiGLU /
//! LoraMatMul / FusedMatMulBiasAct / FusedResidualLN are all
//! decomposed by `rlx_fusion::unfuse_fused_for_autodiff` likewise. Op::If
//! is rewritten to `Where(predicate, then, else)` with both
//! branches inlined; Op::While is bounded-unrolled up to
//! `max_iterations`.

use rlx_ir::op::*;
use rlx_ir::shape::Dim;
use rlx_ir::*;
use std::collections::HashMap;

pub use crate::prepare_ad::{
    AutodiffError, PrepareForAutodiff, grad_with_loss_module, jvp_module, prepare_graph_for_ad,
    prepare_mir_for_ad, prepare_module_for_ad,
};

/// Compute the reverse-mode gradient graph and the loss value.
///
/// Returns a graph whose outputs are
/// `[loss, grad_wrt[0], grad_wrt[1], ...]`. The loss is the original
/// forward output; the gradients are w.r.t. each `wrt` node (typically
/// `Op::Param` ids).
///
/// The returned graph contains a copy of the entire forward graph so
/// activations needed by gradient kernels are recomputed from inputs;
/// it also exposes a new `Op::Input` named `"d_output"` which the
/// caller seeds with the upstream gradient of the loss (typically a
/// scalar `1.0` for "differentiate the loss directly").
///
/// ## Limitations
/// - Forward graph must have exactly one output (the loss / scalar
///   you want to differentiate).
/// - All ops in the forward graph must have an implemented VJP rule.
///   Hitting an op without one is a panic, not a silent miscompute.
pub fn grad_with_loss(forward: &Graph, wrt: &[NodeId]) -> Graph {
    assert_eq!(
        forward.outputs.len(),
        1,
        "grad_with_loss: forward must have exactly one output"
    );

    // Pre-autodiff unfuse: decompose fused ops back to primitives so
    // the per-op VJP rules cover them. Two layers:
    //   1. `UnfuseElementwiseRegions` — splits the chain back to
    //      Activation/Cast/Binary/Compare/Where ops.
    //   2. `rlx_fusion::unfuse_fused_for_autodiff` (below) — handles the
    //      tier-2 fused ops with closed-form decompositions:
    //      FusedMatMulBiasAct, FusedResidualLN, LoraMatMul.
    //
    // FusedSwiGLU / FusedAttentionBlock / FusedTransformerLayer
    // are all decomposed by `rlx_fusion::unfuse_fused_for_autodiff` (each is
    // a multi-stage sub-graph; mirrors what `rlx-tpu/src/unfuse.rs`
    // does for HLO emission).
    let forward_owned = crate::prepare_ad::prepare_graph_for_ad(forward.clone());
    let forward = &forward_owned;

    let mut bwd = Graph::new(format!("{}_grad", forward.name));

    // Mirror every forward node into bwd. The activations needed by
    // gradient kernels (`x` for ReluBackward, `logits` for
    // SoftmaxCrossEntropyBackward, etc.) are looked up by these
    // mirrored ids.
    let mut fwd_to_bwd: HashMap<NodeId, NodeId> = HashMap::new();
    for node in forward.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| fwd_to_bwd[i]).collect();
        let new_id = bwd.add_node(node.op.clone(), inputs, node.shape.clone());
        fwd_to_bwd.insert(node.id, new_id);
    }

    // Seed: the gradient of the loss w.r.t. itself is an external
    // input the caller provides (typically `[1.0]` for a scalar loss).
    let loss_fwd = forward.outputs[0];
    let loss_bwd = fwd_to_bwd[&loss_fwd];
    let loss_shape = forward.node(loss_fwd).shape.clone();
    let d_output = bwd.input("d_output", loss_shape);

    let mut grads: HashMap<NodeId, NodeId> = HashMap::new();
    grads.insert(loss_bwd, d_output);

    for fwd_node in forward.nodes().iter().rev() {
        let bwd_id = fwd_to_bwd[&fwd_node.id];
        let upstream = match grads.get(&bwd_id) {
            Some(g) => *g,
            None => continue,
        };
        let input_grads = vjp(fwd_node, upstream, &fwd_to_bwd, &mut bwd);
        for (idx, grad_id) in input_grads {
            let target = fwd_node.inputs[idx];
            let bwd_target = fwd_to_bwd[&target];
            // Per-consumer gradients accumulate (`+=`).
            let new_grad = if let Some(&prev) = grads.get(&bwd_target) {
                let shape = bwd.node(prev).shape.clone();
                bwd.binary(BinaryOp::Add, prev, grad_id, shape)
            } else {
                grad_id
            };
            grads.insert(bwd_target, new_grad);
        }
    }

    let mut outputs = Vec::with_capacity(1 + wrt.len());
    outputs.push(loss_bwd);
    for &id in wrt {
        let g = grads.get(&fwd_to_bwd[&id]).copied().unwrap_or_else(|| {
            panic!(
                "no gradient flowed to {id} — \
                either the forward graph doesn't depend on it, or one \
                of its consumer ops has no VJP rule"
            )
        });
        outputs.push(g);
    }
    bwd.set_outputs(outputs);
    bwd
}

/// Backwards-compatible single-output alias (parameter gradients only,
/// no loss). Kept for the existing tests; prefer `grad_with_loss` for
/// training.
pub fn grad(forward: &Graph, wrt: &[NodeId]) -> Graph {
    let g = grad_with_loss(forward, wrt);
    let mut g = g;
    // Drop the loss output, keep only gradients.
    let outs = g.outputs.iter().skip(1).copied().collect();
    g.set_outputs(outs);
    g
}

/// Project a gradient back to a smaller shape it was broadcasted from.
/// `target_shape` is the broadcast *source* shape (e.g. `[C]` for a
/// bias added to `[N, C, H, W]`). Sums over leading prepended axes
/// and over any axis where target was 1 but the gradient is larger.
/// Then reshapes to drop the size-1 axes if the rank shrunk.
/// Returns `Some(bits)` if `node_id` is the output of an
/// `Op::FakeQuantize { bits, .. }` (or `FakeQuantizeLSQ`) in the
/// forward graph. Used by the autodiff Conv backward to detect the
/// QAT pattern and emit a specialized weight-grad kernel that can
/// skip dead bins (weights that round to the same code share the
/// gradient). Today only the detection is exposed — the
/// specialization is a follow-up commit.
pub fn quantized_weight_bits(forward: &Graph, node_id: NodeId) -> Option<u8> {
    match &forward.node(node_id).op {
        Op::FakeQuantize { bits, .. } => Some(*bits),
        Op::FakeQuantizeLSQ { bits, .. } => Some(*bits),
        _ => None,
    }
}

fn unbroadcast(grad: NodeId, target_shape: &Shape, bwd: &mut Graph) -> NodeId {
    let grad_shape = bwd.node(grad).shape.clone();
    if grad_shape == *target_shape {
        return grad;
    }
    let g_rank = grad_shape.rank();
    let t_rank = target_shape.rank();
    let extra = g_rank.saturating_sub(t_rank);

    // Axes (in grad's coordinate system) that need summing.
    let mut axes: Vec<usize> = (0..extra).collect();
    for i in 0..t_rank {
        let g_dim = grad_shape.dim(extra + i);
        let t_dim = target_shape.dim(i);
        if matches!(t_dim, Dim::Static(1)) && !matches!(g_dim, Dim::Static(1)) {
            axes.push(extra + i);
        }
    }

    let mut current = grad;
    if !axes.is_empty() {
        // The CPU `Op::Reduce` thunk only supports a *single contiguous*
        // range of axes — `[0, 2, 3]` (the canonical conv-bias-gradient
        // pattern) would silently fall through to a `Nop`. Decompose into
        // a chain of single-axis reductions with `keep_dim=true` so rank
        // stays at `g_rank` and earlier axis indices remain valid; the
        // rank shrink (if any) happens in the reshape step below.
        let mut running_dims: Vec<Dim> = (0..g_rank).map(|i| grad_shape.dim(i)).collect();
        for &ax in &axes {
            running_dims[ax] = Dim::Static(1);
            let step_shape = Shape::from_dims(&running_dims, target_shape.dtype());
            current = bwd.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![ax],
                    keep_dim: true,
                },
                vec![current],
                step_shape,
            );
        }
    }

    // Drop leading 1-axes via Reshape if the target rank is smaller.
    if bwd.node(current).shape.rank() != t_rank {
        let new_shape: Vec<i64> = target_shape
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => *n as i64,
                Dim::Dynamic(_) => -1,
            })
            .collect();
        current = bwd.add_node(
            Op::Reshape { new_shape },
            vec![current],
            target_shape.clone(),
        );
    }
    current
}

/// Reshape a gradient to a target shape (used by Reshape / Mean VJPs).
fn reshape_to(grad: NodeId, target_shape: &Shape, bwd: &mut Graph) -> NodeId {
    if bwd.node(grad).shape == *target_shape {
        return grad;
    }
    let new_shape: Vec<i64> = target_shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    bwd.add_node(Op::Reshape { new_shape }, vec![grad], target_shape.clone())
}

/// VJP for `Op::GroupedMatMul` / dequantized MoE matmul (`dx`, `dw`).
fn grouped_matmul_vjp(
    bwd: &mut Graph,
    upstream: NodeId,
    x_bwd: NodeId,
    w_bwd: NodeId,
    expert_bwd: NodeId,
    x_shape: &Shape,
    w_shape: &Shape,
) -> (NodeId, NodeId) {
    let dtype = x_shape.dtype();
    let m = x_shape.dim(0);
    let k = x_shape.dim(1);
    let e = w_shape.dim(0);
    let n_out = w_shape.dim(2);
    let m_static = match m {
        Dim::Static(v) => v,
        _ => panic!("GroupedMatMul VJP: M must be static"),
    };
    let k_static = match k {
        Dim::Static(v) => v,
        _ => panic!("GroupedMatMul VJP: K must be static"),
    };
    let n_static = match n_out {
        Dim::Static(v) => v,
        _ => panic!("GroupedMatMul VJP: N must be static"),
    };

    let w_per = bwd.add_node(
        Op::Gather { axis: 0 },
        vec![w_bwd, expert_bwd],
        Shape::from_dims(&[m, k, n_out], dtype),
    );

    let up_3d_shape = Shape::from_dims(&[m, Dim::Static(1), n_out], dtype);
    let up_3d = bwd.reshape(
        upstream,
        vec![m_static as i64, 1, n_static as i64],
        up_3d_shape,
    );
    let w_per_t = bwd.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![w_per],
        Shape::from_dims(&[m, n_out, k], dtype),
    );
    let dx_3d_shape = Shape::from_dims(&[m, Dim::Static(1), k], dtype);
    let dx_3d = bwd.matmul(up_3d, w_per_t, dx_3d_shape);
    let dx = bwd.reshape(
        dx_3d,
        vec![m_static as i64, k_static as i64],
        x_shape.clone(),
    );

    let x_3d = bwd.reshape(
        x_bwd,
        vec![m_static as i64, k_static as i64, 1],
        Shape::from_dims(&[m, k, Dim::Static(1)], dtype),
    );
    let up_for_outer = bwd.reshape(
        upstream,
        vec![m_static as i64, 1, n_static as i64],
        Shape::from_dims(&[m, Dim::Static(1), n_out], dtype),
    );
    let dw_per = bwd.matmul(x_3d, up_for_outer, Shape::from_dims(&[m, k, n_out], dtype));
    let dw = bwd.add_node(
        Op::ScatterAdd,
        vec![dw_per, expert_bwd],
        Shape::from_dims(&[e, k, n_out], dtype),
    );
    (dx, dw)
}

/// Build a scalar f32 `Op::Constant` node.
fn scalar_const(value: f32, bwd: &mut Graph) -> NodeId {
    let bytes = value.to_le_bytes().to_vec();
    let shape = Shape::from_dims(&[Dim::Static(1)], DType::F32);
    bwd.add_node(Op::Constant { data: bytes }, vec![], shape)
}

/// Per-op VJP rule. Returns (input_index, gradient_node_id) pairs;
/// inputs not listed receive no gradient (e.g. the labels argument
/// of `SoftmaxCrossEntropyWithLogits` is non-differentiable).
fn vjp(
    node: &Node,
    upstream: NodeId,
    fwd_map: &HashMap<NodeId, NodeId>,
    bwd: &mut Graph,
) -> Vec<(usize, NodeId)> {
    let upstream_shape = bwd.node(upstream).shape.clone();
    match &node.op {
        // Leaves — no inputs → no gradients to attribute.
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => vec![],

        Op::Binary(BinaryOp::Add) => {
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            let g_a = unbroadcast(upstream, &a_shape, bwd);
            let g_b = unbroadcast(upstream, &b_shape, bwd);
            vec![(0, g_a), (1, g_b)]
        }

        Op::Binary(BinaryOp::Sub) => {
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            let neg = bwd.activation(Activation::Neg, upstream, upstream_shape.clone());
            let g_a = unbroadcast(upstream, &a_shape, bwd);
            let g_b = unbroadcast(neg, &b_shape, bwd);
            vec![(0, g_a), (1, g_b)]
        }

        Op::Binary(BinaryOp::Mul) => {
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            // Wirtinger over C64: y = a·b → dL/dā = upstream · conj(b),
            // dL/db̄ = upstream · conj(a). The conjugates turn the
            // standard real Mul rule into the correct complex one
            // without changing the kernel — `BinaryFullC64` does the
            // native complex multiply on whatever inputs we hand it.
            let is_c64 = upstream_shape.dtype() == DType::C64;
            let b_for_a = if is_c64 { bwd.conjugate(b_bwd) } else { b_bwd };
            let a_for_b = if is_c64 { bwd.conjugate(a_bwd) } else { a_bwd };
            let g_a_full = bwd.binary(BinaryOp::Mul, upstream, b_for_a, upstream_shape.clone());
            let g_b_full = bwd.binary(BinaryOp::Mul, upstream, a_for_b, upstream_shape);
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);
            vec![(0, g_a), (1, g_b)]
        }

        Op::Activation(kind) => {
            let x_bwd = fwd_map[&node.inputs[0]];
            // Dedicated `ReluBackward` kernel for the most common case
            // (avoids the per-element kind-dispatch in
            // `ActivationBackward`'s match). Every other activation
            // family hits the generic kernel.
            let dx = match kind {
                Activation::Relu => bwd.relu_backward(x_bwd, upstream),
                _ => bwd.activation_backward(*kind, x_bwd, upstream),
            };
            vec![(0, dx)]
        }

        Op::MatMul => {
            // y [..batch, M, N] = a [..batch_a, M, K] @ b [..batch_b, K, N]
            //   da = upstream @ b^T   (shape [..batch, M, K])
            //   db = a^T   @ upstream (shape [..batch, K, N])
            //
            // The forward shape inference broadcasts `batch_a` and
            // `batch_b` to a common `batch`; if either side was
            // broadcasted, we sum the corresponding gradient back
            // down via `unbroadcast` so it matches the param's actual
            // shape. The transpose swaps the *last two* dims only,
            // leaving batch untouched.
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            assert!(
                a_shape.rank() >= 2 && b_shape.rank() >= 2,
                "MatMul VJP: rank must be ≥ 2, got {} and {}",
                a_shape.rank(),
                b_shape.rank()
            );
            let dtype = upstream_shape.dtype();

            // Transpose-last-two helper.
            let trans_last_two = |bwd: &mut Graph, x: NodeId| -> NodeId {
                let s = bwd.node(x).shape.clone();
                let r = s.rank();
                let mut perm: Vec<usize> = (0..r).collect();
                perm.swap(r - 2, r - 1);
                let mut dims: Vec<Dim> = s.dims().to_vec();
                dims.swap(r - 2, r - 1);
                let new_shape = Shape::from_dims(&dims, s.dtype());
                bwd.add_node(Op::Transpose { perm }, vec![x], new_shape)
            };

            // Build the matmul output shape [..upstream_batch, M_or_K, K_or_N]
            // by swapping in the trailing dims for each gradient.
            let upstream_dims: Vec<Dim> = upstream_shape.dims().to_vec();
            let r_up = upstream_dims.len();

            // ── grad-a = upstream @ b^T (output shape [..up_batch, M, K]) ──
            let b_t = trans_last_two(bwd, b_bwd);
            let mut ga_dims = upstream_dims.clone();
            ga_dims[r_up - 1] = a_shape.dim(a_shape.rank() - 1); // K
            let ga_shape = Shape::from_dims(&ga_dims, dtype);
            let g_a_full = bwd.matmul(upstream, b_t, ga_shape);
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);

            // ── grad-b = a^T @ upstream (output shape [..up_batch, K, N]) ──
            let a_t = trans_last_two(bwd, a_bwd);
            let mut gb_dims = upstream_dims.clone();
            gb_dims[r_up - 2] = a_shape.dim(a_shape.rank() - 1); // K
            let gb_shape = Shape::from_dims(&gb_dims, dtype);
            let g_b_full = bwd.matmul(a_t, upstream, gb_shape);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);

            vec![(0, g_a), (1, g_b)]
        }

        Op::DenseSolve => {
            // X = solve(A, B) ⇒ implicit-function VJP:
            //   dB = solve(Aᵀ, upstream)        same shape as B / X
            //   dA = -dB · Xᵀ                   [N, N]
            //
            // Rank-1 (b: [N]) and rank-2 (B: [N, K]) follow the same
            // formula; rank-1 needs a reshape-to-column trick because
            // we don't have a vector-outer-product op (matmul is
            // matrix-only). Rank-2 is direct matmul.
            let a_bwd = fwd_map[&node.inputs[0]];
            let x_bwd = fwd_map[&node.id];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let x_shape = bwd.node(x_bwd).shape.clone();
            assert_eq!(a_shape.rank(), 2, "DenseSolve VJP: A must be 2-D");
            let n = match a_shape.dim(0) {
                Dim::Static(n) => n,
                Dim::Dynamic(_) => panic!("DenseSolve VJP: dynamic N not supported"),
            };
            let dtype = a_shape.dtype();

            // Aᵀ — shape [N, N] (square, transpose is just a perm).
            let mut a_t_dims: Vec<Dim> = a_shape.dims().to_vec();
            a_t_dims.swap(0, 1);
            let a_t_shape = Shape::from_dims(&a_t_dims, dtype);
            let a_t = bwd.add_node(Op::Transpose { perm: vec![1, 0] }, vec![a_bwd], a_t_shape);

            // dB = solve(Aᵀ, upstream). Same shape as the original B.
            let d_b = bwd.dense_solve(a_t, upstream, x_shape.clone());

            // dA = -dB · Xᵀ.
            let neg_outer = match x_shape.rank() {
                1 => {
                    // b: [N]. Reshape both vectors to matrices for matmul.
                    let col_shape = Shape::from_dims(&[Dim::Static(n), Dim::Static(1)], dtype);
                    let row_shape = Shape::from_dims(&[Dim::Static(1), Dim::Static(n)], dtype);
                    let db_col = bwd.add_node(
                        Op::Reshape {
                            new_shape: vec![n as i64, 1],
                        },
                        vec![d_b],
                        col_shape,
                    );
                    let x_row = bwd.add_node(
                        Op::Reshape {
                            new_shape: vec![1, n as i64],
                        },
                        vec![x_bwd],
                        row_shape,
                    );
                    let outer = bwd.matmul(db_col, x_row, a_shape.clone());
                    bwd.activation(Activation::Neg, outer, a_shape)
                }
                2 => {
                    // B: [N, K]. dA = -MatMul(dB, Xᵀ) where Xᵀ: [K, N].
                    let k = match x_shape.dim(1) {
                        Dim::Static(k) => k,
                        Dim::Dynamic(_) => panic!("DenseSolve VJP: dynamic K not supported"),
                    };
                    let xt_dims = vec![Dim::Static(k), Dim::Static(n)];
                    let xt_shape = Shape::from_dims(&xt_dims, dtype);
                    let x_t =
                        bwd.add_node(Op::Transpose { perm: vec![1, 0] }, vec![x_bwd], xt_shape);
                    let outer = bwd.matmul(d_b, x_t, a_shape.clone());
                    bwd.activation(Activation::Neg, outer, a_shape)
                }
                r => panic!("DenseSolve VJP: B must be rank 1 or 2, got rank {r}"),
            };

            vec![(0, neg_outer), (1, d_b)]
        }

        Op::BatchedDenseSolve => {
            // Per-batch independent. Same implicit-function VJP as
            // DenseSolve, lifted with a leading B axis throughout:
            //   dB = batched_solve(Aᵀ, upstream)        same shape as B/X
            //   dA = -batched_matmul(dB, Xᵀ)            shape [B, N, N]
            // where `Aᵀ` swaps the LAST TWO axes (perm = [0, 2, 1]).
            let a_bwd = fwd_map[&node.inputs[0]];
            let x_bwd = fwd_map[&node.id];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let x_shape = bwd.node(x_bwd).shape.clone();
            assert_eq!(
                a_shape.rank(),
                3,
                "BatchedDenseSolve VJP: A must be rank-3 [B, N, N]"
            );
            let b_dim = match a_shape.dim(0) {
                Dim::Static(b) => b,
                Dim::Dynamic(_) => panic!("BatchedDenseSolve VJP: dynamic B not supported"),
            };
            let n = match a_shape.dim(1) {
                Dim::Static(n) => n,
                Dim::Dynamic(_) => panic!("BatchedDenseSolve VJP: dynamic N not supported"),
            };
            let dtype = a_shape.dtype();

            // Aᵀ across last two dims — perm = [0, 2, 1]. Output shape
            // is [B, N, N] (same as A; transpose of square is square).
            let a_t = bwd.add_node(
                Op::Transpose {
                    perm: vec![0, 2, 1],
                },
                vec![a_bwd],
                a_shape.clone(),
            );

            // dB = batched_solve(Aᵀ, upstream).
            let d_b = bwd.batched_dense_solve(a_t, upstream, x_shape.clone());

            // dA = -batched_matmul(dB, Xᵀ).
            let neg_outer = match x_shape.rank() {
                2 => {
                    // b is [B, N]. Reshape to [B, N, 1] (column) for dB
                    // and [B, 1, N] (row) for X, then batched matmul.
                    let col_shape = Shape::from_dims(
                        &[Dim::Static(b_dim), Dim::Static(n), Dim::Static(1)],
                        dtype,
                    );
                    let row_shape = Shape::from_dims(
                        &[Dim::Static(b_dim), Dim::Static(1), Dim::Static(n)],
                        dtype,
                    );
                    let db_col = bwd.add_node(
                        Op::Reshape {
                            new_shape: vec![b_dim as i64, n as i64, 1],
                        },
                        vec![d_b],
                        col_shape,
                    );
                    let x_row = bwd.add_node(
                        Op::Reshape {
                            new_shape: vec![b_dim as i64, 1, n as i64],
                        },
                        vec![x_bwd],
                        row_shape,
                    );
                    let outer = bwd.matmul(db_col, x_row, a_shape.clone());
                    bwd.activation(Activation::Neg, outer, a_shape)
                }
                3 => {
                    // b is [B, N, K]. dA = -MatMul(dB, Xᵀ) with
                    // Xᵀ = Transpose(perm=[0, 2, 1]) so [B, K, N].
                    let k = match x_shape.dim(2) {
                        Dim::Static(k) => k,
                        Dim::Dynamic(_) => panic!("BatchedDenseSolve VJP: dynamic K not supported"),
                    };
                    let xt_shape = Shape::from_dims(
                        &[Dim::Static(b_dim), Dim::Static(k), Dim::Static(n)],
                        dtype,
                    );
                    let x_t = bwd.add_node(
                        Op::Transpose {
                            perm: vec![0, 2, 1],
                        },
                        vec![x_bwd],
                        xt_shape,
                    );
                    let outer = bwd.matmul(d_b, x_t, a_shape.clone());
                    bwd.activation(Activation::Neg, outer, a_shape)
                }
                r => panic!("BatchedDenseSolve VJP: b must be rank 2 or 3, got rank {r}"),
            };

            vec![(0, neg_outer), (1, d_b)]
        }

        Op::Scan {
            body,
            length,
            save_trajectory,
            num_bcast: _,
            num_xs,
            num_checkpoints,
        } => {
            // After `convert_scans_for_ad`, every scan reaching the AD
            // walk carries its trajectory. Compile body's VJP once
            // — w.r.t. carry AND every xs — so we can extract dinit
            // (Op::ScanBackward) plus dxs_i for each xs
            // (Op::ScanBackwardXs). Each variant runs its own backward
            // sweep; this is `1 + num_xs` independent sweeps. A future
            // optimization can fuse them via packed multi-output.
            let init_bwd = fwd_map[&node.inputs[0]];
            let traj_bwd = fwd_map[&node.id];
            let init_shape = bwd.node(init_bwd).shape.clone();

            // Body Inputs in NodeId order: first = carry, rest = x_t_i.
            let mut body_input_ids: Vec<NodeId> = body
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Input { .. }))
                .map(|n| n.id)
                .collect();
            body_input_ids.sort();

            let body_vjp = grad(body, &body_input_ids);

            let xs_bwd: Vec<NodeId> = (0..*num_xs as usize)
                .map(|i| fwd_map[&node.inputs[1 + i]])
                .collect();

            // Recursive checkpointing: when num_checkpoints is set on
            // the forward Scan, propagate it (and the forward body) to
            // each emitted ScanBackward / ScanBackwardXs so the
            // executor knows to recompute carries via `forward_body`
            // between checkpoints.
            let is_checkpointed = *num_checkpoints != 0 && *num_checkpoints != *length;
            let forward_body_for_bwd = if is_checkpointed {
                Some((**body).clone())
            } else {
                None
            };

            let dinit = bwd.scan_backward_with_checkpoints(
                init_bwd,
                traj_bwd,
                upstream,
                &xs_bwd,
                body_vjp.clone(),
                *length,
                *save_trajectory,
                *num_checkpoints,
                forward_body_for_bwd.clone(),
                init_shape,
            );

            let mut grads: Vec<(usize, NodeId)> = vec![(0, dinit)];
            for i in 0..*num_xs as usize {
                let outer_xs_id = node.inputs[1 + i];
                let xs_shape = bwd.node(fwd_map[&outer_xs_id]).shape.clone();
                let dxs_i = bwd.scan_backward_xs_with_checkpoints(
                    init_bwd,
                    traj_bwd,
                    upstream,
                    &xs_bwd,
                    body_vjp.clone(),
                    *length,
                    *save_trajectory,
                    i as u32,
                    *num_checkpoints,
                    forward_body_for_bwd.clone(),
                    xs_shape,
                );
                grads.push((1 + i, dxs_i));
            }
            grads
        }

        Op::Conv {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_bwd = fwd_map[&node.inputs[1]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_shape = bwd.node(w_bwd).shape.clone();
            let dx = bwd.conv2d_backward_input(
                upstream,
                w_bwd,
                x_shape,
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
                dilation.clone(),
                *groups,
            );
            // Detect the QAT pattern (`Conv` reading from a
            // `FakeQuantize` weight) so a follow-up specialization
            // can skip dead bins (weights that round to the same
            // code share the gradient). For now we still emit the
            // generic backward — the helper just exposes the bits
            // for a future kernel variant.
            // QAT-bits detection requires the forward graph, which isn't
            // threaded through `vjp`. Skip for now; the generic backward
            // is used unconditionally.
            let _qat_bits: Option<u8> = None;
            let dw = bwd.conv2d_backward_weight(
                x_bwd,
                upstream,
                w_shape,
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
                dilation.clone(),
                *groups,
            );
            vec![(0, dx), (1, dw)]
        }

        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size,
            stride,
            padding,
        } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let dx = bwd.maxpool2d_backward(
                x_bwd,
                upstream,
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
            );
            vec![(0, dx)]
        }

        Op::SoftmaxCrossEntropyWithLogits => {
            let logits_bwd = fwd_map[&node.inputs[0]];
            let labels_bwd = fwd_map[&node.inputs[1]];
            let dlogits = bwd.softmax_cross_entropy_backward(logits_bwd, labels_bwd, upstream);
            // labels has no gradient.
            vec![(0, dlogits)]
        }

        Op::Reduce {
            op: ReduceOp::Sum,
            axes,
            keep_dim,
        } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let g = expand_to(upstream, &x_shape, axes, *keep_dim, bwd);
            vec![(0, g)]
        }

        Op::Reduce {
            op: ReduceOp::Mean,
            axes,
            keep_dim,
        } => {
            // Mean = Sum / N. Do the Sum-style expansion first, then
            // multiply the broadcast result by 1/N. Multiplying after
            // the expand keeps the broadcast cleanly anchored at the
            // full input shape and sidesteps the rank-promotion when
            // the reduced output is a scalar (shape `[]`).
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let count: usize = axes
                .iter()
                .map(|&a| match x_shape.dim(a) {
                    Dim::Static(n) => n,
                    _ => panic!("Reduce::Mean VJP requires static reduced dims"),
                })
                .product();
            let expanded = expand_to(upstream, &x_shape, axes, *keep_dim, bwd);
            let inv_count = scalar_const(1.0 / count as f32, bwd);
            let g = bwd.binary(BinaryOp::Mul, expanded, inv_count, x_shape);
            vec![(0, g)]
        }

        Op::Reshape { .. } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dx = reshape_to(upstream, &x_shape, bwd);
            vec![(0, dx)]
        }

        Op::ComplexNormSq => {
            // Wirtinger: ∂|z|²/∂z̄ = z. Cotangent g (real) maps to
            // dz = g·z (complex, element-wise).
            let z_bwd = fwd_map[&node.inputs[0]];
            let dz = bwd.complex_norm_sq_backward(z_bwd, upstream);
            vec![(0, dz)]
        }

        Op::Conjugate => {
            // For w = conj(z): under the JAX-style cotangent (carrying
            // ∂L/∂z̄ for a real-valued L), the rule reduces to
            // cotangent_z = conj(cotangent_w). So the VJP of Conjugate
            // is Conjugate itself. Symmetric — second-order derivatives
            // through complex graphs stay consistent.
            let dz = bwd.conjugate(upstream);
            vec![(0, dz)]
        }

        Op::Cast { .. } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dx = bwd.add_node(
                Op::Cast {
                    to: x_shape.dtype(),
                },
                vec![upstream],
                x_shape,
            );
            vec![(0, dx)]
        }

        // Straight-through estimator: forward simulates the lossy
        // round-trip (x → q → x'), backward pretends it was an
        // identity. `dx = upstream` for both ops. The upstream is the
        // f32 gradient computed by the consumer; the int8 dtype on
        // the input/output is ignored for the gradient — we treat
        // the entire Q/DQ chain as a real-valued no-op for autodiff
        // purposes. This is the foundation for QAT (quantization-
        // aware training): the model trains in fp32 but every
        // forward pass tastes the int8 round-tripped activations,
        // so the learned weights are robust to deployment-time
        // quantization.
        Op::Quantize { .. } | Op::Dequantize { .. } => {
            vec![(0, upstream)]
        }

        Op::FakeQuantizeLSQ { bits, axis } => {
            // LSQ has TWO gradients: dx (STE-clipped) and dscale
            // (closed-form). Route them to inputs[0] (x) and
            // inputs[1] (scale) respectively.
            let x_bwd = fwd_map[&node.inputs[0]];
            let scale_bwd = fwd_map[&node.inputs[1]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let scale_shape = bwd.node(scale_bwd).shape.clone();
            let dx = bwd.add_node(
                Op::FakeQuantizeLSQBackwardX {
                    bits: *bits,
                    axis: *axis,
                },
                vec![x_bwd, scale_bwd, upstream],
                x_shape,
            );
            let dscale = bwd.add_node(
                Op::FakeQuantizeLSQBackwardScale {
                    bits: *bits,
                    axis: *axis,
                },
                vec![x_bwd, scale_bwd, upstream],
                scale_shape,
            );
            vec![(0, dx), (1, dscale)]
        }

        // FakeQuantize backward depends on the STE variant. The
        // default `Identity` is a clean passthrough; the others
        // attenuate the gradient based on `x` and the per-channel
        // scale, so we emit a dedicated `FakeQuantizeBackward` op.
        Op::FakeQuantize {
            bits, axis, ste, ..
        } => {
            use rlx_ir::op::SteKind;
            match ste {
                SteKind::Identity => vec![(0, upstream)],
                _ => {
                    let x_bwd = fwd_map[&node.inputs[0]];
                    let x_shape = bwd.node(x_bwd).shape.clone();
                    let dx = bwd.add_node(
                        Op::FakeQuantizeBackward {
                            bits: *bits,
                            axis: *axis,
                            ste: *ste,
                        },
                        vec![x_bwd, upstream],
                        x_shape,
                    );
                    vec![(0, dx)]
                }
            }
        }

        Op::Expand { .. } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dx = unbroadcast(upstream, &x_shape, bwd);
            vec![(0, dx)]
        }

        Op::LayerNorm { axis, eps } => {
            // y = LayerNorm(x, gamma, beta) over the feature axis.
            // d_x via the dedicated `LayerNormBackwardInput` kernel
            // (closed-form, recomputes mean/var/x̂ inline).
            // d_gamma via `LayerNormBackwardGamma` (sums over batch axes).
            // d_beta = sum(upstream) over batch axes — composable with
            // an unbroadcast back to gamma's shape (gamma and beta share shape).
            let x_bwd = fwd_map[&node.inputs[0]];
            let gamma_bwd = fwd_map[&node.inputs[1]];
            let _beta_bwd = fwd_map[&node.inputs[2]];
            let gamma_shape = bwd.node(gamma_bwd).shape.clone();

            let dx = bwd.layer_norm_backward_input(x_bwd, gamma_bwd, upstream, *axis, *eps);
            let dgamma =
                bwd.layer_norm_backward_gamma(x_bwd, upstream, gamma_shape.clone(), *axis, *eps);
            let dbeta = unbroadcast(upstream, &gamma_shape, bwd);
            vec![(0, dx), (1, dgamma), (2, dbeta)]
        }

        Op::Softmax { axis } => {
            // y = softmax(x, axis)  →  dy/dx[i] = y[i] · (g[i] - Σⱼ y[j]·g[j])
            // where the Σⱼ is over the softmax axis. Compose from existing
            // primitives:  yg = y * upstream
            //              s  = reduce_sum(yg, axis, keep_dim=true)
            //              s' = expand(s, target=y.shape)
            //              dx = y * (upstream - s')
            //
            // The forward `y` lives at `fwd_to_bwd[node.id]` — bwd
            // graph mirrors every forward node so its slot survives
            // through this VJP. We *explicitly* expand `s` to `y.shape`
            // before the Sub rather than relying on `Op::Binary`'s
            // broadcast (which has a known shape-confusion bug for the
            // `[..., 1]` keep-dim case — see the rlx-cpu thunk
            // dispatch). Going through `Op::Expand` runs the
            // dedicated stride-aware broadcast thunk, which is correct.
            let y_bwd = fwd_map[&node.id];
            let y_shape = bwd.node(y_bwd).shape.clone();
            let dtype = y_shape.dtype();
            let rank = y_shape.rank();
            let axis_pos = if *axis < 0 {
                (rank as i32 + *axis) as usize
            } else {
                *axis as usize
            };

            let yg = bwd.binary(BinaryOp::Mul, y_bwd, upstream, y_shape.clone());

            let mut kept_dims: Vec<Dim> = y_shape.dims().to_vec();
            kept_dims[axis_pos] = Dim::Static(1);
            let kept_shape = Shape::from_dims(&kept_dims, dtype);
            let s = bwd.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![axis_pos],
                    keep_dim: true,
                },
                vec![yg],
                kept_shape,
            );

            let target_dims: Vec<i64> = y_shape
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => *n as i64,
                    Dim::Dynamic(_) => -1,
                })
                .collect();
            let s_expanded = bwd.add_node(
                Op::Expand {
                    target_shape: target_dims,
                },
                vec![s],
                y_shape.clone(),
            );

            let diff = bwd.binary(BinaryOp::Sub, upstream, s_expanded, y_shape.clone());
            let dx = bwd.binary(BinaryOp::Mul, y_bwd, diff, y_shape);
            vec![(0, dx)]
        }

        // ── Shape ops: just route the upstream gradient through ──
        Op::Transpose { perm } => {
            // Inverse permutation: if forward maps axis i → perm[i],
            // backward maps perm[i] → i.
            let inv: Vec<usize> = {
                let mut v = vec![0usize; perm.len()];
                for (i, &p) in perm.iter().enumerate() {
                    v[p] = i;
                }
                v
            };
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dx = bwd.add_node(Op::Transpose { perm: inv }, vec![upstream], x_shape);
            vec![(0, dx)]
        }

        Op::Concat { axis } => {
            // Split upstream along the concat axis: each input gets
            // `Narrow(upstream, axis, offset, x_i.dim(axis))`.
            let mut grads = Vec::with_capacity(node.inputs.len());
            let mut offset: usize = 0;
            for (i, &input_id) in node.inputs.iter().enumerate() {
                let x_bwd = fwd_map[&input_id];
                let x_shape = bwd.node(x_bwd).shape.clone();
                let len = match x_shape.dim(*axis) {
                    Dim::Static(n) => n,
                    _ => panic!("Concat VJP: dynamic concat dim"),
                };
                let dx = bwd.add_node(
                    Op::Narrow {
                        axis: *axis,
                        start: offset,
                        len,
                    },
                    vec![upstream],
                    x_shape,
                );
                grads.push((i, dx));
                offset += len;
            }
            grads
        }

        Op::Narrow { axis, start, len } => {
            // Inverse of slicing: pad upstream with zeros on both
            // sides along `axis` so the result matches input shape.
            // Build via Concat[zeros_pre, upstream, zeros_post].
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let full_n = match x_shape.dim(*axis) {
                Dim::Static(n) => n,
                _ => panic!("Narrow VJP: dynamic axis"),
            };
            let pre = *start;
            let post = full_n - *start - *len;

            let zero_buf = |bwd: &mut Graph, len_axis: usize| -> NodeId {
                if len_axis == 0 {
                    return upstream; // sentinel, never used (filtered below)
                }
                let dtype = x_shape.dtype();
                let mut dims: Vec<Dim> = x_shape.dims().to_vec();
                dims[*axis] = Dim::Static(len_axis);
                let s = Shape::from_dims(&dims, dtype);
                let n_elems = dims.iter().fold(1usize, |a, d| match d {
                    Dim::Static(k) => a * k,
                    _ => a,
                });
                // Bytes per element scales with dtype; bytewise-zero is
                // a valid zero at any precision (IEEE +0.0 / signed 0 /
                // unsigned 0), so a vec of zero bytes is safe.
                let bytes = vec![0u8; n_elems * dtype.size_bytes()];
                bwd.add_node(Op::Constant { data: bytes }, vec![], s)
            };

            let mut parts: Vec<NodeId> = Vec::new();
            if pre > 0 {
                parts.push(zero_buf(bwd, pre));
            }
            parts.push(upstream);
            if post > 0 {
                parts.push(zero_buf(bwd, post));
            }

            let dx = if parts.len() == 1 {
                parts[0]
            } else {
                bwd.add_node(Op::Concat { axis: *axis }, parts, x_shape)
            };
            vec![(0, dx)]
        }

        Op::Gather { axis } => {
            let table_bwd = fwd_map[&node.inputs[0]];
            let indices_bwd = fwd_map[&node.inputs[1]];
            let table_shape = bwd.node(table_bwd).shape.clone();
            if *axis == 0 {
                let dtable =
                    bwd.add_node(Op::ScatterAdd, vec![upstream, indices_bwd], table_shape);
                vec![(0, dtable)]
            } else {
                let dtable = bwd.gather_backward(
                    upstream,
                    indices_bwd,
                    table_shape,
                    (*axis).try_into().unwrap(),
                );
                vec![(0, dtable)]
            }
        }

        // ── Non-differentiable predicates / selectors ──
        Op::Compare(_) => {
            // Compare returns a boolean tensor; gradient w.r.t.
            // continuous inputs is zero almost everywhere. We don't
            // propagate (caller will see zero grads for any path
            // that flows through a Compare alone).
            vec![]
        }

        Op::Where => {
            // out = where(cond, a, b). Cond has zero gradient
            // (it's a predicate); a's gradient is `where(cond,
            // upstream, 0)`; b's gradient is `where(cond, 0, upstream)`.
            let cond = fwd_map[&node.inputs[0]];
            let a_bwd = fwd_map[&node.inputs[1]];
            let b_bwd = fwd_map[&node.inputs[2]];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            let out_shape = upstream_shape.clone();

            let zero_a_bytes = vec![0u8; a_shape.num_elements().expect("Where VJP: dynamic a") * 4];
            let zero_b_bytes = vec![0u8; b_shape.num_elements().expect("Where VJP: dynamic b") * 4];
            let zero_a = bwd.add_node(Op::Constant { data: zero_a_bytes }, vec![], a_shape.clone());
            let zero_b = bwd.add_node(Op::Constant { data: zero_b_bytes }, vec![], b_shape.clone());
            // Need to match shapes for Op::Where (cond, a, b same).
            // Upstream shape == out_shape == broadcast of a/b.
            let zero_a_bcast = unbroadcast_inverse(zero_a, &out_shape, bwd);
            let zero_b_bcast = unbroadcast_inverse(zero_b, &out_shape, bwd);
            let g_a_full = bwd.add_node(
                Op::Where,
                vec![cond, upstream, zero_a_bcast],
                out_shape.clone(),
            );
            let g_b_full = bwd.add_node(Op::Where, vec![cond, zero_b_bcast, upstream], out_shape);
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);
            vec![(1, g_a), (2, g_b)]
        }

        // ── Element-wise binary ops ──
        Op::Binary(BinaryOp::Div) => {
            // Real:  d/da (a/b) = 1/b,        d/db (a/b) = -a/b² = -y/b
            // C64 (Wirtinger):
            //        d/dā = upstream / conj(b)
            //        d/db̄ = -upstream · conj(y) / conj(b)
            // Substituting `b ↦ conj(b)` and `y ↦ conj(y)` in the real
            // rule recovers the complex one — the kernel itself is
            // unchanged.
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let y_bwd = fwd_map[&node.id];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            let is_c64 = upstream_shape.dtype() == DType::C64;

            let b_term = if is_c64 { bwd.conjugate(b_bwd) } else { b_bwd };
            let y_term = if is_c64 { bwd.conjugate(y_bwd) } else { y_bwd };

            // d/da: upstream / b_term
            let g_a_full = bwd.binary(BinaryOp::Div, upstream, b_term, upstream_shape.clone());
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);

            // d/db: -upstream * y_term / b_term
            let neg_up = bwd.activation(Activation::Neg, upstream, upstream_shape.clone());
            let neg_up_y = bwd.binary(BinaryOp::Mul, neg_up, y_term, upstream_shape.clone());
            let g_b_full = bwd.binary(BinaryOp::Div, neg_up_y, b_term, upstream_shape);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);

            vec![(0, g_a), (1, g_b)]
        }

        // ── Reductions: gradient flows to where the reduction "saw" ──
        Op::Reduce {
            op: ReduceOp::Max,
            axes,
            keep_dim,
        }
        | Op::Reduce {
            op: ReduceOp::Min,
            axes,
            keep_dim,
        } => {
            // d_x[i] = upstream where x[i] equals the (broadcast)
            // reduce result, else 0. Composed via
            // expand(upstream) * (compare(x, expand(y), Eq) → 1.0).
            let is_max = matches!(
                node.op,
                Op::Reduce {
                    op: ReduceOp::Max,
                    ..
                }
            );
            let _ = is_max;
            let x_bwd = fwd_map[&node.inputs[0]];
            let y_bwd = fwd_map[&node.id];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let y_expanded = expand_to(y_bwd, &x_shape, axes, *keep_dim, bwd);
            let mask_bool = bwd.add_node(
                Op::Compare(CmpOp::Eq),
                vec![x_bwd, y_expanded],
                Shape::from_dims(x_shape.dims(), DType::F32),
            );
            // Convert bool→f32 via Cast (the IR encodes bool/PRED as
            // F32 in our backends already; this is a no-op cast on
            // most paths).
            let mask_f32 = bwd.add_node(
                Op::Cast {
                    to: x_shape.dtype(),
                },
                vec![mask_bool],
                x_shape.clone(),
            );
            let upstream_expanded = expand_to(upstream, &x_shape, axes, *keep_dim, bwd);
            let dx = bwd.binary(BinaryOp::Mul, upstream_expanded, mask_f32, x_shape);
            vec![(0, dx)]
        }

        // ── Rope: backward is forward with negated sin ──
        //
        //   forward:  out = x * cos + rotate(x) * sin
        //   reverse:  dx  = dy * cos + rotate(dy) * (-sin)
        //         =  rope(dy, cos, neg(sin))
        Op::Rope { head_dim, n_rot } => {
            let cos = fwd_map[&node.inputs[1]];
            let sin = fwd_map[&node.inputs[2]];
            let dx = bwd.rope_backward(upstream, cos, sin, *head_dim, *n_rot);
            vec![(0, dx)]
        }

        Op::RmsNorm { axis, eps } => {
            let x = fwd_map[&node.inputs[0]];
            let gamma = fwd_map[&node.inputs[1]];
            let beta = fwd_map[&node.inputs[2]];
            let dx = bwd.rms_norm_backward_input(x, gamma, beta, upstream, *axis, *eps);
            let dgamma = bwd.rms_norm_backward_gamma(x, gamma, beta, upstream, *axis, *eps);
            let dbeta = bwd.rms_norm_backward_beta(x, gamma, beta, upstream, *axis, *eps);
            vec![(0, dx), (1, dgamma), (2, dbeta)]
        }

        Op::GroupNorm { num_groups, eps } => {
            let x = fwd_map[&node.inputs[0]];
            let gamma = fwd_map[&node.inputs[1]];
            let beta = fwd_map[&node.inputs[2]];
            let gamma_shape = bwd.node(gamma).shape.clone();
            let beta_shape = bwd.node(beta).shape.clone();
            let dx = bwd.group_norm_backward_input(x, gamma, beta, upstream, *num_groups, *eps);
            let dgamma =
                bwd.group_norm_backward_gamma(x, upstream, gamma_shape, *num_groups, *eps);
            let dbeta = bwd.group_norm_backward_beta(x, upstream, beta_shape, *num_groups, *eps);
            vec![(0, dx), (1, dgamma), (2, dbeta)]
        }

        // ── Attention → dedicated backward kernels ──────────────
        Op::Attention {
            num_heads,
            head_dim,
            mask_kind,
        } => {
            let q = fwd_map[&node.inputs[0]];
            let k = fwd_map[&node.inputs[1]];
            let v = fwd_map[&node.inputs[2]];
            let mask = match mask_kind {
                MaskKind::Custom | MaskKind::Bias => Some(fwd_map[&node.inputs[3]]),
                _ => None,
            };
            let (dq, dk, dv) = bwd.attention_backward_all(
                q,
                k,
                v,
                upstream,
                *num_heads,
                *head_dim,
                *mask_kind,
                mask,
            );
            vec![(0, dq), (1, dk), (2, dv)]
        }

        // ── Reduce(Prod) ────────────────────────────────────────
        //
        // Forward: y[axes_reduced] = ∏ x[axes_reduced…]
        // Backward: dx[i] = upstream · y / x[i]   (per-row).
        // (Numerically dicey when any x[i] = 0; production users
        //  needing zero-safe Prod-grad should pre-mask.)
        Op::Reduce {
            op: ReduceOp::Prod,
            axes,
            keep_dim,
        } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let y_bwd = fwd_map[&node.id];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let y_expanded = expand_to(y_bwd, &x_shape, axes, *keep_dim, bwd);
            let upstream_expanded = expand_to(upstream, &x_shape, axes, *keep_dim, bwd);
            // dx = upstream_b · y_b / x
            let num = bwd.binary(
                BinaryOp::Mul,
                upstream_expanded,
                y_expanded,
                x_shape.clone(),
            );
            let dx = bwd.binary(BinaryOp::Div, num, x_bwd, x_shape);
            vec![(0, dx)]
        }

        // ── Pool(Mean) ──────────────────────────────────────────
        //
        // Forward: y[..., h_out, w_out] = mean(window).
        // Backward: dx[i] = upstream[output_pos(i)] / |window|
        //   distributed across each pool window.
        //
        // Compose via a Conv2dBackwardInput with a constant
        // 1/|window| kernel of shape [C, 1, kH, kW] and groups=C
        // (depthwise — no channel mixing). This gives the correct
        // "spread upstream over window" behavior including stride
        // and padding handling.
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size,
            stride,
            padding,
        } => {
            assert_eq!(kernel_size.len(), 2, "Pool(Mean) VJP: 2-D pool only");
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dtype = x_shape.dtype();
            // Channels = x_shape.dim(1).
            let c = match x_shape.dim(1) {
                Dim::Static(n) => n,
                _ => panic!("Pool(Mean) VJP: dynamic channel dim"),
            };
            let kh = kernel_size[0];
            let kw = kernel_size[1];
            let inv_n = 1.0_f32 / (kh as f32 * kw as f32);
            let kernel_n = c * kh * kw;
            let mut bytes: Vec<u8> = Vec::with_capacity(kernel_n * 4);
            for _ in 0..kernel_n {
                bytes.extend_from_slice(&inv_n.to_le_bytes());
            }
            let kernel_shape = Shape::from_dims(
                &[
                    Dim::Static(c),
                    Dim::Static(1),
                    Dim::Static(kh),
                    Dim::Static(kw),
                ],
                dtype,
            );
            let kernel = bwd.add_node(Op::Constant { data: bytes }, vec![], kernel_shape);
            let dx = bwd.conv2d_backward_input(
                upstream,
                kernel,
                x_shape,
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
                vec![1, 1],
                c, // groups = c → depthwise
            );
            vec![(0, dx)]
        }

        // ── Binary(Min/Max) ─────────────────────────────────────
        //
        // Element-wise min/max: gradient flows to whichever input
        // was selected (ties go to the first operand by convention).
        //   da = where(a == out, upstream, 0)
        //   db = where(a == out, 0, upstream)   ← exclusive
        Op::Binary(BinaryOp::Min) | Op::Binary(BinaryOp::Max) => {
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let y_bwd = fwd_map[&node.id];
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();
            let dtype = upstream_shape.dtype();

            let bool_shape = Shape::from_dims(upstream_shape.dims(), DType::Bool);
            let mask_pred = bwd.add_node(
                Op::Compare(CmpOp::Eq),
                vec![a_bwd, y_bwd],
                bool_shape,
            );
            let mask_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![mask_pred],
                upstream_shape.clone(),
            );
            let zero_bytes = vec![
                0u8;
                upstream_shape
                    .num_elements()
                    .expect("Min/Max VJP: dyn shape")
                    * 4
            ];
            let zero = bwd.add_node(
                Op::Constant { data: zero_bytes },
                vec![],
                upstream_shape.clone(),
            );
            let g_a_full = bwd.add_node(
                Op::Where,
                vec![mask_f32, upstream, zero],
                upstream_shape.clone(),
            );
            let g_b_full = bwd.add_node(Op::Where, vec![mask_f32, zero, upstream], upstream_shape);
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);
            vec![(0, g_a), (1, g_b)]
        }

        // ── Binary(Pow) ─────────────────────────────────────────
        //
        //   d/da (aᵇ) = b · a^(b-1)
        //   d/db (aᵇ) = aᵇ · ln(a)
        //
        // We don't have a `Pow` activation, but `pow(a, b)` for
        // positive base equals `exp(b · ln(a))`, and the derivative
        // simplifies. Express via `Activation::Log / Exp` and `Mul`.
        Op::Binary(BinaryOp::Pow) => {
            let a_bwd = fwd_map[&node.inputs[0]];
            let b_bwd = fwd_map[&node.inputs[1]];
            let y_bwd = fwd_map[&node.id]; // a^b
            let a_shape = bwd.node(a_bwd).shape.clone();
            let b_shape = bwd.node(b_bwd).shape.clone();

            // d/da: upstream · y / a = upstream · b · a^(b-1).
            // Easier route: upstream · y · b / a.
            let yb = bwd.binary(BinaryOp::Mul, y_bwd, b_bwd, upstream_shape.clone());
            let yb_over_a = bwd.binary(BinaryOp::Div, yb, a_bwd, upstream_shape.clone());
            let g_a_full = bwd.binary(BinaryOp::Mul, upstream, yb_over_a, upstream_shape.clone());
            let g_a = unbroadcast(g_a_full, &a_shape, bwd);

            // d/db: upstream · y · ln(a)
            let ln_a = bwd.activation(Activation::Log, a_bwd, a_shape);
            let ln_a_b = unbroadcast_inverse(ln_a, &upstream_shape, bwd);
            let yln = bwd.binary(BinaryOp::Mul, y_bwd, ln_a_b, upstream_shape.clone());
            let g_b_full = bwd.binary(BinaryOp::Mul, upstream, yln, upstream_shape);
            let g_b = unbroadcast(g_b_full, &b_shape, bwd);

            vec![(0, g_a), (1, g_b)]
        }

        // ── DequantMatMul (QAT-style straight-through) ─────────
        //
        // Forward (Int8BlockAsym):
        //   w_dq = (cast<f32>(w_q) - zp_b) * scale_b
        //   y    = x @ w_dq
        //
        // Backward (QAT convention — scale and zp are typically
        // frozen during fine-tuning; w_q's int8 cast is treated as
        // a no-op for the gradient via straight-through):
        //   dx     = upstream @ w_dq^T
        //   dw_q   = x^T @ upstream * scale_b   (straight-through;
        //            the user's optimizer would project back to
        //            int8 after the step)
        //   dscale = 0   (frozen)
        //   dzp    = 0   (frozen)
        //
        // For full QAT with learnable scale/zp, replace the zero
        // gradients with the closed-form ∂y/∂scale / ∂y/∂zp.
        Op::DequantMatMul { scheme: _ } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_q_bwd = fwd_map[&node.inputs[1]];
            let scale_bwd = fwd_map[&node.inputs[2]];
            let zp_bwd = fwd_map[&node.inputs[3]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_shape = bwd.node(w_q_bwd).shape.clone();
            let scale_shape = bwd.node(scale_bwd).shape.clone();
            let zp_shape = bwd.node(zp_bwd).shape.clone();

            // dx = upstream @ w_dq^T. Recompute w_dq inline.
            // w_q is int8 in the IR — cast to f32 for the matmul
            // backward graph (straight-through equivalent).
            let dtype = x_shape.dtype();
            let w_q_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![w_q_bwd],
                Shape::from_dims(w_shape.dims(), dtype),
            );
            // Broadcast scale/zp to w_shape before subtract/mul.
            let scale_b =
                unbroadcast_inverse(scale_bwd, &Shape::from_dims(w_shape.dims(), dtype), bwd);
            let zp_b = unbroadcast_inverse(zp_bwd, &Shape::from_dims(w_shape.dims(), dtype), bwd);
            let w_centered = bwd.binary(
                BinaryOp::Sub,
                w_q_f32,
                zp_b,
                Shape::from_dims(w_shape.dims(), dtype),
            );
            let w_dq = bwd.binary(
                BinaryOp::Mul,
                w_centered,
                scale_b,
                Shape::from_dims(w_shape.dims(), dtype),
            );

            // Transpose w_dq's last two dims for dx = upstream @ w_dq^T.
            let w_rank = w_shape.rank();
            let mut perm: Vec<usize> = (0..w_rank).collect();
            perm.swap(w_rank - 2, w_rank - 1);
            let mut wdt_dims: Vec<Dim> = w_shape.dims().to_vec();
            wdt_dims.swap(w_rank - 2, w_rank - 1);
            let w_dq_t_shape = Shape::from_dims(&wdt_dims, dtype);
            let w_dq_t = bwd.add_node(Op::Transpose { perm }, vec![w_dq], w_dq_t_shape);
            let dx = bwd.matmul(upstream, w_dq_t, x_shape.clone());

            // dw_q = (x^T @ upstream) * scale_b   (straight-through).
            // The result is in the int8-weight space — caller's
            // optimizer is expected to project back. We emit it as
            // f32 here and let downstream cast.
            let x_rank = x_shape.rank();
            let mut x_perm: Vec<usize> = (0..x_rank).collect();
            x_perm.swap(x_rank - 2, x_rank - 1);
            let mut x_t_dims: Vec<Dim> = x_shape.dims().to_vec();
            x_t_dims.swap(x_rank - 2, x_rank - 1);
            let x_t = bwd.add_node(
                Op::Transpose { perm: x_perm },
                vec![x_bwd],
                Shape::from_dims(&x_t_dims, dtype),
            );
            let dw_unscaled = bwd.matmul(x_t, upstream, Shape::from_dims(w_shape.dims(), dtype));
            let dw_q_f32 = bwd.binary(
                BinaryOp::Mul,
                dw_unscaled,
                scale_b,
                Shape::from_dims(w_shape.dims(), dtype),
            );
            // Cast back to the IR's int8 dtype convention.
            let dw_q = bwd.add_node(
                Op::Cast {
                    to: w_shape.dtype(),
                },
                vec![dw_q_f32],
                w_shape,
            );

            // scale and zp: zero gradients (frozen QAT convention).
            let zero_scale_bytes =
                vec![0u8; scale_shape.num_elements().expect("DQMM VJP: dyn scale") * 4];
            let zero_zp_bytes = vec![0u8; zp_shape.num_elements().expect("DQMM VJP: dyn zp") * 4];
            let dscale = bwd.add_node(
                Op::Constant {
                    data: zero_scale_bytes,
                },
                vec![],
                scale_shape,
            );
            let dzp = bwd.add_node(
                Op::Constant {
                    data: zero_zp_bytes,
                },
                vec![],
                zp_shape,
            );

            vec![(0, dx), (1, dw_q), (2, dscale), (3, dzp)]
        }

        // ── ScatterAdd ──────────────────────────────────────────
        //
        // Forward: out[indices[i], ...] += updates[i, ...].
        // Backward: d_updates[i, ...] = upstream[indices[i], ...]  (gather).
        //   Indices are non-differentiable.
        Op::ScatterAdd => {
            let updates_bwd = fwd_map[&node.inputs[0]];
            let indices_bwd = fwd_map[&node.inputs[1]];
            let updates_shape = bwd.node(updates_bwd).shape.clone();
            let dupdates = bwd.add_node(
                Op::Gather { axis: 0 },
                vec![upstream, indices_bwd],
                updates_shape,
            );
            vec![(0, dupdates)]
        }

        // ── Cumsum ──────────────────────────────────────────────
        //
        Op::Cumsum { axis, exclusive } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let dx = bwd.cumsum_backward(upstream, x_shape, *axis, *exclusive);
            vec![(0, dx)]
        }

        // ── GroupedMatMul (MoE primitive) ──────────────────────
        //
        // Forward: y[i] = x[i] @ w[expert[i]]
        //   x        [M, K]
        //   w        [E, K, N]
        //   expert   [M] (f32-encoded indices)
        //   y        [M, N]
        //
        // Backward (composed via Gather + batched-MatMul + ScatterAdd):
        //   dx[i] = upstream[i] @ w[expert[i]]^T
        //   dw[e, k, n] = sum_{i : expert[i]=e} x[i,k] · upstream[i,n]
        //   dexpert: zero (non-differentiable index input).
        Op::GroupedMatMul => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_bwd = fwd_map[&node.inputs[1]];
            let expert_bwd = fwd_map[&node.inputs[2]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_shape = bwd.node(w_bwd).shape.clone();
            let (dx, dw) = grouped_matmul_vjp(
                bwd,
                upstream,
                x_bwd,
                w_bwd,
                expert_bwd,
                &x_shape,
                &w_shape,
            );
            vec![(0, dx), (1, dw)]
        }

        // ── DequantGroupedMatMul (frozen GGUF MoE weights) ─────
        //
        // Materialize w_dq via `Op::DequantMoEWeights`, then reuse the
        // GroupedMatMul VJP. Packed U8 weights and expert indices are
        // non-differentiable (inference / QAT-frozen convention).
        Op::DequantGroupedMatMul { scheme } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_packed = fwd_map[&node.inputs[1]];
            let expert_bwd = fwd_map[&node.inputs[2]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_packed_shape = bwd.node(w_packed).shape.clone();
            let dtype = x_shape.dtype();
            let k = x_shape.dim(1);
            let n_out = node.shape.dim(node.shape.rank() - 1);
            let k_static = match k {
                Dim::Static(v) => v,
                _ => panic!("DequantGroupedMatMul VJP: K must be static"),
            };
            let n_static = match n_out {
                Dim::Static(v) => v,
                _ => panic!("DequantGroupedMatMul VJP: N must be static"),
            };
            let block_elems = scheme.gguf_block_size() as usize;
            let block_bytes = scheme.gguf_block_bytes() as usize;
            let slab_bytes = (k_static * n_static) / block_elems * block_bytes;
            let total_bytes = w_packed_shape.num_elements().expect("DequantGroupedMatMul VJP: dyn packed");
            let e_static = total_bytes / slab_bytes.max(1);
            let w_shape = Shape::from_dims(
                &[
                    Dim::Static(e_static),
                    Dim::Static(k_static),
                    Dim::Static(n_static),
                ],
                dtype,
            );
            let w_dq = bwd.add_node(
                Op::DequantMoEWeights { scheme: *scheme },
                vec![w_packed],
                w_shape.clone(),
            );
            let (dx, _dw) = grouped_matmul_vjp(
                bwd,
                upstream,
                x_bwd,
                w_dq,
                expert_bwd,
                &x_shape,
                &w_shape,
            );
            vec![(0, dx)]
        }

        // ── QMatMul / QConv2d (straight-through INT8 backward) ──
        //
        // Real INT8 inference kernels. The forward applies
        //   out = clamp(round((x − x_zp) · (w − w_zp) · mult + bias)
        //               + out_zp, [-128, 127])
        // and outputs i8. For training, the standard QAT recipe
        // treats the round/clamp/quantize as straight-through, so
        // the gradient is what a plain f32 MatMul (or Conv) backward
        // would give applied to the dequantized representations.
        // Zero-points and `mult` are typically frozen (calibration
        // outputs); we emit zero gradients for them. Bias gets the
        // standard sum-over-batch gradient.
        Op::QMatMul {
            x_zp,
            w_zp,
            out_zp: _,
            mult,
        } => {
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_bwd = fwd_map[&node.inputs[1]];
            let bias_bwd = fwd_map[&node.inputs[2]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_shape = bwd.node(w_bwd).shape.clone();
            let bias_shape = bwd.node(bias_bwd).shape.clone();
            let dtype = upstream_shape.dtype();

            // Promote x and w to f32 (straight-through); subtract zps.
            let x_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![x_bwd],
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let w_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![w_bwd],
                Shape::from_dims(w_shape.dims(), dtype),
            );
            let xzp_c = scalar_const(*x_zp as f32, bwd);
            let xzp_b = unbroadcast_inverse(xzp_c, &Shape::from_dims(x_shape.dims(), dtype), bwd);
            let _ = bwd.binary(
                BinaryOp::Sub,
                x_f32,
                xzp_b,
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let wzp_c = scalar_const(*w_zp as f32, bwd);
            let wzp_b = unbroadcast_inverse(wzp_c, &Shape::from_dims(w_shape.dims(), dtype), bwd);
            let w_centered = bwd.binary(
                BinaryOp::Sub,
                w_f32,
                wzp_b,
                Shape::from_dims(w_shape.dims(), dtype),
            );

            // mult scaling.
            let mult_c = scalar_const(*mult, bwd);
            let mult_b = unbroadcast_inverse(mult_c, &upstream_shape, bwd);
            let upstream_scaled =
                bwd.binary(BinaryOp::Mul, upstream, mult_b, upstream_shape.clone());

            // dx = upstream_scaled @ w_centered^T   (still i8 dtype
            //  on the input side; cast the gradient back).
            let w_rank = w_shape.rank();
            let mut perm: Vec<usize> = (0..w_rank).collect();
            perm.swap(w_rank - 2, w_rank - 1);
            let mut wt_dims: Vec<Dim> = w_shape.dims().to_vec();
            wt_dims.swap(w_rank - 2, w_rank - 1);
            let w_t = bwd.add_node(
                Op::Transpose { perm },
                vec![w_centered],
                Shape::from_dims(&wt_dims, dtype),
            );
            let dx_f32 = bwd.matmul(
                upstream_scaled,
                w_t,
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let dx = bwd.add_node(
                Op::Cast {
                    to: x_shape.dtype(),
                },
                vec![dx_f32],
                x_shape.clone(),
            );

            // dw = x_centered^T @ upstream_scaled  (similarly cast).
            let x_rank = x_shape.rank();
            let mut x_perm: Vec<usize> = (0..x_rank).collect();
            x_perm.swap(x_rank - 2, x_rank - 1);
            let mut xt_dims: Vec<Dim> = x_shape.dims().to_vec();
            xt_dims.swap(x_rank - 2, x_rank - 1);
            // Need to pull x_centered into scope — recompute inline.
            let x_f32_2 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![x_bwd],
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let x_centered = bwd.binary(
                BinaryOp::Sub,
                x_f32_2,
                xzp_b,
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let x_t = bwd.add_node(
                Op::Transpose { perm: x_perm },
                vec![x_centered],
                Shape::from_dims(&xt_dims, dtype),
            );
            let dw_f32 = bwd.matmul(
                x_t,
                upstream_scaled,
                Shape::from_dims(w_shape.dims(), dtype),
            );
            let dw = bwd.add_node(
                Op::Cast {
                    to: w_shape.dtype(),
                },
                vec![dw_f32],
                w_shape,
            );

            // dbias = sum upstream_scaled over batch axes (matches
            // f32 MatMul-with-bias backward shape).
            let bias_rank = bias_shape.rank();
            let reduce_axes: Vec<usize> = (0..upstream_shape.rank())
                .filter(|&i| i + bias_rank < upstream_shape.rank() || i == 0)
                .collect();
            let dbias_f32 = bwd.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: reduce_axes,
                    keep_dim: false,
                },
                vec![upstream_scaled],
                Shape::from_dims(bias_shape.dims(), dtype),
            );
            let dbias = bwd.add_node(
                Op::Cast {
                    to: bias_shape.dtype(),
                },
                vec![dbias_f32],
                bias_shape,
            );

            vec![(0, dx), (1, dw), (2, dbias)]
        }

        Op::QConv2d {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            x_zp,
            w_zp,
            out_zp: _,
            mult,
        } => {
            // Same straight-through pattern as QMatMul, lifted to
            // 2-D conv via the existing Conv2dBackwardInput / Weight
            // kernels.
            let x_bwd = fwd_map[&node.inputs[0]];
            let w_bwd = fwd_map[&node.inputs[1]];
            let bias_bwd = fwd_map[&node.inputs[2]];
            let x_shape = bwd.node(x_bwd).shape.clone();
            let w_shape = bwd.node(w_bwd).shape.clone();
            let bias_shape = bwd.node(bias_bwd).shape.clone();
            let dtype = upstream_shape.dtype();

            // Promote and dequantize.
            let x_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![x_bwd],
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let w_f32 = bwd.add_node(
                Op::Cast { to: dtype },
                vec![w_bwd],
                Shape::from_dims(w_shape.dims(), dtype),
            );
            let xzp_c = scalar_const(*x_zp as f32, bwd);
            let xzp_b = unbroadcast_inverse(xzp_c, &Shape::from_dims(x_shape.dims(), dtype), bwd);
            let x_centered = bwd.binary(
                BinaryOp::Sub,
                x_f32,
                xzp_b,
                Shape::from_dims(x_shape.dims(), dtype),
            );
            let wzp_c = scalar_const(*w_zp as f32, bwd);
            let wzp_b = unbroadcast_inverse(wzp_c, &Shape::from_dims(w_shape.dims(), dtype), bwd);
            let w_centered = bwd.binary(
                BinaryOp::Sub,
                w_f32,
                wzp_b,
                Shape::from_dims(w_shape.dims(), dtype),
            );

            // mult scaling on upstream.
            let mult_c = scalar_const(*mult, bwd);
            let mult_b = unbroadcast_inverse(mult_c, &upstream_shape, bwd);
            let upstream_scaled =
                bwd.binary(BinaryOp::Mul, upstream, mult_b, upstream_shape.clone());

            // dx, dw via the existing conv-backward kernels.
            let dx_f32 = bwd.conv2d_backward_input(
                upstream_scaled,
                w_centered,
                Shape::from_dims(x_shape.dims(), dtype),
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
                dilation.clone(),
                *groups,
            );
            let dx = bwd.add_node(
                Op::Cast {
                    to: x_shape.dtype(),
                },
                vec![dx_f32],
                x_shape,
            );
            let dw_f32 = bwd.conv2d_backward_weight(
                x_centered,
                upstream_scaled,
                Shape::from_dims(w_shape.dims(), dtype),
                kernel_size.clone(),
                stride.clone(),
                padding.clone(),
                dilation.clone(),
                *groups,
            );
            let dw = bwd.add_node(
                Op::Cast {
                    to: w_shape.dtype(),
                },
                vec![dw_f32],
                w_shape,
            );

            // dbias = sum upstream_scaled over (N, H_out, W_out) keeping C_out.
            let dbias_f32 = bwd.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![0, 2, 3],
                    keep_dim: false,
                },
                vec![upstream_scaled],
                Shape::from_dims(bias_shape.dims(), dtype),
            );
            let dbias = bwd.add_node(
                Op::Cast {
                    to: bias_shape.dtype(),
                },
                vec![dbias_f32],
                bias_shape,
            );

            vec![(0, dx), (1, dw), (2, dbias)]
        }

        // ── Sampling-style ops: non-differentiable ──
        Op::TopK { .. } | Op::Sample { .. } => {
            // TopK selects; Sample multinomial-draws. Gradient w.r.t.
            // the input distribution is undefined / zero in the
            // standard sense. Skip propagation.
            vec![]
        }

        Op::GaussianSplatRender {
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
            ..
        } => {
            use rlx_ir::ops::splat::{
                unpack_gaussian_splat_packed_grads, GaussianSplatBackwardParams, GaussianSplatInputs,
                GaussianSplatRenderParams,
            };
            let render = GaussianSplatRenderParams {
                width: *width,
                height: *height,
                tile_size: *tile_size,
                radius_scale: *radius_scale,
                alpha_cutoff: *alpha_cutoff,
                max_splat_steps: *max_splat_steps,
                transmittance_threshold: *transmittance_threshold,
                max_list_entries: *max_list_entries,
            };
            let inputs = GaussianSplatInputs {
                positions: fwd_map[&node.inputs[0]],
                scales: fwd_map[&node.inputs[1]],
                rotations: fwd_map[&node.inputs[2]],
                opacities: fwd_map[&node.inputs[3]],
                colors: fwd_map[&node.inputs[4]],
                sh_coeffs: fwd_map[&node.inputs[5]],
                meta: fwd_map[&node.inputs[6]],
            };
            let count = bwd.shape(inputs.positions).num_elements().unwrap_or(0) / 3;
            let sh_len = bwd.shape(inputs.sh_coeffs).num_elements().unwrap_or(0);
            let meta_shape = bwd.shape(inputs.meta).clone();
            let packed = bwd.gaussian_splat_render_backward(
                inputs,
                upstream,
                GaussianSplatBackwardParams {
                    render,
                    loss_grad_clip: 1.0,
                    sh_band: 0,
                    max_anisotropy: 10.0,
                },
            );
            let sh_coeff_count = if count == 0 {
                1
            } else {
                (sh_len / (count * 3)).max(1)
            };
            let grads = unpack_gaussian_splat_packed_grads(bwd, packed, count, sh_coeff_count);
            let meta_n = meta_shape.num_elements().unwrap_or(0);
            let zero_meta = bwd.add_node(
                Op::Constant {
                    data: vec![0u8; meta_n * meta_shape.dtype().size_bytes()],
                },
                vec![],
                meta_shape,
            );
            vec![
                (0, grads.positions),
                (1, grads.scales),
                (2, grads.rotations),
                (3, grads.opacities),
                (4, grads.colors),
                (5, grads.sh_coeffs),
                (6, zero_meta),
            ]
        }

        Op::GaussianSplatRenderBackward { .. } => {
            // Scene/meta inputs are not differentiated through this op in v1.
            vec![]
        }

        Op::GaussianSplatPrepare { .. } | Op::GaussianSplatRasterize { .. } => {
            panic!(
                "autodiff: decomposed splat ops must be fused before AD — \
                 `prepare_graph_for_ad` rewrites Prepare→Rasterize into \
                 `GaussianSplatRender`, or use `Op::GaussianSplatRender` directly"
            );
        }

        // ── Anything else: explicit panic with op name ──
        //
        // All ops in the IR have either a per-op VJP rule above
        // or a pre-pass rewrite that decomposes them into ops
        // that do:
        //   * Op::If → control_flow::inline_if (Where + inlined
        //     branches).
        //   * Op::While → control_flow::unroll_while (bounded
        //     unroll up to max_iterations).
        //   * Op::SelectiveScan / Op::FusedTransformerLayer /
        //     Op::FusedAttentionBlock / Op::FusedSwiGLU /
        //     Op::LoraMatMul / Op::FusedMatMulBiasAct /
        //     Op::FusedResidualLN → rlx_fusion::unfuse_fused_for_autodiff.
        //
        // User-defined sub-graph (Op::CustomFn) with override AD.
        // When `vjp_body` is supplied, inline it into `bwd`: each
        // primal Op::Input maps to the outer forward NodeId for that
        // primal; the special-named "primal_output" Input maps to the
        // forward NodeId of this CustomFn node; "d_output" maps to
        // `upstream`. The body's N outputs become this op's N input
        // gradients in declaration order.
        Op::CustomFn {
            vjp_body: Some(vjp_body),
            num_inputs,
            ..
        } => {
            // Map vjp_body NodeIds → bwd NodeIds.
            let mut sub_to_bwd: HashMap<NodeId, NodeId> = HashMap::new();

            // Collect primal-input NodeIds from vjp_body (excluding
            // special names), sorted by NodeId. Position k in this list
            // matches the outer node's input k.
            let mut primal_input_ids: Vec<NodeId> = vjp_body
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } if name != "primal_output" && name != "d_output" => {
                        Some(n.id)
                    }
                    _ => None,
                })
                .collect();
            primal_input_ids.sort();
            assert_eq!(primal_input_ids.len(), *num_inputs as usize);

            // Walk vjp_body in declaration order, cloning each non-Input
            // node into bwd with input remapping.
            for sub_node in vjp_body.nodes() {
                let new_id = match &sub_node.op {
                    Op::Input { name } if name == "primal_output" => fwd_map[&node.id],
                    Op::Input { name } if name == "d_output" => upstream,
                    Op::Input { .. } => {
                        // Find this Input's index in primal_input_ids.
                        let idx = primal_input_ids
                            .iter()
                            .position(|&id| id == sub_node.id)
                            .expect(
                                "custom_fn vjp_body: primal Input \
                                     not found in primal list",
                            );
                        fwd_map[&node.inputs[idx]]
                    }
                    _ => {
                        let new_inputs: Vec<NodeId> =
                            sub_node.inputs.iter().map(|i| sub_to_bwd[i]).collect();
                        bwd.add_node(sub_node.op.clone(), new_inputs, sub_node.shape.clone())
                    }
                };
                sub_to_bwd.insert(sub_node.id, new_id);
            }

            // Collect outputs in set_outputs order — each maps to a
            // primal-input gradient.
            let mut grads: Vec<(usize, NodeId)> = Vec::with_capacity(*num_inputs as usize);
            for (i, out_id) in vjp_body.outputs.iter().enumerate() {
                grads.push((i, sub_to_bwd[out_id]));
            }
            grads
        }

        // CustomFn without vjp_body is inlined by `inline_custom_fn_for_autodiff`
        // before the reverse walk — reaching here means the pre-pass missed it.
        Op::CustomFn { vjp_body: None, .. } => {
            panic!(
                "autodiff: Op::CustomFn has no vjp_body and was not inlined. \
                 This is an internal error in inline_custom_fn_for_autodiff."
            )
        }

        // User-registered custom op — dispatch the VJP through the
        // op registry. The impl emits gradient nodes via the same
        // `bwd` builder built-in arms use; default impl returns
        // `vec![]` (non-differentiable).
        Op::Custom { name, .. } => {
            let ext = rlx_ir::lookup_op(name).unwrap_or_else(|| {
                panic!(
                    "autodiff: Op::Custom('{name}') is not registered \
                        in the op registry — register it via \
                        rlx_ir::register_op before compiling the graph"
                )
            });
            let mut ctx = rlx_ir::VjpContext {
                upstream,
                fwd_map,
                bwd,
            };
            ext.vjp(node, &mut ctx)
        }

        // 1D FFT: y = fft(x; inverse). Both forward and inverse are
        // unnormalized linear operators on the 2N real-block layout,
        // and the DFT matrix's transpose (over the real-block view)
        // equals the unnormalized inverse DFT. So:
        //   VJP(fft)  = ifft(upstream)
        //   VJP(ifft) = fft(upstream)
        // No scaling — the choice to leave both directions unnormalized
        // makes the chain rule a flag flip and nothing else.
        Op::Fft { inverse } => {
            let dx = bwd.fft(upstream, !*inverse);
            vec![(0, dx)]
        }

        // The catch-all below remains as a safety net: if a
        // future op is added without a VJP rule, this panic
        // names it for the implementer.
        other => panic!(
            "autodiff: no VJP rule for {other}. See the matching \
             entry in rlx-opt/src/autodiff.rs (catch-all panic) for \
             a pointer to what's needed to differentiate this op.",
        ),
    }
}

/// Decompose tier-2 fused ops back to their primitive components so
/// the per-op VJP rules cover them. Conceptually identical to what a
/// "training-aware compile" pipeline would do as a pre-pass: avoid
/// running `FuseMatMulBiasAct` / `FuseResidualLN` / `FuseSwiGLU` /
/// `FuseAttentionBlock` if you plan to autodiff afterward. This
/// helper handles the case where they're already in the graph (e.g.
/// from a re-trained inference model).
///
/// Decomposed today: `FusedMatMulBiasAct`, `FusedResidualLN`,
/// `LoraMatMul`, `FusedSwiGLU`, `FusedAttentionBlock`,
/// `FusedTransformerLayer`, and `SelectiveScan` / `GatedDeltaNet` —
/// each rewritten back to its primitive chain (matmul / narrow / attention /
/// layer_norm / residual / activation, plus reduce-sum / concat /
/// Pre-AD pass: convert every `Op::Scan { save_trajectory: false }`
/// into `Op::Scan { save_trajectory: true }` followed by `Narrow` +
/// `Reshape` to extract the final carry. After this rewrite, every
/// scan in the graph carries its full trajectory — which is what the
/// VJP rule needs to compute backward through time. The user-facing
/// shape is unchanged (Narrow + Reshape collapse [length, *carry]
/// back down to *carry).
///
/// Memory cost: trajectory storage is now `O(length × carry_size)`
/// for the duration of the forward + backward pass. For Diffrax-style
/// transients this is the same as Diffrax's `RecursiveCheckpointAdjoint::All`
/// strategy. Recursive checkpointing is a future pass.
/// Pre-AD pass: rewrite `Op::Scan` nodes with `num_bcast > 0` into
/// equivalent `num_bcast = 0` scans by materialising each broadcast
/// input `b` of shape `*bcast` into a per-step xs of shape
/// `[length, *bcast]` (built as `ones([length, *bcast]) × b`). The
/// reverse-mode AD walk and the rest of `convert_scans_for_ad` then
/// see only carry+xs scans — the bcast channel is a forward-only
/// memory optimisation, transparent to backward.
fn materialize_bcasts_for_ad(g: Graph) -> Graph {
    use rlx_ir::op::BinaryOp;

    let needs = g.nodes().iter().any(|n| {
        matches!(
            &n.op, Op::Scan { num_bcast, .. } if *num_bcast > 0
        )
    });
    if !needs {
        return g;
    }

    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        match &node.op {
            Op::Scan {
                body,
                length,
                save_trajectory,
                num_bcast,
                num_xs,
                num_checkpoints,
            } if *num_bcast > 0 => {
                // Each bcast input gets multiplied by an
                // `[length, 1, ..., 1]` ones constant of matching dtype
                // (broadcast against the bcast's natural shape) to
                // produce a `[length, *bcast]` materialised xs.
                let bcast_base = 1;
                let xs_base = 1 + *num_bcast as usize;

                let mut new_scan_inputs = vec![new_inputs[0]];

                // Original xs first remain xs.
                let mut materialised_xs: Vec<NodeId> = Vec::new();
                for i in 0..*num_bcast as usize {
                    let b_id = new_inputs[bcast_base + i];
                    let b_shape = out.node(b_id).shape.clone();
                    let dtype = b_shape.dtype();

                    // ones with shape [length, 1, 1, ...] (matching b's rank
                    // beyond the leading axis we're prepending). Broadcast
                    // against b of shape [*bcast] gives [length, *bcast].
                    let mut ones_dims: Vec<rlx_ir::Dim> =
                        vec![rlx_ir::Dim::Static(*length as usize)];
                    for _ in 0..b_shape.rank() {
                        ones_dims.push(rlx_ir::Dim::Static(1));
                    }
                    let ones_shape = rlx_ir::Shape::from_dims(&ones_dims, dtype);
                    let n_elems: usize = ones_dims
                        .iter()
                        .map(|d| match d {
                            rlx_ir::Dim::Static(n) => *n,
                            rlx_ir::Dim::Dynamic(_) => 1,
                        })
                        .product();
                    let elem_size = dtype.size_bytes();
                    let mut data = Vec::with_capacity(n_elems * elem_size);
                    match dtype {
                        rlx_ir::DType::F64 => {
                            for _ in 0..n_elems {
                                data.extend_from_slice(&1.0_f64.to_le_bytes());
                            }
                        }
                        rlx_ir::DType::F32 => {
                            for _ in 0..n_elems {
                                data.extend_from_slice(&1.0_f32.to_le_bytes());
                            }
                        }
                        other => {
                            panic!("materialize_bcasts_for_ad: unsupported bcast dtype {other:?}")
                        }
                    }
                    let ones = out.add_node(Op::Constant { data }, vec![], ones_shape);

                    // Output shape of broadcast Mul: [length, *bcast].
                    let mut xs_dims: Vec<rlx_ir::Dim> = vec![rlx_ir::Dim::Static(*length as usize)];
                    for i in 0..b_shape.rank() {
                        xs_dims.push(b_shape.dim(i));
                    }
                    let xs_shape = rlx_ir::Shape::from_dims(&xs_dims, dtype);
                    let xs_id = out.add_node(Op::Binary(BinaryOp::Mul), vec![ones, b_id], xs_shape);
                    materialised_xs.push(xs_id);
                }

                new_scan_inputs.extend_from_slice(&materialised_xs);
                for i in 0..*num_xs as usize {
                    new_scan_inputs.push(new_inputs[xs_base + i]);
                }

                let new_id = out.add_node(
                    Op::Scan {
                        body: body.clone(),
                        length: *length,
                        save_trajectory: *save_trajectory,
                        num_bcast: 0,
                        num_xs: *num_bcast + *num_xs,
                        num_checkpoints: *num_checkpoints,
                    },
                    new_scan_inputs,
                    node.shape.clone(),
                );
                id_map.insert(node.id, new_id);
            }
            _ => {
                let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
                id_map.insert(node.id, new_id);
            }
        }
    }

    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

pub fn convert_scans_for_ad(g: Graph) -> Graph {
    use rlx_ir::shape::Shape as IrShape;

    // First, materialise broadcast inputs into per-step xs. The AD
    // walk and the rest of this pre-pass don't know about bcasts
    // (forward-only memory optimisation); after this rewrite the bwd
    // graph treats them as regular xs.
    let g = materialize_bcasts_for_ad(g);

    // Quick check: does any scan need rewriting? Avoid a full graph
    // rebuild when the input is already trajectory-only.
    let needs = g.nodes().iter().any(|n| {
        matches!(
            &n.op,
            Op::Scan {
                save_trajectory: false,
                ..
            }
        )
    });
    if !needs {
        return g;
    }

    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        match &node.op {
            Op::Scan {
                body,
                length,
                save_trajectory: false,
                num_xs,
                num_checkpoints,
                ..
            } => {
                let carry_shape = node.shape.clone();
                // Trajectory shape: [length, *carry_shape].
                //
                // NB: when `num_checkpoints` is set (recursive
                // checkpointing), the executor only writes `K` rows
                // into this buffer (the saved checkpoints, indexed by
                // k=0..K-1 at offsets 0..K·cb). Rows K..length-1 stay
                // zero. The Narrow + Reshape below extracts row
                // `length-1`, which is **zero** in checkpointed mode
                // — i.e. the rewritten forward output is wrong (the
                // FORWARD value of `scan_checkpointed` followed by a
                // direct read is not currently supported).
                //
                // Backward gradients are still correct: Narrow's VJP
                // scatters the upstream into row `length-1` of the
                // gradient tensor, ScanBackward reads upstream[t·cb]
                // for t in 0..length finds zero everywhere except at
                // t=length-1 where it picks up `d_loss`, and the
                // segment-cached recompute uses the K saved
                // checkpoints (at offsets 0..K·cb) plus the forward
                // body to reconstruct intermediate carries.
                let mut traj_dims: Vec<Dim> = Vec::with_capacity(carry_shape.rank() + 1);
                traj_dims.push(Dim::Static(*length as usize));
                for i in 0..carry_shape.rank() {
                    traj_dims.push(carry_shape.dim(i));
                }
                let traj_shape = IrShape::from_dims(&traj_dims, carry_shape.dtype());
                let traj = out.add_node(
                    Op::Scan {
                        body: body.clone(),
                        length: *length,
                        save_trajectory: true,
                        num_bcast: 0,
                        num_xs: *num_xs,
                        num_checkpoints: *num_checkpoints,
                    },
                    new_inputs,
                    traj_shape,
                );
                // Narrow last row → [1, *carry].
                let mut narrow_dims: Vec<Dim> = Vec::with_capacity(carry_shape.rank() + 1);
                narrow_dims.push(Dim::Static(1));
                for i in 0..carry_shape.rank() {
                    narrow_dims.push(carry_shape.dim(i));
                }
                let narrow_shape = IrShape::from_dims(&narrow_dims, carry_shape.dtype());
                let narrowed = out.add_node(
                    Op::Narrow {
                        axis: 0,
                        start: (*length as usize).saturating_sub(1),
                        len: 1,
                    },
                    vec![traj],
                    narrow_shape,
                );
                // Reshape to drop the leading 1 → carry_shape.
                let new_shape: Vec<i64> = (0..carry_shape.rank())
                    .map(|i| match carry_shape.dim(i) {
                        Dim::Static(n) => n as i64,
                        Dim::Dynamic(_) => -1,
                    })
                    .collect();
                let final_id = out.add_node(Op::Reshape { new_shape }, vec![narrowed], carry_shape);
                id_map.insert(node.id, final_id);
            }
            _ => {
                let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
                id_map.insert(node.id, new_id);
            }
        }
    }

    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Pre-AD pass: inline `Op::CustomFn` nodes that have no `vjp_body`
/// by expanding their `fwd_body` into the parent graph. After this
/// rewrite the reverse walk sees only primitive ops and can apply
/// the per-op VJP rules recursively.
pub fn inline_custom_fn_for_autodiff(g: Graph) -> Graph {
    use rlx_fusion::control_flow::inline_subgraph_into;

    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::CustomFn {
                vjp_body: None,
                fwd_body,
                num_inputs,
                ..
            } => {
                assert_eq!(
                    new_inputs.len(),
                    *num_inputs as usize,
                    "custom_fn: outer input count mismatch"
                );
                inline_subgraph_into(fwd_body, &new_inputs, &mut out)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|i| id_map[i]).collect();
    out.set_outputs(new_outputs);
    out
}

/// expand / exp for the SSM unroll). Mirrors the HLO emission
/// decomposition in `rlx-tpu/src/unfuse.rs` and the MLX
/// `lower.rs` SelectiveScan composition.

/// Inverse of `unbroadcast`: broadcast a small tensor up to a target
/// shape via `Op::Expand`. Convenience wrapper for the few VJPs that
/// need it.
pub(crate) fn unbroadcast_inverse(x: NodeId, target: &Shape, bwd: &mut Graph) -> NodeId {
    let target_dims: Vec<i64> = target
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    bwd.add_node(
        Op::Expand {
            target_shape: target_dims,
        },
        vec![x],
        target.clone(),
    )
}

/// Expand a gradient back to its pre-reduction shape: optionally
/// reshape to insert size-1 axes (when forward had `keep_dim=false`),
/// then `Op::Expand` to broadcast to `x_shape`. The reverse of
/// `Reduce::Sum`.
fn expand_to(
    grad: NodeId,
    x_shape: &Shape,
    axes: &[usize],
    keep_dim: bool,
    bwd: &mut Graph,
) -> NodeId {
    let mut current = grad;
    if !keep_dim {
        // Insert size-1 axes at the reduced positions so the rank
        // matches x_shape and Expand can broadcast cleanly.
        let kept_dims: Vec<Dim> = (0..x_shape.rank())
            .map(|i| {
                if axes.contains(&i) {
                    Dim::Static(1)
                } else {
                    x_shape.dim(i)
                }
            })
            .collect();
        let kept = Shape::from_dims(&kept_dims, x_shape.dtype());
        current = reshape_to(current, &kept, bwd);
    }
    let target_shape: Vec<i64> = x_shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    bwd.add_node(Op::Expand { target_shape }, vec![current], x_shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_of_add_is_identity() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.input("y", Shape::new(&[4], DType::F32));
        let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![z]);

        let bwd = grad(&g, &[x, y]);
        // bwd graph should expose two outputs: dz/dx and dz/dy, both = d_output.
        assert_eq!(bwd.outputs.len(), 2);
    }

    #[test]
    fn grad_of_mul_uses_other_operand() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.input("y", Shape::new(&[4], DType::F32));
        let z = g.binary(BinaryOp::Mul, x, y, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![z]);

        let bwd = grad(&g, &[x, y]);
        // bwd should contain Mul nodes (upstream * y, upstream * x).
        assert!(
            bwd.nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
                .count()
                >= 2
        );
    }

    #[test]
    fn grad_with_loss_returns_loss_first() {
        let mut g = Graph::new("loss");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.input("y", Shape::new(&[4], DType::F32));
        let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![z]);

        let bwd = grad_with_loss(&g, &[x, y]);
        // [loss, dz/dx, dz/dy] — three outputs.
        assert_eq!(bwd.outputs.len(), 3);
    }

    #[test]
    fn grad_of_dense_solve_emits_implicit_function_rule() {
        // Forward:
        //   A      : Param [2,2]
        //   b      : Input [2]
        //   x      = solve(A, b)
        //   loss   = sum(x)         (scalar)
        //
        // Backward must contain:
        //   - a Transpose of A
        //   - a second DenseSolve (dx_int = solve(Aᵀ, upstream))
        //   - a MatMul (the outer product dx_int · xᵀ)
        //   - a Neg (the −outer)
        //
        // Outputs are [loss, dA, db].
        let mut g = Graph::new("solve_test");
        let a = g.param("A", Shape::new(&[2, 2], DType::F32));
        let b = g.input("b", Shape::new(&[2], DType::F32));
        let x = g.dense_solve(a, b, Shape::new(&[2], DType::F32));
        let loss = g.reduce(
            x,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F32),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[a, b]);
        assert_eq!(bwd.outputs.len(), 3, "expect [loss, dA, db]");

        let count =
            |pred: fn(&Op) -> bool| -> usize { bwd.nodes().iter().filter(|n| pred(&n.op)).count() };

        // Forward is mirrored into bwd, so we expect 1 + 1 = 2 DenseSolves
        // (forward copy + reverse).
        assert!(
            count(|o| matches!(o, Op::DenseSolve)) >= 2,
            "expected ≥2 DenseSolve nodes (forward mirror + reverse), got\n{bwd}"
        );
        assert!(
            count(|o| matches!(o, Op::Transpose { .. })) >= 1,
            "expected a Transpose for Aᵀ, got\n{bwd}"
        );
        assert!(
            count(|o| matches!(o, Op::MatMul)) >= 1,
            "expected a MatMul for the outer product, got\n{bwd}"
        );
        assert!(
            count(|o| matches!(o, Op::Activation(Activation::Neg))) >= 1,
            "expected a Neg for −outer, got\n{bwd}"
        );
    }

    #[test]
    fn inline_if_replaces_with_where() {
        // Build a parent graph:
        //   x       : Input
        //   pred    : Input (scalar bool)
        //   then_b  : sub-graph with Input(0) → Activation(Relu)
        //   else_b  : sub-graph with Input(0) → Activation(Sigmoid)
        //   out     : If(pred, [x] -> then_b, else_b)
        let s = Shape::new(&[4], DType::F32);
        let pred_s = Shape::new(&[1], DType::F32);

        let mut then_g = Graph::new("then_branch");
        let then_in = then_g.input("captured", s.clone());
        let then_out = then_g.activation(Activation::Relu, then_in, s.clone());
        then_g.set_outputs(vec![then_out]);

        let mut else_g = Graph::new("else_branch");
        let else_in = else_g.input("captured", s.clone());
        let else_out = else_g.activation(Activation::Sigmoid, else_in, s.clone());
        else_g.set_outputs(vec![else_out]);

        let mut g = Graph::new("parent");
        let x = g.input("x", s.clone());
        let pred = g.input("pred", pred_s);
        let if_out = g.add_node(
            Op::If {
                then_branch: Box::new(then_g),
                else_branch: Box::new(else_g),
            },
            vec![pred, x],
            s,
        );
        g.set_outputs(vec![if_out]);

        let inlined = rlx_fusion::control_flow::inline_if(g);

        // After inlining: no Op::If, exactly one Op::Where, one
        // Activation(Relu), one Activation(Sigmoid). Inputs (x,
        // pred) and the original output count are preserved.
        let has_if = inlined
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::If { .. }));
        let has_where = inlined.nodes().iter().any(|n| matches!(n.op, Op::Where));
        let has_relu = inlined
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Activation(Activation::Relu)));
        let has_sigmoid = inlined
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Activation(Activation::Sigmoid)));
        assert!(!has_if, "Op::If should be inlined away");
        assert!(has_where, "Op::Where should replace the Op::If");
        assert!(has_relu, "then_branch's Activation(Relu) should be inlined");
        assert!(
            has_sigmoid,
            "else_branch's Activation(Sigmoid) should be inlined"
        );
        assert_eq!(inlined.outputs.len(), 1);
    }

    #[test]
    fn grad_through_if_propagates() {
        // Sanity: autodiff a graph with Op::If and confirm it
        // produces a gradient (the Where VJP handles the join).
        let s = Shape::new(&[4], DType::F32);
        let pred_s = Shape::new(&[1], DType::F32);

        let mut then_g = Graph::new("th");
        let ti = then_g.input("c", s.clone());
        let to = then_g.binary(BinaryOp::Mul, ti, ti, s.clone());
        then_g.set_outputs(vec![to]);

        let mut else_g = Graph::new("el");
        let ei = else_g.input("c", s.clone());
        let eo = else_g.activation(Activation::Relu, ei, s.clone());
        else_g.set_outputs(vec![eo]);

        let mut g = Graph::new("parent");
        let x = g.input("x", s.clone());
        let pred = g.input("pred", pred_s);
        let z = g.add_node(
            Op::If {
                then_branch: Box::new(then_g),
                else_branch: Box::new(else_g),
            },
            vec![pred, x],
            s,
        );
        g.set_outputs(vec![z]);

        let bwd = grad_with_loss(&g, &[x]);
        // [loss, dz/dx] — two outputs.
        assert_eq!(bwd.outputs.len(), 2, "expected loss + 1 grad output");
    }

    #[test]
    fn unroll_while_replicates_body_n_times() {
        // Build a parent graph:
        //   x   : Input
        //   out : While(cond=trivial, body=Activation(Relu), N=3)
        // After unrolling we expect zero Op::While, three Activation
        // (Relu) nodes (one per replica).
        let s = Shape::new(&[4], DType::F32);
        let bool_s = Shape::new(&[1], DType::F32);

        let mut cond_g = Graph::new("cond");
        let ci = cond_g.input("c", s.clone());
        // dummy bool: just feed input through (cond is not evaluated
        // by the unroll pass, so its body doesn't matter).
        cond_g.set_outputs(vec![ci]);
        // Replace output shape: cond's output is logically a scalar
        // bool — but the unroll pass never inspects it.
        let _ = bool_s;

        let mut body_g = Graph::new("body");
        let bi = body_g.input("c", s.clone());
        let bo = body_g.activation(Activation::Relu, bi, s.clone());
        body_g.set_outputs(vec![bo]);

        let mut g = Graph::new("parent");
        let x = g.input("x", s.clone());
        let w = g.add_node(
            Op::While {
                cond: Box::new(cond_g),
                body: Box::new(body_g),
                max_iterations: Some(3),
            },
            vec![x],
            s,
        );
        g.set_outputs(vec![w]);

        let unrolled = rlx_fusion::control_flow::unroll_while(g);

        let has_while = unrolled
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::While { .. }));
        let relu_count = unrolled
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Activation(Activation::Relu)))
            .count();
        assert!(!has_while, "Op::While should be unrolled away");
        assert_eq!(
            relu_count, 3,
            "body's Activation(Relu) should appear once per iteration"
        );
        assert_eq!(unrolled.outputs.len(), 1);
    }

    #[test]
    fn grad_through_while_propagates() {
        // Sanity: autodiff a graph with Op::While and confirm the
        // gradient pipeline produces a result (the unroll pass turns
        // it into a chain of body replicas before the gradient walk).
        let s = Shape::new(&[4], DType::F32);

        let mut cond_g = Graph::new("cond");
        let ci = cond_g.input("c", s.clone());
        cond_g.set_outputs(vec![ci]);

        let mut body_g = Graph::new("body");
        let bi = body_g.input("c", s.clone());
        let bo = body_g.binary(BinaryOp::Mul, bi, bi, s.clone());
        body_g.set_outputs(vec![bo]);

        let mut g = Graph::new("parent");
        let x = g.input("x", s.clone());
        let w = g.add_node(
            Op::While {
                cond: Box::new(cond_g),
                body: Box::new(body_g),
                max_iterations: Some(2),
            },
            vec![x],
            s,
        );
        g.set_outputs(vec![w]);

        let bwd = grad_with_loss(&g, &[x]);
        assert_eq!(bwd.outputs.len(), 2, "expected loss + 1 grad output");
    }

    /// Build a tiny BERT-style FTL graph with the given bias mode.
    /// Returns (graph, hidden_input_id, all_param_ids).
    fn build_ftl_graph(has_bias: bool) -> (Graph, NodeId, Vec<NodeId>) {
        // B=1, S=2, hidden=4, heads=2, head_dim=2, intermediate=8.
        let mut g = Graph::new("ftl_test");
        let h_shape = Shape::new(&[1, 2, 4], DType::F32);
        let h = g.input("h", h_shape.clone());
        let qkv_w = g.param("qkv_w", Shape::new(&[4, 12], DType::F32));
        let out_w = g.param("out_w", Shape::new(&[4, 4], DType::F32));
        let ln1_g = g.param("ln1_g", Shape::new(&[4], DType::F32));
        let fc1_w = g.param("fc1_w", Shape::new(&[4, 8], DType::F32));
        let fc2_w = g.param("fc2_w", Shape::new(&[8, 4], DType::F32));
        let ln2_g = g.param("ln2_g", Shape::new(&[4], DType::F32));
        let mask = g.input("mask", Shape::new(&[1, 2, 2, 2], DType::F32));

        let (inputs, params) = if has_bias {
            let qkv_b = g.param("qkv_b", Shape::new(&[12], DType::F32));
            let out_b = g.param("out_b", Shape::new(&[4], DType::F32));
            let ln1_b = g.param("ln1_b", Shape::new(&[4], DType::F32));
            let fc1_b = g.param("fc1_b", Shape::new(&[8], DType::F32));
            let fc2_b = g.param("fc2_b", Shape::new(&[4], DType::F32));
            let ln2_b = g.param("ln2_b", Shape::new(&[4], DType::F32));
            (
                vec![
                    h, qkv_w, qkv_b, out_w, out_b, ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b, ln2_g,
                    ln2_b, mask,
                ],
                vec![
                    qkv_w, qkv_b, out_w, out_b, ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b, ln2_g,
                    ln2_b,
                ],
            )
        } else {
            (
                vec![h, qkv_w, out_w, ln1_g, fc1_w, fc2_w, ln2_g, mask],
                vec![qkv_w, out_w, ln1_g, fc1_w, fc2_w, ln2_g],
            )
        };
        let y = g.add_node(
            Op::FusedTransformerLayer {
                num_heads: 2,
                head_dim: 2,
                intermediate_size: 8,
                eps1: 1e-5,
                eps2: 1e-5,
                activation: rlx_ir::op::Activation::Gelu,
                has_bias,
            },
            inputs,
            h_shape,
        );
        g.set_outputs(vec![y]);
        (g, h, params)
    }

    #[test]
    fn unfuse_decomposes_fused_transformer_layer() {
        // After rlx_fusion::unfuse_fused_for_autodiff, the FTL node is gone and
        // primitives appear: at least 4 MatMul (qkv, out, fc1, fc2),
        // 1 Attention, 2 LayerNorm, plus narrows / adds / activation.
        let (g, _h, _params) = build_ftl_graph(true);
        let unfused = rlx_fusion::unfuse_fused_for_autodiff(g);

        let has_ftl = unfused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::FusedTransformerLayer { .. }));
        assert!(!has_ftl, "Op::FusedTransformerLayer should be unfused");

        let count = |pred: fn(&Op) -> bool| -> usize {
            unfused.nodes().iter().filter(|n| pred(&n.op)).count()
        };
        assert!(
            count(|o| matches!(o, Op::MatMul)) >= 4,
            "expected >=4 MatMul after FTL unfuse"
        );
        assert_eq!(
            count(|o| matches!(o, Op::Attention { .. })),
            1,
            "expected exactly 1 Attention after FTL unfuse"
        );
        assert_eq!(
            count(|o| matches!(o, Op::LayerNorm { .. })),
            2,
            "expected exactly 2 LayerNorm after FTL unfuse"
        );
        assert!(
            count(|o| matches!(o, Op::Narrow { .. })) >= 3,
            "expected >=3 Narrow (Q/K/V split) after FTL unfuse"
        );
        assert_eq!(
            count(|o| matches!(o, Op::Activation(_))),
            1,
            "expected exactly 1 Activation (FFN) after FTL unfuse"
        );
    }

    #[test]
    fn grad_through_fused_transformer_layer_propagates() {
        // End-to-end: grad_with_loss through an FTL graph returns
        // [loss, ...grads]. Confirms every primitive emitted by the
        // unfuse has a VJP rule on the gradient walk.
        let (g, _h, params) = build_ftl_graph(true);
        let bwd = grad_with_loss(&g, &params);
        assert_eq!(
            bwd.outputs.len(),
            1 + params.len(),
            "expected loss + {} param grads",
            params.len()
        );
    }

    #[test]
    fn grad_through_fused_transformer_layer_no_bias() {
        // No-bias variant exercises the synthesized zero-beta path
        // for both LayerNorms.
        let (g, _h, params) = build_ftl_graph(false);
        let bwd = grad_with_loss(&g, &params);
        assert_eq!(
            bwd.outputs.len(),
            1 + params.len(),
            "expected loss + {} param grads (no-bias)",
            params.len()
        );
    }

    /// Build a tiny SelectiveScan graph: B=1, S=3, H=2, N=4.
    /// Returns (graph, [x, delta, a, b, c]).
    fn build_ssm_graph() -> (Graph, NodeId, Vec<NodeId>) {
        let mut g = Graph::new("ssm_test");
        let bsh = Shape::new(&[1, 3, 2], DType::F32);
        let hn = Shape::new(&[2, 4], DType::F32);
        let bsn = Shape::new(&[1, 3, 4], DType::F32);

        let x = g.input("x", bsh.clone());
        let delta = g.input("delta", bsh.clone());
        let a = g.param("a", hn);
        let b = g.input("b", bsn.clone());
        let c = g.input("c", bsn);
        let y = g.selective_scan(x, delta, a, b, c, 4, bsh);
        g.set_outputs(vec![y]);
        (g, x, vec![a])
    }

    #[test]
    fn unfuse_decomposes_selective_scan() {
        // After unfuse, no Op::SelectiveScan; instead we see Concat
        // (one for S>1), per-step Reduce(Sum), per-step Activation::Exp,
        // and many Mul / Add / Narrow / Reshape / Expand nodes.
        // S=3 → 3 timesteps.
        let (g, _x, _params) = build_ssm_graph();
        let unfused = rlx_fusion::unfuse_fused_for_autodiff(g);

        let has_ssm = unfused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SelectiveScan { .. }));
        assert!(!has_ssm, "Op::SelectiveScan should be unfused");

        let count = |pred: fn(&Op) -> bool| -> usize {
            unfused.nodes().iter().filter(|n| pred(&n.op)).count()
        };
        assert_eq!(
            count(|o| matches!(o, Op::Concat { .. })),
            1,
            "expected 1 Concat (over the 3 time steps)"
        );
        assert_eq!(
            count(|o| matches!(
                o,
                Op::Reduce {
                    op: ReduceOp::Sum,
                    ..
                }
            )),
            3,
            "expected one Reduce(Sum) per time step (S=3)"
        );
        assert_eq!(
            count(|o| matches!(o, Op::Activation(Activation::Exp))),
            3,
            "expected one exp(δA) per time step (S=3)"
        );
        assert!(
            count(|o| matches!(o, Op::Narrow { .. })) >= 12,
            "expected >=12 Narrows (4 per step × 3 steps)"
        );
    }

    #[test]
    fn grad_through_selective_scan_propagates() {
        // End-to-end: grad_with_loss through SelectiveScan returns
        // [loss, da] — confirms every primitive emitted by the
        // unroll has a VJP rule on the gradient walk (Mul, Add,
        // Activation::Exp, Reduce::Sum, Concat, Narrow, Reshape,
        // Expand).
        let (g, _x, params) = build_ssm_graph();
        let bwd = grad_with_loss(&g, &params);
        assert_eq!(
            bwd.outputs.len(),
            1 + params.len(),
            "expected loss + {} param grads",
            params.len()
        );
    }

    /// Tiny GatedDeltaNet: B=1, S=3, H=2, N=4.
    fn build_gdn_graph() -> (Graph, NodeId, Vec<NodeId>) {
        let (b, s, h, n) = (1usize, 3, 2, 4);
        let mut g = Graph::new("gdn_test");
        let bshn = Shape::new(&[b, s, h, n], DType::F32);
        let bsh = Shape::new(&[b, s, h], DType::F32);
        let q = g.input("q", bshn.clone());
        let k = g.input("k", bshn.clone());
        let v = g.input("v", bshn.clone());
        let g_in = g.input("g", bsh.clone());
        let beta = g.input("beta", bsh);
        let y = g.gated_delta_net(q, k, v, g_in, beta, n, bshn);
        g.set_outputs(vec![y]);
        (g, q, vec![q, k, v, g_in, beta])
    }

    #[test]
    fn unfuse_decomposes_gated_delta_net() {
        let (g, _q, _params) = build_gdn_graph();
        let unfused = rlx_fusion::unfuse_fused_for_autodiff(g);

        let has_gdn = unfused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedDeltaNet { .. }));
        assert!(!has_gdn, "Op::GatedDeltaNet should be unfused");

        let count = |pred: fn(&Op) -> bool| -> usize {
            unfused.nodes().iter().filter(|n| pred(&n.op)).count()
        };
        assert_eq!(
            count(|o| matches!(o, Op::Concat { .. })),
            1,
            "expected 1 Concat over S=3 steps"
        );
        assert!(
            count(|o| matches!(o, Op::MatMul)) >= 3,
            "expected >=3 MatMul per step (sk + out) × S=3"
        );
        assert_eq!(
            count(|o| matches!(o, Op::Activation(Activation::Exp))),
            3,
            "expected one exp(g) per time step"
        );
    }

    #[test]
    fn grad_through_gated_delta_net_propagates() {
        let (g, _q, params) = build_gdn_graph();
        let bwd = grad_with_loss(&g, &params);
        assert_eq!(
            bwd.outputs.len(),
            1 + params.len(),
            "expected loss + {} input grads",
            params.len()
        );
    }

    #[test]
    fn custom_fn_vjp_body_is_inlined_into_bwd() {
        // Forward: y = x² via custom_fn (fwd_body = Mul(x, x)).
        // Override VJP to return Activation::Sin(d_output) — a unique
        // marker that natural autodiff of Mul would never emit. If
        // grad_with_loss inlines the override correctly, the bwd graph
        // must contain a Sin node; if it falls back to recursing into
        // fwd_body, it would emit two Muls (upstream·x + x·upstream)
        // and no Sin.
        let n = 4usize;
        let shape = Shape::new(&[n], DType::F32);

        // fwd_body: x → x · x.
        let mut fwd_body = Graph::new("square_fwd");
        let xb = fwd_body.input("x", shape.clone());
        let yb = fwd_body.binary(BinaryOp::Mul, xb, xb, shape.clone());
        fwd_body.set_outputs(vec![yb]);

        // vjp_body: (x, primal_output, d_output) → sin(d_output).
        let mut vjp_body = Graph::new("square_vjp");
        let _vx = vjp_body.input("x", shape.clone());
        let _vp = vjp_body.input("primal_output", shape.clone());
        let vd = vjp_body.input("d_output", shape.clone());
        let dx = vjp_body.activation(Activation::Sin, vd, shape.clone());
        vjp_body.set_outputs(vec![dx]);

        let mut g = Graph::new("custom_fn_test");
        let x = g.input("x", shape.clone());
        let y = g.custom_fn(vec![x], fwd_body, Some(vjp_body), None);
        let loss = g.reduce(
            y,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F32),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[x]);
        assert_eq!(bwd.outputs.len(), 2, "expect [loss, dx]");
        let sin_count = bwd
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Activation(Activation::Sin)))
            .count();
        assert!(
            sin_count >= 1,
            "expected the vjp_body's Sin to be inlined into bwd, got\n{bwd}"
        );
    }

    #[test]
    fn custom_fn_without_vjp_inlines_fwd_body_for_autodiff() {
        // Forward: y = x² via custom_fn without vjp_body. After the
        // inline pre-pass, autodiff should recurse into Mul and emit
        // dx = 2·x·d_output (two Mul nodes in the backward graph).
        let n = 4usize;
        let shape = Shape::new(&[n], DType::F32);

        let mut fwd_body = Graph::new("square_fwd");
        let xb = fwd_body.input("x", shape.clone());
        let yb = fwd_body.binary(BinaryOp::Mul, xb, xb, shape.clone());
        fwd_body.set_outputs(vec![yb]);

        let mut g = Graph::new("custom_fn_no_vjp");
        let x = g.input("x", shape.clone());
        let y = g.custom_fn(vec![x], fwd_body, None, None);
        let loss = g.reduce(
            y,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F32),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[x]);
        assert_eq!(bwd.outputs.len(), 2, "expect [loss, dx]");
        let custom_fn_count = bwd
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::CustomFn { .. }))
            .count();
        assert_eq!(
            custom_fn_count, 0,
            "CustomFn should be inlined away before autodiff"
        );
        let mul_count = bwd
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
            .count();
        assert!(
            mul_count >= 2,
            "expected Mul-based VJP for x², got\n{bwd}"
        );
    }

    #[test]
    fn convert_scans_for_ad_forces_save_trajectory_true() {
        // grad_with_loss runs `convert_scans_for_ad` as a pre-pass: any
        // forward Op::Scan with `save_trajectory: false` is rewritten
        // to `save_trajectory: true` followed by Narrow + Reshape so
        // the reverse pass has the trajectory it needs. This test
        // verifies the rewrite happens — the bwd graph should contain
        // at least one Scan with save_trajectory == true.
        let n = 2usize;
        let length = 3u32;
        let carry = Shape::new(&[n], DType::F32);
        let xs_shape = Shape::new(&[length as usize, n], DType::F32);

        // body: (carry, x_t) → carry + x_t. One primal Input each.
        let mut body = Graph::new("scan_body");
        let bc = body.input("carry", carry.clone());
        let bx = body.input("x_t", carry.clone());
        let by = body.binary(BinaryOp::Add, bc, bx, carry.clone());
        body.set_outputs(vec![by]);

        let mut g = Graph::new("scan_save_false");
        let init = g.input("init", carry.clone());
        let xs = g.input("xs", xs_shape);
        let scan_out = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 1,
                num_checkpoints: 0,
            },
            vec![init, xs],
            carry.clone(),
        );
        let loss = g.reduce(
            scan_out,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F32),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[init, xs]);
        let saved_traj = bwd.nodes().iter().any(|n| {
            matches!(
                &n.op,
                Op::Scan {
                    save_trajectory: true,
                    ..
                }
            )
        });
        assert!(
            saved_traj,
            "convert_scans_for_ad should rewrite save_trajectory=false → \
             save_trajectory=true in the AD-prepared graph; got\n{bwd}"
        );
    }
}
