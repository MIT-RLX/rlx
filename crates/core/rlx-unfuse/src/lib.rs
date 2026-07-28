// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared IR-level "unfusion" pass for the GPU-family backends.
//!
//! CUDA, ROCm and wgpu all decompose the same composed ops (FusedSwiGLU,
//! LoraMatMul, FusedAttentionBlock, FusedTransformerLayer, DotGeneral, If,
//! While) into primitive sequences, and give Binary / Compare / Where a
//! broadcast prologue when their input shapes mismatch. ~80% of that logic
//! is byte-for-byte identical across the three backends; the divergence is
//! confined to a handful of per-backend policy decisions:
//!
//!   * whether a native `fused_attn_block` kernel can keep certain
//!     FusedAttentionBlock nodes intact instead of expanding them (CUDA);
//!   * whether the autodiff `AttentionBackward` op also gets promoted from
//!     rank-3 to rank-4 (CUDA);
//!   * whether biased projections fold into `FusedMatMulBiasAct` and
//!     residual+norm pairs fold into `FusedResidualLN` (wgpu);
//!   * whether the Attention kernel accepts rank-3 `[B, S, H·D]` inputs via
//!     per-axis strides — eliding the reshape/transpose to `[B, H, S, D]`
//!     and letting the mask ride along the strides (wgpu).
//!
//! Those decisions are captured by [`DecomposePolicy`]; each backend supplies
//! a tiny impl and calls [`unfuse`] (and, for wgpu, [`collapse_reshapes`]).
//! The graph produced for any given backend is byte-for-byte what its old
//! standalone `unfuse` produced.

use std::collections::HashMap;

use rlx_ir::op::{Activation, AttentionBwdWrt, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

/// Per-backend knobs that steer the shared decompose pass.
///
/// Every method has a default that reproduces the plain "materialize
/// everything to primitives, rank-4 attention, no native fused kernels"
/// behavior (which is exactly what ROCm wants). Backends override only the
/// decisions where their lowering diverges.
pub trait DecomposePolicy {
    /// Ops the pass rewrites into primitive sequences. The default is the
    /// allowlist shared by all three GPU-family backends; override only if a
    /// backend grows a different composite surface.
    ///
    /// `FusedAttentionBlock` is listed so the driver *visits* it — the arm
    /// then keeps it native (see [`DecomposePolicy::fab_native`]) or expands
    /// it.
    fn should_unfuse(&self, op: &Op) -> bool {
        match op {
            Op::FusedSwiGLU { .. } => !self.swiglu_native(),
            Op::LoraMatMul { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::DotGeneral { .. }
            | Op::If { .. }
            | Op::While { .. } => true,
            _ => false,
        }
    }

    /// Keep `Op::FusedSwiGLU` intact for a native fused kernel (CUDA/ROCm).
    /// Default: decompose to Narrow + Silu + Mul.
    fn swiglu_native(&self) -> bool {
        false
    }

    /// True when a `FusedAttentionBlock` with this output shape can be served
    /// by a native fused-attention kernel and should be left intact instead of
    /// decomposed. Default: never native ⇒ always expand.
    fn fab_native(&self, _out_shape: &Shape) -> bool {
        false
    }

    /// True when rank-3 `Op::AttentionBackward` should be promoted to rank-4
    /// like the forward `Op::Attention` (i.e. the backend has a rank-4-only
    /// attention-backward kernel). Default: false.
    fn promote_attention_backward(&self) -> bool {
        false
    }

    /// True when biased projections should fold into `Op::FusedMatMulBiasAct`
    /// (matmul + bias [+ activation] in one native kernel) instead of emitting
    /// `matmul` + `BinaryOp::Add` [+ activation]. Default: false.
    fn fold_matmul_bias_act(&self) -> bool {
        false
    }

    /// True when residual+norm pairs should fold into `Op::FusedResidualLN`
    /// instead of emitting `BinaryOp::Add` + `layer_norm`. Default: false.
    fn fold_residual_ln(&self) -> bool {
        false
    }

    /// True when the Attention kernel reads Q/K/V via per-axis strides and
    /// accepts rank-3 `[B, S, H·D]` directly — so the pass skips the
    /// reshape+transpose to `[B, H, S, D]` (and the inverse), and passes the
    /// mask through on its native strides. Default: false ⇒ materialize
    /// rank-4 and reshape+expand the mask.
    fn attention_accepts_rank3(&self) -> bool {
        false
    }
}

pub fn unfuse(graph: Graph, policy: &dyn DecomposePolicy) -> Graph {
    // Skip rebuild only if no fused/composed ops AND every Binary /
    // Compare / Where already has matching input element counts. The
    // element-wise kernels are strict-shape-matched; broadcast
    // prologues get inserted during the rewrite.
    let needs_rewrite = graph.nodes().iter().any(|n| {
        policy.should_unfuse(&n.op)
            || needs_broadcast_prologue(&graph, n)
            || needs_attn_rank3_promotion(&graph, n, policy)
    });
    if !needs_rewrite {
        return graph;
    }

    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|&id| id_map[&id]).collect();

        let new_id = match &node.op {
            Op::FusedSwiGLU { .. } => {
                expand_swiglu(&mut out, &graph, node.inputs[0], &new_inputs, &node.shape)
            }
            Op::LoraMatMul { scale } => expand_lora(
                &mut out,
                &graph,
                &node.inputs,
                &new_inputs,
                &node.shape,
                *scale,
            ),
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => {
                // Keep the block intact when a native fused-attention kernel
                // can serve it; otherwise decompose to the primitive chain.
                if policy.fab_native(&node.shape) {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                } else {
                    expand_fab(
                        &mut out,
                        &graph,
                        &node.inputs,
                        &new_inputs,
                        &node.shape,
                        *num_heads,
                        *head_dim,
                        *has_bias,
                        *has_rope,
                        policy,
                    )
                }
            }
            Op::FusedTransformerLayer {
                num_heads,
                head_dim,
                intermediate_size: _,
                eps1,
                eps2,
                activation,
                has_bias,
            } => expand_ftl(
                &mut out,
                &graph,
                &node.inputs,
                &new_inputs,
                &node.shape,
                *num_heads,
                *head_dim,
                *eps1,
                *eps2,
                *activation,
                *has_bias,
                policy,
            ),
            Op::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => expand_dot_general(
                &mut out,
                &graph,
                node.inputs[0],
                node.inputs[1],
                &new_inputs,
                &node.shape,
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            ),
            Op::If {
                then_branch,
                else_branch,
            } => expand_if(&mut out, &new_inputs, then_branch, else_branch, &node.shape),
            Op::While {
                cond,
                body,
                max_iterations,
            } => expand_while(
                &mut out,
                &new_inputs,
                cond,
                body,
                *max_iterations,
                &node.shape,
            ),
            // Promote rank-3 [B, S, H*D] Attention inputs to rank-4
            // [B, H, S, D] via Reshape + Transpose when the backend's
            // Attention kernel expects rank-4 (policy.attention_accepts_rank3
            // == false); otherwise pass rank-3 straight through.
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => {
                let q_dims = graph.node(node.inputs[0]).shape.dims();
                if q_dims.len() == 3 {
                    expand_attention_rank3(
                        &mut out,
                        &graph,
                        &node.inputs,
                        &new_inputs,
                        &node.shape,
                        *num_heads,
                        *head_dim,
                        *mask_kind,
                        *score_scale,
                        *attn_logit_softcap,
                        policy,
                    )
                } else {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                }
            }
            // Same rank-3 → rank-4 promotion for the autodiff backward op,
            // when the backend opts in (CUDA). Backends that don't just pass
            // it through — identical to the `_` fall-through below.
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => {
                let q_dims = graph.node(node.inputs[0]).shape.dims();
                if policy.promote_attention_backward() && q_dims.len() == 3 {
                    expand_attention_backward_rank3(
                        &mut out,
                        &graph,
                        &node.inputs,
                        &new_inputs,
                        &node.shape,
                        *num_heads,
                        *head_dim,
                        *mask_kind,
                        *wrt,
                    )
                } else {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                }
            }
            // Insert per-axis broadcast prologue for Binary / Compare / Where
            // when input shapes differ — the element-wise kernels are
            // strict-shape-matched. Other backends auto-broadcast in the
            // op itself.
            Op::Binary(_) | Op::Compare(_) | Op::Where => {
                // Complex Binary broadcasts in complex-element units inside the
                // backend's dedicated complex kernel; a lane-wise Expand
                // prologue would corrupt the interleaved `[re, im]` lanes, so
                // pass the mismatched-shape node straight through.
                if node.shape.dtype().is_complex() {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                } else {
                    let broadcasted = broadcast_inputs(&mut out, &new_inputs, &node.shape);
                    out.add_node(node.op.clone(), broadcasted, node.shape.clone())
                }
            }
            // Pass through everything else.
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }

    out.set_outputs(graph.outputs.iter().map(|&id| id_map[&id]).collect());
    out
}

/// Collapse redundant `Reshape`s and drop the ones that become dead.
///
/// The qwen35 `unfuse` (and the DotGeneral/matmul lowering) emit tens of
/// thousands of reshapes — many are pure no-ops (target shape already equals
/// the input's) or reshape-of-reshape chains. wgpu aliases each view onto its
/// parent's slot, so a long-lived view keeps its parent live and defeats the
/// planner's slot reuse (Bonsai-27B: ~27k reshapes → the arena can't reuse
/// activation slots → 28 GiB). Eliding the redundant reshapes shortens those
/// alias chains → tighter liveness → real reuse → smaller arena, and fewer
/// dispatches. Purely structural (semantics-preserving): a no-op reshape is the
/// identity, and `reshape(reshape(x)) == reshape(x)`.
pub fn collapse_reshapes(graph: Graph) -> Graph {
    use std::collections::HashSet;
    let resolve = |redir: &HashMap<NodeId, NodeId>, mut id: NodeId| -> NodeId {
        while let Some(&r) = redir.get(&id) {
            id = r;
        }
        id
    };
    // Pass 1 (topo order): elide no-op reshapes onto their resolved source.
    let mut redir: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        if matches!(node.op, Op::Reshape { .. }) {
            let src = resolve(&redir, node.inputs[0]);
            if graph.node(src).shape.dims() == node.shape.dims() {
                redir.insert(node.id, src);
            }
        }
    }
    if redir.is_empty() {
        return graph;
    }
    // Pass 2: reachability (through redirects) from the outputs for DCE.
    let mut keep: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = graph.outputs.iter().map(|&o| resolve(&redir, o)).collect();
    while let Some(id) = stack.pop() {
        if !keep.insert(id) {
            continue;
        }
        for &inp in &graph.node(id).inputs {
            stack.push(resolve(&redir, inp));
        }
    }
    // Pass 3: rebuild the surviving nodes in topo order.
    let mut out = Graph::new(&graph.name);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        if redir.contains_key(&node.id) || !keep.contains(&node.id) {
            continue;
        }
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|&i| map[&resolve(&redir, i)])
            .collect();
        let nid = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
        map.insert(node.id, nid);
    }
    out.set_outputs(
        graph
            .outputs
            .iter()
            .map(|&o| map[&resolve(&redir, o)])
            .collect(),
    );
    out
}

/// True if `node` is an element-wise op whose inputs don't all share
/// the same element count — i.e. a strict-shape kernel will reject
/// it and we need to insert a broadcast prologue.
fn needs_broadcast_prologue(graph: &Graph, node: &rlx_ir::Node) -> bool {
    let is_elt = matches!(node.op, Op::Binary(_) | Op::Compare(_) | Op::Where);
    if !is_elt {
        return false;
    }
    // Complex element-wise ops broadcast in complex-element units INSIDE the
    // backend's dedicated complex kernel (e.g. wgpu `binary_c64`): each element
    // is 2 (C64) / 4 (C128) interleaved f32 lanes, so a lane-wise Expand
    // prologue would corrupt the `[re, im]` pairing. Skip it — the kernel reads
    // per-operand element counts and does the modulo broadcast itself.
    if node.shape.dtype().is_complex() {
        return false;
    }
    let target_n = node.shape.num_elements().unwrap_or(0);
    node.inputs
        .iter()
        .any(|&id| graph.node(id).shape.num_elements().unwrap_or(0) != target_n)
}

/// True if the node is a rank-3 attention op that needs reshaping +
/// transposing before a rank-4-only kernel can take it. `AttentionBackward`
/// only qualifies when the backend opts into backward promotion.
fn needs_attn_rank3_promotion(
    graph: &Graph,
    node: &rlx_ir::Node,
    policy: &dyn DecomposePolicy,
) -> bool {
    let is_attn = matches!(node.op, Op::Attention { .. })
        || (policy.promote_attention_backward() && matches!(node.op, Op::AttentionBackward { .. }));
    is_attn && graph.node(node.inputs[0]).shape.rank() == 3
}

// ── Expansions ───────────────────────────────────────────────────
//
// `expand_fmb`, `expand_residual_ln` and `expand_residual_rms_norm` are
// currently unused — the fused-op producers reach this pass already
// flattened or natively lowered. Retained as the canonical reference for
// un-fusing those ops if a future code path needs the unfused IR (e.g. a
// debug build that wants it for visualization).

#[allow(dead_code)]
fn expand_fmb(
    out: &mut Graph,
    inputs: &[NodeId],
    shape: &Shape,
    activation: Option<Activation>,
) -> NodeId {
    // inputs: [x, w, b]
    let mm = out.matmul(inputs[0], inputs[1], shape.clone());
    let bias_b = broadcast_to(out, inputs[2], shape);
    let added = out.binary(BinaryOp::Add, mm, bias_b, shape.clone());
    match activation {
        None => added,
        Some(act) => out.activation(act, added, shape.clone()),
    }
}

#[allow(dead_code)]
fn expand_residual_ln(
    out: &mut Graph,
    inputs: &[NodeId],
    shape: &Shape,
    has_bias: bool,
    eps: f32,
) -> NodeId {
    // inputs: [x, residual, [bias], gamma, beta]
    let summed = out.binary(BinaryOp::Add, inputs[0], inputs[1], shape.clone());
    let summed = if has_bias {
        let bias_b = broadcast_to(out, inputs[2], shape);
        out.binary(BinaryOp::Add, summed, bias_b, shape.clone())
    } else {
        summed
    };
    let (gi, bi) = if has_bias { (3, 4) } else { (2, 3) };
    out.layer_norm(summed, inputs[gi], inputs[bi], -1, eps, shape.clone())
}

#[allow(dead_code)]
fn expand_residual_rms_norm(
    out: &mut Graph,
    inputs: &[NodeId],
    shape: &Shape,
    has_bias: bool,
    eps: f32,
) -> NodeId {
    // inputs: [x, residual, [bias], gamma, beta]
    let summed = out.binary(BinaryOp::Add, inputs[0], inputs[1], shape.clone());
    let summed = if has_bias {
        let bias_b = broadcast_to(out, inputs[2], shape);
        out.binary(BinaryOp::Add, summed, bias_b, shape.clone())
    } else {
        summed
    };
    let (gi, bi) = if has_bias { (3, 4) } else { (2, 3) };
    out.rms_norm(summed, inputs[gi], inputs[bi], eps)
}

fn expand_swiglu(
    out: &mut Graph,
    src_graph: &Graph,
    orig_src_id: NodeId,
    inputs: &[NodeId],
    out_shape: &Shape,
) -> NodeId {
    // Op::FusedSwiGLU input is concatenated [up, gate]; output last
    // dim is half. y = up * silu(gate).
    let src_dims = src_graph.node(orig_src_id).shape.dims();
    let last_idx = src_dims.len() - 1;
    let last = src_dims[last_idx].unwrap_static();
    let half = last / 2;

    // Narrow needs the full input shape with the narrow axis adjusted.
    let mut half_dims: Vec<usize> = src_dims.iter().map(|d| d.unwrap_static()).collect();
    half_dims[last_idx] = half;
    let half_shape = Shape::new(&half_dims, src_graph.node(orig_src_id).shape.dtype());

    let up = out.add_node(
        Op::Narrow {
            axis: last_idx,
            start: 0,
            len: half,
        },
        vec![inputs[0]],
        half_shape.clone(),
    );
    let gate = out.add_node(
        Op::Narrow {
            axis: last_idx,
            start: half,
            len: half,
        },
        vec![inputs[0]],
        half_shape.clone(),
    );
    let silu_g = out.activation(Activation::Silu, gate, half_shape.clone());
    out.binary(BinaryOp::Mul, up, silu_g, out_shape.clone())
}

fn expand_lora(
    out: &mut Graph,
    src_graph: &Graph,
    orig_inputs: &[NodeId],
    inputs: &[NodeId],
    out_shape: &Shape,
    scale: f32,
) -> NodeId {
    // out = x @ W + scale * (x @ A) @ B
    // inputs: [x, w, a, b]
    let dtype = out_shape.dtype();
    let m = src_graph.node(orig_inputs[0]).shape.dim(0).unwrap_static();
    let r = src_graph.node(orig_inputs[2]).shape.dim(1).unwrap_static(); // a is [k, r]
    let n = src_graph.node(orig_inputs[3]).shape.dim(1).unwrap_static(); // b is [r, n]

    let xa_shape = Shape::new(&[m, r], dtype);
    let xab_shape = Shape::new(&[m, n], dtype);

    let base = out.matmul(inputs[0], inputs[1], out_shape.clone());
    let xa = out.matmul(inputs[0], inputs[2], xa_shape);
    let xab = out.matmul(xa, inputs[3], xab_shape.clone());

    // scalar Constant [1, 1] (Expand requires equal rank), broadcast to [m, n].
    let s_bytes = scale.to_le_bytes().to_vec();
    let s_const = out.add_node(
        Op::Constant { data: s_bytes },
        vec![],
        Shape::new(&[1, 1], DType::F32),
    );
    let s_exp = out.add_node(
        Op::Expand {
            target_shape: vec![m as i64, n as i64],
        },
        vec![s_const],
        xab_shape.clone(),
    );
    let scaled = out.binary(BinaryOp::Mul, xab, s_exp, xab_shape);
    out.binary(BinaryOp::Add, base, scaled, out_shape.clone())
}

#[allow(clippy::too_many_arguments)]
fn expand_fab(
    out: &mut Graph,
    src_graph: &Graph,
    orig_inputs: &[NodeId],
    inputs: &[NodeId],
    out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    has_bias: bool,
    has_rope: bool,
    policy: &dyn DecomposePolicy,
) -> NodeId {
    // Inputs (per IR doc):
    //   hidden, qkv_w, out_w, mask,
    //   [qkv_b, out_b]      if has_bias,
    //   [rope_cos, rope_sin] if has_rope
    let h_shape = src_graph.node(orig_inputs[0]).shape.dims();
    let batch = h_shape[0].unwrap_static();
    let seq = h_shape[1].unwrap_static();
    let inner = num_heads * head_dim;
    let dtype = out_shape.dtype();

    let qkv_shape = Shape::new(&[batch, seq, 3 * inner], dtype);
    let proj_shape = Shape::new(&[batch, seq, inner], dtype);
    let bhsd_shape = Shape::new(&[batch, num_heads, seq, head_dim], dtype);

    let mut next = 4;
    let (qkv_b_idx, out_b_idx) = if has_bias {
        let r = (next, next + 1);
        next += 2;
        r
    } else {
        (usize::MAX, usize::MAX)
    };
    let (cos_idx, sin_idx) = if has_rope {
        (next, next + 1)
    } else {
        (usize::MAX, usize::MAX)
    };

    // QKV projection. Fold bias into the matmul epilogue via
    // FusedMatMulBiasAct when the backend lowers it natively; otherwise
    // emit matmul + Add.
    let qkv = if has_bias {
        if policy.fold_matmul_bias_act() {
            out.add_node(
                Op::FusedMatMulBiasAct { activation: None },
                vec![inputs[0], inputs[1], inputs[qkv_b_idx]],
                qkv_shape.clone(),
            )
        } else {
            let qkv = out.matmul(inputs[0], inputs[1], qkv_shape.clone());
            out.binary(BinaryOp::Add, qkv, inputs[qkv_b_idx], qkv_shape.clone())
        }
    } else {
        out.matmul(inputs[0], inputs[1], qkv_shape.clone())
    };

    let q = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );
    let k = out.add_node(
        Op::Narrow {
            axis: 2,
            start: inner,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );
    let v = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * inner,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );

    // Rope requires the rank-4 [B, H, S, D] layout. Backends whose Attention
    // kernel accepts rank-3 [B, S, H·D] directly skip the reshape+transpose
    // (and the inverse) on the no-rope path; otherwise everything goes rank-4.
    let attn = if has_rope || !policy.attention_accepts_rank3() {
        let to_bhsd = |out: &mut Graph, t: NodeId| -> NodeId {
            let r = out.reshape(
                t,
                vec![batch as i64, seq as i64, num_heads as i64, head_dim as i64],
                Shape::new(&[batch, seq, num_heads, head_dim], dtype),
            );
            out.add_node(
                Op::Transpose {
                    perm: vec![0, 2, 1, 3],
                },
                vec![r],
                bhsd_shape.clone(),
            )
        };
        let mut q4 = to_bhsd(out, q);
        let mut k4 = to_bhsd(out, k);
        let v4 = to_bhsd(out, v);

        if has_rope {
            q4 = out.add_node(
                Op::Rope {
                    head_dim,
                    n_rot: head_dim,
                    // Unfusing fused attention — NeoX was the only style
                    // pre-fusion; thread the source style once fused ops carry it.
                    style: rlx_ir::op::RopeStyle::NeoX,
                },
                vec![q4, inputs[cos_idx], inputs[sin_idx]],
                bhsd_shape.clone(),
            );
            k4 = out.add_node(
                Op::Rope {
                    head_dim,
                    n_rot: head_dim,
                    // Unfusing fused attention — NeoX was the only style
                    // pre-fusion; thread the source style once fused ops carry it.
                    style: rlx_ir::op::RopeStyle::NeoX,
                },
                vec![k4, inputs[cos_idx], inputs[sin_idx]],
                bhsd_shape.clone(),
            );
        }

        // Attention with the mask passed straight through as Custom.
        let attn_4d = out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: rlx_ir::op::MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q4, k4, v4, inputs[3]],
            bhsd_shape.clone(),
        );

        // [B, H, S, D] → [B, S, H, D] → [B, S, H*D]
        let bsd = out.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![attn_4d],
            Shape::new(&[batch, seq, num_heads, head_dim], dtype),
        );
        out.reshape(
            bsd,
            vec![batch as i64, seq as i64, inner as i64],
            proj_shape.clone(),
        )
    } else {
        // Rank-3 [B, S, H·D] in → rank-3 [B, S, H·D] out (BERT-style, no rope).
        out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: rlx_ir::op::MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, inputs[3]],
            proj_shape.clone(),
        )
    };

    // Out projection, same bias-fold policy as the QKV projection.
    if has_bias {
        if policy.fold_matmul_bias_act() {
            out.add_node(
                Op::FusedMatMulBiasAct { activation: None },
                vec![attn, inputs[2], inputs[out_b_idx]],
                out_shape.clone(),
            )
        } else {
            let out_proj = out.matmul(attn, inputs[2], out_shape.clone());
            out.binary(
                BinaryOp::Add,
                out_proj,
                inputs[out_b_idx],
                out_shape.clone(),
            )
        }
    } else {
        out.matmul(attn, inputs[2], out_shape.clone())
    }
}

#[allow(clippy::too_many_arguments)]
fn expand_ftl(
    out: &mut Graph,
    src_graph: &Graph,
    orig_inputs: &[NodeId],
    inputs: &[NodeId],
    out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    eps1: f32,
    eps2: f32,
    activation: Activation,
    has_bias: bool,
    policy: &dyn DecomposePolicy,
) -> NodeId {
    // BERT-style post-norm transformer layer.
    // Inputs (with bias, 14 entries):
    //   0 hidden, 1 qkv_w, 2 qkv_b, 3 out_w, 4 out_b,
    //   5 ln1_g, 6 ln1_b, 7 fc1_w, 8 fc1_b,
    //   9 fc2_w, 10 fc2_b, 11 ln2_g, 12 ln2_b, 13 mask
    // Without bias (8 entries): hidden, qkv_w, out_w, ln1_g, fc1_w,
    //   fc2_w, ln2_g, mask
    let dtype = out_shape.dtype();
    let h_shape = src_graph.node(orig_inputs[0]).shape.dims();
    let batch = h_shape[0].unwrap_static();
    let seq = h_shape[1].unwrap_static();

    let (
        hidden,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        ln1_g,
        ln1_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        ln2_g,
        ln2_b,
        mask,
    ) = if has_bias {
        (
            inputs[0],
            inputs[1],
            Some(inputs[2]),
            inputs[3],
            Some(inputs[4]),
            inputs[5],
            Some(inputs[6]),
            inputs[7],
            Some(inputs[8]),
            inputs[9],
            Some(inputs[10]),
            inputs[11],
            Some(inputs[12]),
            inputs[13],
        )
    } else {
        // For no-bias case, the 0 NodeId is a placeholder; layer_norm
        // requires a beta input — we synthesize a zero constant.
        let zero = make_zero_const(out, &[h_shape[2].unwrap_static()], dtype);
        (
            inputs[0],
            inputs[1],
            None,
            inputs[2],
            None,
            inputs[3],
            Some(zero),
            inputs[4],
            None,
            inputs[5],
            None,
            inputs[6],
            Some(zero),
            inputs[7],
        )
    };

    // 1. attention block.
    let h_dim = h_shape[2].unwrap_static();
    let inner = num_heads * head_dim;
    let proj_shape = Shape::new(&[batch, seq, inner], dtype);
    let qkv_shape = Shape::new(&[batch, seq, 3 * inner], dtype);

    // QKV projection — fold bias into the matmul epilogue when the backend
    // lowers FusedMatMulBiasAct natively.
    let qkv = match qkv_b {
        Some(b) => {
            if policy.fold_matmul_bias_act() {
                out.add_node(
                    Op::FusedMatMulBiasAct { activation: None },
                    vec![hidden, qkv_w, b],
                    qkv_shape.clone(),
                )
            } else {
                let qkv = out.matmul(hidden, qkv_w, qkv_shape.clone());
                out.binary(BinaryOp::Add, qkv, b, qkv_shape.clone())
            }
        }
        None => out.matmul(hidden, qkv_w, qkv_shape.clone()),
    };
    let q = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );
    let k = out.add_node(
        Op::Narrow {
            axis: 2,
            start: inner,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );
    let v = out.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * inner,
            len: inner,
        },
        vec![qkv],
        proj_shape.clone(),
    );

    // FTL is BERT-style — no Rope. When the Attention kernel accepts rank-3
    // [B, S, H·D] directly (via per-axis strides) we skip the reshape+
    // transpose to [B, H, S, D] AND the inverse on the output; otherwise
    // materialize the rank-4 layout.
    let attn = if policy.attention_accepts_rank3() {
        out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: rlx_ir::op::MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, mask],
            proj_shape.clone(),
        )
    } else {
        let bhsd = Shape::new(&[batch, num_heads, seq, head_dim], dtype);
        let to_bhsd = |out: &mut Graph, t: NodeId| -> NodeId {
            let r = out.reshape(
                t,
                vec![batch as i64, seq as i64, num_heads as i64, head_dim as i64],
                Shape::new(&[batch, seq, num_heads, head_dim], dtype),
            );
            out.add_node(
                Op::Transpose {
                    perm: vec![0, 2, 1, 3],
                },
                vec![r],
                bhsd.clone(),
            )
        };
        let q = to_bhsd(out, q);
        let k = to_bhsd(out, k);
        let v = to_bhsd(out, v);
        let attn_4d = out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind: rlx_ir::op::MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, mask],
            bhsd.clone(),
        );
        let attn_bsd = out.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![attn_4d],
            Shape::new(&[batch, seq, num_heads, head_dim], dtype),
        );
        out.reshape(
            attn_bsd,
            vec![batch as i64, seq as i64, inner as i64],
            proj_shape.clone(),
        )
    };

    // Out projection, same bias-fold policy.
    let attn_out = match out_b {
        Some(b) => {
            if policy.fold_matmul_bias_act() {
                out.add_node(
                    Op::FusedMatMulBiasAct { activation: None },
                    vec![attn, out_w, b],
                    out_shape.clone(),
                )
            } else {
                let attn_out = out.matmul(attn, out_w, out_shape.clone());
                out.binary(BinaryOp::Add, attn_out, b, out_shape.clone())
            }
        }
        None => out.matmul(attn, out_w, out_shape.clone()),
    };

    // 2. residual + LayerNorm 1.
    let h1 = if policy.fold_residual_ln() {
        out.add_node(
            Op::FusedResidualLN {
                has_bias: false,
                eps: eps1,
            },
            vec![attn_out, hidden, ln1_g, ln1_b.unwrap()],
            out_shape.clone(),
        )
    } else {
        let pre1 = out.binary(BinaryOp::Add, hidden, attn_out, out_shape.clone());
        out.layer_norm(pre1, ln1_g, ln1_b.unwrap(), -1, eps1, out_shape.clone())
    };

    // 3. FFN: act(h1 @ fc1_w + fc1_b) @ fc2_w + fc2_b.
    // Derive intermediate dim from fc1_w shape (which is [in, intermediate]).
    let fc1_w_shape = src_graph
        .node(if has_bias {
            orig_inputs[7]
        } else {
            orig_inputs[4]
        })
        .shape
        .dims();
    let inter_dim = fc1_w_shape[1].unwrap_static();
    let inter_shape = Shape::new(&[batch, seq, inter_dim], dtype);
    // Fold matmul + bias + activation into one dispatch when bias is present
    // and the backend lowers it; the activation rides the matmul epilogue.
    let fc1_act = match fc1_b {
        Some(b) => {
            if policy.fold_matmul_bias_act() {
                out.add_node(
                    Op::FusedMatMulBiasAct {
                        activation: Some(activation),
                    },
                    vec![h1, fc1_w, b],
                    inter_shape.clone(),
                )
            } else {
                let fc1_out = out.matmul(h1, fc1_w, inter_shape.clone());
                let fc1_out = out.binary(BinaryOp::Add, fc1_out, b, inter_shape.clone());
                out.activation(activation, fc1_out, inter_shape.clone())
            }
        }
        None => {
            let fc1_out = out.matmul(h1, fc1_w, inter_shape.clone());
            out.activation(activation, fc1_out, inter_shape.clone())
        }
    };
    let ffn_out = match fc2_b {
        Some(b) => {
            if policy.fold_matmul_bias_act() {
                out.add_node(
                    Op::FusedMatMulBiasAct { activation: None },
                    vec![fc1_act, fc2_w, b],
                    out_shape.clone(),
                )
            } else {
                let fc2_out = out.matmul(fc1_act, fc2_w, out_shape.clone());
                out.binary(BinaryOp::Add, fc2_out, b, out_shape.clone())
            }
        }
        None => out.matmul(fc1_act, fc2_w, out_shape.clone()),
    };

    // 4. residual + LayerNorm 2.
    let _ = h_dim;
    if policy.fold_residual_ln() {
        out.add_node(
            Op::FusedResidualLN {
                has_bias: false,
                eps: eps2,
            },
            vec![ffn_out, h1, ln2_g, ln2_b.unwrap()],
            out_shape.clone(),
        )
    } else {
        let pre2 = out.binary(BinaryOp::Add, h1, ffn_out, out_shape.clone());
        out.layer_norm(pre2, ln2_g, ln2_b.unwrap(), -1, eps2, out_shape.clone())
    }
}

/// Broadcast every input of an element-wise op to `target_shape`,
/// using `broadcast_to` per input. The Binary / Compare / Where
/// kernels expect strict shape match across operands; other backends
/// auto-broadcast inside the op.
fn broadcast_inputs(out: &mut Graph, inputs: &[NodeId], target: &Shape) -> Vec<NodeId> {
    inputs
        .iter()
        .map(|&id| broadcast_to(out, id, target))
        .collect()
}

/// Broadcast `src` (the new-graph NodeId of a tensor) to `target_shape`.
/// If src already matches target, returns src unchanged. Otherwise:
///   1. Reshape src to match target's rank by left-padding with 1s.
///   2. Expand the rank-matched intermediate to target_shape.
///
/// Used to attach a `[N]` bias to a `[..., N]` activation in backends
/// which have no implicit broadcasting in Binary.
fn broadcast_to(out: &mut Graph, src: NodeId, target: &Shape) -> NodeId {
    let src_dims_dim = out.node(src).shape.dims().to_vec();
    let target_dims: Vec<usize> = target.dims().iter().map(|d| d.unwrap_static()).collect();
    let src_dims: Vec<usize> = src_dims_dim.iter().map(|d| d.unwrap_static()).collect();
    if src_dims == target_dims {
        return src;
    }

    let dtype = target.dtype();
    let target_rank = target_dims.len();
    let src_rank = src_dims.len();
    debug_assert!(
        src_rank <= target_rank,
        "broadcast_to: src rank exceeds target"
    );

    // Left-pad with 1s so src has the same rank as target.
    let padded: Vec<usize> = std::iter::repeat_n(1usize, target_rank - src_rank)
        .chain(src_dims.iter().copied())
        .collect();
    let reshaped = if padded.len() == src_rank {
        src
    } else {
        let new_shape_dims: Vec<i64> = padded.iter().map(|&d| d as i64).collect();
        out.reshape(src, new_shape_dims, Shape::new(&padded, dtype))
    };

    if padded == target_dims {
        return reshaped;
    }

    let target_i64: Vec<i64> = target_dims.iter().map(|&d| d as i64).collect();
    out.add_node(
        Op::Expand {
            target_shape: target_i64,
        },
        vec![reshaped],
        target.clone(),
    )
}

/// Lower DotGeneral to (Transpose + Reshape +) batched MatMul + Reshape.
///
/// Algorithm:
///   • LHS  → permute to `[batch..., outer..., contracting...]`, flatten to `[B, M, K]`
///   • RHS  → permute to `[batch..., contracting..., outer...]`, flatten to `[B, K, N]`
///   • MatMul (true batched when B > 1, plain 2D when B = 1)
///   • Reshape result back to `[batch_dims, lhs_outer..., rhs_outer...]`
///
/// Single-axis contracting per side; multi-axis contracting is handled
/// implicitly by the flatten step (K = product of all contracting sizes).
#[allow(clippy::too_many_arguments)]
fn expand_dot_general(
    out: &mut Graph,
    src: &Graph,
    orig_lhs: NodeId,
    orig_rhs: NodeId,
    inputs: &[NodeId],
    out_shape: &Shape,
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> NodeId {
    let dtype = out_shape.dtype();
    let lhs_dims: Vec<usize> = src
        .node(orig_lhs)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let rhs_dims: Vec<usize> = src
        .node(orig_rhs)
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();

    assert_eq!(
        lhs_batch.len(),
        rhs_batch.len(),
        "DotGeneral: lhs_batch and rhs_batch lengths must match"
    );
    assert_eq!(
        lhs_contracting.len(),
        rhs_contracting.len(),
        "DotGeneral: lhs and rhs contracting lengths must match"
    );

    // Categorize axes.
    let lhs_outer: Vec<usize> = (0..lhs_dims.len())
        .filter(|i| !lhs_contracting.contains(i) && !lhs_batch.contains(i))
        .collect();
    let rhs_outer: Vec<usize> = (0..rhs_dims.len())
        .filter(|i| !rhs_contracting.contains(i) && !rhs_batch.contains(i))
        .collect();

    // Build perms.
    let lhs_perm: Vec<usize> = lhs_batch
        .iter()
        .chain(lhs_outer.iter())
        .chain(lhs_contracting.iter())
        .copied()
        .collect();
    let rhs_perm: Vec<usize> = rhs_batch
        .iter()
        .chain(rhs_contracting.iter())
        .chain(rhs_outer.iter())
        .copied()
        .collect();

    let permute_if_needed =
        |out: &mut Graph, x: NodeId, dims: &[usize], perm: &[usize]| -> NodeId {
            let identity: Vec<usize> = (0..dims.len()).collect();
            if perm == identity.as_slice() {
                return x;
            }
            let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
            out.add_node(
                Op::Transpose {
                    perm: perm.to_vec(),
                },
                vec![x],
                Shape::new(&new_dims, dtype),
            )
        };

    let lhs_t = permute_if_needed(out, inputs[0], &lhs_dims, &lhs_perm);
    let rhs_t = permute_if_needed(out, inputs[1], &rhs_dims, &rhs_perm);

    let b: usize = lhs_batch
        .iter()
        .map(|&i| lhs_dims[i])
        .product::<usize>()
        .max(1);
    let m: usize = lhs_outer
        .iter()
        .map(|&i| lhs_dims[i])
        .product::<usize>()
        .max(1);
    let k: usize = lhs_contracting
        .iter()
        .map(|&i| lhs_dims[i])
        .product::<usize>()
        .max(1);
    let n: usize = rhs_outer
        .iter()
        .map(|&i| rhs_dims[i])
        .product::<usize>()
        .max(1);

    let mm_node = if lhs_batch.is_empty() {
        // 2D × 2D path.
        let lhs_2d = if lhs_outer.len() == 1 && lhs_contracting.len() == 1 {
            lhs_t
        } else {
            out.reshape(lhs_t, vec![m as i64, k as i64], Shape::new(&[m, k], dtype))
        };
        let rhs_2d = if rhs_outer.len() == 1 && rhs_contracting.len() == 1 {
            rhs_t
        } else {
            out.reshape(rhs_t, vec![k as i64, n as i64], Shape::new(&[k, n], dtype))
        };
        out.matmul(lhs_2d, rhs_2d, Shape::new(&[m, n], dtype))
    } else {
        // Batched [B, M, K] × [B, K, N] → [B, M, N].
        let lhs_3d = out.reshape(
            lhs_t,
            vec![b as i64, m as i64, k as i64],
            Shape::new(&[b, m, k], dtype),
        );
        let rhs_3d = out.reshape(
            rhs_t,
            vec![b as i64, k as i64, n as i64],
            Shape::new(&[b, k, n], dtype),
        );
        out.matmul(lhs_3d, rhs_3d, Shape::new(&[b, m, n], dtype))
    };

    // Reshape result back to the declared output shape.
    let out_dims_i64: Vec<i64> = out_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let out_dims_usize: Vec<usize> = out_shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let canonical_dims: Vec<usize> = if lhs_batch.is_empty() {
        vec![m, n]
    } else {
        vec![b, m, n]
    };
    if out_dims_usize == canonical_dims {
        mm_node
    } else {
        out.reshape(mm_node, out_dims_i64, out_shape.clone())
    }
}

fn mask_dims_dtype(src: &Graph, id: NodeId) -> DType {
    src.node(id).shape.dtype()
}

/// Promote (or pass through) a rank-3 `[B, S, H·D]` Attention call.
///
/// When the backend's kernel is rank-4-only
/// (`policy.attention_accepts_rank3() == false`): reshape + transpose Q/K/V
/// into `[B, H, S, D]`, reshape+expand a `MaskKind::Custom` mask to
/// `[B, H, S_q, S_k]`, then transpose + reshape the output back.
///
/// When the kernel reads Q/K/V (and the mask) via per-axis strides
/// (`== true`): pass the rank-3 tensors straight through, forwarding a
/// `MaskKind::Custom | MaskKind::Bias` mask on its native strides — no
/// transposes, no explicit mask materialization for the common shapes.
#[allow(clippy::too_many_arguments)]
fn expand_attention_rank3(
    out: &mut Graph,
    src: &Graph,
    orig_inputs: &[NodeId],
    new_inputs: &[NodeId],
    out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    score_scale: Option<f32>,
    attn_logit_softcap: Option<f32>,
    policy: &dyn DecomposePolicy,
) -> NodeId {
    let dtype = out_shape.dtype();
    let q_dims: Vec<usize> = src
        .node(orig_inputs[0])
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let k_dims: Vec<usize> = src
        .node(orig_inputs[1])
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let batch = q_dims[0];
    let seq_q = q_dims[1];
    let seq_k = k_dims[1];

    if policy.attention_accepts_rank3() {
        // Rank-3 straight through. The kernel's per-axis stride params
        // (q/k/v/o batch_stride, head_stride, seq_stride) handle either
        // canonical [B, H, S, D] or the transpose-elided [B, S, H·D] layout
        // uniformly, so we skip the 3 input + 1 output Transpose dispatches.
        //
        // The mask is likewise forwarded on its native strides
        // (mask_batch_stride / mask_head_stride / seq_q_stride / seq_k_stride):
        // a [B, S], [B, S_q, S_k] or [B, H, S_q, S_k] mask needs no explicit
        // [B, H, S_q, S_k] expansion.
        let bsd_q = Shape::new(&[batch, seq_q, num_heads * head_dim], dtype);
        let bsd_k = Shape::new(&[batch, seq_k, num_heads * head_dim], dtype);

        let mut attn_inputs = vec![new_inputs[0], new_inputs[1], new_inputs[2]];
        if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
            let mask_id = new_inputs[3];
            let mask_dims: Vec<usize> = src
                .node(orig_inputs[3])
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let mask_b = match mask_dims.len() {
                // [B, S]: stride math (S, 0, 0, 1) handles broadcast.
                2 => mask_id,
                // [B, S_q, S_k]: stride math (S_q·S_k, 0, S_k, 1).
                3 => mask_id,
                // [B, H, S_q, S_k]: canonical layout, strides are full.
                4 => mask_id,
                _ => {
                    // Unusual shape — fall back to the explicit expand path
                    // that materializes a [B, H, S_q, S_k] tensor.
                    let _ = mask_dims; // unused fallback path
                    let reshaped = mask_id;
                    let target_i64: Vec<i64> =
                        vec![batch as i64, num_heads as i64, seq_q as i64, seq_k as i64];
                    out.add_node(
                        Op::Expand {
                            target_shape: target_i64,
                        },
                        vec![reshaped],
                        Shape::new(&[batch, num_heads, seq_q, seq_k], dtype),
                    )
                }
            };
            attn_inputs.push(mask_b);
        }
        let _ = bsd_k; // documentation aid
        let _ = bsd_q;

        // Attention output declared in the [B, S, H·D] layout matching the
        // inputs — saves the inverse transpose + reshape pair.
        out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            },
            attn_inputs,
            out_shape.clone(),
        )
    } else {
        // Rank-4-only kernel: reshape + transpose Q/K/V to [B, H, S, D].
        let to_bhsd = |out: &mut Graph, x: NodeId, seq: usize| -> NodeId {
            let r = out.reshape(
                x,
                vec![batch as i64, seq as i64, num_heads as i64, head_dim as i64],
                Shape::new(&[batch, seq, num_heads, head_dim], dtype),
            );
            out.add_node(
                Op::Transpose {
                    perm: vec![0, 2, 1, 3],
                },
                vec![r],
                Shape::new(&[batch, num_heads, seq, head_dim], dtype),
            )
        };

        let q4 = to_bhsd(out, new_inputs[0], seq_q);
        let k4 = to_bhsd(out, new_inputs[1], seq_k);
        let v4 = to_bhsd(out, new_inputs[2], seq_k);

        let bhsd = Shape::new(&[batch, num_heads, seq_q, head_dim], dtype);
        let mut attn_inputs = vec![q4, k4, v4];
        if matches!(mask_kind, MaskKind::Custom) {
            // BERT passes [B, S]; the kernel reads [B, H, S_q, S_k]. We reshape
            // the mask through Reshape + Expand to broadcast across heads and
            // queries (additive padding mask: same per (q, k) pair regardless
            // of which head or which query position it gates).
            let mask_id = new_inputs[3];
            let mask_dims: Vec<usize> = src
                .node(orig_inputs[3])
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let target = Shape::new(&[batch, num_heads, seq_q, seq_k], dtype);
            let target_dims = vec![batch, num_heads, seq_q, seq_k];
            let mask_b = if mask_dims == target_dims {
                mask_id
            } else {
                // Reshape [B, S] → [B, 1, 1, S] then expand.
                let padded_dims = match mask_dims.len() {
                    2 => vec![mask_dims[0], 1, 1, mask_dims[1]],
                    3 => vec![mask_dims[0], mask_dims[1], 1, mask_dims[2]],
                    _ => mask_dims.clone(),
                };
                let reshaped = if padded_dims.len() != mask_dims.len() {
                    let new_shape_i64: Vec<i64> = padded_dims.iter().map(|&d| d as i64).collect();
                    out.reshape(
                        mask_id,
                        new_shape_i64,
                        Shape::new(&padded_dims, mask_dims_dtype(src, orig_inputs[3])),
                    )
                } else {
                    mask_id
                };
                let target_i64: Vec<i64> = target_dims.iter().map(|&d| d as i64).collect();
                out.add_node(
                    Op::Expand {
                        target_shape: target_i64,
                    },
                    vec![reshaped],
                    target.clone(),
                )
            };
            attn_inputs.push(mask_b);
        }
        let attn_4d = out.add_node(
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            },
            attn_inputs,
            bhsd.clone(),
        );

        // Inverse: [B, H, S, D] → [B, S, H, D] → [B, S, H*D].
        let bsd = out.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![attn_4d],
            Shape::new(&[batch, seq_q, num_heads, head_dim], dtype),
        );
        out.reshape(
            bsd,
            vec![batch as i64, seq_q as i64, (num_heads * head_dim) as i64],
            out_shape.clone(),
        )
    }
}

/// Promote a rank-3 `[B, S, H·D]` AttentionBackward to rank-4 `[B, H, S, D]`
/// for a rank-4-only kernel: reshape + transpose Q/K/V **and dY** on the way
/// in, then transpose + reshape the emitted (dQ|dK|dV) back to the declared
/// rank-3 shape. Mirrors the rank-4 branch of [`expand_attention_rank3`]; the
/// one difference is that the output sequence length follows `wrt` (dQ has
/// `seq_q`; dK/dV have `seq_k`).
#[allow(clippy::too_many_arguments)]
fn expand_attention_backward_rank3(
    out: &mut Graph,
    src: &Graph,
    orig_inputs: &[NodeId],
    new_inputs: &[NodeId],
    out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
    wrt: AttentionBwdWrt,
) -> NodeId {
    let dtype = out_shape.dtype();
    let q_dims: Vec<usize> = src
        .node(orig_inputs[0])
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let k_dims: Vec<usize> = src
        .node(orig_inputs[1])
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let batch = q_dims[0];
    let seq_q = q_dims[1];
    let seq_k = k_dims[1];

    let to_bhsd = |out: &mut Graph, x: NodeId, seq: usize| -> NodeId {
        let r = out.reshape(
            x,
            vec![batch as i64, seq as i64, num_heads as i64, head_dim as i64],
            Shape::new(&[batch, seq, num_heads, head_dim], dtype),
        );
        out.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![r],
            Shape::new(&[batch, num_heads, seq, head_dim], dtype),
        )
    };

    // dY is the gradient of the forward output [B, seq_q, H*D].
    let q4 = to_bhsd(out, new_inputs[0], seq_q);
    let k4 = to_bhsd(out, new_inputs[1], seq_k);
    let v4 = to_bhsd(out, new_inputs[2], seq_k);
    let dy4 = to_bhsd(out, new_inputs[3], seq_q);

    let mut bwd_inputs = vec![q4, k4, v4, dy4];
    if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
        // Mask is input[4]; broadcast to [B, H, S_q, S_k] as the forward does.
        let mask_id = new_inputs[4];
        let mask_dims: Vec<usize> = src
            .node(orig_inputs[4])
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let target_dims = vec![batch, num_heads, seq_q, seq_k];
        let target = Shape::new(&[batch, num_heads, seq_q, seq_k], dtype);
        let mask_b = if mask_dims == target_dims {
            mask_id
        } else {
            let padded_dims = match mask_dims.len() {
                2 => vec![mask_dims[0], 1, 1, mask_dims[1]],
                3 => vec![mask_dims[0], mask_dims[1], 1, mask_dims[2]],
                _ => mask_dims.clone(),
            };
            let reshaped = if padded_dims.len() != mask_dims.len() {
                let ns: Vec<i64> = padded_dims.iter().map(|&d| d as i64).collect();
                out.reshape(
                    mask_id,
                    ns,
                    Shape::new(&padded_dims, mask_dims_dtype(src, orig_inputs[4])),
                )
            } else {
                mask_id
            };
            let ti: Vec<i64> = target_dims.iter().map(|&d| d as i64).collect();
            out.add_node(
                Op::Expand { target_shape: ti },
                vec![reshaped],
                target.clone(),
            )
        };
        bwd_inputs.push(mask_b);
    }

    // dQ has seq_q rows; dK/dV have seq_k rows.
    let seq_wrt = match wrt {
        AttentionBwdWrt::Query => seq_q,
        AttentionBwdWrt::Key | AttentionBwdWrt::Value => seq_k,
    };
    let bhsd = Shape::new(&[batch, num_heads, seq_wrt, head_dim], dtype);
    let grad4 = out.add_node(
        Op::AttentionBackward {
            num_heads,
            head_dim,
            mask_kind,
            wrt,
        },
        bwd_inputs,
        bhsd,
    );

    // Inverse: [B, H, S_wrt, D] → [B, S_wrt, H, D] → [B, S_wrt, H*D].
    let bsd = out.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![grad4],
        Shape::new(&[batch, seq_wrt, num_heads, head_dim], dtype),
    );
    out.reshape(
        bsd,
        vec![batch as i64, seq_wrt as i64, (num_heads * head_dim) as i64],
        out_shape.clone(),
    )
}

/// Inline a sub-graph into `out`, binding the sub-graph's `Op::Input`
/// nodes positionally to `captures` (in the order they appear in the
/// sub-graph). `Op::Param` nodes look up by name — if the parent
/// already has a Param with that name we reuse it; otherwise a fresh
/// Param is added (the eventual user of the compiled graph still needs
/// to call set_param for it). `Op::Constant` nodes are cloned inline.
///
/// Returns the new NodeIds (in `out`) of the sub-graph's outputs.
fn inline_subgraph(out: &mut Graph, subgraph: &Graph, captures: &[NodeId]) -> Vec<NodeId> {
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut input_idx: usize = 0;
    for sub_node in subgraph.nodes() {
        let new_id = match &sub_node.op {
            Op::Input { .. } => {
                // Positional capture binding (matches the MLX convention).
                let cap = captures.get(input_idx).copied().unwrap_or_else(|| {
                    panic!(
                        "inline_subgraph: subgraph has more Op::Input nodes \
                         than captures provided ({} > {})",
                        input_idx + 1,
                        captures.len()
                    )
                });
                input_idx += 1;
                cap
            }
            Op::Param { name } => {
                // Try to find an existing Param in `out` with the same name.
                let existing = out.nodes().iter().find_map(|n| match &n.op {
                    Op::Param { name: n2 } if n2 == name => Some(n.id),
                    _ => None,
                });
                match existing {
                    Some(id) => id,
                    None => out.param(name.clone(), sub_node.shape.clone()),
                }
            }
            other => {
                let new_inputs: Vec<NodeId> =
                    sub_node.inputs.iter().map(|&id| id_map[&id]).collect();
                out.add_node(other.clone(), new_inputs, sub_node.shape.clone())
            }
        };
        id_map.insert(sub_node.id, new_id);
    }
    subgraph.outputs.iter().map(|&id| id_map[&id]).collect()
}

/// Expand `Op::If`: inline both branches against the captures, then
/// combine via Where(predicate, then_out, else_out).
fn expand_if(
    out: &mut Graph,
    inputs: &[NodeId],
    then_branch: &Graph,
    else_branch: &Graph,
    out_shape: &Shape,
) -> NodeId {
    if inputs.is_empty() {
        panic!("expand_if: missing predicate input");
    }
    let pred = inputs[0];
    let captures = &inputs[1..];
    let then_outs = inline_subgraph(out, then_branch, captures);
    let else_outs = inline_subgraph(out, else_branch, captures);
    if then_outs.len() != 1 || else_outs.len() != 1 {
        panic!(
            "expand_if: each branch must produce exactly 1 output \
                (then={}, else={})",
            then_outs.len(),
            else_outs.len()
        );
    }
    out.add_node(
        Op::Where,
        vec![pred, then_outs[0], else_outs[0]],
        out_shape.clone(),
    )
}

/// Expand `Op::While`: bounded unroll, gating updates with
/// Where(active && cond, body_out, carried) so that once `cond` flips
/// false the carried value freezes. Requires `max_iterations` —
/// without a static bound the unroll has no terminating count.
fn expand_while(
    out: &mut Graph,
    inputs: &[NodeId],
    cond: &Graph,
    body: &Graph,
    max_iterations: Option<usize>,
    out_shape: &Shape,
) -> NodeId {
    let max_iter = max_iterations.unwrap_or_else(|| {
        panic!(
            "expand_while: max_iterations is required \
                — this backend has no runtime loop primitive"
        )
    });
    if inputs.is_empty() {
        panic!("expand_while: at least one loop-carried value required");
    }

    // Active mask starts at all-ones, same shape as the carried value.
    // We use a Constant (f32 1.0) broadcast to the carried shape.
    let mut carried: Vec<NodeId> = inputs.to_vec();
    let active_shape = out.node(carried[0]).shape.clone();
    let n_elems: usize = active_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .product();
    let ones = vec![1.0f32; n_elems];
    let ones_bytes: Vec<u8> = ones.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut active = out.add_node(
        Op::Constant { data: ones_bytes },
        vec![],
        active_shape.clone(),
    );

    for _ in 0..max_iter {
        let cond_outs = inline_subgraph(out, cond, &carried);
        if cond_outs.len() != 1 {
            panic!(
                "expand_while: cond sub-graph must produce 1 output \
                    (got {})",
                cond_outs.len()
            );
        }
        let cond_f = cond_outs[0];
        // active *= cond_f (cond's output dtype should already be f32 0.0/1.0
        // in our f32-uniform arena where Bool is stored as f32).
        let cond_b = broadcast_to(out, cond_f, &active_shape);
        active = out.binary(BinaryOp::Mul, active, cond_b, active_shape.clone());

        let body_outs = inline_subgraph(out, body, &carried);
        if body_outs.len() != carried.len() {
            panic!(
                "expand_while: body produced {} outputs but {} \
                    loop-carried values were expected",
                body_outs.len(),
                carried.len()
            );
        }
        let mut next: Vec<NodeId> = Vec::with_capacity(carried.len());
        for (b_out, c_in) in body_outs.into_iter().zip(carried.iter()) {
            let n = out.add_node(
                Op::Where,
                vec![active, b_out, *c_in],
                out.node(*c_in).shape.clone(),
            );
            next.push(n);
        }
        carried = next;
    }

    // Single-output convention: return carried[0]. If the declared
    // output shape differs (e.g. caller wired through a Reshape), do a
    // final Reshape to match.
    let final_id = carried[0];
    let final_shape = out.node(final_id).shape.clone();
    let want_dims: Vec<usize> = out_shape.dims().iter().map(|d| d.unwrap_static()).collect();
    let have_dims: Vec<usize> = final_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    if want_dims == have_dims {
        final_id
    } else {
        let want_i64: Vec<i64> = want_dims.iter().map(|&d| d as i64).collect();
        out.reshape(final_id, want_i64, out_shape.clone())
    }
}

/// Allocate a zero Constant of the given shape (f32-uniform arena).
fn make_zero_const(out: &mut Graph, dims: &[usize], dtype: DType) -> NodeId {
    let n: usize = dims.iter().product();
    // F32-only constants in the GPU-family backends (f32-uniform arena);
    // any other dtype here means an upstream graph error we'd want
    // to surface explicitly — for now coerce.
    let _ = dtype;
    let bytes = vec![0u8; n * 4];
    out.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(dims, DType::F32),
    )
}
