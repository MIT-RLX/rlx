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

//! IR-level "unfusion" pass for the CUDA backend.
//!
//! Logic ported verbatim from `rlx-wgpu`'s unfuse: composed ops
//! (FusedSwiGLU, LoraMatMul, FusedAttentionBlock, FusedTransformerLayer,
//! DotGeneral, If, While) decompose into primitive sequences, and
//! Binary/Compare/Where get a broadcast prologue when their input
//! shapes mismatch. CUDA's matmul kernel handles bias + activation in
//! its epilogue, and `fused_residual_ln.cu` does Add+LN in one kernel,
//! so those two stay native (not unfused).

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

pub fn unfuse(graph: Graph) -> Graph {
    // Skip rebuild only if no fused/composed ops AND every Binary /
    // Compare / Where already has matching input element counts. The
    // wgpu element-wise kernels are strict-shape-matched; broadcast
    // prologues get inserted during the rewrite.
    let needs_rewrite = graph.nodes().iter().any(|n| {
        should_unfuse(&n.op)
            || needs_broadcast_prologue(&graph, n)
            || needs_attn_rank3_promotion(&graph, n)
    });
    if !needs_rewrite {
        return graph;
    }

    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|&id| id_map[&id]).collect();

        let new_id = match &node.op {
            Op::FusedSwiGLU { cast_to: _, .. } => {
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
            } => expand_fab(
                &mut out,
                &graph,
                &node.inputs,
                &new_inputs,
                &node.shape,
                *num_heads,
                *head_dim,
                *has_bias,
                *has_rope,
            ),
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
            // [B, H, S, D] via Reshape + Transpose; the wgpu Attention
            // kernel expects rank-4. Output gets transposed + reshaped
            // back to the declared shape.
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
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
                    )
                } else {
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                }
            }
            // Insert per-axis broadcast prologue for Binary / Compare / Where
            // when input shapes differ — the wgpu element-wise kernels are
            // strict-shape-matched. Other backends auto-broadcast in the
            // op itself.
            Op::Binary(_) | Op::Compare(_) | Op::Where => {
                let broadcasted = broadcast_inputs(&mut out, &new_inputs, &node.shape);
                out.add_node(node.op.clone(), broadcasted, node.shape.clone())
            }
            // Pass through everything else.
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }

    out.set_outputs(graph.outputs.iter().map(|&id| id_map[&id]).collect());
    out
}

/// True if `node` is an element-wise op whose inputs don't all share
/// the same element count — i.e. wgpu's strict-shape kernel will reject
/// it and we need to insert a broadcast prologue.
fn needs_broadcast_prologue(graph: &Graph, node: &rlx_ir::Node) -> bool {
    let is_elt = matches!(node.op, Op::Binary(_) | Op::Compare(_) | Op::Where);
    if !is_elt {
        return false;
    }
    let target_n = node.shape.num_elements().unwrap_or(0);
    node.inputs
        .iter()
        .any(|&id| graph.node(id).shape.num_elements().unwrap_or(0) != target_n)
}

fn should_unfuse(op: &Op) -> bool {
    // FusedMatMulBiasAct and FusedResidualLN are now lowered natively
    // — the matmul kernel folds bias + activation into its epilogue,
    // and `fused_residual_ln.wgsl` does (Add[+bias] + LayerNorm) in
    // one pass.
    matches!(
        op,
        Op::FusedSwiGLU { .. }
            | Op::LoraMatMul { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::DotGeneral { .. }
            | Op::If { .. }
            | Op::While { .. }
    )
}

/// True if the node is a rank-3 Op::Attention — those need to be
/// reshaped + transposed before our rank-4-only kernel can take them.
fn needs_attn_rank3_promotion(graph: &Graph, node: &rlx_ir::Node) -> bool {
    matches!(node.op, Op::Attention { .. }) && graph.node(node.inputs[0]).shape.rank() == 3
}

// ── Expansions ───────────────────────────────────────────────────

#[allow(dead_code)] // FusedMatMulBiasAct is lowered natively in CUDA's matmul.
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

#[allow(dead_code)] // FusedResidualLN is lowered natively in CUDA.
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

    let qkv = out.matmul(inputs[0], inputs[1], qkv_shape.clone());
    let qkv = if has_bias {
        out.binary(BinaryOp::Add, qkv, inputs[qkv_b_idx], qkv_shape.clone())
    } else {
        qkv
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
            Op::Rope { head_dim, n_rot: head_dim },
            vec![q4, inputs[cos_idx], inputs[sin_idx]],
            bhsd_shape.clone(),
        );
        k4 = out.add_node(
            Op::Rope { head_dim, n_rot: head_dim },
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
    let attn = out.reshape(
        bsd,
        vec![batch as i64, seq as i64, inner as i64],
        proj_shape.clone(),
    );

    let out_proj = out.matmul(attn, inputs[2], out_shape.clone());
    if has_bias {
        out.binary(
            BinaryOp::Add,
            out_proj,
            inputs[out_b_idx],
            out_shape.clone(),
        )
    } else {
        out_proj
    }
}

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

    // 1. attention block via FAB expansion.
    let h_dim = h_shape[2].unwrap_static();
    let proj_shape = Shape::new(&[batch, seq, num_heads * head_dim], dtype);
    let bhsd = Shape::new(&[batch, num_heads, seq, head_dim], dtype);

    let qkv = out.matmul(
        hidden,
        qkv_w,
        Shape::new(&[batch, seq, 3 * num_heads * head_dim], dtype),
    );
    let qkv = match qkv_b {
        Some(b) => out.binary(
            BinaryOp::Add,
            qkv,
            b,
            Shape::new(&[batch, seq, 3 * num_heads * head_dim], dtype),
        ),
        None => qkv,
    };
    let inner = num_heads * head_dim;
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
    let attn = out.reshape(
        attn_bsd,
        vec![batch as i64, seq as i64, inner as i64],
        proj_shape.clone(),
    );
    let attn_out = out.matmul(attn, out_w, out_shape.clone());
    let attn_out = match out_b {
        Some(b) => out.binary(BinaryOp::Add, attn_out, b, out_shape.clone()),
        None => attn_out,
    };

    // 2. residual + LayerNorm 1.
    let pre1 = out.binary(BinaryOp::Add, hidden, attn_out, out_shape.clone());
    let h1 = out.layer_norm(pre1, ln1_g, ln1_b.unwrap(), -1, eps1, out_shape.clone());

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
    let fc1_out = out.matmul(h1, fc1_w, inter_shape.clone());
    let fc1_out = match fc1_b {
        Some(b) => out.binary(BinaryOp::Add, fc1_out, b, inter_shape.clone()),
        None => fc1_out,
    };
    let fc1_act = out.activation(activation, fc1_out, inter_shape.clone());
    let fc2_out = out.matmul(fc1_act, fc2_w, out_shape.clone());
    let ffn_out = match fc2_b {
        Some(b) => out.binary(BinaryOp::Add, fc2_out, b, out_shape.clone()),
        None => fc2_out,
    };

    // 4. residual + LayerNorm 2.
    let pre2 = out.binary(BinaryOp::Add, h1, ffn_out, out_shape.clone());
    let _ = h_dim;
    out.layer_norm(pre2, ln2_g, ln2_b.unwrap(), -1, eps2, out_shape.clone())
}

/// Broadcast every input of an element-wise op to `target_shape`,
/// using `broadcast_to` per input. The wgpu Binary / Compare / Where
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
/// Used by FMB / residual-LN unfusion to attach a `[N]` bias to a
/// `[..., N]` activation in the wgpu backend (which has no implicit
/// broadcasting in Binary).
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

/// Promote a rank-3 [B, S, H*D] Attention call to rank-4 [B, H, S, D]
/// for the wgpu kernel: reshape + transpose Q/K/V on the way in, then
/// transpose + reshape the output back to the declared shape.
fn expand_attention_rank3(
    out: &mut Graph,
    src: &Graph,
    orig_inputs: &[NodeId],
    new_inputs: &[NodeId],
    out_shape: &Shape,
    num_heads: usize,
    head_dim: usize,
    mask_kind: MaskKind,
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
                        "rlx-wgpu inline_subgraph: subgraph has more Op::Input nodes \
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
        panic!("rlx-wgpu expand_if: missing predicate input");
    }
    let pred = inputs[0];
    let captures = &inputs[1..];
    let then_outs = inline_subgraph(out, then_branch, captures);
    let else_outs = inline_subgraph(out, else_branch, captures);
    if then_outs.len() != 1 || else_outs.len() != 1 {
        panic!(
            "rlx-wgpu expand_if: each branch must produce exactly 1 output \
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
            "rlx-wgpu expand_while: max_iterations is required \
                — wgpu has no runtime loop primitive"
        )
    });
    if inputs.is_empty() {
        panic!("rlx-wgpu expand_while: at least one loop-carried value required");
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
                "rlx-wgpu expand_while: cond sub-graph must produce 1 output \
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
                "rlx-wgpu expand_while: body produced {} outputs but {} \
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
    // F32-only constants in the wgpu backend (arena is f32-uniform);
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
