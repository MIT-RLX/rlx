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

//! Fusion passes — pattern-match and replace subgraphs with fused ops.
//!
//! Each pass scans the graph in reverse topological order, looking for
//! specific multi-node patterns and replacing them with single fused nodes.
//! These are the same fusions we hand-coded in burnembed's ndarray_fused.rs.

use rlx_ir::op::*;
use rlx_ir::*;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

mod ada_layer_norm;
mod attention_block;
mod conv_bias_act;
mod gated_residual;
mod mark_elementwise;
mod matmul_bias_act;
mod residual_ln;
mod residual_rmsnorm;
mod rmsnorm_reshape;
mod shared_input_matmul;
mod swiglu;
mod swiglu_dual;
mod transformer_layer;
mod unfuse_elementwise;

pub use ada_layer_norm::*;
pub use attention_block::*;
pub use conv_bias_act::*;
pub use gated_residual::*;
pub use mark_elementwise::*;
pub use matmul_bias_act::*;
pub use residual_ln::*;
pub use residual_rmsnorm::*;
pub use rmsnorm_reshape::*;
pub use shared_input_matmul::*;
pub use swiglu::*;
pub use swiglu_dual::*;
pub use transformer_layer::*;
pub use unfuse_elementwise::*;

/// Activations that may be folded into `FusedMatMulBiasAct` epilogues.
fn fusible_mm_bias_epilogue_activation(act: Activation) -> bool {
    matches!(act, Activation::Gelu | Activation::Silu)
}

fn leading_flatten_shape(in_shape: &Shape, new_shape: &[i64]) -> Option<Shape> {
    rlx_ir::shape::leading_flatten_shape(in_shape, new_shape)
}

fn sole_consumer(graph: &Graph, id: NodeId) -> Option<NodeId> {
    graph
        .nodes()
        .iter()
        .find(|n| n.inputs.contains(&id))
        .map(|n| n.id)
}

/// Match a single producer node id that produces a tensor consumed by `narrow`.
fn narrow_parent(node: &Node) -> Option<(NodeId, usize, usize, usize)> {
    match &node.op {
        Op::Narrow { axis, start, len } => Some((node.inputs[0], *axis, *start, *len)),
        _ => None,
    }
}

/// Match `FusedMatMulBiasAct{activation: None}` and return its (input, weight, bias) tuple.
fn fused_mm_bias_none(node: &Node) -> Option<(NodeId, NodeId, NodeId)> {
    if let Op::FusedMatMulBiasAct { activation: None } = &node.op
        && node.inputs.len() == 3
    {
        return Some((node.inputs[0], node.inputs[1], node.inputs[2]));
    }
    None
}

/// Match `FusedResidualLN { has_bias: false }` and return `(x, residual, gamma, beta, eps)`.
fn fused_residual_ln_no_bias(node: &Node) -> Option<(NodeId, NodeId, NodeId, NodeId, f32)> {
    if let Op::FusedResidualLN {
        has_bias: false,
        eps,
    } = &node.op
        && node.inputs.len() == 4
    {
        return Some((
            node.inputs[0],
            node.inputs[1],
            node.inputs[2],
            node.inputs[3],
            *eps,
        ));
    }
    None
}

/// Match `FusedMatMulBiasAct { activation: Some(a) }` and return `(input, weight, bias, activation)`.
fn fused_mm_bias_act(node: &Node) -> Option<(NodeId, NodeId, NodeId, Activation)> {
    if let Op::FusedMatMulBiasAct {
        activation: Some(a),
    } = &node.op
        && node.inputs.len() == 3
    {
        return Some((node.inputs[0], node.inputs[1], node.inputs[2], *a));
    }
    None
}

/// Match `FusedAttentionBlock { has_bias: true, has_rope: false }` BERT shape.
fn fused_attn_block_bert(
    node: &Node,
) -> Option<(usize, usize, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId)> {
    if let Op::FusedAttentionBlock {
        num_heads,
        head_dim,
        has_bias: true,
        has_rope: false,
    } = &node.op
        && node.inputs.len() == 6
    {
        // [hidden, qkv_w, out_w, mask, qkv_b, out_b]
        return Some((
            *num_heads,
            *head_dim,
            node.inputs[0],
            node.inputs[1],
            node.inputs[2],
            node.inputs[3],
            node.inputs[4],
            node.inputs[5],
        ));
    }
    None
}

/// Unfuse only `ElementwiseRegion` nodes that exceed [`crate::limits::FusionLimits`].
///
/// Run after [`MarkElementwiseRegions`] when marking may still produce
/// oversized chains (e.g. limits tightened per backend).
pub fn clip_elementwise_regions(graph: Graph, limits: crate::limits::FusionLimits) -> Graph {
    let oversize = |n: &rlx_ir::Node| -> bool {
        matches!(
            &n.op,
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                ..
            } if *num_inputs > limits.max_elementwise_inputs
                || chain.len() as u32 > limits.max_elementwise_steps
        )
    };
    if !graph.nodes().iter().any(oversize) {
        return graph;
    }

    let mut rw = Rewriter::new(&graph.name);
    for node in graph.nodes() {
        if !oversize(node) {
            rw.copy_node(node);
            continue;
        }

        let Op::ElementwiseRegion {
            chain,
            num_inputs: _,
            scalar_input_mask: _,
            input_modulus: _,
            prologue: _,
            prologue_input: _,
        } = &node.op
        else {
            unreachable!();
        };

        let region_inputs: Vec<NodeId> = node.inputs.iter().map(|id| rw.map(*id)).collect();
        let mut step_ids: Vec<NodeId> = Vec::with_capacity(chain.len());
        let region_shape = node.shape.clone();
        let region_dims: Vec<_> = region_shape.dims().to_vec();
        let mut step_dtypes: Vec<rlx_ir::DType> = Vec::with_capacity(chain.len());
        let region_dtype = region_shape.dtype();
        let dtype_of = |op: &ChainOperand,
                        ins: &[NodeId],
                        step_dt: &[rlx_ir::DType],
                        rw: &Rewriter|
         -> rlx_ir::DType {
            match *op {
                ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.dtype(),
                ChainOperand::Step(i) => step_dt[i as usize],
            }
        };
        let shape_of =
            |op: &ChainOperand, ins: &[NodeId], step_ids: &[NodeId], rw: &Rewriter| -> Shape {
                match *op {
                    ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.clone(),
                    ChainOperand::Step(i) => rw.new_graph.node(step_ids[i as usize]).shape.clone(),
                }
            };
        for step in chain {
            let resolve = |op: &ChainOperand| -> NodeId {
                match *op {
                    ChainOperand::Input(i) => region_inputs[i as usize],
                    ChainOperand::Step(i) => step_ids[i as usize],
                }
            };
            let (new_id, dt) = match step {
                ChainStep::Activation(a, src) => {
                    let s = resolve(src);
                    let dt = dtype_of(src, &region_inputs, &step_dtypes, &rw);
                    let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                    let dims: Vec<_> = src_shape.dims().to_vec();
                    let shape = Shape::from_dims(&dims, dt);
                    (
                        rw.new_graph.add_node(Op::Activation(*a), vec![s], shape),
                        dt,
                    )
                }
                ChainStep::Cast(to, src) => {
                    let s = resolve(src);
                    let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                    let dims: Vec<_> = src_shape.dims().to_vec();
                    let shape = Shape::from_dims(&dims, *to);
                    (
                        rw.new_graph.add_node(Op::Cast { to: *to }, vec![s], shape),
                        *to,
                    )
                }
                ChainStep::Binary(op, lhs, rhs) => {
                    let l = resolve(lhs);
                    let r = resolve(rhs);
                    let dt = dtype_of(lhs, &region_inputs, &step_dtypes, &rw);
                    let l_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                    let r_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                    let bcast = l_shape
                        .broadcast_with(&r_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
                    let dims: Vec<_> = bcast.dims().to_vec();
                    let shape = Shape::from_dims(&dims, dt);
                    (
                        rw.new_graph.add_node(Op::Binary(*op), vec![l, r], shape),
                        dt,
                    )
                }
                ChainStep::Compare(op, lhs, rhs) => {
                    let l = resolve(lhs);
                    let r = resolve(rhs);
                    let l_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                    let r_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                    let bcast = l_shape
                        .broadcast_with(&r_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
                    let dims: Vec<_> = bcast.dims().to_vec();
                    let shape = Shape::from_dims(&dims, rlx_ir::DType::U8);
                    (
                        rw.new_graph.add_node(Op::Compare(*op), vec![l, r], shape),
                        rlx_ir::DType::U8,
                    )
                }
                ChainStep::Where(cond, x, y) => {
                    let cn = resolve(cond);
                    let xn = resolve(x);
                    let yn = resolve(y);
                    let dt = dtype_of(x, &region_inputs, &step_dtypes, &rw);
                    let x_shape = shape_of(x, &region_inputs, &step_ids, &rw);
                    let y_shape = shape_of(y, &region_inputs, &step_ids, &rw);
                    let c_shape = shape_of(cond, &region_inputs, &step_ids, &rw);
                    let bcast_xy = x_shape
                        .broadcast_with(&y_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
                    let bcast = c_shape.broadcast_with(&bcast_xy).unwrap_or_else(|e| {
                        panic!("clip_elementwise_regions: cannot broadcast cond {c_shape:?} ⊗ {bcast_xy:?} for Where: {e}")
                    });
                    let dims: Vec<_> = bcast.dims().to_vec();
                    let shape = Shape::from_dims(&dims, dt);
                    (
                        rw.new_graph.add_node(Op::Where, vec![cn, xn, yn], shape),
                        dt,
                    )
                }
            };
            step_ids.push(new_id);
            step_dtypes.push(dt);
        }
        let _ = (region_dtype, region_dims);
        let last = *step_ids
            .last()
            .expect("oversize region has non-empty chain");
        rw.replace(node.id, last);
    }
    rw.finish(&graph.outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::FusionLimits;
    use crate::pass::{Pass, run_passes};

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn fuse_matmul_bias_gelu() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let w = g.param("w", f32_shape(&[384, 1536]));
        let b = g.param("b", f32_shape(&[1536]));
        let mm = g.matmul(x, w, f32_shape(&[4, 15, 1536]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[4, 15, 1536]));
        let out = g.activation(Activation::Gelu, add, f32_shape(&[4, 15, 1536]));
        g.set_outputs(vec![out]);

        assert_eq!(g.len(), 6); // input, w, b, mm, add, gelu

        let fused = FuseMatMulBiasAct.run(g);
        println!("{fused}");

        // Should be: input, w, b, fused_mm_bias_gelu
        assert_eq!(fused.len(), 4);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        ));
    }

    #[test]
    fn fuse_matmul_bias_no_act() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let w = g.param("w", f32_shape(&[384, 384]));
        let b = g.param("b", f32_shape(&[384]));
        let mm = g.matmul(x, w, f32_shape(&[4, 15, 384]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[4, 15, 384]));
        g.set_outputs(vec![add]);

        let fused = FuseMatMulBiasAct.run(g);
        assert_eq!(fused.len(), 4);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedMatMulBiasAct { activation: None }
        ));
    }

    #[test]
    fn fuse_matmul_bias_skips_unsupported_activation_epilogue() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[8, 1024]));
        let w = g.param("w", f32_shape(&[1024, 16]));
        let b = g.param("b", f32_shape(&[16]));
        let mm = g.matmul(x, w, f32_shape(&[8, 16]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[8, 16]));
        let exp = g.activation(Activation::Exp, add, f32_shape(&[8, 16]));
        g.set_outputs(vec![exp]);

        let fused = FuseMatMulBiasAct.run(g);
        // mm + bias fuse; Exp stays separate (qwen35 softplus pattern).
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(out_node.op, Op::Activation(Activation::Exp)));
        let add_node = fused.node(out_node.inputs[0]);
        assert!(matches!(
            add_node.op,
            Op::FusedMatMulBiasAct { activation: None }
        ));
    }

    #[test]
    fn fuse_matmul_bias_act_with_late_bias_param() {
        use rlx_ir::infer::GraphExt;

        let mut g = Graph::new("late_bias");
        let x = g.input("x", f32_shape(&[8, 16]));
        let w = g.param("w", f32_shape(&[16, 32]));
        let out = {
            let mm = g.mm(x, w);
            let b = g.param("b", f32_shape(&[32]));
            let biased = g.add(mm, b);
            g.gelu(biased)
        };
        g.set_outputs(vec![out]);

        let fused = FuseMatMulBiasAct.run(g);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. })),
            "bias param declared after matmul must still fuse:\n{fused}"
        );
    }

    #[test]
    fn swiglu_ffn_builder_fuses_end_to_end() {
        let mut g = Graph::new("swiglu_block");
        let x = g.input("x", f32_shape(&[4, 768]));
        let up_w = g.param("up", f32_shape(&[768, 2048]));
        let gate_w = g.param("gate", f32_shape(&[768, 2048]));
        let down_w = g.param("down", f32_shape(&[2048, 768]));
        let out = g.swiglu_ffn(x, up_w, gate_w, down_w);
        g.set_outputs(vec![out]);

        let g = FuseSharedInputMatMul.run(g);
        let g = FuseSwiGLU.run(g);
        assert!(
            g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. })),
            "swiglu_ffn builder should match FuseSwiGLU:\n{g}"
        );
    }

    #[test]
    fn fuse_swiglu_dual_matmul_gate_first() {
        use rlx_ir::infer::GraphExt;

        let mut g = Graph::new("qwen3_ffn");
        let x = g.input("x", f32_shape(&[4, 768]));
        let gate_w = g.param("gate", f32_shape(&[768, 2048]));
        let up_w = g.param("up", f32_shape(&[768, 2048]));
        let gate = g.mm(x, gate_w);
        let up = g.mm(x, up_w);
        let gate_act = g.silu(gate);
        let out = g.mul(gate_act, up);
        g.set_outputs(vec![out]);

        let fused = FuseSwiGLUDualMatmul.run(g);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. })),
            "gate-first dual matmul should fuse:\n{fused}"
        );
        assert!(
            fused.len() <= 6,
            "dual fusion should collapse to x + weights + concat + mm + fused_swiglu, got {} nodes",
            fused.len()
        );
    }

    #[test]
    fn fuse_shared_input_matmul_three_way_qkv() {
        let mut g = Graph::new("qkv");
        let x = g.input("x", f32_shape(&[8, 512]));
        let wq = g.param("wq", f32_shape(&[512, 128]));
        let wk = g.param("wk", f32_shape(&[512, 128]));
        let wv = g.param("wv", f32_shape(&[512, 128]));
        let q = g.matmul(x, wq, f32_shape(&[8, 128]));
        let k = g.matmul(x, wk, f32_shape(&[8, 128]));
        let v = g.matmul(x, wv, f32_shape(&[8, 128]));
        g.set_outputs(vec![q, k, v]);

        let fused = FuseSharedInputMatMul.run(g);
        assert_eq!(
            fused.len(),
            9,
            "x + 3 weights + concat + mm + 3 narrows = 9"
        );
        for &out in &fused.outputs {
            assert!(matches!(fused.node(out).op, Op::Narrow { .. }));
        }
    }

    #[test]
    fn fuse_residual_layer_norm() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let residual = g.input("residual", f32_shape(&[4, 15, 384]));
        let gamma = g.param("gamma", f32_shape(&[384]));
        let beta = g.param("beta", f32_shape(&[384]));
        let add = g.binary(BinaryOp::Add, x, residual, f32_shape(&[4, 15, 384]));
        let ln = g.layer_norm(add, gamma, beta, -1, 1e-12, f32_shape(&[4, 15, 384]));
        g.set_outputs(vec![ln]);

        assert_eq!(g.len(), 6); // x, residual, gamma, beta, add, ln

        let fused = FuseResidualLN.run(g);
        println!("{fused}");

        // Should be: x, residual, gamma, beta, fused_residual_ln
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedResidualLN {
                has_bias: false,
                ..
            }
        ));
    }

    #[test]
    fn fuse_residual_rms_norm() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let residual = g.input("residual", f32_shape(&[4, 15, 384]));
        let gamma = g.param("gamma", f32_shape(&[384]));
        let beta = g.param("beta", f32_shape(&[384]));
        let add = g.binary(BinaryOp::Add, x, residual, f32_shape(&[4, 15, 384]));
        let rn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![add, gamma, beta],
            f32_shape(&[4, 15, 384]),
        );
        g.set_outputs(vec![rn]);

        assert_eq!(g.len(), 6);

        let fused = FuseResidualRmsNorm.run(g);
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedResidualRmsNorm {
                has_bias: false,
                ..
            }
        ));
    }

    /// Qwen/Bonsai post-attn: `h+=attn; n=rms(h); h+=ffn(n)` — add dst stays
    /// live, so FuseResidualRmsNorm must refuse.
    #[test]
    fn fuse_residual_rms_norm_skips_live_reuse() {
        let mut g = Graph::new("post_attn_reuse");
        let h0 = g.input("h", f32_shape(&[1, 512]));
        let attn = g.input("attn", f32_shape(&[1, 512]));
        let ffn = g.input("ffn", f32_shape(&[1, 512]));
        let gamma = g.param("gamma", f32_shape(&[512]));
        let beta = g.param("beta", f32_shape(&[512]));
        let h = g.binary(BinaryOp::Add, h0, attn, f32_shape(&[1, 512]));
        let _n = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![h, gamma, beta],
            f32_shape(&[1, 512]),
        );
        let out = g.binary(BinaryOp::Add, h, ffn, f32_shape(&[1, 512]));
        g.set_outputs(vec![out]);

        let fused = FuseResidualRmsNorm.run(g);
        assert!(
            fused
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::FusedResidualRmsNorm { .. })),
            "must not fuse when add result feeds both rms and a later residual"
        );
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::RmsNorm { .. })),
            "rms must remain unfused"
        );
    }

    #[test]
    fn fuse_rms_norm_reshape() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[1, 8, 512]));
        let gamma = g.param("gamma", f32_shape(&[512]));
        let beta = g.param("beta", f32_shape(&[512]));
        let rn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            f32_shape(&[1, 8, 512]),
        );
        let flat = g.add_node(
            Op::Reshape {
                new_shape: vec![8, 512],
            },
            vec![rn],
            f32_shape(&[8, 512]),
        );
        let w = g.param("w", f32_shape(&[512, 128]));
        let mm = g.matmul(flat, w, f32_shape(&[8, 128]));
        g.set_outputs(vec![mm]);

        let fused = FuseRmsNormReshape.run(g);
        // x, gamma, beta, rms_norm(2d), w, matmul — no separate reshape
        assert_eq!(fused.len(), 6);
        let rn_node = fused.node(fused.node(fused.outputs[0]).inputs[0]);
        assert!(matches!(rn_node.op, Op::RmsNorm { .. }));
        assert_eq!(rn_node.shape.dim(0).unwrap_static(), 8);
        assert_eq!(rn_node.shape.dim(1).unwrap_static(), 512);
    }

    #[test]
    fn fuse_shared_input_matmul() {
        let mut g = Graph::new("swiglu");
        let x = g.input("x", f32_shape(&[60, 768]));
        let w1 = g.param("fc11", f32_shape(&[768, 2048]));
        let w2 = g.param("fc12", f32_shape(&[768, 2048]));
        let mm1 = g.matmul(x, w1, f32_shape(&[60, 2048]));
        let mm2 = g.matmul(x, w2, f32_shape(&[60, 2048]));
        g.set_outputs(vec![mm1, mm2]);

        assert_eq!(g.len(), 5); // x, w1, w2, mm1, mm2

        let fused = FuseSharedInputMatMul.run(g);
        println!("{fused}");

        // Should have: x, w1, w2, concat(w1,w2), combined_mm, narrow1, narrow2
        assert!(fused.len() <= 7);
        // Both outputs should be Narrow ops
        for &out in &fused.outputs {
            assert!(matches!(fused.node(out).op, Op::Narrow { .. }));
        }
    }

    /// F5 AdaLN packs dozens of linears on one time embed; leave them unfused
    /// so backends never materialize a ~0.5 GiB Concat weight.
    #[test]
    fn fuse_shared_input_matmul_skips_oversized_groups() {
        let mut g = Graph::new("adaln_pack");
        let x = g.input("t", f32_shape(&[1, 64]));
        let mut outs = Vec::new();
        for i in 0..8 {
            let w = g.param(format!("w{i}"), f32_shape(&[64, 128]));
            outs.push(g.matmul(x, w, f32_shape(&[1, 128])));
        }
        g.set_outputs(outs);

        let fused = FuseSharedInputMatMul.run(g);
        let concat_count = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Concat { .. }))
            .count();
        assert_eq!(
            concat_count, 0,
            "groups larger than MAX_SHARED_INPUT_MATMULS must stay unfused"
        );
        for &out in &fused.outputs {
            assert!(
                matches!(fused.node(out).op, Op::MatMul),
                "expected unfused MatMul, got {:?}",
                fused.node(out).op
            );
        }
    }

    /// Regression: `FuseSharedInputMatMul` used to panic when `w2` is
    /// declared after `mm1`. `ensure_mapped` now copies late operands.
    #[test]
    fn fuse_shared_input_matmul_with_late_w2_param() {
        let mut g = Graph::new("late_w2");
        let x = g.input("x", f32_shape(&[8, 16]));
        let w1 = g.param("w1", f32_shape(&[16, 8]));
        let mm1 = g.matmul(x, w1, f32_shape(&[8, 8]));
        let w2 = g.param("w2", f32_shape(&[16, 8]));
        let mm2 = g.matmul(x, w2, f32_shape(&[8, 8]));
        g.set_outputs(vec![mm1, mm2]);

        let fused = FuseSharedInputMatMul.run(g);
        for &out in &fused.outputs {
            assert!(
                matches!(fused.node(out).op, Op::Narrow { .. }),
                "late w2 should still fuse via ensure_mapped, got {:?}",
                fused.node(out).op
            );
        }
    }

    /// Regression: qwen35moe FFN declares router / shared-expert matmuls on the
    /// same flattened hidden state with weights scattered through the block.
    #[test]
    fn fuse_shared_input_matmul_moe_ffn_pattern() {
        let mut g = Graph::new("moe_ffn");
        let rows = 4usize;
        let n_embd = 16usize;
        let n_expert = 4usize;
        let n_ff = 16usize;

        let h_in = g.input("h", f32_shape(&[1, rows, n_embd]));
        let h_2d = g.reshape_(h_in, vec![rows as i64, n_embd as i64]);

        let router_w = g.param("router_w", f32_shape(&[n_embd, n_expert]));
        let router_logits = g.matmul(h_2d, router_w, f32_shape(&[rows, n_expert]));

        // MoE body omitted — only the shared-expert tail matters for fusion order.
        let shared_router_w = g.param("shared_router_w", f32_shape(&[n_embd, 1]));
        let shared_logits = g.matmul(h_2d, shared_router_w, f32_shape(&[rows, 1]));
        let shared_gate = g.activation(Activation::Sigmoid, shared_logits, f32_shape(&[rows, 1]));

        let s_gate_w = g.param("s_gate_w", f32_shape(&[n_embd, n_ff]));
        let s_up_w = g.param("s_up_w", f32_shape(&[n_embd, n_ff]));
        let s_gate = g.matmul(h_2d, s_gate_w, f32_shape(&[rows, n_ff]));
        let s_up = g.matmul(h_2d, s_up_w, f32_shape(&[rows, n_ff]));
        let s_gate_silu = g.silu(s_gate);
        let s_swiglu = g.mul(s_gate_silu, s_up);

        g.set_outputs(vec![router_logits, shared_gate, s_swiglu]);

        let fused = FuseSharedInputMatMul.run(g);
        let narrow_count = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Narrow { .. }))
            .count();
        assert!(
            narrow_count >= 4,
            "expected four narrow slices from fused h_2d matmuls, got {narrow_count}"
        );
    }

    /// Full pipeline: build a BERT FFN subgraph and run all fusion passes.
    #[test]
    fn full_bert_ffn_fusion() {
        let mut g = Graph::new("bert_ffn");
        let f = DType::F32;

        let x = g.input("hidden", Shape::new(&[4, 15, 384], f));
        let residual = g.input("residual", Shape::new(&[4, 15, 384], f));

        // Output projection result + residual + LN
        let out_w = g.param("out.w", Shape::new(&[384, 384], f));
        let out_b = g.param("out.b", Shape::new(&[384], f));
        let out_mm = g.matmul(x, out_w, Shape::new(&[4, 15, 384], f));
        let out_add = g.binary(BinaryOp::Add, out_mm, out_b, Shape::new(&[4, 15, 384], f));
        let res_add = g.binary(
            BinaryOp::Add,
            out_add,
            residual,
            Shape::new(&[4, 15, 384], f),
        );
        let gamma = g.param("ln.g", Shape::new(&[384], f));
        let beta = g.param("ln.b", Shape::new(&[384], f));
        let ln = g.layer_norm(
            res_add,
            gamma,
            beta,
            -1,
            1e-12,
            Shape::new(&[4, 15, 384], f),
        );

        // FFN intermediate: matmul + bias + gelu
        let int_w = g.param("int.w", Shape::new(&[384, 1536], f));
        let int_b = g.param("int.b", Shape::new(&[1536], f));
        let int_mm = g.matmul(ln, int_w, Shape::new(&[4, 15, 1536], f));
        let int_add = g.binary(BinaryOp::Add, int_mm, int_b, Shape::new(&[4, 15, 1536], f));
        let gelu = g.activation(Activation::Gelu, int_add, Shape::new(&[4, 15, 1536], f));

        // FFN output: matmul + bias
        let out2_w = g.param("out2.w", Shape::new(&[1536, 384], f));
        let out2_b = g.param("out2.b", Shape::new(&[384], f));
        let out2_mm = g.matmul(gelu, out2_w, Shape::new(&[4, 15, 384], f));
        let out2_add = g.binary(BinaryOp::Add, out2_mm, out2_b, Shape::new(&[4, 15, 384], f));

        g.set_outputs(vec![out2_add]);

        let before = g.len();
        println!("=== BEFORE fusion ({before} nodes) ===\n{g}");

        // Run all passes
        let passes: Vec<&dyn Pass> = vec![&FuseMatMulBiasAct, &FuseResidualLN];
        let optimized = run_passes(g, &passes, false);
        let after = optimized.len();
        println!("=== AFTER fusion ({after} nodes) ===\n{optimized}");

        // Should have eliminated:
        // - 2 Add + 1 Gelu from matmul_bias_gelu fusion (×2 matmuls)
        // - 1 Add from residual_ln fusion
        assert!(
            after < before,
            "fusion should reduce node count: {before} → {after}"
        );

        // Check that fused ops exist
        let ops: Vec<String> = optimized
            .nodes()
            .iter()
            .map(|n| format!("{}", n.op))
            .collect();
        let has_fused_mm = ops.iter().any(|s| s.contains("fused_mm_bias"));
        assert!(has_fused_mm, "should have fused_mm_bias_act: {ops:?}");
    }

    /// FuseSwiGLU fires on the canonical Nomic-style pattern produced by
    /// `FuseSharedInputMatMul` (concat'd matmul → narrow×2 → silu → mul).
    #[test]
    fn fuse_swiglu_canonical() {
        let mut g = Graph::new("nomic_ffn");
        let f = DType::F32;
        // After FuseSharedInputMatMul: cat = mm(x, concat(fc11, fc12)) → [60, 4096]
        let cat = g.input("cat", Shape::new(&[60, 4096], f));
        let up = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let gate = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 2048,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let silu = g.activation(Activation::Silu, gate, Shape::new(&[60, 2048], f));
        let out = g.binary(BinaryOp::Mul, up, silu, Shape::new(&[60, 2048], f));
        g.set_outputs(vec![out]);

        let before = g.len();
        let fused = FuseSwiGLU.run(g);
        let after = fused.len();
        // Removed: up, gate, silu, mul → replaced by FusedSwiGLU.
        // Net: -3 nodes (4 removed, 1 added).
        assert_eq!(
            after,
            before - 3,
            "should remove narrows+silu+mul, add FusedSwiGLU"
        );
        let out_node = fused.node(fused.outputs[0]);
        assert!(
            matches!(
                out_node.op,
                Op::FusedSwiGLU {
                    cast_to: None,
                    gate_first: false
                }
            ),
            "output should be FusedSwiGLU, got {}",
            out_node.op
        );
        // FusedSwiGLU's input is the cat tensor.
        let in_id = out_node.inputs[0];
        assert!(matches!(fused.node(in_id).op, Op::Input { .. }));
    }

    /// FuseSwiGLU does NOT fire when narrows are shared with another consumer
    /// (would corrupt the second consumer's view of the data).
    #[test]
    fn fuse_swiglu_skips_when_narrow_has_extra_user() {
        let mut g = Graph::new("contended");
        let f = DType::F32;
        let cat = g.input("cat", Shape::new(&[60, 4096], f));
        let up = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let gate = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 2048,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let silu = g.activation(Activation::Silu, gate, Shape::new(&[60, 2048], f));
        let out = g.binary(BinaryOp::Mul, up, silu, Shape::new(&[60, 2048], f));
        // Extra user of `up` — this should block fusion.
        let extra = g.activation(Activation::Relu, up, Shape::new(&[60, 2048], f));
        g.set_outputs(vec![out, extra]);

        let before = g.len();
        let fused = FuseSwiGLU.run(g);
        // Pass should be a no-op when fusion is unsafe.
        assert_eq!(fused.len(), before);
        // No FusedSwiGLU node anywhere.
        let any_fused = fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::FusedSwiGLU { .. }));
        assert!(!any_fused, "should not fuse when narrow has extra user");
    }

    // ── MarkElementwiseRegions (PLAN L2) ────────────────────────────

    #[test]
    fn region_collapses_add_mul_relu_chain() {
        // Build: out = relu(add(a, b) * c). All same shape, single consumer
        // chain. Should fuse into one ElementwiseRegion.
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[8], f));
        let b = g.input("b", Shape::new(&[8], f));
        let c = g.input("c", Shape::new(&[8], f));
        let s = Shape::new(&[8], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let mul = g.binary(BinaryOp::Mul, add, c, s.clone());
        let relu = g.activation(Activation::Relu, mul, s.clone());
        g.set_outputs(vec![relu]);

        let before = g.len();
        let fused = MarkElementwiseRegions.run(g);

        // Three element-wise ops collapsed into one region node.
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1, "expected one ElementwiseRegion");
        let region = regions[0];
        assert_eq!(
            region.inputs.len(),
            3,
            "region has 3 external inputs (a, b, c)"
        );
        if let Op::ElementwiseRegion {
            chain, num_inputs, ..
        } = &region.op
        {
            assert_eq!(*num_inputs, 3);
            assert_eq!(chain.len(), 3);
            // Step 0: Add(Input(0), Input(1))
            match &chain[0] {
                ChainStep::Binary(
                    BinaryOp::Add,
                    ChainOperand::Input(0),
                    ChainOperand::Input(1),
                ) => {}
                other => panic!("step 0 unexpected: {other:?}"),
            }
            // Step 1: Mul(Step(0), Input(2))
            match &chain[1] {
                ChainStep::Binary(BinaryOp::Mul, ChainOperand::Step(0), ChainOperand::Input(2)) => {
                }
                other => panic!("step 1 unexpected: {other:?}"),
            }
            // Step 2: Activation(Relu, Step(1))
            match &chain[2] {
                ChainStep::Activation(Activation::Relu, ChainOperand::Step(1)) => {}
                other => panic!("step 2 unexpected: {other:?}"),
            }
        } else {
            unreachable!();
        }
        // Original chain (3 ops) replaced by 1 region; net node count is
        // (inputs 3) + (region 1) = 4 (vs 3 + 3 = 6 before).
        assert!(fused.len() < before);
    }

    #[test]
    fn region_does_not_fuse_when_intermediate_has_multiple_consumers() {
        // out1 = add(a, b); out2 = relu(out1). out1 also fed to out_extra.
        // Multi-consumer on out1 forbids fusion.
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let relu = g.activation(Activation::Relu, add, s.clone());
        let extra = g.activation(Activation::Sigmoid, add, s.clone());
        g.set_outputs(vec![relu, extra]);

        let before = g.len();
        let fused = MarkElementwiseRegions.run(g);
        // No region: add has two consumers (relu and extra), so the chain
        // can't extend through it. Each downstream activation is alone in
        // its region (size 1, doesn't fuse).
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 0);
        assert_eq!(fused.len(), before);
    }

    #[test]
    fn region_skips_chains_of_length_one() {
        // Single relu — no fusion (size 1 = degenerate).
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[4], f));
        let r = g.activation(Activation::Relu, a, Shape::new(&[4], f));
        g.set_outputs(vec![r]);

        let fused = MarkElementwiseRegions.run(g);
        let any_region = fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }));
        assert!(!any_region);
    }

    #[test]
    fn unfuse_decomposes_region_back_to_atomic_ops() {
        // Build the same chain, fuse it, then unfuse — expect the
        // original atomic ops back (Add, Mul, Relu).
        let f = DType::F32;
        let mut g = Graph::new("ew_unfuse");
        let a = g.input("a", Shape::new(&[8], f));
        let b = g.input("b", Shape::new(&[8], f));
        let c = g.input("c", Shape::new(&[8], f));
        let s = Shape::new(&[8], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let mul = g.binary(BinaryOp::Mul, add, c, s.clone());
        let relu = g.activation(Activation::Relu, mul, s);
        g.set_outputs(vec![relu]);

        let fused = MarkElementwiseRegions.run(g);
        // Sanity: fusion happened.
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        );

        let unfused = UnfuseElementwiseRegions::FOR_CPU.run(fused);
        // No region nodes left.
        assert!(
            !unfused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        );
        // Original atomic ops are back: Add, Mul, Relu.
        let bin_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(_)))
            .count();
        let act_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Activation(_)))
            .count();
        assert_eq!(bin_count, 2, "Add + Mul restored");
        assert_eq!(act_count, 1, "Relu restored");
    }

    #[test]
    fn clip_unfuses_region_over_step_cap() {
        use rlx_ir::op::{Activation, ChainOperand, ChainStep};

        let mut g = Graph::new("clip");
        let x = g.input("x", f32_shape(&[4]));
        let mut chain: Vec<ChainStep> = Vec::new();
        let mut prev = ChainOperand::Input(0);
        for _ in 0..40 {
            chain.push(ChainStep::Activation(Activation::Relu, prev));
            prev = ChainOperand::Step(chain.len() as u32 - 1);
        }
        let y = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![x],
            f32_shape(&[4]),
        );
        g.set_outputs(vec![y]);

        let clipped = clip_elementwise_regions(g, FusionLimits::GPU_NATIVE);
        assert!(
            !clipped
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. })),
            "oversized region should be decomposed"
        );
        assert!(clipped.len() > 5);
    }

    #[test]
    fn unfuse_is_noop_when_no_region_present() {
        let f = DType::F32;
        let mut g = Graph::new("noop");
        let a = g.input("a", Shape::new(&[4], f));
        let r = g.activation(Activation::Relu, a, Shape::new(&[4], f));
        g.set_outputs(vec![r]);
        let n_before = g.len();
        let result = UnfuseElementwiseRegions::FOR_CPU.run(g);
        // Pass returns unchanged graph (early return on no-region check).
        assert_eq!(result.len(), n_before);
    }

    #[test]
    fn region_includes_where_step() {
        // Build: cmp = a > b; sel = where(cmp, a, b); out = sel + a
        // The compare → where → add chain is fully element-wise; the
        // Where step lands inside the region thanks to the L2-quality
        // extension that adds `Op::Where` to the chain-eligible set.
        let f = DType::F32;
        let mut g = Graph::new("region_where");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let cmp = g.add_node(Op::Compare(CmpOp::Gt), vec![a, b], s.clone());
        let sel = g.add_node(Op::Where, vec![cmp, a, b], s.clone());
        let add = g.binary(BinaryOp::Add, sel, a, s.clone());
        g.set_outputs(vec![add]);

        let fused = MarkElementwiseRegions.run(g);
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1, "expected one ElementwiseRegion");
        if let Op::ElementwiseRegion { chain, .. } = &regions[0].op {
            // 3 steps: Compare a > b, Where, Add
            assert_eq!(chain.len(), 3);
            assert!(
                matches!(chain[1], ChainStep::Where(_, _, _)),
                "step 1 should be Where, got {:?}",
                chain[1]
            );
        } else {
            unreachable!();
        }
    }

    #[test]
    fn unfuse_decomposes_where_step_back_to_op_where() {
        // Round-trip: build a region with a Where step, decompose it,
        // verify the resulting graph contains an Op::Where node.
        let f = DType::F32;
        let mut g = Graph::new("unfuse_where");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let cmp = g.add_node(Op::Compare(CmpOp::Gt), vec![a, b], s.clone());
        let sel = g.add_node(Op::Where, vec![cmp, a, b], s.clone());
        let add = g.binary(BinaryOp::Add, sel, a, s.clone());
        g.set_outputs(vec![add]);
        let fused = MarkElementwiseRegions.run(g);
        let unfused = UnfuseElementwiseRegions::FOR_CPU.run(fused);
        let where_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Where))
            .count();
        assert_eq!(
            where_count, 1,
            "decomposer should re-emit one Op::Where for the chain step"
        );
    }

    /// Synthetic BERT attention block: input [B,S,H] → QKV proj (matmul+bias) →
    /// narrow×3 → Attention(mask) → OutProj (matmul+bias) → output [B,S,H].
    /// Runs FuseMatMulBiasAct then FuseAttentionBlock and asserts collapse.
    #[test]
    fn fuse_attention_block_collapses_qkv_attn_outproj() {
        let nh: usize = 4;
        let dh: usize = 8;
        let h: usize = nh * dh; // 32
        let b: usize = 1;
        let s: usize = 4; // tiny — keep b*s ≤ 64 so should_fuse fires

        let mut g = Graph::new("attn-block");
        let hidden = g.input("hidden", f32_shape(&[b, s, h]));
        let mask = g.input("attention_mask", f32_shape(&[b, s]));

        // QKV projection (matmul + bias).
        let qkv_w = g.param("qkv_w", f32_shape(&[h, 3 * h]));
        let qkv_b = g.param("qkv_b", f32_shape(&[3 * h]));
        let qkv_mm = g.matmul(hidden, qkv_w, f32_shape(&[b, s, 3 * h]));
        let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, f32_shape(&[b, s, 3 * h]));

        // Three narrows on the innermost axis.
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 2 * h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );

        // Attention with custom (input) mask.
        let attn = g.attention(q, k, v, mask, nh, dh, f32_shape(&[b, s, h]));

        // OutProj (matmul + bias).
        let out_w = g.param("out_w", f32_shape(&[h, h]));
        let out_b = g.param("out_b", f32_shape(&[h]));
        let out_mm = g.matmul(attn, out_w, f32_shape(&[b, s, h]));
        let out = g.binary(BinaryOp::Add, out_mm, out_b, f32_shape(&[b, s, h]));
        g.set_outputs(vec![out]);

        // Step 1: FuseMatMulBiasAct collapses each matmul+bias into one node.
        let fused1 = FuseMatMulBiasAct.run(g);
        let mm_bias_count = fused1
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedMatMulBiasAct { activation: None }))
            .count();
        assert_eq!(mm_bias_count, 2, "QKV + OutProj should each fuse");

        // Step 2: FuseAttentionBlock collapses QKV-MM → narrow×3 → Attention → OutProj-MM
        // into one FusedAttentionBlock node.
        let fused2 = FuseAttentionBlock.run(fused1);
        let fab_count = fused2
            .nodes()
            .iter()
            .filter(|n| {
                matches!(
                    n.op,
                    Op::FusedAttentionBlock {
                        has_bias: true,
                        has_rope: false,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            fab_count, 1,
            "should produce exactly one FusedAttentionBlock"
        );

        // No stray Narrow / Attention / FusedMatMulBiasAct should remain from
        // the collapsed chain.
        let narrow_count = fused2
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Narrow { .. }))
            .count();
        let attention_count = fused2
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Attention { .. }))
            .count();
        let mm_bias_remaining = fused2
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. }))
            .count();
        assert_eq!(narrow_count, 0, "QKV narrows absorbed");
        assert_eq!(attention_count, 0, "Attention absorbed");
        assert_eq!(mm_bias_remaining, 0, "both projections absorbed");

        let out_node = fused2.node(fused2.outputs[0]);
        assert!(matches!(out_node.op, Op::FusedAttentionBlock { .. }));
    }

    /// Synthetic full BERT layer (one block): hidden → FusedAttentionBlock →
    /// FusedResidualLN → FusedMatMulBiasAct(GeLU) → FusedMatMulBiasAct →
    /// FusedResidualLN. Confirm FuseTransformerLayer collapses to one node.
    #[test]
    fn fuse_transformer_layer_collapses_full_bert_block() {
        let nh: usize = 4;
        let dh: usize = 8;
        let h: usize = nh * dh;
        let inter = 4 * h;
        let eps1: f32 = 1e-12;
        let eps2: f32 = 1e-12;
        let b: usize = 1;
        let s: usize = 4;

        let mut g = Graph::new("bert-layer");
        let hidden = g.input("hidden", f32_shape(&[b, s, h]));
        let mask = g.input("attention_mask", f32_shape(&[b, s]));

        // === Attention block ===
        let qkv_w = g.param("qkv_w", f32_shape(&[h, 3 * h]));
        let qkv_b = g.param("qkv_b", f32_shape(&[3 * h]));
        let qkv_mm = g.matmul(hidden, qkv_w, f32_shape(&[b, s, 3 * h]));
        let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, f32_shape(&[b, s, 3 * h]));
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 2 * h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let attn = g.attention(q, k, v, mask, nh, dh, f32_shape(&[b, s, h]));
        let out_w = g.param("out_w", f32_shape(&[h, h]));
        let out_b = g.param("out_b", f32_shape(&[h]));
        let out_mm = g.matmul(attn, out_w, f32_shape(&[b, s, h]));
        let attn_out = g.binary(BinaryOp::Add, out_mm, out_b, f32_shape(&[b, s, h]));

        // === Post-attn residual + LN ===
        let res1 = g.binary(BinaryOp::Add, attn_out, hidden, f32_shape(&[b, s, h]));
        let ln1_g = g.param("ln1_g", f32_shape(&[h]));
        let ln1_b = g.param("ln1_b", f32_shape(&[h]));
        let h1 = g.add_node(
            Op::LayerNorm {
                axis: -1,
                eps: eps1,
            },
            vec![res1, ln1_g, ln1_b],
            f32_shape(&[b, s, h]),
        );

        // === FFN ===
        let fc1_w = g.param("fc1_w", f32_shape(&[h, inter]));
        let fc1_b = g.param("fc1_b", f32_shape(&[inter]));
        let fc1_mm = g.matmul(h1, fc1_w, f32_shape(&[b, s, inter]));
        let fc1_add = g.binary(BinaryOp::Add, fc1_mm, fc1_b, f32_shape(&[b, s, inter]));
        let fc1_act = g.activation(Activation::Gelu, fc1_add, f32_shape(&[b, s, inter]));
        let fc2_w = g.param("fc2_w", f32_shape(&[inter, h]));
        let fc2_b = g.param("fc2_b", f32_shape(&[h]));
        let fc2_mm = g.matmul(fc1_act, fc2_w, f32_shape(&[b, s, h]));
        let ffn_out = g.binary(BinaryOp::Add, fc2_mm, fc2_b, f32_shape(&[b, s, h]));

        // === Post-FFN residual + LN ===
        let res2 = g.binary(BinaryOp::Add, ffn_out, h1, f32_shape(&[b, s, h]));
        let ln2_g = g.param("ln2_g", f32_shape(&[h]));
        let ln2_b = g.param("ln2_b", f32_shape(&[h]));
        let out = g.add_node(
            Op::LayerNorm {
                axis: -1,
                eps: eps2,
            },
            vec![res2, ln2_g, ln2_b],
            f32_shape(&[b, s, h]),
        );
        g.set_outputs(vec![out]);

        // Run the same pipeline order the production pipeline uses.
        let g = FuseMatMulBiasAct.run(g);
        let g = FuseResidualLN.run(g);
        let g = FuseAttentionBlock.run(g);
        let g = FuseTransformerLayer.run(g);

        let ftl_count = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedTransformerLayer { .. }))
            .count();
        assert_eq!(
            ftl_count, 1,
            "single layer should collapse to one FusedTransformerLayer"
        );

        // After the full pipeline, the layer's intermediate fused ops should
        // be gone — only the parameter / input nodes and the single
        // FusedTransformerLayer remain.
        let leftover_fab = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedAttentionBlock { .. }))
            .count();
        let leftover_frln = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedResidualLN { .. }))
            .count();
        let leftover_fmba = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. }))
            .count();
        assert_eq!(leftover_fab, 0, "attn block absorbed into layer");
        assert_eq!(leftover_frln, 0, "both residual+LNs absorbed");
        assert_eq!(leftover_fmba, 0, "FFN matmuls absorbed");

        let out_node = g.node(g.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedTransformerLayer {
                num_heads: 4,
                head_dim: 8,
                intermediate_size: 128,
                has_bias: true,
                ..
            }
        ));
        assert_eq!(out_node.inputs.len(), 14);
    }

    /// `should_fuse` must reject the pass when batch·seq exceeds the threshold,
    /// so attention block fusion stays opt-in for small inputs.
    #[test]
    fn fuse_attention_block_skips_large_inputs() {
        let nh: usize = 4;
        let dh: usize = 8;
        let h: usize = nh * dh;
        let b: usize = 16;
        let s: usize = 128; // b*s = 2048 ≫ 64 default threshold

        let mut g = Graph::new("attn-block-large");
        let hidden = g.input("hidden", f32_shape(&[b, s, h]));
        let mask = g.input("attention_mask", f32_shape(&[b, s]));
        let qkv_w = g.param("qkv_w", f32_shape(&[h, 3 * h]));
        let qkv_b = g.param("qkv_b", f32_shape(&[3 * h]));
        let qkv_mm = g.matmul(hidden, qkv_w, f32_shape(&[b, s, 3 * h]));
        let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, f32_shape(&[b, s, 3 * h]));
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 2 * h,
                len: h,
            },
            vec![qkv],
            f32_shape(&[b, s, h]),
        );
        let attn = g.attention(q, k, v, mask, nh, dh, f32_shape(&[b, s, h]));
        let out_w = g.param("out_w", f32_shape(&[h, h]));
        let out_b = g.param("out_b", f32_shape(&[h]));
        let out_mm = g.matmul(attn, out_w, f32_shape(&[b, s, h]));
        let out = g.binary(BinaryOp::Add, out_mm, out_b, f32_shape(&[b, s, h]));
        g.set_outputs(vec![out]);

        let fused1 = FuseMatMulBiasAct.run(g);
        let fused2 = FuseAttentionBlock.run(fused1);
        let fab_count = fused2
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedAttentionBlock { .. }))
            .count();
        assert_eq!(fab_count, 0, "block-fusion must skip large batches");
    }

    #[test]
    fn fuse_ada_layer_norm_one_plus_scale() {
        use rlx_ir::infer::GraphExt;
        let (b, s, d) = (2usize, 4usize, 8usize);
        let eps = 1e-5f32;
        let mut g = Graph::new("ada");
        let x = g.input("x", f32_shape(&[b, s, d]));
        let scale = g.input("scale", f32_shape(&[b, 1, d]));
        let shift = g.input("shift", f32_shape(&[b, 1, d]));
        let gamma = g.full(&[d], 1.0, DType::F32);
        let beta = g.zeros(&[d], DType::F32);
        let n = g.layer_norm(x, gamma, beta, -1, eps, f32_shape(&[b, s, d]));
        let scale_e = g.add_node(
            Op::Expand {
                target_shape: vec![b as i64, s as i64, d as i64],
            },
            vec![scale],
            f32_shape(&[b, s, d]),
        );
        let one = g.full(&[1], 1.0, DType::F32);
        let one_plus = g.binary(BinaryOp::Add, one, scale_e, f32_shape(&[b, s, d]));
        let scaled = g.binary(BinaryOp::Mul, n, one_plus, f32_shape(&[b, s, d]));
        let shift_e = g.add_node(
            Op::Expand {
                target_shape: vec![b as i64, s as i64, d as i64],
            },
            vec![shift],
            f32_shape(&[b, s, d]),
        );
        let out = g.binary(BinaryOp::Add, scaled, shift_e, f32_shape(&[b, s, d]));
        g.set_outputs(vec![out]);

        let fused = FuseAdaLayerNorm.run(g);
        let out_node = fused.node(fused.outputs[0]);
        assert!(
            matches!(
                out_node.op,
                Op::AdaLayerNorm {
                    norm: AdaNormKind::LayerNorm,
                    ..
                }
            ),
            "expected AdaLayerNorm, got {:?}",
            out_node.op
        );
        assert_eq!(out_node.inputs.len(), 3);
    }

    #[test]
    fn fuse_gated_residual_with_expand() {
        let (b, s, d) = (2usize, 4usize, 8usize);
        let mut g = Graph::new("gate");
        let x = g.input("x", f32_shape(&[b, s, d]));
        let y = g.input("y", f32_shape(&[b, s, d]));
        let gate = g.input("gate", f32_shape(&[b, 1, d]));
        let gate_e = g.add_node(
            Op::Expand {
                target_shape: vec![b as i64, s as i64, d as i64],
            },
            vec![gate],
            f32_shape(&[b, s, d]),
        );
        let gy = g.binary(BinaryOp::Mul, gate_e, y, f32_shape(&[b, s, d]));
        let out = g.binary(BinaryOp::Add, x, gy, f32_shape(&[b, s, d]));
        g.set_outputs(vec![out]);

        let fused = FuseGatedResidual.run(g);
        let out_node = fused.node(fused.outputs[0]);
        assert!(
            matches!(out_node.op, Op::GatedResidual),
            "expected GatedResidual, got {:?}",
            out_node.op
        );
        assert_eq!(out_node.inputs.len(), 3);
        // Prefer pre-Expand gate.
        let gate_in = fused.node(out_node.inputs[2]);
        assert!(matches!(gate_in.op, Op::Input { .. }));
        assert_eq!(gate_in.shape.dims(), f32_shape(&[b, 1, d]).dims());
    }
}
