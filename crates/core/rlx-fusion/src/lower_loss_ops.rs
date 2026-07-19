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

//! Lower `SoftmaxCrossEntropy` (dense / soft-label) /
//! `SoftmaxCrossEntropyWithLogits` / `SoftmaxCrossEntropyBackward` to
//! primitives for backends (CUDA, Metal) that lack native kernels.

use std::collections::HashMap;

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, CmpOp, ReduceOp};
use rlx_ir::shape;
use rlx_ir::*;

fn scalar_const(g: &mut Graph, v: f32, dtype: DType) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], dtype),
    )
}

fn broadcast_scalar(g: &mut Graph, scalar: NodeId, like: &Shape) -> NodeId {
    let dims: Vec<i64> = like
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let out = Shape::new(
        &dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        like.dtype(),
    );
    g.add_node(Op::Expand { target_shape: dims }, vec![scalar], out)
}

fn compare_eq(g: &mut Graph, lhs: NodeId, rhs: NodeId) -> NodeId {
    let s = shape::compare_shape(g.shape(lhs), g.shape(rhs)).expect("compare eq");
    g.add_node(Op::Compare(CmpOp::Eq), vec![lhs, rhs], s)
}

fn one_hot_2d(g: &mut Graph, labels: NodeId, n: usize, c: usize, dt: DType) -> NodeId {
    let labels_shape = g.shape(labels).clone();
    let one = scalar_const(g, 1.0, dt);
    let zero = scalar_const(g, 0.0, dt);
    let one_b = broadcast_scalar(g, one, &labels_shape);
    let zero_b = broadcast_scalar(g, zero, &labels_shape);
    let mut cols = Vec::with_capacity(c);
    for ci in 0..c {
        let class = scalar_const(g, ci as f32, dt);
        let class_b = broadcast_scalar(g, class, &labels_shape);
        let eq = compare_eq(g, labels, class_b);
        let col = g.add_node(Op::Where, vec![eq, one_b, zero_b], labels_shape.clone());
        // Each class column is `[N]`; reshape to `[N,1]` and concat along the
        // class axis so `one_hot[n][ci] = (labels[n] == ci)`. (Concatenating the
        // `[N]` columns along axis 0 and reshaping to `[N,c]` scrambles the layout
        // whenever N ≠ c — it interleaves classes across examples.)
        let col2d = g.reshape_(col, vec![n as i64, 1]);
        cols.push(col2d);
    }
    g.concat_(cols, 1)
}

/// `loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]`
/// (dense / soft-label cross-entropy).
fn lower_softmax_cross_entropy_dense(
    g: &mut Graph,
    logits: NodeId,
    targets: NodeId,
    out_shape: Shape,
) -> NodeId {
    let logits_shape = g.shape(logits).clone();
    let n = logits_shape.dim(0).unwrap_static();
    let axis = 1usize;

    let m_shape = shape::reduce_shape(&logits_shape, &[axis], true).expect("max reduce");
    let m = g.reduce(logits, ReduceOp::Max, vec![axis], true, m_shape);
    let shifted = g.sub(logits, m);
    let exp_d = g.exp(shifted);
    let sum_shape = shape::reduce_shape(&logits_shape, &[axis], false).expect("sum reduce");
    let sum_exp = g.reduce(exp_d, ReduceOp::Sum, vec![axis], false, sum_shape.clone());
    let log_sum = g.add_node(
        Op::Activation(Activation::Log),
        vec![sum_exp],
        sum_shape.clone(),
    );
    let m_squeezed = g.reshape_(m, vec![n as i64]);
    let lse = g.add(m_squeezed, log_sum);

    // Σ_c targets[n,c]·logits[n,c] along the class axis.
    let prod = g.mul(logits, targets);
    let dot = g.reduce(prod, ReduceOp::Sum, vec![axis], false, out_shape.clone());

    g.sub(lse, dot)
}

/// `loss[n] = logsumexp(logits[n]) - logits[n, labels[n]]`.
fn lower_softmax_cross_entropy_with_logits(
    g: &mut Graph,
    logits: NodeId,
    labels: NodeId,
    out_shape: Shape,
) -> NodeId {
    let logits_shape = g.shape(logits).clone();
    let n = logits_shape.dim(0).unwrap_static();
    let c = logits_shape.dim(1).unwrap_static();
    let dt = logits_shape.dtype();
    let axis = 1usize;

    let labels_flat = if g.shape(labels).rank() == 1 {
        labels
    } else {
        g.reshape_(labels, vec![n as i64])
    };

    let m_shape = shape::reduce_shape(&logits_shape, &[axis], true).expect("max reduce");
    let m = g.reduce(logits, ReduceOp::Max, vec![axis], true, m_shape);
    let shifted = g.sub(logits, m);
    let exp_d = g.exp(shifted);
    let sum_shape = shape::reduce_shape(&logits_shape, &[axis], false).expect("sum reduce");
    let sum_exp = g.reduce(exp_d, ReduceOp::Sum, vec![axis], false, sum_shape.clone());
    let log_sum = g.add_node(
        Op::Activation(Activation::Log),
        vec![sum_exp],
        sum_shape.clone(),
    );
    let m_squeezed = g.reshape_(m, vec![n as i64]);
    let lse = g.add(m_squeezed, log_sum);

    let one_hot = one_hot_2d(g, labels_flat, n, c, dt);
    let masked = g.mul(logits, one_hot);
    let logit_at_label = g.reduce(masked, ReduceOp::Sum, vec![axis], false, out_shape.clone());

    g.sub(lse, logit_at_label)
}

/// `dlogits = (softmax(logits) - one_hot(labels)) * d_loss`.
fn lower_softmax_cross_entropy_backward(
    g: &mut Graph,
    logits: NodeId,
    labels: NodeId,
    d_loss: NodeId,
    out_shape: Shape,
) -> NodeId {
    let logits_shape = g.shape(logits).clone();
    let n = logits_shape.dim(0).unwrap_static();
    let c = logits_shape.dim(1).unwrap_static();
    let dt = logits_shape.dtype();

    let sm = g.softmax(logits, -1, out_shape.clone());
    let labels_flat = if g.shape(labels).rank() == 1 {
        labels
    } else {
        g.reshape_(labels, vec![n as i64])
    };
    let one_hot = one_hot_2d(g, labels_flat, n, c, dt);
    let diff = g.sub(sm, one_hot);
    // `d_loss` is the per-example upstream gradient `[N]`; it scales each ROW
    // (example) of `dlogits`, matching the CPU kernel `(p − one_hot)·d_loss[n]`.
    // Reshape to `[N,1]` before expanding to `[N,C]` so it broadcasts across the
    // CLASS axis — a bare `[N]→[N,C]` expand mis-aligns `N` with `C` under
    // trailing-dim broadcasting (wrong, and out-of-bounds, whenever N≠C).
    let dl_2d = g.reshape_(d_loss, vec![n as i64, 1]);
    let dl_b = g.add_node(
        Op::Expand {
            target_shape: vec![n as i64, c as i64],
        },
        vec![dl_2d],
        out_shape.clone(),
    );
    g.mul(diff, dl_b)
}

/// Rewrite SCE forward/backward nodes to primitive ops.
pub struct LowerSoftmaxCrossEntropy;

impl Pass for LowerSoftmaxCrossEntropy {
    fn name(&self) -> &str {
        "lower_softmax_cross_entropy"
    }

    fn run(&self, graph: Graph) -> Graph {
        let needs = graph.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::SoftmaxCrossEntropy
                    | Op::SoftmaxCrossEntropyWithLogits
                    | Op::SoftmaxCrossEntropyBackward
            )
        });
        if !needs {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::SoftmaxCrossEntropy => {
                    let logits = id_map[&node.inputs[0]];
                    let targets = id_map[&node.inputs[1]];
                    lower_softmax_cross_entropy_dense(
                        &mut new_graph,
                        logits,
                        targets,
                        node.shape.clone(),
                    )
                }
                Op::SoftmaxCrossEntropyWithLogits => {
                    let logits = id_map[&node.inputs[0]];
                    let labels = id_map[&node.inputs[1]];
                    lower_softmax_cross_entropy_with_logits(
                        &mut new_graph,
                        logits,
                        labels,
                        node.shape.clone(),
                    )
                }
                Op::SoftmaxCrossEntropyBackward => {
                    let logits = id_map[&node.inputs[0]];
                    let labels = id_map[&node.inputs[1]];
                    let d_loss = id_map[&node.inputs[2]];
                    lower_softmax_cross_entropy_backward(
                        &mut new_graph,
                        logits,
                        labels,
                        d_loss,
                        node.shape.clone(),
                    )
                }
                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                    new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
                }
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_sce_for_cuda_primitives() {
        let f = DType::F32;
        let mut g = Graph::new("sce");
        let logits = g.input("logits", Shape::new(&[4, 4], f));
        let labels = g.input("labels", Shape::new(&[4], f));
        let loss = g.softmax_cross_entropy_with_logits(logits, labels);
        g.set_outputs(vec![loss]);

        let cuda_like = &[
            OpKind::Input,
            OpKind::Constant,
            OpKind::Reduce,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Activation,
            OpKind::Reshape,
            OpKind::Compare,
            OpKind::Where,
            OpKind::Concat,
            OpKind::Softmax,
        ];
        let lowered = LowerSoftmaxCrossEntropy.run(g);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::SoftmaxCrossEntropyWithLogits))
        );
        let _ = lowered;
        let _ = cuda_like;
    }
}
