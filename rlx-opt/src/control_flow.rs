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

//! Control-flow lowering passes: `Op::If` → `Where` + inlined
//! branches; `Op::While` → bounded unroll of body replicas.
//!
//! Backends that don't have native sub-graph executors run
//! `LowerControlFlow` BEFORE the legalize / supported-set check so
//! they never see `Op::If` or `Op::While`. Used by rlx-cpu and
//! rlx-metal (the runtime-side `run_if` / `run_while` helpers exist
//! but the executor wiring through their thunk schedules is more
//! invasive than this rewrite). Other backends (rlx-wgpu, rlx-cuda,
//! rlx-rocm, rlx-tpu) ship their own per-backend unfuse passes that
//! do equivalent work — this module is the portable, IR-level
//! version.
//!
//! Trade-offs:
//!   * `Op::If` always evaluates **both** branches in the rewritten
//!     graph. That's the price of expressing it via primitives. Fine
//!     for inference where Op::If is rare; if a workload hits a
//!     hot Op::If on a path where both branches are expensive, the
//!     fix is a backend-native If executor, not this rewrite.
//!   * `Op::While` requires `max_iterations = Some(N)` — unbounded
//!     loops have no terminating count and panic with a clear
//!     message pointing at `rlx_runtime::subgraph::run_while` for
//!     the dynamic alternative.
//!
//! Capture binding (used by both passes): each sub-graph's
//! `Op::Input` nodes appear in the same order as the parent's
//! captures (`inputs[1..]` for `Op::If` past the predicate, all
//! `inputs[..]` for `Op::While`). Sub-graph `Op::Input[i]` rewires
//! to `captures[i]` when inlined into the parent.

use crate::pass::Pass;
use rlx_ir::shape::Dim;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

/// Pass form: rewrites `Op::If` and `Op::While` into primitive ops.
/// No-op when neither op is present.
pub struct LowerControlFlow;

impl Pass for LowerControlFlow {
    fn name(&self) -> &str {
        "LowerControlFlow"
    }
    fn run(&self, graph: Graph) -> Graph {
        let g = inline_if(graph);
        unroll_while(g)
    }
}

/// Inline `Op::If` sub-graphs into the parent and replace the If
/// node with `Where(predicate, then_output, else_output)`. Both
/// branches are present in the rewritten graph and always evaluate.
pub fn inline_if(g: Graph) -> Graph {
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::If {
                then_branch,
                else_branch,
            } => {
                let captures: Vec<NodeId> = new_inputs[1..].to_vec();
                let then_out = inline_subgraph_into(then_branch, &captures, &mut out);
                let else_out = inline_subgraph_into(else_branch, &captures, &mut out);
                // Most backends' Where kernel requires the predicate
                // to share the output's element count (no broadcast
                // inside the kernel). Expand a smaller predicate up
                // to the output shape so the rewritten graph runs
                // out of the box on CPU/Metal.
                let predicate = expand_to_shape(new_inputs[0], &node.shape, &mut out);
                out.add_node(
                    Op::Where,
                    vec![predicate, then_out, else_out],
                    node.shape.clone(),
                )
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|i| id_map[i]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Bounded-unroll `Op::While` up to `max_iterations`. Each iteration
/// inlines the body sub-graph, threading the single loop-carried
/// value through. The cond sub-graph is NOT evaluated at unroll time
/// — the unrolled graph always runs the full N iterations.
pub fn unroll_while(g: Graph) -> Graph {
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::While {
                body,
                max_iterations: Some(n),
                ..
            } => {
                if new_inputs.is_empty() {
                    panic!(
                        "Op::While unroll: at least one \
                            loop-carried input required"
                    );
                }
                let mut current = new_inputs[0];
                for _ in 0..*n {
                    let captures: [NodeId; 1] = [current];
                    current = inline_subgraph_into(body, &captures, &mut out);
                }
                current
            }
            Op::While {
                max_iterations: None,
                ..
            } => {
                panic!(
                    "LowerControlFlow: Op::While requires \
                        max_iterations = Some(N) for unrolling. \
                        Either set a bounded max_iterations on the \
                        forward graph, or use the dynamic \
                        `rlx_runtime::subgraph::run_while` helper."
                );
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|i| id_map[i]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Expand a tensor up to `target` via `Op::Expand` if its shape
/// (specifically its element count) differs from the target. Used to
/// promote a scalar / smaller predicate up to the Where output shape
/// during `Op::If` lowering.
fn expand_to_shape(src: NodeId, target: &rlx_ir::Shape, out: &mut Graph) -> NodeId {
    let src_shape = out.node(src).shape.clone();
    let src_n = src_shape
        .dims()
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .product::<usize>();
    let tgt_n = target
        .dims()
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .product::<usize>();
    if src_shape.dims() == target.dims() {
        return src;
    }
    let target_dims_i64: Vec<i64> = target
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            _ => -1,
        })
        .collect();
    // Op::Expand requires equal rank (broadcast via 1-dim only).
    // If src has a smaller rank, left-pad with 1s via a Reshape first.
    let src_rank = src_shape.rank();
    let tgt_rank = target.dims().len();
    let to_expand = if src_rank < tgt_rank {
        let mut padded_dims: Vec<Dim> = std::iter::repeat_n(Dim::Static(1), tgt_rank - src_rank)
            .chain(src_shape.dims().iter().copied())
            .collect();
        // Width of last dim follows src; rank gain pads with 1s.
        let _ = src_n;
        let _ = tgt_n;
        let dtype = src_shape.dtype();
        let pad_dims_i64: Vec<i64> = padded_dims
            .iter()
            .map(|d| match d {
                Dim::Static(n) => *n as i64,
                _ => -1,
            })
            .collect();
        // Borrow the padded shape for Reshape's output.
        let pad_shape = rlx_ir::Shape::from_dims(&padded_dims, dtype);
        padded_dims.clear();
        out.reshape(src, pad_dims_i64, pad_shape)
    } else {
        src
    };
    out.add_node(
        Op::Expand {
            target_shape: target_dims_i64,
        },
        vec![to_expand],
        target.clone(),
    )
}

/// Helper: copy `sub`'s nodes into `out`, mapping each Op::Input
/// by position to the corresponding capture. Returns the new
/// NodeId in `out` of the sub-graph's first declared output.
pub(crate) fn inline_subgraph_into(sub: &Graph, captures: &[NodeId], out: &mut Graph) -> NodeId {
    let mut sub_to_parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut input_idx = 0usize;
    for sub_node in sub.nodes() {
        let new_id = match &sub_node.op {
            Op::Input { .. } => {
                let parent_id = captures[input_idx];
                input_idx += 1;
                parent_id
            }
            _ => {
                let new_inputs: Vec<NodeId> =
                    sub_node.inputs.iter().map(|i| sub_to_parent[i]).collect();
                out.add_node(sub_node.op.clone(), new_inputs, sub_node.shape.clone())
            }
        };
        sub_to_parent.insert(sub_node.id, new_id);
    }
    sub_to_parent[&sub.outputs[0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Shape};

    #[test]
    fn lower_control_flow_pass_handles_both_if_and_while() {
        let s = Shape::new(&[2], DType::F32);

        let mut then_g = Graph::new("th");
        let ti = then_g.input("c", s.clone());
        let to = then_g.activation(Activation::Relu, ti, s.clone());
        then_g.set_outputs(vec![to]);
        let mut else_g = Graph::new("el");
        let ei = else_g.input("c", s.clone());
        let eo = else_g.activation(Activation::Sigmoid, ei, s.clone());
        else_g.set_outputs(vec![eo]);

        let mut body_g = Graph::new("body");
        let bi = body_g.input("c", s.clone());
        let bo = body_g.binary(BinaryOp::Mul, bi, bi, s.clone());
        body_g.set_outputs(vec![bo]);
        let mut cond_g = Graph::new("cond");
        let ci = cond_g.input("c", s.clone());
        cond_g.set_outputs(vec![ci]);

        let mut g = Graph::new("parent");
        let x = g.input("x", s.clone());
        let pred = g.input("p", Shape::new(&[1], DType::F32));
        let if_out = g.add_node(
            Op::If {
                then_branch: Box::new(then_g),
                else_branch: Box::new(else_g),
            },
            vec![pred, x],
            s.clone(),
        );
        let w_out = g.add_node(
            Op::While {
                cond: Box::new(cond_g),
                body: Box::new(body_g),
                max_iterations: Some(2),
            },
            vec![if_out],
            s.clone(),
        );
        g.set_outputs(vec![w_out]);

        let lowered = LowerControlFlow.run(g);
        let has_if = lowered
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::If { .. }));
        let has_while = lowered
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::While { .. }));
        assert!(
            !has_if && !has_while,
            "LowerControlFlow should erase both If and While"
        );
        // Where introduced for If; body's Mul appears twice (N=2).
        let n_where = lowered
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Where))
            .count();
        let n_mul = lowered
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
            .count();
        assert_eq!(n_where, 1, "expected 1 Where from If lowering");
        assert_eq!(n_mul, 2, "expected 2 Mul from While unroll (N=2)");
    }
}
