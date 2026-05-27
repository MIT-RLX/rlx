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
use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
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
/// inlines `cond` and `body` with all loop-carried captures, then
/// applies `Where(active, body_out, carried)` per carry (MLX semantics).
pub fn unroll_while(g: Graph) -> Graph {
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    let scalar_f32 = Shape::new(&[1], DType::F32);

    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::While {
                cond,
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
                let one = out.add_node(
                    Op::Constant {
                        data: 1.0_f32.to_le_bytes().to_vec(),
                    },
                    vec![],
                    scalar_f32.clone(),
                );
                let mut active = one;
                let mut carried = new_inputs;
                for _ in 0..*n {
                    let cond_out = inline_subgraph_into(cond, &carried, &mut out);
                    let cond_f = cond_to_scalar_f32(cond_out, &mut out, &scalar_f32);
                    active = out.binary(BinaryOp::Mul, active, cond_f, scalar_f32.clone());

                    let body_outs = inline_subgraph_into_outputs(body, &carried, &mut out);
                    assert_eq!(
                        body_outs.len(),
                        carried.len(),
                        "Op::While: body output count must match loop-carried arity"
                    );
                    let mut next = Vec::with_capacity(carried.len());
                    for (body_out, &prev) in body_outs.iter().zip(carried.iter()) {
                        let shape = out.node(prev).shape.clone();
                        let mask = expand_to_shape(active, &shape, &mut out);
                        let merged = out.add_node(Op::Where, vec![mask, *body_out, prev], shape);
                        next.push(merged);
                    }
                    carried = next;
                }
                carried[0]
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

/// Fold a cond-subgraph output into a scalar f32 loop flag for `active`.
/// Vector conds are reduced with min(nonzero) so every element must be
/// truthy for the loop to keep running (matches treating 0 as false).
fn cond_to_scalar_f32(cond_out: NodeId, out: &mut Graph, scalar_f32: &Shape) -> NodeId {
    let cond_shape = out.node(cond_out).shape.clone();
    let n = cond_shape
        .dims()
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .product::<usize>();
    let as_f32 = if cond_shape.dtype() == DType::F32 {
        cond_out
    } else {
        out.add_node(
            Op::Cast { to: DType::F32 },
            vec![cond_out],
            cond_shape.with_dtype(DType::F32),
        )
    };
    if n <= 1 {
        return as_f32;
    }
    let as_f32_shape = out.node(as_f32).shape.clone();
    let rank = as_f32_shape.rank();
    let zero = out.add_node(
        Op::Constant {
            data: 0.0_f32.to_le_bytes().to_vec(),
        },
        vec![],
        scalar_f32.clone(),
    );
    let nonzero = out.add_node(
        Op::Compare(CmpOp::Ne),
        vec![as_f32, zero],
        as_f32_shape.clone().with_dtype(DType::Bool),
    );
    let nonzero_f = out.add_node(
        Op::Cast { to: DType::F32 },
        vec![nonzero],
        as_f32_shape.with_dtype(DType::F32),
    );
    let axes: Vec<usize> = (0..rank).collect();
    out.reduce(nonzero_f, ReduceOp::Min, axes, true, scalar_f32.clone())
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

/// Inline `sub` into `out`, wiring `Op::Input` slots to `captures` in
/// subgraph node order. Returns every output node (declaration order).
pub fn inline_subgraph_into_outputs(
    sub: &Graph,
    captures: &[NodeId],
    out: &mut Graph,
) -> Vec<NodeId> {
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
    assert_eq!(
        input_idx,
        captures.len(),
        "Op::While/If sub-graph: {} Op::Input nodes but {} captures",
        input_idx,
        captures.len()
    );
    sub.outputs.iter().map(|o| sub_to_parent[o]).collect()
}

/// Helper: copy `sub`'s nodes into `out`, mapping each Op::Input
/// by position to the corresponding capture. Returns the new
/// NodeId in `out` of the sub-graph's first declared output.
pub fn inline_subgraph_into(sub: &Graph, captures: &[NodeId], out: &mut Graph) -> NodeId {
    inline_subgraph_into_outputs(sub, captures, out)[0]
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
        // 1 Where from If; While unroll adds 1 Where per iteration per
        // carry (MLX semantics, see `unroll_while`).
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
        assert_eq!(
            n_where, 3,
            "expected 1 Where from If + 2 from While (N=2, 1 carry)"
        );
        assert_eq!(
            n_mul, 4,
            "expected 2 body Mul + 2 active*cond_f Mul from While (N=2)"
        );
    }

    #[test]
    fn unroll_while_multi_carry_cond_freezes_updates() {
        let v_shape = Shape::new(&[2], DType::F32);
        let s_shape = Shape::new(&[1], DType::F32);

        let mut body = Graph::new("body");
        let v_in = body.input("v", v_shape.clone());
        let s_in = body.input("s", s_shape.clone());
        let one = body.add_node(
            Op::Constant {
                data: 1.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            s_shape.clone(),
        );
        let v_out = body.binary(BinaryOp::Add, v_in, one, v_shape.clone());
        body.set_outputs(vec![v_out, s_in]);

        let mut cond = Graph::new("cond");
        let v_c = cond.input("v", v_shape.clone());
        let _s_c = cond.input("s", s_shape.clone());
        let ten = cond.add_node(
            Op::Constant {
                data: 10.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            s_shape.clone(),
        );
        let lt = cond.add_node(
            Op::Compare(rlx_ir::op::CmpOp::Lt),
            vec![v_c, ten],
            Shape::new(&[1], DType::Bool),
        );
        cond.set_outputs(vec![lt]);

        let mut g = Graph::new("parent");
        let v0 = g.input("v0", v_shape.clone());
        let s0 = g.input("s0", s_shape.clone());
        let w = g.add_node(
            Op::While {
                cond: Box::new(cond),
                body: Box::new(body),
                max_iterations: Some(3),
            },
            vec![v0, s0],
            v_shape.clone(),
        );
        g.set_outputs(vec![w]);

        let lowered = unroll_while(g);
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::While { .. })),
            "While should be erased"
        );
        let n_where = lowered
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Where))
            .count();
        assert_eq!(n_where, 6, "expected 3 iters × 2 carries Where masks");
    }
}
