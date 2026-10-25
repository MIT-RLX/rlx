// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Remove transposes that cancel.
//!
//! Layout conversions come in pairs: an importer (ONNX from TensorFlow, a
//! channels-last front end) or a backend that prefers another layout wraps an
//! op in `Transpose(p)` … `Transpose(p⁻¹)`, and between two such ops the inner
//! pair is pure data movement. This pass removes it, by four local rules applied
//! to a fixed point:
//!
//! * **identity** — `Transpose([0, 1, …])` is its input;
//! * **compose** — `Transpose(q)(Transpose(p)(x))` is `Transpose(p∘q)(x)`, and
//!   vanishes when that is the identity;
//! * **constant** — a transpose of a constant is a constant;
//! * **sink** — `Transpose(q)` over a single-use elementwise op whose operands
//!   are all `Transpose(p)` (or scalars) moves below that op as `Transpose(p∘q)`
//!   on each operand, which the compose rule then cancels.
//!
//! The sink rule only fires when it makes a transpose meet another one, so it
//! never adds work. Shapes are unchanged at every graph output.

use std::collections::HashMap;

use rlx_fusion::pass::{IRStatus, Pass, PassResult};
use rlx_ir::{Graph, NodeId, Op, Shape};

pub struct SimplifyTransposes;

impl Pass for SimplifyTransposes {
    fn name(&self) -> &str {
        "simplify_transposes"
    }

    fn run(&self, graph: Graph) -> Graph {
        self.run_with_status(graph).graph
    }

    fn run_with_status(&self, mut graph: Graph) -> PassResult {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Transpose { .. }))
        {
            return PassResult::unchanged(graph);
        }
        let mut changed = false;
        for _ in 0..8 {
            let (g, fired) = one_round(&graph);
            graph = g;
            if !fired {
                break;
            }
            changed = true;
        }
        if !changed {
            return PassResult::unchanged(graph);
        }
        let graph = crate::DeadCodeElimination.run(graph);
        PassResult::from_status(graph, IRStatus::Changed)
    }
}

/// `out[i] = in[perm[i]]` composed: applying `inner` then `outer` is one
/// transpose by `inner[outer[i]]`.
fn compose(inner: &[usize], outer: &[usize]) -> Vec<usize> {
    outer.iter().map(|&i| inner[i]).collect()
}

fn is_identity(p: &[usize]) -> bool {
    p.iter().enumerate().all(|(i, &d)| i == d)
}

/// Elementwise ops a transpose commutes with.
fn is_elementwise(op: &Op) -> bool {
    matches!(
        op,
        Op::Activation(_) | Op::Binary(_) | Op::Compare(_) | Op::Where | Op::Cast { .. }
    )
}

/// A rank-0 / single-element operand broadcasts the same way in any layout.
fn is_scalar(s: &Shape) -> bool {
    s.num_elements() == Some(1)
}

fn permuted_shape(s: &Shape, perm: &[usize]) -> Shape {
    let dims = s.dims();
    let new: Vec<_> = perm.iter().map(|&p| dims[p]).collect();
    Shape::from_dims(&new, s.dtype())
}

/// Permute a constant's row-major element bytes.
fn permute_bytes(data: &[u8], shape: &Shape, perm: &[usize]) -> Option<Vec<u8>> {
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .map(|d| d.is_static().then(|| d.unwrap_static()))
        .collect::<Option<_>>()?;
    let n: usize = dims.iter().product();
    let esz = shape.dtype().size_bytes();
    if n == 0 || data.len() != n * esz {
        return None;
    }
    let rank = dims.len();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * dims[i + 1];
    }
    let out_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let mut out = vec![0u8; data.len()];
    let mut idx = vec![0usize; rank];
    for o in 0..n {
        let src: usize = (0..rank).map(|i| idx[i] * in_strides[perm[i]]).sum();
        out[o * esz..(o + 1) * esz].copy_from_slice(&data[src * esz..(src + 1) * esz]);
        for i in (0..rank).rev() {
            idx[i] += 1;
            if idx[i] < out_dims[i] {
                break;
            }
            idx[i] = 0;
        }
    }
    Some(out)
}

/// One forward rebuild applying the rules; `true` if any fired.
fn one_round(g: &Graph) -> (Graph, bool) {
    let mut uses: HashMap<NodeId, usize> = HashMap::new();
    for n in g.nodes() {
        for &i in &n.inputs {
            *uses.entry(i).or_default() += 1;
        }
    }
    for &o in &g.outputs {
        *uses.entry(o).or_default() += 1;
    }

    let mut out = Graph::new(&g.name);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    // New id → the old node it copies (absent for nodes a rule created).
    let mut rev: HashMap<NodeId, NodeId> = HashMap::new();
    let mut fired = false;

    for n in g.nodes() {
        let inputs: Vec<NodeId> = n.inputs.iter().map(|i| map[i]).collect();
        let new_id = match &n.op {
            Op::Transpose { perm } => {
                let src = inputs[0];
                let src_node = out.node(src).clone();
                if let Some(id) = simplify(&mut out, g, &uses, &rev, src, &src_node, perm) {
                    fired = true;
                    id
                } else {
                    copy(&mut out, n, inputs)
                }
            }
            _ => copy(&mut out, n, inputs),
        };
        map.insert(n.id, new_id);
        rev.entry(new_id).or_insert(n.id);
    }
    out.set_outputs(g.outputs.iter().map(|o| map[o]).collect());
    (out, fired)
}

fn copy(out: &mut Graph, n: &rlx_ir::Node, inputs: Vec<NodeId>) -> NodeId {
    let id = out.add_node(n.op.clone(), inputs, n.shape.clone());
    let m = out.node_mut(id);
    m.name = n.name.clone();
    m.origin = n.origin.clone();
    id
}

/// Transpose `src` (a node of `out`) by `perm`, applying the compose and
/// identity rules at once.
fn transpose(out: &mut Graph, src: NodeId, perm: &[usize]) -> NodeId {
    let s = out.node(src).clone();
    if is_identity(perm) {
        return src;
    }
    if let Op::Transpose { perm: inner } = &s.op {
        let p = compose(inner, perm);
        let base = s.inputs[0];
        return transpose(out, base, &p);
    }
    let shape = permuted_shape(&s.shape, perm);
    out.add_node(
        Op::Transpose {
            perm: perm.to_vec(),
        },
        vec![src],
        shape,
    )
}

fn simplify(
    out: &mut Graph,
    g: &Graph,
    uses: &HashMap<NodeId, usize>,
    rev: &HashMap<NodeId, NodeId>,
    src: NodeId,
    src_node: &rlx_ir::Node,
    perm: &[usize],
) -> Option<NodeId> {
    if is_identity(perm) {
        return Some(src);
    }
    match &src_node.op {
        Op::Transpose { .. } => Some(transpose(out, src, perm)),
        Op::Constant { data } => {
            let bytes = permute_bytes(data, &src_node.shape, perm)?;
            let shape = permuted_shape(&src_node.shape, perm);
            Some(out.add_node(Op::Constant { data: bytes }, vec![], shape))
        }
        op if is_elementwise(op) => {
            // Sink only when the whole elementwise subtree under `src` bottoms
            // out in transposes by one shared `p` (or scalars), through values
            // nothing else reads — then the moved transposes cancel or compose
            // instead of multiplying.
            sink_perm(out, g, uses, rev, src, 0)?;
            Some(sink(out, src, perm))
        }
        _ => None,
    }
}

/// The shared inner permutation if `id` can absorb a transpose: a
/// `Transpose(p)`, or a single-use elementwise op (copied from the old graph)
/// whose non-scalar operands all can, with the same `p`.
fn sink_perm(
    out: &Graph,
    g: &Graph,
    uses: &HashMap<NodeId, usize>,
    rev: &HashMap<NodeId, NodeId>,
    id: NodeId,
    depth: usize,
) -> Option<Vec<usize>> {
    let n = out.node(id);
    if let Op::Transpose { perm } = &n.op {
        return Some(perm.clone());
    }
    if depth >= 8 || !is_elementwise(&n.op) {
        return None;
    }
    let old = *rev.get(&id)?;
    if uses.get(&old).copied().unwrap_or(0) != 1 || g.outputs.contains(&old) {
        return None;
    }
    let mut shared: Option<Vec<usize>> = None;
    for &i in &n.inputs {
        let opnd = out.node(i);
        if is_scalar(&opnd.shape) {
            continue;
        }
        if opnd.shape.dims() != n.shape.dims() {
            return None;
        }
        let p = sink_perm(out, g, uses, rev, i, depth + 1)?;
        match &shared {
            None => shared = Some(p),
            Some(q) if *q == p => {}
            Some(_) => return None,
        }
    }
    shared
}

/// `Transpose(perm)` applied to `id` by pushing it to the leaves
/// [`sink_perm`] accepted.
fn sink(out: &mut Graph, id: NodeId, perm: &[usize]) -> NodeId {
    let n = out.node(id).clone();
    if matches!(n.op, Op::Transpose { .. }) {
        return transpose(out, id, perm);
    }
    let inputs: Vec<NodeId> = n
        .inputs
        .iter()
        .map(|&i| {
            if is_scalar(&out.node(i).shape) {
                i
            } else {
                sink(out, i, perm)
            }
        })
        .collect();
    let shape = permuted_shape(&n.shape, perm);
    out.add_node(n.op.clone(), inputs, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, OpKind};

    const F: DType = DType::F32;

    fn count(g: &Graph, k: OpKind) -> usize {
        g.nodes().iter().filter(|n| n.op.kind() == k).count()
    }

    fn t(g: &mut Graph, x: NodeId, perm: &[usize]) -> NodeId {
        let s = permuted_shape(&g.node(x).shape, perm);
        g.add_node(
            Op::Transpose {
                perm: perm.to_vec(),
            },
            vec![x],
            s,
        )
    }

    #[test]
    fn inverse_pair_cancels() {
        let mut g = Graph::new("pair");
        let x = g.input("x", Shape::new(&[1, 3, 4, 5], F));
        let a = t(&mut g, x, &[0, 2, 3, 1]); // NCHW → NHWC
        let b = t(&mut g, a, &[0, 3, 1, 2]); // NHWC → NCHW
        let y = g.add_node(
            Op::Activation(Activation::Relu),
            vec![b],
            Shape::new(&[1, 3, 4, 5], F),
        );
        g.set_outputs(vec![y]);
        let r = SimplifyTransposes.run(g);
        assert_eq!(count(&r, OpKind::Transpose), 0);
        assert_eq!(
            r.node(r.node(r.outputs[0]).inputs[0]).op.kind(),
            OpKind::Input
        );
    }

    #[test]
    fn chain_composes_to_one() {
        let mut g = Graph::new("chain");
        let x = g.input("x", Shape::new(&[2, 3, 4], F));
        let a = t(&mut g, x, &[1, 0, 2]);
        let b = t(&mut g, a, &[0, 2, 1]);
        g.set_outputs(vec![b]);
        let r = SimplifyTransposes.run(g);
        assert_eq!(count(&r, OpKind::Transpose), 1);
        let Op::Transpose { perm } = &r.node(r.outputs[0]).op else {
            panic!()
        };
        // a = x[1,0,2] → dims (3,2,4); b = a[0,2,1] → (3,4,2) = x dims (1,2,0).
        assert_eq!(perm, &vec![1, 2, 0]);
        assert_eq!(
            r.node(r.outputs[0]).shape.dims(),
            Shape::new(&[3, 4, 2], F).dims()
        );
    }

    #[test]
    fn sinks_through_elementwise_to_cancel() {
        // T⁻¹(relu(T(x)) + T(y)) == relu(x) + y
        let mut g = Graph::new("sink");
        let x = g.input("x", Shape::new(&[1, 3, 4, 5], F));
        let yv = g.input("y", Shape::new(&[1, 3, 4, 5], F));
        let tx = t(&mut g, x, &[0, 2, 3, 1]);
        let ty = t(&mut g, yv, &[0, 2, 3, 1]);
        let s = g.node(tx).shape.clone();
        let r = g.add_node(Op::Activation(Activation::Relu), vec![tx], s.clone());
        let a = g.binary(BinaryOp::Add, r, ty, s);
        let back = t(&mut g, a, &[0, 3, 1, 2]);
        g.set_outputs(vec![back]);
        let out = SimplifyTransposes.run(g);
        assert_eq!(count(&out, OpKind::Transpose), 0, "{out:?}");
    }

    #[test]
    fn does_not_sink_through_a_shared_value() {
        let mut g = Graph::new("shared");
        let x = g.input("x", Shape::new(&[2, 3], F));
        let tx = t(&mut g, x, &[1, 0]);
        let s = g.node(tx).shape.clone();
        let r = g.add_node(Op::Activation(Activation::Relu), vec![tx], s);
        let back = t(&mut g, r, &[1, 0]);
        g.set_outputs(vec![back, r]);
        let out = SimplifyTransposes.run(g);
        assert_eq!(
            count(&out, OpKind::Activation),
            1,
            "relu must not be duplicated"
        );
    }

    #[test]
    fn folds_constant() {
        let mut g = Graph::new("const");
        let data: Vec<u8> = (0..6u32).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let c = g.add_node(Op::Constant { data }, vec![], Shape::new(&[2, 3], F));
        let tc = t(&mut g, c, &[1, 0]);
        g.set_outputs(vec![tc]);
        let out = SimplifyTransposes.run(g);
        let n = out.node(out.outputs[0]);
        let Op::Constant { data } = &n.op else {
            panic!("not folded: {:?}", n.op)
        };
        let v: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(v, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn no_transpose_is_unchanged() {
        let mut g = Graph::new("none");
        let x = g.input("x", Shape::new(&[2], F));
        let y = g.add_node(
            Op::Activation(Activation::Relu),
            vec![x],
            Shape::new(&[2], F),
        );
        g.set_outputs(vec![y]);
        assert_eq!(
            SimplifyTransposes.run_with_status(g).ir_changed,
            IRStatus::Unchanged
        );
    }
}
