// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The **pre-head representation** of a model graph, found from the graph.
//!
//! Classifier graphs end the same way whatever the architecture: the
//! representation passes through one parametric linear op — a `MatMul`, or a
//! `Conv` whose kernel covers the remaining extent — then bias adds, reshapes
//! and squeezes. A caller that wants that representation (a frozen-encoder
//! probe, a similarity search, transfer to another task) needs no per-model
//! head names to get it:
//!
//! * [`input_dependent`] — which nodes depend on a graph `Input` (the
//!   activation side) versus only on params and constants (the weight side, so
//!   a transpose or max-norm applied to a weight is never mistaken for data).
//! * [`pre_heads`] — walk back from an output to the last `MatMul`/`Conv` and
//!   return its activation operand; through combining ops (`Add`, `Sub`, `Max`,
//!   `Concat`) whose operands each end in their own head (ATCNet sums a dense
//!   head per sliding window) it returns one node per head. Any other fork is an
//!   error naming the op, never a guess.
//! * [`expose_pre_heads`] — append those nodes to the graph's outputs, so one
//!   run returns `[outputs.., features..]`.
//! * [`head_slice`] — the head alone as a standalone graph whose inputs are the
//!   features; run it with the model's params and it must give the logits.
//! * [`freeze_node`] — replace a node by a constant tensor. Doing it to the
//!   pre-head nodes leaves a graph that computes the head alone from injected
//!   features, which is how a caller *proves* the exposed tensor is the head's
//!   input: the logits must come back.

use crate::op::BinaryOp;
use crate::{Graph, NodeId, Op};

/// Why a pre-head node could not be found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadError(pub String);

impl std::fmt::Display for HeadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HeadError {}

fn err(msg: String) -> HeadError {
    HeadError(msg)
}

/// A parametric linear op a classifier head is built from.
pub fn is_head_op(op: &Op) -> bool {
    matches!(
        op,
        Op::MatMul | Op::Conv { .. } | Op::ConvTranspose2d { .. } | Op::Conv3d { .. }
    )
}

fn is_combining(op: &Op) -> bool {
    matches!(
        op,
        Op::Binary(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Max) | Op::Concat { .. }
    )
}

/// For each node, whether it depends (transitively) on a graph `Input`.
/// Indexed by `NodeId.0`; relies on the graph's construction-order topology.
pub fn input_dependent(g: &Graph) -> Vec<bool> {
    let mut dep = vec![false; g.len()];
    for n in g.nodes() {
        let i = n.id.0 as usize;
        dep[i] = matches!(n.op, Op::Input { .. }) || n.inputs.iter().any(|x| dep[x.0 as usize]);
    }
    dep
}

/// Every pre-head node feeding `out`, in operand order (one per head).
pub fn pre_heads(g: &Graph, out: NodeId) -> Result<Vec<NodeId>, HeadError> {
    let dep = input_dependent(g);
    let mut found = Vec::new();
    walk(g, &dep, out, &mut found)?;
    let mut seen = std::collections::HashSet::new();
    found.retain(|n| seen.insert(*n));
    Ok(found)
}

/// The single pre-head node of `out`; an error if the logits combine heads.
pub fn pre_head(g: &Graph, out: NodeId) -> Result<NodeId, HeadError> {
    match pre_heads(g, out)?.as_slice() {
        [one] => Ok(*one),
        many => Err(err(format!(
            "the logits combine {} heads; use pre_heads",
            many.len()
        ))),
    }
}

fn walk(g: &Graph, dep: &[bool], start: NodeId, found: &mut Vec<NodeId>) -> Result<(), HeadError> {
    let mut cur = start;
    for _ in 0..g.len() {
        let n = g.node(cur);
        let act: Vec<NodeId> = n
            .inputs
            .iter()
            .copied()
            .filter(|x| dep[x.0 as usize])
            .collect();
        if is_head_op(&n.op) {
            return match act.as_slice() {
                [a] => {
                    found.push(*a);
                    Ok(())
                }
                [] => Err(err(format!(
                    "head op {:?} at {cur:?} has no input-dependent operand",
                    n.op
                ))),
                _ => Err(err(format!(
                    "head op {:?} at {cur:?} has {} input-dependent operands (attention-like), \
                     so there is no single weight-side head to strip",
                    n.op,
                    act.len()
                ))),
            };
        }
        match act.as_slice() {
            [a] => cur = *a,
            [] => {
                return Err(err(format!(
                    "reached {:?} at {cur:?} without passing a MatMul/Conv head",
                    n.op
                )));
            }
            many if is_combining(&n.op) => {
                for a in many {
                    walk(g, dep, *a, found)?;
                }
                return Ok(());
            }
            _ => {
                return Err(err(format!(
                    "the path forks at {:?} ({cur:?}): {} input-dependent operands, not a combination of heads",
                    n.op,
                    act.len()
                )));
            }
        }
    }
    Err(err(format!("no head op found walking back from {start:?}")))
}

/// Append the pre-head node(s) of `outputs[0]` to the outputs. Returns them.
pub fn expose_pre_heads(g: &mut Graph) -> Result<Vec<NodeId>, HeadError> {
    let logits = *g
        .outputs
        .first()
        .ok_or_else(|| err("graph has no outputs".into()))?;
    let feats = pre_heads(g, logits)?;
    let mut outs = g.outputs.clone();
    outs.extend(feats.iter().copied());
    g.set_outputs(outs);
    Ok(feats)
}

/// Replace `node` by a constant holding `data` (the node's own shape),
/// cutting its inputs. Everything upstream of it becomes dead for this node's
/// consumers. The bytes are written in the node's OWN dtype — an f64 graph
/// (the SPD-manifold models run in f64) given f32 bytes would read garbage
/// rather than fail.
pub fn freeze_node(g: &mut Graph, node: NodeId, data: &[f32]) {
    let bytes: Vec<u8> = match g.shape(node).dtype() {
        crate::DType::F64 => data
            .iter()
            .flat_map(|x| (*x as f64).to_le_bytes())
            .collect(),
        _ => data.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    g.node_mut(node).op = Op::Constant { data: bytes };
    g.set_inputs(node, vec![]);
}

/// The head alone as a standalone graph: one `Input` per pre-head node
/// (`features_0`, `features_1`, … — their shapes), plus copies of exactly the
/// nodes between those inputs and `out`, with their params and constants.
/// Run it with the model's own params and the features, and it computes the
/// logits — the cleanest proof that the features are the head's input.
///
/// Fails if the head also reads a graph input other than the features (then
/// it is not a function of them alone, e.g. a skip connection around it).
/// Returns the graph and its input names, in `features` order.
pub fn head_slice(
    g: &Graph,
    features: &[NodeId],
    out: NodeId,
) -> Result<(Graph, Vec<String>), HeadError> {
    let mut h = Graph::new(format!("{}.head", g.name));
    let mut map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let mut names = Vec::with_capacity(features.len());
    for (i, f) in features.iter().enumerate() {
        let name = format!("features_{i}");
        let id = h.push(
            Op::Input { name: name.clone() },
            vec![],
            g.shape(*f).clone(),
            None,
        );
        map.insert(*f, id);
        names.push(name);
    }
    // iterative post-order copy, so a deep head cannot overflow the stack
    let mut stack: Vec<(NodeId, bool)> = vec![(out, false)];
    while let Some((n, expanded)) = stack.pop() {
        if map.contains_key(&n) {
            continue;
        }
        let node = g.node(n);
        if let Op::Input { name } = &node.op {
            return Err(err(format!(
                "the head reads graph input `{name}` besides the features, so it is not a function of them alone"
            )));
        }
        if !expanded {
            stack.push((n, true));
            for i in node.inputs.iter().rev() {
                if !map.contains_key(i) {
                    stack.push((*i, false));
                }
            }
            continue;
        }
        let inputs = node.inputs.iter().map(|i| map[i]).collect();
        let id = h.push(
            node.op.clone(),
            inputs,
            node.shape.clone(),
            node.name.clone(),
        );
        map.insert(n, id);
    }
    h.set_outputs(vec![map[&out]]);
    Ok((h, names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, GraphExt, Shape};

    fn mlp() -> (Graph, NodeId) {
        let mut g = Graph::new("mlp");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w1 = g.param("w1", Shape::new(&[4, 3], DType::F32));
        let h = g.matmul(x, w1, Shape::new(&[2, 3], DType::F32));
        let h = g.relu(h);
        let w2 = g.param("w2t", Shape::new(&[5, 3], DType::F32));
        let w2 = g.transpose_(w2, vec![1, 0]);
        let y = g.matmul(h, w2, Shape::new(&[2, 5], DType::F32));
        let b = g.param("b", Shape::new(&[5], DType::F32));
        let y = g.add(y, b);
        g.set_outputs(vec![y]);
        (g, h)
    }

    #[test]
    fn the_weight_side_transpose_is_not_the_activation_path() {
        let (g, h) = mlp();
        assert_eq!(pre_head(&g, g.outputs[0]).unwrap(), h);
    }

    #[test]
    fn summed_heads_return_one_node_per_head() {
        let mut g = Graph::new("two");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let wa = g.param("wa", Shape::new(&[4, 3], DType::F32));
        let ha = g.matmul(x, wa, Shape::new(&[2, 3], DType::F32));
        let wb = g.param("wb", Shape::new(&[4, 2], DType::F32));
        let hb = g.matmul(x, wb, Shape::new(&[2, 2], DType::F32));
        let ka = g.param("ka", Shape::new(&[3, 5], DType::F32));
        let kb = g.param("kb", Shape::new(&[2, 5], DType::F32));
        let ya = g.matmul(ha, ka, Shape::new(&[2, 5], DType::F32));
        let yb = g.matmul(hb, kb, Shape::new(&[2, 5], DType::F32));
        let y = g.add(ya, yb);
        g.set_outputs(vec![y]);
        assert_eq!(pre_heads(&g, y).unwrap(), vec![ha, hb]);
        assert!(pre_head(&g, y).is_err());
    }

    #[test]
    fn a_non_combining_fork_is_refused() {
        let mut g = Graph::new("fork");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let y = g.input("y", Shape::new(&[2, 4], DType::F32));
        let z = g.mul(x, y);
        g.set_outputs(vec![z]);
        assert!(pre_heads(&g, z).unwrap_err().0.contains("forks"));
    }

    #[test]
    fn expose_and_freeze() {
        let (mut g, h) = mlp();
        assert_eq!(expose_pre_heads(&mut g).unwrap(), vec![h]);
        assert_eq!(g.outputs.len(), 2);
        freeze_node(&mut g, h, &[0.0; 6]);
        assert!(matches!(g.node(h).op, Op::Constant { .. }) && g.node(h).inputs.is_empty());
    }

    #[test]
    fn head_slice_keeps_only_the_head() {
        let (g, h) = mlp();
        let y = g.outputs[0];
        let (hs, names) = head_slice(&g, &[h], y).unwrap();
        assert_eq!(names, vec!["features_0".to_string()]);
        // features input, w2t, transpose, matmul, b, add — not x, w1, the first matmul or relu
        assert_eq!(hs.len(), 6, "{hs}");
        assert!(
            hs.nodes()
                .iter()
                .all(|n| !matches!(&n.op, Op::Input { name } if name == "x"))
        );
        assert!(
            hs.nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Param { name } if name == "w2t"))
        );
        assert_eq!(hs.shape(hs.outputs[0]), g.shape(y));
    }

    #[test]
    fn head_slice_refuses_a_head_that_reads_another_input() {
        let mut g = Graph::new("skip");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        let w = g.param("w", Shape::new(&[3, 3], DType::F32));
        let h = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let y = g.add(h, x); // skip connection around the "head"
        g.set_outputs(vec![y]);
        let e = head_slice(&g, &[h], y).unwrap_err();
        assert!(e.0.contains("graph input `x`"), "{e}");
    }

    #[test]
    fn a_frozen_f64_node_gets_f64_bytes() {
        let mut g = Graph::new("f64");
        let x = g.input("x", Shape::new(&[1, 2], DType::F64));
        let w = g.param("w", Shape::new(&[2, 2], DType::F64));
        let h = g.matmul(x, w, Shape::new(&[1, 2], DType::F64));
        g.set_outputs(vec![h]);
        freeze_node(&mut g, x, &[1.5, -2.0]);
        let Op::Constant { data } = &g.node(x).op else {
            panic!()
        };
        assert_eq!(data.len(), 16, "two f64 values");
        assert_eq!(f64::from_le_bytes(data[..8].try_into().unwrap()), 1.5);
    }
}
