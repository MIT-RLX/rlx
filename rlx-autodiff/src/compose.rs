// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Graph composition helpers for higher-order reverse-mode AD.

use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::{HashMap, VecDeque};

/// Build an `Op::Constant` filled with `1.0`, matching `like`'s dtype and rank.
///
/// True scalars (`rank == 0`) stay rank-0 — do not promote to `[1]`.
pub fn constant_one(like: &Shape) -> Op {
    let n = like.num_elements().unwrap_or(1).max(1);
    let data = match like.dtype() {
        DType::F32 => vec![1.0f32; n]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F64 => vec![1.0f64; n]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F16 => (0..n)
            .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
            .collect(),
        DType::BF16 => (0..n)
            .flat_map(|_| half::bf16::from_f32(1.0).to_le_bytes())
            .collect(),
        other => panic!("constant_one: unsupported dtype {other:?}"),
    };
    Op::Constant { data }
}

/// Build a zero constant with the same shape/dtype as `like`.
pub fn constant_zero(like: &Shape) -> Op {
    let n = like.num_elements().unwrap_or(1).max(1);
    let data = match like.dtype() {
        DType::F32 => vec![0.0f32; n]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F64 => vec![0.0f64; n]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::F16 => vec![0u8; 2 * n],
        DType::BF16 => vec![0u8; 2 * n],
        other => panic!("constant_zero: unsupported dtype {other:?}"),
    };
    Op::Constant { data }
}

/// Replace the `"d_output"` seed input with an in-place `Constant(1.0)`.
///
/// In-place rewrite preserves insertion order so downstream `prepare_graph_for_ad`
/// passes can remap nodes without hitting missing `id_map` entries.
pub fn internalize_d_output(g: &mut Graph) {
    let d_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id);
    let Some(d_id) = d_id else {
        return;
    };
    let shape = g.node(d_id).shape.clone();
    g.node_mut(d_id).op = constant_one(&shape);
    g.node_mut(d_id).inputs.clear();
}

/// Find an `Op::Input` or `Op::Param` node by name.
pub fn find_input_by_name(g: &Graph, name: &str) -> Option<NodeId> {
    g.nodes().iter().find_map(|n| match &n.op {
        Op::Input { name: n_name } | Op::Param { name: n_name } if n_name == name => Some(n.id),
        _ => None,
    })
}

/// Whether `output` depends on `wrt` through a differentiable path.
///
/// Graph reachability alone is insufficient: piecewise activations (ReLU/Abs)
/// routed only through `Op::Compare` would otherwise look reachable and panic
/// at 3rd order. This walk skips non-differentiable edges.
pub fn output_depends_on_differentiable(g: &Graph, output: NodeId, wrt: NodeId) -> bool {
    if output == wrt {
        return true;
    }
    let mut seen = std::collections::HashSet::new();
    let mut q = VecDeque::from([output]);
    while let Some(id) = q.pop_front() {
        if id == wrt {
            return true;
        }
        if !seen.insert(id) {
            continue;
        }
        let node = g.node(id);
        let inputs = diff_inputs(&node.op, &node.inputs);
        for &inp in inputs {
            q.push_back(inp);
        }
    }
    false
}

/// Inputs that can carry gradients w.r.t. continuous variables.
fn diff_inputs<'a>(op: &'a Op, inputs: &'a [NodeId]) -> &'a [NodeId] {
    match op {
        Op::Compare(_)
        | Op::TopK { .. }
        | Op::Sample { .. }
        | Op::RngNormal { .. }
        | Op::RngUniform { .. } => &[],
        Op::Where => {
            if inputs.len() >= 3 {
                &inputs[1..3]
            } else {
                &[]
            }
        }
        Op::Cast { to } if !to.is_float() => &[],
        _ => inputs,
    }
}

/// Structural-equality CSE for pure (non-leaf) ops.
pub fn cse(g: Graph) -> Graph {
    let mut out = Graph::new(format!("{}_cse", g.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut cache: HashMap<String, NodeId> = HashMap::new();

    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        if is_cse_eligible(&node.op) {
            let key = format!("{:?}|{:?}|{:?}", node.op, new_inputs, node.shape);
            if let Some(&existing) = cache.get(&key) {
                id_map.insert(node.id, existing);
                continue;
            }
            let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
            id_map.insert(node.id, new_id);
            cache.insert(key, new_id);
            continue;
        }
        let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|o| id_map[o]).collect());
    out
}

fn is_cse_eligible(op: &Op) -> bool {
    !matches!(
        op,
        Op::Input { .. } | Op::Param { .. } | Op::Scan { .. } | Op::ScanBackward { .. }
    )
}

/// Scalar-zero graph: `output = 0` with a single input `wrt_name`.
pub fn zero_derivative_graph(name: &str, wrt_name: &str, dtype: DType) -> Graph {
    let mut g = Graph::new(name);
    let wrt = g.input(wrt_name, Shape::scalar(dtype));
    let zero = g.add_node(
        constant_zero(&Shape::scalar(dtype)),
        vec![],
        Shape::scalar(dtype),
    );
    g.set_outputs(vec![zero]);
    let _ = wrt;
    g
}

/// Expand a rank-0 or rank-1 scalar constant to `target` via `Expand` when needed.
pub fn broadcast_scalar(g: &mut Graph, scalar: NodeId, target: &Shape) -> NodeId {
    let s = g.node(scalar).shape.clone();
    if s == *target {
        return scalar;
    }
    let target_i64: Vec<i64> = target
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    let mut node = scalar;
    if s.rank() == 0 && target.rank() > 0 {
        node = g.add_node(
            Op::Reshape {
                new_shape: vec![1; target.rank()],
            },
            vec![scalar],
            Shape::from_dims(&vec![Dim::Static(1); target.rank()], s.dtype()),
        );
    } else if s.rank() < target.rank() {
        // Row vector `[N]` → matrix `[N, C]` must become `[N, 1]` (trailing
        // ones), not `[1, N]` — otherwise Expand walks OOB on the buffer.
        let align_row = s.rank() == 1 && target.rank() >= 2 && s.dim(0) == target.dim(0);
        let padded: Vec<Dim> = if align_row {
            let mut dims = s.dims().to_vec();
            while dims.len() < target.rank() {
                dims.push(Dim::Static(1));
            }
            dims
        } else {
            let mut dims = s.dims().to_vec();
            while dims.len() < target.rank() {
                dims.insert(0, Dim::Static(1));
            }
            dims
        };
        let padded_shape = Shape::from_dims(&padded, s.dtype());
        if s != padded_shape {
            let new_shape: Vec<i64> = padded
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => *n as i64,
                    Dim::Dynamic(_) => -1,
                })
                .collect();
            node = g.add_node(Op::Reshape { new_shape }, vec![node], padded_shape);
        }
    }
    g.add_node(
        Op::Expand {
            target_shape: target_i64,
        },
        vec![node],
        target.clone(),
    )
}

fn is_scalar_const_source(g: &Graph, id: NodeId) -> bool {
    match &g.node(id).op {
        Op::Constant { .. } => g.node(id).shape.num_elements() == Some(1),
        Op::Reshape { .. } => {
            g.node(id).shape.num_elements() == Some(1)
                && is_scalar_const_source(g, g.node(id).inputs[0])
        }
        _ => false,
    }
}

fn peel_eligible_consumer(op: &Op) -> bool {
    matches!(
        op,
        Op::Activation(_)
            | Op::Binary(_)
            | Op::Cast { .. }
            | Op::Compare(_)
            | Op::Where
            | Op::Softmax { .. }
            | Op::Reduce { .. }
            | Op::Reshape { .. }
    )
}

/// Drop `Expand(constant)` when the sole consumer can broadcast the scalar.
///
/// Enables `MarkElementwiseRegions` to fuse stacked grad chains that would
/// otherwise alternate tiny `Expand` + elementwise ops on Metal.
pub fn peel_scalar_expands(g: Graph) -> Graph {
    let mut g = g;
    loop {
        let before = count_expands(&g);
        g = peel_scalar_expands_once(g);
        if count_expands(&g) >= before {
            break;
        }
    }
    g
}

fn count_expands(g: &Graph) -> usize {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Expand { .. }))
        .count()
}

fn peel_scalar_expands_once(g: Graph) -> Graph {
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in g.nodes() {
        for &inp in &node.inputs {
            *consumers.entry(inp).or_insert(0) += 1;
        }
    }

    let mut peel: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let Op::Expand { .. } = &node.op else {
            continue;
        };
        if consumers.get(&node.id).copied().unwrap_or(0) != 1 {
            continue;
        }
        let src = node.inputs[0];
        if !is_scalar_const_source(&g, src) {
            continue;
        }
        let Some(consumer) = g.nodes().iter().find(|n| n.inputs.contains(&node.id)) else {
            continue;
        };
        if !peel_eligible_consumer(&consumer.op) {
            continue;
        }
        let out_elems = consumer.shape.num_elements();
        let src_elems = g.node(src).shape.num_elements();
        let ok = match (src_elems, out_elems) {
            (Some(s), Some(o)) if s == o => true,
            (Some(s), Some(o)) if s > 0 && o % s == 0 => true,
            _ => false,
        };
        if ok {
            peel.insert(node.id, src);
        }
    }

    if peel.is_empty() {
        return g;
    }

    let mut out = Graph::new(format!("{}_peel", g.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        if peel.contains_key(&node.id) {
            id_map.insert(node.id, id_map[&peel[&node.id]]);
            continue;
        }
        let new_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| {
                if let Some(&src) = peel.get(i) {
                    id_map[&src]
                } else {
                    id_map[i]
                }
            })
            .collect();
        let new_id = out.add_node(node.op.clone(), new_inputs, node.shape.clone());
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|o| id_map[o]).collect());
    out
}

/// Merge `sub` into `base`, wiring named inputs to existing `base` nodes.
///
/// Returns `sub` node id → id in `base`. Outputs of `sub` can be remapped via this map.
pub fn merge_subgraph(
    base: &mut Graph,
    sub: &Graph,
    bind_inputs: &std::collections::HashMap<String, NodeId>,
) -> std::collections::HashMap<NodeId, NodeId> {
    let mut id_map = std::collections::HashMap::new();
    for node in sub.nodes() {
        if let Op::Input { name } | Op::Param { name } = &node.op {
            if let Some(&ext) = bind_inputs.get(name) {
                id_map.insert(node.id, ext);
                continue;
            }
        }
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = base.add_node(node.op.clone(), inputs, node.shape.clone());
        if let Some(inferred) = rlx_ir::infer_shape::infer_output_shape(base, base.node(new_id)) {
            base.node_mut(new_id).shape = inferred;
        }
        id_map.insert(node.id, new_id);
    }
    id_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Graph, Op, Shape};

    #[test]
    fn peel_scalar_expands_drops_redundant_expand() {
        let mut g = Graph::new("peel");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[4, 4], f));
        let one = g.add_node(constant_one(&Shape::scalar(f)), vec![], Shape::scalar(f));
        let expanded = broadcast_scalar(&mut g, one, &Shape::new(&[4, 4], f));
        let y = g.binary(BinaryOp::Add, x, expanded, Shape::new(&[4, 4], f));
        g.set_outputs(vec![y]);
        let before = g.nodes().len();
        let peeled = peel_scalar_expands(g);
        assert!(
            peeled
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::Expand { .. })),
            "expected Expand nodes removed"
        );
        assert!(peeled.nodes().len() < before);
    }

    #[test]
    fn peel_scalar_expands_before_softmax() {
        let mut g = Graph::new("peel_softmax");
        let f = DType::F32;
        let _x = g.input("x", Shape::new(&[4, 4], f));
        let one = g.add_node(constant_one(&Shape::scalar(f)), vec![], Shape::scalar(f));
        let expanded = broadcast_scalar(&mut g, one, &Shape::new(&[4, 4], f));
        let y = g.sm(expanded, -1);
        g.set_outputs(vec![y]);
        let peeled = peel_scalar_expands(g);
        assert!(
            peeled
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::Expand { .. })),
            "expected Expand removed before Softmax"
        );
    }

    #[test]
    fn broadcast_scalar_row_vector_to_matrix() {
        let f = DType::F32;
        let mut g = Graph::new("row_bcast");
        let row = g.input("row", Shape::new(&[3], f));
        let out = broadcast_scalar(&mut g, row, &Shape::new(&[3, 4], f));
        g.set_outputs(vec![out]);
        let expand = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Expand { .. }))
            .expect("expand");
        let expand_in = g.node(expand.inputs[0]);
        assert_eq!(expand_in.shape.dims(), &[Dim::Static(3), Dim::Static(1)]);
    }
}
