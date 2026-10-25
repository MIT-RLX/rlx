// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rewrites that specialize a portable graph for one backend, keeping what
//! a package function carries (variables, symbolic attributes, derived
//! symbols).
//!
//! A package stores a function once, for every backend. What is fastest
//! differs: f16 matmul weights halve the bytes a Metal decode reads but are
//! widened every step on the CPU; attention that indexes the shared K/V
//! heads itself saves the `repeat_kv` broadcast where its kernel does that.
//! A build for a target adds the rewritten function beside the portable one
//! (`step@metal`), and the runtime runs the one for its device.

use std::collections::{HashMap, HashSet};

use crate::dynamic::SymAttr;
use crate::shape::Dim;
use crate::{DType, Graph, NodeId, Op, Shape};

/// Declare `graph`'s matmul weights `dtype` (F16 / BF16): each 2-D f32
/// param whose every use is a `MatMul`'s right operand. Norms, biases and an
/// embedding table (a `Gather`) stay f32. The runtime binds a weight in the
/// dtype its param declares. The count changed.
pub fn matmul_weights_as(graph: &mut Graph, dtype: DType) -> usize {
    let mut only_matmul_rhs: HashMap<NodeId, bool> = HashMap::new();
    for n in graph.nodes() {
        for (i, input) in n.inputs.iter().enumerate() {
            let ok = matches!(n.op, Op::MatMul) && i == 1;
            only_matmul_rhs
                .entry(*input)
                .and_modify(|v| *v &= ok)
                .or_insert(ok);
        }
    }
    let targets: Vec<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(n.op, Op::Param { .. })
                && n.shape.rank() == 2
                && n.shape.dtype() == DType::F32
                && only_matmul_rhs.get(&n.id) == Some(&true)
        })
        .map(|n| n.id)
        .collect();
    for id in &targets {
        let node = graph.node_mut(*id);
        node.shape = Shape::from_dims(node.shape.dims(), dtype);
    }
    targets.len()
}

/// Store `graph`'s embedding tables as `dtype` (F16 / BF16): each 2-D f32
/// param whose every use is the table of an axis-0 `Gather` becomes `dtype`,
/// and each of those gathers yields `dtype` rows cast back to f32 — the graph
/// computes as before, from half the table bytes (a backend that keeps the
/// table half gathers packed rows and widens only them). The count changed.
pub fn embedding_tables_as(graph: &mut Graph, dtype: DType) -> usize {
    let mut only_table: HashMap<NodeId, bool> = HashMap::new();
    for n in graph.nodes() {
        for (i, input) in n.inputs.iter().enumerate() {
            let ok = matches!(n.op, Op::Gather { axis: 0 }) && i == 0;
            only_table
                .entry(*input)
                .and_modify(|v| *v &= ok)
                .or_insert(ok);
        }
    }
    let targets: HashSet<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(n.op, Op::Param { .. })
                && n.shape.rank() == 2
                && n.shape.dtype() == DType::F32
                && only_table.get(&n.id) == Some(&true)
        })
        .map(|n| n.id)
        .collect();
    if targets.is_empty() {
        return 0;
    }
    let rebuilt = crate::stateful::rebuild(graph, |out, _, node, inputs| {
        if targets.contains(&node.id) {
            let shape = Shape::from_dims(node.shape.dims(), dtype);
            let id = out.add_node(node.op.clone(), vec![], shape);
            out.node_mut(id).name = node.name.clone();
            return Ok(Some(id));
        }
        if matches!(node.op, Op::Gather { axis: 0 })
            && node.inputs.first().is_some_and(|t| targets.contains(t))
        {
            let half = Shape::from_dims(node.shape.dims(), dtype);
            let rows = out.add_node(node.op.clone(), inputs.to_vec(), half);
            let id = out.add_node(
                Op::Cast {
                    to: node.shape.dtype(),
                },
                vec![rows],
                node.shape.clone(),
            );
            out.node_mut(id).name = node.name.clone();
            return Ok(Some(id));
        }
        Ok(None)
    });
    match rebuilt {
        Ok(g) => {
            *graph = g;
            targets.len()
        }
        Err(_) => 0,
    }
}

/// The K / V an `Attention` reads before `repeat_kv`'s broadcast:
/// `reshape([…, nkv*g*dh]) ← expand([…, nkv, g, dh]) ← reshape([…, nkv, 1,
/// dh]) ← x[…, nkv*dh]`, so `x`.
fn before_repeat_kv(g: &Graph, id: NodeId) -> Option<NodeId> {
    let flat = g.node(id);
    let Op::Reshape { .. } = flat.op else {
        return None;
    };
    let wide = g.node(*flat.inputs.first()?);
    let Op::Expand { .. } = wide.op else {
        return None;
    };
    let split = g.node(*wide.inputs.first()?);
    let Op::Reshape { .. } = split.op else {
        return None;
    };
    let x = *split.inputs.first()?;
    let (sd, wd, xd) = (
        split.shape.dims(),
        wide.shape.dims(),
        g.node(x).shape.dims(),
    );
    let r = sd.len();
    // `[…, nkv, 1, dh]` → `[…, nkv, g, dh]`, `x` the same leading dims and
    // `nkv*dh` wide.
    if r < 3 || wd.len() != r || xd.len() != r - 2 {
        return None;
    }
    let same_lead = sd[..r - 3] == wd[..r - 3] && sd[..r - 3] == xd[..r - 3];
    let broadcast = sd[r - 2] == Dim::Static(1)
        && sd[r - 3] == wd[r - 3]
        && sd[r - 1] == wd[r - 1]
        && matches!(wd[r - 2], Dim::Static(g) if g > 1);
    let width = match (sd[r - 3], sd[r - 1], xd[r - 3]) {
        (Dim::Static(nkv), Dim::Static(dh), Dim::Static(w)) => nkv * dh == w,
        _ => false,
    };
    (same_lead && broadcast && width).then_some(x)
}

/// Attention reading the shared K / V heads itself: each `Attention` whose
/// K and V come through `repeat_kv`'s broadcast reads them from before it,
/// and the broadcasts go. For backends whose attention indexes a query
/// head's K/V head (CPU, Metal); the count rewired.
pub fn gqa_native_attention(graph: &Graph) -> Result<(Graph, usize), String> {
    let mut rewired = 0;
    let g = crate::stateful::rebuild(graph, |out, _, node, inputs| {
        let Op::Attention { .. } = node.op else {
            return Ok(None);
        };
        let (Some(k), Some(v)) = (
            before_repeat_kv(graph, node.inputs[1]),
            before_repeat_kv(graph, node.inputs[2]),
        ) else {
            return Ok(None);
        };
        // `inputs` are the rebuilt ids; the K / V sources come before the
        // attention, so the map has them: find them by walking the new graph
        // the same way.
        let (Some(k_new), Some(v_new)) = (
            before_repeat_kv(out, inputs[1]),
            before_repeat_kv(out, inputs[2]),
        ) else {
            return Err(format!(
                "{}: K / V {k} / {v} lost their broadcast in the rebuild",
                node.id
            ));
        };
        let mut ins = inputs.to_vec();
        ins[1] = k_new;
        ins[2] = v_new;
        rewired += 1;
        let id = out.add_node(node.op.clone(), ins, node.shape.clone());
        out.node_mut(id).name = node.name.clone();
        Ok(Some(id))
    })?;
    Ok((prune(&g), rewired))
}

/// `graph` without the nodes no output needs, its variables, symbolic
/// attributes and derived symbols carried over.
pub fn prune(graph: &Graph) -> Graph {
    let mut live: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = graph.outputs.clone();
    stack.extend(graph.vars().iter().map(|(i, _)| *i));
    while let Some(id) = stack.pop() {
        if live.insert(id) {
            stack.extend(graph.node(id).inputs.iter().copied());
        }
    }
    let mut out = Graph::new(&graph.name);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        if !live.contains(&node.id) {
            continue;
        }
        let inputs = node.inputs.iter().map(|i| map[i]).collect();
        let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
        let n = out.node_mut(id);
        n.name = node.name.clone();
        n.origin = node.origin.clone();
        map.insert(node.id, id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| map[o]).collect());
    out.set_vars(graph.vars().iter().map(|(i, o)| (map[i], *o)).collect());
    out.set_sym_attrs(
        graph
            .sym_attrs()
            .iter()
            .filter_map(|a| {
                Some(SymAttr {
                    node: *map.get(&a.node)?,
                    ..a.clone()
                })
            })
            .collect(),
    );
    out.set_sym_defs(graph.sym_defs().to_vec());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `q[1, 1, 4*2]` against K / V of 2 heads broadcast to 4.
    fn gqa_graph() -> Graph {
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let mut g = Graph::new("attn");
        let q = g.input("q", f(&[1, 1, 8]));
        let mut kv = Vec::new();
        for name in ["k", "v"] {
            let x = g.input(name, f(&[1, 5, 4]));
            let split = g.reshape(x, vec![1, 5, 2, 1, 2], f(&[1, 5, 2, 1, 2]));
            let wide = g.add_node(
                Op::Expand {
                    target_shape: vec![1, 5, 2, 2, 2],
                },
                vec![split],
                f(&[1, 5, 2, 2, 2]),
            );
            kv.push(g.reshape(wide, vec![1, 5, 8], f(&[1, 5, 8])));
        }
        let a = g.add_node(
            Op::Attention {
                num_heads: 4,
                head_dim: 2,
                v_head_dim: None,
                mask_kind: crate::op::MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, kv[0], kv[1]],
            f(&[1, 1, 8]),
        );
        g.set_outputs(vec![a]);
        g
    }

    #[test]
    fn attention_reads_the_shared_heads_and_the_broadcasts_go() {
        let (g, n) = gqa_native_attention(&gqa_graph()).unwrap();
        assert_eq!(n, 1);
        let a = g.node(g.outputs[0]);
        for i in [1, 2] {
            assert!(matches!(g.node(a.inputs[i]).op, Op::Input { .. }));
        }
        assert!(!g.nodes().iter().any(|n| matches!(n.op, Op::Expand { .. })));
        assert!(crate::verify::verify_all(&g).is_empty());
    }

    #[test]
    fn only_matmul_weights_narrow() {
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let mut g = Graph::new("mm");
        let x = g.input("x", f(&[1, 4]));
        let w = g.param("w", f(&[4, 4]));
        let table = g.param("table", f(&[4, 4]));
        let ids = g.input("ids", Shape::new(&[1], DType::I64));
        let y = g.matmul(x, w, f(&[1, 4]));
        let e = g.gather(table, ids, 0, f(&[1, 4]));
        g.set_outputs(vec![y, e]);
        assert_eq!(matmul_weights_as(&mut g, DType::F16), 1);
        assert_eq!(g.node(w).shape.dtype(), DType::F16);
        assert_eq!(g.node(table).shape.dtype(), DType::F32);
    }
}
