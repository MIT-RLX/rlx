// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Give `Attention` K / V one head per query head, for backends whose
//! kernel cannot index a shared one.
//!
//! `Op::Attention` takes K / V with fewer heads than Q (grouped-query
//! attention): the CPU, Metal and wgpu kernels map query head `h` to K/V
//! head `h / (heads / kv_heads)`. The CUDA (and so ROCm, which shares its
//! kernels) and Vulkan kernels read K / V as if they had `num_heads` heads,
//! and returned a wrong tensor without an error (0.62 / 0.72 off on an
//! 8-over-2 case). A graph from `rlx_ir::retarget::gqa_native_attention`,
//! or any builder that leaves the heads shared, is such a graph. This puts
//! `repeat_kv`'s broadcast back for them: `[…, kv*dh] → [… , kv, 1, dh] →
//! expand [… , kv, g, dh] → [… , kv*g*dh]`.

use std::collections::HashMap;

use rlx_ir::{Dim, Graph, NodeId, Op, OpKind, Shape};

use crate::Pass;

/// The K / V head count of `shape` (`[B, S, kv*dh]`, static) when fewer
/// than `heads` divide them; `None` when it has them all or another shape.
fn shared_heads(shape: &Shape, heads: usize, dh: usize) -> Option<usize> {
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic(_) => None,
        })
        .collect::<Option<_>>()?;
    let [_, _, w] = dims[..] else {
        return None;
    };
    if dh == 0 || !w.is_multiple_of(dh) {
        return None;
    }
    let kv = w / dh;
    (kv > 0 && kv < heads && heads.is_multiple_of(kv)).then_some(kv)
}

/// K / V of `[…, kv*dh]` repeated to `heads` heads; `None` when it already
/// has them or the shape is not static `[B, S, kv*dh]`.
fn repeat_heads(g: &mut Graph, x: NodeId, heads: usize, dh: usize) -> Option<NodeId> {
    let shape = g.node(x).shape.clone();
    let kv = shared_heads(&shape, heads, dh)?;
    let [b, s, _] = shape.dims()[..] else {
        return None;
    };
    let (b, s) = (b.unwrap_static(), s.unwrap_static());
    let group = heads / kv;
    let dt = shape.dtype();
    let at = |d: &[usize]| d.iter().map(|&n| n as i64).collect::<Vec<i64>>();
    let split = [b, s, kv, 1, dh];
    let wide = [b, s, kv, group, dh];
    let flat = [b, s, heads * dh];
    let split = g.add_node(
        Op::Reshape {
            new_shape: at(&split),
        },
        vec![x],
        Shape::new(&split, dt),
    );
    let wide_id = g.add_node(
        Op::Expand {
            target_shape: at(&wide),
        },
        vec![split],
        Shape::new(&wide, dt),
    );
    Some(g.add_node(
        Op::Reshape {
            new_shape: at(&flat),
        },
        vec![wide_id],
        Shape::new(&flat, dt),
    ))
}

/// See the module docs.
pub struct ExpandSharedKvHeads;

impl Pass for ExpandSharedKvHeads {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Attention]
    }

    fn name(&self) -> &str {
        "expand_shared_kv_heads"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Most graphs have no shared heads: give them back as they are.
        let shared = graph.nodes().iter().any(|n| match n.op {
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                ..
            } if n.inputs.len() >= 3 => {
                let shape = |i: usize| &graph.node(n.inputs[i]).shape;
                shared_heads(shape(1), num_heads, head_dim).is_some()
                    || shared_heads(shape(2), num_heads, v_head_dim.unwrap_or(head_dim)).is_some()
            }
            _ => false,
        });
        if !shared {
            return graph;
        }
        let mut out = Graph::new(&graph.name);
        let mut map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let mut inputs: Vec<NodeId> = node.inputs.iter().map(|i| map[i]).collect();
            if let Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                ..
            } = node.op
                && inputs.len() >= 3
            {
                if let Some(k) = repeat_heads(&mut out, inputs[1], num_heads, head_dim) {
                    inputs[1] = k;
                }
                let dv = v_head_dim.unwrap_or(head_dim);
                if let Some(v) = repeat_heads(&mut out, inputs[2], num_heads, dv) {
                    inputs[2] = v;
                }
            }
            let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
            let n = out.node_mut(id);
            n.name = node.name.clone();
            n.origin = node.origin.clone();
            map.insert(node.id, id);
        }
        out.set_outputs(graph.outputs.iter().map(|o| map[o]).collect());
        out.set_vars(graph.vars().iter().map(|(i, o)| (map[i], *o)).collect());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[test]
    fn shared_heads_are_repeated_and_full_ones_left() {
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let attn = |kv_width: usize| {
            let mut g = Graph::new("a");
            let q = g.input("q", f(&[1, 2, 8]));
            let k = g.input("k", f(&[1, 5, kv_width]));
            let v = g.input("v", f(&[1, 5, kv_width]));
            let a = g.add_node(
                Op::Attention {
                    num_heads: 4,
                    head_dim: 2,
                    v_head_dim: None,
                    mask_kind: rlx_ir::op::MaskKind::None,
                    score_scale: None,
                    attn_logit_softcap: None,
                },
                vec![q, k, v],
                f(&[1, 2, 8]),
            );
            g.set_outputs(vec![a]);
            g
        };
        let g = ExpandSharedKvHeads.run(attn(4));
        let a = g.node(g.outputs[0]);
        assert_eq!(g.node(a.inputs[1]).shape, f(&[1, 5, 8]));
        assert_eq!(g.node(a.inputs[2]).shape, f(&[1, 5, 8]));
        assert!(rlx_ir::verify::verify_all(&g).is_empty());
        let same = ExpandSharedKvHeads.run(attn(8));
        assert_eq!(same.len(), 4);
    }
}
