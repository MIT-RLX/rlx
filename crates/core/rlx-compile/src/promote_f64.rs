// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Promote an f32 graph to F64** (`NumberFormat::F64` / `F64x2`).
//!
//! Every F32 value becomes F64 while the graph keeps its f32 interface:
//!
//! * F32 inputs stay F32 and are cast to F64 on entry (inputs are small and
//!   callers keep feeding f32);
//! * F32 params become F64 — they are the bulk of the data, so the runtime
//!   widens them once on upload rather than the graph casting them on every
//!   run ([`PromotedF64::widened_params`]);
//! * F32 constants are re-encoded as f64;
//! * F32 outputs are cast back to F32 on exit.
//!
//! Integer and boolean values (token ids, masks) are left alone; a cast from
//! them to F32 now casts to F64. F32 values used as indices (token ids fed as
//! floats into a `Gather`) reach the index operand still in F32. Whether a backend can run the result is the
//! caller's question: on the CPU every op must have an f64 kernel, on GPUs
//! the graph goes on to `lower_f64`.

use std::collections::HashMap;

use rlx_ir::{DType, Graph, NodeId, Op};

/// A graph after [`promote_to_f64`].
#[derive(Debug, Clone)]
pub struct PromotedF64 {
    pub graph: Graph,
    /// Params that were F32 and are now F64; callers still upload f32.
    pub widened_params: Vec<String>,
}

/// Rewrite `g`'s f32 computation in F64, keeping its f32 inputs and outputs.
pub fn promote_to_f64(g: &Graph) -> PromotedF64 {
    let mut out = Graph::new(&g.name);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    // F32 inputs as fed, before their cast to F64.
    let mut as_fed: HashMap<NodeId, NodeId> = HashMap::new();
    let mut widened_params = Vec::new();
    for n in g.nodes() {
        let mut ids: Vec<NodeId> = n.inputs.iter().map(|i| map[i]).collect();
        for &k in index_operands(&n.op) {
            let old = n.inputs[k];
            if g.node(old).shape.dtype() == DType::F32 {
                ids[k] = match as_fed.get(&old) {
                    Some(&x) => x,
                    None => out.add_node(
                        Op::Cast { to: DType::F32 },
                        vec![ids[k]],
                        g.node(old).shape.clone(),
                    ),
                };
            }
        }
        let f32_value = n.shape.dtype() == DType::F32;
        let wide = n.shape.clone().with_dtype(DType::F64);
        let id = match &n.op {
            Op::Input { .. } if f32_value => {
                let x = out.add_node(n.op.clone(), vec![], n.shape.clone());
                as_fed.insert(n.id, x);
                out.add_node(Op::Cast { to: DType::F64 }, vec![x], wide)
            }
            Op::Param { name } if f32_value => {
                widened_params.push(name.clone());
                out.add_node(n.op.clone(), vec![], wide)
            }
            Op::Constant { data } if f32_value => {
                let bytes = data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|c| (f32::from_le_bytes(*c) as f64).to_le_bytes())
                    .collect();
                out.add_node(Op::Constant { data: bytes }, vec![], wide)
            }
            // A cast to f32 of something already promoted is now a no-op.
            Op::Cast { to: DType::F32 } if out.node(ids[0]).shape.dtype() == DType::F64 => ids[0],
            Op::Cast { to: DType::F32 } => out.add_node(Op::Cast { to: DType::F64 }, ids, wide),
            _ if f32_value => out.add_node(n.op.clone(), ids, wide),
            _ => out.add_node(n.op.clone(), ids, n.shape.clone()),
        };
        map.insert(n.id, id);
    }
    let outs: Vec<NodeId> = g
        .outputs
        .iter()
        .map(|o| {
            let id = map[o];
            let s = &g.node(*o).shape;
            if s.dtype() == DType::F32 && out.node(id).shape.dtype() == DType::F64 {
                out.add_node(Op::Cast { to: DType::F32 }, vec![id], s.clone())
            } else {
                id
            }
        })
        .collect();
    out.set_outputs(outs);
    PromotedF64 {
        graph: out,
        widened_params,
    }
}

/// Operands of `op` that hold indices rather than values.
fn index_operands(op: &Op) -> &'static [usize] {
    match op {
        Op::Gather { .. } | Op::GatherElements { .. } | Op::GatherNd { .. } => &[1],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Shape;

    #[test]
    fn f32_graph_computes_in_f64_behind_an_f32_interface() {
        let f = DType::F32;
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2, 3], f));
        let w = g.param("w", Shape::new(&[3, 4], f));
        let mm = g.matmul(x, w, Shape::new(&[2, 4], f));
        g.set_outputs(vec![mm]);
        let p = promote_to_f64(&g);
        assert_eq!(p.widened_params, vec!["w".to_string()]);
        let o = p.graph.node(p.graph.outputs[0]);
        assert!(matches!(o.op, Op::Cast { to: DType::F32 }));
        let mm = p.graph.node(o.inputs[0]);
        assert!(matches!(mm.op, Op::MatMul));
        assert_eq!(mm.shape.dtype(), DType::F64);
        for i in &mm.inputs {
            assert_eq!(p.graph.node(*i).shape.dtype(), DType::F64);
        }
        let x = p
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }));
        assert_eq!(x.unwrap().shape.dtype(), DType::F32);
    }

    #[test]
    fn float_token_ids_stay_f32_as_gather_indices() {
        let f = DType::F32;
        let mut g = Graph::new("t");
        let ids = g.input("ids", Shape::new(&[4], f));
        let table = g.param("emb", Shape::new(&[10, 3], f));
        let e = g.add_node(
            Op::Gather { axis: 0 },
            vec![table, ids],
            Shape::new(&[4, 3], f),
        );
        g.set_outputs(vec![e]);
        let p = promote_to_f64(&g);
        let gather = p
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Gather { .. }));
        let idx = p.graph.node(gather.unwrap().inputs[1]);
        assert!(matches!(idx.op, Op::Input { .. }));
        assert_eq!(idx.shape.dtype(), DType::F32);
    }

    #[test]
    fn integer_values_are_untouched() {
        let mut g = Graph::new("t");
        let ids = g.input("ids", Shape::new(&[4], DType::I32));
        let c = g.add_node(
            Op::Cast { to: DType::F32 },
            vec![ids],
            Shape::new(&[4], DType::F32),
        );
        g.set_outputs(vec![c]);
        let p = promote_to_f64(&g);
        let ids = p
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }));
        assert_eq!(ids.unwrap().shape.dtype(), DType::I32);
        let o = p.graph.node(p.graph.outputs[0]);
        assert_eq!(o.shape.dtype(), DType::F32);
        assert_eq!(p.graph.node(o.inputs[0]).shape.dtype(), DType::F64);
    }
}
