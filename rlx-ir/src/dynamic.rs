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

//! Dynamic / symbolic dimensions — compile once, specialize at runtime.
//!
//! Plan #54: graphs built with [`Dim::Dynamic`] symbols specialize via
//! [`DimBinding`] before buffer planning and backend lowering.

use std::collections::{BTreeSet, HashMap};

use crate::shape::{Dim, DimBinding, Shape};
use crate::{DType, Graph, Op};

/// Well-known dynamic dimension symbols. Reuse the same id across shapes
/// so `[?0, ?1, H]` and `[?0, ?1, 4]` share batch/seq bindings.
pub mod sym {
    pub const BATCH: u32 = 0;
    pub const SEQ: u32 = 1;
    /// Cached prefix length for decode KV (past_k / past_v axis 1).
    pub const PAST_SEQ: u32 = 3;
    /// Product of leading axes (e.g. `batch * seq` flatten).
    pub const ROWS: u32 = 2;
}

/// Allocate named dynamic symbols for model builders.
#[derive(Debug, Clone, Default)]
pub struct DimEnv {
    next: u32,
    names: HashMap<String, u32>,
}

impl DimEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the symbol id for `name`, allocating on first use.
    pub fn sym(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = self.next;
        self.next += 1;
        self.names.insert(name.into(), id);
        id
    }

    pub fn name(&self, symbol: u32) -> Option<&str> {
        self.names
            .iter()
            .find_map(|(n, &s)| (s == symbol).then_some(n.as_str()))
    }
}

impl Shape {
    /// `[batch, seq, hidden]` with symbolic leading axes.
    pub fn batch_seq(batch: u32, seq: u32, hidden: usize, dtype: DType) -> Self {
        Self::from_dims(
            &[Dim::Dynamic(batch), Dim::Dynamic(seq), Dim::Static(hidden)],
            dtype,
        )
    }

    /// `[batch, seq]` matrix.
    pub fn batch_seq_2d(batch: u32, seq: u32, dtype: DType) -> Self {
        Self::from_dims(&[Dim::Dynamic(batch), Dim::Dynamic(seq)], dtype)
    }

    /// `[batch, seq, heads, head_dim]` attention layout.
    pub fn batch_seq_heads(
        batch: u32,
        seq: u32,
        heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        Self::from_dims(
            &[
                Dim::Dynamic(batch),
                Dim::Dynamic(seq),
                Dim::Static(heads),
                Dim::Static(head_dim),
            ],
            dtype,
        )
    }
}

impl DimBinding {
    pub fn from_pairs(pairs: &[(u32, usize)]) -> Self {
        let mut b = Self::new();
        for &(sym, size) in pairs {
            b.set(sym, size);
        }
        b
    }

    pub fn batch_seq(batch: usize, seq: usize) -> Self {
        let mut b = Self::from_pairs(&[(sym::BATCH, batch), (sym::SEQ, seq)]);
        if batch > 1 {
            b.set(sym::ROWS, batch * seq);
        }
        b
    }

    pub fn batch_past_seq(batch: usize, past_seq: usize) -> Self {
        Self::from_pairs(&[(sym::BATCH, batch), (sym::PAST_SEQ, past_seq)])
    }
}

/// True if any node shape references a dynamic dimension.
pub fn has_dynamic_dims(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| n.shape.dims().iter().any(|d| matches!(d, Dim::Dynamic(_))))
}

/// Collect all dynamic symbols referenced anywhere in the graph.
pub fn collect_dynamic_symbols(graph: &Graph) -> Vec<u32> {
    let mut syms = BTreeSet::new();
    for node in graph.nodes() {
        for s in node.shape.dynamic_symbols() {
            syms.insert(s);
        }
    }
    syms.into_iter().collect()
}

/// Specialize every node's shape against `bindings`.
///
/// Node ids are preserved (nodes are cloned in insertion order), so
/// edges and outputs remain valid without remapping.
pub fn bind_graph(graph: &Graph, bindings: &DimBinding) -> Graph {
    let mut out = Graph::new(&graph.name);
    for node in graph.nodes() {
        let bound = node.shape.bind(bindings);
        out.push_ext(
            node.op.clone(),
            node.inputs.clone(),
            bound,
            node.name.clone(),
            node.origin.clone(),
        );
    }
    out.set_outputs(graph.outputs.clone());
    out
}

/// After [`bind_graph`], sync `Op::Reshape { new_shape }` with bound node shapes.
pub fn sync_reshape_ops(graph: &mut Graph) {
    use crate::Op;
    for node in graph.nodes_mut() {
        if let Op::Reshape { new_shape } = &mut node.op {
            if node.shape.is_static() {
                *new_shape = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i64)
                    .collect();
            }
        }
    }
}

/// Recompute all inferrable output shapes after binding (propagates concat fixes).
pub fn sync_graph_shapes(graph: &mut Graph) {
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        if let Some(shape) = crate::infer_shape::infer_output_shape(graph, node) {
            graph.node_mut(node.id).shape = shape;
        }
    }
}

/// Recompute `Op::Concat` output shapes from bound inputs (fixes mixed static+dynamic axes).
pub fn sync_concat_shapes(graph: &mut Graph) {
    use crate::Op;
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        let Op::Concat { axis } = &node.op else {
            continue;
        };
        let shapes: Vec<Shape> = node
            .inputs
            .iter()
            .map(|&id| graph.node(id).shape.clone())
            .collect();
        let refs: Vec<&Shape> = shapes.iter().collect();
        if let Ok(out) = crate::shape::concat_shape(&refs, *axis) {
            graph.node_mut(node.id).shape = out;
        }
    }
}

/// Clamp `Op::Narrow` start indices after bind (template may bake in max_seq placeholders).
pub fn sync_narrow_ops(graph: &mut Graph) {
    use crate::Op;
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        let Op::Narrow { axis, start, len } = &node.op else {
            continue;
        };
        let in_shape = graph.node(node.inputs[0]).shape.clone();
        if *axis >= in_shape.rank() || !in_shape.is_static() {
            continue;
        }
        let ax_len = in_shape.dims()[*axis].unwrap_static();
        if *start + *len > ax_len {
            graph.node_mut(node.id).op = Op::Narrow {
                axis: *axis,
                start: ax_len.saturating_sub(*len),
                len: *len,
            };
        }
    }
}

/// Infer symbol sizes from runtime input element counts.
///
/// Each `Op::Input` may have at most one dynamic dimension; its size is
/// `data_len / product(static_dims)`.
pub fn infer_bindings_from_inputs(
    graph: &Graph,
    inputs: &[(&str, usize)],
) -> Result<DimBinding, String> {
    let by_name: HashMap<&str, usize> = inputs.iter().copied().collect();
    let mut binding = DimBinding::new();
    for node in graph.nodes() {
        let Op::Input { name } = &node.op else {
            continue;
        };
        let Some(&n_elems) = by_name.get(name.as_str()) else {
            continue;
        };
        let mut static_prod: usize = 1;
        let mut dynamic_sym: Option<u32> = None;
        for d in node.shape.dims() {
            match d {
                Dim::Static(n) => static_prod *= *n,
                Dim::Dynamic(sym) => {
                    if dynamic_sym.is_some() {
                        return Err(format!(
                            "Input '{name}' has multiple dynamic dims; \
                             pass an explicit DimBinding"
                        ));
                    }
                    dynamic_sym = Some(*sym);
                }
            }
        }
        let Some(sym) = dynamic_sym else {
            continue;
        };
        if static_prod == 0 {
            return Err(format!("Input '{name}': static dim product is zero"));
        }
        if n_elems % static_prod != 0 {
            return Err(format!(
                "Input '{name}': len {n_elems} not divisible by static product {static_prod}"
            ));
        }
        let size = n_elems / static_prod;
        if let Some(prev) = binding.get(sym) {
            if prev != size {
                return Err(format!(
                    "symbol {sym} bound to {prev} and {size} from different inputs"
                ));
            }
        } else {
            binding.set(sym, size);
        }
    }
    Ok(binding)
}

/// Infer bindings from f32 slice lengths (convenience for tests/runtime).
pub fn infer_bindings_from_f32_inputs(
    graph: &Graph,
    inputs: &[(&str, &[f32])],
) -> Result<DimBinding, String> {
    infer_bindings_from_inputs(
        graph,
        &inputs
            .iter()
            .map(|(n, d)| (*n, d.len()))
            .collect::<Vec<_>>(),
    )
}

pub fn same_binding(a: &DimBinding, b: &DimBinding) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(sym, size)| b.get(sym) == Some(size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::GraphExt;

    #[test]
    fn bind_graph_specializes_matmul() {
        let batch = sym::BATCH;
        let seq = sym::SEQ;
        let mut g = Graph::new("dyn");
        let x = g.input("x", Shape::batch_seq(batch, seq, 4, DType::F32));
        let w = g.param("w", Shape::new(&[4, 8], DType::F32));
        let y = g.mm(x, w);
        g.set_outputs(vec![y]);

        assert!(has_dynamic_dims(&g));
        let binding = DimBinding::batch_seq(2, 16);
        let bound = bind_graph(&g, &binding);
        assert!(!has_dynamic_dims(&bound));
        assert_eq!(
            bound.node(bound.outputs[0]).shape,
            Shape::new(&[2, 16, 8], DType::F32)
        );
    }

    #[test]
    fn infer_bindings_from_input_data() {
        let mut g = Graph::new("dyn");
        let x = g.input(
            "x",
            Shape::from_dims(
                &[Dim::Static(3), Dim::Dynamic(sym::SEQ), Dim::Static(64)],
                DType::F32,
            ),
        );
        g.set_outputs(vec![x]);

        let b = infer_bindings_from_f32_inputs(&g, &[("x", &vec![0.0f32; 3 * 128 * 64])])
            .expect("infer");
        assert_eq!(b.get(sym::SEQ), Some(128));
    }
}
