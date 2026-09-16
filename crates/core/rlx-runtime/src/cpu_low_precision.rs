// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// CPU thunks execute f32/f64 kernels. Graphs authored in F16/BF16 are
// promoted to F32 at compile time on CPU and GPU backends; boundary dtypes
// are preserved for typed I/O (`run_typed` / `set_param_typed`).

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

/// Declared boundary dtypes from the user graph (before f32 promotion).
#[derive(Debug, Clone, Default)]
pub struct IoDtypeManifest {
    #[allow(dead_code)]
    pub inputs: HashMap<String, DType>,
    #[allow(dead_code)]
    pub params: HashMap<String, DType>,
    pub outputs: Vec<DType>,
}

impl IoDtypeManifest {
    pub fn from_graph(g: &Graph) -> Self {
        let mut inputs = HashMap::new();
        let mut params = HashMap::new();
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } => {
                    inputs.insert(name.clone(), node.shape.dtype());
                }
                Op::Param { name } => {
                    params.insert(name.clone(), node.shape.dtype());
                }
                _ => {}
            }
        }
        let outputs = g
            .outputs
            .iter()
            .map(|&id| g.node(id).shape.dtype())
            .collect();
        Self {
            inputs,
            params,
            outputs,
        }
    }

    pub fn output_dtype(&self, idx: usize, fallback: DType) -> DType {
        self.outputs.get(idx).copied().unwrap_or(fallback)
    }
}

/// Capture boundary dtypes, then promote F16/BF16 graphs to F32 for GPU/CPU exec.
#[allow(dead_code)]
pub fn prepare_f32_exec_graph(graph: Graph) -> (Graph, IoDtypeManifest) {
    let manifest = IoDtypeManifest::from_graph(&graph);
    let exec = if needs_f32_exec(&graph) {
        promote_to_f32(graph)
    } else {
        graph
    };
    (exec, manifest)
}

/// Widen integer *control* `Param` nodes (duration carry, masks, trip counts)
/// to F32 so f32-uniform backends don't treat raw i64 uploads as denormals.
///
/// Mirrors Metal's `widen_integer_activations_to_f32` Param branch. Packed
/// U8/I8 weights and floating params are left alone. Callers must also widen
/// matching `set_param_typed` uploads via [`crate::backend::widen_bytes_to_f32`].
#[allow(dead_code)] // mlx_backend only; metal/wgpu widen params in-graph instead
pub fn widen_integer_control_params_to_f32(mut graph: Graph) -> Graph {
    for node in graph.nodes_mut() {
        if !matches!(node.op, Op::Param { .. }) {
            continue;
        }
        let old = node.shape.dtype();
        if matches!(
            old,
            DType::I32 | DType::I64 | DType::U32 | DType::Bool | DType::I16
        ) {
            node.shape = node.shape.clone().with_dtype(DType::F32);
        }
    }
    graph
}

pub fn needs_f32_exec(g: &Graph) -> bool {
    g.nodes().iter().any(|n| {
        if !matches!(n.shape.dtype(), DType::F16 | DType::BF16) {
            return false;
        }
        // User-registered `Op::Custom` kernels may execute natively at
        // F16/BF16; only built-in ops need the f32 promotion rewrite.
        !matches!(
            &n.op,
            Op::Custom { .. } | Op::Constant { .. } | Op::Input { .. } | Op::Param { .. }
        )
    })
}

fn promote_dtype(dt: DType) -> DType {
    match dt {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

fn promote_shape(shape: &Shape) -> Shape {
    shape.clone().with_dtype(promote_dtype(shape.dtype()))
}

fn widen_constant_bytes(data: &[u8], from: DType) -> Vec<u8> {
    match from {
        DType::F16 => data
            .chunks_exact(2)
            .flat_map(|c| {
                let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                v.to_le_bytes()
            })
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .flat_map(|c| {
                let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                v.to_le_bytes()
            })
            .collect(),
        _ => data.to_vec(),
    }
}

fn is_lowp_layout_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Concat { .. }
            | Op::Narrow { .. }
            | Op::Reshape { .. }
            | Op::Transpose { .. }
            | Op::Expand { .. }
            // A single-row copy into the cache it aliases — it moves bits and
            // computes nothing, so there is no arithmetic to promote. Promoting
            // it split the op from its own buffer: the cache is a boundary
            // tensor and keeps its declared F16/BF16, while every backend takes
            // the row stride from the KvAppend NODE's dtype, so an F32 node over
            // an F16 cache strides a layout it does not have.
            | Op::KvAppend { .. }
    )
}

/// Rewrite F16/BF16 node shapes (and constant payloads) to F32 for CPU exec.
///
/// Boundary tensors (`Param` / `Input`) keep their declared dtype so Metal (and
/// other backends with native F16 weight paths) can store Linear weights at
/// half width. Pure layout ops that only rearrange F16/BF16 tensors (e.g. the
/// weight `Concat` from shared-input MatMul fusion) stay low-precision too —
/// promoting those would materialize a full F32 copy of every packed weight.
pub fn promote_to_f32(graph: Graph) -> Graph {
    if !needs_f32_exec(&graph) {
        return graph;
    }
    let mut out = Graph::new(format!("{}_f32_exec", graph.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // Node ids in the *source* graph whose dtype we intentionally kept low-p.
    let mut kept_lowp: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let is_boundary = matches!(&node.op, Op::Param { .. } | Op::Input { .. });
        let is_layout_of_lowp = is_lowp_layout_op(&node.op)
            && matches!(node.shape.dtype(), DType::F16 | DType::BF16)
            && !node.inputs.is_empty()
            && node.inputs.iter().all(|&i| {
                kept_lowp.contains(&i)
                    || matches!(graph.node(i).shape.dtype(), DType::F16 | DType::BF16)
            });
        let keep_dtype = is_boundary || is_layout_of_lowp;
        let shape = if keep_dtype {
            node.shape.clone()
        } else {
            promote_shape(&node.shape)
        };
        if keep_dtype && matches!(shape.dtype(), DType::F16 | DType::BF16) {
            kept_lowp.insert(node.id);
        }
        let op = match &node.op {
            Op::Constant { data } => Op::Constant {
                data: widen_constant_bytes(data, node.shape.dtype()),
            },
            Op::Cast { to } => Op::Cast {
                to: promote_dtype(*to),
            },
            other => other.clone(),
        };
        let new_id = out.add_node(op, inputs, shape);
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::GraphExt;

    /// **`KvAppend` must keep the dtype of the cache it writes into.**
    ///
    /// The op aliases its cache input's buffer, and every backend derives the
    /// row stride from the KvAppend NODE's dtype. The cache is a boundary
    /// tensor, so it keeps its declared F16/BF16 through this pass — promoting
    /// the node to F32 while leaving the cache half-width therefore strides a
    /// layout the buffer does not have, and the new token lands in the wrong
    /// row with nothing to report it.
    #[test]
    fn a_low_precision_kv_append_is_not_promoted() {
        for dt in [DType::F16, DType::BF16] {
            let mut g = Graph::new("kv");
            let cache = g.input("cache", Shape::new(&[1, 8, 4], dt));
            let row = g.input("row", Shape::new(&[1, 1, 4], dt));
            let out = g.add_node(
                Op::KvAppend { axis: 1, pos: 3 },
                vec![cache, row],
                Shape::new(&[1, 4, 4], dt),
            );
            g.set_outputs(vec![out]);

            let promoted = promote_to_f32(g);
            let node = promoted.node(promoted.outputs[0]);
            assert!(
                matches!(node.op, Op::KvAppend { .. }),
                "{dt:?}: the append itself vanished"
            );
            assert_eq!(
                node.shape.dtype(),
                dt,
                "{dt:?}: KvAppend was promoted away from its cache's dtype"
            );
        }
    }

    /// The other half: a genuinely computed operand still promotes, so this
    /// exemption cannot be read as "low precision executes natively".
    #[test]
    fn arithmetic_over_the_same_operands_still_promotes() {
        let mut g = Graph::new("mul");
        let a = g.input("a", Shape::new(&[4], DType::F16));
        let b = g.input("b", Shape::new(&[4], DType::F16));
        let out = g.mul(a, b);
        g.set_outputs(vec![out]);

        let promoted = promote_to_f32(g);
        assert_eq!(promoted.node(promoted.outputs[0]).shape.dtype(), DType::F32);
    }
}
