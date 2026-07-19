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
