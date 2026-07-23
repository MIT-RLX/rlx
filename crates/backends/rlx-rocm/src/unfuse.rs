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

//! IR-level "unfusion" pass for the ROCm/HIP backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies ROCm's [`DecomposePolicy`]. ROCm shares CUDA's `.cu` kernels but
//! (unlike CUDA) does not keep any `FusedAttentionBlock` native and does not
//! promote `AttentionBackward` — so the policy is the plain default: lower
//! every composite to primitives, materialize rank-4 attention, no
//! FusedMatMulBiasAct / FusedResidualLN folding.
//!
//! [`Op::PartitionedConv`] has no native kernel; expand it to the
//! Fft/MatMul GEMM path ROCm already runs (same decomposition as TPU /
//! CPU / Metal / CUDA).

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;

/// ROCm's decompose policy — keep native `FusedSwiGLU` (shared `.cu` kernel);
/// everything else uses the plain defaults (see [`DecomposePolicy`]).
pub(crate) struct RocmPolicy;

impl DecomposePolicy for RocmPolicy {
    fn swiglu_native(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    let graph = expand_partitioned_conv(graph);
    rlx_unfuse::unfuse(graph, &RocmPolicy)
}

/// Expand [`Op::PartitionedConv`] → batched-GEMM frequency-domain primitives
/// (`Fft` / `MatMul` / …) via `unfuse_fused_for_autodiff`.
fn expand_partitioned_conv(g: Graph) -> Graph {
    let needs = g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::PartitionedConv { .. }));
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::PartitionedConv { .. } => {
                inline_unfused(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    let mut mini = Graph::new("rocm_unfuse_pc");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, &src) in inputs.iter().enumerate() {
        let sh = out.node(src).shape.clone();
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op {
            if let Some(rest) = name.strip_prefix("in") {
                if let Ok(i) = rest.parse::<usize>() {
                    map.insert(n.id, inputs[i]);
                    continue;
                }
            }
        }
        let mapped: Vec<NodeId> = n.inputs.iter().map(|id| map[id]).collect();
        let nid = out.add_node(n.op.clone(), mapped, n.shape.clone());
        map.insert(n.id, nid);
    }
    map[&expanded.outputs[0]]
}
