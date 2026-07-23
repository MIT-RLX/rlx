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

//! IR-level "unfusion" pass for the CUDA backend.
//!
//! The shared decompose driver lives in `rlx-unfuse`; this module only
//! supplies CUDA's [`DecomposePolicy`]. CUDA lowers `FusedMatMulBiasAct`
//! (matmul folds bias + activation into its epilogue) and
//! `fused_residual_{ln,rms_norm}.cu` (Add[+bias] + norm in one kernel)
//! natively, so those stay fused rather than being decomposed to
//! FusedMatMulBiasAct/FusedResidualLN — i.e. the folds are OFF here. CUDA's
//! Attention/AttentionBackward kernels are rank-4-only, and a native
//! `fused_attn_block` kernel can serve small-sequence FusedAttentionBlock
//! nodes intact.
//!
//! [`Op::PartitionedConv`] has no native kernel; expand it to the
//! Fft/MatMul GEMM path CUDA already runs (same decomposition as TPU /
//! CPU / Metal).

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;

/// CUDA's decompose policy: keep small `FusedAttentionBlock` nodes native
/// (`fused_attn_block` kernel), promote rank-3 `AttentionBackward` to rank-4
/// like the forward op, and lower everything to primitives (no
/// FusedMatMulBiasAct / FusedResidualLN folding, rank-4-only attention).
pub(crate) struct CudaPolicy;

impl DecomposePolicy for CudaPolicy {
    /// True when the native `fused_attn_block` kernel can serve this block: the
    /// `[seq, seq]` score matrix must fit the GPU's default 48 KB dynamic
    /// shared-memory budget (with margin). Larger sequences decompose to the
    /// primitive chain. The CUDA arena is f32-uniform, so dtype is always fine.
    fn fab_native(&self, out_shape: &Shape) -> bool {
        let dims = out_shape.dims();
        if dims.len() != 3 {
            return false;
        }
        let s = dims[1].unwrap_static();
        // seq*seq*4 bytes of shared memory; 96 → 36 KB, comfortably under 48 KB.
        s > 0 && s <= 96
    }

    fn promote_attention_backward(&self) -> bool {
        true
    }

    fn swiglu_native(&self) -> bool {
        true
    }
}

pub fn unfuse(graph: Graph) -> Graph {
    let graph = expand_partitioned_conv(graph);
    rlx_unfuse::unfuse(graph, &CudaPolicy)
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
    let mut mini = Graph::new("cuda_unfuse_pc");
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
