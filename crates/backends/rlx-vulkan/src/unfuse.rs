// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! IR-level unfusion for the Vulkan backend via shared `rlx-unfuse`.
//!
//! Vulkan keeps `FusedSwiGLU` / `FusedResidualLN` / `FusedResidualRmsNorm`
//! native, accepts rank-3 Attention via strides (like wgpu), and folds biased
//! projections into `FusedMatMulBiasAct` (schedule composes matmul + bias +
//! act from existing SPIR-V steps). `GatedDeltaNet` expands to MatMul / Mul /
//! Add / … primitives before legalize (same compose path as TPU).

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;
use std::collections::HashMap;

/// Vulkan decompose policy: native SwiGLU / residual-norm, fold matmul+bias+act,
/// and accept rank-3 Attention via strides (same as wgpu).
pub(crate) struct VulkanPolicy;

impl DecomposePolicy for VulkanPolicy {
    fn swiglu_native(&self) -> bool {
        true
    }

    fn fold_matmul_bias_act(&self) -> bool {
        true
    }

    fn fold_residual_ln(&self) -> bool {
        true
    }

    fn attention_accepts_rank3(&self) -> bool {
        true
    }
}

/// Apply shared `rlx-unfuse` with [`VulkanPolicy`].
pub fn unfuse(graph: Graph) -> Graph {
    rlx_unfuse::unfuse(graph, &VulkanPolicy)
}

/// Expand `GatedDeltaNet` via `unfuse_fused_for_autodiff` (time-unrolled
/// MatMul / Mul / Add / Sub / Exp chain). Vulkan has no dedicated GDN kernel;
/// SelectiveScan is a different recurrence, so compose-to-primitives is the
/// native path.
pub fn expand_gated_delta_net(g: Graph) -> Graph {
    let needs = g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::GatedDeltaNet { .. }));
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::GatedDeltaNet { .. } => {
                inline_unfused_compose(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused_compose(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    let mut mini = Graph::new("vulkan_unfuse");
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
