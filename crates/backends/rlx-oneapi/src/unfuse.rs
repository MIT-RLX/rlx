// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IR-level unfusion for the OneAPI backend.
//!
//! Shared decompose lives in `rlx-unfuse`. OneAPI keeps native
//! `FusedSwiGLU` / residual-norm / `FusedConvBiasAct` kernels, folds biased
//! projections into `FusedMatMulBiasAct` (schedule composes matmul + bias +
//! act), and expands composed ops (`LoraMatMul`, `FusedTransformerLayer`,
//! `DotGeneral`, `If`/`While`, `FusedAttentionBlock`, `PartitionedConv`,
//! `GatedDeltaNet`) to the primitive set the host / OpenCL path runs.

use rlx_ir::{Graph, NodeId, Op, Shape};
use rlx_unfuse::DecomposePolicy;
use std::collections::HashMap;

/// OneAPI policy: native SwiGLU + residual-norm; fold matmul+bias+act;
/// expand FAB / LoRA / FTL / DotGeneral / CF.
pub(crate) struct OneApiPolicy;

impl DecomposePolicy for OneApiPolicy {
    fn swiglu_native(&self) -> bool {
        true
    }

    fn fold_matmul_bias_act(&self) -> bool {
        true
    }

    fn fold_residual_ln(&self) -> bool {
        true
    }
}

/// Apply shared `rlx-unfuse` with [`OneApiPolicy`].
pub fn unfuse(graph: Graph) -> Graph {
    rlx_unfuse::unfuse(graph, &OneApiPolicy)
}

/// Expand fused ops that CPU HostOp would Nop (`PartitionedConv`).
/// `FusedConvBiasAct` stays first-class — native `fused_conv_bias_act.cl`
/// when kernels are embedded; else CPU host-fallback.
pub fn expand_cpu_nop_fused(g: Graph) -> Graph {
    let needs = g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::PartitionedConv { .. }));
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::PartitionedConv { .. } => {
                inline_unfused_compose(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Expand `GatedDeltaNet` via `unfuse_fused_for_autodiff` (time-unrolled
/// MatMul / Mul / Add / Sub / Exp chain). OneAPI has no dedicated GDN kernel;
/// compose-to-primitives is the native path (same as Vulkan).
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
    let mut mini = Graph::new("oneapi_unfuse");
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
