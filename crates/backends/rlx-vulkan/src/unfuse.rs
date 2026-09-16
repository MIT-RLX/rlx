// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Apply shared `rlx-unfuse` with `VulkanPolicy`.
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

/// Expand `PartitionedConv` into the frequency-domain chain
/// (`rfft -> complex matmul over partitions -> irfft`).
///
/// Vulkan claims `PartitionedConv` for legalize, so it never lands in the
/// unsupported set that would make `legalize_or_rewrite_for_backend` expand it,
/// and `rlx-unfuse` (unlike `rlx-fusion`'s unfuse) has no arm for it. That left
/// the node intact all the way to the scheduler, which routed it to the CPU
/// host fallback — where it is a **Nop**, because rlx-cpu expands the op before
/// building thunks and so has no kernel for it either.
///
/// The result was a silent buffer of zeros on every Vulkan implementation
/// (MoltenVK, NVIDIA, RADV all agreed). Expanding here, in Vulkan's own compile
/// entry, is the same fix oneapi already carries as `expand_cpu_nop_fused`.
pub fn expand_partitioned_conv(g: Graph) -> Graph {
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
            Op::PartitionedConv { .. } => inline_compose(
                &mut out,
                &node.op,
                &new_inputs,
                &node.shape,
                rlx_opt::unfuse_fused_for_autodiff,
            ),
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused_compose(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    // Unconditional: Vulkan has no GDN kernel, so the flag-gated
    // `unfuse_fused_for_autodiff` would leave the node fused and it would
    // reach the scheduler and panic.
    inline_compose(
        out,
        op,
        inputs,
        shape,
        rlx_opt::unfuse_gated_delta_net_always,
    )
}

/// Splice `op` into a one-node graph, run `expand` over it, and inline the
/// resulting primitives back into `out`.
fn inline_compose(
    out: &mut Graph,
    op: &Op,
    inputs: &[NodeId],
    shape: &Shape,
    expand: fn(Graph) -> Graph,
) -> NodeId {
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
    let expanded = expand(mini);
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
