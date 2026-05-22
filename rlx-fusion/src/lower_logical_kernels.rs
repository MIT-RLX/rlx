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
//! Lower logical kernels to common IR when native backend ops are unavailable.
//!
//! Runs before [`rlx_compile::rewrite_for_backend`] so native `supported_ops` still win under
//! [`KernelDispatchPolicy::PreferNative`].

use crate::lower_vae_ops::{LowerGroupNorm, LowerResizeNearest2x};
use crate::pass::Pass;
use rlx_ir::logical_kernel::{self, KernelDispatchConfig};
use rlx_ir::logical_kernel::splat_common;
use rlx_ir::{Graph, NodeId, Op, OpKind};
use std::collections::HashMap;

/// Apply common lowers for logical kernels that are not natively supported (or forced).
pub fn lower_logical_kernels(
    graph: Graph,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> Graph {
    if supported.is_empty()
        && config.policy != rlx_ir::logical_kernel::KernelDispatchPolicy::ForceCommon
        && config.force_common_kinds.is_empty()
    {
        return graph;
    }

    let to_lower = logical_kernel::logical_kinds_in_graph(&graph, supported, config);
    if to_lower.is_empty() {
        return graph;
    }

    let mut g = graph;
    for kind in to_lower {
        g = match kind {
            OpKind::GroupNorm => LowerGroupNorm.run(g),
            OpKind::ResizeNearest2x => LowerResizeNearest2x.run(g),
            OpKind::GaussianSplatRender => lower_gaussian_splat_render_pass(g),
            OpKind::GaussianSplatRenderBackward => lower_gaussian_splat_backward_pass(g),
            _ => g,
        };
    }
    g
}

fn lower_gaussian_splat_render_pass(graph: Graph) -> Graph {
    lower_gaussian_splat_nodes(graph, |g, node| {
        if let Op::GaussianSplatRender {
            width,
            height,
            tile_size: _,
            radius_scale: _,
            alpha_cutoff: _,
            max_splat_steps: _,
            transmittance_threshold: _,
            max_list_entries: _,
        } = &node.op
        {
            let inputs = &node.inputs;
            splat_common::lower_gaussian_splat_render(
                g,
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                *width,
                *height,
                node.shape.clone(),
            )
        } else {
            unreachable!()
        }
    })
}

fn lower_gaussian_splat_backward_pass(graph: Graph) -> Graph {
    lower_gaussian_splat_nodes(graph, |g, node| {
        if let Op::GaussianSplatRenderBackward {
            width,
            height,
            loss_grad_clip: _,
            sh_band: _,
            max_anisotropy: _,
            tile_size: _,
            radius_scale: _,
            alpha_cutoff: _,
            max_splat_steps: _,
            transmittance_threshold: _,
            max_list_entries: _,
        } = &node.op
        {
            let inputs = &node.inputs;
            splat_common::lower_gaussian_splat_render_backward(
                g,
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                *width,
                *height,
                node.shape.clone(),
            )
        } else {
            unreachable!()
        }
    })
}

fn lower_gaussian_splat_nodes<F>(graph: Graph, mut lower_one: F) -> Graph
where
    F: FnMut(&mut Graph, &rlx_ir::Node) -> NodeId,
{
    if !graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. }))
    {
        return graph;
    }

    let mut new_graph = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_id = if matches!(
            node.op,
            Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. }
        ) {
            lower_one(&mut new_graph, node)
        } else {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
    new_graph.set_outputs(new_outputs);
    new_graph
}
