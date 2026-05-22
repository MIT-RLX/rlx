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
//! Host execution for [`Op::GaussianSplatRender`] on TPU-orchestrated graphs.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op};

/// Tensor store keyed by original graph [`NodeId`].
pub type HostTensors = HashMap<NodeId, Vec<f32>>;

pub fn run_splat_render(graph: &Graph, node: NodeId, env: &mut HostTensors) {
    let n = graph.node(node);
    let Op::GaussianSplatRender {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = &n.op
    else {
        panic!("run_splat_render: expected GaussianSplatRender, got {:?}", n.op);
    };

    let get = |id: NodeId| -> &[f32] {
        env.get(&id)
            .unwrap_or_else(|| panic!("rlx-tpu splat: missing tensor for node {id:?}"))
            .as_slice()
    };

    let out = rlx_cpu::splat::render_host_slices(
        get(n.inputs[0]),
        get(n.inputs[1]),
        get(n.inputs[2]),
        get(n.inputs[3]),
        get(n.inputs[4]),
        get(n.inputs[5]),
        get(n.inputs[6]),
        *width,
        *height,
        *tile_size,
        *radius_scale,
        *alpha_cutoff,
        *max_splat_steps,
        *transmittance_threshold,
        *max_list_entries,
    );
    env.insert(node, out);
}

pub fn run_splat_backward(graph: &Graph, node: NodeId, env: &mut HostTensors) {
    let n = graph.node(node);
    let Op::GaussianSplatRenderBackward {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
        loss_grad_clip,
        sh_band,
        max_anisotropy,
    } = &n.op
    else {
        panic!(
            "run_splat_backward: expected GaussianSplatRenderBackward, got {:?}",
            n.op
        );
    };

    let get = |id: NodeId| -> &[f32] {
        env.get(&id)
            .unwrap_or_else(|| panic!("rlx-tpu splat: missing tensor for node {id:?}"))
            .as_slice()
    };

    let packed = rlx_cpu::splat::backward_host_slices(
        get(n.inputs[0]),
        get(n.inputs[1]),
        get(n.inputs[2]),
        get(n.inputs[3]),
        get(n.inputs[4]),
        get(n.inputs[5]),
        get(n.inputs[6]),
        get(n.inputs[7]),
        *width,
        *height,
        *tile_size,
        *radius_scale,
        *alpha_cutoff,
        *max_splat_steps,
        *transmittance_threshold,
        *max_list_entries,
        *loss_grad_clip,
        *sh_band,
        *max_anisotropy,
    );
    env.insert(node, packed);
}
