// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuse `GaussianSplatPrepare` → `GaussianSplatRasterize` into [`Op::GaussianSplatRender`] for AD.

use rlx_ir::{Graph, NodeId, Op};

/// When rasterize has a single-use prepare producer, rewrite rasterize to monolithic render
/// so the existing [`Op::GaussianSplatRender`] VJP applies.
pub fn fuse_decomposed_gaussian_splat(mut g: Graph) -> Graph {
    let n = g.len();
    let mut users = vec![0usize; n];
    for node in g.nodes() {
        for &inp in &node.inputs {
            users[inp.0 as usize] += 1;
        }
    }

    for i in 0..n {
        let rid = NodeId(i as u32);
        let raster = g.node(rid).clone();
        let Op::GaussianSplatRasterize {
            width,
            height,
            tile_size,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
        } = raster.op
        else {
            continue;
        };

        let prep_id = raster.inputs[0];
        if users[prep_id.0 as usize] != 1 {
            continue;
        }
        let (radius_scale, prep_inputs) = match &g.node(prep_id).op {
            Op::GaussianSplatPrepare { radius_scale, .. } => {
                (*radius_scale, g.node(prep_id).inputs.clone())
            }
            _ => continue,
        };

        g.node_mut(rid).op = Op::GaussianSplatRender {
            width,
            height,
            tile_size,
            radius_scale,
            alpha_cutoff,
            max_splat_steps,
            transmittance_threshold,
            max_list_entries,
        };
        g.set_inputs(rid, prep_inputs);
    }
    g
}
