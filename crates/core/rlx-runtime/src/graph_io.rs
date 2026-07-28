// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static IO / sync profile for compiled graphs (Phase 0 — fusion planning).

use rlx_ir::{Graph, Node, Op, OpKind};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Tuning for static IO analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GraphIoOptions {
    /// Count each `Op::Fft` as a host-sync boundary (non-native fallback).
    pub fft_host_sync: bool,
}

/// Host-visible traffic and sync points for one forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct GraphIoProfile {
    /// Kernel / thunk dispatches (one per non-view executable node).
    pub kernel_launches: usize,
    /// GPU flush + host-side thunk boundaries (Metal LogMel, host FFT fallback, …).
    pub sync_points: usize,
    /// Bytes returned to the caller via graph outputs (`CompiledGraph::run`).
    pub host_output_bytes: u64,
    /// Bytes moved inside the device arena (read inputs + write outputs per node).
    pub device_traffic_bytes: u64,
}

impl GraphIoProfile {
    pub fn host_readback_bytes(&self, unified_memory: bool) -> u64 {
        if unified_memory {
            self.host_output_bytes
        } else {
            self.host_output_bytes
                .saturating_add(self.device_traffic_bytes / 4)
        }
    }
}

/// Ops that force a GPU sync + host thunk on Metal today.
pub fn metal_host_sync_kinds() -> &'static [OpKind] {
    &[
        OpKind::LogMel,
        OpKind::LogMelBackward,
        OpKind::Custom,
        OpKind::WelchPeaks,
    ]
}

/// Profile a graph before compile (conservative static estimate).
pub fn profile_graph_io(graph: &Graph) -> GraphIoProfile {
    profile_graph_io_with_options(graph, GraphIoOptions::default())
}

pub fn profile_graph_io_with_options(graph: &Graph, opts: GraphIoOptions) -> GraphIoProfile {
    let mut profile = GraphIoProfile::default();
    let output_nodes: HashSet<_> = graph.outputs.iter().copied().collect();

    for node in graph.nodes() {
        if is_metadata_op(&node.op) {
            continue;
        }
        profile.kernel_launches += 1;
        profile.device_traffic_bytes += node_io_bytes(node, graph);

        let kind = node.op.kind();
        if metal_host_sync_kinds().contains(&kind) {
            profile.sync_points += 1;
        }
        if opts.fft_host_sync && kind == OpKind::Fft {
            profile.sync_points += 1;
        }

        if output_nodes.contains(&node.id) {
            profile.host_output_bytes += tensor_bytes(&node.shape);
        }
    }

    profile
}

/// Profile with only selected outputs materialized on the host (peaks-only, logits-only, …).
pub fn profile_graph_io_outputs(graph: &Graph, output_indices: &[usize]) -> GraphIoProfile {
    let mut profile = profile_graph_io(graph);
    profile.host_output_bytes = graph
        .outputs
        .iter()
        .enumerate()
        .filter(|(i, _)| output_indices.contains(i))
        .filter_map(|(_, id)| graph.node(*id).shape.num_elements())
        .map(|n| (n * 4) as u64)
        .sum();
    profile
}

fn is_metadata_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Reshape { .. }
            | Op::Transpose { .. }
            | Op::Narrow { .. }
    )
}

fn node_io_bytes(node: &Node, graph: &Graph) -> u64 {
    let out = tensor_bytes(&node.shape);
    let inputs: u64 = node
        .inputs
        .iter()
        .map(|&id| tensor_bytes(&graph.node(id).shape))
        .sum();
    inputs.saturating_add(out)
}

fn tensor_bytes(shape: &rlx_ir::Shape) -> u64 {
    shape
        .num_elements()
        .map(|n| (n * shape.dtype().size_bytes()) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Shape};

    #[test]
    fn fft_graph_io_profile() {
        let mut g = Graph::new("fft");
        let x = g.input("x", Shape::new(&[8, 512], DType::F32));
        let zeros = g.sub(x, x);
        let block = g.concat_(vec![x, zeros], 1);
        let y = g.fft(block, false);
        g.set_outputs(vec![y]);
        let p = profile_graph_io(&g);
        assert!(p.kernel_launches >= 3);
        assert_eq!(p.host_output_bytes, (8 * 512 * 2 * 4) as u64);
    }

    #[test]
    fn peaks_only_output_smaller_readback() {
        let mut g = Graph::new("peaks");
        let spec = g.input("spec", Shape::new(&[4, 512], DType::F32));
        let peaks = g.welch_peaks(spec, 16, 2);
        g.set_outputs(vec![peaks]);
        let full = profile_graph_io(&g);
        assert_eq!(full.host_output_bytes, (2 * 16 * 2 * 4) as u64);
        assert!(full.device_traffic_bytes > full.host_output_bytes);
    }
}
