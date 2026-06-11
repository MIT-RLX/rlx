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

//! Phase 3 — estimate whether a fusion rewrite reduces predicted IO/sync cost.

/// Host-visible traffic and sync points for one forward pass (mirrors `rlx_runtime::graph_io::GraphIoProfile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphIoProfile {
    pub kernel_launches: usize,
    pub sync_points: usize,
    pub host_output_bytes: u64,
    pub device_traffic_bytes: u64,
}

/// Predicted savings from `before` → `after` (positive = fusion helps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionBenefit {
    pub launches_saved: isize,
    pub sync_points_saved: isize,
    pub host_readback_bytes_saved: i64,
    pub device_traffic_bytes_saved: i64,
}

impl FusionBenefit {
    pub fn score_ns(
        &self,
        dispatch_ns: f64,
        roundtrip_ns: f64,
        memory_bw: f64,
        host_readback_bw: f64,
        unified_memory: bool,
    ) -> f64 {
        let rb_before = if unified_memory {
            0.0
        } else {
            host_readback_bw
        };
        (self.launches_saved as f64) * dispatch_ns
            + (self.sync_points_saved as f64) * roundtrip_ns
            + (self.device_traffic_bytes_saved as f64) / memory_bw.max(1.0)
            + (self.host_readback_bytes_saved as f64) / rb_before.max(1.0)
    }

    pub fn should_fuse(
        &self,
        dispatch_ns: f64,
        roundtrip_ns: f64,
        memory_bw: f64,
        host_readback_bw: f64,
        unified_memory: bool,
        min_gain_ns: f64,
    ) -> bool {
        self.score_ns(
            dispatch_ns,
            roundtrip_ns,
            memory_bw,
            host_readback_bw,
            unified_memory,
        ) >= min_gain_ns
    }
}

pub fn fusion_benefit(before: &GraphIoProfile, after: &GraphIoProfile) -> FusionBenefit {
    FusionBenefit {
        launches_saved: before.kernel_launches as isize - after.kernel_launches as isize,
        sync_points_saved: before.sync_points as isize - after.sync_points as isize,
        host_readback_bytes_saved: before.host_output_bytes as i64 - after.host_output_bytes as i64,
        device_traffic_bytes_saved: before.device_traffic_bytes as i64
            - after.device_traffic_bytes as i64,
    }
}

/// Default compile-time IO cost knobs per backend (no runtime calibration).
#[derive(Debug, Clone, Copy)]
pub struct IoFusionGate {
    pub dispatch_ns: f64,
    pub roundtrip_ns: f64,
    pub memory_bw: f64,
    pub host_readback_bw: f64,
    pub unified_memory: bool,
    /// Penalty per added host-sync thunk (`sync_points_saved < 0`), e.g. tail-host `WelchPeaks`.
    pub host_thunk_penalty_ns: f64,
    pub min_gain_ns: f64,
}

use rlx_ir::{Graph, Node, Op, OpKind};
use std::collections::HashSet;

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

fn tensor_bytes(shape: &rlx_ir::Shape) -> u64 {
    shape
        .num_elements()
        .map(|n| (n * shape.dtype().size_bytes()) as u64)
        .unwrap_or(0)
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

/// Ops that force a GPU sync + host thunk on Metal today (mirrors `rlx_runtime::graph_io`).
pub fn host_sync_op_kinds() -> &'static [OpKind] {
    &[
        OpKind::LogMel,
        OpKind::LogMelBackward,
        OpKind::Custom,
        OpKind::WelchPeaks,
    ]
}

/// Static IO profile for a graph (compile-time fusion planning).
pub fn profile_graph_io(graph: &Graph) -> GraphIoProfile {
    profile_graph_io_outputs(graph, &graph.outputs)
}

/// Profile with an explicit output node list (peaks-only vs spectrum readback).
pub fn profile_graph_io_outputs(graph: &Graph, output_ids: &[rlx_ir::NodeId]) -> GraphIoProfile {
    let mut profile = GraphIoProfile::default();
    let output_nodes: HashSet<_> = output_ids.iter().copied().collect();

    for node in graph.nodes() {
        if is_metadata_op(&node.op) {
            continue;
        }
        profile.kernel_launches += 1;
        profile.device_traffic_bytes += node_io_bytes(node, graph);

        let kind = node.op.kind();
        if host_sync_op_kinds().contains(&kind) {
            profile.sync_points += 1;
        }

        if output_nodes.contains(&node.id) {
            profile.host_output_bytes += tensor_bytes(&node.shape);
        }
    }

    profile
}

impl IoFusionGate {
    pub fn score_ns(&self, benefit: &FusionBenefit) -> f64 {
        let mut score = benefit.score_ns(
            self.dispatch_ns,
            self.roundtrip_ns,
            self.memory_bw,
            self.host_readback_bw,
            self.unified_memory,
        );
        if benefit.sync_points_saved < 0 {
            score -= (-benefit.sync_points_saved as f64) * self.host_thunk_penalty_ns;
        }
        score
    }

    pub fn should_fuse(&self, before: &GraphIoProfile, after: &GraphIoProfile) -> bool {
        let benefit = fusion_benefit(before, after);
        self.score_ns(&benefit) >= self.min_gain_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_fusion_saves_readback() {
        let dense = GraphIoProfile {
            kernel_launches: 2,
            sync_points: 1,
            host_output_bytes: 8192,
            device_traffic_bytes: 16384,
        };
        let fused = GraphIoProfile {
            kernel_launches: 1,
            sync_points: 1,
            host_output_bytes: 512,
            device_traffic_bytes: 16384,
        };
        let b = fusion_benefit(&dense, &fused);
        assert!(b.host_readback_bytes_saved > 0);
        assert!(b.launches_saved >= 0);
    }

    #[test]
    fn io_gate_favors_welch_peaks_fusion_on_metal() {
        let dense = GraphIoProfile {
            kernel_launches: 3,
            sync_points: 0,
            host_output_bytes: 33_554_432,
            device_traffic_bytes: 184_549_376,
        };
        let fused = GraphIoProfile {
            kernel_launches: 4,
            sync_points: 1,
            host_output_bytes: 1_048_576,
            device_traffic_bytes: 219_152_384,
        };
        let gate = IoFusionGate {
            dispatch_ns: 500.0,
            roundtrip_ns: 5_000.0,
            memory_bw: 200.0,
            host_readback_bw: 200.0,
            unified_memory: true,
            host_thunk_penalty_ns: 2_000_000.0,
            min_gain_ns: 1_000.0,
        };
        assert!(gate.should_fuse(&dense, &fused));
    }

    #[test]
    fn io_gate_rejects_welch_peaks_fusion_on_wgpu() {
        let dense = GraphIoProfile {
            kernel_launches: 3,
            sync_points: 0,
            host_output_bytes: 33_554_432,
            device_traffic_bytes: 184_549_376,
        };
        let fused = GraphIoProfile {
            kernel_launches: 4,
            sync_points: 1,
            host_output_bytes: 1_048_576,
            device_traffic_bytes: 219_152_384,
        };
        let gate = IoFusionGate {
            dispatch_ns: 3_000.0,
            roundtrip_ns: 30_000.0,
            memory_bw: 100.0,
            host_readback_bw: 40.0,
            unified_memory: false,
            host_thunk_penalty_ns: 25_000_000.0,
            min_gain_ns: 10_000.0,
        };
        assert!(!gate.should_fuse(&dense, &fused));
    }
}
