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
    pub min_gain_ns: f64,
}

impl IoFusionGate {
    pub fn should_fuse(&self, before: &GraphIoProfile, after: &GraphIoProfile) -> bool {
        fusion_benefit(before, after).should_fuse(
            self.dispatch_ns,
            self.roundtrip_ns,
            self.memory_bw,
            self.host_readback_bw,
            self.unified_memory,
            self.min_gain_ns,
        )
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
            min_gain_ns: 1_000.0,
        };
        assert!(gate.should_fuse(&dense, &fused));
    }
}
