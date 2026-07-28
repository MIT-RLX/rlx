// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Per-node monitoring** — build/forward timing, resident memory, and
//! throughput, collected by the coordinator and rendered as a table.

use serde::{Deserialize, Serialize};
use std::ops::Range;

/// What a worker reports about building + serving its stage (JSON on stdout).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReport {
    pub addr: String,
    pub layers: Range<usize>,
    pub device: String,
    pub precision: String,
    /// Time to build the stage graph + load weights (ms).
    pub build_ms: u64,
    /// Resident bytes after build.
    pub resident_bytes: u64,
    pub n_params: usize,
    pub n_packed: usize,
}

/// Coordinator-side per-stage timing (network + compute for one forward).
#[derive(Debug, Clone)]
pub struct StageTiming {
    pub addr: String,
    pub layers: Range<usize>,
    pub device: String,
    pub build_ms: u64,
    pub forward_ms: u64,
    pub resident_bytes: u64,
}

/// Full run result: the model output + per-node monitoring.
#[derive(Debug, Clone)]
pub struct ClusterRun {
    pub timings: Vec<StageTiming>,
    pub total_forward_ms: u64,
    /// Final-stage output (e.g. logits), flat.
    pub output: Vec<f32>,
}

impl ClusterRun {
    /// Throughput in tokens/s given the sequence length driven.
    pub fn tokens_per_s(&self, seq: usize) -> f64 {
        if self.total_forward_ms == 0 {
            return 0.0;
        }
        seq as f64 / (self.total_forward_ms as f64 / 1000.0)
    }

    /// Aggregate a per-node metric with a reduce op — the SAME [`ReduceKind`]
    /// semantics the collective all-reduce uses for tensor-parallel partial sums.
    /// (Here over locally-gathered stats; [`crate::cluster::all_reduce`] runs the
    /// real cross-node collective.)
    pub fn reduce_resident_gb(&self, kind: rlx_driver::ReduceKind) -> f64 {
        use rlx_driver::ReduceKind::*;
        let vals = self.timings.iter().map(|t| t.resident_bytes as f64 / 1e9);
        match kind {
            Sum => vals.sum(),
            Mean => {
                let n = self.timings.len().max(1) as f64;
                vals.sum::<f64>() / n
            }
            Max => vals.fold(f64::MIN, f64::max),
            Min => vals.fold(f64::MAX, f64::min),
        }
    }
    /// Critical-path forward time = MAX over stages (a reduce-Max); the pipeline
    /// bottleneck the planner's throughput policy minimizes.
    pub fn critical_path_ms(&self) -> u64 {
        self.timings.iter().map(|t| t.forward_ms).max().unwrap_or(0)
    }

    /// A monitor table for stdout.
    pub fn table(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{:<22}{:<10}{:<8}{:>10}{:>12}{:>12}\n",
            "node", "layers", "device", "build", "forward", "resident"
        ));
        s.push_str(&"─".repeat(74));
        s.push('\n');
        for t in &self.timings {
            s.push_str(&format!(
                "{:<22}{:<10}{:<8}{:>9}s{:>11}s{:>10.1}G\n",
                t.addr,
                format!("{}..{}", t.layers.start, t.layers.end),
                t.device,
                format!("{:.1}", t.build_ms as f64 / 1000.0),
                format!("{:.1}", t.forward_ms as f64 / 1000.0),
                t.resident_bytes as f64 / 1e9,
            ));
        }
        s.push_str(&"─".repeat(74));
        s.push('\n');
        // Cluster aggregates via reduce ops: Σ resident, max critical-path stage.
        use rlx_driver::ReduceKind::{Max, Sum};
        s.push_str(&format!(
            "reduce: Σ resident {:.1} GB, max resident {:.1} GB | total forward {:.1}s, critical path {:.1}s across {} stages\n",
            self.reduce_resident_gb(Sum),
            self.reduce_resident_gb(Max),
            self.total_forward_ms as f64 / 1000.0,
            self.critical_path_ms() as f64 / 1000.0,
            self.timings.len(),
        ));
        s
    }
}
