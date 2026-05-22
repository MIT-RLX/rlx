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

//! Pre-load memory estimation (plan #35).
//!
//! Borrowed from MAX's `max/python/max/pipelines/` pattern: model
//! peak memory is estimated *before* weights load. On Apple
//! Silicon this matters disproportionately — unified memory is
//! shared with the OS, so a model that "would fit on a 96 GB
//! Mac" can still OOM if you spawn it during a heavy Spotlight
//! re-index.
//!
//! Three components:
//!
//!   - **Activation working set** — peak arena bytes. Already
//!     computed by `rlx_opt::memory::plan_memory(&graph)`; we
//!     just expose it.
//!   - **Weight bytes** — sum of registered weights from a
//!     [`WeightRegistry`]. Aliases (tied embeddings) don't
//!     double-count.
//!   - **Per-batch input bytes** — bytes the user is going to
//!     hand in via `compiled.run()`. Driven by graph inputs.
//!
//! [`MemoryEstimate::peak_bytes`] is the sum. [`MemoryEstimate::
//! fits_in`] takes a budget and returns the gating decision plus
//! a structured reason.

use crate::expert_pool::gpu_expert_budget_from_vram;
use crate::weight_registry::WeightRegistry;
use rlx_ir::Graph;
use rlx_opt::memory::plan_memory;

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Peak working-set during one forward pass — output of the
    /// memory planner.
    pub activation_bytes: usize,
    /// Total weight bytes, deduplicated against tied aliases.
    pub weight_bytes: usize,
    /// Sum of input tensor sizes for one call (computed from
    /// graph inputs, ignoring outputs since they overlap with
    /// the activation arena).
    pub input_bytes: usize,
}

impl MemoryEstimate {
    pub fn peak_bytes(&self) -> usize {
        self.activation_bytes + self.weight_bytes + self.input_bytes
    }

    /// True if the estimate fits within `budget_bytes`. The
    /// `Result` carries the deficit on failure so callers can
    /// surface a useful error message.
    pub fn fits_in(&self, budget_bytes: usize) -> Result<(), MemoryDeficit> {
        let peak = self.peak_bytes();
        if peak <= budget_bytes {
            Ok(())
        } else {
            Err(MemoryDeficit {
                budget_bytes,
                peak_bytes: peak,
                activation_bytes: self.activation_bytes,
                weight_bytes: self.weight_bytes,
                input_bytes: self.input_bytes,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryDeficit {
    pub budget_bytes: usize,
    pub peak_bytes: usize,
    pub activation_bytes: usize,
    pub weight_bytes: usize,
    pub input_bytes: usize,
}

impl std::fmt::Display for MemoryDeficit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let over = self.peak_bytes - self.budget_bytes;
        write!(
            f,
            "estimated peak {peak_mb:.1} MiB exceeds budget {budget_mb:.1} MiB by {over_mb:.1} MiB \
             (activation {act_mb:.1}, weights {w_mb:.1}, inputs {in_mb:.1})",
            peak_mb = self.peak_bytes as f64 / 1024.0 / 1024.0,
            budget_mb = self.budget_bytes as f64 / 1024.0 / 1024.0,
            over_mb = over as f64 / 1024.0 / 1024.0,
            act_mb = self.activation_bytes as f64 / 1024.0 / 1024.0,
            w_mb = self.weight_bytes as f64 / 1024.0 / 1024.0,
            in_mb = self.input_bytes as f64 / 1024.0 / 1024.0,
        )
    }
}

impl std::error::Error for MemoryDeficit {}

/// Estimate peak memory for running `graph` on a session bound to
/// `registry`. Pure analysis — runs the memory planner internally
/// and queries the registry for weight bytes; doesn't compile or
/// execute.
/// MoE offload sizing (TIDE `enable_predictive_expert_offload`).
#[derive(Debug, Clone)]
pub struct MoeOffloadEstimate {
    /// Bytes for one expert FFN (gate+up+down) at runtime dtype.
    pub expert_param_bytes: usize,
    pub num_moe_layers: usize,
    pub num_experts: usize,
    /// Experts pinned on device per layer after budget clamp.
    pub gpu_expert_budget_per_layer: usize,
    /// All experts resident on host+device (upper bound).
    pub all_expert_weight_bytes: usize,
    /// Only `gpu_expert_budget_per_layer` experts per layer on device.
    pub resident_expert_weight_bytes: usize,
}

impl MoeOffloadEstimate {
    /// Resident expert weights + non-expert peak from [`MemoryEstimate`].
    pub fn peak_with_offload(&self, base: &MemoryEstimate) -> usize {
        base.activation_bytes
            + base.input_bytes
            + (base.weight_bytes - self.all_expert_weight_bytes)
            + self.resident_expert_weight_bytes
    }
}

/// Compute GPU expert budget from a memory budget (unified RAM or VRAM).
pub fn estimate_moe_offload(
    expert_param_bytes: usize,
    num_moe_layers: usize,
    num_experts: usize,
    max_gpu_experts_per_layer: usize,
    memory_budget_bytes: usize,
    reserve_fraction: f32,
) -> MoeOffloadEstimate {
    let reserve_bytes = (memory_budget_bytes as f64 * reserve_fraction as f64) as usize;
    let gpu_budget = gpu_expert_budget_from_vram(
        memory_budget_bytes,
        reserve_bytes,
        expert_param_bytes,
        num_moe_layers,
        max_gpu_experts_per_layer,
        num_experts,
    );
    let all_expert = expert_param_bytes
        .saturating_mul(num_experts)
        .saturating_mul(num_moe_layers);
    let resident_expert = expert_param_bytes
        .saturating_mul(gpu_budget)
        .saturating_mul(num_moe_layers);
    MoeOffloadEstimate {
        expert_param_bytes,
        num_moe_layers,
        num_experts,
        gpu_expert_budget_per_layer: gpu_budget,
        all_expert_weight_bytes: all_expert,
        resident_expert_weight_bytes: resident_expert,
    }
}

pub fn estimate(graph: &Graph, registry: &WeightRegistry) -> MemoryEstimate {
    let plan = plan_memory(graph);
    // Inputs: walk the graph once; sum each Op::Input's shape.
    let mut input_bytes = 0usize;
    for node in graph.nodes() {
        if matches!(node.op, rlx_ir::Op::Input { .. }) {
            input_bytes += node.shape.size_bytes().unwrap_or(0);
        }
    }
    MemoryEstimate {
        activation_bytes: plan.arena_size,
        weight_bytes: registry.total_bytes(),
        input_bytes,
    }
}

/// Available unified-memory budget on the running machine. On
/// macOS reads `hw.memsize` via sysctl; everywhere else returns
/// `None` so callers can fall back to a user-supplied budget.
pub fn available_unified_memory() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        let cname = CString::new("hw.memsize").ok()?;
        let mut val: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::os::raw::c_char,
                oldp: *mut std::os::raw::c_void,
                oldlenp: *mut usize,
                newp: *mut std::os::raw::c_void,
                newlen: usize,
            ) -> std::os::raw::c_int;
        }
        let rc = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                &mut val as *mut u64 as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 { Some(val as usize) } else { None }
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_registry::{WeightKind, WeightRegistry};
    use rlx_ir::*;
    use std::sync::Arc;

    fn small_graph() -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("est");
        let x = g.input("x", Shape::new(&[2, 16], f)); // 128 B
        let w = g.param("w", Shape::new(&[16, 32], f)); // weight via registry
        let mm = g.matmul(x, w, Shape::new(&[2, 32], f)); // 256 B activation
        g.set_outputs(vec![mm]);
        g
    }

    #[test]
    fn estimate_sums_components() {
        let g = small_graph();
        let mut reg = WeightRegistry::new();
        reg.register(
            "w",
            Shape::new(&[16, 32], DType::F32),
            Arc::from(vec![0u8; 16 * 32 * 4]),
            WeightKind::Base,
        );
        let est = estimate(&g, &reg);
        assert!(
            est.activation_bytes >= 256,
            "activation arena should hold mm output"
        );
        assert_eq!(est.weight_bytes, 16 * 32 * 4);
        assert_eq!(est.input_bytes, 2 * 16 * 4);
        assert!(
            est.peak_bytes() >= est.activation_bytes + est.weight_bytes + est.input_bytes
                || est.peak_bytes() == est.activation_bytes + est.weight_bytes + est.input_bytes
        );
    }

    #[test]
    fn fits_in_passes_with_room() {
        let g = small_graph();
        let mut reg = WeightRegistry::new();
        reg.register(
            "w",
            Shape::new(&[16, 32], DType::F32),
            Arc::from(vec![0u8; 2048]),
            WeightKind::Base,
        );
        let est = estimate(&g, &reg);
        assert!(
            est.fits_in(1 << 30).is_ok(),
            "1 GiB budget should fit a tiny graph"
        );
    }

    #[test]
    fn fits_in_reports_deficit() {
        let g = small_graph();
        let mut reg = WeightRegistry::new();
        reg.register(
            "w",
            Shape::new(&[16, 32], DType::F32),
            Arc::from(vec![0u8; 100_000_000]),
            WeightKind::Base,
        );
        let est = estimate(&g, &reg);
        let err = est.fits_in(1024).unwrap_err();
        assert!(err.peak_bytes > err.budget_bytes);
        assert!(format!("{err}").contains("exceeds"));
    }

    #[test]
    fn available_memory_returns_something_on_macos() {
        // basic test — on macOS this should return Some(>0).
        // Elsewhere (CI) we accept None.
        if cfg!(target_os = "macos") {
            let mem = available_unified_memory();
            assert!(mem.is_some());
            assert!(mem.unwrap() > 0);
        }
    }
}
