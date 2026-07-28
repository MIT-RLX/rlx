// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declared optional features of an [`crate::ExecutableGraph`].
//!
//! The [`ExecutableGraph`](crate::ExecutableGraph) trait grew a long tail of
//! defaulted methods (MoE residency, GPU handles, KV row feeds, async
//! pipeline, …). Most backends leave those as no-ops / `false`. Callers that
//! need to branch on support should prefer:
//!
//! 1. [`ExecutableGraph::capabilities`] for a cheap advisory bitmask, then
//! 2. the concrete method's return value (`false` / `None`) as the source of
//!    truth — capabilities can lag a backend that forgot to override them.
//!
//! Method groups on [`ExecutableGraph`](crate::ExecutableGraph):
//!
//! | Group | Methods |
//! |-------|---------|
//! | Core run | `set_param`, `run`, `run_raw`, `run_slots`, `arena_ptr`, `finalize_params` |
//! | Clone | `clone_box` |
//! | Extent / RNG | `set_active_extent`, `set_rng`, `rng` |
//! | MoE | `set_moe_resident_experts*`, `enable_moe_topk_capture`, `take_moe_*` |
//! | Persistent / GPU handles | `bind_handle`, `read_handle`, `bind_gpu_handle`, … |
//! | KV resident | `register_kv_row_feed`, `feed_kv_row`, `seed_resident_kv_prefix_from`, … |
//! | Typed I/O | `set_param_typed`, `run_typed` |
//! | Async pipeline | `commit_no_wait`, `sync_pending`, `run_pipelined` |

/// Advisory feature flags for a compiled executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutableCapabilities {
    /// [`crate::ExecutableGraph::clone_box`] is implemented (not the default panic).
    pub clone: bool,
    /// MoE residency / TopK capture hooks are live.
    pub moe: bool,
    /// Persistent named handles (`bind_handle` / `read_handle`).
    pub persistent_handles: bool,
    /// GPU-resident input handles (`bind_gpu_handle`, feeds, …).
    pub gpu_handles: bool,
    /// Device-resident KV row feed / D2D seed path.
    pub kv_resident: bool,
    /// Native non-F32 `set_param_typed` / `run_typed` (not F32-only default).
    pub typed_io: bool,
    /// `commit_no_wait` / `run_pipelined` do real async work.
    pub async_pipeline: bool,
    /// `set_active_extent` is honored for bucketed decode.
    pub active_extent: bool,
}

impl ExecutableCapabilities {
    /// All bits clear — every optional hook is a no-op / unsupported.
    pub const NONE: Self = Self {
        clone: false,
        moe: false,
        persistent_handles: false,
        gpu_handles: false,
        kv_resident: false,
        typed_io: false,
        async_pipeline: false,
        active_extent: false,
    };

    /// Human-readable list of enabled capability names (for `device_report`).
    pub fn enabled_names(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.clone {
            out.push("clone");
        }
        if self.moe {
            out.push("moe");
        }
        if self.persistent_handles {
            out.push("persistent_handles");
        }
        if self.gpu_handles {
            out.push("gpu_handles");
        }
        if self.kv_resident {
            out.push("kv_resident");
        }
        if self.typed_io {
            out.push("typed_io");
        }
        if self.async_pipeline {
            out.push("async_pipeline");
        }
        if self.active_extent {
            out.push("active_extent");
        }
        out
    }
}
