// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-backend caps for fused IR (elementwise region chains, etc.).

use std::cell::Cell;

/// Hardware / encoder limits for fusion passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionLimits {
    /// Max steps in one `Op::ElementwiseRegion` chain (Metal/wgpu/CUDA: 32).
    pub max_elementwise_steps: u32,
    /// Max distinct external inputs in one region (Metal/wgpu/CUDA: 16).
    pub max_elementwise_inputs: u32,
}

impl FusionLimits {
    /// Caps shared by native elementwise-region kernels today.
    pub const GPU_NATIVE: Self = Self {
        max_elementwise_steps: 32,
        max_elementwise_inputs: 16,
    };

    /// No practical cap — used when regions are unfused to primitives (CPU).
    pub const UNBOUNDED: Self = Self {
        max_elementwise_steps: u32::MAX,
        max_elementwise_inputs: u32::MAX,
    };
}

impl Default for FusionLimits {
    fn default() -> Self {
        Self::GPU_NATIVE
    }
}

thread_local! {
    static ACTIVE_LIMITS: Cell<FusionLimits> = Cell::new(FusionLimits::default());
}

/// Limits used by [`crate::fusion::MarkElementwiseRegions`] during this compile.
pub fn active_fusion_limits() -> FusionLimits {
    ACTIVE_LIMITS.with(|c| c.get())
}

/// Run `f` with `limits` installed for mark/clip passes (single-threaded compile).
pub fn with_fusion_limits<T>(limits: FusionLimits, f: impl FnOnce() -> T) -> T {
    ACTIVE_LIMITS.with(|c| {
        let prev = c.get();
        c.set(limits);
        let out = f();
        c.set(prev);
        out
    })
}
