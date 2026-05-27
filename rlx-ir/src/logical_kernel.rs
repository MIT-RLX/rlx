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
//! One logical kernel, many backends — dispatch policy and registry.
//!
//! A **logical kernel** is a single [`OpKind`] (e.g. [`OpKind::GaussianSplatRender`]) with a
//! documented semantic contract. Backends may provide a **native** implementation (fast path:
//! custom thunk, MSL, MPS, etc.). When native is unavailable or [`KernelDispatchPolicy::ForceCommon`]
//! is set, the compiler lowers to a **common** subgraph built only from primitive MIR ops so each
//! backend schedules the same math through its usual fusion/GEMM/elementwise paths.
//!
//! Native kernels are never removed from backends; common lowering is additive.

use crate::env;
use crate::op::OpKind;

pub mod splat_common;

/// When to use native backend kernels vs the shared IR common body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelDispatchPolicy {
    /// Native thunk when `OpKind` is in the backend `supported_ops`; else common IR lower.
    #[default]
    PreferNative,
    /// Always lower registered logical kernels to common IR (parity / minimal backends).
    ForceCommon,
    /// Never common-lower; legalization must succeed with native ops only.
    ForceNative,
}

impl KernelDispatchPolicy {
    pub fn from_env() -> Self {
        let v = env::var("KERNEL_DISPATCH").or_else(|| env::var("RLX_KERNEL_DISPATCH"));
        match v.as_deref() {
            Some("common") | Some("force_common") | Some("ForceCommon") => Self::ForceCommon,
            Some("native") | Some("force_native") | Some("ForceNative") => Self::ForceNative,
            _ => Self::PreferNative,
        }
    }
}

/// Registered logical kernel: native [`OpKind`] plus optional common lower pass name.
#[derive(Debug, Clone, Copy)]
pub struct LogicalKernelEntry {
    pub kind: OpKind,
    /// Human-readable id (logging / docs).
    pub name: &'static str,
}

/// Logical kernels that have a registered common IR body in `rlx-fusion`.
pub fn registered_logical_kernels() -> &'static [LogicalKernelEntry] {
    &[
        LogicalKernelEntry {
            kind: OpKind::GroupNorm,
            name: "group_norm",
        },
        LogicalKernelEntry {
            kind: OpKind::ResizeNearest2x,
            name: "resize_nearest_2x",
        },
        LogicalKernelEntry {
            kind: OpKind::GaussianSplatRender,
            name: "gaussian_splat_render",
        },
        LogicalKernelEntry {
            kind: OpKind::GaussianSplatRenderBackward,
            name: "gaussian_splat_render_backward",
        },
    ]
}

/// Per-compile overrides on top of [`KernelDispatchPolicy`].
#[derive(Debug, Clone, Copy, Default)]
pub struct KernelDispatchConfig {
    pub policy: KernelDispatchPolicy,
    /// Always common-lower these kinds (e.g. splat on CPU while keeping native matmul).
    pub force_common_kinds: &'static [OpKind],
    /// Never common-lower these kinds (overrides `ForceCommon` for listed kinds).
    pub force_native_kinds: &'static [OpKind],
}

impl KernelDispatchConfig {
    pub fn new(policy: KernelDispatchPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub fn from_env() -> Self {
        Self::new(KernelDispatchPolicy::from_env())
    }
}

/// Whether `kind` should be common-lowered for this backend claim set and config.
pub fn should_lower_to_common(
    kind: OpKind,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> bool {
    if !registered_logical_kernels().iter().any(|e| e.kind == kind) {
        return false;
    }
    if config.force_native_kinds.contains(&kind) {
        return false;
    }
    if config.force_common_kinds.contains(&kind) {
        return true;
    }
    match config.policy {
        KernelDispatchPolicy::ForceCommon => true,
        KernelDispatchPolicy::ForceNative => false,
        KernelDispatchPolicy::PreferNative => !supported.is_empty() && !supported.contains(&kind),
    }
}

/// Op kinds that appear in the graph and may need common lowering.
pub fn logical_kinds_in_graph(
    graph: &crate::Graph,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> Vec<OpKind> {
    let mut kinds = Vec::new();
    for node in graph.nodes() {
        let k = node.op.kind();
        if should_lower_to_common(k, supported, config) && !kinds.contains(&k) {
            kinds.push(k);
        }
    }
    kinds
}
