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

//! RLX optimizer facade — re-exports [`rlx_fusion`], [`rlx_autodiff`], and
//! [`rlx_compile`] for backward-compatible `rlx_opt::` paths.
//!
//! # Crates
//!
//! | Crate | Role |
//! |-------|------|
//! | [`rlx_fusion`] | Fusion passes + [`rlx_fusion::unfuse_fused_for_autodiff`] |
//! | [`rlx_autodiff`] | `grad_with_loss`, `jvp`, `vmap`, [`prepare_graph_for_ad`] (feature `training`) |
//! | [`rlx_compile`] | [`CompilePipeline`], memory plan, legalization (feature `compile`) |
//!
//! # Features
//!
//! - `compile` (default) — HIR → MIR → LIR pipeline
//! - `training` (default) — autodiff transforms
//! - `full` — both

pub use rlx_fusion;

#[cfg(feature = "training")]
pub use rlx_autodiff;

#[cfg(feature = "compile")]
pub use rlx_compile;

// ── Backward-compatible module paths ─────────────────────────────

pub use rlx_fusion::control_flow;
pub use rlx_fusion::fusion;
pub use rlx_fusion::fusion_report;
pub use rlx_fusion::lower_dot_general;
pub use rlx_fusion::pass;
pub use rlx_fusion::unfuse;

#[cfg(feature = "training")]
pub mod autodiff {
    pub use rlx_autodiff::autodiff::*;
    pub use rlx_autodiff::prepare_ad::*;
}

#[cfg(feature = "training")]
pub mod autodiff_fwd {
    pub use rlx_autodiff::autodiff_fwd::*;
}

#[cfg(feature = "training")]
pub mod prepare_ad {
    pub use rlx_autodiff::prepare_ad::*;
}

#[cfg(feature = "compile")]
pub mod compiler {
    pub use rlx_compile::compiler::*;
}

#[cfg(feature = "compile")]
pub mod memory {
    pub use rlx_compile::memory::*;
}

#[cfg(feature = "compile")]
pub mod fusion_pipeline {
    pub use rlx_compile::fusion_pipeline::*;
}

#[cfg(feature = "compile")]
pub mod inspect {
    pub use rlx_compile::inspect::*;
}

#[cfg(feature = "compile")]
pub mod legalize {
    pub use rlx_compile::legalize::*;
}

#[cfg(feature = "compile")]
pub mod legalize_broadcast {
    pub use rlx_compile::legalize_broadcast::*;
}

#[cfg(feature = "compile")]
pub mod const_fold {
    pub use rlx_compile::const_fold::*;
}

#[cfg(feature = "compile")]
pub mod dce {
    pub use rlx_compile::dce::*;
}

#[cfg(feature = "compile")]
pub mod precision {
    pub use rlx_compile::precision::*;
}

#[cfg(feature = "compile")]
pub mod quant_insert {
    pub use rlx_compile::quant_insert::*;
}

#[cfg(feature = "compile")]
pub mod quant_propagate {
    pub use rlx_compile::quant_propagate::*;
}

#[cfg(feature = "compile")]
pub mod promote_params {
    pub use rlx_compile::promote_params::*;
}

#[cfg(feature = "compile")]
pub mod inline {
    pub use rlx_compile::inline::*;
}

#[cfg(feature = "compile")]
pub mod svg {
    pub use rlx_compile::svg::*;
}

#[cfg(feature = "training")]
pub mod vmap {
    pub use rlx_autodiff::vmap::*;
}

// ── Root re-exports (legacy `use rlx_opt::…`) ─────────────────────

pub use rlx_fusion::{
    FuseAttentionBlock, FuseMatMulBiasAct, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, FusionReport, LowerControlFlow,
    LowerDotGeneral, MarkElementwiseRegions, MissReason, MissedFusion, Pass,
    UnfuseElementwiseRegions, inline_if, inline_subgraph_into, run_passes,
    unfuse_fused_for_autodiff, unroll_while,
};

#[cfg(feature = "training")]
pub use rlx_autodiff::{
    AutodiffError, MirAutodiffExt, PrepareForAutodiff, grad, grad_with_loss, grad_with_loss_module,
    hvp, jvp, jvp_module, prepare_graph_for_ad, prepare_mir_for_ad, prepare_module_for_ad,
    quantized_weight_bits,
};

#[cfg(feature = "training")]
pub use rlx_autodiff::vmap::vmap;

#[cfg(feature = "training")]
pub use rlx_autodiff::autodiff::{convert_scans_for_ad, inline_custom_fn_for_autodiff};

#[cfg(all(feature = "compile", feature = "training"))]
pub use rlx_compile::{TrainingCompileError, TrainingCompileResult, backward_cleanup_passes};

#[cfg(feature = "compile")]
pub use rlx_compile::{
    AutoMixedPrecision, CalibrationEntry, CalibrationRecord, CastConfig, CompilePipeline,
    CompileResult, ConstantFolding, DeadCodeElimination, DispatchPath, FusionLimits, FusionOptions,
    FusionTarget, KernelDispatchConfig, KernelDispatchPolicy, KernelDispatchReport,
    KindDispatchSummary, LegalizeBroadcast, LegalizeResult, MemoryPlanOptions, OpKind,
    PipelineInspect, Precision, PrecisionPolicy, SharedWeightLayout, WeightSlot, analyze_dispatch,
    format_dispatch_report, format_legalize_error, fusion_limits_for_target, fusion_passes,
    fusion_passes_for_supported, inline_into, insert_q_dq, inspect_compiled, inspect_fusion,
    inspect_pipeline, is_pure_view, legalize_for_backend, legalize_or_rewrite_for_backend,
    legalize_or_rewrite_for_backend_with_config, legalize_or_rewrite_for_backend_with_dispatch,
    maybe_dump_pipeline, maybe_log_dispatch_report, plan_memory_backward, plan_memory_with_options,
    prepare_graph_for_backend_with_report, promote_params_to_inputs, rewrite_for_backend,
    rewrite_for_backend_with_config, rewrite_for_backend_with_dispatch, supported_for_target,
    supports_op,
};
