// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIR → MIR → LIR compile pipeline: fusion orchestration, legalization,
//! memory planning, and diagnostics.

pub mod algebraic_simplify;
pub mod compiler;
pub mod const_fold;
pub mod cse;
pub mod dce;
pub mod dispatch_report;
pub mod fusion_benefit;
pub mod fusion_pipeline;
pub mod fusion_target;
pub mod hardening;
pub mod inline;
pub mod inspect;
pub mod io_output_gate;
pub mod legalize;
pub mod legalize_broadcast;
pub mod memory;
pub mod numeric_lint;
pub mod param_hoist;
pub mod param_specialize;
pub mod precision;
pub mod promote_params;
pub mod quant_insert;
pub mod quant_propagate;
pub mod rewrite;
pub mod scaled_quant_insert;
pub mod sccp;
pub mod svg;

#[cfg(feature = "training")]
pub mod training_compile;

pub use algebraic_simplify::{AlgebraicSimplify, algebraic_simplify};
pub use compiler::{CompilePipeline, CompileResult};
pub use const_fold::ConstantFolding;
pub use cse::CommonSubexpressionElimination;
pub use dce::DeadCodeElimination;
pub use dispatch_report::{
    DispatchPath, KernelDispatchReport, KindDispatchSummary, analyze_dispatch,
    format_dispatch_report, maybe_log_dispatch_report, prepare_graph_for_backend_with_report,
};
pub use fusion_benefit::{
    FusionBenefit, GraphIoProfile as FusionIoProfile, IoFusionGate, fusion_benefit,
    profile_graph_io as profile_fusion_graph_io, profile_graph_io_outputs,
};
pub use fusion_pipeline::{
    FusionOptions, FusionTarget, fk_passes_after_elementwise_regions, fusion_limits_for_target,
    fusion_passes, fusion_passes_for_supported, io_fusion_gate_for_target, run_fusion_pipeline,
    should_fuse_with_target, supported_for_target, supports_op,
};
pub use fusion_target::{active_fusion_target, with_fusion_target};
pub use inline::inline_into;
pub use inspect::{
    PipelineInspect, inspect_compiled, inspect_fusion, inspect_pipeline, maybe_dump_pipeline,
};
pub use io_output_gate::SelectPeaksOnlyOutputs;
pub use legalize::{LegalizeResult, format_legalize_error, legalize_for_backend};
pub use legalize_broadcast::LegalizeBroadcast;
pub use memory::{
    ArenaWidthPolicy, MemoryPlanOptions, SharedWeightLayout, WeightSlot, is_pure_view,
    plan_memory_backward, plan_memory_f32_uniform, plan_memory_hybrid, plan_memory_native,
    plan_memory_native_in_order, plan_memory_with_options,
};
pub use numeric_lint::{NumericLint, lint_numerics};
pub use param_hoist::{HoistSplit, split_param_invariant};
pub use param_specialize::{SpecializeParams, specialize_params};
pub use precision::{AutoMixedPrecision, CastConfig, OpKind, Precision, PrecisionPolicy};
pub use promote_params::promote_params_to_inputs;
pub use quant_insert::{CalibrationEntry, CalibrationRecord, insert_q_dq};
pub use rewrite::{
    legalize_or_rewrite_for_backend, legalize_or_rewrite_for_backend_with_config,
    legalize_or_rewrite_for_backend_with_dispatch, lower_custom_ops, rewrite_for_backend,
    rewrite_for_backend_with_config, rewrite_for_backend_with_dispatch,
};
pub use rlx_fusion::FusionLimits;
pub use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
#[cfg(feature = "training")]
pub use training_compile::{TrainingCompileError, TrainingCompileResult, backward_cleanup_passes};
