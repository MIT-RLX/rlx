// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! HIR → MIR → LIR compile pipeline: fusion orchestration, legalization,
//! memory planning, and diagnostics.

pub mod compiler;
pub mod const_fold;
pub mod dce;
pub mod dispatch_report;
pub mod fusion_pipeline;
pub mod hardening;
pub mod inline;
pub mod inspect;
pub mod legalize;
pub mod legalize_broadcast;
pub mod memory;
pub mod precision;
pub mod promote_params;
pub mod quant_insert;
pub mod quant_propagate;
pub mod rewrite;
pub mod svg;

#[cfg(feature = "training")]
pub mod training_compile;

pub use compiler::{CompilePipeline, CompileResult};
pub use const_fold::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use dispatch_report::{
    DispatchPath, KernelDispatchReport, KindDispatchSummary, analyze_dispatch,
    format_dispatch_report, maybe_log_dispatch_report, prepare_graph_for_backend_with_report,
};
pub use fusion_pipeline::{
    FusionOptions, FusionTarget, fusion_limits_for_target, fusion_passes,
    fusion_passes_for_supported, supported_for_target, supports_op,
};
pub use inline::inline_into;
pub use inspect::{
    PipelineInspect, inspect_compiled, inspect_fusion, inspect_pipeline, maybe_dump_pipeline,
};
pub use legalize::{LegalizeResult, format_legalize_error, legalize_for_backend};
pub use legalize_broadcast::LegalizeBroadcast;
pub use memory::{
    MemoryPlanOptions, SharedWeightLayout, WeightSlot, is_pure_view, plan_memory_backward,
    plan_memory_with_options,
};
pub use precision::{AutoMixedPrecision, CastConfig, OpKind, Precision, PrecisionPolicy};
pub use promote_params::promote_params_to_inputs;
pub use quant_insert::{CalibrationEntry, CalibrationRecord, insert_q_dq};
pub use rewrite::{
    legalize_or_rewrite_for_backend, legalize_or_rewrite_for_backend_with_config,
    legalize_or_rewrite_for_backend_with_dispatch, rewrite_for_backend,
    rewrite_for_backend_with_config, rewrite_for_backend_with_dispatch,
};
pub use rlx_fusion::FusionLimits;
pub use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
#[cfg(feature = "training")]
pub use training_compile::{TrainingCompileError, TrainingCompileResult, backward_cleanup_passes};
