// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! MIR fusion passes and fused-op decomposition.
//!
//! Pattern-matching fusion (`FuseMatMulBiasAct`, `FuseSwiGLU`, …) and
//! the inverse [`unfuse_fused_for_autodiff`] rewrite used before autodiff.

pub mod control_flow;
pub mod fusion;
pub mod fusion_report;
pub mod limits;
pub mod lower_dot_general;
pub mod lower_logical_kernels;
pub mod lower_vae_ops;
pub mod pass;
pub mod unfuse;

pub use control_flow::{
    LowerControlFlow, inline_if, inline_subgraph_into, inline_subgraph_into_outputs, unroll_while,
};
pub use fusion::{
    FuseAttentionBlock, FuseMatMulBiasAct, FuseResidualLN, FuseResidualRmsNorm,
    FuseRmsNormReshape, FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul,
    MarkElementwiseRegions, UnfuseElementwiseRegions, clip_elementwise_regions,
};
pub use fusion_report::{FusionReport, MissReason, MissedFusion};
pub use limits::{FusionLimits, active_fusion_limits, with_fusion_limits};
pub use lower_dot_general::LowerDotGeneral;
pub use lower_logical_kernels::lower_logical_kernels;
pub use lower_vae_ops::{LowerGroupNorm, LowerResizeNearest2x};
pub use pass::{Pass, run_passes};
pub use unfuse::unfuse_fused_for_autodiff;
