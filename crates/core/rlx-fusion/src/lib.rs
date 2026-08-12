// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MIR fusion passes and fused-op decomposition.
//!
//! Pattern-matching fusion (`FuseMatMulBiasAct`, `FuseSwiGLU`, …) and
//! the inverse [`unfuse_fused_for_autodiff`] rewrite used before autodiff.

pub mod analysis;
pub mod control_flow;
pub mod fk_fusion;
pub mod fk_graphs;
pub mod fusion;
pub mod fusion_fragment;
pub mod fusion_report;
pub mod graph_rewrite;
pub mod limits;
pub mod lower_axial_rope2d;
pub mod lower_backward_ops;
pub mod lower_cumulative;
pub mod lower_dot_general;
pub mod lower_fake_quantize;
pub mod lower_fma;
pub mod lower_histogram;
pub mod lower_logical_kernels;
pub mod lower_loss_ops;
pub mod lower_pad;
pub mod lower_reduce_axes;
pub mod lower_scaled_grouped_matmul;
pub mod lower_slice;
pub mod lower_spectral;
pub mod lower_spline_activation;
pub mod lower_spline_backward;
pub mod lower_structural;
pub mod lower_synth_matmul;
pub mod lower_synth_matmul_backward;
pub mod lower_synth_reconstruct;
pub mod lower_vae_ops;
pub mod pass;
pub mod rewriter;
pub mod unfuse;

pub use control_flow::{
    LowerControlFlow, LowerScan, inline_if, inline_subgraph_into, inline_subgraph_into_outputs,
    maybe_unroll_scans, maybe_unroll_scans_budget, unroll_scan, unroll_while,
};
pub use fk_fusion::{
    DecomposeFusionRegions, FuseBatchPreprocess, FuseRegionPrologue, MarkBatchSliceRegions,
    MarkTransformRegions,
};
pub use fk_graphs::{
    batch_narrow_relu_primitive_graph, batch_narrow_relu_regions_graph, nchw, resize_relu_graph,
    resize_relu_region_graph,
};
pub use fusion::{
    FuseAdaLayerNorm, FuseAttentionBlock, FuseConvAffineAct, FuseConvBiasAct, FuseGatedResidual,
    FuseMatMulBiasAct, FuseMatMulResidual, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, FuseTransformerLayer,
    MarkElementwiseRegions, UnfuseElementwiseRegions, clip_elementwise_regions,
    fusible_conv_activation,
};
pub use fusion_fragment::{
    FusionFragment, FusionRole, fusion_fragments, is_registered_transform_op,
    prologue_for_transform_op, register_fusion_fragment, transform_chain_eligible,
};
pub use fusion_report::{FusionReport, MissReason, MissedFusion};
pub use limits::{FusionLimits, active_fusion_limits, with_fusion_limits};
pub use lower_axial_rope2d::{LowerAxialRope2d, lower_axial_rope2d};
pub use lower_backward_ops::LowerBackwardOps;
pub use lower_cumulative::LowerCumulative;
pub use lower_dot_general::LowerDotGeneral;
pub use lower_fake_quantize::{LowerFakeQuantize, lower_fake_quantize};
pub use lower_fma::LowerFma;
pub use lower_histogram::{LowerHistogram, lower_histogram};
pub use lower_logical_kernels::lower_logical_kernels;
pub use lower_loss_ops::LowerSoftmaxCrossEntropy;
pub use lower_pad::{LowerPad, lower_pad};
pub use lower_reduce_axes::LowerNonLastAxisReduce;
pub use lower_scaled_grouped_matmul::LowerScaledGroupedMatMul;
pub use lower_slice::{LowerSlice, lower_slice};
pub use lower_spectral::LowerSpectral;
pub use lower_spline_activation::LowerSplineActivation;
pub use lower_spline_backward::LowerSplineActivationBackward;
pub use lower_structural::LowerStructural;
pub use lower_synth_matmul::LowerSynthMatMul;
pub use lower_synth_matmul_backward::LowerSynthMatMulBackward;
pub use lower_synth_reconstruct::LowerSynthReconstruct;
pub use lower_vae_ops::{LowerBatchNormInference, LowerGroupNorm, LowerResizeNearest2x};
pub use pass::{
    Pass, register_ir_pass, registered_ir_passes, run_passes, run_registered_ir_passes,
};
pub use unfuse::{
    unfuse_attention_block, unfuse_dit_modulation, unfuse_fused_for_autodiff, unfuse_recurrent_ops,
};
