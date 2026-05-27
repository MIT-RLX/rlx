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

//! Centralized fusion pass pipelines per backend target.
//!
//! [`fusion_passes_for_supported`] selects passes from a backend's
//! [`rlx_ir::OpKind`] claim set so fusion never emits fused ops the
//! target cannot lower. [`fusion_passes`] keeps the legacy
//! [`FusionTarget`] entry point and delegates to the same selector.

use rlx_ir::OpKind;

use crate::DeadCodeElimination;
use rlx_fusion::control_flow::LowerControlFlow;
use rlx_fusion::fusion::{
    FuseAttentionBlock, FuseMatMulBiasAct, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, MarkElementwiseRegions,
    UnfuseElementwiseRegions,
};
use rlx_fusion::limits::FusionLimits;
use rlx_fusion::lower_dot_general::LowerDotGeneral;
use rlx_fusion::pass::Pass;

/// Compile target that selects a fusion pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionTarget {
    Cpu,
    Metal,
    Mlx,
    Wgpu,
    Cuda,
    Rocm,
    Tpu,
}

/// Per-target fusion toggles (env-driven on Metal today).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusionOptions {
    /// Skip all pattern fusions (Metal: `RLX_METAL_NO_FUSION`).
    pub skip_fusion: bool,
    /// Break `ElementwiseRegion` back into primitives after marking.
    pub unfuse_elementwise_regions: bool,
    /// Caps for fused elementwise chains (encoder / scratch limits).
    pub fusion_limits: FusionLimits,
}

impl FusionOptions {
    /// Read Metal-specific env overrides.
    pub fn from_metal_env() -> Self {
        Self {
            skip_fusion: rlx_ir::env::flag("RLX_METAL_NO_FUSION"),
            unfuse_elementwise_regions: rlx_ir::env::flag("RLX_METAL_UNFUSE_REGIONS"),
            ..Self::default()
        }
    }

    /// CPU executes element-wise chains as per-op thunks — mark then unfuse.
    pub fn for_cpu() -> Self {
        Self {
            unfuse_elementwise_regions: true,
            fusion_limits: FusionLimits::UNBOUNDED,
            ..Self::default()
        }
    }
}

/// Elementwise-region caps for `target` (matches GPU kernel encoders).
pub fn fusion_limits_for_target(target: FusionTarget) -> FusionLimits {
    match target {
        FusionTarget::Cpu => FusionLimits::UNBOUNDED,
        FusionTarget::Tpu => FusionLimits {
            max_elementwise_steps: 32,
            max_elementwise_inputs: 16,
        },
        _ => FusionLimits::GPU_NATIVE,
    }
}

/// True when `supported` is empty (no claim) or contains `kind`.
#[inline]
pub fn supports_op(supported: &[OpKind], kind: OpKind) -> bool {
    supported.is_empty() || supported.contains(&kind)
}

/// Return the ordered fusion passes allowed for `supported`.
///
/// When `supported` is empty every fusion pass runs (legacy "accept
/// all" backends). When non-empty, each pattern fusion pass is
/// included only if the backend claims the fused [`OpKind`] it
/// emits. Lowering passes (`LowerControlFlow`, `LowerDotGeneral`) and
/// `FuseRmsNormReshape` (topology-only) always run unless
/// `skip_fusion` is set.
pub fn fusion_passes_for_supported(
    supported: &[OpKind],
    opts: FusionOptions,
) -> Vec<&'static dyn Pass> {
    if opts.skip_fusion {
        return vec![&LowerControlFlow, &LowerDotGeneral];
    }

    let mut passes: Vec<&'static dyn Pass> = vec![&LowerControlFlow, &LowerDotGeneral];

    if supports_op(supported, OpKind::FusedAttentionBlock) {
        passes.push(&FuseAttentionBlock);
    }
    if supports_op(supported, OpKind::FusedMatMulBiasAct) {
        passes.push(&FuseMatMulBiasAct);
    }
    if supports_op(supported, OpKind::FusedResidualLN) {
        passes.push(&FuseResidualLN);
    }
    if supports_op(supported, OpKind::FusedResidualRmsNorm) {
        passes.push(&FuseResidualRmsNorm);
    }
    passes.push(&FuseRmsNormReshape);

    if supports_op(supported, OpKind::FusedSwiGLU) {
        passes.push(&FuseSwiGLUDualMatmul);
    }
    if supports_op(supported, OpKind::MatMul) {
        passes.push(&FuseSharedInputMatMul);
    }
    if supports_op(supported, OpKind::FusedSwiGLU) {
        passes.push(&FuseSwiGLU);
    }

    // Mark eligible element-wise chains. Backends that don't lower
    // ElementwiseRegion natively unfuse immediately afterward.
    passes.push(&MarkElementwiseRegions);
    let keep_regions =
        supports_op(supported, OpKind::ElementwiseRegion) && !opts.unfuse_elementwise_regions;
    if !keep_regions {
        passes.push(&UnfuseElementwiseRegions);
    }

    finish_pipeline(passes)
}

/// Return the ordered fusion passes for `target`.
pub fn fusion_passes(target: FusionTarget, opts: FusionOptions) -> Vec<&'static dyn Pass> {
    let mut opts = opts;
    if matches!(target, FusionTarget::Cpu) && !opts.unfuse_elementwise_regions {
        opts.unfuse_elementwise_regions = true;
    }
    if opts.fusion_limits == FusionLimits::default() {
        opts.fusion_limits = fusion_limits_for_target(target);
    }
    fusion_passes_for_supported(supported_for_target(target), opts)
}

/// Per-target op claims used when a backend doesn't supply an explicit
/// `supported_ops` slice. Must stay aligned with each backend's
/// `*_SUPPORTED_OPS` in `rlx-runtime/src/backend.rs`.
pub fn supported_for_target(target: FusionTarget) -> &'static [OpKind] {
    use OpKind::*;
    match target {
        FusionTarget::Cpu => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            FusedAttentionBlock,
        ],
        FusionTarget::Metal => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
        ],
        FusionTarget::Mlx => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
        ],
        FusionTarget::Wgpu => &[
            MatMul,
            ElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            FusedAttentionBlock,
            FusedTransformerLayer,
        ],
        FusionTarget::Cuda | FusionTarget::Rocm => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
        ],
        FusionTarget::Tpu => &[
            MatMul,
            ElementwiseRegion,
            FusedMatMulBiasAct,
            FusedResidualLN,
        ],
    }
}

fn finish_pipeline(mut passes: Vec<&'static dyn Pass>) -> Vec<&'static dyn Pass> {
    passes.push(&DeadCodeElimination);
    passes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_pipeline_includes_attention_block() {
        let passes = fusion_passes(FusionTarget::Cpu, FusionOptions::default());
        assert_eq!(passes.len(), 13);
        assert_eq!(passes[2].name(), "fuse_attention_block");
        assert_eq!(passes.last().unwrap().name(), "dead_code_elimination");
    }

    #[test]
    fn metal_skip_fusion_only_lowers_dot() {
        let passes = fusion_passes(
            FusionTarget::Metal,
            FusionOptions {
                skip_fusion: true,
                ..FusionOptions::default()
            },
        );
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].name(), "LowerControlFlow");
        assert_eq!(passes[1].name(), "lower_dot_general");
    }

    #[test]
    fn metal_supported_ops_omit_attention_block_fusion() {
        let passes = fusion_passes_for_supported(
            supported_for_target(FusionTarget::Metal),
            FusionOptions::default(),
        );
        assert!(
            !passes.iter().any(|p| p.name() == "fuse_attention_block"),
            "Metal should not run FuseAttentionBlock"
        );
        assert!(
            passes.iter().any(|p| p.name() == "fuse_matmul_bias_act"),
            "Metal should fuse matmul+bias+act"
        );
    }

    #[test]
    fn cuda_supported_ops_fuse_matmul_bias_act() {
        let passes = fusion_passes_for_supported(
            supported_for_target(FusionTarget::Cuda),
            FusionOptions::default(),
        );
        assert!(
            passes.iter().any(|p| p.name() == "fuse_matmul_bias_act"),
            "CUDA should fuse matmul+bias+act when claimed"
        );
        assert!(
            !passes.iter().any(|p| p.name() == "fuse_swiglu"),
            "CUDA should not fuse SwiGLU"
        );
    }

    #[test]
    fn cpu_unfuses_elementwise_regions() {
        let passes = fusion_passes_for_supported(
            supported_for_target(FusionTarget::Cpu),
            FusionOptions::for_cpu(),
        );
        assert!(
            passes
                .iter()
                .any(|p| p.name() == "unfuse_elementwise_regions")
        );
    }

    #[test]
    fn metal_keeps_elementwise_regions_by_default() {
        let passes = fusion_passes_for_supported(
            supported_for_target(FusionTarget::Metal),
            FusionOptions::default(),
        );
        assert!(
            !passes
                .iter()
                .any(|p| p.name() == "unfuse_elementwise_regions")
        );
    }
}
