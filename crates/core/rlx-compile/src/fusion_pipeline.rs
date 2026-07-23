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
use crate::io_output_gate::SelectPeaksOnlyOutputs;
use rlx_fusion::control_flow::LowerControlFlow;
use rlx_fusion::fk_fusion::{
    DecomposeFusionRegions, FuseBatchPreprocess, FuseRegionPrologue, MarkBatchSliceRegions,
    MarkTransformRegions,
};
use rlx_fusion::fusion::{
    FuseAdaLayerNorm, FuseAttentionBlock, FuseConvAffineAct, FuseConvBiasAct, FuseGatedResidual,
    FuseMatMulBiasAct, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, FuseTransformerLayer,
    MarkElementwiseRegions, UnfuseElementwiseRegions,
};
use rlx_fusion::limits::{FusionLimits, with_fusion_limits};
use rlx_fusion::lower_dot_general::LowerDotGeneral;
use rlx_fusion::pass::{Pass, run_passes};
use rlx_ir::Graph;

use crate::fusion_target::with_fusion_target;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionOptions {
    /// Skip all pattern fusions (Metal: `RLX_METAL_NO_FUSION`).
    pub skip_fusion: bool,
    /// Break `ElementwiseRegion` back into primitives after marking.
    pub unfuse_elementwise_regions: bool,
    /// Keep fused `ElementwiseRegion` through lowering (env: `RLX_KEEP_ELEMENTWISE_REGIONS`).
    pub keep_elementwise_regions: bool,
    /// Decompose FKL-style transform / batch regions before backend lowering.
    pub decompose_fusion_regions: bool,
    /// Run FKL passes (`MarkTransformRegions`, prologue, batch). Env: `RLX_NO_FK_FUSION=1` disables.
    pub fk_fusion: bool,
    /// Fold `ResizeNearest2x` into `ElementwiseRegion` prologue. Env: `RLX_FUSE_REGION_PROLOGUE=0` disables.
    pub fuse_region_prologue: bool,
    /// Merge parallel region slices into `BatchElementwiseRegion`. Env: `RLX_FUSE_BATCH_PREPROCESS=0` disables.
    pub fuse_batch_preprocess: bool,
    /// Keep `TransformRegion` / `BatchElementwiseRegion` in MIR for native lowering. Env: `RLX_NATIVE_FK_REGIONS=1`.
    pub native_fk_regions: bool,
    /// Caps for fused elementwise chains (encoder / scratch limits).
    pub fusion_limits: FusionLimits,
    /// Skip Conv+Bias+Act fusion (`RLX_DISABLE_CONV_BIAS_ACT_FUSION`).
    pub disable_conv_bias_act_fusion: bool,
    /// Disable IO-gated peaks-only output selection (`RLX_NO_IO_PEAKS_OUTPUT`).
    pub no_io_peaks_output: bool,
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            skip_fusion: false,
            unfuse_elementwise_regions: false,
            keep_elementwise_regions: false,
            decompose_fusion_regions: false,
            fk_fusion: true,
            fuse_region_prologue: true,
            fuse_batch_preprocess: true,
            native_fk_regions: false,
            fusion_limits: FusionLimits::default(),
            disable_conv_bias_act_fusion: false,
            no_io_peaks_output: false,
        }
    }
}

impl FusionOptions {
    /// Read Metal-specific env overrides.
    pub fn from_metal_env() -> Self {
        Self {
            skip_fusion: rlx_ir::env::flag("RLX_METAL_NO_FUSION"),
            unfuse_elementwise_regions: rlx_ir::env::flag("RLX_METAL_UNFUSE_REGIONS"),
            keep_elementwise_regions: rlx_ir::env::flag("RLX_KEEP_ELEMENTWISE_REGIONS"),
            decompose_fusion_regions: rlx_ir::env::flag("RLX_DECOMPOSE_FUSION_REGIONS"),
            fk_fusion: !rlx_ir::env::flag("RLX_NO_FK_FUSION"),
            fuse_region_prologue: if rlx_ir::env::is_unset("RLX_FUSE_REGION_PROLOGUE") {
                true
            } else {
                rlx_ir::env::flag("RLX_FUSE_REGION_PROLOGUE")
            },
            fuse_batch_preprocess: if rlx_ir::env::is_unset("RLX_FUSE_BATCH_PREPROCESS") {
                true
            } else {
                rlx_ir::env::flag("RLX_FUSE_BATCH_PREPROCESS")
            },
            native_fk_regions: rlx_ir::env::flag("RLX_NATIVE_FK_REGIONS"),
            ..Self::default()
        }
    }

    /// Merge session options with compile-time env overrides.
    pub fn merge_env(mut self) -> Self {
        if rlx_ir::env::flag("RLX_METAL_NO_FUSION") {
            self.skip_fusion = true;
        }
        if rlx_ir::env::flag("RLX_METAL_UNFUSE_REGIONS") {
            self.unfuse_elementwise_regions = true;
        }
        if rlx_ir::env::flag("RLX_KEEP_ELEMENTWISE_REGIONS") {
            self.keep_elementwise_regions = true;
        }
        if rlx_ir::env::flag("RLX_DECOMPOSE_FUSION_REGIONS") {
            self.decompose_fusion_regions = true;
        }
        if rlx_ir::env::flag("RLX_NO_FK_FUSION") {
            self.fk_fusion = false;
        }
        if !rlx_ir::env::is_unset("RLX_FUSE_REGION_PROLOGUE") {
            self.fuse_region_prologue = rlx_ir::env::flag("RLX_FUSE_REGION_PROLOGUE");
        }
        if !rlx_ir::env::is_unset("RLX_FUSE_BATCH_PREPROCESS") {
            self.fuse_batch_preprocess = rlx_ir::env::flag("RLX_FUSE_BATCH_PREPROCESS");
        }
        if rlx_ir::env::flag("RLX_NATIVE_FK_REGIONS") {
            self.native_fk_regions = true;
        }
        if rlx_ir::env_registry::flag("RLX_NO_NATIVE_FK_REGIONS") {
            self.native_fk_regions = false;
        }
        if rlx_ir::env_registry::flag("RLX_DISABLE_CONV_BIAS_ACT_FUSION") {
            self.disable_conv_bias_act_fusion = true;
        }
        if rlx_ir::env_registry::flag("RLX_NO_IO_PEAKS_OUTPUT") {
            self.no_io_peaks_output = true;
        }
        self
    }

    /// GPU-class targets keep native FKL regions unless opted out.
    pub fn apply_native_fk_defaults(mut self, target: FusionTarget) -> Self {
        if rlx_ir::env::flag("RLX_NO_NATIVE_FK_REGIONS") {
            self.native_fk_regions = false;
            return self;
        }
        if self.native_fk_regions || rlx_ir::env::flag("RLX_NATIVE_FK_REGIONS") {
            self.native_fk_regions = true;
            return self;
        }
        if matches!(
            target,
            FusionTarget::Metal
                | FusionTarget::Cuda
                | FusionTarget::Rocm
                | FusionTarget::Wgpu
                | FusionTarget::Mlx
                | FusionTarget::Tpu
        ) {
            self.native_fk_regions = true;
        }
        self
    }

    /// CPU executes element-wise chains as per-op thunks — mark then unfuse.
    pub fn for_cpu() -> Self {
        Self {
            unfuse_elementwise_regions: true,
            fusion_limits: FusionLimits::UNBOUNDED,
            ..Self::default()
        }
    }

    /// Metal keeps RMSNorm / matmul fusions but unfuses `ElementwiseRegion`
    /// (fused MSL mis-lowers long chains on deep transformer graphs).
    pub fn for_metal() -> Self {
        let mut opts = Self::from_metal_env();
        opts.unfuse_elementwise_regions = true;
        opts
    }

    /// wgpu region kernel only supports trailing/scalar broadcast via
    /// modulus — unfuse so LegalizeBroadcast Expand + Binary run separately.
    pub fn for_wgpu() -> Self {
        let keep = rlx_ir::env::flag("RLX_KEEP_ELEMENTWISE_REGIONS");
        Self {
            unfuse_elementwise_regions: !keep,
            keep_elementwise_regions: keep,
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
    target: FusionTarget,
) -> Vec<&'static dyn Pass> {
    let opts = opts.apply_native_fk_defaults(target);
    if opts.skip_fusion {
        return vec![&LowerControlFlow, &LowerDotGeneral];
    }

    let mut passes: Vec<&'static dyn Pass> = vec![&LowerControlFlow, &LowerDotGeneral];

    // ORDER: FuseMatMulBiasAct first, then FuseAttentionBlock. The block-level
    // pass matches the post-fusion shape
    //   FusedMatMulBiasAct(qkv) → narrow×3 → Attention → FusedMatMulBiasAct(out)
    // which is the pattern BERT-family encoders actually present after the
    // per-layer matmul+bias fusion has collapsed Q, K, V, and out projections.
    if supports_op(supported, OpKind::FusedMatMulBiasAct) {
        passes.push(&FuseMatMulBiasAct);
    }
    // Conv + bias + activation → cuDNN's fused conv-bias-activation (CUDA only
    // claims `FusedConvBiasAct`; every other backend keeps it decomposed).
    // `RLX_DISABLE_CONV_BIAS_ACT_FUSION=1` skips it (ablation / A-B benchmarking
    // vs the unfused conv+bias+act path). `FuseConvAffineAct` folds a
    // host-pre-folded BatchNorm affine (`conv→Mul→Add→relu`) into the same op.
    if supports_op(supported, OpKind::FusedConvBiasAct) && !opts.disable_conv_bias_act_fusion {
        passes.push(&FuseConvBiasAct);
        passes.push(&FuseConvAffineAct);
    }
    // Block-level fusion: `Op::FusedAttentionBlock`. All backends that claim
    // this op now produce parity-correct output (the MLX
    // `Op::FusedAttentionBlock` lowering at `rlx-mlx/src/lower.rs:1689`
    // historically diverged on `MaskKind::Custom` BERT masks because it
    // bypassed the binary→additive conversion and the contiguous
    // materialization the unfused `Op::Attention` path applies — fixed
    // alongside this pass landing).
    if supports_op(supported, OpKind::FusedAttentionBlock) {
        passes.push(&FuseAttentionBlock);
    }
    // FuseResidualLN must run BEFORE FuseTransformerLayer: the layer-level
    // pass matches `FAB → FusedResidualLN → FMBA(GeLU) → FMBA → FusedResidualLN`
    // and needs the residual+LN ops already collapsed.
    if supports_op(supported, OpKind::FusedResidualLN) {
        passes.push(&FuseResidualLN);
    }
    if supports_op(supported, OpKind::FusedResidualRmsNorm) {
        passes.push(&FuseResidualRmsNorm);
    }
    // DiT adaLN-Zero / gated residual — after residual-LN so we don't
    // compete for Add→LN patterns; gated residual is independent.
    if supports_op(supported, OpKind::AdaLayerNorm) {
        passes.push(&FuseAdaLayerNorm);
    }
    if supports_op(supported, OpKind::GatedResidual) {
        passes.push(&FuseGatedResidual);
    }
    passes.push(&FuseRmsNormReshape);

    // Layer-level fusion runs AFTER FuseResidualLN so it can match the
    // post-fusion shape `FAB → FusedResidualLN → FMBA(GeLU) → FMBA →
    // FusedResidualLN`. Opt-in via `RLX_ENABLE_FUSE_TRANSFORMER_LAYER`
    // because backend perf wins are uneven: WGPU un-fuses with no
    // dispatch reduction; MLX's lowering is correct (per the FAB fix
    // above) but the MLX `compile()` already collapses sub-ops, so the
    // extra IR-level fusion doesn't beat the natural pipeline. The pass
    // exists for backends planning a monolithic transformer-layer kernel.
    if rlx_ir::env::flag("RLX_ENABLE_FUSE_TRANSFORMER_LAYER")
        && supports_op(supported, OpKind::FusedTransformerLayer)
        && supports_op(supported, OpKind::FusedAttentionBlock)
    {
        passes.push(&FuseTransformerLayer);
    }

    if supports_op(supported, OpKind::FusedSwiGLU) {
        passes.push(&FuseSwiGLUDualMatmul);
    }
    // Opt out: `RLX_NO_SHARED_INPUT_MATMUL=1` (debug / parity vs unfused AdaLN).
    if supports_op(supported, OpKind::MatMul) && !rlx_ir::env::flag("RLX_NO_SHARED_INPUT_MATMUL") {
        passes.push(&FuseSharedInputMatMul);
    }
    if supports_op(supported, OpKind::FusedSwiGLU) {
        passes.push(&FuseSwiGLU);
    }

    // Mark eligible element-wise chains only when the backend keeps regions.
    // CPU/Metal unfuse immediately afterward — marking first duplicates the
    // full graph for no benefit.
    let keep_regions =
        supports_op(supported, OpKind::ElementwiseRegion) && !opts.unfuse_elementwise_regions;
    if keep_regions {
        passes.push(&MarkElementwiseRegions);
    }
    if opts.fk_fusion {
        passes.push(&MarkBatchSliceRegions);
        passes.push(&MarkTransformRegions);
        if opts.fuse_region_prologue {
            passes.push(&FuseRegionPrologue);
        }
        if opts.fuse_batch_preprocess {
            passes.push(&FuseBatchPreprocess);
        }
    }
    let backend_native_fk = supports_op(supported, OpKind::TransformRegion)
        && supports_op(supported, OpKind::BatchElementwiseRegion);
    let keep_native_fk = opts.native_fk_regions && backend_native_fk;
    if opts.decompose_fusion_regions || !keep_native_fk {
        passes.push(&DecomposeFusionRegions);
    }
    if !keep_regions {
        let unfuse = if matches!(target, FusionTarget::Cpu) {
            &UnfuseElementwiseRegions::FOR_CPU
        } else {
            &UnfuseElementwiseRegions::FOR_GPU
        };
        passes.push(unfuse);
    }

    if supports_op(supported, OpKind::Fft) && supports_op(supported, OpKind::WelchPeaks) {
        passes.push(&SelectPeaksOnlyOutputs);
    }

    finish_pipeline(passes)
}

/// FKL passes to run after [`MarkElementwiseRegions`] (e.g. `TpuExecutable::compile`).
pub fn fk_passes_after_elementwise_regions(
    supported: &[OpKind],
    opts: FusionOptions,
) -> Vec<&'static dyn Pass> {
    let mut passes: Vec<&'static dyn Pass> = Vec::new();
    if !opts.fk_fusion {
        let backend_native_fk = supports_op(supported, OpKind::TransformRegion)
            && supports_op(supported, OpKind::BatchElementwiseRegion);
        let keep_native_fk = opts.native_fk_regions && backend_native_fk;
        if opts.decompose_fusion_regions || !keep_native_fk {
            passes.push(&DecomposeFusionRegions);
        }
        return finish_pipeline(passes);
    }
    passes.push(&MarkBatchSliceRegions);
    passes.push(&MarkTransformRegions);
    if opts.fuse_region_prologue {
        passes.push(&FuseRegionPrologue);
    }
    if opts.fuse_batch_preprocess {
        passes.push(&FuseBatchPreprocess);
    }
    let backend_native_fk = supports_op(supported, OpKind::TransformRegion)
        && supports_op(supported, OpKind::BatchElementwiseRegion);
    let keep_native_fk = opts.native_fk_regions && backend_native_fk;
    if opts.decompose_fusion_regions || !keep_native_fk {
        passes.push(&DecomposeFusionRegions);
    }
    finish_pipeline(passes)
}

/// IO gate decision for a rewrite on `target` (convenience for compile passes / model crates).
pub fn should_fuse_with_target(
    target: FusionTarget,
    before: &crate::fusion_benefit::GraphIoProfile,
    after: &crate::fusion_benefit::GraphIoProfile,
) -> bool {
    io_fusion_gate_for_target(target).should_fuse(before, after)
}

/// Phase 3 — IO-aware gate defaults for fusion rewrites on `target`.
pub fn io_fusion_gate_for_target(target: FusionTarget) -> crate::fusion_benefit::IoFusionGate {
    use crate::fusion_benefit::IoFusionGate;
    match target {
        FusionTarget::Metal | FusionTarget::Mlx => IoFusionGate {
            dispatch_ns: 500.0,
            roundtrip_ns: 5_000.0,
            memory_bw: 200.0,
            host_readback_bw: 200.0,
            unified_memory: true,
            host_thunk_penalty_ns: 2_000_000.0,
            min_gain_ns: 1_000.0,
        },
        FusionTarget::Cuda | FusionTarget::Rocm => IoFusionGate {
            dispatch_ns: 2_000.0,
            roundtrip_ns: 20_000.0,
            memory_bw: 800.0,
            host_readback_bw: 50.0,
            unified_memory: false,
            host_thunk_penalty_ns: 15_000_000.0,
            min_gain_ns: 5_000.0,
        },
        FusionTarget::Wgpu | FusionTarget::Tpu => IoFusionGate {
            dispatch_ns: 3_000.0,
            roundtrip_ns: 30_000.0,
            memory_bw: 100.0,
            host_readback_bw: 40.0,
            unified_memory: false,
            host_thunk_penalty_ns: 25_000_000.0,
            min_gain_ns: 10_000.0,
        },
        FusionTarget::Cpu => IoFusionGate {
            dispatch_ns: 50.0,
            roundtrip_ns: 0.0,
            memory_bw: 50.0,
            host_readback_bw: 50.0,
            unified_memory: true,
            host_thunk_penalty_ns: 0.0,
            min_gain_ns: 0.0,
        },
    }
}

/// Return the ordered fusion passes for `target`.
pub fn fusion_passes(target: FusionTarget, opts: FusionOptions) -> Vec<&'static dyn Pass> {
    let mut opts = opts;
    // CPU thunks execute element-wise chains per-op. Metal's fused
    // `ElementwiseRegion` MSL kernel mis-lowers long chains on deep
    // transformer graphs (NaNs past ~14 blocks); keep FAB/RMSNorm fusions.
    if !opts.keep_elementwise_regions
        && matches!(target, FusionTarget::Cpu | FusionTarget::Metal)
        && !opts.unfuse_elementwise_regions
    {
        opts.unfuse_elementwise_regions = true;
    }
    if opts.fusion_limits == FusionLimits::default() {
        opts.fusion_limits = fusion_limits_for_target(target);
    }
    opts = opts.apply_native_fk_defaults(target);
    fusion_passes_for_supported(supported_for_target(target), opts, target)
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
            AdaLayerNorm,
            GatedResidual,
        ],
        FusionTarget::Metal => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            AdaLayerNorm,
            GatedResidual,
        ],
        FusionTarget::Mlx => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            AdaLayerNorm,
            GatedResidual,
        ],
        FusionTarget::Wgpu => &[
            MatMul,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedSwiGLU,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            AdaLayerNorm,
            GatedResidual,
            FusedAttentionBlock,
            FusedTransformerLayer,
        ],
        // CUDA lowers `FusedConvBiasAct` to cuDNN's fused conv-bias-activation.
        // ROCm/MIOpen has no fused-conv path yet, so it is NOT claimed there —
        // its conv-bias-activation stays decomposed until a MIOpen lowering lands.
        FusionTarget::Cuda => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedMatMulBiasAct,
            FusedConvBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            AdaLayerNorm,
            GatedResidual,
        ],
        FusionTarget::Rocm => &[
            MatMul,
            DotGeneral,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedMatMulBiasAct,
            FusedResidualLN,
            FusedResidualRmsNorm,
            AdaLayerNorm,
            GatedResidual,
        ],
        FusionTarget::Tpu => &[
            MatMul,
            ElementwiseRegion,
            TransformRegion,
            BatchElementwiseRegion,
            FusedMatMulBiasAct,
            FusedResidualLN,
            AdaLayerNorm,
            GatedResidual,
        ],
    }
}

fn finish_pipeline(mut passes: Vec<&'static dyn Pass>) -> Vec<&'static dyn Pass> {
    passes.push(&DeadCodeElimination);
    passes
}

/// Run the fusion pipeline for `target` on a MIR graph (IO-gated passes included).
pub fn run_fusion_pipeline(
    graph: Graph,
    target: FusionTarget,
    supported: &[OpKind],
    opts: FusionOptions,
) -> Graph {
    let mut opts = opts.apply_native_fk_defaults(target);
    if opts.fusion_limits == FusionLimits::default() {
        opts.fusion_limits = fusion_limits_for_target(target);
    }
    let limits = opts.fusion_limits;
    let passes = fusion_passes_for_supported(supported, opts, target);
    with_fusion_target(target, || {
        with_fusion_limits(limits, || run_passes(graph, &passes, false))
    })
}

impl FusionOptions {
    /// Canonical option set for a target, matching what each backend's
    /// compile path uses: CPU unfuses element-wise regions (it executes them
    /// as per-op thunks), Metal/MLX keep matmul/norm fusions but unfuse long
    /// element-wise chains, wgpu unfuses for its modulus-broadcast kernel, and
    /// the remaining GPU-class targets take the defaults (native FKL regions
    /// are then enabled by [`apply_native_fk_defaults`](Self::apply_native_fk_defaults)).
    /// This is the option set the one-call [`fuse`] / [`Fuse`] entries pick.
    pub fn for_target(target: FusionTarget) -> Self {
        match target {
            FusionTarget::Cpu => Self::for_cpu(),
            FusionTarget::Metal | FusionTarget::Mlx => Self::for_metal(),
            FusionTarget::Wgpu => Self::for_wgpu(),
            FusionTarget::Cuda | FusionTarget::Rocm | FusionTarget::Tpu => Self::default(),
        }
    }
}

/// One-call fusion: optimize `graph` for `target` with canonical defaults.
///
/// The ergonomic front door over [`run_fusion_pipeline`]. Callers no longer
/// assemble the supported-op table or pick a per-target [`FusionOptions`] —
/// both easy to get wrong. Exactly equivalent to:
///
/// ```ignore
/// run_fusion_pipeline(
///     graph, target,
///     supported_for_target(target),
///     FusionOptions::for_target(target),
/// )
/// ```
///
/// `Op::Input` / `Op::Param` leaves survive fusion, so recover handles into
/// the rewritten graph by name with [`rlx_ir::Graph::node_id_by_name`];
/// outputs stay positionally stable in `graph.outputs`. For custom limits,
/// skip-fusion, or a change report, use the [`Fuse`] builder.
pub fn fuse(graph: Graph, target: FusionTarget) -> Graph {
    Fuse::new(target).run(graph)
}

/// Fluent fusion builder. `Fuse::new(target)` starts from the canonical
/// [`FusionOptions::for_target`] defaults; setters override individual toggles
/// before [`run`](Self::run) / [`run_with_report`](Self::run_with_report).
///
/// ```ignore
/// let (fused, report) = Fuse::new(FusionTarget::Cpu)
///     .limits(FusionLimits::UNBOUNDED)
///     .run_with_report(graph);
/// eprintln!("{}", report.summary_line());
/// ```
#[derive(Debug, Clone)]
pub struct Fuse {
    target: FusionTarget,
    options: FusionOptions,
}

impl Fuse {
    /// Start from the canonical defaults for `target`.
    pub fn new(target: FusionTarget) -> Self {
        Self {
            target,
            options: FusionOptions::for_target(target),
        }
    }

    /// Replace the whole option set.
    pub fn options(mut self, options: FusionOptions) -> Self {
        self.options = options;
        self
    }

    /// Override the element-wise fusion limits.
    pub fn limits(mut self, limits: FusionLimits) -> Self {
        self.options.fusion_limits = limits;
        self
    }

    /// Disable all pattern fusion (lowering legalization only).
    pub fn skip_fusion(mut self, skip: bool) -> Self {
        self.options.skip_fusion = skip;
        self
    }

    /// Layer compile-time env overrides (`RLX_*`) on top of the builder state.
    pub fn merge_env(mut self) -> Self {
        self.options = self.options.merge_env();
        self
    }

    /// Run the pipeline, returning the optimized graph.
    pub fn run(self, graph: Graph) -> Graph {
        run_fusion_pipeline(
            graph,
            self.target,
            supported_for_target(self.target),
            self.options,
        )
    }

    /// Run the pipeline and also return a [`FusionReport`] describing the
    /// before→after change (op/region deltas, fusions left on the table) for
    /// logging or assertions.
    ///
    /// [`FusionReport`]: rlx_fusion::fusion_report::FusionReport
    pub fn run_with_report(self, graph: Graph) -> (Graph, rlx_fusion::fusion_report::FusionReport) {
        let before = graph.clone();
        let after = self.run(graph);
        let report = rlx_fusion::fusion_report::FusionReport::analyze(&before, &after);
        (after, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_FK_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The one-call `fuse` / `Fuse` API folds matmul+bias+act on CPU and keeps
    /// Input/Param leaves recoverable by name in the rewritten graph — the
    /// ergonomic contract callers rely on instead of threading a remap table.
    #[test]
    fn fuse_one_call_collapses_matmul_bias_act_and_keeps_handles() {
        use rlx_ir::op::{Activation, BinaryOp};
        use rlx_ir::{DType, Graph, Shape};

        let f = DType::F32;
        let mut g = Graph::new("mm_bias_act");
        let x = g.input("x", Shape::new(&[4, 8], f));
        let w = g.param("w", Shape::new(&[8, 16], f));
        let bias = g.param("bias", Shape::new(&[16], f));
        let mm = g.matmul(x, w, Shape::new(&[4, 16], f));
        let ba = g.binary(BinaryOp::Add, mm, bias, Shape::new(&[4, 16], f));
        let act = g.activation(Activation::Relu, ba, Shape::new(&[4, 16], f));
        g.set_outputs(vec![act]);
        let before = g.len();

        let (fused, report) = Fuse::new(FusionTarget::Cpu).run_with_report(g);

        assert!(
            fused.len() < before,
            "matmul+bias+relu should fuse on CPU: {before} -> {} ({})",
            fused.len(),
            report.summary_line()
        );
        // Leaves survive a fusion rewrite; handles recoverable by name.
        assert!(fused.input_id("x").is_some(), "input x lost after fusion");
        assert!(fused.param_id("w").is_some(), "param w lost after fusion");
        assert!(fused.param_id("bias").is_some(), "param bias lost");
        assert!(fused.node_id_by_name("w").is_some());
        // Outputs stay positionally stable.
        assert_eq!(fused.outputs.len(), 1);
    }

    #[test]
    fn cpu_pipeline_includes_attention_block() {
        let passes = fusion_passes(FusionTarget::Cpu, FusionOptions::default());
        assert_eq!(
            passes.len(),
            19,
            "CPU default supported_ops omit Fft/WelchPeaks and mark_elementwise (unfuse-only backends skip mark); includes FuseAdaLayerNorm + FuseGatedResidual"
        );
        assert_eq!(passes[2].name(), "fuse_matmul_bias_act");
        assert_eq!(passes[3].name(), "fuse_attention_block");
        assert!(
            passes.iter().any(|p| p.name() == "fuse_region_prologue"),
            "default CPU pipeline should run FKL prologue fusion"
        );
        assert!(
            !passes
                .iter()
                .any(|p| p.name() == "mark_elementwise_regions"),
            "CPU unfuse backends should not mark elementwise regions before unfusing"
        );
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
            FusionTarget::Metal,
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
            FusionTarget::Cuda,
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
            FusionTarget::Cpu,
        );
        assert!(
            passes
                .iter()
                .any(|p| p.name() == "unfuse_elementwise_regions")
        );
    }

    #[test]
    fn metal_unfuses_elementwise_regions_by_default() {
        let passes = fusion_passes(FusionTarget::Metal, FusionOptions::default());
        assert!(
            passes
                .iter()
                .any(|p| p.name() == "unfuse_elementwise_regions")
        );
    }

    #[test]
    fn metal_default_unfuse_preserves_prologue_regions() {
        let mut g = rlx_ir::Graph::new("t");
        let shape_in = rlx_ir::Shape::new(&[1, 3, 8, 8], rlx_ir::DType::F32);
        let shape_out = rlx_ir::Shape::new(&[1, 3, 16, 16], rlx_ir::DType::F32);
        let x = g.input("x", shape_in);
        let up = g.add_node(rlx_ir::Op::ResizeNearest2x, vec![x], shape_out.clone());
        let r = g.add_node(
            rlx_ir::Op::Activation(rlx_ir::op::Activation::Relu),
            vec![up],
            shape_out,
        );
        g.set_outputs(vec![r]);

        let passes = fusion_passes(FusionTarget::Metal, FusionOptions::default());
        let out = rlx_fusion::pass::run_passes(g, &passes, false);
        assert!(out.nodes().iter().any(|n| {
            matches!(
                n.op,
                rlx_ir::Op::ElementwiseRegion {
                    prologue: rlx_ir::RegionPrologue::ResizeNearest2x,
                    ..
                }
            )
        }));
    }

    #[test]
    fn fk_passes_after_elementwise_includes_batch_fusion() {
        let opts = FusionOptions::default().apply_native_fk_defaults(FusionTarget::Tpu);
        let passes =
            fk_passes_after_elementwise_regions(supported_for_target(FusionTarget::Tpu), opts);
        let names: Vec<_> = passes.iter().map(|p| p.name()).collect();
        assert!(names.contains(&"mark_batch_slice_regions"));
        assert!(names.contains(&"fuse_batch_preprocess"));
        assert!(
            !names.contains(&"decompose_fusion_regions"),
            "TPU native FK defaults should keep batch/transform regions"
        );
    }

    #[test]
    fn tpu_native_fk_region_pass_policy() {
        let _lock = ENV_FK_TEST_LOCK.lock().unwrap();
        let default_passes = fusion_passes(FusionTarget::Tpu, FusionOptions::default());
        assert!(
            !default_passes
                .iter()
                .any(|p| p.name() == "decompose_fusion_regions"),
            "default TPU pipeline keeps batch/transform regions via native_fk_defaults"
        );

        rlx_ir::env::set("RLX_NO_NATIVE_FK_REGIONS", "1");
        let opt_out = fusion_passes(FusionTarget::Tpu, FusionOptions::default());
        rlx_ir::env::unset("RLX_NO_NATIVE_FK_REGIONS");
        assert!(
            opt_out
                .iter()
                .any(|p| p.name() == "decompose_fusion_regions"),
            "RLX_NO_NATIVE_FK_REGIONS should force decompose on TPU"
        );
    }

    #[test]
    fn native_fk_regions_skips_decompose_on_tpu() {
        let passes = fusion_passes(
            FusionTarget::Tpu,
            FusionOptions {
                native_fk_regions: true,
                decompose_fusion_regions: false,
                unfuse_elementwise_regions: false,
                ..FusionOptions::default()
            },
        );
        assert!(
            !passes
                .iter()
                .any(|p| p.name() == "decompose_fusion_regions"),
            "native_fk_regions should skip decompose on TPU when batch/transform are supported"
        );
    }

    #[test]
    fn native_fk_regions_skips_decompose_on_metal() {
        let passes = fusion_passes(
            FusionTarget::Metal,
            FusionOptions {
                native_fk_regions: true,
                decompose_fusion_regions: false,
                unfuse_elementwise_regions: false,
                ..FusionOptions::default()
            },
        );
        assert!(
            !passes
                .iter()
                .any(|p| p.name() == "decompose_fusion_regions"),
            "native_fk_regions should skip decompose when backend claims batch/transform ops"
        );
    }

    #[test]
    fn metal_keeps_elementwise_regions_when_requested() {
        let passes = fusion_passes(
            FusionTarget::Metal,
            FusionOptions {
                keep_elementwise_regions: true,
                unfuse_elementwise_regions: false,
                ..FusionOptions::default()
            },
        );
        assert!(
            !passes
                .iter()
                .any(|p| p.name() == "unfuse_elementwise_regions"),
            "keep_elementwise_regions should skip unfuse pass"
        );
        assert!(
            passes.iter().any(|p| p.name() == "fuse_region_prologue"),
            "FKL prologue fusion should still run"
        );
    }

    #[test]
    fn metal_audio_ops_pipeline_includes_peaks_output_gate() {
        let mut supported = supported_for_target(FusionTarget::Metal).to_vec();
        supported.push(OpKind::Fft);
        supported.push(OpKind::WelchPeaks);
        let passes =
            fusion_passes_for_supported(&supported, FusionOptions::default(), FusionTarget::Metal);
        assert!(
            passes
                .iter()
                .any(|p| p.name() == "select_peaks_only_outputs"),
            "Metal + Fft/WelchPeaks should run IO peaks-only output gate"
        );
    }

    #[test]
    fn should_fuse_with_target_matches_gate() {
        use crate::fusion_benefit::GraphIoProfile;
        let dense = GraphIoProfile {
            kernel_launches: 3,
            sync_points: 0,
            host_output_bytes: 33_554_432,
            device_traffic_bytes: 184_549_376,
        };
        let fused = GraphIoProfile {
            kernel_launches: 4,
            sync_points: 1,
            host_output_bytes: 1_048_576,
            device_traffic_bytes: 219_152_384,
        };
        assert!(should_fuse_with_target(FusionTarget::Metal, &dense, &fused));
        assert!(!should_fuse_with_target(FusionTarget::Wgpu, &dense, &fused));
    }
}
