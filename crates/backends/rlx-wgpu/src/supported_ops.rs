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

//! OpKinds this backend claims for legalization (`Backend::supported_ops`).
//!
//! Source of truth for the coverage matrix in `docs/op-coverage.md`.
//! Kept in the backend crate so adding an op is a local edit, not a change
//! to `rlx-runtime`'s mega-`backend.rs`.

use rlx_ir::OpKind;

pub const SUPPORTED_OPS: &[OpKind] = &[
    OpKind::Input,
    OpKind::Param,
    OpKind::Constant,
    OpKind::Activation,
    OpKind::Cast,
    OpKind::StopGradient,
    OpKind::Binary,
    OpKind::Compare,
    OpKind::Where,
    OpKind::Fma,
    OpKind::ElementwiseRegion,
    OpKind::TransformRegion,
    OpKind::BatchElementwiseRegion,
    OpKind::MatMul,
    OpKind::DotGeneral,
    OpKind::LayerNorm,
    OpKind::LayerNorm2d,
    OpKind::GroupNorm,
    OpKind::ResizeNearest2x,
    OpKind::RmsNorm,
    OpKind::Attention,
    OpKind::AttentionBackward,
    OpKind::RmsNormBackwardInput,
    OpKind::RmsNormBackwardGamma,
    OpKind::RmsNormBackwardBeta,
    // LayerNorm backward family:
    //   * Input  — single workgroup-per-row fused kernel.
    //   * Gamma  — two-dispatch (partial + reduce) that uses a tail
    //              scratch zone in the arena to hold per-chunk
    //              partial sums; the reduce kernel sums them.
    // Both beat the autodiff-decomposed primitive chain.
    OpKind::LayerNormBackwardInput,
    OpKind::LayerNormBackwardGamma,
    OpKind::RopeBackward,
    OpKind::CumsumBackward,
    OpKind::GatherBackward,
    // Native host conv backward (D2H→CPU→H2D). Kept out of the autodiff
    // decompose set: the static im2col-gather decomposition corrupts the
    // grad on wgpu when the conv input is a runtime input (arena aliasing);
    // the host kernels read the operands from the arena once and are correct.
    // Weight-grad + input-grad together = full multi-conv CNN training.
    OpKind::Conv2dBackwardWeight,
    OpKind::Conv2dBackwardInput,
    OpKind::Rope,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Narrow,
    OpKind::Concat,
    OpKind::Expand,
    OpKind::Gather,
    OpKind::Reverse,
    OpKind::Reduce,
    OpKind::Softmax,
    OpKind::SoftmaxCrossEntropy,
    OpKind::ArgMax,
    OpKind::ArgMin,
    OpKind::Cumsum,
    OpKind::TopK,
    OpKind::Sample,
    OpKind::Conv,
    OpKind::Im2Col,
    OpKind::Pool,
    OpKind::GroupedMatMul,
    OpKind::DequantGroupedMatMul,
    OpKind::DequantMoEWeights,
    OpKind::ScatterAdd,
    OpKind::ScatterNd,
    OpKind::ScatterElements,
    OpKind::GatherNd,
    OpKind::GatherElements,
    OpKind::SelectiveScan,
    OpKind::Lstm,
    OpKind::Gru,
    OpKind::Rnn,
    OpKind::Mamba2,
    // Keep GDN fused: unrolling into Expand/MatMul at legalize explodes the
    // arena (~100 GiB for Bonsai-27B @ max_seq=96). Host-eval via gdn_host,
    // matching Metal/CUDA/MLX claim-then-host pattern.
    OpKind::GatedDeltaNet,
    // Transposed conv (vision U-Net decoder) — host fallback via the CPU kernel.
    OpKind::ConvTranspose2d,
    // 3-D convs (volumetric UNETR-style decoders) — CPU NCDHW kernels.
    OpKind::Conv3d,
    OpKind::ConvTranspose3d,
    OpKind::DequantMatMul,
    OpKind::FusedMatMulBiasAct,
    OpKind::FusedResidualLN,
    OpKind::FusedResidualRmsNorm,
    OpKind::AdaLayerNorm,
    OpKind::GatedResidual,
    OpKind::AdaLayerNormBackward,
    OpKind::GatedResidualBackward,
    OpKind::FusedSwiGLU,
    OpKind::FusedAttentionBlock,
    OpKind::FusedTransformerLayer,
    // Native FFT (WGSL radix-2): f32 only, power-of-2 N ≤ 1024.
    // Anything outside that envelope panics at lowering with a
    // "pin to Device::Cpu" hint. No host fallback — WGPU has no
    // unified memory, so silent CPU round-trip would be a hidden
    // performance cliff.
    OpKind::Fft,
    // Op::Scan (arbitrary-body recurrence) via readback host fallback —
    // compile the body once, loop it on the CPU against an arena readback.
    // Enables IIR (`biquad`/`sosfilt`) on wgpu.
    OpKind::Scan,
    OpKind::ScanBackward,
    OpKind::ScanBackwardXs,
    OpKind::LogMel,
    OpKind::LogMelBackward,
    OpKind::WelchPeaks,
    // 3D Gaussian splat: native Metal / CPU reference per backend.
    OpKind::GaussianSplatRender,
    OpKind::GaussianSplatRenderBackward,
    OpKind::GaussianSplatPrepare,
    OpKind::GaussianSplatRasterize,
    OpKind::Custom,
    OpKind::RngNormal,
    OpKind::RngUniform,
    // Core Riemannian / SPD-manifold ops — no WGSL eigen kernel, so they
    // host-fallback to `rlx_cpu::spd` (F64) via an arena readback. See
    // `rlx_wgpu::spd_host`.
    OpKind::BiMap,
    OpKind::ReEig,
    OpKind::LogEig,
    OpKind::SpdBatchNorm,
    OpKind::SpdKarcherMean,
    OpKind::SpdKarcherMeanWeighted,
    OpKind::SpdLogMap,
    OpKind::SpdExpMap,
    OpKind::SpdParallelTransport,
    OpKind::SpdMatrixFnBatch,
    OpKind::ReEigBackward,
    OpKind::LogEigBackward,
    OpKind::SpdBatchNormBackwardX,
    OpKind::SpdBatchNormBackwardG,
    OpKind::SpdLogMapBackward,
    OpKind::SpdExpMapBackward,
    OpKind::SpdParallelTransportBackward,
    OpKind::SpdMatrixFnBatchBackward,
    OpKind::Eigh,
    OpKind::EighBackward,
    OpKind::EighBatch,
    OpKind::EighBatchBackward,
    // LoRA, If, While: not yet wired in wgpu — fail loudly.
];
