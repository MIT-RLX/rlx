// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    OpKind::Interpolate3d,
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
    OpKind::Conv3dBackwardWeight,
    OpKind::Conv3dBackwardInput,
    OpKind::MaxPool3dBackward,
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
    OpKind::CumProd,
    OpKind::CumMax,
    OpKind::TopK,
    OpKind::Sample,
    OpKind::Conv,
    OpKind::Im2Col,
    OpKind::Pool,
    OpKind::GroupedMatMul,
    OpKind::ScaledGroupedMatMul,
    OpKind::DequantGroupedMatMul,
    OpKind::DequantGroupedMatMulMlx,
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
    // Training / vision (native WGSL where listed; else host / HostOp).
    OpKind::AxialRope2d,
    OpKind::MaxPool2dBackward,
    OpKind::SoftmaxCrossEntropyWithLogits,
    OpKind::SoftmaxCrossEntropyBackward,
    OpKind::GroupNormBackwardInput,
    OpKind::GroupNormBackwardGamma,
    OpKind::GroupNormBackwardBeta,
    OpKind::BatchNormInference,
    OpKind::BatchNormInferenceBackwardInput,
    OpKind::BatchNormInferenceBackwardGamma,
    OpKind::BatchNormInferenceBackwardBeta,
    // Native activation_backward.wgsl (Fixed kinds, op id 0–16).
    OpKind::ReluBackward,
    OpKind::ActivationBackward,
    // Scaled FP8 (Metal-parity host).
    OpKind::ScaledMatMul,
    OpKind::ScaledQuantize,
    OpKind::ScaledQuantScale,
    OpKind::ScaledDequantize,
    // Native FakeQuantize Fixed/PerBatch; EMA + LSQ/Backward stay HostOp.
    OpKind::FakeQuantize,
    OpKind::FakeQuantizeBackward,
    OpKind::FakeQuantizeLSQ,
    OpKind::FakeQuantizeLSQBackwardX,
    OpKind::FakeQuantizeLSQBackwardScale,
    OpKind::Quantize,
    OpKind::Dequantize,
    OpKind::QMatMul,
    OpKind::QConv2d,
    OpKind::DenseSolve,
    OpKind::BatchedDenseSolve,
    // Host-staged to CPU LAPACK (potrf / trsm / getrf) via the `Step::HostOp`
    // catch-all in `compile/lower.rs`, same as DenseSolve.
    OpKind::Cholesky,
    OpKind::TriangularSolve,
    OpKind::Det,
    OpKind::LogDet,
    // Sort / ArgSort host-stage to CPU (stable strided sort) via the
    // `Step::HostOp` catch-all in `compile/lower.rs`, same as Det / LogDet.
    OpKind::Sort,
    OpKind::Svd,
    OpKind::Qr,
    OpKind::ArgSort,
    // Native WGSL C64 Wirtinger surface (`complex_wirtinger.wgsl`).
    OpKind::ComplexNormSq,
    OpKind::ComplexNormSqBackward,
    OpKind::Conjugate,
    // Native WGSL ternary-pruned FFT butterfly (`fft_butterfly_stage.wgsl`).
    OpKind::FftButterflyStage,
    // Remaining QAT / INT8 / fuse forms (HostOp → CPU).
    OpKind::LoraMatMul,
    OpKind::FusedConvBiasAct,
    // NOT PartitionedConv: the HostOp path returned zeros (cpu=0.4 vs gpu=0),
    // and declining the claim lets the shared unfuse decompose it into the
    // rfft → complex-GEMM → irfft primitives it already has WGSL kernels for —
    // which runs on the GPU instead of staging back to the host.
    OpKind::CustomFn,
    // Session + WgpuExecutable compile run LowerControlFlow first.
    OpKind::If,
    OpKind::While,
];
