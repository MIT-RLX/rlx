// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OpKinds this backend claims for legalization (`Backend::supported_ops`).
//!
//! Source of truth for the coverage matrix in `docs/op-coverage.md`.
//! Kept in the backend crate so adding an op is a local edit, not a change
//! to `rlx-runtime`'s mega-`backend.rs`.

pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        Activation,
        Cast,
        StopGradient,
        Binary,
        Compare,
        Where,
        Fma,
        ElementwiseRegion,
        TransformRegion,
        BatchElementwiseRegion,
        MatMul,
        ScaledMatMul,
        ScaledGroupedMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        DotGeneral,
        LayerNorm,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        LayerNorm2d,
        GroupNorm,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        BatchNormInference,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        ResizeNearest2x,
        Interpolate3d,
        AxialRope2d,
        Reverse,
        Pad,
        Slice,
        ArgMax,
        ArgMin,
        RmsNorm,
        Attention,
        AttentionBackward,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        RopeBackward,
        CumsumBackward,
        GatherBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        MaxPool2dBackward,
        Conv3dBackwardInput,
        Conv3dBackwardWeight,
        MaxPool3dBackward,
        Rope,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        KvAppend,
        Expand,
        Gather,
        Reduce,
        Softmax,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        ReluBackward,
        ActivationBackward,
        Cumsum,
        CumProd,
        CumMax,
        TopK,
        Sample,
        Conv,
        Conv3d,
        ConvTranspose2d,
        ConvTranspose3d,
        Pool,
        GroupedMatMul,
        DequantGroupedMatMul,
        DequantGroupedMatMulMlx,
        DequantMoEWeights,
        ScatterAdd,
        ScatterNd,
        ScatterElements,
        GatherNd,
        GatherElements,
        DequantMatMul,
        SelectiveScan,
        // Native CUDA kernel + host fallback. Must be claimed: otherwise
        // legalize unfuses the scan into primitives that do not write the
        // final SSM back into the carry Input, so prefill-cache export
        // reads zeros and Qwen3.5/Bonsai decode diverges from Metal/CPU.
        GatedDeltaNet,
        Lstm,
        // Native CUDA kernel (L=1/unidir/no-carry, hidden≤1024) + host fallback.
        Gru,
        Rnn,
        // Native CUDA kernel (state_size≤256) + host fallback.
        Mamba2,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad) via
        // D2H→CPU→H2D host fallback (forces eager, not graph-captured).
        Scan,
        ScanBackward,
        ScanBackwardXs,
        FusedMatMulBiasAct,
        FusedConvBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        FusedSwiGLU,
        AdaLayerNorm,
        GatedResidual,
        AdaLayerNormBackward,
        GatedResidualBackward,
        // Native Fixed + PerBatch; LSQ forward reuses Fixed; LSQ bwd + STE
        // FakeQuantizeBackward + INT8 Quantize/Dequantize are native kernels.
        FakeQuantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        FakeQuantizeBackward,
        Quantize,
        Dequantize,
        // Fused, then decomposed by the backend's own `unfuse` pass
        // (rlx-cuda / rlx-rocm) before lowering — no monolithic
        // fused-attention kernel yet, same fuse-then-unfuse as WGPU.
        FusedAttentionBlock,
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        Custom,
        Fft,
        // Fixed-point `Op::FftQ`, host fallback. The arena stores integers as
        // f32 values, so the adapter converts at the boundary; exact while every
        // value stays inside f32's exact-integer range (see run_fft1d_q_valued).
        FftQ,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        Im2Col,
        RngNormal,
        RngUniform,
        // Core Riemannian / SPD-manifold ops (F64) via CPU host fallback
        // (D2H → CPU reference → H2D; see `rlx_cuda::spd`). No CUDA
        // eigendecomposition kernel; runs the exact `rlx-cpu` thunk kernels.
        BiMap,
        ReEig,
        LogEig,
        SpdBatchNorm,
        SpdKarcherMean,
        SpdKarcherMeanWeighted,
        SpdLogMap,
        SpdExpMap,
        SpdParallelTransport,
        SpdMatrixFnBatch,
        ReEigBackward,
        LogEigBackward,
        SpdBatchNormBackwardX,
        SpdBatchNormBackwardG,
        SpdLogMapBackward,
        SpdExpMapBackward,
        SpdParallelTransportBackward,
        SpdMatrixFnBatchBackward,
        Eigh,
        EighBackward,
        EighBatch,
        EighBatchBackward,
        // C64 Wirtinger surface — native `complex_wirtinger.cu` (shared with
        // ROCm). Interleaved [re, im] pairs matching CPU / Metal MSL /
        // wgpu WGSL semantics.
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        // Decomposed by the backend `unfuse` pass (`rlx_unfuse::expand_lora`
        // → MatMul + Mul + Add) before lowering — same path as wgpu. No fused
        // LoRA kernel; claiming keeps legalize from rejecting the op.
        LoraMatMul,
        // Same unfuse pass expands these to primitives CUDA already runs
        // (`expand_ftl` / `expand_if` / bounded `expand_while`).
        FusedTransformerLayer,
        If,
        While,
        // DenseSolve / BatchedDenseSolve: native F32 via cuSOLVER/cuBLAS;
        // other dtypes stay HostOp. QMatMul / QConv2d are native INT8;
        // CustomFn remains host-staged. PartitionedConv expands in
        // `crate::unfuse` to Fft/MatMul (batched-GEMM frequency path).
        DenseSolve,
        BatchedDenseSolve,
        // Cholesky / TriangularSolve / Det / LogDet host-stage to CPU LAPACK
        // (potrf / trsm / getrf) via the `Step::HostOp` catch-all in
        // `backend/compile.rs`. Native cuSOLVER is a future perf follow-up.
        Cholesky,
        TriangularSolve,
        Det,
        LogDet,
        // Sort / ArgSort host-stage to CPU (stable strided sort) via the
        // `Step::HostOp` catch-all in `backend/compile.rs`, same as Det / LogDet.
        Sort,
        Svd,
        Qr,
        ArgSort,
        QMatMul,
        QConv2d,
        FftButterflyStage,
        PartitionedConv,
        CustomFn,
    ]
};

/// Does this op get handed to rlx-cpu instead of a CUDA kernel?
///
/// Exposed so the routing can be checked from outside the crate **without a
/// device**, by `rlx-runtime/tests/host_fallback_never_nops.rs`. The
/// composition it guards is invisible to any single-crate test: [`SUPPORTED_OPS`]
/// claims an op (so nothing upstream expands it), no kernel lowers it, this
/// returns true, and rlx-cpu has no thunk arm for it either — the result is a
/// `Thunk::Nop` over a zeroed slot, with no panic and no unsupported-op error.
/// Vulkan's `PartitionedConv` shipped exactly that way.
///
/// rlx-cuda is structurally safer than that: anything reaching its compile
/// match with no arm hits a `panic!("op … not yet lowered")` rather than a
/// host route, so an unhandled claim is loud. This predicate covers the ops
/// that are *deliberately* handed to rlx-cpu — the generic `Step::HostOp`
/// catch-all, plus the three op-specific host steps (`Step::ScanHost`,
/// `Step::HostOp` for the scan backwards, and `Step::CpuIndexing`).
pub fn routes_to_cpu_host(op: &rlx_ir::Op) -> bool {
    use rlx_ir::Op;
    matches!(
        op,
        // Generic `Step::HostOp` catch-all in `backend/compile.rs`: F64 (and
        // other dtypes) via CPU LAPACK, plus the opaque `CustomFn` body.
        Op::DenseSolve
            | Op::BatchedDenseSolve
            | Op::Cholesky
            | Op::TriangularSolve { .. }
            | Op::Det
            | Op::LogDet
            | Op::Sort { .. }
            | Op::Svd { .. }
            | Op::Qr { .. }
            | Op::ArgSort { .. }
            | Op::CustomFn { .. }
            // `Step::ScanHost` / `Step::HostOp`.
            | Op::Scan { .. }
            | Op::ScanBackward { .. }
            | Op::ScanBackwardXs { .. }
            // `Step::CpuIndexing`. These four now have on-device kernels
            // (`Step::IndexingNd`), so this is the *residual* host route: the
            // shapes `rlx_gpu_host::indexing_plan` declines — genuine packed
            // I64 indices, non-f32 gathered elements, Mul/Max/Min reductions,
            // and the two ScatterElements branches that infer a layout. The
            // predicate stays coarse on purpose: it answers "can this op reach
            // rlx-cpu", and it still can.
            | Op::ScatterNd { .. }
            | Op::ScatterElements { .. }
            | Op::GatherNd { .. }
            | Op::GatherElements { .. }
    )
}
