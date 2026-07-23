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
        AxialRope2d,
        Reverse,
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
        Rope,
        Reshape,
        Transpose,
        Narrow,
        Concat,
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
        TopK,
        Sample,
        Conv,
        Conv3d,
        ConvTranspose2d,
        ConvTranspose3d,
        Pool,
        GroupedMatMul,
        DequantGroupedMatMul,
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
        QMatMul,
        QConv2d,
        FftButterflyStage,
        PartitionedConv,
        CustomFn,
    ]
};
