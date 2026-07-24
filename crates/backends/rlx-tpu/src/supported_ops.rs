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
        DotGeneral,
        LayerNorm,
        LayerNorm2d,
        GroupNorm,
        BatchNormInference,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        RmsNorm,
        Attention,
        Rope,
        AxialRope2d,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Reverse,
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
        ArgMax,
        ArgMin,
        Conv,
        Conv3d,
        ConvTranspose2d,
        ConvTranspose3d,
        Pool,
        Im2Col,
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
        // Real-INT8 path + fake-quant.
        QMatMul,
        QConv2d,
        Quantize,
        Dequantize,
        FakeQuantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        FakeQuantizeBackward,
        FusedMatMulBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        FusedSwiGLU,
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native composed HLO in `rlx-tpu/src/lower/lower.rs`.
        AdaLayerNormBackward,
        GatedResidualBackward,
        // Claimed for legalize/coverage; `prepare_graph_for_hlo` /
        // `rlx_tpu::unfuse` decomposes these to the primitive chain
        // ahead of HLO emission.
        FusedAttentionBlock,
        FusedTransformerLayer,
        LoraMatMul,
        PartitionedConv,
        GatedDeltaNet,
        FusedConvBiasAct,
        If,
        While,
        // Recurrent — claimed then expanded by `rlx_fusion::unfuse_recurrent_ops`.
        Gru,
        Rnn,
        Lstm,
        Mamba2,
        // Wirtinger / complex — native XLA C64 opcodes.
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        // Host segments (CPU LAPACK / vision kernels between HLO).
        DenseSolve,
        BatchedDenseSolve,
        Fft,
        FftButterflyStage,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        RngNormal,
        RngUniform,
        // Training / bwd — composed HLO; AttentionBackward expands in
        // `prepare_graph_for_hlo` (autodiff decompose → MatMul/Softmax).
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        MaxPool2dBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        RopeBackward,
        CumsumBackward,
        GatherBackward,
        AttentionBackward,
        // Vision / scan / custom / scaled.
        // Interpolate3d / Conv3dBackward* / MaxPool3dBackward: not lowered on
        // TPU yet — omit from SUPPORTED_OPS (use CPU/GPU or im2col decompose).
        ResizeNearest2x,
        Scan,
        ScanBackward,
        ScanBackwardXs,
        Custom,
        CustomFn,
        ScaledMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        // SPD / Eigh — LowerSpectral (f32) or host (f64 / bwd / Eigh).
        BiMap,
        ReEig,
        LogEig,
        SpdBatchNorm,
        SpdKarcherMean,
        ReEigBackward,
        LogEigBackward,
        SpdBatchNormBackwardX,
        SpdBatchNormBackwardG,
        SpdKarcherMeanWeighted,
        SpdLogMap,
        SpdExpMap,
        SpdParallelTransport,
        SpdMatrixFnBatch,
        SpdLogMapBackward,
        SpdExpMapBackward,
        SpdParallelTransportBackward,
        SpdMatrixFnBatchBackward,
        Eigh,
        EighBackward,
        EighBatch,
        EighBatchBackward,
        // Splat: host segments (`splat_host` / `GaussianSplatPrepare|Rasterize`).
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
    ]
};
