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
        MatMul,
        DotGeneral,
        DenseSolve,
        BatchedDenseSolve,
        Scan,
        ScanBackward,
        ScanBackwardXs,
        LayerNorm,
        LayerNorm2d,
        GroupNorm,
        BatchNormInference,
        RmsNorm,
        ResizeNearest2x,
        AxialRope2d,
        Attention,
        Rope,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Reverse,
        Reduce,
        Softmax,
        Cumsum,
        ArgMax,
        ArgMin,
        TopK,
        Sample,
        RngNormal,
        RngUniform,
        Conv,
        Im2Col,
        ConvTranspose2d,
        Conv3d,
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
        LoraMatMul,
        DequantMatMul,
        ScaledMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        SelectiveScan,
        GatedDeltaNet,
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        FusedSwiGLU,
        FusedMatMulBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        FusedAttentionBlock,
        AdaLayerNorm,
        GatedResidual,
        AdaLayerNormBackward,
        GatedResidualBackward,
        // Backward ops emitted by `rlx_opt::autodiff::grad_with_loss`.
        // Their thunks live in rlx-cpu/src/thunk.rs alongside the
        // forward kernels; without these entries the legalize step
        // below would reject any compiled gradient graph.
        ReluBackward,
        ActivationBackward,
        FakeQuantize,
        FakeQuantizeBackward,
        // LSQ (learned step size) QAT — native CPU thunks in thunk.rs.
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        MaxPool2dBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        AttentionBackward,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        // GroupNorm backward (native thunks in rlx-cpu/training_bwd):
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        RopeBackward,
        CumsumBackward,
        GatherBackward,
        // 3D Gaussian splat CPU reference render/backward (requires `rlx-cpu/splat`).
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        // User-registered custom ops dispatched through
        // `rlx_cpu::op_registry`. Lowering panics with a clear
        // message if the named CPU kernel isn't registered.
        Custom,
        // User-defined sub-graph with optional override AD rules
        // (JAX-shaped custom_vjp / custom_jvp). Body is a regular
        // Graph compiled recursively in compile_thunks.
        CustomFn,
        // FFT primitive (1D last-axis, 2N real-block layout, f64
        // power-of-2 sizes). Other backends panic at lowering;
        // pin FFT-containing graphs to Device::Cpu for now.
        Fft,
        FftButterflyStage,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        // C64 Wirtinger AD surface. ComplexNormSq is the canonical
        // real-valued loss for complex inputs; Conjugate is emitted
        // by the new Wirtinger VJP rules for BinaryOp::Mul/Div on
        // C64. Both have CPU thunks in rlx-cpu.
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        // Riemannian / SPD-manifold layers (SPDNet + SPD batch-norm).
        // CPU-first (F64); other backends fall through their catch-alls.
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
    ]
};
