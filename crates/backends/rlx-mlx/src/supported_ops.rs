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
        ElementwiseRegion,
        TransformRegion,
        BatchElementwiseRegion,
        MatMul,
        DotGeneral,
        DenseSolve,
        BatchedDenseSolve,
        // Host-staged to CPU LAPACK via `host_eval_op_typed`
        // (`is_mlx_typed_host_op`), the same route as Eigh.
        Cholesky,
        TriangularSolve,
        Det,
        LogDet,
        // Sort / ArgSort host-stage to CPU via `host_eval_op_typed`
        // (`is_mlx_typed_host_op`), same route as Det / LogDet.
        Sort,
        Svd,
        Qr,
        ArgSort,
        LayerNorm,
        LayerNorm2d,
        GroupNorm,
        ResizeNearest2x,
        Interpolate3d,
        RmsNorm,
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
        CumProd,
        CumMax,
        ArgMax,
        ArgMin,
        TopK,
        RngNormal,
        RngUniform,
        Sample,
        Conv,
        Im2Col,
        ConvTranspose2d,
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
        LoraMatMul,
        DequantMatMul,
        SelectiveScan,
        GatedDeltaNet,
        FusedSwiGLU,
        FusedMatMulBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native composed lowering in `rlx-mlx/src/lower/`.
        AdaLayerNormBackward,
        GatedResidualBackward,
        FusedAttentionBlock,
        FusedTransformerLayer,
        If,
        While,
        // Loop-unrolled scan (Op::Scan body is statically unrolled
        // `length` times into MLX ops; mirror of Op::While's
        // bounded-unroll lowering). ScanBackward is the AD
        // companion — handled the same way.
        Scan,
        ScanBackward,
        ScanBackwardXs,
        // Recurrent ops are kept intact (NOT unfused into a giant
        // unrolled graph — that makes MLX compile crawl on long
        // sequences, e.g. a 256-step LSTM over a spatial field). MLX
        // host-evals them via the shared CPU kernel (see
        // `first_host_eval_op` + the `Op::Lstm` arm in rlx-mlx/lower.rs),
        // matching the ROCm/CoreML host-fallback pattern.
        Lstm,
        // GRU / Elman-RNN: native on-device unrolls in `rlx-mlx/lower.rs`
        // (`native_gru` / `native_rnn`), same story as `Lstm`. Without these
        // the runtime would decompose them via `unfuse` and the native
        // (fp16-capable) unrolls would never run.
        Gru,
        Rnn,
        // Tier 1 autodiff backward ops — lowered as primitive
        // compositions in `rlx-mlx/src/lower/`.
        ReluBackward,
        ActivationBackward,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        AttentionBackward,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        // GroupNorm backward — native MLX lowering in `lower.rs`
        // (group-reshape + reduce, mirrors GroupNormBackwardInput).
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        // Tier 2 — conv backward via `mc::conv_general` with the
        // same parameter-mapping MLX uses inside its built-in vjp.
        // Currently groups=1 only; grouped conv backward will
        // surface as a clear error from `lower.rs`.
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        // 3D training bwd — typed CPU host-eval (no native MLX path yet).
        Conv3dBackwardInput,
        Conv3dBackwardWeight,
        MaxPool3dBackward,
        // Tier 3 — max-pool backward via slice-strided argmax over
        // pool windows + a per-kernel-slot scatter-add, matching
        // the CPU thunk's "first-hit-wins" tiebreaking.
        MaxPool2dBackward,
        // QAT — `FakeQuantize` (PerBatch + Fixed scale modes;
        // EMA returns a clear error from `lower.rs`) and the
        // `FakeQuantizeBackward` family covering all 4 STE
        // variants. Closes the last gap vs `rlx_cpu::SUPPORTED_OPS`.
        FakeQuantize,
        FakeQuantizeBackward,
        // User-registered custom ops dispatched through
        // `rlx_mlx::op_registry`. Lowering looks up the
        // registered `MlxKernel` and calls its `execute` method
        // to produce the lazy MLX `Array` for this node.
        Custom,
        Fft,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        // Op::Fft on MLX: native `mlx::fft::fft` via rlx_mlx_op_fft shim.
        // 2N real-block f32/f64 and complex64 inputs supported.
        // ── Full OpKind coverage (native MLX or typed CPU host-eval) ──
        // Native: Quantize/Dequantize/LSQ/QMatMul/QConv2d/Complex*/FftButterfly/Mamba2.
        // Host typed: Scaled*, CustomFn, GaussianSplat prepare/rasterize, SPD/Eigh.
        Quantize,
        Dequantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        Fma,
        BatchNormInference,
        AxialRope2d,
        Conv3d,
        ConvTranspose3d,
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        RopeBackward,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        CumsumBackward,
        GatherBackward,
        PartitionedConv,
        QMatMul,
        QConv2d,
        ScaledMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        Mamba2,
        FusedConvBiasAct,
        CustomFn,
        FftButterflyStage,
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
    ]
};
