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
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        DotGeneral,
        LayerNorm,
        LayerNorm2d,
        GroupNorm,
        RmsNorm,
        ResizeNearest2x,
        Interpolate3d,
        AxialRope2d,
        Attention,
        AttentionBackward,
        AttentionBackwardAll,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        RopeBackward,
        Cumsum,
        CumProd,
        CumMax,
        CumsumBackward,
        GatherBackward,
        Conv2dBackwardInput,
        Conv3dBackwardInput,
        Conv3dBackwardWeight,
        MaxPool3dBackward,
        Conv2dBackwardWeight,
        MaxPool2dBackward,
        Rope,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Reverse,
        Pad,
        Slice,
        Reduce,
        Softmax,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        ArgMax,
        ArgMin,
        TopK,
        Sample,
        RngNormal,
        RngUniform,
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
        DequantMatMul,
        SynthMatMul,
        // Native fused reconstruct (`synth_reconstruct_nk`, writes `w_bt[n,k]` in
        // one dispatch). Claimed so a forward-only INFERENCE path can emit it — it
        // wins the forward (~35 vs 47ms). NOT emitted by `Tensor::synth_reconstruct`
        // during training, where it MEASURED net-worse: the opaque op costs ~+20ms
        // in the backward (hidden from the CSE + transpose-simplification that make
        // the decomposed fold's `dx` free) — more than the ~12ms forward win.
        SynthReconstruct,
        // NOTE: `SynthMatMulBackward` intentionally NOT claimed — the native fused
        // kernels (MSL `synth_bwd_dx`/`synth_bwd_codebook`) are built and
        // correctness-validated, but MEASURED slower than decomposing to Gather +
        // MPS-tiled sgemm (a hand GEMM ≈ 40% of MPS, same as the forward). So the
        // op decomposes via `LowerSynthMatMulBackward`; the kernels stay dormant
        // for a future tiled/simdgroup implementation that could beat MPS.
        SplineActivation,
        SplineActivationBackwardX,
        SplineActivationBackwardCoeff,
        GatedDeltaNet,
        SelectiveScan,
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        FusedSwiGLU,
        FusedMatMulBiasAct,
        // FusedMatMulResidual intentionally NOT claimed: decode is weight-read
        // bandwidth-bound, so folding the residual into the matmul saves zero
        // GPU time (measured: identical 40.5ms wait with 56 fewer dispatches),
        // and its f32-only epilogue kernel would force the o_proj/down_proj
        // weights to materialize F32 — blocking the F16-resident weight win
        // (RLX_QWEN3_F16_WEIGHTS: 23.9→33.2 tok/s). The op + pass + kernel stay
        // in-tree for a future non-bandwidth-bound path; Metal just doesn't opt in.
        // (Re-confirmed on the dispatch-bound training path 2026-08: fusing all 16
        // residuals gave 1.00× — removing small dispatches doesn't move wall time
        // here; only removing real recompute, like the attention-bwd fusion, does.)
        FusedResidualLN,
        FusedResidualRmsNorm,
        // Claimed so the Metal fusion pipeline may emit it;
        // `MetalExecutable::compile_inner` decomposes it back to the
        // primitive chain (no monolithic fused-attention MSL kernel
        // yet — the per-run cost is dominated by wait_until_completed,
        // not encode, so a dispatch-wrapper fusion buys nothing).
        FusedAttentionBlock,
        // DiT adaLN-Zero / gated residual — native MSL kernels avoid
        // Expand of `[B,1,D]` modulation over the sequence axis.
        AdaLayerNorm,
        GatedResidual,
        AdaLayerNormBackward,
        GatedResidualBackward,
        // User-registered custom ops dispatched through
        // `rlx_metal::op_registry`. Lowering panics with a clear
        // message if the named MetalKernel isn't registered;
        // executor inserts a sync point + runs the host kernel
        // against the unified-memory arena.
        Custom,
        // Op::Fft is supported via the same host-fallback pattern
        // as Custom: sync the GPU, run rlx-cpu's FFT against the
        // unified-memory arena, restart cmd_buf. A native Metal
        // compute kernel will replace this when a workload makes
        // the sync the bottleneck.
        Fft,
        // Op::Scan (arbitrary-body recurrence) via the same host
        // fallback: compile the body once, loop it on the CPU against
        // the unified-memory arena. Enables IIR (`biquad`/`sosfilt`).
        Scan,
        ScanBackward,
        ScanBackwardXs,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        // Host-fallback splat (unified-memory arena + rlx-cpu/splat).
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        // Core Riemannian / SPD-manifold ops. No MSL eigen kernel; they
        // host-fallback to `rlx_cpu::spd` (F64) against the unified-memory
        // arena via the same sync pattern as Fft/Custom (see
        // `rlx_metal::spd` + `Thunk::SpdHost`). The SPD subgraph's F64
        // tensors are widened to f32 for arena planning; the host step does
        // the f32↔f64 conversion.
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
        // Full OpKind coverage (host-fallback via `Thunk::HostOp` /
        // `eval_single_op_f32`, or primitive expand for fused ops the
        // CPU catch-all would Nop). See `lower_cpu_nop_fused_for_metal`
        // + the HostOp arm in `thunk/compile.rs`.
        Quantize,
        Dequantize,
        FakeQuantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        DenseSolve,
        BatchedDenseSolve,
        // Cholesky / TriangularSolve / Det / LogDet host-stage to CPU
        // LAPACK via the `_other => Thunk::HostOp` catch-all (same as
        // DenseSolve F64).
        Cholesky,
        TriangularSolve,
        Det,
        LogDet,
        // Sort / ArgSort host-stage to CPU (stable strided sort) via the
        // same `_other => Thunk::HostOp` catch-all as Det / LogDet.
        Sort,
        Svd,
        Qr,
        ArgSort,
        BatchNormInference,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        Conv3d,
        ConvTranspose3d,
        // Native MSL ReluBackward / ActivationBackward (Fixed kinds).
        ReluBackward,
        ActivationBackward,
        FakeQuantizeBackward,
        // Native MSL C64 Wirtinger surface (`complex_norm_sq` /
        // `complex_norm_sq_backward` / `conjugate_c64`).
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        // Native MSL ternary-pruned FFT butterfly (`fft_butterfly_stage`).
        FftButterflyStage,
        LoraMatMul,
        PartitionedConv,
        QMatMul,
        QConv2d,
        FusedConvBiasAct,
        FusedTransformerLayer,
        If,
        While,
        CustomFn,
    ]
};
