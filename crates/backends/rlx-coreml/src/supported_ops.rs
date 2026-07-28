// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OpKinds this backend claims for legalization (`Backend::supported_ops`).
//!
//! Source of truth for the coverage matrix in `docs/op-coverage.md`.
//! Kept in the backend crate so adding an op is a local edit, not a change
//! to `rlx-runtime`'s mega-`backend.rs`.

/// Ops with a MIL lowering today (see `rlx_coreml::mil`).
///
/// Single source of truth, shared by `coreml_backend::CoremlBackend::
/// supported_ops` and `device_ext::supports(Device::Ane, ..)` so the two
/// never drift. Ungated so the support probe compiles on every target
/// (it's only *consulted* when the `coreml` backend is available).
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        Activation,
        Cast,
        Binary,
        MatMul,
        LayerNorm,
        RmsNorm,
        Reduce,
        Softmax,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Gather,
        Rope,
        Attention,
        // Claimed first-class; `CoremlExecutable::compile_with_options`
        // decomposes it to the primitive chain (matmul → narrow → rope →
        // attention → matmul) since the MIL lowering has no fused-attention
        // op. FAB-only decompose, so native LoraMatMul below is untouched.
        FusedAttentionBlock,
        // DiT modulation — native composed MIL (implicit broadcast, no Expand).
        AdaLayerNorm,
        GatedResidual,
        Compare,
        Where,
        Expand,
        Cumsum,
        ScatterAdd,
        ScatterNd,
        ScatterElements,
        GatherNd,
        GatherElements,
        BatchNormInference,
        GroupNorm,
        LayerNorm2d,
        LoraMatMul,
        Conv,
        ConvTranspose2d,
        Pool,
        TopK,
        AxialRope2d,
        ResizeNearest2x,
        Interpolate3d,
        StopGradient,
        GroupedMatMul,
        DequantMatMul,
        DequantMoEWeights,
        DequantGroupedMatMul,
        Quantize,
        Dequantize,
        SelectiveScan,
        GatedDeltaNet,
        ArgMax,
        ArgMin,
        Reverse,
        Fft,
        LogMel,
        Sample,
        RngNormal,
        RngUniform,
        Lstm,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad) runs on
        // the host between MIL segments via rlx-cpu's execute_scan_host.
        Scan,
        ScanBackward,
        ScanBackwardXs,
        Gru,
        Rnn,
        Mamba2,
        WelchPeaks,
        Custom,
        // Full OpKind coverage: ops without a MIL arm run as hybrid host
        // segments via `host_exec::is_host_op` + `run_host_op_node_f32`.
        // Fused forms the CPU catch-all would Nop are expanded in
        // `CoremlExecutable::compile_with_options` before planning.
        FakeQuantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        Fma,
        ElementwiseRegion,
        TransformRegion,
        BatchElementwiseRegion,
        DotGeneral,
        DenseSolve,
        BatchedDenseSolve,
        // Host-staged to CPU LAPACK (potrf / trsm / getrf) via `is_host_node`,
        // same as DenseSolve (no MIL arm).
        Cholesky,
        TriangularSolve,
        Det,
        LogDet,
        // Sort / ArgSort host-stage to CPU (stable strided sort) via
        // `is_host_node`, same as Det / LogDet (no MIL arm).
        Sort,
        Svd,
        Qr,
        ArgSort,
        Im2Col,
        Conv3d,
        ConvTranspose3d,
        ReluBackward,
        ActivationBackward,
        FakeQuantizeBackward,
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        MaxPool2dBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        MaxPool3dBackward,
        Conv3dBackwardInput,
        Conv3dBackwardWeight,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        AttentionBackward,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        RopeBackward,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
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
        FusedSwiGLU,
        FusedMatMulBiasAct,
        FusedConvBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        FusedTransformerLayer,
        If,
        While,
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        CustomFn,
        FftButterflyStage,
        LogMelBackward,
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
        AdaLayerNormBackward,
        GatedResidualBackward,
    ]
};

/// Backward / training `OpKind`s the CoreML backend can run **via decomposition**
/// (`rlx_autodiff::decompose_backward_ops_except` in the legalize/rewrite pass):
/// each lowers to a chain of primitives that are all in [`SUPPORTED_OPS`].
///
/// Kept separate from `SUPPORTED_OPS` on purpose: these must NOT be in the
/// list handed to `legalize_or_rewrite_for_backend` (otherwise they'd be treated
/// as directly lowerable and skip the decompose, and the MIL lowering would choke
/// on a raw `*Backward` op). They feed only the *device-selection* probe
/// (`device_ext::coreml_supports`) so the runtime picks `Device::Ane` for a graph
/// that carries them. As native MIL backward kernels land (see `rlx_coreml::mil`),
/// the corresponding kind graduates into `SUPPORTED_OPS` and is removed
/// here. Only consulted under the `training` feature.
///
/// Excluded deliberately: `Conv2dBackwardWeight` (decomposes via `Im2Col`, which
/// has no MIL lowering yet — lands with the Phase-2 conv kernels) and the
/// conditional / domain backward ops (`ScanBackward*`, `LogMelBackward`,
/// `GaussianSplatRenderBackward`, `ComplexNormSqBackward`, `FakeQuantizeLSQ*`).
#[allow(dead_code)] // only read under `feature = "training"`.
pub const BACKWARD_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        ReluBackward,
        ActivationBackward,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        RopeBackward,
        AttentionBackward,
        SoftmaxCrossEntropyBackward,
        CumsumBackward,
        GatherBackward,
        FakeQuantizeBackward,
    ]
};

/// Backward `OpKind`s the CoreML backend lowers through a **native MIL kernel**
/// (`rlx_coreml::mil`, gated by `rlx-coreml/training`) rather than decomposition.
/// Unlike [`BACKWARD_OPS`], these ARE added to the list handed to
/// `legalize_or_rewrite_for_backend` (under `training`), so the rewrite leaves
/// them intact for the lowering's dedicated arm.
///
/// These cases land here:
/// - **RMSNorm backward** — the dominant norm in modern transformers; a
///   hand-composed MIL kernel (implicit broadcasting, ~13 ops) beats the autodiff
///   decomposition (~27 ops of `Expand`-with-ones). `ReluBackward`/
///   `ActivationBackward` are NOT here: their decomposition is already `dy·f'(x)`
///   (~3 MIL ops), so a native kernel would emit identical MIL — they ride the
///   decompose route instead.
/// - **MaxPool2d backward** — MUST be native: the decomposition builds an
///   N²-sized dense scatter that blows RLX's size cap on any real CNN. The native
///   kernel is O(input) (reshape + reduce_max/min + select). Non-overlapping,
///   unpadded pooling only (the CNN-training norm, e.g. MNIST 2×2/2); other
///   configs return `Unsupported`.
/// - **Conv2d backward** (input via `conv_transpose`, weight via the transpose-conv
///   trick) — native because the autodiff input decomposition emits the wrong op.
/// - **Softmax-cross-entropy (forward `WithLogits` + backward)** — MUST be native
///   for LLM-scale training: the decompose builds the one-hot by concatenating C
///   class columns, O(C) graph nodes that explode at vocab size. MIL `one_hot` is a
///   single node. Both halves are listed so the loss op stays out of `bad` and the
///   shared `LowerSoftmaxCrossEntropy` pass never re-decomposes the backward.
/// - **DiT packed reverse** (`AdaLayerNormBackward` / `GatedResidualBackward`) —
///   native composed MIL (affine-free LN/RMS dx + unbroadcast + concat pack)
///   beats the Expand-heavy autodiff decomposition on ANE.
///
/// MUST stay in lock-step with the lowering arms in `rlx_coreml::mil` (a kind
/// here without an arm would skip decompose and hit the `Unsupported` fallback).
/// The norm arms mirror the autodiff decomposition's math — the RMSNorm input
/// cross-term is `inv_r³` (finite-difference-verified, the same formula every
/// backend now uses) — so ANE gradients stay consistent with the rest of the
/// training path.
#[allow(dead_code)] // only read under `feature = "training"`.
pub const NATIVE_BACKWARD_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        MaxPool2dBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        AttentionBackward,
        // The SCE training pair (integer-label loss + its gradient). Both native so
        // neither lands in `bad` — otherwise the shared `LowerSoftmaxCrossEntropy`
        // pass fires on the forward and re-decomposes the backward into the O(C)
        // one-hot concat. `SoftmaxCrossEntropyWithLogits` is a forward op but is
        // training-only, so it lives here rather than in the base inference set.
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        // DiT packed reverse — native composed MIL (implicit broadcast +
        // concat pack); kept intact by legalize under `training`.
        AdaLayerNormBackward,
        GatedResidualBackward,
    ]
};

/// Op claim under the `training` feature. Native backward kernels are already
/// in [`SUPPORTED_OPS`] (full coverage); this alias keeps
/// `CoremlBackend::supported_ops` and the fusion pipeline on one list so MIL
/// arms (MaxPool2d / Conv2d / RmsNorm / … backward) stay intact instead of
/// being decomposed before lowering.
#[cfg(feature = "training")]
pub const SUPPORTED_OPS_TRAINING: &[rlx_ir::OpKind] = SUPPORTED_OPS;
