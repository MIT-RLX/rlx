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
        ScaledMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        DotGeneral,
        LayerNorm,
        LayerNorm2d,
        GroupNorm,
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
        Rope,
        Reshape,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Reduce,
        Softmax,
        Cumsum,
        TopK,
        Sample,
        Conv,
        ConvTranspose2d,
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
        // Native ROCm/host GDN. Must be claimed so legalize does not unfuse
        // into a scan that drops carry writeback (same as CUDA).
        GatedDeltaNet,
        Lstm,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad) via
        // D2H→CPU→H2D host fallback (forces eager, not graph-captured).
        Scan,
        ScanBackward,
        ScanBackwardXs,
        FusedMatMulBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        AdaLayerNorm,
        GatedResidual,
        AdaLayerNormBackward,
        GatedResidualBackward,
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
        // (D2H → CPU reference → H2D; see `rlx_rocm::spd`). No ROCm
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
    ]
};
