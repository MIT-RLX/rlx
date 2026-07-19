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
        LayerNorm,
        RmsNorm,
        Attention,
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
        // Real-INT8 path + fake-quant.
        QMatMul,
        QConv2d,
        Quantize,
        Dequantize,
        FusedMatMulBiasAct,
        FusedResidualLN,
        FusedResidualRmsNorm,
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native composed HLO in `rlx-tpu/src/lower/lower.rs`.
        AdaLayerNormBackward,
        GatedResidualBackward,
        // Claimed for legalize/coverage; `rlx_tpu::unfuse::unfuse`
        // decomposes it to the primitive chain ahead of HLO emission.
        FusedAttentionBlock,
        Fft,
        LogMel,
        LogMelBackward,
        WelchPeaks,
        RngNormal,
        RngUniform,
        // Splat: no on-chip kernel — lowered to common primitive MIR via logical_kernel.
    ]
};
