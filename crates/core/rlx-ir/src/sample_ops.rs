// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One constructible [`Op`] per [`OpKind`], for gates that must cover the
//! whole op set rather than the handful a test author thought of.
//!
//! Feature `test-support`.
//!
//! # Why this exists
//!
//! Tables keyed on the op set — arity, capabilities, backend support — drift
//! the moment an op is added and one table is not. Nothing could iterate the
//! op set, so every such gate was limited to ops some test happened to build:
//! `rlx-corpus` reaches 13 of 187 kinds.
//!
//! [`sample`] is a `match` over `OpKind`, so **adding a variant fails to
//! compile here** until a sample is supplied. That is the enforcement; the
//! count assertion is only a second line of defence.
//!
//! # What a sample is and is not
//!
//! Field values are placeholders chosen to construct, not to be meaningful: a
//! sample is **not** a valid graph node and will not pass shape inference. Use
//! it to ask questions about the *op*, not about a computation.
//!
//! Fields that change an op\'s arity (`has_bias`, `num_inputs`, `mask`, …) take
//! one arbitrary setting, so a sample pins one point of a field-dependent
//! arity. Cover the other settings explicitly — see
//! `arity_is_consistent_for_field_dependent_ops`.

use crate::DType;
use crate::OpKind;
use crate::op::{
    Activation, AdaNormKind, AttentionBwdWrt, BinaryOp, CmpOp, MaskKind, Op, PadMode, QrPart,
    ReduceOp, RegionPrologue, RopeStyle, ScaleMode, ScatterNdReduction, SpdMatFn, SteKind, SvdPart,
    SynthBwdWrt, SynthKind,
};

/// A constructible `Op` for `kind`.
///
/// Exhaustive by construction: this is a `match` over every [`OpKind`].
pub fn sample(kind: OpKind) -> Op {
    match kind {
        OpKind::Input => Op::Input { name: "x".into() },
        OpKind::Param => Op::Param { name: "x".into() },
        OpKind::Constant => Op::Constant { data: Vec::new() },
        OpKind::Activation => Op::Activation(Activation::Gelu),
        OpKind::Cast => Op::Cast { to: DType::F32 },
        OpKind::StopGradient => Op::StopGradient,
        OpKind::Quantize => Op::Quantize {
            axis: None,
            scales: Vec::new(),
            zero_points: Vec::new(),
        },
        OpKind::Dequantize => Op::Dequantize {
            axis: None,
            scales: Vec::new(),
            zero_points: Vec::new(),
        },
        OpKind::FakeQuantize => Op::FakeQuantize {
            bits: 1,
            axis: None,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        OpKind::FakeQuantizeLSQ => Op::FakeQuantizeLSQ {
            bits: 1,
            axis: None,
        },
        OpKind::FakeQuantizeLSQBackwardX => Op::FakeQuantizeLSQBackwardX {
            bits: 1,
            axis: None,
        },
        OpKind::FakeQuantizeLSQBackwardScale => Op::FakeQuantizeLSQBackwardScale {
            bits: 1,
            axis: None,
        },
        OpKind::Binary => Op::Binary(BinaryOp::Add),
        OpKind::Compare => Op::Compare(CmpOp::Eq),
        OpKind::Where => Op::Where,
        OpKind::Fma => Op::Fma,
        OpKind::ElementwiseRegion => Op::ElementwiseRegion {
            chain: Vec::new(),
            num_inputs: 1,
            scalar_input_mask: 1,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 1,
        },
        OpKind::TransformRegion => Op::TransformRegion {
            steps: Vec::new(),
            num_inputs: 1,
        },
        OpKind::BatchElementwiseRegion => Op::BatchElementwiseRegion {
            chain: Vec::new(),
            num_batch_inputs: 1,
            scalar_input_mask: 1,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 1,
        },
        OpKind::MatMul => Op::MatMul,
        OpKind::DotGeneral => Op::DotGeneral {
            lhs_contracting: Vec::new(),
            rhs_contracting: Vec::new(),
            lhs_batch: Vec::new(),
            rhs_batch: Vec::new(),
        },
        OpKind::DenseSolve => Op::DenseSolve,
        OpKind::BatchedDenseSolve => Op::BatchedDenseSolve,
        OpKind::Cholesky => Op::Cholesky,
        OpKind::TriangularSolve => Op::TriangularSolve {
            lower: false,
            transpose: false,
        },
        OpKind::Det => Op::Det,
        OpKind::LogDet => Op::LogDet,
        OpKind::Sort => Op::Sort {
            axis: 1,
            descending: false,
        },
        OpKind::ArgSort => Op::ArgSort {
            axis: 1,
            descending: false,
        },
        OpKind::Svd => Op::Svd { part: SvdPart::U },
        OpKind::Qr => Op::Qr { part: QrPart::Q },
        OpKind::LayerNorm => Op::LayerNorm { axis: 1, eps: 1.0 },
        OpKind::LayerNorm2d => Op::LayerNorm2d { eps: 1.0 },
        OpKind::GroupNorm => Op::GroupNorm {
            num_groups: 1,
            eps: 1.0,
        },
        OpKind::BatchNormInference => Op::BatchNormInference { eps: 1.0 },
        OpKind::RmsNorm => Op::RmsNorm { axis: 1, eps: 1.0 },
        OpKind::ResizeNearest2x => Op::ResizeNearest2x,
        OpKind::Interpolate3d => Op::Interpolate3d { size: Vec::new() },
        OpKind::Attention => Op::Attention {
            num_heads: 1,
            head_dim: 1,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        OpKind::Rope => Op::Rope {
            head_dim: 1,
            n_rot: 1,
            style: RopeStyle::NeoX,
        },
        OpKind::AxialRope2d => Op::AxialRope2d {
            end_x: 1,
            end_y: 1,
            head_dim: 1,
            num_heads: 1,
            theta: 1.0,
            repeat_factor: 1,
        },
        OpKind::Reshape => Op::Reshape {
            new_shape: Vec::new(),
        },
        OpKind::Transpose => Op::Transpose { perm: Vec::new() },
        OpKind::Narrow => Op::Narrow {
            axis: 1,
            start: 1,
            len: 1,
        },
        OpKind::Concat => Op::Concat { axis: 1 },
        OpKind::KvAppend => Op::KvAppend { axis: 1, pos: 1 },
        OpKind::Expand => Op::Expand {
            target_shape: Vec::new(),
        },
        OpKind::Gather => Op::Gather { axis: 1 },
        OpKind::Reverse => Op::Reverse { axes: Vec::new() },
        OpKind::Pad => Op::Pad {
            pads: Vec::new(),
            mode: PadMode::Constant(0.0),
        },
        OpKind::Slice => Op::Slice {
            axis: 1,
            start: 1,
            len: 1,
            step: 1,
        },
        OpKind::Roll => Op::Roll {
            shifts: Vec::new(),
            dims: Vec::new(),
        },
        OpKind::Clamp => Op::Clamp { min: 1.0, max: 1.0 },
        OpKind::Tile => Op::Tile { reps: Vec::new() },
        OpKind::Trilu => Op::Trilu {
            upper: false,
            diagonal: 1,
        },
        OpKind::Reduce => Op::Reduce {
            op: ReduceOp::Sum,
            axes: Vec::new(),
            keep_dim: false,
        },
        OpKind::Histogram => Op::Histogram {
            bins: 1,
            min: 1.0,
            max: 1.0,
        },
        OpKind::Softmax => Op::Softmax { axis: 1 },
        OpKind::Cumsum => Op::Cumsum {
            axis: 1,
            exclusive: false,
        },
        OpKind::CumProd => Op::CumProd {
            axis: 1,
            exclusive: false,
        },
        OpKind::CumMax => Op::CumMax {
            axis: 1,
            exclusive: false,
        },
        OpKind::ArgMax => Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        OpKind::ArgMin => Op::ArgMin {
            axis: 1,
            keep_dim: false,
        },
        OpKind::TopK => Op::TopK { k: 1 },
        OpKind::Sample => Op::Sample {
            top_k: 1,
            top_p: 1.0,
            temperature: 1.0,
            seed: 1,
        },
        OpKind::RngNormal => Op::RngNormal {
            mean: 1.0,
            scale: 1.0,
            key: 1,
            op_seed: None,
        },
        OpKind::RngUniform => Op::RngUniform {
            low: 1.0,
            high: 1.0,
            key: 1,
            op_seed: None,
        },
        OpKind::Conv => Op::Conv {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
        },
        OpKind::Im2Col => Op::Im2Col {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
        },
        OpKind::ConvTranspose2d => Op::ConvTranspose2d {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            output_padding: Vec::new(),
            groups: 1,
        },
        OpKind::Conv3d => Op::Conv3d {
            stride: [1; 3],
            padding: [1; 3],
            dilation: [1; 3],
            groups: 1,
        },
        OpKind::ConvTranspose3d => Op::ConvTranspose3d {
            stride: [1; 3],
            padding: [1; 3],
            dilation: [1; 3],
            output_padding: [1; 3],
            groups: 1,
        },
        OpKind::Pool => Op::Pool {
            kind: ReduceOp::Sum,
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
        },
        OpKind::ReluBackward => Op::ReluBackward,
        OpKind::ActivationBackward => Op::ActivationBackward {
            kind: Activation::Gelu,
        },
        OpKind::FakeQuantizeBackward => Op::FakeQuantizeBackward {
            bits: 1,
            axis: None,
            ste: SteKind::Identity,
        },
        OpKind::ComplexNormSq => Op::ComplexNormSq,
        OpKind::ComplexNormSqBackward => Op::ComplexNormSqBackward,
        OpKind::Conjugate => Op::Conjugate,
        OpKind::LayerNormBackwardInput => Op::LayerNormBackwardInput { axis: 1, eps: 1.0 },
        OpKind::LayerNormBackwardGamma => Op::LayerNormBackwardGamma { axis: 1, eps: 1.0 },
        OpKind::RmsNormBackwardInput => Op::RmsNormBackwardInput { axis: 1, eps: 1.0 },
        OpKind::RmsNormBackwardGamma => Op::RmsNormBackwardGamma { axis: 1, eps: 1.0 },
        OpKind::RmsNormBackwardBeta => Op::RmsNormBackwardBeta { axis: 1, eps: 1.0 },
        OpKind::RopeBackward => Op::RopeBackward {
            head_dim: 1,
            n_rot: 1,
            style: RopeStyle::NeoX,
        },
        OpKind::GroupNormBackwardInput => Op::GroupNormBackwardInput {
            num_groups: 1,
            eps: 1.0,
        },
        OpKind::GroupNormBackwardGamma => Op::GroupNormBackwardGamma {
            num_groups: 1,
            eps: 1.0,
        },
        OpKind::GroupNormBackwardBeta => Op::GroupNormBackwardBeta {
            num_groups: 1,
            eps: 1.0,
        },
        OpKind::BatchNormInferenceBackwardInput => Op::BatchNormInferenceBackwardInput { eps: 1.0 },
        OpKind::BatchNormInferenceBackwardGamma => Op::BatchNormInferenceBackwardGamma { eps: 1.0 },
        OpKind::BatchNormInferenceBackwardBeta => Op::BatchNormInferenceBackwardBeta,
        OpKind::CumsumBackward => Op::CumsumBackward {
            axis: 1,
            exclusive: false,
        },
        OpKind::GatherBackward => Op::GatherBackward { axis: 1 },
        OpKind::MaxPool2dBackward => Op::MaxPool2dBackward {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
        },
        OpKind::Conv2dBackwardInput => Op::Conv2dBackwardInput {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
        },
        OpKind::Conv2dBackwardWeight => Op::Conv2dBackwardWeight {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
        },
        OpKind::MaxPool3dBackward => Op::MaxPool3dBackward {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
        },
        OpKind::Conv3dBackwardInput => Op::Conv3dBackwardInput {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
        },
        OpKind::Conv3dBackwardWeight => Op::Conv3dBackwardWeight {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
        },
        OpKind::SoftmaxCrossEntropy => Op::SoftmaxCrossEntropy,
        OpKind::SoftmaxCrossEntropyWithLogits => Op::SoftmaxCrossEntropyWithLogits,
        OpKind::SoftmaxCrossEntropyBackward => Op::SoftmaxCrossEntropyBackward,
        OpKind::AttentionBackward => Op::AttentionBackward {
            num_heads: 1,
            head_dim: 1,
            mask_kind: MaskKind::None,
            wrt: AttentionBwdWrt::Query,
        },
        OpKind::AttentionBackwardAll => Op::AttentionBackwardAll {
            num_heads: 1,
            head_dim: 1,
            mask_kind: MaskKind::None,
        },
        OpKind::GroupedMatMul => Op::GroupedMatMul,
        OpKind::DequantGroupedMatMul => Op::DequantGroupedMatMul {
            scheme: crate::quant::QuantScheme::Int8Block { block_size: 32 },
        },
        OpKind::DequantGroupedMatMulMlx => Op::DequantGroupedMatMulMlx {
            scheme: crate::quant::QuantScheme::Int8Block { block_size: 32 },
        },
        OpKind::DequantMoEWeights => Op::DequantMoEWeights {
            scheme: crate::quant::QuantScheme::Int8Block { block_size: 32 },
        },
        OpKind::ScaledGroupedMatMul => Op::ScaledGroupedMatMul {
            lhs_format: crate::quant::ScaledFormat::F8E4M3,
            rhs_format: crate::quant::ScaledFormat::F8E4M3,
            scale_layout: crate::quant::ScaleLayout::PerTensor,
            has_bias: false,
        },
        OpKind::ScatterAdd => Op::ScatterAdd { axis: 1 },
        OpKind::ScatterNd => Op::ScatterNd {
            reduction: ScatterNdReduction::None,
        },
        OpKind::ScatterElements => Op::ScatterElements {
            axis: 1,
            reduction: ScatterNdReduction::None,
        },
        OpKind::GatherNd => Op::GatherNd { batch_dims: 1 },
        OpKind::GatherElements => Op::GatherElements { axis: 1 },
        OpKind::LoraMatMul => Op::LoraMatMul { scale: 1.0 },
        OpKind::PartitionedConv => Op::PartitionedConv { block: 1 },
        OpKind::DequantMatMul => Op::DequantMatMul {
            scheme: crate::quant::QuantScheme::Int8Block { block_size: 32 },
        },
        OpKind::SynthMatMul => Op::SynthMatMul {
            kind: SynthKind::Codebook {
                entry_dim: 2,
                num_entries: 2,
            },
        },
        OpKind::SynthMatMulBackward => Op::SynthMatMulBackward {
            kind: SynthKind::Codebook {
                entry_dim: 2,
                num_entries: 2,
            },
            wrt: SynthBwdWrt::Dx,
        },
        OpKind::SynthReconstruct => Op::SynthReconstruct {
            kind: SynthKind::Codebook {
                entry_dim: 2,
                num_entries: 2,
            },
        },
        OpKind::SplineActivation => Op::SplineActivation {
            num_basis: 1,
            grid_min: 1.0,
            grid_max: 1.0,
        },
        OpKind::SplineActivationBackwardX => Op::SplineActivationBackwardX {
            num_basis: 1,
            grid_min: 1.0,
            grid_max: 1.0,
        },
        OpKind::SplineActivationBackwardCoeff => Op::SplineActivationBackwardCoeff {
            num_basis: 1,
            grid_min: 1.0,
            grid_max: 1.0,
        },
        OpKind::QMatMul => Op::QMatMul {
            x_zp: 1,
            w_zp: 1,
            out_zp: 1,
            mult: 1.0,
        },
        OpKind::QConv2d => Op::QConv2d {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
            x_zp: 1,
            w_zp: 1,
            out_zp: 1,
            mult: 1.0,
        },
        OpKind::ScaledMatMul => Op::ScaledMatMul {
            lhs_format: crate::quant::ScaledFormat::F8E4M3,
            rhs_format: crate::quant::ScaledFormat::F8E4M3,
            scale_layout: crate::quant::ScaleLayout::PerTensor,
            has_bias: false,
        },
        OpKind::ScaledQuantize => Op::ScaledQuantize {
            format: crate::quant::ScaledFormat::F8E4M3,
            scale_layout: crate::quant::ScaleLayout::PerTensor,
        },
        OpKind::ScaledQuantScale => Op::ScaledQuantScale {
            format: crate::quant::ScaledFormat::F8E4M3,
            scale_layout: crate::quant::ScaleLayout::PerTensor,
        },
        OpKind::ScaledDequantize => Op::ScaledDequantize {
            format: crate::quant::ScaledFormat::F8E4M3,
            scale_layout: crate::quant::ScaleLayout::PerTensor,
        },
        OpKind::SelectiveScan => Op::SelectiveScan { state_size: 1 },
        OpKind::GatedDeltaNet => Op::GatedDeltaNet {
            state_size: 1,
            carry_state: false,
            gate_per_channel: false,
        },
        OpKind::GatedDeltaNetBackward => Op::GatedDeltaNetBackward {
            state_size: 1,
            carry_state: false,
            gate_per_channel: false,
        },
        OpKind::Lstm => Op::Lstm {
            hidden_size: 1,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        OpKind::Gru => Op::Gru {
            hidden_size: 1,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        OpKind::Rnn => Op::Rnn {
            hidden_size: 1,
            num_layers: 1,
            bidirectional: false,
            carry: false,
            relu: false,
        },
        OpKind::Mamba2 => Op::Mamba2 {
            head_dim: 1,
            state_size: 1,
        },
        OpKind::FusedSwiGLU => Op::FusedSwiGLU {
            cast_to: None,
            gate_first: false,
        },
        OpKind::FusedMatMulBiasAct => Op::FusedMatMulBiasAct { activation: None },
        OpKind::FusedMatMulResidual => Op::FusedMatMulResidual,
        OpKind::FusedConvBiasAct => Op::FusedConvBiasAct {
            kernel_size: Vec::new(),
            stride: Vec::new(),
            padding: Vec::new(),
            dilation: Vec::new(),
            groups: 1,
            activation: None,
            has_residual: false,
        },
        OpKind::FusedResidualLN => Op::FusedResidualLN {
            has_bias: false,
            eps: 1.0,
        },
        OpKind::FusedResidualRmsNorm => Op::FusedResidualRmsNorm {
            has_bias: false,
            eps: 1.0,
        },
        OpKind::FusedAttentionBlock => Op::FusedAttentionBlock {
            num_heads: 1,
            head_dim: 1,
            has_bias: false,
            has_rope: false,
        },
        OpKind::FusedTransformerLayer => Op::FusedTransformerLayer {
            num_heads: 1,
            head_dim: 1,
            intermediate_size: 1,
            eps1: 1.0,
            eps2: 1.0,
            activation: Activation::Gelu,
            has_bias: false,
        },
        OpKind::If => Op::If {
            then_branch: Box::new(crate::Graph::new("b")),
            else_branch: Box::new(crate::Graph::new("b")),
        },
        OpKind::While => Op::While {
            cond: Box::new(crate::Graph::new("b")),
            body: Box::new(crate::Graph::new("b")),
            max_iterations: None,
        },
        OpKind::Scan => Op::Scan {
            body: Box::new(crate::Graph::new("b")),
            length: 1,
            save_trajectory: false,
            num_bcast: 1,
            num_xs: 1,
            num_checkpoints: 1,
        },
        OpKind::ScanBackward => Op::ScanBackward {
            body_vjp: Box::new(crate::Graph::new("b")),
            length: 1,
            save_trajectory: false,
            num_xs: 1,
            num_checkpoints: 1,
            forward_body: None,
        },
        OpKind::ScanBackwardXs => Op::ScanBackwardXs {
            body_vjp: Box::new(crate::Graph::new("b")),
            length: 1,
            save_trajectory: false,
            num_xs: 1,
            xs_idx: 1,
            num_checkpoints: 1,
            forward_body: None,
        },
        OpKind::GaussianSplatRender => Op::GaussianSplatRender {
            width: 1,
            height: 1,
            tile_size: 1,
            radius_scale: 1.0,
            alpha_cutoff: 1.0,
            max_splat_steps: 1,
            transmittance_threshold: 1.0,
            max_list_entries: 1,
        },
        OpKind::GaussianSplatRenderBackward => Op::GaussianSplatRenderBackward {
            width: 1,
            height: 1,
            tile_size: 1,
            radius_scale: 1.0,
            alpha_cutoff: 1.0,
            max_splat_steps: 1,
            transmittance_threshold: 1.0,
            max_list_entries: 1,
            loss_grad_clip: 1.0,
            sh_band: 1,
            max_anisotropy: 1.0,
        },
        OpKind::GaussianSplatPrepare => Op::GaussianSplatPrepare {
            width: 1,
            height: 1,
            tile_size: 1,
            radius_scale: 1.0,
            alpha_cutoff: 1.0,
            max_splat_steps: 1,
            transmittance_threshold: 1.0,
            max_list_entries: 1,
        },
        OpKind::GaussianSplatRasterize => Op::GaussianSplatRasterize {
            width: 1,
            height: 1,
            tile_size: 1,
            alpha_cutoff: 1.0,
            max_splat_steps: 1,
            transmittance_threshold: 1.0,
            max_list_entries: 1,
        },
        OpKind::Custom => Op::Custom {
            name: "x".into(),
            num_inputs: 1,
            attrs: Vec::new(),
        },
        OpKind::CustomFn => Op::CustomFn {
            fwd_body: Box::new(crate::Graph::new("b")),
            vjp_body: None,
            jvp_body: None,
            num_inputs: 1,
        },
        OpKind::Fft => Op::Fft {
            inverse: false,
            norm: crate::fft::FftNorm::Backward,
        },
        OpKind::FftQ => Op::FftQ {
            inverse: false,
            norm: crate::fft::FftNorm::Backward,
            scale: crate::fft::FftQScale::None,
        },
        OpKind::FftButterflyStage => Op::FftButterflyStage { stage: 1, n_fft: 1 },
        OpKind::LogMel => Op::LogMel,
        OpKind::LogMelBackward => Op::LogMelBackward,
        OpKind::WelchPeaks => Op::WelchPeaks {
            k: 1,
            n_segments: 1,
        },
        OpKind::BiMap => Op::BiMap,
        OpKind::ReEig => Op::ReEig { eps: 1.0 },
        OpKind::LogEig => Op::LogEig { eps: 1.0 },
        OpKind::SpdBatchNorm => Op::SpdBatchNorm { eps: 1.0 },
        OpKind::SpdKarcherMean => Op::SpdKarcherMean { iters: 1, tol: 1.0 },
        OpKind::SpdKarcherMeanWeighted => Op::SpdKarcherMeanWeighted { iters: 1, tol: 1.0 },
        OpKind::SpdLogMap => Op::SpdLogMap,
        OpKind::SpdExpMap => Op::SpdExpMap,
        OpKind::SpdParallelTransport => Op::SpdParallelTransport,
        OpKind::SpdMatrixFnBatch => Op::SpdMatrixFnBatch {
            kind: SpdMatFn::Logm,
        },
        OpKind::ReEigBackward => Op::ReEigBackward { eps: 1.0 },
        OpKind::LogEigBackward => Op::LogEigBackward { eps: 1.0 },
        OpKind::SpdBatchNormBackwardX => Op::SpdBatchNormBackwardX { eps: 1.0 },
        OpKind::SpdBatchNormBackwardG => Op::SpdBatchNormBackwardG { eps: 1.0 },
        OpKind::SpdLogMapBackward => Op::SpdLogMapBackward,
        OpKind::SpdExpMapBackward => Op::SpdExpMapBackward,
        OpKind::SpdParallelTransportBackward => Op::SpdParallelTransportBackward,
        OpKind::SpdMatrixFnBatchBackward => Op::SpdMatrixFnBatchBackward {
            kind: SpdMatFn::Logm,
        },
        OpKind::Eigh => Op::Eigh,
        OpKind::EighBackward => Op::EighBackward,
        OpKind::EighBatch => Op::EighBatch,
        OpKind::EighBatchBackward => Op::EighBatchBackward,
        OpKind::AdaLayerNorm => Op::AdaLayerNorm {
            norm: AdaNormKind::LayerNorm,
            eps: 1.0,
        },
        OpKind::AdaLayerNormBackward => Op::AdaLayerNormBackward {
            norm: AdaNormKind::LayerNorm,
            eps: 1.0,
        },
        OpKind::GatedResidual => Op::GatedResidual,
        OpKind::GatedResidualBackward => Op::GatedResidualBackward,
    }
}

/// Every [`OpKind`], each paired with a constructible sample.
///
/// Derived from [`sample`], so it cannot fall behind the op set.
pub fn all() -> Vec<(OpKind, Op)> {
    ALL_KINDS.iter().map(|&k| (k, sample(k))).collect()
}

/// Every [`OpKind`] variant.
pub const ALL_KINDS: &[OpKind] = &[
    OpKind::Input,
    OpKind::Param,
    OpKind::Constant,
    OpKind::Activation,
    OpKind::Cast,
    OpKind::StopGradient,
    OpKind::Quantize,
    OpKind::Dequantize,
    OpKind::FakeQuantize,
    OpKind::FakeQuantizeLSQ,
    OpKind::FakeQuantizeLSQBackwardX,
    OpKind::FakeQuantizeLSQBackwardScale,
    OpKind::Binary,
    OpKind::Compare,
    OpKind::Where,
    OpKind::Fma,
    OpKind::ElementwiseRegion,
    OpKind::TransformRegion,
    OpKind::BatchElementwiseRegion,
    OpKind::MatMul,
    OpKind::DotGeneral,
    OpKind::DenseSolve,
    OpKind::BatchedDenseSolve,
    OpKind::Cholesky,
    OpKind::TriangularSolve,
    OpKind::Det,
    OpKind::LogDet,
    OpKind::Sort,
    OpKind::ArgSort,
    OpKind::Svd,
    OpKind::Qr,
    OpKind::LayerNorm,
    OpKind::LayerNorm2d,
    OpKind::GroupNorm,
    OpKind::BatchNormInference,
    OpKind::RmsNorm,
    OpKind::ResizeNearest2x,
    OpKind::Interpolate3d,
    OpKind::Attention,
    OpKind::Rope,
    OpKind::AxialRope2d,
    OpKind::Reshape,
    OpKind::Transpose,
    OpKind::Narrow,
    OpKind::Concat,
    OpKind::KvAppend,
    OpKind::Expand,
    OpKind::Gather,
    OpKind::Reverse,
    OpKind::Pad,
    OpKind::Slice,
    OpKind::Roll,
    OpKind::Clamp,
    OpKind::Tile,
    OpKind::Trilu,
    OpKind::Reduce,
    OpKind::Histogram,
    OpKind::Softmax,
    OpKind::Cumsum,
    OpKind::CumProd,
    OpKind::CumMax,
    OpKind::ArgMax,
    OpKind::ArgMin,
    OpKind::TopK,
    OpKind::Sample,
    OpKind::RngNormal,
    OpKind::RngUniform,
    OpKind::Conv,
    OpKind::Im2Col,
    OpKind::ConvTranspose2d,
    OpKind::Conv3d,
    OpKind::ConvTranspose3d,
    OpKind::Pool,
    OpKind::ReluBackward,
    OpKind::ActivationBackward,
    OpKind::FakeQuantizeBackward,
    OpKind::ComplexNormSq,
    OpKind::ComplexNormSqBackward,
    OpKind::Conjugate,
    OpKind::MaxPool2dBackward,
    OpKind::Conv2dBackwardInput,
    OpKind::Conv2dBackwardWeight,
    OpKind::MaxPool3dBackward,
    OpKind::Conv3dBackwardInput,
    OpKind::Conv3dBackwardWeight,
    OpKind::SoftmaxCrossEntropy,
    OpKind::SoftmaxCrossEntropyWithLogits,
    OpKind::SoftmaxCrossEntropyBackward,
    OpKind::AttentionBackward,
    OpKind::AttentionBackwardAll,
    OpKind::LayerNormBackwardInput,
    OpKind::LayerNormBackwardGamma,
    OpKind::RmsNormBackwardInput,
    OpKind::RmsNormBackwardGamma,
    OpKind::RmsNormBackwardBeta,
    OpKind::RopeBackward,
    OpKind::GroupNormBackwardInput,
    OpKind::GroupNormBackwardGamma,
    OpKind::GroupNormBackwardBeta,
    OpKind::BatchNormInferenceBackwardInput,
    OpKind::BatchNormInferenceBackwardGamma,
    OpKind::BatchNormInferenceBackwardBeta,
    OpKind::CumsumBackward,
    OpKind::GatherBackward,
    OpKind::GroupedMatMul,
    OpKind::DequantGroupedMatMul,
    OpKind::DequantGroupedMatMulMlx,
    OpKind::DequantMoEWeights,
    OpKind::ScaledGroupedMatMul,
    OpKind::ScatterAdd,
    OpKind::ScatterNd,
    OpKind::ScatterElements,
    OpKind::GatherNd,
    OpKind::GatherElements,
    OpKind::LoraMatMul,
    OpKind::PartitionedConv,
    OpKind::DequantMatMul,
    OpKind::SynthMatMul,
    OpKind::SynthMatMulBackward,
    OpKind::SynthReconstruct,
    OpKind::SplineActivation,
    OpKind::SplineActivationBackwardX,
    OpKind::SplineActivationBackwardCoeff,
    OpKind::QMatMul,
    OpKind::QConv2d,
    OpKind::ScaledMatMul,
    OpKind::ScaledQuantize,
    OpKind::ScaledQuantScale,
    OpKind::ScaledDequantize,
    OpKind::SelectiveScan,
    OpKind::GatedDeltaNet,
    OpKind::GatedDeltaNetBackward,
    OpKind::Lstm,
    OpKind::Gru,
    OpKind::Rnn,
    OpKind::Mamba2,
    OpKind::FusedSwiGLU,
    OpKind::FusedMatMulBiasAct,
    OpKind::FusedMatMulResidual,
    OpKind::FusedConvBiasAct,
    OpKind::FusedResidualLN,
    OpKind::FusedResidualRmsNorm,
    OpKind::FusedAttentionBlock,
    OpKind::FusedTransformerLayer,
    OpKind::If,
    OpKind::While,
    OpKind::Scan,
    OpKind::ScanBackward,
    OpKind::ScanBackwardXs,
    OpKind::GaussianSplatRender,
    OpKind::GaussianSplatRenderBackward,
    OpKind::GaussianSplatPrepare,
    OpKind::GaussianSplatRasterize,
    OpKind::Custom,
    OpKind::CustomFn,
    OpKind::Fft,
    OpKind::FftQ,
    OpKind::FftButterflyStage,
    OpKind::LogMel,
    OpKind::LogMelBackward,
    OpKind::WelchPeaks,
    OpKind::BiMap,
    OpKind::ReEig,
    OpKind::LogEig,
    OpKind::SpdBatchNorm,
    OpKind::SpdKarcherMean,
    OpKind::ReEigBackward,
    OpKind::LogEigBackward,
    OpKind::SpdBatchNormBackwardX,
    OpKind::SpdBatchNormBackwardG,
    OpKind::SpdKarcherMeanWeighted,
    OpKind::SpdLogMap,
    OpKind::SpdExpMap,
    OpKind::SpdParallelTransport,
    OpKind::SpdMatrixFnBatch,
    OpKind::SpdLogMapBackward,
    OpKind::SpdExpMapBackward,
    OpKind::SpdParallelTransportBackward,
    OpKind::SpdMatrixFnBatchBackward,
    OpKind::Eigh,
    OpKind::EighBackward,
    OpKind::EighBatch,
    OpKind::EighBatchBackward,
    OpKind::AdaLayerNorm,
    OpKind::AdaLayerNormBackward,
    OpKind::GatedResidual,
    OpKind::GatedResidualBackward,
];

// ── arity that depends on a field ──────────────────────────────────────────

/// Kinds whose operand count is *carried in a field*, so no static upper bound
/// exists: `Custom { num_inputs }`, `Scan { num_xs }`, and friends accept
/// whatever the field says.
pub const COUNT_FROM_FIELD: &[OpKind] = &[
    OpKind::Custom,
    OpKind::CustomFn,
    OpKind::ElementwiseRegion,
    OpKind::Scan,
    OpKind::ScanBackward,
    OpKind::ScanBackwardXs,
];

/// Every setting of `kind` that changes its arity.
///
/// [`sample`] pins one point of a field-dependent arity — `has_bias: false`,
/// `MaskKind::None`, and so on. A gate that reasons about the *widest* legal
/// operand list needs the other settings too, or it reports a legitimate read
/// as out of range.
pub fn variants(kind: OpKind) -> Vec<Op> {
    let base = sample(kind);
    let mut out = vec![base.clone()];
    let mut alt = base.clone();
    // Flip the one field that moves the count. `..` keeps every other field,
    // so this cannot fall out of step with the variant's shape.
    let changed = match &mut alt {
        Op::Attention { mask_kind, .. }
        | Op::AttentionBackward { mask_kind, .. }
        | Op::AttentionBackwardAll { mask_kind, .. } => {
            *mask_kind = MaskKind::Custom;
            true
        }
        Op::Lstm { carry, .. } | Op::Gru { carry, .. } | Op::Rnn { carry, .. } => {
            *carry = !*carry;
            true
        }
        Op::GatedDeltaNet { carry_state, .. } | Op::GatedDeltaNetBackward { carry_state, .. } => {
            *carry_state = !*carry_state;
            true
        }
        Op::ScaledMatMul { has_bias, .. }
        | Op::ScaledGroupedMatMul { has_bias, .. }
        | Op::FusedResidualLN { has_bias, .. }
        | Op::FusedResidualRmsNorm { has_bias, .. }
        | Op::FusedTransformerLayer { has_bias, .. } => {
            *has_bias = !*has_bias;
            true
        }
        Op::FusedConvBiasAct { has_residual, .. } => {
            *has_residual = !*has_residual;
            true
        }
        Op::FakeQuantize { scale_mode, .. } => {
            *scale_mode = ScaleMode::Fixed;
            true
        }
        Op::DequantMatMul { scheme } => {
            *scheme = crate::quant::QuantScheme::GgufQ4K;
            true
        }
        _ => false,
    };
    if changed {
        out.push(alt);
    }
    // Two independent flags, so the widest arity needs both set — flipping
    // one at a time would understate it.
    if let Op::FusedAttentionBlock { .. } = base {
        let mut both = base.clone();
        if let Op::FusedAttentionBlock {
            has_bias, has_rope, ..
        } = &mut both
        {
            *has_bias = true;
            *has_rope = true;
        }
        out.push(both);
    }
    out
}

/// The widest legal operand count for `kind`, or `None` when there is no
/// static upper bound.
///
/// Unbounded for two distinct reasons, and a caller exempting an op from an
/// upper-bound check should know which:
///
/// * the count is carried in a field ([`COUNT_FROM_FIELD`]) — `Custom`, `Scan`;
/// * the op is genuinely variadic ([`Arity::AtLeast`]) — `Concat`, `If`,
///   `While`.
///
/// Both are legitimate; `count_from_field_is_exactly_the_unbounded_set` pins
/// that nothing else slips into the exempt set, since an op exempted by
/// accident is never checked at all.
pub fn max_operands(kind: OpKind) -> Option<usize> {
    if COUNT_FROM_FIELD.contains(&kind) {
        return None;
    }
    variants(kind)
        .iter()
        .map(|op| op.arity().max_operands())
        .try_fold(0usize, |acc, m| m.map(|m| acc.max(m)))
}
