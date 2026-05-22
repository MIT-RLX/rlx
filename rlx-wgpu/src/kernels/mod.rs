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

//! WGSL kernel sources + per-kernel pipeline cache.
//!
//! Pipelines are content-addressed: same WGSL source + same entry
//! point yields the same pipeline. We hold them in `OnceLock`s so a
//! single device dispatches every (graph, op) pair against a cached
//! compilation.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};

pub const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
pub const MATMUL_WIDE_WGSL: &str = include_str!("matmul_wide.wgsl");
pub const MATMUL_F16W_WGSL: &str = include_str!("matmul_f16w.wgsl");
pub const MATMUL_F16_COMPUTE_WGSL: &str = include_str!("matmul_f16_compute.wgsl");
pub const MATMUL_COOP16_WGSL: &str = include_str!("matmul_coop16.wgsl");
pub const MATMUL_COOP_F32_WGSL: &str = include_str!("matmul_coop_f32.wgsl");
pub const CAST_F32_TO_F16_WGSL: &str = include_str!("cast_f32_to_f16.wgsl");
pub const BINARY_WGSL: &str = include_str!("binary.wgsl");
pub const UNARY_WGSL: &str = include_str!("unary.wgsl");
pub const COMPARE_WGSL: &str = include_str!("compare.wgsl");
pub const WHERE_WGSL: &str = include_str!("where.wgsl");
pub const REDUCE_WGSL: &str = include_str!("reduce.wgsl");
pub const SOFTMAX_WGSL: &str = include_str!("softmax.wgsl");
pub const LAYERNORM_WGSL: &str = include_str!("layernorm.wgsl");
pub const RMS_NORM_BWD_WGSL: &str = include_str!("rms_norm_backward.wgsl");
pub const CUMSUM_BWD_WGSL: &str = include_str!("cumsum_backward.wgsl");
pub const ROPE_BWD_WGSL: &str = include_str!("rope_backward.wgsl");
pub const GATHER_BWD_WGSL: &str = include_str!("gather_backward.wgsl");
pub const CUMSUM_WGSL: &str = include_str!("cumsum.wgsl");
pub const FFT_WGSL: &str = include_str!("fft.wgsl");
pub const COPY_WGSL: &str = include_str!("copy.wgsl");
pub const ELEMENTWISE_REGION_WGSL: &str = include_str!("elementwise_region.wgsl");
pub const TRANSPOSE_WGSL: &str = include_str!("transpose.wgsl");
pub const NARROW_WGSL: &str = include_str!("narrow.wgsl");
pub const CONCAT_WGSL: &str = include_str!("concat.wgsl");
pub const GATHER_WGSL: &str = include_str!("gather.wgsl");
pub const GATHER_AXIS_WGSL: &str = include_str!("gather_axis.wgsl");
pub const ATTENTION_WGSL: &str = include_str!("attention.wgsl");
pub const ATTENTION_BWD_WGSL: &str = include_str!("attention_bwd.wgsl");
pub const ROPE_WGSL: &str = include_str!("rope.wgsl");
pub const EXPAND_WGSL: &str = include_str!("expand.wgsl");
pub const ARGMAX_WGSL: &str = include_str!("argmax.wgsl");
pub const POOL2D_WGSL: &str = include_str!("pool2d.wgsl");
pub const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
pub const POOL1D_WGSL: &str = include_str!("pool1d.wgsl");
pub const POOL3D_WGSL: &str = include_str!("pool3d.wgsl");
pub const CONV1D_WGSL: &str = include_str!("conv1d.wgsl");
pub const CONV3D_WGSL: &str = include_str!("conv3d.wgsl");
pub const SCATTER_ADD_WGSL: &str = include_str!("scatter_add.wgsl");
pub const TOPK_WGSL: &str = include_str!("topk.wgsl");
pub const GROUPED_MATMUL_WGSL: &str = include_str!("grouped_matmul.wgsl");
pub const SAMPLE_WGSL: &str = include_str!("sample.wgsl");
pub const SELECTIVE_SCAN_WGSL: &str = include_str!("selective_scan.wgsl");
pub const DEQUANT_MATMUL_WGSL: &str = include_str!("dequant_matmul.wgsl");
pub const FUSED_RESIDUAL_LN_WGSL: &str = include_str!("fused_residual_ln.wgsl");
pub const FUSED_RESIDUAL_LN_TEE_WGSL: &str = include_str!("fused_residual_ln_tee.wgsl");
pub const MATMUL_QKV_WGSL: &str = include_str!("matmul_qkv.wgsl");
pub const MATMUL_QKV_COOP_F32_WGSL: &str = include_str!("matmul_qkv_coop_f32.wgsl");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub batch: u32,
    pub a_batch_stride: u32,
    pub b_batch_stride: u32,
    pub c_batch_stride: u32,
    pub has_bias: u32,
    pub bias_off: u32,
    pub act_id: u32, // 0xFFFF = no activation
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Shared layout for binary, compare. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for unary kernel. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UnaryParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for where (3-input select). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhereParams {
    pub n: u32,
    pub cond_off: u32,
    pub x_off: u32,
    pub y_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for reductions. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ReduceParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for softmax. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SoftmaxParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for LayerNorm / RmsNorm.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayerNormParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub eps_bits: u32, // bitcast::<u32>(eps)
    pub op: u32,       // 0=LayerNorm, 1=RmsNorm
}

/// RMSNorm backward kernel params (f32 element offsets). `wrt`: 0=dx, 1=dgamma, 2=dbeta.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RmsNormBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub wrt: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumsumBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub exclusive: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RopeBwdParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub head_dim: u32,
    pub n_rot: u32,
    pub dy_off: u32,
    pub cos_off: u32,
    pub sin_off: u32,
    pub dx_off: u32,
    pub cos_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherBwdParams {
    pub outer: u32,
    pub axis_dim: u32,
    pub num_idx: u32,
    pub trailing: u32,
    pub dy_off: u32,
    pub idx_off: u32,
    pub dst_off: u32,
    pub _p0: u32,
}

/// Layout for cumsum. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumsumParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub exclusive: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for FFT. 32 bytes. Matches `fft.wgsl::Params`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftParams {
    pub src_off: u32,
    pub dst_off: u32,
    pub n: u32,
    pub log2n: u32,
    pub inverse: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// PLAN L2 — interpreted N-ary element-wise region. Chain encoded
/// as 4 u32s per step (op_kind, op_sub, lhs_enc, rhs_enc). Operand
/// encoding: bit 31 = src kind (0=Input, 1=Step), bits 0..30 = index.
/// `scalar_input_mask` is the per-input scalar fast-path bitfield;
/// `input_modulus[i]` is the per-input element count for trailing-
/// shape broadcast (`0` ⇒ no broadcast, kernel reads gid; `>0` ⇒
/// kernel reads `gid % input_modulus[i]`). Fixed cap at 32 steps +
/// 16 inputs (ample for chains rlx produces). 12 padding bytes
/// after `scalar_input_mask` align the next array on WGSL's
/// 16-byte uniform alignment boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ElementwiseRegionParams {
    pub len: u32,
    pub num_inputs: u32,
    pub num_steps: u32,
    pub dst_off: u32,
    pub input_offs: [u32; 16],
    pub chain: [u32; 128], // 32 steps * 4 u32s
    pub scalar_input_mask: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub input_modulus: [u32; 16],
}

/// Layout shared by Reshape / Cast / generic full copy. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CopyParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Layout for transpose (uses the 3-binding bind layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TransposeParams {
    pub rank: u32,
    pub out_total: u32,
    pub in_off: u32,
    pub out_off: u32,
    /// PLAN L1 — precomputed at compile time. `1` when `perm[0] == 0`
    /// (= bucket axis stays at output axis 0). Active-extent path
    /// scales `out_total` proportionally only when this is `1`.
    pub bucket_outermost: u32,
    /// PLAN L1 — `out_dims[0]` for active-extent scaling math.
    pub out_dim_0: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for narrow / concat (the same struct serves both).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NarrowConcatParams {
    pub total: u32, // total elements (output for narrow, input for concat)
    pub outer: u32,
    pub inner: u32,
    pub axis_in_size: u32,
    pub axis_out_size: u32,
    pub start: u32,
    pub in_off: u32,
    pub out_off: u32,
}

/// Layout for gather.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherParams {
    pub n_out: u32,
    pub n_idx: u32,
    pub dim: u32,
    pub vocab: u32,
    pub in_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
    pub _p0: u32,
}

/// Layout for gather along a non-zero axis.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatherAxisParams {
    pub total: u32,
    pub outer: u32,
    pub axis_dim: u32,
    pub num_idx: u32,
    pub trailing: u32,
    pub table_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
}

/// Layout for fused SDPA.
///
/// Per-tensor (Q, K, V, output) strides are passed explicitly so the
/// kernel can read either canonical [B, H, S, D] or transposed
/// [B, S, H, D] without inserting upstream Transpose dispatches. The
/// layout-elimination saves ~24 transpose dispatches per BERT-L6
/// forward (one per Q/K/V/output × layers), each ~50µs at small batch.
///
/// The `seq_q_stride` / `seq_k_stride` fields are retained because
/// they describe the MASK layout `[B, H, S_q, S_k]` (separate from
/// Q/K/V layout), used by `MaskKind::Custom`.
///
/// 144 bytes (36 u32s); WebGPU uniform-buffer 16-byte alignment OK.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttentionParams {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub out_off: u32,
    pub mask_off: u32,
    pub mask_kind: u32,
    pub scale_bits: u32,
    pub window: u32,
    /// MASK address strides. Mask address math (per-element):
    ///   addr = mask_off
    ///        + b  * mask_batch_stride
    ///        + h  * mask_head_stride
    ///        + qi * seq_q_stride         (per-query stride)
    ///        + s  * seq_k_stride         (per-key   stride)
    /// Setting some strides to 0 lets the kernel read a *broadcast*
    /// mask without materializing the broadcast. e.g. BERT padding mask
    /// `[B, S]`: mask_batch_stride=S, mask_head_stride=0, seq_q_stride=0,
    /// seq_k_stride=1. Saves the Expand pre-pass that unfuse used to
    /// emit per attention block.
    pub seq_q_stride: u32,
    pub seq_k_stride: u32,
    pub mask_batch_stride: u32,
    pub mask_head_stride: u32,
    pub _pad_mask_0: u32,
    pub _pad_mask_1: u32,
    pub _pad_mask_2: u32,

    // Q stride triple (in f32 elements). For [B, H, S, D]:
    //   q_batch_stride = H·S·D, q_head_stride = S·D, q_seq_stride = D
    // For [B, S, H, D]:
    //   q_batch_stride = S·H·D, q_head_stride = D,   q_seq_stride = H·D
    pub q_batch_stride: u32,
    pub q_head_stride: u32,
    pub q_seq_stride: u32,
    pub _pad_q: u32,

    pub k_batch_stride: u32,
    pub k_head_stride: u32,
    pub k_seq_stride: u32,
    pub _pad_k: u32,

    pub v_batch_stride: u32,
    pub v_head_stride: u32,
    pub v_seq_stride: u32,
    pub _pad_v: u32,

    pub o_batch_stride: u32,
    pub o_head_stride: u32,
    pub o_seq_stride: u32,
    pub _pad_o: u32,
}

/// Layout for [`attention_bwd.wgsl`] — forward strides + `dy_off` + `wrt`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttentionBwdParams {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub mask_off: u32,
    pub mask_kind: u32,
    pub scale_bits: u32,
    pub window: u32,
    pub wrt: u32,
    pub seq_q_stride: u32,
    pub seq_k_stride: u32,
    pub mask_batch_stride: u32,
    pub mask_head_stride: u32,
    pub _pad_mask_0: u32,
    pub _pad_mask_1: u32,
    pub _pad_mask_2: u32,
    pub q_batch_stride: u32,
    pub q_head_stride: u32,
    pub q_seq_stride: u32,
    pub _pad_q: u32,
    pub k_batch_stride: u32,
    pub k_head_stride: u32,
    pub k_seq_stride: u32,
    pub _pad_k: u32,
    pub v_batch_stride: u32,
    pub v_head_stride: u32,
    pub v_seq_stride: u32,
    pub _pad_v: u32,
    pub o_batch_stride: u32,
    pub o_head_stride: u32,
    pub o_seq_stride: u32,
    pub _pad_o: u32,
}

/// Layout for Rope.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RopeParams {
    pub n_total: u32,
    pub seq: u32,
    pub head_dim: u32,
    pub half: u32,
    pub in_off: u32,
    pub cos_off: u32,
    pub sin_off: u32,
    pub out_off: u32,
    pub last_dim: u32,
    /// PLAN L1 — set at compile time. Together with `seq_stride`,
    /// lets the WGSL kernel decompose iteration index into
    /// `(bi, si, d)` while indexing into the underlying full-extent
    /// buffer. `n_total` is the runtime-scaled iteration bound;
    /// `seq_stride` is the compile-time-fixed full seq for stride.
    pub batch: u32,
    pub seq_stride: u32,
    pub _p2: u32,
}

/// Layout for Expand. Mirrors TransposeParams (rank, total, offsets);
/// per-axis dims/strides ride in the meta storage buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ExpandParams {
    pub rank: u32,
    pub out_total: u32,
    pub in_off: u32,
    pub out_off: u32,
    /// PLAN L1 — precomputed at compile time. `1` when the bucket
    /// axis stays at output axis 0 after the expand mapping.
    pub bucket_outermost: u32,
    /// PLAN L1 — `out_dims[0]` for active-extent scaling math.
    pub out_dim_0: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for argmax (matches Reduce shape).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ArgmaxParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Layout for Pool2D NCHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool2dParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for Conv2D NCHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv2dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub h: u32,
    pub w: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
}

/// Layout for Pool1D NCL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool1dParams {
    pub n: u32,
    pub c: u32,
    pub l: u32,
    pub l_out: u32,
    pub kl: u32,
    pub sl: u32,
    pub pl: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
}

/// Layout for Pool3D NCDHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Pool3dParams {
    pub n: u32,
    pub c: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub op: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for Conv1D NCL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv1dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub l: u32,
    pub l_out: u32,
    pub kl: u32,
    pub sl: u32,
    pub pl: u32,
    pub dl: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for DequantMatMul. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantMatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub block_size: u32,
    pub scheme_id: u32,
    pub x_off: u32,
    pub w_off: u32,
    pub scale_off: u32,
    pub zp_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for FusedResidualLN-Tee. 48 bytes (12 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedResidualLnTeeParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub residual_off: u32,
    pub bias_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub sum_off: u32,
    pub ln_out_off: u32,
    pub eps_bits: u32,
    pub has_bias: u32,
    pub _p0: u32,
}

/// Layout for matmul_qkv (split-write QKV matmul).
/// 64 bytes (16 u32s); WebGPU uniform-buffer 16-byte alignment OK.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MatmulQkvParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub head_width: u32,
    pub has_bias: u32,
    pub bias_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Layout for FusedResidualLN. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FusedResidualLnParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub residual_off: u32,
    pub bias_off: u32,
    pub gamma_off: u32,
    pub beta_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub has_bias: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for SelectiveScan. 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SelectiveScanParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub state_size: u32,
    pub x_off: u32,
    pub delta_off: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    /// PLAN L1 — full-extent seq stride for per-batch offset math.
    /// Stays at compile-time `seq` even when runtime `seq` is scaled,
    /// so per-batch arena offsets stay correct under active-extent.
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
}

/// Layout for Sample. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SampleParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub top_k: u32,
    pub top_p_bits: u32,
    pub temp_bits: u32,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GroupedMatMul. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroupedMatmulParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub num_experts: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub idx_off: u32,
    pub out_off: u32,
}

/// Layout for TopK. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TopKParams {
    pub outer: u32,
    pub inner: u32,
    pub k: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for ScatterAdd. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScatterAddParams {
    pub op: u32, // 0 = zero phase, 1 = accumulate phase
    pub out_off: u32,
    pub upd_off: u32,
    pub idx_off: u32,
    pub out_total: u32,
    pub num_updates: u32,
    pub trailing: u32,
    pub out_dim: u32,
}

/// Layout for Conv3D NCDHW.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dParams {
    pub n: u32,
    pub c_in: u32,
    pub c_out: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub d_out: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kd: u32,
    pub kh: u32,
    pub kw: u32,
    pub sd: u32,
    pub sh: u32,
    pub sw: u32,
    pub pd: u32,
    pub ph: u32,
    pub pw: u32,
    pub dd: u32,
    pub dh: u32,
    pub dw: u32,
    pub groups: u32,
    pub in_off: u32,
    pub w_off: u32,
    pub out_off: u32,
    pub _p0: u32,
}

/// Lazy-init container for a compute pipeline + its bind-group layout.
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

/// Build a 4-binding compute kernel: storage(rw) / uniform / storage(ro)
/// / storage(ro). Currently unused — `matmul_coop16` switched to a
/// 3-binding layout (A is staged from arena through workgroup memory
/// instead of from a separate f16 binding). Kept for future kernels
/// that genuinely need a 4th binding.
#[allow(dead_code)]
/// Used by the cooperative-matrix matmul which needs a
/// fourth binding for the f16 activation shadow buffer.
fn build_kernel_4(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

fn build_kernel_3(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

fn build_kernel(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}

static MATMUL: OnceLock<Kernel> = OnceLock::new();
static MATMUL_WIDE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16W: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16_COMPUTE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP16: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F32: OnceLock<Kernel> = OnceLock::new();
static CAST_F32_TO_F16: OnceLock<Kernel> = OnceLock::new();
static BINARY: OnceLock<Kernel> = OnceLock::new();
static UNARY: OnceLock<Kernel> = OnceLock::new();
static COMPARE: OnceLock<Kernel> = OnceLock::new();
static WHEREK: OnceLock<Kernel> = OnceLock::new();
static REDUCE: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX: OnceLock<Kernel> = OnceLock::new();
static LAYERNORM: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD_PARAM: OnceLock<Kernel> = OnceLock::new();
static CUMSUM_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE_BWD: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ZERO: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ACC: OnceLock<Kernel> = OnceLock::new();
static CUMSUM: OnceLock<Kernel> = OnceLock::new();
static FFT: OnceLock<Kernel> = OnceLock::new();
static COPY: OnceLock<Kernel> = OnceLock::new();
static ELEMENTWISE_REGION: OnceLock<Kernel> = OnceLock::new();
static TRANSPOSE: OnceLock<Kernel> = OnceLock::new();
static NARROW: OnceLock<Kernel> = OnceLock::new();
static CONCAT: OnceLock<Kernel> = OnceLock::new();
static GATHER: OnceLock<Kernel> = OnceLock::new();
static GATHER_AXIS: OnceLock<Kernel> = OnceLock::new();
static ATTENTION: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE: OnceLock<Kernel> = OnceLock::new();
static EXPAND: OnceLock<Kernel> = OnceLock::new();
static ARGMAX: OnceLock<Kernel> = OnceLock::new();
static POOL2D: OnceLock<Kernel> = OnceLock::new();
static CONV2D: OnceLock<Kernel> = OnceLock::new();
static POOL1D: OnceLock<Kernel> = OnceLock::new();
static POOL3D: OnceLock<Kernel> = OnceLock::new();
static CONV1D: OnceLock<Kernel> = OnceLock::new();
static CONV3D: OnceLock<Kernel> = OnceLock::new();
static SCATTER_ADD: OnceLock<Kernel> = OnceLock::new();
static TOPK: OnceLock<Kernel> = OnceLock::new();
static GROUPED_MATMUL: OnceLock<Kernel> = OnceLock::new();
static SAMPLE: OnceLock<Kernel> = OnceLock::new();
static SELECTIVE_SCAN: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_MATMUL: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN_TEE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F32: OnceLock<Kernel> = OnceLock::new();

pub fn matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL.get_or_init(|| build_kernel(device, "rlx-wgpu matmul", MATMUL_WGSL, "matmul"))
}
pub fn matmul_wide_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_WIDE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide",
            MATMUL_WIDE_WGSL,
            "matmul_wide",
        )
    })
}
/// f16-weight matmul (f32 compute). Returns Some only when the device
/// exposes the `SHADER_F16` feature. EXPERIMENTAL: currently slower
/// than the f32 baseline on Apple Silicon — kept as foundation; see
/// `matmul_f16w.wgsl` for the empirical analysis.
pub fn matmul_f16w_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(MATMUL_F16W.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_f16w",
            MATMUL_F16W_WGSL,
            "matmul_f16w",
        )
    }))
}
/// f16-compute matmul: f16 operands, f16 multiply, f32 accumulator.
/// Targets the 2× f16 ALU throughput on Apple Silicon. Returns Some
/// only when the device exposes `SHADER_F16`.
pub fn matmul_f16_compute_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(MATMUL_F16_COMPUTE.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_f16_compute",
            MATMUL_F16_COMPUTE_WGSL,
            "matmul_f16_compute",
        )
    }))
}
/// Cooperative-matrix matmul (8×8 tiles, hardware GEMM units).
/// Lowers to MSL `simdgroup_matrix` on Metal and SPIR-V's
/// `OpCooperativeMatrixMulAddKHR` on Vulkan. Returns Some only when
/// the device exposes both `SHADER_F16` and
/// `EXPERIMENTAL_COOPERATIVE_MATRIX`.
pub fn matmul_coop16_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::SHADER_F16)
        || !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return None;
    }
    Some(MATMUL_COOP16.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu matmul_coop16",
            MATMUL_COOP16_WGSL,
            "matmul_coop16",
        )
    }))
}
/// Pure-f32 cooperative-matrix matmul. No SHADER_F16 needed — uses
/// `coop_mat8x8<f32>` throughout (lowers to `simdgroup_float8x8` on
/// Apple). Returns None if the cooperative-matrix feature is missing
/// OR if the device's WGSL→backend lowering can't compile it (some
/// implementations only expose half-precision coop matrices).
pub fn matmul_coop_f32_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return None;
    }
    Some(MATMUL_COOP_F32.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_coop_f32",
            MATMUL_COOP_F32_WGSL,
            "matmul_coop_f32",
        )
    }))
}
/// Mirrors a region of the f32 arena into the f16 shadow buffer.
/// Used before `matmul_coop16` for the matmul's activation operand
/// (intermediate activations don't go through `set_param` /
/// `write_f32`, so they aren't in the f16 buffer otherwise).
pub fn cast_f32_to_f16_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(CAST_F32_TO_F16.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu cast_f32_to_f16",
            CAST_F32_TO_F16_WGSL,
            "cast_f32_to_f16",
        )
    }))
}
pub fn binary_kernel(device: &wgpu::Device) -> &'static Kernel {
    BINARY.get_or_init(|| build_kernel(device, "rlx-wgpu binary", BINARY_WGSL, "binary"))
}
pub fn unary_kernel(device: &wgpu::Device) -> &'static Kernel {
    UNARY.get_or_init(|| build_kernel(device, "rlx-wgpu unary", UNARY_WGSL, "unary"))
}
pub fn compare_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPARE.get_or_init(|| build_kernel(device, "rlx-wgpu compare", COMPARE_WGSL, "compare"))
}
pub fn where_kernel(device: &wgpu::Device) -> &'static Kernel {
    WHEREK.get_or_init(|| build_kernel(device, "rlx-wgpu where", WHERE_WGSL, "where_select"))
}
pub fn reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    REDUCE.get_or_init(|| build_kernel(device, "rlx-wgpu reduce", REDUCE_WGSL, "reduce"))
}
pub fn softmax_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX.get_or_init(|| build_kernel(device, "rlx-wgpu softmax", SOFTMAX_WGSL, "softmax"))
}
pub fn layernorm_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYERNORM.get_or_init(|| build_kernel(device, "rlx-wgpu layernorm", LAYERNORM_WGSL, "norm"))
}
pub fn rms_norm_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    RMS_NORM_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rms_norm_bwd",
            RMS_NORM_BWD_WGSL,
            "rms_norm_bwd",
        )
    })
}
pub fn rms_norm_backward_param_kernel(device: &wgpu::Device) -> &'static Kernel {
    RMS_NORM_BWD_PARAM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu rms_norm_bwd_param",
            RMS_NORM_BWD_WGSL,
            "rms_norm_bwd_param",
        )
    })
}
pub fn cumsum_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM_BWD.get_or_init(|| {
        build_kernel(device, "rlx-wgpu cumsum_bwd", CUMSUM_BWD_WGSL, "cumsum_bwd")
    })
}
pub fn rope_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ROPE_BWD.get_or_init(|| build_kernel(device, "rlx-wgpu rope_bwd", ROPE_BWD_WGSL, "rope_bwd"))
}
pub fn gather_backward_zero_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ZERO.get_or_init(|| {
        build_kernel(device, "rlx-wgpu gather_bwd_zero", GATHER_BWD_WGSL, "gather_bwd_zero")
    })
}
pub fn gather_backward_acc_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ACC.get_or_init(|| {
        build_kernel(device, "rlx-wgpu gather_bwd_acc", GATHER_BWD_WGSL, "gather_bwd_acc")
    })
}
pub fn cumsum_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM.get_or_init(|| build_kernel(device, "rlx-wgpu cumsum", CUMSUM_WGSL, "cumsum"))
}
pub fn fft_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT.get_or_init(|| build_kernel(device, "rlx-wgpu fft", FFT_WGSL, "fft_radix2"))
}
pub fn copy_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY.get_or_init(|| build_kernel(device, "rlx-wgpu copy", COPY_WGSL, "copy"))
}
pub fn elementwise_region_kernel(device: &wgpu::Device) -> &'static Kernel {
    // Region params bind as a STORAGE buffer (not uniform) — WGSL's
    // uniform-storage spec requires 16-byte stride for `array<T, N>`,
    // which our packed `array<u32, N>` chain layout doesn't satisfy.
    // Storage allows arbitrary stride.
    ELEMENTWISE_REGION.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu elementwise_region",
            ELEMENTWISE_REGION_WGSL,
            "elementwise_region",
        )
    })
}

fn build_kernel_region(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // Region params: read-only storage (vs uniform).
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    Kernel { pipeline, bgl }
}
pub fn transpose_kernel(device: &wgpu::Device) -> &'static Kernel {
    TRANSPOSE
        .get_or_init(|| build_kernel_3(device, "rlx-wgpu transpose", TRANSPOSE_WGSL, "transpose"))
}
pub fn narrow_kernel(device: &wgpu::Device) -> &'static Kernel {
    NARROW.get_or_init(|| build_kernel(device, "rlx-wgpu narrow", NARROW_WGSL, "narrow"))
}
pub fn concat_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONCAT.get_or_init(|| build_kernel(device, "rlx-wgpu concat", CONCAT_WGSL, "concat"))
}
pub fn gather_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER.get_or_init(|| build_kernel(device, "rlx-wgpu gather", GATHER_WGSL, "gather"))
}
pub fn gather_axis_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_AXIS.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_axis",
            GATHER_AXIS_WGSL,
            "gather_axis",
        )
    })
}
pub fn attention_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION
        .get_or_init(|| build_kernel(device, "rlx-wgpu attention", ATTENTION_WGSL, "attention"))
}
pub fn attention_bwd_kernel(device: &wgpu::Device) -> &'static Kernel {
    ATTENTION_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu attention_bwd",
            ATTENTION_BWD_WGSL,
            "attention_bwd",
        )
    })
}
pub fn rope_kernel(device: &wgpu::Device) -> &'static Kernel {
    ROPE.get_or_init(|| build_kernel(device, "rlx-wgpu rope", ROPE_WGSL, "rope"))
}
pub fn expand_kernel(device: &wgpu::Device) -> &'static Kernel {
    EXPAND.get_or_init(|| build_kernel_3(device, "rlx-wgpu expand", EXPAND_WGSL, "expand"))
}
pub fn argmax_kernel(device: &wgpu::Device) -> &'static Kernel {
    ARGMAX.get_or_init(|| build_kernel(device, "rlx-wgpu argmax", ARGMAX_WGSL, "argmax"))
}
pub fn pool2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL2D.get_or_init(|| build_kernel(device, "rlx-wgpu pool2d", POOL2D_WGSL, "pool2d"))
}
pub fn conv2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV2D.get_or_init(|| build_kernel(device, "rlx-wgpu conv2d", CONV2D_WGSL, "conv2d"))
}
pub fn pool1d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL1D.get_or_init(|| build_kernel(device, "rlx-wgpu pool1d", POOL1D_WGSL, "pool1d"))
}
pub fn pool3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    POOL3D.get_or_init(|| build_kernel(device, "rlx-wgpu pool3d", POOL3D_WGSL, "pool3d"))
}
pub fn conv1d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV1D.get_or_init(|| build_kernel(device, "rlx-wgpu conv1d", CONV1D_WGSL, "conv1d"))
}
pub fn conv3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D.get_or_init(|| build_kernel(device, "rlx-wgpu conv3d", CONV3D_WGSL, "conv3d"))
}
pub fn scatter_add_kernel(device: &wgpu::Device) -> &'static Kernel {
    SCATTER_ADD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu scatter_add",
            SCATTER_ADD_WGSL,
            "scatter_add",
        )
    })
}
pub fn topk_kernel(device: &wgpu::Device) -> &'static Kernel {
    TOPK.get_or_init(|| build_kernel(device, "rlx-wgpu topk", TOPK_WGSL, "topk"))
}
pub fn grouped_matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUPED_MATMUL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu grouped_matmul",
            GROUPED_MATMUL_WGSL,
            "grouped_matmul",
        )
    })
}
pub fn sample_kernel(device: &wgpu::Device) -> &'static Kernel {
    SAMPLE.get_or_init(|| build_kernel(device, "rlx-wgpu sample", SAMPLE_WGSL, "sample"))
}
pub fn selective_scan_kernel(device: &wgpu::Device) -> &'static Kernel {
    SELECTIVE_SCAN.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu selective_scan",
            SELECTIVE_SCAN_WGSL,
            "selective_scan",
        )
    })
}
pub fn dequant_matmul_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_MATMUL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu dequant_matmul",
            DEQUANT_MATMUL_WGSL,
            "dequant_matmul",
        )
    })
}
pub fn fused_residual_ln_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_LN.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_ln",
            FUSED_RESIDUAL_LN_WGSL,
            "fused_residual_ln",
        )
    })
}
pub fn fused_residual_ln_tee_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_LN_TEE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_ln_tee",
            FUSED_RESIDUAL_LN_TEE_WGSL,
            "fused_residual_ln_tee",
        )
    })
}
pub fn matmul_qkv_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_QKV
        .get_or_init(|| build_kernel(device, "rlx-wgpu matmul_qkv", MATMUL_QKV_WGSL, "matmul_qkv"))
}
pub fn matmul_qkv_coop_f32_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return None;
    }
    Some(MATMUL_QKV_COOP_F32.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_qkv_coop_f32",
            MATMUL_QKV_COOP_F32_WGSL,
            "matmul_qkv_coop_f32",
        )
    }))
}
