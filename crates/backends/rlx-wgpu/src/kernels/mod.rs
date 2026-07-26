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
pub const MATMUL_WIDE_NV_WGSL: &str = include_str!("matmul_wide_nv.wgsl");
pub const MATMUL_F16W_WGSL: &str = include_str!("matmul_f16w.wgsl");
pub const MATMUL_F16_COMPUTE_WGSL: &str = include_str!("matmul_f16_compute.wgsl");
pub const MATMUL_COOP16_WGSL: &str = include_str!("matmul_coop16.wgsl");
pub const MATMUL_COOP_F32_WGSL: &str = include_str!("matmul_coop_f32.wgsl");
pub const MATMUL_COOP_F32_PORTABLE_WGSL: &str = include_str!("matmul_coop_f32_portable.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WGSL: &str = include_str!("matmul_coop_f16_vulkan.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WIDEN_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_widen.wgsl");
pub const MATMUL_COOP_F16_VULKAN_F32ACC_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_f32acc.wgsl");
pub const MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC_WGSL: &str =
    include_str!("matmul_coop_f16_vulkan_widen_f32acc.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WGSL: &str = include_str!("matmul_qkv_coop_f16_vk.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WIDEN_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_widen.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_F32ACC_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_f32acc.wgsl");
pub const MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC_WGSL: &str =
    include_str!("matmul_qkv_coop_f16_vk_widen_f32acc.wgsl");
pub const CAST_F32_TO_F16_WGSL: &str = include_str!("cast_f32_to_f16.wgsl");
pub const BINARY_WGSL: &str = include_str!("binary.wgsl");
pub const UNARY_WGSL: &str = include_str!("unary.wgsl");
pub const UNARY_F16_MIRROR_WGSL: &str = include_str!("unary_f16_mirror.wgsl");
pub const COMPARE_WGSL: &str = include_str!("compare.wgsl");
pub const WHERE_WGSL: &str = include_str!("where.wgsl");
pub const FMA_WGSL: &str = include_str!("fma.wgsl");
pub const ACTIVATION_BACKWARD_WGSL: &str = include_str!("activation_backward.wgsl");
pub const REDUCE_WGSL: &str = include_str!("reduce.wgsl");
pub const SOFTMAX_WGSL: &str = include_str!("softmax.wgsl");
pub const SOFTMAX_CROSS_ENTROPY_WGSL: &str = include_str!("softmax_cross_entropy.wgsl");
pub const SOFTMAX_CROSS_ENTROPY_BWD_WGSL: &str = include_str!("softmax_cross_entropy_bwd.wgsl");
pub const MAXPOOL2D_BWD_WGSL: &str = include_str!("maxpool2d_backward.wgsl");
pub const MAXPOOL3D_BWD_WGSL: &str = include_str!("maxpool3d_backward.wgsl");
pub const CONV3D_BWD_INPUT_WGSL: &str = include_str!("conv3d_backward_input.wgsl");
pub const CONV3D_BWD_WEIGHT_WGSL: &str = include_str!("conv3d_backward_weight.wgsl");
pub const LAYERNORM_WGSL: &str = include_str!("layernorm.wgsl");
pub const RMS_NORM_BWD_WGSL: &str = include_str!("rms_norm_backward.wgsl");
pub const LAYER_NORM_BWD_WGSL: &str = include_str!("layer_norm_backward.wgsl");
pub const CUMSUM_BWD_WGSL: &str = include_str!("cumsum_backward.wgsl");
pub const ROPE_BWD_WGSL: &str = include_str!("rope_backward.wgsl");
pub const GATHER_BWD_WGSL: &str = include_str!("gather_backward.wgsl");
pub const CUMSUM_WGSL: &str = include_str!("cumsum.wgsl");
pub const CUM_SCAN_WGSL: &str = include_str!("cum_scan.wgsl");
pub const FFT_GPU_WGSL: &str = include_str!("fft_gpu.wgsl");
/// native-gpu-fft: 32 KB on-chip radix-2/4/8 kernels (n<=4096) in a separate
/// module — only instantiated on devices with >=32 KB workgroup storage.
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_BIG_WGSL: &str = include_str!("fft_gpu_big.wgsl");
/// native-gpu-fft: portable 16 KB radix-4 kernel for n<=2048 (the default
/// on-chip path on wgpu — higher occupancy than the 32 KB module).
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_R4_16K_WGSL: &str = include_str!("fft_gpu_r4_16k.wgsl");
/// native-gpu-fft: multi-row on-chip FFT for small n (packs rows/workgroup).
#[cfg(feature = "native-gpu-fft")]
pub const FFT_GPU_MULTIROW_WGSL: &str = include_str!("fft_gpu_multirow.wgsl");
pub const COPY_WGSL: &str = include_str!("copy.wgsl");
pub const CAST_WGSL: &str = include_str!("cast.wgsl");
pub const COMPLEX_CAST_WGSL: &str = include_str!("complex_cast.wgsl");
pub const BINARY_C64_WGSL: &str = include_str!("binary_c64.wgsl");
pub const COMPLEX_WIRINGER_WGSL: &str = include_str!("complex_wirtinger.wgsl");
pub const FFT_BUTTERFLY_STAGE_WGSL: &str = include_str!("fft_butterfly_stage.wgsl");
pub const ELEMENTWISE_REGION_WGSL: &str = include_str!("elementwise_region.wgsl");
pub const TRANSPOSE_WGSL: &str = include_str!("transpose.wgsl");
pub const NARROW_WGSL: &str = include_str!("narrow.wgsl");
pub const CONCAT_WGSL: &str = include_str!("concat.wgsl");
pub const GATHER_WGSL: &str = include_str!("gather.wgsl");
pub const GATHER_SPLIT_WGSL: &str = include_str!("gather_split.wgsl");
pub const GATHER_AXIS_WGSL: &str = include_str!("gather_axis.wgsl");
pub const ATTENTION_WGSL: &str = include_str!("attention.wgsl");
pub const ATTENTION_BWD_WGSL: &str = include_str!("attention_bwd.wgsl");
pub const ROPE_WGSL: &str = include_str!("rope.wgsl");
pub const EXPAND_WGSL: &str = include_str!("expand.wgsl");
pub const ARGMAX_WGSL: &str = include_str!("argmax.wgsl");
pub const POOL2D_WGSL: &str = include_str!("pool2d.wgsl");
pub const CONV2D_WGSL: &str = include_str!("conv2d.wgsl");
pub const CONV1D_TILED_WGSL: &str = include_str!("conv1d_tiled.wgsl");
pub const IM2COL2D_WGSL: &str = include_str!("im2col2d.wgsl");
pub const POOL1D_WGSL: &str = include_str!("pool1d.wgsl");
pub const POOL3D_WGSL: &str = include_str!("pool3d.wgsl");
pub const CONV1D_WGSL: &str = include_str!("conv1d.wgsl");
pub const CONV3D_WGSL: &str = include_str!("conv3d.wgsl");
pub const CONV_TRANSPOSE3D_WGSL: &str = include_str!("conv_transpose3d.wgsl");
pub const GROUP_NORM_BWD_WGSL: &str = include_str!("group_norm_backward.wgsl");
pub const AXIAL_ROPE2D_WGSL: &str = include_str!("axial_rope2d.wgsl");
pub const FAKE_QUANTIZE_WGSL: &str = include_str!("fake_quantize.wgsl");
pub const SCATTER_ADD_WGSL: &str = include_str!("scatter_add.wgsl");
pub const TOPK_WGSL: &str = include_str!("topk.wgsl");
pub const WELCH_PEAKS_GPU_WGSL: &str = include_str!("welch_peaks_gpu.wgsl");
pub const UMAP_KNN_WGSL: &str = include_str!("umap_knn.wgsl");
pub const GROUPED_MATMUL_WGSL: &str = include_str!("grouped_matmul.wgsl");
pub const SAMPLE_WGSL: &str = include_str!("sample.wgsl");
pub const SELECTIVE_SCAN_WGSL: &str = include_str!("selective_scan.wgsl");
pub const GATED_DELTA_NET_WGSL: &str = include_str!("gated_delta_net.wgsl");
pub const MAMBA2_WGSL: &str = include_str!("mamba2.wgsl");
pub const GRU_WGSL: &str = include_str!("gru.wgsl");
pub const RNN_WGSL: &str = include_str!("rnn.wgsl");
pub const DEQUANT_MATMUL_WGSL: &str = include_str!("dequant_matmul.wgsl");
pub const DEQUANT_MATMUL_MLX_WGSL: &str = include_str!("dequant_matmul_mlx.wgsl");
pub const DEQUANT_GGUF_WGSL: &str = include_str!("dequant_gguf.wgsl");
pub const DEQUANT_GEMV_GGUF_WGSL: &str = include_str!("dequant_gemv_gguf.wgsl");
pub const DEQUANT_GEMM_Q1_0_WGSL: &str = include_str!("dequant_gemm_q1_0.wgsl");
pub const FUSED_RESIDUAL_LN_WGSL: &str = include_str!("fused_residual_ln.wgsl");
pub const FUSED_RESIDUAL_LN_TEE_WGSL: &str = include_str!("fused_residual_ln_tee.wgsl");
pub const FUSED_RESIDUAL_RMS_NORM_WGSL: &str = include_str!("fused_residual_rms_norm.wgsl");
pub const ADA_LAYER_NORM_WGSL: &str = include_str!("ada_layer_norm.wgsl");
pub const GATED_RESIDUAL_WGSL: &str = include_str!("gated_residual.wgsl");
pub const ADA_LAYER_NORM_BACKWARD_WGSL: &str = include_str!("ada_layer_norm_backward.wgsl");
pub const GATED_RESIDUAL_BACKWARD_WGSL: &str = include_str!("gated_residual_backward.wgsl");
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

/// Layout for fma (3-input fused multiply-add). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FmaParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for ReluBackward / ActivationBackward. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ActivationBackwardParams {
    pub n: u32,
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for reductions. 32 bytes.
///
/// Supports arbitrary-axis reductions. The reduce kernel walks the
/// input as a 3D tensor `[outer, reduce_dim, inner]` where:
///   * `outer` = product of dims BEFORE the reduce axis
///   * `reduce_dim` = the reduce axis itself
///   * `inner` = product of dims AFTER the reduce axis (=1 for the
///     last-axis case, which is what the v3 dispatcher emitted).
/// Output shape is `[outer, inner]` (or with the reduce axis kept as 1
/// when `keep_dim`; the dispatcher handles the shape arithmetic).
#[repr(C)]
pub struct ReduceParams {
    pub outer: u32,
    pub reduce_dim: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub op: u32,
    pub _p0: u32,
    pub _p1: u32,
}

// Manual impls to avoid issues with structural derives if any field
// arrangement subtly trips bytemuck.
unsafe impl Pod for ReduceParams {}
unsafe impl Zeroable for ReduceParams {}
impl Copy for ReduceParams {}
impl Clone for ReduceParams {
    fn clone(&self) -> Self {
        *self
    }
}
impl std::fmt::Debug for ReduceParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReduceParams {{ outer: {}, reduce_dim: {}, inner: {}, op: {} }}",
            self.outer, self.reduce_dim, self.inner, self.op
        )
    }
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

/// Layout for the fused dense softmax cross-entropy. 32 bytes.
/// Also used by integer-label `SoftmaxCrossEntropyWithLogits` (labels in
/// `targets_off`, one f32 per row).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceParams {
    pub outer: u32,
    pub inner: u32,
    pub logits_off: u32,
    pub targets_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for integer-label softmax cross-entropy backward. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub logits_off: u32,
    pub labels_off: u32,
    pub d_loss_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for MaxPool2d backward. 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaxPool2dBwdParams {
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
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub _p0: u32,
}

/// Layout for MaxPool3d backward. 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaxPool3dBwdParams {
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
    pub x_off: u32,
    pub dy_off: u32,
    pub dx_off: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Layout for Conv3d BackwardInput. 112 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dBwdInputParams {
    pub n: u32,
    pub c_in: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub c_out: u32,
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
    pub dy_off: u32,
    pub w_off: u32,
    pub dx_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for Conv3d BackwardWeight. 112 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Conv3dBwdWeightParams {
    pub n: u32,
    pub c_in: u32,
    pub d: u32,
    pub h: u32,
    pub w: u32,
    pub c_out: u32,
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
    pub x_off: u32,
    pub dy_off: u32,
    pub dw_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GroupNorm (NCHW) backward. 48 bytes. Shared by dx/dgamma/dbeta
/// entry points (`group_norm_bwd_{input,gamma,beta}`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroupNormBwdParams {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub num_groups: u32,
    pub eps_bits: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for AxialRope2d. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AxialRope2dParams {
    pub batch: u32,
    pub seq: u32,
    pub hidden: u32,
    pub end_x: u32,
    pub end_y: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub repeat_factor: u32,
    pub theta: f32,
    pub in_off: u32,
    pub out_off: u32,
    pub n_total: u32,
}

/// Layout for FakeQuantize (Fixed / PerBatch). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FakeQuantizeParams {
    pub n: u32,
    pub chan_dim: u32,
    pub inner: u32,
    pub q_max: f32,
    pub in_off: u32,
    pub scale_off: u32,
    pub out_off: u32,
    pub _pad: u32,
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

/// LayerNorm backward kernel params (f32 element offsets). Shared by
/// the three entry points; the dispatcher picks `layer_norm_bwd_input`,
/// `layer_norm_bwd_gamma_partial`, or `layer_norm_bwd_gamma_reduce`
/// based on which Step variant fired. dbeta isn't a dedicated op — it's
/// a plain `Reduce::Sum` over the batch dim of `dy`, handled by the
/// general reduce kernel.
///
/// `scratch_off` is the f32-element offset of the tail scratch zone
/// (only used by the gamma partial/reduce kernels). For the reduce
/// kernel `outer` carries the number of partial chunks emitted by the
/// partial kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayerNormBwdParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub gamma_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub scratch_off: u32,
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

/// Layout for cum_scan (cumprod / cummax). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CumScanParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub exclusive: u32,
    pub is_max: u32,
    pub _p0: u32,
    pub _p1: u32,
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
    pub norm_scale: f32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Uniform block for multi-kernel FFT (`fft_gpu.wgsl::Params`). 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftGpuParams {
    pub off: u32,
    pub dst_off: u32,
    pub n: u32,
    pub log2n: u32,
    pub inverse: u32,
    pub norm_scale: f32,
    pub outer: u32,
    pub tile: u32,
    pub inner_stages: u32,
    pub q_or_hs: u32,
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
    pub prologue: u32,
    pub out_n: u32,
    pub out_c: u32,
    pub out_h: u32,
    pub out_w: u32,
    pub prologue_input: u32,
    pub input_modulus: [u32; 16],
}

/// FKL batch region: `batch_input_offs[slice]` + shared chain (no prologue).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BatchElementwiseRegionParams {
    pub slice_len: u32,
    pub num_batch: u32,
    pub num_steps: u32,
    pub base_dst_off: u32,
    pub slice_elems: u32,
    pub batch_input_offs: [u32; 64],
    pub chain: [u32; 128],
    pub scalar_input_mask: u32,
    pub input_modulus: [u32; 16],
}

/// Layout for a numeric `Op::Cast` (`cast.wgsl`). 32 bytes. `mode`:
/// 0 identity, 1 float→int (trunc + saturate to `[lo_bits, hi_bits]`, NaN→0),
/// 2 →Bool (`value != 0`). `lo_bits`/`hi_bits` are f32-as-u32 clamp bounds.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CastParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub mode: u32,
    pub lo_bits: u32,
    pub hi_bits: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Standalone complex `Op::Cast` (`complex_cast.wgsl`). 32 bytes.
/// `mode` selects one of the six lane-move directions (real↔C64, real↔C128,
/// C64↔C128); `n` is the complex-element count; offsets are f32-element.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ComplexCastParams {
    pub n: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub mode: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// C64 element-wise binary (`binary_c64.wgsl`). 32 bytes. `n` is the output
/// complex-element count; `n_a`/`n_b` are the operands' complex-element counts
/// (broadcast via `k % n_x`); offsets are f32-element (lane `2*m + j`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BinaryC64Params {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub op: u32,
    pub n_a: u32,
    pub n_b: u32,
    pub _p0: u32,
}

/// C64 Wirtinger surface (`complex_wirtinger.wgsl`). 32 bytes.
/// Shared by ComplexNormSq / ComplexNormSqBackward / Conjugate:
/// - NormSq / Conjugate: `a_off`=src, `c_off`=dst (`b_off` unused)
/// - NormSqBackward: `a_off`=z, `b_off`=g, `c_off`=dz
/// `n` is the complex-element count; offsets are f32-element.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ComplexWirtingerParams {
    pub n: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
}

/// Ternary-pruned radix-2 butterfly (`fft_butterfly_stage.wgsl`). 48 bytes.
/// Offsets are f32-element; `half = n_fft / 2`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FftButterflyStageParams {
    pub batch: u32,
    pub n_fft: u32,
    pub stage: u32,
    pub half: u32,
    pub state_off: u32,
    pub out_off: u32,
    pub gate_off: u32,
    pub rev_off: u32,
    pub tw_re_off: u32,
    pub tw_im_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout shared by Reshape / same-dtype Cast / generic full copy. 32 bytes.
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
    /// GQA/MQA: number of key/value heads that the query heads share. Equals
    /// `heads` for plain MHA; 0 means unset and the shader falls back to MHA.
    pub kv_heads: u32,
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
    /// RoPE pairing flavor: `0` = NeoX rotate-half `(i, i+half)`, `1` = GPT-J /
    /// llama.cpp-NORM interleaved adjacent pairs `(2i, 2i+1)`. GGUF Llama weights
    /// are permuted for the GPT-J layout, so GGUF-backed decode needs `style=1`.
    pub style: u32,
    /// Partial rotary: half of `n_rot` (the rotated width). Dims `[n_rot,
    /// head_dim)` are copied through unchanged. Equals `half` for full rotation
    /// (n_rot == head_dim); smaller for p-RoPE (Gemma 4 global layers). The
    /// cos/sin row stride stays `half` (head_dim/2), matching the CPU reference.
    pub rot_half: u32,
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

/// Layout for GPU im2col (NCHW, N==1, groups==1). 80 bytes (20 u32).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Im2Col2dParams {
    pub c_in: u32,
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
    pub in_off: u32,
    pub col_off: u32,
    pub k_total: u32,
    pub spatial: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
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

/// Layout for dequant_gguf. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGgufParams {
    pub w_byte_off: u32,
    pub dst_f32_off: u32,
    pub scheme_id: u32,
    pub num_blocks: u32,
}

/// Layout for the fused GGUF K-quant GEMV (`dequant_gemv_gguf.wgsl`). 32 bytes.
/// Offsets are relative to each kernel binding's windowed base.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGemvGgufParams {
    pub k: u32,
    pub n: u32,
    pub scheme_id: u32,
    pub x_f32_off: u32,
    pub w_byte_off: u32,
    pub out_f32_off: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for fused Q1_0 GEMM (`dequant_gemm_q1_0.wgsl`). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantGemmQ10Params {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub x_f32_off: u32,
    pub w_byte_off: u32,
    pub out_f32_off: u32,
    pub _p0: u32,
    pub _p1: u32,
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

/// Layout for MLX DequantMatMul. 48 bytes (12 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DequantMatmulMlxParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub kind: u32,
    pub bits: u32,
    pub group_size: u32,
    pub x_byte_off: u32,
    pub w_byte_off: u32,
    pub scale_byte_off: u32,
    pub zp_byte_off: u32,
    pub out_byte_off: u32,
    pub _pad: u32,
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

/// Layout for FusedResidualRmsNorm (same bind layout as FusedResidualLN).
pub type FusedResidualRmsNormParams = FusedResidualLnParams;

/// Layout for AdaLayerNorm. 112 bytes (28 u32s).
/// `lead_pack` is 20 u32s (5×vec4 in WGSL); first 17 from IR are used.
/// Prefixed by 8 scalar u32s (32 bytes) so the vec4 array is naturally
/// 16-byte aligned — no implicit WGSL padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AdaLayerNormParams {
    pub outer: u32,
    pub inner: u32,
    pub in_off: u32,
    pub scale_off: u32,
    pub shift_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub layer_norm: u32,
    pub lead_pack: [u32; 20],
}

/// Layout for GatedResidual. 128 bytes (32 u32s).
/// Six scalar fields (24 B) + 8 B explicit pad so `lead_pack` (vec4 array)
/// starts at offset 32, matching WGSL uniform layout; trailing pad to 128.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedResidualParams {
    pub outer: u32,
    pub inner: u32,
    pub x_off: u32,
    pub y_off: u32,
    pub gate_off: u32,
    pub out_off: u32,
    pub _pre0: u32,
    pub _pre1: u32,
    pub lead_pack: [u32; 20],
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Expand the 17-u32 IR lead pack into the 20-u32 WGSL uniform slot.
#[inline]
pub fn lead_pack_uniform(src: [u32; 17]) -> [u32; 20] {
    let mut out = [0u32; 20];
    out[..17].copy_from_slice(&src);
    out
}

/// Layout for AdaLayerNormBackward. 48 bytes (12 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AdaLayerNormBackwardParams {
    pub mod_rows: u32,
    pub seq_per_mod: u32,
    pub inner: u32,
    pub x_off: u32,
    pub scale_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub eps_bits: u32,
    pub layer_norm: u32,
    pub _p0: u32,
    pub _p1: u32,
    pub _p2: u32,
}

/// Layout for GatedResidualBackward. 32 bytes (8 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedResidualBackwardParams {
    pub mod_rows: u32,
    pub seq_per_mod: u32,
    pub inner: u32,
    pub y_off: u32,
    pub gate_off: u32,
    pub dy_off: u32,
    pub out_off: u32,
    pub _p0: u32,
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

/// Layout for Mamba2 (SSD scan). 68→72 bytes padded to 16.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Mamba2Params {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    pub x_off: u32,
    pub dt_off: u32,
    pub a_off: u32,
    pub b_off: u32,
    pub c_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
}

/// Layout for GRU (native WGSL). 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GruParams {
    pub batch: u32,
    pub seq: u32,
    pub input_size: u32,
    pub hidden: u32,
    pub x_off: u32,
    pub wih_off: u32,
    pub whh_off: u32,
    pub bih_off: u32,
    pub bhh_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
}

/// Layout for Elman RNN (native WGSL). 68→padded. `relu` selects activation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RnnParams {
    pub batch: u32,
    pub seq: u32,
    pub input_size: u32,
    pub hidden: u32,
    pub x_off: u32,
    pub wih_off: u32,
    pub whh_off: u32,
    pub bias_off: u32,
    pub out_off: u32,
    pub seq_stride: u32,
    pub relu: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
    pub _p4: u32,
    pub _p5: u32,
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

/// Layout for GatedDeltaNet. 64 bytes (16 u32s).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GatedDeltaNetParams {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub state_size: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
    pub g_off: u32,
    pub beta_off: u32,
    pub state_off: u32,
    pub out_off: u32,
    pub use_carry: u32,
    /// PLAN L1 — full-extent seq stride for per-batch offset math.
    pub seq_stride: u32,
    pub _p1: u32,
    pub _p2: u32,
    pub _p3: u32,
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

/// Native GPU WelchPeaks dispatch parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WelchPeaksGpuParams {
    pub spec_off: u32,
    pub dst_off: u32,
    pub welch_batch: u32,
    pub n_fft: u32,
    pub n_segments: u32,
    pub k: u32,
    pub n_bins: u32,
    pub _p0: u32,
    pub _p1: u32,
}

/// Layout for UMAP k-NN on a pairwise `[n, n]` matrix. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UmapKnnParams {
    pub n: u32,
    pub k: u32,
    pub pw_off: u32,
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

impl Kernel {
    pub fn bind_two(
        &self,
        device: &wgpu::Device,
        arena: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu fft gpu bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    }
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

/// 4-binding layout: storage(ro) + uniform + storage(ro) + storage(rw).
/// For the GGUF GEMV: x (ro arena window) + params + weight (ro arena window) +
/// out (rw separate buffer). The arena is bound read-only twice (allowed), and
/// the single read-write binding is a distinct buffer — sidestepping wgpu's
/// "STORAGE_READ_WRITE is exclusive" rule for same-buffer aliasing.
fn build_kernel_ro_u_ro_rw(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(true),
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
                ty: storage(true),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage(false),
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

/// f16 shadow (rw) + uniform + f32 arena (rw) — `cast_f32_to_f16` only.
/// Separate from `build_kernel_3`: cast reads f32 written by a prior unary in
/// the same arena; other 3-binding kernels keep binding 2 read-only.
fn build_kernel_cast_f32_to_f16(
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

/// f32 arena (rw) + uniform + f16 shadow (rw) — unary with CoopF16Vk mirror.
fn build_kernel_f32_rw_uniform_f16_rw(
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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

/// f16 shadow (read) + f32 arena (rw) + uniform — Vulkan/DX12 coop f16 matmul.
fn build_kernel_coop_f16_vk(
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

fn try_build_kernel_coop_f16_vk(
    device: &wgpu::Device,
    label: &'static str,
    wgsl: &str,
    entry_point: &'static str,
) -> Option<Kernel> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build_kernel_coop_f16_vk(device, label, wgsl, entry_point)
    }))
    .ok()
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
static MATMUL_WIDE_NV: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16W: OnceLock<Kernel> = OnceLock::new();
static MATMUL_F16_COMPUTE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP16: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F32: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F32_PORTABLE: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_WIDEN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static CAST_F32_TO_F16: OnceLock<Kernel> = OnceLock::new();
static BINARY: OnceLock<Kernel> = OnceLock::new();
static BINARY_C64: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_CAST: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_NORM_SQ: OnceLock<Kernel> = OnceLock::new();
static COMPLEX_NORM_SQ_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static CONJUGATE_C64: OnceLock<Kernel> = OnceLock::new();
static FFT_BUTTERFLY_STAGE: OnceLock<Kernel> = OnceLock::new();
static UNARY: OnceLock<Kernel> = OnceLock::new();
static UNARY_F16_MIRROR: OnceLock<Kernel> = OnceLock::new();
static COMPARE: OnceLock<Kernel> = OnceLock::new();
static WHEREK: OnceLock<Kernel> = OnceLock::new();
static FMAK: OnceLock<Kernel> = OnceLock::new();
static ACTIVATION_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static REDUCE: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY_WITH_LOGITS: OnceLock<Kernel> = OnceLock::new();
static SOFTMAX_CROSS_ENTROPY_BWD: OnceLock<Kernel> = OnceLock::new();
static MAXPOOL2D_BWD: OnceLock<Kernel> = OnceLock::new();
static MAXPOOL3D_BWD: OnceLock<Kernel> = OnceLock::new();
static CONV3D_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static CONV3D_BWD_WEIGHT: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_GAMMA: OnceLock<Kernel> = OnceLock::new();
static GROUP_NORM_BWD_BETA: OnceLock<Kernel> = OnceLock::new();
static AXIAL_ROPE2D: OnceLock<Kernel> = OnceLock::new();
static FAKE_QUANTIZE_FIXED: OnceLock<Kernel> = OnceLock::new();
static FAKE_QUANTIZE_PERBATCH: OnceLock<Kernel> = OnceLock::new();
static LAYERNORM: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD: OnceLock<Kernel> = OnceLock::new();
static RMS_NORM_BWD_PARAM: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_INPUT: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_GAMMA: OnceLock<Kernel> = OnceLock::new();
static LAYER_NORM_BWD_GAMMA_REDUCE: OnceLock<Kernel> = OnceLock::new();
static CUMSUM_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE_BWD: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ZERO: OnceLock<Kernel> = OnceLock::new();
static GATHER_BWD_ACC: OnceLock<Kernel> = OnceLock::new();
static CUMSUM: OnceLock<Kernel> = OnceLock::new();
static CUM_SCAN: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_RADIX2: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_RADIX2_BIG: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R2: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R4: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_BIG_R8: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_R4_16K: OnceLock<Kernel> = OnceLock::new();
#[cfg(feature = "native-gpu-fft")]
static FFT_GPU_MULTIROW: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_BITREV: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_INNER: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_OUTER_R4: OnceLock<Kernel> = OnceLock::new();
static FFT_GPU_OUTER_R2: OnceLock<Kernel> = OnceLock::new();
static COPY: OnceLock<Kernel> = OnceLock::new();
static CAST: OnceLock<Kernel> = OnceLock::new();
static ELEMENTWISE_REGION: OnceLock<Kernel> = OnceLock::new();
static ELEMENTWISE_REGION_SPATIAL: OnceLock<Kernel> = OnceLock::new();
static TRANSPOSE: OnceLock<Kernel> = OnceLock::new();
static NARROW: OnceLock<Kernel> = OnceLock::new();
static CONCAT: OnceLock<Kernel> = OnceLock::new();
static GATHER: OnceLock<Kernel> = OnceLock::new();
static GATHER_SPLIT: OnceLock<Kernel> = OnceLock::new();
static GATHER_AXIS: OnceLock<Kernel> = OnceLock::new();
static ATTENTION: OnceLock<Kernel> = OnceLock::new();
static ATTENTION_BWD: OnceLock<Kernel> = OnceLock::new();
static ROPE: OnceLock<Kernel> = OnceLock::new();
static EXPAND: OnceLock<Kernel> = OnceLock::new();
static ARGMAX: OnceLock<Kernel> = OnceLock::new();
static POOL2D: OnceLock<Kernel> = OnceLock::new();
static CONV2D: OnceLock<Kernel> = OnceLock::new();
static CONV1D_TILED: OnceLock<Kernel> = OnceLock::new();
static IM2COL2D: OnceLock<Kernel> = OnceLock::new();
static POOL1D: OnceLock<Kernel> = OnceLock::new();
static POOL3D: OnceLock<Kernel> = OnceLock::new();
static CONV1D: OnceLock<Kernel> = OnceLock::new();
static CONV3D: OnceLock<Kernel> = OnceLock::new();
static CONV_TRANSPOSE3D: OnceLock<Kernel> = OnceLock::new();
static SCATTER_ADD: OnceLock<Kernel> = OnceLock::new();
static TOPK: OnceLock<Kernel> = OnceLock::new();
static WELCH_PEAKS_GPU: OnceLock<Kernel> = OnceLock::new();
static UMAP_KNN: OnceLock<Kernel> = OnceLock::new();
static GROUPED_MATMUL: OnceLock<Kernel> = OnceLock::new();
static SAMPLE: OnceLock<Kernel> = OnceLock::new();
static SELECTIVE_SCAN: OnceLock<Kernel> = OnceLock::new();
static GATED_DELTA_NET: OnceLock<Kernel> = OnceLock::new();
static MAMBA2: OnceLock<Kernel> = OnceLock::new();
static GRU: OnceLock<Kernel> = OnceLock::new();
static RNN: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_MATMUL: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_MATMUL_MLX: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GGUF: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMV_GGUF: OnceLock<Kernel> = OnceLock::new();
static DEQUANT_GEMM_Q1_0: OnceLock<Kernel> = OnceLock::new();
static MATMUL_BT: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_LN_TEE: OnceLock<Kernel> = OnceLock::new();
static FUSED_RESIDUAL_RMS_NORM: OnceLock<Kernel> = OnceLock::new();
static ADA_LAYER_NORM: OnceLock<Kernel> = OnceLock::new();
static GATED_RESIDUAL: OnceLock<Kernel> = OnceLock::new();
static ADA_LAYER_NORM_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static GATED_RESIDUAL_BACKWARD: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F32: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_WIDEN: OnceLock<Kernel> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();
static MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC: OnceLock<Option<Kernel>> = OnceLock::new();

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
/// 64×64 / 256-thread variant for discrete GPUs (Vulkan path).
pub fn matmul_wide_nv_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_WIDE_NV.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_wide_nv",
            MATMUL_WIDE_NV_WGSL,
            "matmul_wide_nv",
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
/// Vulkan/DX12-oriented coop f32 matmul (`coopLoad`, 8×8 workgroups).
pub fn matmul_coop_f32_portable_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    let feats = device.features();
    if !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        || !crate::device::coop_f32_8x8_supported()
    {
        return None;
    }
    Some(MATMUL_COOP_F32_PORTABLE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu matmul_coop_f32_portable",
            MATMUL_COOP_F32_PORTABLE_WGSL,
            "matmul_coop_f32_portable",
        )
    }))
}
fn coop_f16_vk_device_ready(device: &wgpu::Device) -> bool {
    // Cooperative-matrix Vulkan/DX12 matmul is OFF by default — see
    // `coop_f16_vk_eligible` in `backend.rs` for the rationale. Opt in
    // with `RLX_WGPU_COOP_F16_VK_ENABLE=1`. Legacy
    // `RLX_WGPU_COOP_F16_VK_DISABLE=1` also fully disables.
    if rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_DISABLE")
        || !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_ENABLE")
    {
        return false;
    }
    device.features().contains(wgpu::Features::SHADER_F16)
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && crate::device::coop_f16_16x16_supported()
        && crate::device::coop_discrete_backend()
}

fn coop_f16_vk_f32acc_device_ready(device: &wgpu::Device) -> bool {
    coop_f16_vk_device_ready(device) && crate::device::coop_f16_16x16_f32_acc_supported()
}

pub fn matmul_coop_f16_vulkan_f32acc_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_COOP_F16_VULKAN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_coop_f16_vulkan_f32acc",
                MATMUL_COOP_F16_VULKAN_F32ACC_WGSL,
                "matmul_coop_f16_vulkan_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_coop_f16_vulkan_widen_f32acc_kernel(
    device: &wgpu::Device,
) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_coop_f16_vulkan_widen_f32acc",
                MATMUL_COOP_F16_VULKAN_WIDEN_F32ACC_WGSL,
                "matmul_coop_f16_vulkan_widen_f32acc",
            )
        })
        .as_ref()
}

fn coop_f16_vk_use_f32acc(device: &wgpu::Device) -> bool {
    !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_NO_F32ACC")
        && matmul_coop_f16_vulkan_f32acc_kernel(device).is_some()
}

fn pick_coop_f16_vk_matmul(
    device: &wgpu::Device,
    n: u32,
    loadt: fn(&wgpu::Device) -> Option<&'static Kernel>,
    loadt_f32acc: fn(&wgpu::Device) -> Option<&'static Kernel>,
    widen: fn(&wgpu::Device) -> Option<&'static Kernel>,
    widen_f32acc: fn(&wgpu::Device) -> Option<&'static Kernel>,
) -> Option<&'static Kernel> {
    if coop_f16_vk_use_f32acc(device) {
        if coop_f16_vk_widen_b_load(n) {
            return widen_f32acc(device).or_else(|| loadt_f32acc(device));
        }
        return loadt_f32acc(device);
    }
    if coop_f16_vk_widen_b_load(n) {
        widen(device).or_else(|| loadt(device))
    } else {
        loadt(device)
    }
}

/// Matmul CoopF16Vk kernel for column count `n`.
pub fn matmul_coop_f16_vulkan_active_kernel(
    device: &wgpu::Device,
    n: u32,
) -> Option<&'static Kernel> {
    pick_coop_f16_vk_matmul(
        device,
        n,
        matmul_coop_f16_vulkan_kernel,
        matmul_coop_f16_vulkan_f32acc_kernel,
        matmul_coop_f16_vulkan_widen_kernel,
        matmul_coop_f16_vulkan_widen_f32acc_kernel,
    )
}

pub fn matmul_coop_f16_vulkan_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_COOP_F16_VULKAN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_coop_f16_vulkan",
            MATMUL_COOP_F16_VULKAN_WGSL,
            "matmul_coop_f16_vulkan",
        )
    }))
}
/// N above which coop may use the row-major B-load variant (`RLX_WGPU_COOP_F16_VK_LARGE_N`).
pub const COOP_F16_VK_WIDEN_N: u32 = 768;

/// Use `coopLoad` on B instead of `coopLoadT` when N > 768 and `RLX_WGPU_COOP_F16_VK_LOAD_T` is unset.
pub fn coop_f16_vk_widen_b_load(n: u32) -> bool {
    n > COOP_F16_VK_WIDEN_N && !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_LOAD_T")
}

pub fn matmul_coop_f16_vulkan_widen_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_COOP_F16_VULKAN_WIDEN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_coop_f16_vulkan_widen",
            MATMUL_COOP_F16_VULKAN_WIDEN_WGSL,
            "matmul_coop_f16_vulkan_widen",
        )
    }))
}
pub fn coop_f16_vk_f32acc_available(device: &wgpu::Device) -> bool {
    matmul_coop_f16_vulkan_f32acc_kernel(device).is_some()
}
/// CoopF32 kernel for the active wgpu backend (Metal simdgroup vs Vulkan/DX12 portable).
pub fn matmul_coop_f32_active_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    match crate::device::wgpu_device().map(|d| d.backend) {
        Some(wgpu::Backend::Metal) => matmul_coop_f32_kernel(device),
        Some(wgpu::Backend::Vulkan) | Some(wgpu::Backend::Dx12) => {
            matmul_coop_f32_portable_kernel(device)
        }
        _ => None,
    }
}
/// Wide f32 matmul kernel for the active backend.
pub fn matmul_wide_active_kernel(device: &wgpu::Device) -> &'static Kernel {
    match crate::device::wgpu_device().map(|d| d.backend) {
        Some(wgpu::Backend::Vulkan) | Some(wgpu::Backend::Dx12) => matmul_wide_nv_kernel(device),
        _ => matmul_wide_kernel(device),
    }
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
        build_kernel_cast_f32_to_f16(
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
pub fn binary_c64_kernel(device: &wgpu::Device) -> &'static Kernel {
    BINARY_C64.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu binary_c64",
            BINARY_C64_WGSL,
            "binary_c64_main",
        )
    })
}
pub fn complex_cast_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_CAST.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_cast",
            COMPLEX_CAST_WGSL,
            "complex_cast_main",
        )
    })
}
pub fn complex_norm_sq_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_NORM_SQ.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_norm_sq",
            COMPLEX_WIRINGER_WGSL,
            "complex_norm_sq",
        )
    })
}
pub fn complex_norm_sq_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPLEX_NORM_SQ_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu complex_norm_sq_backward",
            COMPLEX_WIRINGER_WGSL,
            "complex_norm_sq_backward",
        )
    })
}
pub fn conjugate_c64_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONJUGATE_C64.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conjugate_c64",
            COMPLEX_WIRINGER_WGSL,
            "conjugate_c64",
        )
    })
}
pub fn fft_butterfly_stage_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_BUTTERFLY_STAGE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_butterfly_stage",
            FFT_BUTTERFLY_STAGE_WGSL,
            "fft_butterfly_stage",
        )
    })
}
pub fn unary_kernel(device: &wgpu::Device) -> &'static Kernel {
    UNARY.get_or_init(|| build_kernel(device, "rlx-wgpu unary", UNARY_WGSL, "unary"))
}
pub fn unary_f16_mirror_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    Some(UNARY_F16_MIRROR.get_or_init(|| {
        build_kernel_f32_rw_uniform_f16_rw(
            device,
            "rlx-wgpu unary_f16_mirror",
            UNARY_F16_MIRROR_WGSL,
            "unary_f16_mirror",
        )
    }))
}
pub fn compare_kernel(device: &wgpu::Device) -> &'static Kernel {
    COMPARE.get_or_init(|| build_kernel(device, "rlx-wgpu compare", COMPARE_WGSL, "compare"))
}
pub fn where_kernel(device: &wgpu::Device) -> &'static Kernel {
    WHEREK.get_or_init(|| build_kernel(device, "rlx-wgpu where", WHERE_WGSL, "where_select"))
}
pub fn fma_kernel(device: &wgpu::Device) -> &'static Kernel {
    FMAK.get_or_init(|| build_kernel(device, "rlx-wgpu fma", FMA_WGSL, "fma_main"))
}
pub fn activation_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ACTIVATION_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu activation_backward",
            ACTIVATION_BACKWARD_WGSL,
            "activation_backward",
        )
    })
}
pub fn reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    REDUCE.get_or_init(|| build_kernel(device, "rlx-wgpu reduce", REDUCE_WGSL, "reduce"))
}
pub fn softmax_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX.get_or_init(|| build_kernel(device, "rlx-wgpu softmax", SOFTMAX_WGSL, "softmax"))
}
pub fn softmax_cross_entropy_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy",
            SOFTMAX_CROSS_ENTROPY_WGSL,
            "softmax_cross_entropy",
        )
    })
}
pub fn softmax_cross_entropy_with_logits_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy_with_logits",
            SOFTMAX_CROSS_ENTROPY_WGSL,
            "softmax_cross_entropy_with_logits",
        )
    })
}
pub fn softmax_cross_entropy_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    SOFTMAX_CROSS_ENTROPY_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu softmax_cross_entropy_backward",
            SOFTMAX_CROSS_ENTROPY_BWD_WGSL,
            "softmax_cross_entropy_backward",
        )
    })
}
pub fn maxpool2d_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAXPOOL2D_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu maxpool2d_backward",
            MAXPOOL2D_BWD_WGSL,
            "maxpool2d_backward",
        )
    })
}
pub fn maxpool3d_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAXPOOL3D_BWD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu maxpool3d_backward",
            MAXPOOL3D_BWD_WGSL,
            "maxpool3d_backward",
        )
    })
}
pub fn conv3d_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv3d_backward_input",
            CONV3D_BWD_INPUT_WGSL,
            "conv3d_backward_input",
        )
    })
}
pub fn conv3d_backward_weight_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV3D_BWD_WEIGHT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv3d_backward_weight",
            CONV3D_BWD_WEIGHT_WGSL,
            "conv3d_backward_weight",
        )
    })
}
pub fn group_norm_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_input",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_input",
        )
    })
}
pub fn group_norm_backward_gamma_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_GAMMA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_gamma",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_gamma",
        )
    })
}
pub fn group_norm_backward_beta_kernel(device: &wgpu::Device) -> &'static Kernel {
    GROUP_NORM_BWD_BETA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu group_norm_bwd_beta",
            GROUP_NORM_BWD_WGSL,
            "group_norm_bwd_beta",
        )
    })
}
pub fn axial_rope2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    AXIAL_ROPE2D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu axial_rope2d",
            AXIAL_ROPE2D_WGSL,
            "axial_rope2d",
        )
    })
}
pub fn fake_quantize_fixed_kernel(device: &wgpu::Device) -> &'static Kernel {
    FAKE_QUANTIZE_FIXED.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fake_quantize_fixed",
            FAKE_QUANTIZE_WGSL,
            "fake_quantize_fixed",
        )
    })
}
pub fn fake_quantize_perbatch_kernel(device: &wgpu::Device) -> &'static Kernel {
    FAKE_QUANTIZE_PERBATCH.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fake_quantize_perbatch",
            FAKE_QUANTIZE_WGSL,
            "fake_quantize_perbatch",
        )
    })
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
pub fn layer_norm_backward_input_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_INPUT.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_input",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_input",
        )
    })
}
pub fn layer_norm_backward_gamma_partial_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_GAMMA.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_gamma_partial",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_gamma_partial",
        )
    })
}

pub fn layer_norm_backward_gamma_reduce_kernel(device: &wgpu::Device) -> &'static Kernel {
    LAYER_NORM_BWD_GAMMA_REDUCE.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu layer_norm_bwd_gamma_reduce",
            LAYER_NORM_BWD_WGSL,
            "layer_norm_bwd_gamma_reduce",
        )
    })
}
pub fn cumsum_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM_BWD
        .get_or_init(|| build_kernel(device, "rlx-wgpu cumsum_bwd", CUMSUM_BWD_WGSL, "cumsum_bwd"))
}
pub fn rope_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ROPE_BWD.get_or_init(|| build_kernel(device, "rlx-wgpu rope_bwd", ROPE_BWD_WGSL, "rope_bwd"))
}
pub fn gather_backward_zero_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ZERO.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_bwd_zero",
            GATHER_BWD_WGSL,
            "gather_bwd_zero",
        )
    })
}
pub fn gather_backward_acc_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_BWD_ACC.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gather_bwd_acc",
            GATHER_BWD_WGSL,
            "gather_bwd_acc",
        )
    })
}
pub fn cumsum_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUMSUM.get_or_init(|| build_kernel(device, "rlx-wgpu cumsum", CUMSUM_WGSL, "cumsum"))
}
pub fn cum_scan_kernel(device: &wgpu::Device) -> &'static Kernel {
    CUM_SCAN.get_or_init(|| build_kernel(device, "rlx-wgpu cum_scan", CUM_SCAN_WGSL, "cum_scan"))
}
pub fn fft_gpu_radix2_full_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_RADIX2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_full",
            FFT_GPU_WGSL,
            "fft_radix2_full",
        )
    })
}
/// native-gpu-fft: single-kernel on-chip FFT for n in (1024, 2048] (16 KB).
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_radix2_full_big_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_RADIX2_BIG.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_full_big",
            FFT_GPU_WGSL,
            "fft_radix2_full_big",
        )
    })
}
/// native-gpu-fft: 32 KB on-chip kernels (n<=4096). Only call when the device
/// reports >=32 KB workgroup storage — pipeline creation otherwise exceeds the
/// limit.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r2_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix2_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix2_big",
        )
    })
}
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r4_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix4_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix4_big",
        )
    })
}
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_big_r8_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BIG_R8.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix8_big",
            FFT_GPU_BIG_WGSL,
            "fft_radix8_big",
        )
    })
}
/// native-gpu-fft: portable 16 KB radix-4 (n<=2048); no device-limit gate.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_r4_16k_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_R4_16K.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_radix4_16k",
            FFT_GPU_R4_16K_WGSL,
            "fft_radix4_16k",
        )
    })
}
/// native-gpu-fft: multi-row small-n FFT (16 KB); no device-limit gate.
#[cfg(feature = "native-gpu-fft")]
pub fn fft_gpu_multirow_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_MULTIROW.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_multirow",
            FFT_GPU_MULTIROW_WGSL,
            "fft_multirow",
        )
    })
}
pub fn fft_gpu_bit_reverse_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_BITREV.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_bit_reverse",
            FFT_GPU_WGSL,
            "fft_bit_reverse",
        )
    })
}
pub fn fft_gpu_inner_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_INNER
        .get_or_init(|| build_kernel(device, "rlx-wgpu fft_inner", FFT_GPU_WGSL, "fft_inner"))
}
pub fn fft_gpu_outer_r4_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_OUTER_R4.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_outer_r4",
            FFT_GPU_WGSL,
            "fft_outer_r4",
        )
    })
}
pub fn fft_gpu_outer_r2_kernel(device: &wgpu::Device) -> &'static Kernel {
    FFT_GPU_OUTER_R2.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fft_outer_r2",
            FFT_GPU_WGSL,
            "fft_outer_r2",
        )
    })
}
pub fn copy_kernel(device: &wgpu::Device) -> &'static Kernel {
    COPY.get_or_init(|| build_kernel(device, "rlx-wgpu copy", COPY_WGSL, "copy"))
}
pub fn cast_kernel(device: &wgpu::Device) -> &'static Kernel {
    CAST.get_or_init(|| build_kernel(device, "rlx-wgpu cast", CAST_WGSL, "cast_main"))
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

pub fn elementwise_region_spatial_kernel(device: &wgpu::Device) -> &'static Kernel {
    ELEMENTWISE_REGION_SPATIAL.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu elementwise_region_spatial",
            ELEMENTWISE_REGION_WGSL,
            "elementwise_region_spatial",
        )
    })
}

static BATCH_ELEMENTWISE_REGION: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();

pub fn batch_elementwise_region_kernel(device: &wgpu::Device) -> &'static Kernel {
    BATCH_ELEMENTWISE_REGION.get_or_init(|| {
        build_kernel_region(
            device,
            "rlx-wgpu batch_elementwise_region",
            ELEMENTWISE_REGION_WGSL,
            "batch_elementwise_region",
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
/// Split-binding gather: table (ro) + uniform + idx (ro) + out (rw, separate
/// buffer). For >4 GiB arenas where the embedding output lies outside the
/// table's bind window. See [`build_kernel_ro_u_ro_rw`].
pub fn gather_split_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATHER_SPLIT.get_or_init(|| {
        build_kernel_ro_u_ro_rw(device, "rlx-wgpu gather_split", GATHER_SPLIT_WGSL, "gather")
    })
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
pub fn im2col2d_kernel(device: &wgpu::Device) -> &'static Kernel {
    IM2COL2D.get_or_init(|| build_kernel(device, "rlx-wgpu im2col2d", IM2COL2D_WGSL, "im2col2d"))
}
pub fn conv1d_tiled_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV1D_TILED.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv1d_tiled",
            CONV1D_TILED_WGSL,
            "conv1d_tiled",
        )
    })
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
pub fn conv_transpose3d_kernel(device: &wgpu::Device) -> &'static Kernel {
    CONV_TRANSPOSE3D.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu conv_transpose3d",
            CONV_TRANSPOSE3D_WGSL,
            "conv_transpose3d",
        )
    })
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
pub fn welch_peaks_gpu_kernel(device: &wgpu::Device) -> &'static Kernel {
    WELCH_PEAKS_GPU.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu welch_peaks_gpu",
            WELCH_PEAKS_GPU_WGSL,
            "welch_peaks_gpu",
        )
    })
}
pub fn umap_knn_kernel(device: &wgpu::Device) -> &'static Kernel {
    UMAP_KNN.get_or_init(|| build_kernel(device, "rlx-wgpu umap_knn", UMAP_KNN_WGSL, "umap_knn"))
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
pub fn gated_delta_net_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_DELTA_NET.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_delta_net",
            GATED_DELTA_NET_WGSL,
            "gated_delta_net",
        )
    })
}
pub fn mamba2_kernel(device: &wgpu::Device) -> &'static Kernel {
    MAMBA2.get_or_init(|| build_kernel(device, "rlx-wgpu mamba2", MAMBA2_WGSL, "mamba2"))
}
pub fn gru_kernel(device: &wgpu::Device) -> &'static Kernel {
    GRU.get_or_init(|| build_kernel(device, "rlx-wgpu gru", GRU_WGSL, "gru"))
}
pub fn rnn_kernel(device: &wgpu::Device) -> &'static Kernel {
    RNN.get_or_init(|| build_kernel(device, "rlx-wgpu rnn", RNN_WGSL, "rnn"))
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
pub fn dequant_matmul_mlx_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_MATMUL_MLX.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu dequant_matmul_mlx",
            DEQUANT_MATMUL_MLX_WGSL,
            "dequant_matmul_mlx",
        )
    })
}
pub fn dequant_gguf_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GGUF.get_or_init(|| {
        build_kernel_3(
            device,
            "rlx-wgpu dequant_gguf",
            DEQUANT_GGUF_WGSL,
            "dequant_gguf",
        )
    })
}
pub fn matmul_bt_kernel(device: &wgpu::Device) -> &'static Kernel {
    MATMUL_BT.get_or_init(|| build_kernel(device, "rlx-wgpu matmul_bt", MATMUL_WGSL, "matmul_bt"))
}
pub fn dequant_gemv_gguf_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMV_GGUF.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemv_gguf",
            DEQUANT_GEMV_GGUF_WGSL,
            "dequant_gemv",
        )
    })
}
pub fn dequant_gemm_q1_0_kernel(device: &wgpu::Device) -> &'static Kernel {
    DEQUANT_GEMM_Q1_0.get_or_init(|| {
        build_kernel_ro_u_ro_rw(
            device,
            "rlx-wgpu dequant_gemm_q1_0",
            DEQUANT_GEMM_Q1_0_WGSL,
            "dequant_gemm_q1_0",
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
pub fn fused_residual_rms_norm_kernel(device: &wgpu::Device) -> &'static Kernel {
    FUSED_RESIDUAL_RMS_NORM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu fused_residual_rms_norm",
            FUSED_RESIDUAL_RMS_NORM_WGSL,
            "fused_residual_rms_norm",
        )
    })
}
pub fn ada_layer_norm_kernel(device: &wgpu::Device) -> &'static Kernel {
    ADA_LAYER_NORM.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu ada_layer_norm",
            ADA_LAYER_NORM_WGSL,
            "ada_layer_norm",
        )
    })
}
pub fn gated_residual_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_RESIDUAL.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_residual",
            GATED_RESIDUAL_WGSL,
            "gated_residual",
        )
    })
}
pub fn ada_layer_norm_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    ADA_LAYER_NORM_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu ada_layer_norm_backward",
            ADA_LAYER_NORM_BACKWARD_WGSL,
            "ada_layer_norm_backward",
        )
    })
}
pub fn gated_residual_backward_kernel(device: &wgpu::Device) -> &'static Kernel {
    GATED_RESIDUAL_BACKWARD.get_or_init(|| {
        build_kernel(
            device,
            "rlx-wgpu gated_residual_backward",
            GATED_RESIDUAL_BACKWARD_WGSL,
            "gated_residual_backward",
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
pub fn matmul_qkv_coop_f16_vk_f32acc_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_QKV_COOP_F16_VK_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_qkv_coop_f16_vk_f32acc",
                MATMUL_QKV_COOP_F16_VK_F32ACC_WGSL,
                "matmul_qkv_coop_f16_vk_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_qkv_coop_f16_vk_widen_f32acc_kernel(
    device: &wgpu::Device,
) -> Option<&'static Kernel> {
    if !coop_f16_vk_f32acc_device_ready(device) {
        return None;
    }
    MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC
        .get_or_init(|| {
            try_build_kernel_coop_f16_vk(
                device,
                "rlx-wgpu matmul_qkv_coop_f16_vk_widen_f32acc",
                MATMUL_QKV_COOP_F16_VK_WIDEN_F32ACC_WGSL,
                "matmul_qkv_coop_f16_vk_widen_f32acc",
            )
        })
        .as_ref()
}

pub fn matmul_qkv_coop_f16_vk_active_kernel(
    device: &wgpu::Device,
    n: u32,
) -> Option<&'static Kernel> {
    pick_coop_f16_vk_matmul(
        device,
        n,
        matmul_qkv_coop_f16_vk_kernel,
        matmul_qkv_coop_f16_vk_f32acc_kernel,
        matmul_qkv_coop_f16_vk_widen_kernel,
        matmul_qkv_coop_f16_vk_widen_f32acc_kernel,
    )
}

pub fn matmul_qkv_coop_f16_vk_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_QKV_COOP_F16_VK.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_qkv_coop_f16_vk",
            MATMUL_QKV_COOP_F16_VK_WGSL,
            "matmul_qkv_coop_f16_vk",
        )
    }))
}
pub fn matmul_qkv_coop_f16_vk_widen_kernel(device: &wgpu::Device) -> Option<&'static Kernel> {
    if !coop_f16_vk_device_ready(device) {
        return None;
    }
    Some(MATMUL_QKV_COOP_F16_VK_WIDEN.get_or_init(|| {
        build_kernel_coop_f16_vk(
            device,
            "rlx-wgpu matmul_qkv_coop_f16_vk_widen",
            MATMUL_QKV_COOP_F16_VK_WIDEN_WGSL,
            "matmul_qkv_coop_f16_vk_widen",
        )
    }))
}
