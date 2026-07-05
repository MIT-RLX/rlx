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

//! `CudaExecutable` — lowers an rlx-ir Graph into a sequence of CUDA
//! kernel launches against a pre-allocated device buffer.
//!
//! v2 op coverage: MatMul (tiled SGEMM), Binary, Compare, Activation, Where,
//! Reduce, Softmax, LayerNorm, RmsNorm, FusedResidualLN, Gather, Narrow,
//! Argmax, Reshape/Cast (no-op via slot aliasing), leaf nodes. Anything
//! else panics at compile time with a "fall back to CPU/Metal/MLX/WGPU"
//! diagnostic. Op coverage is grown incrementally — each new op is one
//! `.cu` source + one Step variant + one match arm.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_opt::rlx_fusion::pass::Pass as _;

use crate::arena::{Arena, plan_f32_uniform};
use crate::device::{
    CUBLASLT_WORKSPACE_BYTES, CUDNN_WORKSPACE_BYTES, cuda_blas, cuda_blas_lt_handle,
    cuda_blas_lt_workspace, cuda_context, cuda_dnn_handle, cuda_dnn_workspace,
};
use crate::host_staging::F32HostSlot;
use crate::kernels::{
    argmax_kernel, attention_bwd_kernel, attention_kernel, attention_row_kernel,
    batch_elementwise_region_kernel, binary_kernel, compare_kernel, concat_kernel,
    conv_transpose2d_kernel, conv1d_kernel, conv2d_kernel, conv3d_kernel, copy_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, dispatch_grid_1d,
    dispatch_grid_prologue_nchw, elementwise_region_kernel, expand_kernel, fused_attn_kernel,
    fused_binary_unary_kernel, fused_residual_ln_kernel, fused_residual_rms_norm_kernel,
    gather_axis_kernel, gather_backward_kernel, gather_kernel, group_norm_kernel,
    grouped_matmul_kernel, im2col_kernel, layer_norm2d_kernel, layernorm_kernel,
    matmul_epilogue_kernel, matmul_kernel, matmul_wmma_kernel, maxpool2d_backward_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};

/// Opt-in WMMA Tensor Core matmul. Reads `RLX_CUDA_WMMA=1` from env at
/// process start (cached behind a `OnceLock`). When true and cuBLAS is
/// unavailable, the scalar matmul kernel is replaced by the WMMA kernel
/// for plain (non-fused) matmul. Tensor Cores require SM 70+; on older
/// hardware NVRTC's `load_module` will fail and we fall back to scalar.
fn use_wmma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_WMMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Strict f32 matmul for encoder parity: tiled `matmul.cu` kernel (same
/// family as wgpu), not cuBLASLt / cuBLAS heuristics.
fn matmul_parity_mode() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::flag("RLX_CUDA_NO_TF32")
            || rlx_ir::env::flag("RLX_CUDA_PARITY")
            || rlx_ir::env::flag("RLX_CUDA_NO_CUBLASLT")
    })
}

/// One launch step in the compiled schedule.
#[derive(Clone)]
enum Step {
    Matmul {
        m: u32,
        k: u32,
        n: u32,
        a_off_f32: u32,
        b_off_f32: u32,
        c_off_f32: u32,
        batch: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
        c_batch_stride: u32,
        has_bias: u32,
        bias_off_f32: u32,
        act_id: u32,
    },
    /// Native FP8 tensor-core GEMM (cublasLt). TN: lhs[m,k]·rhs[n,k]ᵀ. All
    /// offsets are BYTES into the arena (codes are u8, scales/out/bias f32).
    ScaledMatMul {
        m: u32,
        k: u32,
        n: u32,
        lhs_byte_off: u32,
        rhs_byte_off: u32,
        lhs_scale_byte_off: u32,
        rhs_scale_byte_off: u32,
        out_byte_off: u32,
        has_bias: u32,
        bias_byte_off: u32,
        lhs_e5m2: u32,
        rhs_e5m2: u32,
    },
    /// Per-tensor amax → f32 scale for a tensor about to be FP8-quantized.
    ScaledQuantScale {
        x_off_f32: u32,
        scale_off_f32: u32,
        n: u32,
        max_finite: f32,
    },
    /// Encode f32 → FP8 codes (per-tensor scale). `e5m2`: 0=E4M3, 1=E5M2.
    ScaledQuantizeFp8 {
        x_off_f32: u32,
        scale_off_f32: u32,
        out_byte_off: u32,
        n: u32,
        e5m2: u32,
    },
    /// Decode-and-accumulate GEMM fallback (non-tensor-core) for block / FP4 /
    /// FP6 configs cublasLt can't do. Byte offsets for codes/scales; f32-element
    /// offsets for out/bias.
    ScaledMatMulDecode {
        m: u32,
        k: u32,
        n: u32,
        lhs_byte_off: u32,
        rhs_byte_off: u32,
        lhs_scale_byte_off: u32,
        rhs_scale_byte_off: u32,
        out_off_f32: u32,
        lhs_fmt: u32,
        rhs_fmt: u32,
        scale_mode: u32,
        block: u32,
        has_bias: u32,
        bias_off_f32: u32,
    },
    /// General (all-format/all-layout) scale producer.
    ScaledQuantScaleGeneral {
        x_off_f32: u32,
        scale_byte_off: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    /// General (all-format/all-layout) quantize producer.
    ScaledQuantizeGeneral {
        x_off_f32: u32,
        scale_byte_off: u32,
        out_byte_off: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    ScaledDequantizeGeneral {
        codes_byte_off: u32,
        scale_byte_off: u32,
        out_off_f32: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    Binary {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Compare {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Unary {
        n: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Where {
        n: u32,
        cond_off: u32,
        x_off: u32,
        y_off: u32,
        out_off: u32,
    },
    Reduce {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Softmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    LayerNorm {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        gamma_off: u32,
        beta_off: u32,
        eps_bits: u32,
        op: u32,
    },
    FusedResidualLn {
        outer: u32,
        inner: u32,
        in_off: u32,
        residual_off: u32,
        bias_off: u32,
        gamma_off: u32,
        beta_off: u32,
        out_off: u32,
        eps_bits: u32,
        has_bias: u32,
    },
    FusedResidualRmsNorm {
        outer: u32,
        inner: u32,
        in_off: u32,
        residual_off: u32,
        bias_off: u32,
        gamma_off: u32,
        beta_off: u32,
        out_off: u32,
        eps_bits: u32,
        has_bias: u32,
    },
    Gather {
        n_out: u32,
        n_idx: u32,
        dim: u32,
        vocab: u32,
        in_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    GatherAxis {
        total: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
        table_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    Narrow {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Argmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    Transpose {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Expand {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Concat {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Attention {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        softcap_bits: u32,
        window: u32,
        seq_q_stride: u32,
        seq_k_stride: u32,
        mask_batch_stride: u32,
        mask_head_stride: u32,
        q_batch_stride: u32,
        q_head_stride: u32,
        q_seq_stride: u32,
        k_batch_stride: u32,
        k_head_stride: u32,
        k_seq_stride: u32,
        v_batch_stride: u32,
        v_head_stride: u32,
        v_seq_stride: u32,
        o_batch_stride: u32,
        o_head_stride: u32,
        o_seq_stride: u32,
    },
    /// Native fused-attention core (`fused_attn_block` kernel): inline RoPE +
    /// SDPA over the packed QKV scratch `[B,S,3*inner]` → attn scratch
    /// `[B,S,inner]`. One block per (batch·head); `seq*seq` f32 of shared
    /// memory hold the score matrix. The QKV / out projections are separate
    /// `Step::Matmul`s emitted by the same `Op::FusedAttentionBlock` arm.
    FusedAttn {
        qkv_off: u32,
        mask_off: u32,
        cos_off: u32,
        sin_off: u32,
        out_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        mask_kind: u32,
        scale_bits: u32,
        has_rope: u32,
    },
    AttentionBackward {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        dy_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        window: u32,
        wrt: u32,
    },
    Rope {
        n_total: u32,
        seq: u32,
        head_dim: u32,
        half: u32,
        /// Partial rotary: half of the rotated width `n_rot` (Gemma 4 global
        /// layers use n_rot < head_dim). Equals `half` for full rotation.
        rot_half: u32,
        in_off: u32,
        cos_off: u32,
        sin_off: u32,
        out_off: u32,
        last_dim: u32,
        interleaved: u32,
    },
    Cumsum {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        exclusive: u32,
    },
    TopK {
        outer: u32,
        inner: u32,
        k: u32,
        in_off: u32,
        out_off: u32,
    },
    GroupedMatmul {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        in_off: u32,
        w_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    ScatterAddZero {
        out_off: u32,
        out_total: u32,
    },
    ScatterAddAcc {
        out_off: u32,
        upd_off: u32,
        idx_off: u32,
        num_updates: u32,
        trailing: u32,
        out_dim: u32,
    },
    DequantMatmul {
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        scheme_id: u32,
        x_off: u32,
        w_off: u32,
        scale_off: u32,
        zp_off: u32,
        out_off: u32,
    },
    /// GGUF K-quant weights — GPU dequant scratch + cuBLAS (host fallback).
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        out_byte_off: u32,
    },
    DequantGroupedMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        idx_byte_off: u32,
        out_byte_off: u32,
    },
    Sample {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        top_k: u32,
        top_p_bits: u32,
        temp_bits: u32,
        seed_lo: u32,
        seed_hi: u32,
    },
    /// Host fill for [`Op::RngNormal`].
    RngNormal {
        dst_byte_off: u32,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    RngUniform {
        dst_byte_off: u32,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    SelectiveScan {
        batch: u32,
        seq: u32,
        hidden: u32,
        state_size: u32,
        x_off: u32,
        delta_off: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        out_off: u32,
    },
    /// 1D FFT — native GPU (f32 pow2) or host fallback.
    Fft {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype_tag: u32,
        use_gpu: bool,
        /// When true, `src_byte_off` points at an `n`-wide **real** signal (row
        /// stride `n`) instead of the 2N `[re|im]` block — the native FFT kernel
        /// reads `re` from it and uses `im = 0`, fusing the real→complex
        /// `Sub`+`Concat` zero-pad away. Only set by the native-cuda-fft fusion
        /// for stockham-eligible sizes.
        real_input: bool,
    },
    /// Log-mel from block-layout FFT spectrum — host fallback.
    LogMelHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    LogMelBackwardHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dy_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    /// Welch PSD top-K from block-layout spectra — host fallback.
    WelchPeaksHost {
        spec_byte_off: u32,
        dst_byte_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
    },
    /// Native GPU WelchPeaks (in-arena, no D2H).
    WelchPeaksGpu {
        spec_off: u32,
        dst_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
        n_bins: u32,
    },
    /// NCHW im2col — GPU kernel or host fallback (dynamic batch / `RLX_CUDA_IM2COL_HOST=1`).
    Im2ColHost {
        x_byte_off: u32,
        col_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw_dil: u32,
        use_gpu: bool,
    },
    /// Host-staged batch-general reverse/flip.
    ReverseHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u32,
    },
    /// Host-staged ArgMax/ArgMin (f32-encoded indices).
    ArgReduceHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
    },
    /// Host-staged axial 2-D RoPE.
    AxialRope2dHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        hidden: u32,
        end_x: u32,
        end_y: u32,
        head_dim: u32,
        num_heads: u32,
        theta: f32,
        repeat_factor: u32,
    },
    /// Gated-DeltaNet — host scan between GPU segments (qwen35 linear layers).
    GatedDeltaNet {
        q_byte_off: u32,
        k_byte_off: u32,
        v_byte_off: u32,
        g_byte_off: u32,
        beta_byte_off: u32,
        state_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
        use_carry: bool,
    },
    /// Single-layer LSTM via host fallback (D2H → CPU → H2D).
    Lstm {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        bias_byte_off: u32,
        h0_byte_off: u32,
        c0_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// General `Op::Scan` recurrence (e.g. IIR biquad) via host fallback
    /// (D2H → CPU body loop → H2D). Not CUDA-Graph-capture-safe.
    ScanHost {
        plan: std::sync::Arc<rlx_cpu::thunk::ScanBodyPlan>,
        outer_init_off: usize,
        outer_final_off: usize,
        length: u32,
        save_trajectory: bool,
        xs_outer: Vec<(usize, usize)>,
        bcast_outer: Vec<(usize, usize)>,
    },
    /// LLaDA2 / TIDE group-limited MoE gate (host TopK between GPU segments).
    Llada2GroupLimitedGate {
        sig_off: u32,
        route_off: u32,
        out_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
    },
    /// Fused multi-scale deformable attention (host compute between GPU segments).
    MsDeformAttnHost {
        in_offs: Vec<(u32, u32)>, // (f32_off, f32_len) per input
        out_off: u32,
        out_len: u32,
        attrs: Vec<u8>,
    },
    UmapKnn {
        pairwise_off: u32,
        out_off: u32,
        n: u32,
        k: u32,
    },
    /// 3D Gaussian splat — host reference between GPU segments.
    GaussianSplatRender {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        dst_off: u32,
        dst_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    GaussianSplatRenderBackward {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        d_loss_off: u32,
        d_loss_len: u32,
        packed_off: u32,
        packed_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
        loss_grad_clip: f32,
        sh_band: u32,
        max_anisotropy: f32,
    },
    GaussianSplatPrepare {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        meta_len: u32,
        prep_off: u32,
        prep_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    GaussianSplatRasterize {
        prep_off: u32,
        prep_len: u32,
        meta_off: u32,
        meta_len: u32,
        dst_off: u32,
        dst_len: u32,
        count: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    RmsNormBackwardInput {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardGamma {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dgamma_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardBeta {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dbeta_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RopeBackward {
        dy_byte_off: u32,
        cos_byte_off: u32,
        sin_byte_off: u32,
        dx_byte_off: u32,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        n_rot: u32,
        cos_len: u32,
    },
    CumsumBackward {
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    GatherBackward {
        dy_byte_off: u32,
        indices_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    MaxPool2dBackward {
        x_byte_off: u32,
        dy_byte_off: u32,
        dx_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
    },
    Conv2dBackwardInput {
        dy_byte_off: u32,
        w_byte_off: u32,
        dx_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
    },
    Conv2dBackwardWeight {
        x_byte_off: u32,
        dy_byte_off: u32,
        dw_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw_dil: u32,
        groups: u32,
    },
    Pool1d {
        n: u32,
        c: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool2d {
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool3d {
        n: u32,
        c: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Conv1d {
        n: u32,
        c_in: u32,
        c_out: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        dl: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    Conv2d {
        n: u32,
        c_in: u32,
        c_out: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    Conv3d {
        n: u32,
        c_in: u32,
        c_out: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        dd: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    /// NCHW LayerNorm2d (SAM semantics).
    LayerNorm2d {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps_bits: u32,
    },
    /// NCHW ConvTranspose2d (PyTorch weight layout).
    ConvTranspose2d {
        src_off: u32,
        w_off: u32,
        dst_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
    },
    /// NCHW group norm.
    GroupNorm {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps_bits: u32,
    },
    /// Nearest-neighbor 2× upsample on NCHW.
    ResizeNearest2x {
        src_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Backend-level fusion of `Binary → Unary` element-wise chains.
    /// Emitted by `fuse_elementwise_chains` when the intermediate
    /// offset has exactly one consumer in the schedule. Avoids one
    /// kernel launch + one round-trip to global memory for the
    /// intermediate result.
    FusedBinaryUnary {
        n: u32,
        a_off: u32,
        b_off: u32,
        out_off: u32,
        bin_op: u32,
        un_op: u32,
    },
    /// PLAN L2 — interpreted N-ary element-wise chain. The chain
    /// encoding (input_offs[8] + chain[64]) lives in `meta_buffers`
    /// and is indexed via `meta_idx`. One thread per output element;
    /// each thread walks the chain in registers and writes the final
    /// result to `arena[dst_off + i]`. Caps: 16 steps, 8 inputs.
    /// Emitted from `Op::ElementwiseRegion` by `MarkElementwiseRegions`
    /// (replaces the prior `UnfuseElementwiseRegions` decomposer
    /// fallback). `input_offs` mirrors what's packed in `meta` and is
    /// kept in the Step so the multi-stream scheduler can resolve
    /// producer-consumer dependencies without unpacking metadata.
    ElementwiseRegion {
        len: u32,
        num_inputs: u32,
        num_steps: u32,
        dst_off: u32,
        input_offs: [u32; 16],
        /// PLAN L2 quality fast path: per-input scalar bitfield.
        /// Bit `i` ⇒ input `i` is a single-element broadcast.
        scalar_input_mask: u32,
        /// PLAN L2 quality general broadcast: per-input element count.
        /// `0` ⇒ no broadcast (kernel reads gid); `>0` ⇒ kernel reads
        /// `arena[input_offs[i] + (gid % input_modulus[i])]`.
        input_modulus: [u32; 16],
        meta_idx: usize,
        /// When true, launch a W×H×(N·C) grid (resize prologue).
        spatial_prologue: bool,
        prologue_w: u32,
        prologue_h: u32,
        prologue_nc: u32,
    },
    /// FKL batch region: one launch over `num_batch` slices (`blockIdx.z`).
    BatchElementwiseRegion {
        slice_len: u32,
        num_batch: u32,
        num_steps: u32,
        base_dst_off: u32,
        slice_elems: u32,
        /// Host copy for schedule dependency edges.
        batch_input_offs: [u32; 64],
        batch_offs_idx: usize,
        meta_idx: usize,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
    },
}

/// When kernels turn into PTX device code.
///
/// `Jit` is the default — each kernel NVRTC-compiles on first dispatch,
/// then the cuModule is cached for the rest of the process. `Aot`
/// pre-compiles every kernel at executable construction so the first
/// `run()` doesn't pay any compile latency. The full AOT pass is ~1-3s
/// (10-100ms × 32 kernels) but moves that cost out of the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    #[default]
    Jit,
    Aot,
}

/// How the schedule executes.
///
/// `Stream` (default) launches each Step on the default stream every
/// `run()`. `Graph` captures the full schedule into a CUDA Graph on
/// first run and replays the captured graph on subsequent runs —
/// eliminates per-launch dispatch overhead (~10-20% on small-batch
/// inference). `Eager` is a one-shot helper that compiles + runs +
/// drops the executable in one call; useful for interactive debugging.
/// `MultiStream(n)` allocates a pool of `n` streams and assigns each
/// `Step` to a stream based on data dependencies — independent ops
/// (e.g. unfused Q/K/V projections, FFN gate/up) run in parallel.
/// Cross-stream synchronization uses CUDA events at producer-consumer
/// boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    Stream,
    Graph,
    Eager,
    MultiStream(usize),
}

pub struct CudaExecutable {
    ctx: Arc<CudaContext>,
    /// cuBLAS handle bound to the same default stream as `ctx`. Used for
    /// plain matmul (no fused bias/activation); falls back to the custom
    /// kernel when cuBLAS isn't available (e.g., on Mac via the panic-
    /// catch probe).
    blas: Option<Arc<Mutex<CudaBlas>>>,
    /// cuBLASLt handle for fused matmul + bias + activation. Falls back
    /// to plain cuBLAS sgemm + epilogue kernel when unavailable.
    blas_lt: Option<cublaslt_sys::cublasLtHandle_t>,
    /// Shared cuBLASLt scratch — process singleton, only referenced when
    /// the schedule uses cublasLt-fusable matmul.
    blas_lt_workspace: Option<Arc<Mutex<cudarc::driver::CudaSlice<u8>>>>,
    /// cuDNN handle for convolution dispatch (conv1d/2d/3d). Falls back
    /// to the custom direct-convolution kernels when unavailable.
    dnn: Option<cudnn_sys::cudnnHandle_t>,
    /// Shared cuDNN scratch — process singleton, only referenced when the
    /// schedule contains conv steps.
    dnn_workspace: Option<Arc<Mutex<cudarc::driver::CudaSlice<u8>>>>,
    /// Scratch f16 buffer for casting activations on-the-fly when the
    /// matching weight is half-stored. Sized to fit the largest
    /// per-call M·K product seen in matmul dispatch; grown lazily.
    half_act_scratch: Option<cudarc::driver::CudaSlice<u16>>,
    /// Byte offset in the f32 arena for GGUF dequant scratch (max k×n f32).
    dequant_scratch_off: usize,
    graph: Graph,
    arena: Arena,
    schedule: Vec<Step>,
    input_offsets: HashMap<String, NodeId>,
    param_offsets: HashMap<String, NodeId>,
    /// Per-step side buffers for kernels that need per-axis u32 metadata
    /// (Transpose, Expand). Indexed via `Step::Transpose.meta_idx` etc.
    meta_buffers: Vec<cudarc::driver::CudaSlice<u32>>,
    exec_mode: ExecMode,
    /// Captured CUDA Graph (built on first `run()` when `exec_mode ==
    /// Graph`). Replayed on subsequent runs to skip per-launch dispatch.
    captured_graph: Option<cudarc::driver::CudaGraph>,
    /// Stream pool for `ExecMode::MultiStream(n)`. Empty for the other
    /// modes (which use the context's default stream).
    streams: Vec<Arc<cudarc::driver::CudaStream>>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every step in `schedule` is in the
    /// safe set, `run` bypasses the captured CUDA Graph (recorded at
    /// full extent) and dispatches per-step with scaled launch dims.
    /// Otherwise full-extent fallback. See PLAN L1.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Reused host output buffers (stable addresses for CUDA Graph dtoh capture).
    output_staging: Vec<F32HostSlot>,
    /// Pinned/pageable host staging for fixed-size graph inputs.
    input_staging: HashMap<String, F32HostSlot>,
    /// cuFFT plan cache + interleaved scratch (only with the `cufft` feature).
    #[cfg(feature = "cufft")]
    cufft_state: crate::cufft_dispatch::CufftState,
    /// Reused event for graph replay completion (avoids full stream sync when possible).
    replay_event: Option<cudarc::driver::CudaEvent>,
    /// Persistent KV inputs (host mirror + device upload each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Row feeds: after decode, copy output row `src_row` into handle row `dst_row`.
    kv_row_feeds: HashMap<String, usize>,
    gpu_handle_resident: std::collections::HashSet<String>,
    /// When set, only these output indices are read back from device (KV feeds stay on GPU).
    pending_read_indices: Option<Vec<usize>>,
    /// Reused sorted/deduped output indices for the current run (avoids alloc in `readback_plan`).
    readback_plan_buf: Vec<usize>,
    /// Output indices baked into the captured CUDA graph (must match on replay).
    captured_readback_plan: Option<Vec<usize>>,
    /// Graph input names in declaration order (parallel to `input_slots`).
    input_slot_names: Vec<String>,
    /// Graph inputs in declaration order: `(arena_byte_offset, max_f32_elems)`.
    input_slots: Vec<(usize, usize)>,
    /// Host readback layout: `(byte_offset_in_host_arena, f32_elems)` per graph output.
    output_slots: Vec<(usize, usize)>,
    /// Pinned/pageable host mirror for `run_slots` / `arena_ptr` (not GPU arena).
    host_arena: Vec<f32>,
    /// Runtime-mutable RNG policy for in-graph random ops.
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

impl CudaExecutable {
    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.rng.write().expect("rng lock") = rng;
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Initial coverage: simple element-wise ops + reductions + softmax +
    /// LayerNorm + cumsum. Matmul, Attention, Conv, Pool, GroupedMatmul,
    /// DequantMatmul, Sample, SelectiveScan, Rope, ScatterAdd, Transpose,
    /// Expand, Concat, Narrow, Gather, GatherAxis, Argmax, TopK still
    /// default to unsafe — opt in once each Step's per-tier dispatch +
    /// kernel offset arithmetic has been verified to scale safely.
    pub fn safe_for_active_extent(&self) -> bool {
        matches!(
            self,
            Step::Binary { .. }
                | Step::Compare { .. }
                | Step::Unary { .. }
                | Step::Where { .. }
                | Step::Reduce { .. }
                | Step::Softmax { .. }
                | Step::LayerNorm { .. }
                | Step::FusedResidualLn { .. }
                | Step::FusedResidualRmsNorm { .. }
                | Step::Cumsum { .. }
                | Step::FusedBinaryUnary { .. }
                | Step::ElementwiseRegion { .. }
                | Step::BatchElementwiseRegion { .. }
        )
    }

    /// False when the step performs host-side work or stream sync during dispatch.
    pub fn graph_capture_safe(&self) -> bool {
        match self {
            Step::Im2ColHost { use_gpu, .. } | Step::Fft { use_gpu, .. } => *use_gpu,
            Step::GatedDeltaNet { .. }
            | Step::Llada2GroupLimitedGate { .. }
            | Step::MsDeformAttnHost { .. }
            | Step::UmapKnn { .. }
            | Step::LogMelHost { .. }
            | Step::LogMelBackwardHost { .. }
            | Step::WelchPeaksHost { .. }
            | Step::RngNormal { .. }
            | Step::RngUniform { .. }
            | Step::ReverseHost { .. }
            | Step::ArgReduceHost { .. }
            | Step::AxialRope2dHost { .. }
            | Step::ScanHost { .. }
            | Step::GaussianSplatRender { .. }
            | Step::GaussianSplatRenderBackward { .. }
            | Step::GaussianSplatPrepare { .. } => false,
            _ => true,
        }
    }
}

fn schedule_graph_capture_safe(schedule: &[Step]) -> bool {
    schedule.iter().all(Step::graph_capture_safe)
}

fn step_is_tail_host(step: &Step) -> bool {
    matches!(
        step,
        Step::LogMelHost { .. } | Step::LogMelBackwardHost { .. } | Step::WelchPeaksHost { .. }
    )
}

fn run_tail_host_audio_ops(
    schedule: &[Step],
    stream: &Arc<cudarc::driver::CudaStream>,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    pre_sync: bool,
) {
    if !schedule.iter().any(step_is_tail_host) {
        return;
    }
    if pre_sync {
        stream
            .synchronize()
            .expect("rlx-cuda: tail host pre-sync failed");
    }
    for step in schedule {
        match step {
            Step::LogMelHost {
                spec_byte_off,
                filt_byte_off,
                dst_byte_off,
                outer,
                n_fft,
                n_bins,
                n_mels,
            } => {
                crate::log_mel_host::run_log_mel(
                    stream,
                    buffer,
                    *spec_byte_off as usize,
                    *filt_byte_off as usize,
                    *dst_byte_off as usize,
                    *outer as usize,
                    *n_fft as usize,
                    *n_bins as usize,
                    *n_mels as usize,
                    false,
                );
            }
            Step::LogMelBackwardHost {
                spec_byte_off,
                filt_byte_off,
                dy_byte_off,
                dst_byte_off,
                outer,
                n_fft,
                n_bins,
                n_mels,
            } => {
                crate::log_mel_backward_host::run_log_mel_backward(
                    stream,
                    buffer,
                    *spec_byte_off as usize,
                    *filt_byte_off as usize,
                    *dy_byte_off as usize,
                    *dst_byte_off as usize,
                    *outer as usize,
                    *n_fft as usize,
                    *n_bins as usize,
                    *n_mels as usize,
                    false,
                );
            }
            Step::WelchPeaksHost {
                spec_byte_off,
                dst_byte_off,
                welch_batch,
                n_fft,
                n_segments,
                k,
            } => {
                crate::welch_peaks_host::run_welch_peaks(
                    stream,
                    buffer,
                    *spec_byte_off as usize,
                    *dst_byte_off as usize,
                    *welch_batch as usize,
                    *n_fft as usize,
                    *n_segments as usize,
                    *k as usize,
                    false,
                );
            }
            _ => {}
        }
    }
}

fn schedule_needs_blas_lt(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Matmul { act_id, .. } if cublaslt_act_supported(*act_id)
        )
    })
}

fn schedule_needs_dnn(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Conv1d { .. } | Step::Conv2d { .. } | Step::Conv3d { .. }
        )
    })
}

/// Map our internal activation id (matches the `unary` kernel table)
/// to a cuBLASLt epilogue activation, if it's natively fusable.
/// cuBLASLt only supports Relu and Gelu in the epilogue — anything else
/// (sigmoid, tanh, silu, abs, neg, sqrt) returns None and the caller
/// falls back to plain sgemm + the matmul_epilogue kernel.
fn cublaslt_act_for(act_id: u32) -> Option<cublaslt_sys::cublasLtEpilogue_t> {
    None.or(match act_id {
        // Identity
        0xFFFFu32 => Some(None),
        // Relu = 0; Gelu = 9; GeluApprox = 11 (treat as Gelu).
        0 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        )),
        9 | 11 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        )),
        _ => Some(None),
    })
    .flatten()
}

/// True when `act_id` is fusable in cuBLASLt's epilogue (or absent).
fn cublaslt_act_supported(act_id: u32) -> bool {
    matches!(act_id, 0xFFFFu32 | 0 | 9 | 11)
}

/// Single cuBLASLt fused matmul. Consumes one descriptor + three matrix
/// layouts + one preference object per call (descriptors are cheap to
/// create; future optimization could cache them by shape). Returns
/// `Err` on any setup failure so the caller can fall back to plain
/// cuBLAS sgemm + epilogue kernel.
unsafe fn cublaslt_matmul_fused(
    handle: cublaslt_sys::cublasLtHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
    has_bias: bool,
    bias_off_f32: u32,
    epilogue_act: Option<cublaslt_sys::cublasLtEpilogue_t>,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    cu_stream: cudarc::driver::sys::CUstream,
) -> Result<(), cublaslt_result::CublasError> {
    use core::ffi::c_void;
    use core::mem;

    // cuBLASLt is column-major. We swap A↔B so that "computing C^T =
    // B^T·A^T in column-major" matches "C = A·B in row-major".
    let a_ptr = (arena_dev_ptr + (b_off_f32 as u64) * 4) as *const c_void; // = our B
    let b_ptr = (arena_dev_ptr + (a_off_f32 as u64) * 4) as *const c_void; // = our A
    let c_ptr = (arena_dev_ptr + (c_off_f32 as u64) * 4) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    let dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;

    // Layouts. After A↔B swap: cuBLASLt sees a [n,k] · [k,m] = [n,m].
    let a_layout = cublaslt_result::create_matrix_layout(dt, n as u64, k as u64, n as i64)?;
    let b_layout = cublaslt_result::create_matrix_layout(dt, k as u64, m as u64, k as i64)?;
    let c_layout = cublaslt_result::create_matrix_layout(dt, n as u64, m as u64, n as i64)?;

    if batch > 1 {
        unsafe {
            let bsz = batch as i32;
            for &layout in &[a_layout, b_layout, c_layout] {
                cublaslt_result::set_matrix_layout_attribute(
                layout,
                cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &bsz as *const _ as *const _,
                mem::size_of::<i32>(),
            )?;
            }
            let stride_b = b_batch_stride as i64;
            let stride_a = a_batch_stride as i64;
            let stride_c = c_batch_stride as i64;
            cublaslt_result::set_matrix_layout_attribute(
            a_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_b as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            b_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_a as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            c_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c as *const _ as *const _, mem::size_of::<i64>())?;
        }
    }

    // CUBLAS_COMPUTE_32F_FAST_TF32 enables Tensor-Core paths on Ampere+.
    // Set RLX_CUDA_NO_TF32=1 (or RLX_CUDA_PARITY=1) for strict f32 parity
    // vs CPU / wgpu reference paths.
    let compute_type =
        if rlx_ir::env::flag("RLX_CUDA_NO_TF32") || rlx_ir::env::flag("RLX_CUDA_PARITY") {
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        } else {
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        };
    let matmul_desc = cublaslt_result::create_matmul_desc(compute_type, dt)?;

    // Pick the epilogue mode. cuBLASLt fuses bias broadcast over the
    // M dimension (in cuBLASLt's view). With our A↔B swap, cuBLASLt's
    // M = our row-major N, so a bias[N] vector broadcasts across M
    // rows of row-major C — exactly what we want.
    let epilogue = match (has_bias, epilogue_act) {
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS
        }
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        }
        (true, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (false, Some(act)) => act,
        (false, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        _ => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
    };
    unsafe {
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue as *const _ as *const _,
            mem::size_of::<cublaslt_sys::cublasLtEpilogue_t>(),
        )?;
    }

    if has_bias {
        let bias_dev_ptr = arena_dev_ptr + (bias_off_f32 as u64) * 4;
        unsafe {
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_dev_ptr as *const _ as *const _,
                mem::size_of::<u64>(),
            )?;
        }
    }

    let matmul_pref = cublaslt_result::create_matmul_pref()?;
    unsafe {
        cublaslt_result::set_matmul_pref_attribute(
            matmul_pref,
            cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        cublaslt_result::get_matmul_algo_heuristic(
            handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )
    }?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let workspace_ptr = workspace_dev_ptr as *mut c_void;

    let result = unsafe {
        cublaslt_result::matmul(
            handle,
            matmul_desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            c_ptr,
            c_layout,
            d_ptr,
            c_layout,
            &heuristic.algo as *const _,
            workspace_ptr,
            workspace_size,
            cu_stream as cublaslt_sys::cudaStream_t,
        )
    };

    // Always destroy descriptors (success or fail).
    unsafe {
        let _ = cublaslt_result::destroy_matmul_pref(matmul_pref);
        let _ = cublaslt_result::destroy_matmul_desc(matmul_desc);
        let _ = cublaslt_result::destroy_matrix_layout(c_layout);
        let _ = cublaslt_result::destroy_matrix_layout(b_layout);
        let _ = cublaslt_result::destroy_matrix_layout(a_layout);
    }

    result
}

/// Native **FP8 tensor-core GEMM** via cuBLASLt (Hopper/Ada sm_89+).
/// Computes row-major `D[m,n] = (lhs[m,k] · rhs[n,k]ᵀ) · lhs_scale · rhs_scale`
/// where `lhs`/`rhs` are FP8 (E4M3/E5M2) codes and the scales are device f32
/// scalars. This is RLX's `Op::ScaledMatMul` (TN layout) — the operands are fed
/// straight into the tensor cores with f32 accumulation, the real low-precision
/// throughput win that the decode-then-sgemm storage path leaves on the table.
///
/// Mapping to cuBLASLt's column-major `D = op(A)·op(B)`: we compute the
/// transpose `Dᵀ[n,m]` in column-major (= our row-major `D[m,n]`) with
///   A = rhs  (col-major `[k,n]`, op = **T**)   — FP8 requires transa=T
///   B = lhs  (col-major `[k,m]`, op = **N**)   — FP8 requires transb=N
/// so A↔scale: A_SCALE = rhs_scale, B_SCALE = lhs_scale. Offsets are **bytes**
/// (FP8 codes are 1 byte; scales/out/bias are f32).
#[allow(clippy::too_many_arguments)]
unsafe fn cublaslt_matmul_fp8(
    handle: cublaslt_sys::cublasLtHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    lhs_byte_off: u64,
    rhs_byte_off: u64,
    lhs_scale_byte_off: u64,
    rhs_scale_byte_off: u64,
    out_byte_off: u64,
    has_bias: bool,
    bias_byte_off: u64,
    lhs_e5m2: bool,
    rhs_e5m2: bool,
    cu_stream: cudarc::driver::sys::CUstream,
) -> Result<(), cublaslt_result::CublasError> {
    use core::ffi::c_void;
    use core::mem;

    let fp8 = |e5m2: bool| {
        if e5m2 {
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E5M2
        } else {
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E4M3
        }
    };
    let a_dt = fp8(rhs_e5m2); // A = rhs
    let b_dt = fp8(lhs_e5m2); // B = lhs
    let out_dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;

    let a_ptr = (arena_dev_ptr + rhs_byte_off) as *const c_void;
    let b_ptr = (arena_dev_ptr + lhs_byte_off) as *const c_void;
    let c_ptr = (arena_dev_ptr + out_byte_off) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    // A = rhs col-major [k,n] ld=k; B = lhs col-major [k,m] ld=k;
    // D = col-major [n,m] ld=n  (== row-major [m,n]).
    let a_layout = cublaslt_result::create_matrix_layout(a_dt, k as u64, n as u64, k as i64)?;
    let b_layout = cublaslt_result::create_matrix_layout(b_dt, k as u64, m as u64, k as i64)?;
    let cd_layout = cublaslt_result::create_matrix_layout(out_dt, n as u64, m as u64, n as i64)?;

    // FP8 accumulation is f32; scale type (alpha/beta) f32.
    let compute_type = cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let matmul_desc = cublaslt_result::create_matmul_desc(
        compute_type,
        cublaslt_sys::cudaDataType_t::CUDA_R_32F,
    )?;

    // cuBLASLt FP8 requires transa = T, transb = N (cublasOperation_t as i32).
    let op_t: i32 = 1; // CUBLAS_OP_T
    let op_n: i32 = 0; // CUBLAS_OP_N
    unsafe {
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &op_t as *const i32 as *const _,
            mem::size_of::<i32>(),
        )?;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &op_n as *const i32 as *const _,
            mem::size_of::<i32>(),
        )?;

        // Per-tensor dequant scales: D = a_scale · b_scale · (A·B).
        let a_scale_ptr = arena_dev_ptr + rhs_scale_byte_off;
        let b_scale_ptr = arena_dev_ptr + lhs_scale_byte_off;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale_ptr as *const u64 as *const _,
            mem::size_of::<u64>(),
        )?;
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale_ptr as *const u64 as *const _,
            mem::size_of::<u64>(),
        )?;

        if has_bias {
            let epi = cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epi as *const _ as *const _,
                mem::size_of::<cublaslt_sys::cublasLtEpilogue_t>(),
            )?;
            let bias_ptr = arena_dev_ptr + bias_byte_off;
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_ptr as *const u64 as *const _,
                mem::size_of::<u64>(),
            )?;
        }
    }

    let matmul_pref = cublaslt_result::create_matmul_pref()?;
    unsafe {
        cublaslt_result::set_matmul_pref_attribute(
            matmul_pref,
            cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        cublaslt_result::get_matmul_algo_heuristic(
            handle,
            matmul_desc,
            a_layout,
            b_layout,
            cd_layout,
            cd_layout,
            matmul_pref,
        )
    }?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let result = unsafe {
        cublaslt_result::matmul(
            handle,
            matmul_desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            c_ptr,
            cd_layout,
            d_ptr,
            cd_layout,
            &heuristic.algo as *const _,
            workspace_dev_ptr as *mut c_void,
            workspace_size,
            cu_stream as cublaslt_sys::cudaStream_t,
        )
    };

    unsafe {
        let _ = cublaslt_result::destroy_matmul_pref(matmul_pref);
        let _ = cublaslt_result::destroy_matmul_desc(matmul_desc);
        let _ = cublaslt_result::destroy_matrix_layout(cd_layout);
        let _ = cublaslt_result::destroy_matrix_layout(b_layout);
        let _ = cublaslt_result::destroy_matrix_layout(a_layout);
    }
    result
}

/// cuDNN forward 2D convolution against arena offsets. NCHW input,
/// KCRS filter, NCHW output. Uses the v7 algorithm heuristic to pick
/// the fastest algo that fits in the supplied workspace. Returns
/// `Err` on any setup failure so the caller can fall back to the
/// direct-convolution kernel.
unsafe fn cudnn_conv2d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;

    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            y_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        // Pick the fastest fwd algo via the v7 heuristic.
        let mut returned_count: i32 = 0;
        let mut perf = std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;

        let needed = cudnn_result::get_convolution_forward_workspace_size(
            handle, x_desc, w_desc, conv_desc, y_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// cuDNN backward-data 2-D convolution: dx (input grad) from dy and w.
/// Mirrors `cudnn_conv2d_forward`; returns Err so the caller can fall back
/// to the host reference.
#[allow(clippy::too_many_arguments)]
unsafe fn cudnn_conv2d_backward_data(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    dy_off_f32: u32,
    w_off_f32: u32,
    dx_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;
    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let dx_desc = cudnn_result::create_tensor_descriptor()?;
    let dy_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let w_desc = unsafe {
        let mut u = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(u.as_mut_ptr()).result()?;
        u.assume_init()
    };
    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            dx_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            dy_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };
    let result = setup.and_then(|()| unsafe {
        let mut returned_count: i32 = 0;
        let mut perf =
            std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_backward_data_algorithm(
            handle,
            w_desc,
            dy_desc,
            conv_desc,
            dx_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;
        let needed = cudnn_result::get_convolution_backward_data_workspace_size(
            handle, w_desc, dy_desc, conv_desc, dx_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let dy_ptr = (arena_dev_ptr + (dy_off_f32 as u64) * 4) as *const c_void;
        let dx_ptr = (arena_dev_ptr + (dx_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;
        cudnn_result::convolution_backward_data(
            handle,
            &alpha as *const _ as *const c_void,
            w_desc,
            w_ptr,
            dy_desc,
            dy_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            dx_desc,
            dx_ptr,
        )
    });
    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dy_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dx_desc);
    }
    result
}

/// cuDNN backward-filter 2-D convolution: dw (weight grad) from x and dy.
#[allow(clippy::too_many_arguments)]
unsafe fn cudnn_conv2d_backward_filter(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    x_off_f32: u32,
    dy_off_f32: u32,
    dw_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;
    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let dy_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let dw_desc = unsafe {
        let mut u = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(u.as_mut_ptr()).result()?;
        u.assume_init()
    };
    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            dy_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            dw_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };
    let result = setup.and_then(|()| unsafe {
        let mut returned_count: i32 = 0;
        let mut perf =
            std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_backward_filter_algorithm(
            handle,
            x_desc,
            dy_desc,
            conv_desc,
            dw_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;
        let needed = cudnn_result::get_convolution_backward_filter_workspace_size(
            handle, x_desc, dy_desc, conv_desc, dw_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (x_off_f32 as u64) * 4) as *const c_void;
        let dy_ptr = (arena_dev_ptr + (dy_off_f32 as u64) * 4) as *const c_void;
        let dw_ptr = (arena_dev_ptr + (dw_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;
        cudnn_result::convolution_backward_filter(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            dy_desc,
            dy_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            dw_desc,
            dw_ptr,
        )
    });
    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(dw_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(dy_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }
    result
}

/// cuDNN forward 3-D convolution. NCDHW input, KCDRS filter, NCDHW
/// output. Uses cuDNN's nd-descriptor APIs (set_tensornd / set_filternd
/// / set_convolutionnd) since the 4D versions only cover up to 2D conv.
unsafe fn cudnn_conv3d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    // 5-D tensor: [N, C, D, H, W] with row-major contiguous strides.
    let x_dims: [i32; 5] = [n as i32, c_in as i32, d as i32, h as i32, w as i32];
    let x_strides: [i32; 5] = [
        (c_in * d * h * w) as i32,
        (d * h * w) as i32,
        (h * w) as i32,
        w as i32,
        1,
    ];
    let y_dims: [i32; 5] = [
        n as i32,
        c_out as i32,
        d_out as i32,
        h_out as i32,
        w_out as i32,
    ];
    let y_strides: [i32; 5] = [
        (c_out * d_out * h_out * w_out) as i32,
        (d_out * h_out * w_out) as i32,
        (h_out * w_out) as i32,
        w_out as i32,
        1,
    ];
    let f_dims: [i32; 5] = [
        c_out as i32,
        (c_in / groups.max(1)) as i32,
        kd as i32,
        kh as i32,
        kw as i32,
    ];
    let pads: [i32; 3] = [pd as i32, ph as i32, pw as i32];
    let strides: [i32; 3] = [sd as i32, sh as i32, sw as i32];
    let dilations: [i32; 3] = [dd as i32, dh as i32, dw as i32];

    let setup = unsafe {
        cudnn_result::set_tensornd_descriptor(x_desc, dt, 5, x_dims.as_ptr(), x_strides.as_ptr())?;
        cudnn_result::set_tensornd_descriptor(y_desc, dt, 5, y_dims.as_ptr(), y_strides.as_ptr())?;
        cudnn_result::set_filternd_descriptor(w_desc, dt, fmt, 5, f_dims.as_ptr())?;
        cudnn_result::set_convolutionnd_descriptor(
            conv_desc,
            3,
            pads.as_ptr(),
            strides.as_ptr(),
            dilations.as_ptr(),
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        let mut returned_count: i32 = 0;
        let mut perf = std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;

        let needed = cudnn_result::get_convolution_forward_workspace_size(
            handle, x_desc, w_desc, conv_desc, y_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// Per-`Op::FusedAttentionBlock` scratch: packed QKV `[B,S,3*inner]` followed
/// by the attention output `[B,S,inner]`, both f32, 16-byte aligned per block.
/// Returns the total scratch size in BYTES and a map from each surviving FAB
/// node to its `(qkv, attn)` f32-element offsets *relative to the scratch
/// base*. Empty when the unfuse pass decomposed every FAB to primitives.
fn fab_scratch_plan(graph: &Graph) -> (usize, HashMap<rlx_ir::NodeId, (u32, u32)>) {
    let mut map = HashMap::new();
    let mut cur: usize = 0; // f32 elements
    for node in graph.nodes() {
        if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            ..
        } = &node.op
        {
            let dims = node.shape.dims();
            let b = dims[0].unwrap_static();
            let s = dims[1].unwrap_static();
            let inner = num_heads * head_dim;
            let qkv_rel = cur as u32;
            cur += b * s * 3 * inner;
            let attn_rel = cur as u32;
            cur += b * s * inner;
            cur = (cur + 3) & !3; // 16-byte align the next block's region
            map.insert(node.id, (qkv_rel, attn_rel));
        }
    }
    (cur * 4, map)
}

/// Decode a Matmul/FusedMatMulBiasAct node's input shapes into the
/// (m, k, n, batch, a_stride, b_stride, c_stride, a_id, b_id) tuple
/// the kernel expects. Three patterns:
///   • 2D × 2D                       → batch=1, all strides 0
///   • [..,M,K] × [K,N] (broadcast)  → batch=1, leading dims flattened into M
///   • [..,M,K] × [..,K,N] (matched) → batch=prod(leading), per-batch strides
fn matmul_shape(
    graph: &Graph,
    node: &rlx_ir::Node,
    op_label: &str,
) -> (u32, u32, u32, u32, u32, u32, u32, NodeId, NodeId) {
    let a_id = node.inputs[0];
    let b_id = node.inputs[1];
    let a_shape = graph.node(a_id).shape.dims();
    let b_shape = graph.node(b_id).shape.dims();
    let out_shape = node.shape.dims();
    if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2 {
        let m = a_shape[0].unwrap_static() as u32;
        let k = a_shape[1].unwrap_static() as u32;
        let n = b_shape[1].unwrap_static() as u32;
        (m, k, n, 1, 0, 0, 0, a_id, b_id)
    } else if a_shape.len() >= 2 && b_shape.len() == 2 && out_shape.len() == a_shape.len() {
        let leading: usize = a_shape[..a_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .product();
        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
        let n_inner = b_shape[1].unwrap_static();
        (
            (leading * m_inner) as u32,
            k_inner as u32,
            n_inner as u32,
            1,
            0,
            0,
            0,
            a_id,
            b_id,
        )
    } else if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
        let leading_a: Vec<usize> = a_shape[..a_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let leading_b: Vec<usize> = b_shape[..b_shape.len() - 2]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if leading_a != leading_b {
            panic!(
                "rlx-cuda {op_label}: batched shape mismatch \
                    a_leading={leading_a:?} b_leading={leading_b:?}"
            );
        }
        let b_count: usize = leading_a.iter().product();
        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
        (
            m_inner as u32,
            k_inner as u32,
            n_inner as u32,
            b_count as u32,
            (m_inner * k_inner) as u32,
            (k_inner * n_inner) as u32,
            (m_inner * n_inner) as u32,
            a_id,
            b_id,
        )
    } else {
        panic!(
            "rlx-cuda {op_label}: unsupported shapes a={a_shape:?} b={b_shape:?} out={out_shape:?}"
        );
    }
}

fn binary_op_id(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

fn compare_op_id(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

fn reduce_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

/// Op code for the `pool{1,2,3}d.cu` kernels, whose legend is `0=max, 1=mean,
/// 2=sum, 3=min, 4=prod` — this differs from [`reduce_op_id`] (which swaps Max
/// and Sum). Using `reduce_op_id` here made max-pooling compute the window sum.
fn pool_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Max => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Sum => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

fn activation_op_id(act: Activation) -> u32 {
    match act {
        Activation::Relu => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Exp => 3,
        Activation::Log => 4,
        Activation::Sqrt => 5,
        Activation::Rsqrt => 6,
        Activation::Neg => 7,
        Activation::Abs => 8,
        Activation::Gelu => 9,
        Activation::Silu => 10,
        Activation::GeluApprox => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
    }
}

/// Mixed-precision matmul tier-0: when the weight (B input) is stored
/// in the half-arena, cast f32 activations to f16/bf16 in the scratch
/// buffer and run `cublasGemmEx` with both inputs half + f32
/// accumulator. Returns `true` on success.
///
/// Free function (rather than `&mut self` method) so the caller can
/// hold `&self.schedule` across the call without violating disjoint-
/// field borrow checks.
#[allow(clippy::too_many_arguments)]
fn try_mixed_precision_gemm(
    ctx: &Arc<CudaContext>,
    arena: &mut crate::arena::Arena,
    half_act_scratch: &mut Option<cudarc::driver::CudaSlice<u16>>,
    blas: Option<&Arc<Mutex<CudaBlas>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
) -> bool {
    let (half_off, half_dtype) = match arena.half_by_f32_off.get(&b_off_f32).copied() {
        Some(v) => v,
        None => return false,
    };
    let blas = match blas {
        Some(b) => b,
        None => return false,
    };

    let act_elems = (m * k * batch.max(1)) as usize;
    let need_resize = half_act_scratch
        .as_ref()
        .is_none_or(|s| s.len() < act_elems);
    if need_resize {
        *half_act_scratch = stream.alloc_zeros::<u16>(act_elems.max(4)).ok();
    }
    if half_act_scratch.is_none() {
        return false;
    }

    // Phase 1: cast activations f32 → f16/bf16 into the scratch.
    let n_total = m * k * batch.max(1);
    let dtype_id: u32 = match half_dtype {
        crate::arena::HalfDtype::F16 => 0,
        crate::arena::HalfDtype::Bf16 => 1,
    };
    {
        let kernel = crate::kernels::cast_f32_to_half_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n_total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let src_view = arena
            .f32_buf()
            .slice(a_off_f32 as usize..a_off_f32 as usize + n_total as usize);
        let scratch_mut = half_act_scratch.as_mut().unwrap();
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&src_view)
            .arg(scratch_mut)
            .arg(&n_total)
            .arg(&dtype_id);
        if unsafe { launcher.launch(cfg) }.is_err() {
            return false;
        }
    }

    // Phase 2: cublasGemmEx with both inputs half + f32 output.
    let blas = blas.lock().unwrap();
    let arena_ptr_u64 = {
        let (p, _ar) = arena.buffer.device_ptr_mut(stream);
        p
    };
    let (half_buf_ptr, _hb) = arena.half_buffer.as_mut().unwrap().device_ptr_mut(stream);
    let scratch_ptr_u64 = {
        let s = half_act_scratch.as_mut().unwrap();
        let (p, _r) = s.device_ptr_mut(stream);
        p
    };
    let weight_dev = half_buf_ptr + (half_off as u64) * 2; // u16 = 2 bytes
    let act_dev = scratch_ptr_u64;
    let c_dev = arena_ptr_u64 + (c_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let cuda_dt = match half_dtype {
        crate::arena::HalfDtype::F16 => cublas_sys::cudaDataType_t::CUDA_R_16F,
        crate::arena::HalfDtype::Bf16 => cublas_sys::cudaDataType_t::CUDA_R_16BF,
    };
    let compute_ty = match half_dtype {
        crate::arena::HalfDtype::F16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F
        }
        crate::arena::HalfDtype::Bf16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
        }
    };
    let result = unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32 as *const _,
            weight_dev as *const _,
            cuda_dt,
            n as i32,
            act_dev as *const _,
            cuda_dt,
            k as i32,
            &beta as *const f32 as *const _,
            c_dev as *mut _,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            compute_ty,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if let Err(ref e) = result {
        log_fallback("matmul.gemmEx (mixed-precision)", e);
    }
    result.is_ok()
}

/// One-time-per-tier log when a fast-path dispatch silently falls
/// back. Helps cloud-GPU debugging see *why* the slow path took over —
/// otherwise the only signal is unexpectedly low throughput.
/// Gated behind `RLX_CUDA_LOG_FALLBACK=1` so production isn't spammed.
fn log_fallback(tier: &str, err: impl std::fmt::Debug) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_LOG_FALLBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if enabled {
        eprintln!("rlx-cuda: tier '{tier}' fell back: {err:?}");
    }
}

/// Stable, profiler-friendly name for an NVTX range covering a Step
/// dispatch. Matches the variant name; nsight-systems / nvprof show
/// these as range boundaries in the timeline.
fn fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-cuda Op::Fft: unsupported dtype {other:?}"),
    }
}

fn fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-cuda Op::Fft: bad dtype tag {other}"),
    }
}

fn step_name(step: &Step) -> &'static str {
    match step {
        Step::Matmul { .. } => "rlx::Matmul",
        Step::ScaledMatMul { .. } => "rlx::ScaledMatMul",
        Step::ScaledQuantScale { .. } => "rlx::ScaledQuantScale",
        Step::ScaledQuantizeFp8 { .. } => "rlx::ScaledQuantizeFp8",
        Step::ScaledMatMulDecode { .. } => "rlx::ScaledMatMulDecode",
        Step::ScaledQuantScaleGeneral { .. } => "rlx::ScaledQuantScaleGeneral",
        Step::ScaledQuantizeGeneral { .. } => "rlx::ScaledQuantizeGeneral",
        Step::ScaledDequantizeGeneral { .. } => "rlx::ScaledDequantizeGeneral",
        Step::Binary { .. } => "rlx::Binary",
        Step::Compare { .. } => "rlx::Compare",
        Step::Unary { .. } => "rlx::Unary",
        Step::Where { .. } => "rlx::Where",
        Step::Reduce { .. } => "rlx::Reduce",
        Step::Softmax { .. } => "rlx::Softmax",
        Step::LayerNorm { .. } => "rlx::LayerNorm",
        Step::FusedResidualLn { .. } => "rlx::FusedResidualLN",
        Step::FusedResidualRmsNorm { .. } => "rlx::FusedResidualRmsNorm",
        Step::Gather { .. } => "rlx::Gather",
        Step::GatherAxis { .. } => "rlx::GatherAxis",
        Step::Narrow { .. } => "rlx::Narrow",
        Step::Concat { .. } => "rlx::Concat",
        Step::Transpose { .. } => "rlx::Transpose",
        Step::Expand { .. } => "rlx::Expand",
        Step::Argmax { .. } => "rlx::Argmax",
        Step::Attention { .. } => "rlx::Attention",
        Step::FusedAttn { .. } => "rlx::FusedAttn",
        Step::AttentionBackward { .. } => "rlx::AttentionBackward",
        Step::Rope { .. } => "rlx::Rope",
        Step::Cumsum { .. } => "rlx::Cumsum",
        Step::TopK { .. } => "rlx::TopK",
        Step::GroupedMatmul { .. } => "rlx::GroupedMatmul",
        Step::ScatterAddZero { .. } => "rlx::ScatterAdd::zero",
        Step::ScatterAddAcc { .. } => "rlx::ScatterAdd::acc",
        Step::DequantMatmul { .. } => "rlx::DequantMatmul",
        Step::DequantMatmulGguf { .. } => "rlx::DequantMatmulGguf",
        Step::DequantGroupedMatmulGguf { .. } => "rlx::DequantGroupedMatmulGguf",
        Step::Sample { .. } => "rlx::Sample",
        Step::RngNormal { .. } => "rlx::RngNormal",
        Step::RngUniform { .. } => "rlx::RngUniform",
        Step::SelectiveScan { .. } => "rlx::SelectiveScan",
        Step::Fft { .. } => "rlx::Fft",
        Step::LogMelHost { .. } => "rlx::LogMelHost",
        Step::LogMelBackwardHost { .. } => "rlx::LogMelBackwardHost",
        Step::WelchPeaksHost { .. } => "rlx::WelchPeaksHost",
        Step::WelchPeaksGpu { .. } => "rlx::WelchPeaksGpu",
        Step::Im2ColHost { .. } => "rlx::Im2ColHost",
        Step::ReverseHost { .. } => "rlx::ReverseHost",
        Step::ArgReduceHost { .. } => "rlx::ArgReduceHost",
        Step::AxialRope2dHost { .. } => "rlx::AxialRope2dHost",
        Step::GatedDeltaNet { .. } => "rlx::GatedDeltaNet",
        Step::Lstm { .. } => "rlx::Lstm",
        Step::ScanHost { .. } => "rlx::ScanHost",
        Step::Llada2GroupLimitedGate { .. } => "rlx::Llada2GroupLimitedGate",
        Step::MsDeformAttnHost { .. } => "rlx::MsDeformAttnHost",
        Step::UmapKnn { .. } => "rlx::UmapKnn",
        Step::GaussianSplatRender { .. } => "rlx::GaussianSplatRender",
        Step::GaussianSplatRenderBackward { .. } => "rlx::GaussianSplatRenderBackward",
        Step::GaussianSplatPrepare { .. } => "rlx::GaussianSplatPrepare",
        Step::GaussianSplatRasterize { .. } => "rlx::GaussianSplatRasterize",
        Step::RmsNormBackwardInput { .. } => "rlx::RmsNormBackwardInput",
        Step::RmsNormBackwardGamma { .. } => "rlx::RmsNormBackwardGamma",
        Step::RmsNormBackwardBeta { .. } => "rlx::RmsNormBackwardBeta",
        Step::RopeBackward { .. } => "rlx::RopeBackward",
        Step::CumsumBackward { .. } => "rlx::CumsumBackward",
        Step::GatherBackward { .. } => "rlx::GatherBackward",
        Step::MaxPool2dBackward { .. } => "rlx::MaxPool2dBackward",
        Step::Conv2dBackwardInput { .. } => "rlx::Conv2dBackwardInput",
        Step::Conv2dBackwardWeight { .. } => "rlx::Conv2dBackwardWeight",
        Step::Pool1d { .. } => "rlx::Pool1d",
        Step::Pool2d { .. } => "rlx::Pool2d",
        Step::Pool3d { .. } => "rlx::Pool3d",
        Step::Conv1d { .. } => "rlx::Conv1d",
        Step::Conv2d { .. } => "rlx::Conv2d",
        Step::Conv3d { .. } => "rlx::Conv3d",
        Step::LayerNorm2d { .. } => "rlx::LayerNorm2d",
        Step::ConvTranspose2d { .. } => "rlx::ConvTranspose2d",
        Step::GroupNorm { .. } => "rlx::GroupNorm",
        Step::ResizeNearest2x { .. } => "rlx::ResizeNearest2x",
        Step::FusedBinaryUnary { .. } => "rlx::FusedBinaryUnary",
        Step::ElementwiseRegion { .. } => "rlx::ElementwiseRegion",
        Step::BatchElementwiseRegion { .. } => "rlx::BatchElementwiseRegion",
    }
}

/// Walk a freshly-built schedule and merge `Binary → Unary` element-wise
/// chains into `FusedBinaryUnary`. Conditions for fusion:
///   1. The pair has matching element count `n`.
///   2. The Unary's input offset == the Binary's output offset.
///   3. The intermediate offset has exactly one consumer in the
///      schedule (= no other Step reads it). This guarantees we can
///      drop the round-trip to global memory for the intermediate
///      without breaking any other Step's input.
fn fuse_elementwise_chains(schedule: Vec<Step>) -> Vec<Step> {
    // Tally consumer counts per offset: how many Steps in the schedule
    // read each offset.
    let mut consumer_counts: HashMap<u32, usize> = HashMap::new();
    for step in &schedule {
        let (reads, _) = step_offsets(step);
        for r in &reads {
            *consumer_counts.entry(*r).or_insert(0) += 1;
        }
    }

    let mut out = Vec::with_capacity(schedule.len());
    let mut i = 0;
    while i < schedule.len() {
        if i + 1 < schedule.len() {
            let pair = (&schedule[i], &schedule[i + 1]);
            if let (
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op: bin_op,
                },
                Step::Unary {
                    n: n2,
                    in_off,
                    out_off,
                    op: un_op,
                },
            ) = pair
            {
                let single_consumer = consumer_counts.get(c_off).copied() == Some(1);
                if n == n2 && c_off == in_off && single_consumer {
                    out.push(Step::FusedBinaryUnary {
                        n: *n,
                        a_off: *a_off,
                        b_off: *b_off,
                        out_off: *out_off,
                        bin_op: *bin_op,
                        un_op: *un_op,
                    });
                    i += 2;
                    continue;
                }
            }
        }
        out.push(schedule[i].clone());
        i += 1;
    }
    out
}

/// (read offsets, write offsets) for a Step. Used by the multi-stream
/// scheduler to decide which streams each step depends on. Offsets are
/// the leading f32-element offset of each input/output tensor — a
/// coarse approximation that's correct for our planner since each
/// node has its own slot (Reshape/Cast aliasing maps consumers to the
/// same slot, which is exactly what the dependency tracker wants).
fn step_offsets(step: &Step) -> (Vec<u32>, Vec<u32>) {
    match step {
        Step::ScanHost {
            outer_init_off,
            outer_final_off,
            xs_outer,
            bcast_outer,
            ..
        } => {
            let mut reads = vec![(*outer_init_off / 4) as u32];
            reads.extend(bcast_outer.iter().map(|&(o, _)| (o / 4) as u32));
            reads.extend(xs_outer.iter().map(|&(o, _)| (o / 4) as u32));
            (reads, vec![(*outer_final_off / 4) as u32])
        }
        Step::Matmul {
            a_off_f32,
            b_off_f32,
            c_off_f32,
            has_bias,
            bias_off_f32,
            ..
        } => {
            let mut r = vec![*a_off_f32, *b_off_f32];
            if *has_bias != 0 {
                r.push(*bias_off_f32);
            }
            (r, vec![*c_off_f32])
        }
        // Offsets here are coarse f32-element slot keys; byte offsets ÷4 land in
        // the right slot since the planner aligns each tensor's slot.
        Step::ScaledMatMul {
            lhs_byte_off,
            rhs_byte_off,
            lhs_scale_byte_off,
            rhs_scale_byte_off,
            out_byte_off,
            has_bias,
            bias_byte_off,
            ..
        } => {
            let mut r = vec![
                *lhs_byte_off / 4,
                *rhs_byte_off / 4,
                *lhs_scale_byte_off / 4,
                *rhs_scale_byte_off / 4,
            ];
            if *has_bias != 0 {
                r.push(*bias_byte_off / 4);
            }
            (r, vec![*out_byte_off / 4])
        }
        Step::ScaledQuantScale {
            x_off_f32,
            scale_off_f32,
            ..
        } => (vec![*x_off_f32], vec![*scale_off_f32]),
        Step::ScaledQuantizeFp8 {
            x_off_f32,
            scale_off_f32,
            out_byte_off,
            ..
        } => (vec![*x_off_f32, *scale_off_f32], vec![*out_byte_off / 4]),
        Step::ScaledMatMulDecode {
            lhs_byte_off,
            rhs_byte_off,
            lhs_scale_byte_off,
            rhs_scale_byte_off,
            out_off_f32,
            has_bias,
            bias_off_f32,
            ..
        } => {
            let mut r = vec![
                *lhs_byte_off / 4,
                *rhs_byte_off / 4,
                *lhs_scale_byte_off / 4,
                *rhs_scale_byte_off / 4,
            ];
            if *has_bias != 0 {
                r.push(*bias_off_f32);
            }
            (r, vec![*out_off_f32])
        }
        Step::ScaledQuantScaleGeneral {
            x_off_f32,
            scale_byte_off,
            ..
        } => (vec![*x_off_f32], vec![*scale_byte_off / 4]),
        Step::ScaledQuantizeGeneral {
            x_off_f32,
            scale_byte_off,
            out_byte_off,
            ..
        } => (
            vec![*x_off_f32, *scale_byte_off / 4],
            vec![*out_byte_off / 4],
        ),
        Step::ScaledDequantizeGeneral {
            codes_byte_off,
            scale_byte_off,
            out_off_f32,
            ..
        } => (
            vec![*codes_byte_off / 4, *scale_byte_off / 4],
            vec![*out_off_f32],
        ),
        Step::Binary {
            a_off,
            b_off,
            c_off,
            ..
        }
        | Step::Compare {
            a_off,
            b_off,
            c_off,
            ..
        } => (vec![*a_off, *b_off], vec![*c_off]),
        Step::Unary {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Where {
            cond_off,
            x_off,
            y_off,
            out_off,
            ..
        } => (vec![*cond_off, *x_off, *y_off], vec![*out_off]),
        Step::Reduce {
            in_off, out_off, ..
        }
        | Step::Softmax {
            in_off, out_off, ..
        }
        | Step::Argmax {
            in_off, out_off, ..
        }
        | Step::Cumsum {
            in_off, out_off, ..
        }
        | Step::Sample {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::RngNormal { dst_byte_off, .. } | Step::RngUniform { dst_byte_off, .. } => {
            (vec![], vec![*dst_byte_off / 4])
        }
        Step::TopK {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::LayerNorm {
            in_off,
            gamma_off,
            beta_off,
            out_off,
            ..
        } => (vec![*in_off, *gamma_off, *beta_off], vec![*out_off]),
        Step::FusedResidualLn {
            in_off,
            residual_off,
            bias_off,
            gamma_off,
            beta_off,
            out_off,
            has_bias,
            ..
        } => {
            let mut r = vec![*in_off, *residual_off, *gamma_off, *beta_off];
            if *has_bias != 0 {
                r.push(*bias_off);
            }
            (r, vec![*out_off])
        }
        Step::FusedResidualRmsNorm {
            in_off,
            residual_off,
            bias_off,
            gamma_off,
            beta_off,
            out_off,
            has_bias,
            ..
        } => {
            let mut r = vec![*in_off, *residual_off, *gamma_off, *beta_off];
            if *has_bias != 0 {
                r.push(*bias_off);
            }
            (r, vec![*out_off])
        }
        Step::Gather {
            in_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *idx_off], vec![*out_off]),
        Step::GatherAxis {
            table_off,
            idx_off,
            out_off,
            ..
        } => (vec![*table_off, *idx_off], vec![*out_off]),
        Step::Narrow {
            in_off, out_off, ..
        }
        | Step::Concat {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Transpose {
            in_off, out_off, ..
        }
        | Step::Expand {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Attention {
            q_off,
            k_off,
            v_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::FusedAttn {
            qkv_off,
            mask_off,
            cos_off,
            sin_off,
            out_off,
            mask_kind,
            has_rope,
            ..
        } => {
            let mut r = vec![*qkv_off];
            if *mask_kind == 2 {
                r.push(*mask_off);
            }
            if *has_rope != 0 {
                r.push(*cos_off);
                r.push(*sin_off);
            }
            (r, vec![*out_off])
        }
        Step::AttentionBackward {
            q_off,
            k_off,
            v_off,
            dy_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off, *dy_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::Rope {
            in_off,
            cos_off,
            sin_off,
            out_off,
            ..
        } => (vec![*in_off, *cos_off, *sin_off], vec![*out_off]),
        Step::GroupedMatmul {
            in_off,
            w_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off, *idx_off], vec![*out_off]),
        Step::ScatterAddZero { out_off, .. } => (vec![], vec![*out_off]),
        Step::ScatterAddAcc {
            upd_off,
            idx_off,
            out_off,
            ..
        } =>
        // out_off is read-modify-write — list it as both a read and
        // a write so the scheduler waits on the prior zero.
        {
            (vec![*upd_off, *idx_off, *out_off], vec![*out_off])
        }
        Step::DequantMatmul {
            x_off,
            w_off,
            scale_off,
            zp_off,
            out_off,
            scheme_id,
            ..
        } => {
            let mut r = vec![*x_off, *w_off, *scale_off];
            if *scheme_id == 1 {
                r.push(*zp_off);
            }
            (r, vec![*out_off])
        }
        Step::DequantMatmulGguf {
            x_byte_off,
            w_byte_off,
            out_byte_off,
            ..
        } => (vec![x_byte_off / 4, w_byte_off / 4], vec![out_byte_off / 4]),
        Step::DequantGroupedMatmulGguf {
            x_byte_off,
            w_byte_off,
            idx_byte_off,
            out_byte_off,
            ..
        } => (
            vec![x_byte_off / 4, w_byte_off / 4, idx_byte_off / 4],
            vec![out_byte_off / 4],
        ),
        Step::SelectiveScan {
            x_off,
            delta_off,
            a_off,
            b_off,
            c_off,
            out_off,
            ..
        } => (
            vec![*x_off, *delta_off, *a_off, *b_off, *c_off],
            vec![*out_off],
        ),
        Step::Fft {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::LogMelHost {
            spec_byte_off,
            filt_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![*spec_byte_off / 4, *filt_byte_off / 4],
            vec![*dst_byte_off / 4],
        ),
        Step::LogMelBackwardHost {
            spec_byte_off,
            filt_byte_off,
            dy_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![*spec_byte_off / 4, *filt_byte_off / 4, *dy_byte_off / 4],
            vec![*dst_byte_off / 4],
        ),
        Step::WelchPeaksHost {
            spec_byte_off,
            dst_byte_off,
            ..
        } => (vec![*spec_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::WelchPeaksGpu {
            spec_off, dst_off, ..
        } => (vec![*spec_off], vec![*dst_off]),
        Step::Im2ColHost {
            x_byte_off,
            col_byte_off,
            ..
        } => (vec![*x_byte_off / 4], vec![*col_byte_off / 4]),
        Step::ReverseHost {
            src_byte_off,
            dst_byte_off,
            ..
        }
        | Step::ArgReduceHost {
            src_byte_off,
            dst_byte_off,
            ..
        }
        | Step::AxialRope2dHost {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::GatedDeltaNet {
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            state_byte_off,
            dst_byte_off,
            use_carry,
            ..
        } => {
            let mut reads = vec![
                q_byte_off / 4,
                k_byte_off / 4,
                v_byte_off / 4,
                g_byte_off / 4,
                beta_byte_off / 4,
            ];
            if *use_carry {
                reads.push(state_byte_off / 4);
            }
            let mut writes = vec![dst_byte_off / 4];
            if *use_carry {
                writes.push(state_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Lstm {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            bias_byte_off,
            h0_byte_off,
            c0_byte_off,
            dst_byte_off,
            carry,
            ..
        } => {
            let mut reads = vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                bias_byte_off / 4,
            ];
            let mut writes = vec![dst_byte_off / 4];
            if *carry {
                // h0/c0 are read and (decode) written back in place.
                reads.push(h0_byte_off / 4);
                reads.push(c0_byte_off / 4);
                writes.push(h0_byte_off / 4);
                writes.push(c0_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Llada2GroupLimitedGate {
            sig_off,
            route_off,
            out_off,
            ..
        } => (vec![*sig_off, *route_off], vec![*out_off]),
        Step::MsDeformAttnHost {
            in_offs, out_off, ..
        } => (in_offs.iter().map(|(o, _)| *o).collect(), vec![*out_off]),
        Step::UmapKnn {
            pairwise_off,
            out_off,
            ..
        } => (vec![*pairwise_off], vec![*out_off]),
        Step::GaussianSplatRender {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            dst_off,
            dst_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![dst_off / 4],
        ),
        Step::GaussianSplatRenderBackward {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            d_loss_off,
            d_loss_len: _,
            packed_off,
            packed_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
                d_loss_off / 4,
            ],
            vec![packed_off / 4],
        ),
        Step::RmsNormBackwardInput {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dx_byte_off / 4],
        ),
        Step::RmsNormBackwardGamma {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dgamma_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dgamma_byte_off / 4],
        ),
        Step::RmsNormBackwardBeta {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dbeta_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dbeta_byte_off / 4],
        ),
        Step::RopeBackward {
            dy_byte_off,
            cos_byte_off,
            sin_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, cos_byte_off / 4, sin_byte_off / 4],
            vec![dx_byte_off / 4],
        ),
        Step::CumsumBackward {
            dy_byte_off,
            dx_byte_off,
            ..
        } => (vec![dy_byte_off / 4], vec![dx_byte_off / 4]),
        Step::GatherBackward {
            dy_byte_off,
            indices_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, indices_byte_off / 4],
            vec![dst_byte_off / 4],
        ),
        Step::MaxPool2dBackward {
            x_byte_off,
            dy_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![*x_byte_off / 4, *dy_byte_off / 4],
            vec![*dx_byte_off / 4],
        ),
        Step::Conv2dBackwardInput {
            dy_byte_off,
            w_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![*dy_byte_off / 4, *w_byte_off / 4],
            vec![*dx_byte_off / 4],
        ),
        Step::Conv2dBackwardWeight {
            x_byte_off,
            dy_byte_off,
            dw_byte_off,
            ..
        } => (
            vec![*x_byte_off / 4, *dy_byte_off / 4],
            vec![*dw_byte_off / 4],
        ),
        Step::Pool1d {
            in_off, out_off, ..
        }
        | Step::Pool2d {
            in_off, out_off, ..
        }
        | Step::Pool3d {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Conv1d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv2d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv3d {
            in_off,
            w_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off], vec![*out_off]),
        Step::LayerNorm2d {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::ConvTranspose2d {
            src_off,
            w_off,
            dst_off,
            ..
        } => (vec![*src_off, *w_off], vec![*dst_off]),
        Step::GroupNorm {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::ResizeNearest2x {
            src_off, dst_off, ..
        } => (vec![*src_off], vec![*dst_off]),
        Step::FusedBinaryUnary {
            a_off,
            b_off,
            out_off,
            ..
        } => (vec![*a_off, *b_off], vec![*out_off]),
        Step::ElementwiseRegion {
            dst_off,
            input_offs,
            num_inputs,
            ..
        } => {
            let n = (*num_inputs as usize).min(input_offs.len());
            (input_offs[..n].to_vec(), vec![*dst_off])
        }
        Step::BatchElementwiseRegion {
            base_dst_off,
            batch_input_offs,
            num_batch,
            ..
        } => {
            let n = (*num_batch as usize).min(64);
            (batch_input_offs[..n].to_vec(), vec![*base_dst_off])
        }
        Step::GaussianSplatPrepare {
            positions_off,
            scales_off,
            rotations_off,
            opacities_off,
            colors_off,
            sh_coeffs_off,
            meta_off,
            prep_off,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![prep_off / 4],
        ),
        Step::GaussianSplatRasterize {
            prep_off,
            meta_off,
            dst_off,
            ..
        } => (vec![prep_off / 4, meta_off / 4], vec![dst_off / 4]),
    }
}

/// Pre-compile every NVRTC kernel against `ctx`. Used by AOT mode to
/// move JIT compile cost out of the first-run critical path. Runs at
/// most once per process — later `CompileMode::Aot` compiles skip it.
static AOT_PREWARM_ONCE: Once = Once::new();

fn prewarm_all(ctx: &Arc<CudaContext>) {
    AOT_PREWARM_ONCE.call_once(|| prewarm_all_kernels(ctx));
}

fn prewarm_all_kernels(ctx: &Arc<CudaContext>) {
    use crate::kernels::*;
    let _ = binary_kernel(ctx);
    let _ = fused_binary_unary_kernel(ctx);
    let _ = unary_kernel(ctx);
    let _ = copy_kernel(ctx);
    let _ = matmul_kernel(ctx);
    let _ = matmul_epilogue_kernel(ctx);
    let _ = compare_kernel(ctx);
    let _ = where_kernel(ctx);
    let _ = reduce_kernel(ctx);
    let _ = softmax_kernel(ctx);
    let _ = layernorm_kernel(ctx);
    let _ = fused_residual_ln_kernel(ctx);
    let _ = fused_residual_rms_norm_kernel(ctx);
    let _ = gather_kernel(ctx);
    let _ = gather_axis_kernel(ctx);
    let _ = narrow_kernel(ctx);
    let _ = concat_kernel(ctx);
    let _ = transpose_kernel(ctx);
    let _ = expand_kernel(ctx);
    let _ = attention_kernel(ctx);
    let _ = attention_row_kernel(ctx);
    let _ = attention_bwd_kernel(ctx);
    let _ = argmax_kernel(ctx);
    let _ = rope_kernel(ctx);
    let _ = cumsum_kernel(ctx);
    let _ = topk_kernel(ctx);
    let _ = grouped_matmul_kernel(ctx);
    let _ = scatter_add_zero_kernel(ctx);
    let _ = scatter_add_acc_kernel(ctx);
    let _ = dequant_matmul_kernel(ctx);
    let _ = dequant_matmul_gguf_kernel(ctx);
    let _ = dequant_gguf_kernel(ctx);
    let _ = sample_kernel(ctx);
    let _ = selective_scan_kernel(ctx);
    let _ = pool1d_kernel(ctx);
    let _ = pool2d_kernel(ctx);
    let _ = pool3d_kernel(ctx);
    let _ = conv1d_kernel(ctx);
    let _ = conv2d_kernel(ctx);
    let _ = im2col_kernel(ctx);
    let _ = conv3d_kernel(ctx);
    let _ = layer_norm2d_kernel(ctx);
    let _ = conv_transpose2d_kernel(ctx);
    let _ = group_norm_kernel(ctx);
    let _ = resize_nearest_2x_kernel(ctx);
    let _ = elementwise_region_kernel(ctx);
    let _ = batch_elementwise_region_kernel(ctx);
    // matmul_wmma deliberately excluded: requires SM 70+ and may fail
    // load_module on older GPUs. Compile lazily on first opt-in dispatch.
}

fn im2col_use_gpu(n: u32, exec_mode: ExecMode) -> bool {
    if rlx_ir::env::var("RLX_CUDA_IM2COL_HOST").is_some() {
        return false;
    }
    if matches!(exec_mode, ExecMode::Graph) {
        return n > 0;
    }
    n > 0
}

fn pinned_host_io_disabled() -> bool {
    rlx_ir::env::var("RLX_CUDA_PINNED_IO").is_some_and(|v| v.eq_ignore_ascii_case("0"))
}

/// Pinned host output staging (faster D2H). On by default; set `RLX_CUDA_PINNED_IO=0` to disable.
fn pinned_output_staging_enabled() -> bool {
    !pinned_host_io_disabled()
}

/// Pinned host input staging for H2D. Graph mode always; stream mode when `RLX_CUDA_PINNED_IO=1`.
fn pinned_input_staging_enabled(exec_mode: ExecMode) -> bool {
    if pinned_host_io_disabled() {
        return false;
    }
    matches!(exec_mode, ExecMode::Graph)
        || rlx_ir::env::var("RLX_CUDA_PINNED_IO").is_some_and(|v| !v.eq_ignore_ascii_case("0"))
}

fn normalize_read_indices(buf: &mut Vec<usize>) {
    if buf.len() > 1 {
        buf.sort_unstable();
        buf.dedup();
    }
}

fn compile_mode_from_env() -> CompileMode {
    match rlx_ir::env::var("RLX_CUDA_COMPILE_MODE").as_deref() {
        Some(mode) if mode.eq_ignore_ascii_case("aot") => CompileMode::Aot,
        _ => CompileMode::Jit,
    }
}

fn exec_mode_from_env() -> ExecMode {
    match rlx_ir::env::var("RLX_CUDA_EXEC_MODE").as_deref() {
        Some(mode) if mode.eq_ignore_ascii_case("graph") => ExecMode::Graph,
        Some(mode) => {
            let lower = mode.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("multistream") {
                let n = rest.trim_start_matches([':', '=']).parse().unwrap_or(2);
                ExecMode::MultiStream(n.max(1))
            } else {
                ExecMode::Stream
            }
        }
        _ => ExecMode::Stream,
    }
}

impl CudaExecutable {
    /// JIT compile, stream-mode execution. Default entry point.
    ///
    /// Honors `RLX_CUDA_COMPILE_MODE=aot` and `RLX_CUDA_EXEC_MODE=graph|multistream:N`.
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with_rng(
            graph,
            compile_mode_from_env(),
            exec_mode_from_env(),
            rlx_ir::RngOptions::default(),
        )
    }

    /// Compile with explicit RNG policy and env-selected compile/exec modes.
    pub fn compile_rng(graph: Graph, rng: rlx_ir::RngOptions) -> Self {
        Self::compile_with_rng(graph, compile_mode_from_env(), exec_mode_from_env(), rng)
    }

    /// Compile with explicit RNG policy (used by [`rlx-runtime`]).
    pub fn compile_with_rng(
        graph: Graph,
        compile_mode: CompileMode,
        exec_mode: ExecMode,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        let ctx = cuda_context().expect("rlx-cuda: no CUDA driver available");

        if compile_mode == CompileMode::Aot {
            prewarm_all(&ctx);
        }

        // Decompose composed ops we don't yet have native kernels for
        // (FusedMatMulBiasAct, canonical DotGeneral) into primitives
        // before memory planning. Fusion may reintroduce mid-axis Reduce
        // (e.g. EEG temporal mean); CUDA only schedules last-axis Reduce.
        let graph = LowerNonLastAxisReduce.run(crate::unfuse::unfuse(graph));

        let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
        // Native `Op::FusedAttentionBlock`: per-block packed-QKV + attn scratch
        // (the projections are GEMMs into these buffers, read by the
        // `fused_attn_block` kernel). Empty when every FAB was decomposed.
        let (fab_scratch_bytes, fab_scratch_map) = fab_scratch_plan(&graph);
        let mut plan = plan_f32_uniform(&graph, 16);
        let dequant_scratch_off = if dequant_scratch > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            plan.arena_size = aligned + dequant_scratch;
            aligned
        } else {
            0
        };
        let fab_scratch_off = if fab_scratch_bytes > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            plan.arena_size = aligned + fab_scratch_bytes;
            aligned
        } else {
            0
        };
        let fab_scratch_base_f32 = (fab_scratch_off / 4) as u32;
        if rlx_ir::env::flag("RLX_CUDA_ARENA_DEBUG") {
            eprintln!(
                "[cuda-arena] plan.arena_size={:.3} GiB (dequant_scratch={:.3} GiB, fab_scratch={:.3} GiB)",
                plan.arena_size as f64 / (1u64 << 30) as f64,
                dequant_scratch as f64 / (1u64 << 30) as f64,
                fab_scratch_bytes as f64 / (1u64 << 30) as f64,
            );
        }
        let mut arena = Arena::from_plan(&ctx, &plan);
        for node in graph.nodes() {
            let slot_bytes = node
                .shape
                .size_bytes()
                .unwrap_or_else(|| node.shape.num_elements().unwrap_or(0) * 4);
            arena.set_actual_len(node.id, slot_bytes);
        }

        // Initial param/input offset maps for fast lookup at run time.
        let mut input_offsets = HashMap::new();
        let mut param_offsets = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_offsets.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_offsets.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        // Initialise Constants directly into the arena.
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                // The arena is f32; widen the constant to f32 by its declared
                // dtype (e.g. an i64 `arange` constant would otherwise be read
                // as garbage f32 bit-for-bit).
                let f32_data: Vec<f32> = match node.shape.dtype() {
                    rlx_ir::DType::F32 => data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    rlx_ir::DType::F64 => data
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    rlx_ir::DType::I64 => data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    rlx_ir::DType::I32 | rlx_ir::DType::U32 => data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect(),
                    rlx_ir::DType::I16 => data
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                        .collect(),
                    rlx_ir::DType::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
                    rlx_ir::DType::U8 | rlx_ir::DType::Bool => {
                        data.iter().map(|&b| b as f32).collect()
                    }
                    // f16/bf16/c64: raw bytes already narrower/complex — keep the
                    // bit-reinterpret path.
                    _ => data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                };
                let n_f32 = f32_data.len().min(arena.len_of(node.id) / 4);
                let off_f32 = arena.offset(node.id) / 4;
                let stream = ctx.default_stream();
                let mut slot = arena.f32_buf_mut().slice_mut(off_f32..off_f32 + n_f32);
                stream
                    .memcpy_htod(&f32_data[..n_f32], &mut slot)
                    .expect("rlx-cuda: constant upload failed");
            }
        }

        let mut schedule = Vec::new();
        let mut meta_buffers: Vec<cudarc::driver::CudaSlice<u32>> = Vec::new();
        let mut packed_bshd_attn: HashMap<NodeId, (NodeId, u32)> = HashMap::new();
        if !rlx_ir::env::flag("RLX_CUDA_NO_PACKED_BSHD_ATTN") {
            for node in graph.nodes() {
                let Op::Attention { .. } = &node.op else {
                    continue;
                };
                if node.inputs.len() < 3 {
                    continue;
                }
                if let Some((parent, head_width, _)) = rlx_ir::detect_packed_bshd_qkv_attention(
                    &graph,
                    node.inputs[0],
                    node.inputs[1],
                    node.inputs[2],
                ) {
                    packed_bshd_attn.insert(node.id, (parent, head_width as u32));
                }
            }
        }
        // #6 real→complex fusion analysis (native-cuda-fft): find forward FFTs
        // whose input is `Concat([signal, Sub(x,x)])` (a real signal zero-padded
        // to the 2N complex block), where the Concat and the zeros are
        // single-use. Such FFTs read `signal` directly (im=0), and the Concat +
        // Sub are dropped from the schedule — eliminating two memory-bound
        // kernels that together cost more than the FFT itself. Conservative:
        // only fuses stockham-eligible sizes; `RLX_FFT_FUSE_REAL=0` disables.
        #[cfg(feature = "native-cuda-fft")]
        let (fft_real_skip, fft_real_src) = {
            let mut skip: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
            let mut srcmap: HashMap<NodeId, NodeId> = HashMap::new();
            let fuse = !rlx_ir::env::var("RLX_FFT_FUSE_REAL")
                .is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("off"));
            if fuse {
                let mut uses: HashMap<NodeId, u32> = HashMap::new();
                for node in graph.nodes() {
                    for &inp in &node.inputs {
                        *uses.entry(inp).or_insert(0) += 1;
                    }
                }
                for node in graph.nodes() {
                    let Op::Fft { inverse: false, .. } = &node.op else {
                        continue;
                    };
                    let meta = rlx_ir::fft::fft_meta(&graph.node(node.inputs[0]).shape);
                    if !crate::native_fft_dispatch::stockham_eligible(meta.n_complex as u32) {
                        continue;
                    }
                    let concat_id = node.inputs[0];
                    let cnode = graph.node(concat_id);
                    let Op::Concat { axis } = &cnode.op else {
                        continue;
                    };
                    if cnode.inputs.len() != 2
                        || *axis != cnode.shape.rank() - 1
                        || uses.get(&concat_id) != Some(&1)
                    {
                        continue;
                    }
                    let (x_id, z_id) = (cnode.inputs[0], cnode.inputs[1]);
                    let znode = graph.node(z_id);
                    let is_zeros = matches!(&znode.op, Op::Binary(BinaryOp::Sub))
                        && znode.inputs.len() == 2
                        && znode.inputs[0] == znode.inputs[1];
                    // The FFT now reads `signal` at a point the arena planned its
                    // liveness only up to the (skipped) Concat — so require it to
                    // be a resident Input/Param, whose region is never aliased
                    // away mid-run. (Covers the real-FFT graph; conservative.)
                    let xnode = graph.node(x_id);
                    let x_resident = matches!(&xnode.op, Op::Input { .. } | Op::Param { .. });
                    // signal must hold exactly the `n` real values the FFT consumes.
                    let x_ok = x_resident
                        && xnode.shape.dim(xnode.shape.rank() - 1).unwrap_static()
                            == meta.n_complex;
                    if is_zeros && uses.get(&z_id) == Some(&1) && x_ok {
                        skip.insert(concat_id);
                        skip.insert(z_id);
                        srcmap.insert(node.id, x_id);
                    }
                }
            }
            (skip, srcmap)
        };

        for node in graph.nodes() {
            #[cfg(feature = "native-cuda-fft")]
            if fft_real_skip.contains(&node.id) {
                continue; // fused into the real-input FFT below
            }
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::Reshape { .. } | Op::Cast { .. } | Op::StopGradient => {
                    // No-op: arena.plan_f32_uniform already aliased the
                    // output slot to the input. The same row-major bytes
                    // are visible under the new node ID. StopGradient is a
                    // pure identity in the forward pass (the AD pass already
                    // consumed its gradient-blocking semantics upstream).
                }
                Op::ScaledMatMul {
                    lhs_format,
                    rhs_format,
                    scale_layout,
                    has_bias,
                } => {
                    let out_dims = node.shape.dims();
                    let m = out_dims[0].unwrap_static() as u32;
                    let n = out_dims[1].unwrap_static() as u32;
                    let k = graph.node(node.inputs[0]).shape.dims()[1].unwrap_static() as u32;
                    let bias_byte = if *has_bias {
                        arena.offset(node.inputs[4]) as u32
                    } else {
                        0
                    };
                    let native = lhs_format.is_native_fp8()
                        && rhs_format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor);
                    if native {
                        // Per-tensor FP8 → native cublasLt tensor-core GEMM.
                        schedule.push(Step::ScaledMatMul {
                            m,
                            k,
                            n,
                            lhs_byte_off: arena.offset(node.inputs[0]) as u32,
                            rhs_byte_off: arena.offset(node.inputs[1]) as u32,
                            lhs_scale_byte_off: arena.offset(node.inputs[2]) as u32,
                            rhs_scale_byte_off: arena.offset(node.inputs[3]) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            has_bias: u32::from(*has_bias),
                            bias_byte_off: bias_byte,
                            lhs_e5m2: u32::from(*lhs_format == rlx_ir::ScaledFormat::F8E5M2),
                            rhs_e5m2: u32::from(*rhs_format == rlx_ir::ScaledFormat::F8E5M2),
                        });
                    } else {
                        // Block / FP4 / FP6 → on-device decode-and-accumulate.
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledMatMulDecode {
                            m,
                            k,
                            n,
                            lhs_byte_off: arena.offset(node.inputs[0]) as u32,
                            rhs_byte_off: arena.offset(node.inputs[1]) as u32,
                            lhs_scale_byte_off: arena.offset(node.inputs[2]) as u32,
                            rhs_scale_byte_off: arena.offset(node.inputs[3]) as u32,
                            out_off_f32: (arena.offset(node.id) / 4) as u32,
                            lhs_fmt: lhs_format.kernel_id(),
                            rhs_fmt: rhs_format.kernel_id(),
                            scale_mode,
                            block,
                            has_bias: u32::from(*has_bias),
                            bias_off_f32: bias_byte / 4,
                        });
                    }
                }
                Op::ScaledQuantScale {
                    format,
                    scale_layout,
                } => {
                    let x_id = node.inputs[0];
                    if format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor)
                    {
                        let n = graph.node(x_id).shape.num_elements().unwrap() as u32;
                        schedule.push(Step::ScaledQuantScale {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_off_f32: (arena.offset(node.id) / 4) as u32,
                            n,
                            max_finite: format.max_finite(),
                        });
                    } else {
                        let xs = graph.node(x_id).shape.dims();
                        let cols = xs[xs.len() - 1].unwrap_static() as u32;
                        let rows =
                            graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledQuantScaleGeneral {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_byte_off: arena.offset(node.id) as u32,
                            rows,
                            cols,
                            fmt: format.kernel_id(),
                            scale_mode,
                            block,
                        });
                    }
                }
                Op::ScaledQuantize {
                    format,
                    scale_layout,
                } => {
                    let x_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    if format.is_native_fp8()
                        && matches!(scale_layout, rlx_ir::ScaleLayout::PerTensor)
                    {
                        let n = graph.node(x_id).shape.num_elements().unwrap() as u32;
                        schedule.push(Step::ScaledQuantizeFp8 {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_off_f32: (arena.offset(scale_id) / 4) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            n,
                            e5m2: u32::from(*format == rlx_ir::ScaledFormat::F8E5M2),
                        });
                    } else {
                        let xs = graph.node(x_id).shape.dims();
                        let cols = xs[xs.len() - 1].unwrap_static() as u32;
                        let rows =
                            graph.node(x_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                        let (scale_mode, block) = scale_layout.mode_block();
                        schedule.push(Step::ScaledQuantizeGeneral {
                            x_off_f32: (arena.offset(x_id) / 4) as u32,
                            scale_byte_off: arena.offset(scale_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            rows,
                            cols,
                            fmt: format.kernel_id(),
                            scale_mode,
                            block,
                        });
                    }
                }
                Op::ScaledDequantize {
                    format,
                    scale_layout,
                } => {
                    // codes (U8, input 0) + scale (input 1) → f32. Logical shape
                    // follows the codes. One general kernel covers all layouts.
                    let codes_id = node.inputs[0];
                    let scale_id = node.inputs[1];
                    let xs = graph.node(codes_id).shape.dims();
                    let cols = xs[xs.len() - 1].unwrap_static() as u32;
                    let rows =
                        graph.node(codes_id).shape.num_elements().unwrap() as u32 / cols.max(1);
                    let (scale_mode, block) = scale_layout.mode_block();
                    schedule.push(Step::ScaledDequantizeGeneral {
                        codes_byte_off: arena.offset(codes_id) as u32,
                        scale_byte_off: arena.offset(scale_id) as u32,
                        out_off_f32: (arena.offset(node.id) / 4) as u32,
                        rows,
                        cols,
                        fmt: format.kernel_id(),
                        scale_mode,
                        block,
                    });
                }
                Op::MatMul => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "MatMul");
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32: (arena.offset(a_id) / 4) as u32,
                        b_off_f32: (arena.offset(b_id) / 4) as u32,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: 0,
                        bias_off_f32: 0,
                        act_id: 0xFFFF,
                    });
                }
                Op::FusedMatMulBiasAct { activation } => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "FusedMatMulBiasAct");
                    let bias_id = node.inputs[2];
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32: (arena.offset(a_id) / 4) as u32,
                        b_off_f32: (arena.offset(b_id) / 4) as u32,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: 1,
                        bias_off_f32: (arena.offset(bias_id) / 4) as u32,
                        act_id,
                    });
                }
                Op::Binary(bop) => {
                    schedule.push(Step::Binary {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: binary_op_id(*bop),
                    });
                }
                Op::Activation(act) => {
                    schedule.push(Step::Unary {
                        n: elems,
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: activation_op_id(*act),
                    });
                }
                Op::Compare(cop) => {
                    schedule.push(Step::Compare {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: compare_op_id(*cop),
                    });
                }
                Op::Where => {
                    schedule.push(Step::Where {
                        n: elems,
                        cond_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        x_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        y_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::BatchElementwiseRegion {
                    chain,
                    num_batch_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    let n = *num_batch_inputs as usize;
                    if n == 0 || chain.len() > 32 {
                        panic!(
                            "rlx-cuda BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                            chain.len()
                        );
                    }
                    let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                    let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                        .expect("batch region static shape");
                    let base_dst_off = (arena.offset(node.id) / 4) as u32;
                    let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                    if use_single {
                        let mut batch_input_offs = [0u32; 64];
                        for i in 0..n {
                            batch_input_offs[i] = (arena.offset(node.inputs[i]) / 4) as u32;
                        }
                        let input_offs_meta = [0u32; 16];
                        let meta_arr = rlx_ir::encode_elementwise_region_meta(
                            &input_offs_meta,
                            chain,
                            *prologue,
                            &slice_shape,
                            *prologue_input,
                        );
                        let meta = ctx
                            .default_stream()
                            .clone_htod(&meta_arr.to_vec())
                            .expect("rlx-cuda: batch elementwise_region meta upload failed");
                        let meta_idx = meta_buffers.len();
                        meta_buffers.push(meta);
                        let batch_vec: Vec<u32> = batch_input_offs[..n].to_vec();
                        let batch_dev = ctx
                            .default_stream()
                            .clone_htod(&batch_vec)
                            .expect("rlx-cuda: batch input offs upload failed");
                        let batch_offs_idx = meta_buffers.len();
                        meta_buffers.push(batch_dev);
                        schedule.push(Step::BatchElementwiseRegion {
                            slice_len: slice_elems,
                            num_batch: n as u32,
                            num_steps: chain.len() as u32,
                            base_dst_off,
                            slice_elems,
                            batch_input_offs,
                            batch_offs_idx,
                            meta_idx,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                        });
                    } else {
                        for i in 0..n {
                            let mut input_offs = [0u32; 16];
                            input_offs[0] = (arena.offset(node.inputs[i]) / 4) as u32;
                            let meta_arr = rlx_ir::encode_elementwise_region_meta(
                                &input_offs,
                                chain,
                                *prologue,
                                &slice_shape,
                                *prologue_input,
                            );
                            let meta = ctx
                                .default_stream()
                                .clone_htod(&meta_arr.to_vec())
                                .expect("rlx-cuda: batch elementwise_region meta upload failed");
                            let meta_idx = meta_buffers.len();
                            meta_buffers.push(meta);
                            let spatial =
                                matches!(*prologue, rlx_ir::RegionPrologue::ResizeNearest2x);
                            let grid = rlx_ir::PrologueLaunchGrid::from_output_shape(&slice_shape);
                            schedule.push(Step::ElementwiseRegion {
                                len: slice_elems,
                                num_inputs: 1,
                                num_steps: chain.len() as u32,
                                dst_off: rlx_ir::batch_region_slice_dst_off_f32(
                                    base_dst_off,
                                    slice_elems,
                                    i,
                                ),
                                input_offs,
                                scalar_input_mask: *scalar_input_mask,
                                input_modulus: *input_modulus,
                                meta_idx,
                                spatial_prologue: spatial,
                                prologue_w: grid.map(|g| g.width).unwrap_or(0),
                                prologue_h: grid.map(|g| g.height).unwrap_or(0),
                                prologue_nc: grid.map(|g| g.depth).unwrap_or(0),
                            });
                        }
                    }
                }
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // 72-u32 metadata buffer (8 input offsets + 16 steps *
                    // 4 u32s) uploaded once at compile time; the kernel
                    // walks the chain interpretively in registers. Caps
                    // match the cross-backend Metal MSL / wgpu WGSL
                    // encoders.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-cuda ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Run UnfuseElementwiseRegions to fall back \
                                to atomic ops.",
                            chain.len()
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = (arena.offset(id) / 4) as u32;
                    }
                    let meta_arr = rlx_ir::encode_elementwise_region_meta(
                        &input_offs,
                        chain,
                        *prologue,
                        &node.shape,
                        *prologue_input,
                    );
                    let meta_data: Vec<u32> = meta_arr.to_vec();
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: elementwise_region meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    let spatial = matches!(*prologue, rlx_ir::RegionPrologue::ResizeNearest2x);
                    let grid = rlx_ir::PrologueLaunchGrid::from_output_shape(&node.shape);
                    schedule.push(Step::ElementwiseRegion {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        input_offs,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                        meta_idx,
                        spatial_prologue: spatial,
                        prologue_w: grid.map(|g| g.width).unwrap_or(0),
                        prologue_h: grid.map(|g| g.height).unwrap_or(0),
                        prologue_nc: grid.map(|g| g.depth).unwrap_or(0),
                    });
                }
                Op::Reduce {
                    op,
                    axes,
                    keep_dim: _,
                } => {
                    // v2: reduce along the LAST axis only — same v1
                    // simplification rlx-wgpu had.
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    if axes.len() != 1 || axes[0] != in_dims.len() - 1 {
                        panic!(
                            "rlx-cuda Reduce: only single last-axis supported \
                                (got axes={axes:?}, rank={})",
                            in_dims.len()
                        );
                    }
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Reduce {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: reduce_op_id(*op),
                    });
                }
                Op::Softmax { axis: _ } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Softmax {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let is_layer = matches!(&node.op, Op::LayerNorm { .. });
                    let gamma_id = node.inputs[1];
                    let beta_id = if is_layer && node.inputs.len() >= 3 {
                        node.inputs[2]
                    } else {
                        gamma_id
                    };
                    schedule.push(Step::LayerNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        gamma_off: (arena.offset(gamma_id) / 4) as u32,
                        beta_off: (arena.offset(beta_id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        op: if is_layer { 0 } else { 1 },
                    });
                }
                Op::FusedResidualLN { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::FusedResidualLn {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                    });
                }
                Op::FusedResidualRmsNorm { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::FusedResidualRmsNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                    });
                }
                Op::Gather { axis } => {
                    let table_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    if *axis == 0 {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let vocab = table_shape[0].unwrap_static() as u32;
                        let dim: u32 = table_shape[1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let n_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        schedule.push(Step::Gather {
                            n_out: elems,
                            n_idx,
                            dim,
                            vocab,
                            in_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    } else {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let outer: u32 = table_shape[..*axis]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let trailing: u32 = table_shape[*axis + 1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let axis_dim = table_shape[*axis].unwrap_static() as u32;
                        let num_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let total = outer * num_idx * trailing;
                        schedule.push(Step::GatherAxis {
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    }
                }
                Op::Narrow { axis, start, len } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let outer: u32 = in_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = in_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_in = in_dims[*axis].unwrap_static() as u32;
                    schedule.push(Step::Narrow {
                        total: elems,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Transpose { perm } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let rank = perm.len();
                    let in_dims_u: Vec<u32> =
                        in_dims.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Cumulative input strides (row-major, innermost = 1).
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims_u[i + 1];
                    }
                    let out_dims_u: Vec<u32> = perm.iter().map(|&i| in_dims_u[i]).collect();
                    let strides_for_out: Vec<u32> = perm.iter().map(|&i| in_strides[i]).collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims_u);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Transpose {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let rank = target_shape.len();
                    if rank < in_shape.len() {
                        panic!(
                            "rlx-cuda Expand: cannot reduce rank (in={}, target={})",
                            in_shape.len(),
                            rank
                        );
                    }
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let pad = rank - in_shape.len();
                    let mut in_dims: Vec<u32> = vec![1; pad];
                    in_dims.extend(in_shape.iter().map(|d| d.unwrap_static() as u32));
                    let mut in_strides_row = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides_row[i] = in_strides_row[i + 1] * in_dims[i + 1];
                    }
                    let strides_for_out: Vec<u32> = (0..rank)
                        .map(|i| {
                            if in_dims[i] == 1 && out_dims[i] != 1 {
                                0
                            } else {
                                in_strides_row[i]
                            }
                        })
                        .collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Expand {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Concat { axis } => {
                    // Caller convention: one Step::Concat per input, copying
                    // each input's slice into the output at the right axis offset.
                    let mut start: u32 = 0;
                    let out_dims = node.shape.dims();
                    let outer: u32 = out_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = out_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_out_size = out_dims[*axis].unwrap_static() as u32;
                    for &in_id in &node.inputs {
                        let in_dims = graph.node(in_id).shape.dims();
                        let axis_in = in_dims[*axis].unwrap_static() as u32;
                        let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                        schedule.push(Step::Concat {
                            total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size,
                            start,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                        start += axis_in;
                    }
                }
                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } => {
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-cuda Attention: unfuse should have promoted to rank-4");
                    }
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let batch = geom.batch as u32;
                    let heads = geom.heads as u32;
                    let seq_q = geom.seq_q as u32;
                    let seq_k = geom.seq_k as u32;
                    let hd = *head_dim as u32;
                    // Honor the op's score_scale (Gemma pre-scales Q and passes 1.0;
                    // Gemma 4 passes unit scale). Only fall back to head_dim^-0.5 when
                    // unset — matches rlx-cpu executor.rs and rlx-wgpu backend.rs.
                    let scale = score_scale.unwrap_or(1.0_f32 / (hd as f32).sqrt());
                    // Gemma 2 attention logit soft-cap (0 = disabled). Applied
                    // pre-mask in the kernel; matches rlx-cpu executor.rs.
                    let softcap_bits = attn_logit_softcap.unwrap_or(0.0).to_bits();
                    let mask_shape = if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
                        Some(graph.node(node.inputs[3]).shape.dims())
                    } else {
                        None
                    };
                    let packed_parent = packed_bshd_attn.get(&node.id).copied();
                    let st = if let Some((_, head_width)) = packed_parent {
                        let (qb, qh, qs) =
                            rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
                        let (ob, oh, os) =
                            rlx_ir::strides_for_shape(node.shape.dims(), heads, hd, seq_q, false);
                        let (mb, mh, mq, mk) = mask_shape
                            .map(|m| rlx_ir::mask_strides_for_shape(m, heads, seq_q, seq_k))
                            .unwrap_or_else(|| rlx_ir::mask_strides_bhsd(heads, seq_q, seq_k));
                        rlx_ir::AttentionLaunchStrides {
                            q_batch: qb,
                            q_head: qh,
                            q_seq: qs,
                            k_batch: qb,
                            k_head: qh,
                            k_seq: qs,
                            v_batch: qb,
                            v_head: qh,
                            v_seq: qs,
                            o_batch: ob,
                            o_head: oh,
                            o_seq: os,
                            mask_batch: mb,
                            mask_head: mh,
                            mask_q: mq,
                            mask_k: mk,
                        }
                    } else {
                        rlx_ir::attention_launch_strides(
                            geom,
                            q_shape,
                            k_shape,
                            graph.node(v_id).shape.dims(),
                            node.shape.dims(),
                            mask_shape,
                        )
                    };
                    let (q_off, k_off, v_off) = if let Some((parent, head_width)) = packed_parent {
                        let p = (arena.offset(parent) / 4) as u32;
                        (
                            p,
                            p.saturating_add(head_width),
                            p.saturating_add(head_width * 2),
                        )
                    } else {
                        (
                            (arena.offset(q_id) / 4) as u32,
                            (arena.offset(k_id) / 4) as u32,
                            (arena.offset(v_id) / 4) as u32,
                        )
                    };
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                    };
                    schedule.push(Step::Attention {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off,
                        k_off,
                        v_off,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        softcap_bits,
                        window,
                        seq_q_stride: st.mask_q,
                        seq_k_stride: st.mask_k,
                        mask_batch_stride: st.mask_batch,
                        mask_head_stride: st.mask_head,
                        q_batch_stride: st.q_batch,
                        q_head_stride: st.q_head,
                        q_seq_stride: st.q_seq,
                        k_batch_stride: st.k_batch,
                        k_head_stride: st.k_head,
                        k_seq_stride: st.k_seq,
                        v_batch_stride: st.v_batch,
                        v_head_stride: st.v_head,
                        v_seq_stride: st.v_seq,
                        o_batch_stride: st.o_batch,
                        o_head_stride: st.o_head,
                        o_seq_stride: st.o_seq,
                    });
                }
                Op::FusedAttentionBlock {
                    num_heads,
                    head_dim,
                    has_bias,
                    has_rope,
                } => {
                    // Native lowering (the unfuse pass only keeps FAB nodes
                    // the `fused_attn_block` kernel can serve): two GEMMs into
                    // packed scratch around the fused RoPE+SDPA kernel.
                    //   1. qkv = hidden @ qkv_w [+ qkv_b]   → qkv scratch [B,S,3I]
                    //   2. attn = fused_attn(qkv, mask, cos, sin) → attn scratch [B,S,I]
                    //   3. out  = attn @ out_w [+ out_b]    → node output [B,S,I]
                    let nh = *num_heads as u32;
                    let hd = *head_dim as u32;
                    let inner = (*num_heads * *head_dim) as u32;
                    let dims = node.shape.dims();
                    let b = dims[0].unwrap_static() as u32;
                    let s = dims[1].unwrap_static() as u32;
                    let m = b * s;

                    if rlx_ir::env::flag("RLX_CUDA_TRACE_FAB") {
                        eprintln!(
                            "[rlx-cuda] native fused_attn_block: b={b} s={s} heads={nh} \
                             head_dim={hd} rope={has_rope} bias={has_bias}"
                        );
                    }
                    let (qkv_rel, attn_rel) = *fab_scratch_map
                        .get(&node.id)
                        .expect("rlx-cuda: FusedAttentionBlock scratch offset missing");
                    let qkv_off = fab_scratch_base_f32 + qkv_rel;
                    let attn_off = fab_scratch_base_f32 + attn_rel;

                    let hidden_off = (arena.offset(node.inputs[0]) / 4) as u32;
                    let qkv_w_off = (arena.offset(node.inputs[1]) / 4) as u32;
                    let out_w_off = (arena.offset(node.inputs[2]) / 4) as u32;
                    let mask_off = (arena.offset(node.inputs[3]) / 4) as u32;
                    let mut next = 4usize;
                    let (qkv_b_off, out_b_off) = if *has_bias {
                        let q = (arena.offset(node.inputs[next]) / 4) as u32;
                        let o = (arena.offset(node.inputs[next + 1]) / 4) as u32;
                        next += 2;
                        (q, o)
                    } else {
                        (0u32, 0u32)
                    };
                    let (cos_off, sin_off) = if *has_rope {
                        let c = (arena.offset(node.inputs[next]) / 4) as u32;
                        let si = (arena.offset(node.inputs[next + 1]) / 4) as u32;
                        (c, si)
                    } else {
                        (0u32, 0u32)
                    };

                    let bias_flag = u32::from(*has_bias);
                    schedule.push(Step::Matmul {
                        m,
                        k: inner,
                        n: 3 * inner,
                        batch: 1,
                        a_batch_stride: 0,
                        b_batch_stride: 0,
                        c_batch_stride: 0,
                        a_off_f32: hidden_off,
                        b_off_f32: qkv_w_off,
                        c_off_f32: qkv_off,
                        has_bias: bias_flag,
                        bias_off_f32: qkv_b_off,
                        act_id: 0xFFFF,
                    });
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    schedule.push(Step::FusedAttn {
                        qkv_off,
                        mask_off,
                        cos_off,
                        sin_off,
                        out_off: attn_off,
                        batch: b,
                        seq: s,
                        heads: nh,
                        head_dim: hd,
                        mask_kind: 2, // Custom binary [B,S] — the only FAB mask
                        scale_bits: scale.to_bits(),
                        has_rope: u32::from(*has_rope),
                    });
                    schedule.push(Step::Matmul {
                        m,
                        k: inner,
                        n: inner,
                        batch: 1,
                        a_batch_stride: 0,
                        b_batch_stride: 0,
                        c_batch_stride: 0,
                        a_off_f32: attn_off,
                        b_off_f32: out_w_off,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: bias_flag,
                        bias_off_f32: out_b_off,
                        act_id: 0xFFFF,
                    });
                }
                Op::AttentionBackward {
                    num_heads: _,
                    head_dim,
                    mask_kind,
                    wrt,
                } => {
                    use rlx_ir::op::AttentionBwdWrt;
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-cuda AttentionBackward: unfuse should have promoted to rank-4");
                    }
                    let batch = q_shape[0].unwrap_static() as u32;
                    let heads = q_shape[1].unwrap_static() as u32;
                    let seq_q = q_shape[2].unwrap_static() as u32;
                    let seq_k = k_shape[2].unwrap_static() as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                    };
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    schedule.push(Step::AttentionBackward {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        wrt: wrt_id,
                    });
                }
                Op::Rope {
                    head_dim,
                    n_rot,
                    style,
                } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-cuda Rope: last_dim {} not multiple of head_dim {}",
                            last, head_dim
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-cuda Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    let interleaved = match style {
                        rlx_ir::op::RopeStyle::NeoX => 0u32,
                        rlx_ir::op::RopeStyle::GptJ => 1u32,
                    };
                    schedule.push(Step::Rope {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        // Partial rotary: rotate only n_rot dims (Gemma 4 global
                        // layers use n_rot < head_dim). Equals half for full rope.
                        rot_half: (*n_rot / 2) as u32,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        cos_off: (arena.offset(cos_id) / 4) as u32,
                        sin_off: (arena.offset(sin_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        last_dim: last as u32,
                        interleaved,
                    });
                }
                Op::Cumsum { axis: _, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Cumsum {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                    });
                }
                Op::TopK { k } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::TopK {
                        outer,
                        inner,
                        k: *k as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::GroupedMatMul => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = w_dims[2].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    schedule.push(Step::GroupedMatmul {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::DequantGroupedMatMul { scheme } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let slab_bytes = (k as usize * n as usize) / block_elems * block_bytes;
                    let total_bytes = graph.node(w_id).shape.num_elements().unwrap();
                    let ne = (total_bytes / slab_bytes.max(1)) as u32;
                    schedule.push(Step::DequantGroupedMatmulGguf {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                        x_byte_off: arena.offset(in_id) as u32,
                        w_byte_off: arena.offset(w_id) as u32,
                        idx_byte_off: arena.offset(idx_id) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                    });
                }
                Op::ScatterAdd => {
                    let upd_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    let upd_dims = graph.node(upd_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let num_updates = upd_dims[0].unwrap_static() as u32;
                    let trailing: u32 = upd_dims
                        .iter()
                        .skip(1)
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let out_dim = out_dims[0].unwrap_static() as u32;
                    let out_total = out_dim * trailing;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    schedule.push(Step::ScatterAddZero { out_off, out_total });
                    schedule.push(Step::ScatterAddAcc {
                        out_off,
                        upd_off: (arena.offset(upd_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        num_updates,
                        trailing,
                        out_dim,
                    });
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::quant::QuantScheme;
                    let x_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    // Rank-agnostic GEMM dims (mirrors rlx-wgpu / rlx-metal thunk
                    // lowering). A 2D-only `out_dims[1]` read collapses 3D decode
                    // output `[1, 1, hidden]` to `n = 1` and breaks GGUF dequant.
                    let out_total = node.shape.num_elements().unwrap_or(0) as u32;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u32;
                    let m = out_total / n.max(1);
                    let x_total = graph.node(x_id).shape.num_elements().unwrap_or(0) as u32;
                    let k = x_total / m.max(1);
                    if scheme.is_gguf() {
                        schedule.push(Step::DequantMatmulGguf {
                            m,
                            k,
                            n,
                            scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                            x_byte_off: arena.offset(x_id) as u32,
                            w_byte_off: arena.offset(w_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                        });
                    } else {
                        let (block_size, scheme_id) = match scheme {
                            QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                            QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                            QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                            QuantScheme::Fp8E4m3 => (1, 3u32),
                            QuantScheme::Fp8E5m2 => (1, 4u32),
                            QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                            other => panic!("rlx-cuda DequantMatMul: unsupported scheme {other:?}"),
                        };
                        let scale_id = node.inputs[2];
                        let zp_id = node.inputs[3];
                        schedule.push(Step::DequantMatmul {
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            w_off: (arena.offset(w_id) / 4) as u32,
                            scale_off: (arena.offset(scale_id) / 4) as u32,
                            zp_off: (arena.offset(zp_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    }
                }
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!("rlx-cuda SelectiveScan: state_size {state_size} > 256 cap");
                    }
                    let x_id = node.inputs[0];
                    let dt_id = node.inputs[1];
                    let a_id = node.inputs[2];
                    let b_id = node.inputs[3];
                    let c_id = node.inputs[4];
                    let in_dims = graph.node(x_id).shape.dims();
                    schedule.push(Step::SelectiveScan {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq: in_dims[1].unwrap_static() as u32,
                        hidden: in_dims[2].unwrap_static() as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        delta_off: (arena.offset(dt_id) / 4) as u32,
                        a_off: (arena.offset(a_id) / 4) as u32,
                        b_off: (arena.offset(b_id) / 4) as u32,
                        c_off: (arena.offset(c_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Fft { inverse, norm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.clone();
                    let meta = rlx_ir::fft::fft_meta(&in_shape);
                    let dtype = in_shape.dtype();
                    let use_gpu = matches!(dtype, rlx_ir::DType::F32)
                        && meta.n_complex.is_power_of_two()
                        && meta.n_complex >= 2;
                    // #6: if this forward FFT's input is a fused real→complex
                    // zero-pad (`Concat([signal, zeros])`), read `signal` directly
                    // with `im = 0` and skip the Concat+Sub entirely.
                    #[cfg(feature = "native-cuda-fft")]
                    let (src_id, real_input) = match fft_real_src.get(&node.id) {
                        Some(&sig) => (sig, true),
                        None => (in_id, false),
                    };
                    #[cfg(not(feature = "native-cuda-fft"))]
                    let (src_id, real_input) = (in_id, false);
                    schedule.push(Step::Fft {
                        src_byte_off: arena.offset(src_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_complex: meta.n_complex as u32,
                        inverse: *inverse,
                        norm_tag: norm.tag(),
                        dtype_tag: fft_dtype_tag(dtype),
                        use_gpu,
                        real_input,
                    });
                }
                Op::LogMel => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                    schedule.push(Step::LogMelHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::LogMelBackward => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                    schedule.push(Step::LogMelBackwardHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dy_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::WelchPeaks { k, n_segments } => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                        .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                    let use_gpu = rlx_ir::audio::welch_peaks_gpu_native_eligible(
                        &spec_shape,
                        *k,
                        *n_segments,
                    )
                    .unwrap_or(false);
                    if use_gpu {
                        schedule.push(Step::WelchPeaksGpu {
                            spec_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            dst_off: (arena.offset(node.id) / 4) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                            n_bins: meta.n_bins as u32,
                        });
                    } else {
                        schedule.push(Step::WelchPeaksHost {
                            spec_byte_off: arena.offset(node.inputs[0]) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                        });
                    }
                }
                Op::Im2Col {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    if kernel_size.len() != 2 || x_shape.rank() != 4 {
                        panic!("rlx-cuda Im2Col: 2D NCHW only");
                    }
                    let n = match x_shape.dim(0) {
                        rlx_ir::shape::Dim::Static(v) => v as u32,
                        _ => 0,
                    };
                    let c_in = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    let kh = kernel_size[0] as u32;
                    let kw = kernel_size[1] as u32;
                    let sh = stride.first().copied().unwrap_or(1) as u32;
                    let sw = stride.get(1).copied().unwrap_or(1) as u32;
                    let ph = padding.first().copied().unwrap_or(0) as u32;
                    let pw = padding.get(1).copied().unwrap_or(0) as u32;
                    let dh = dilation.first().copied().unwrap_or(1) as u32;
                    let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
                    let h_out = rlx_ir::shape::conv2d_spatial_output(
                        h as usize,
                        kh as usize,
                        sh as usize,
                        ph as usize,
                        dh as usize,
                    ) as u32;
                    let w_out = rlx_ir::shape::conv2d_spatial_output(
                        w as usize,
                        kw as usize,
                        sw as usize,
                        pw as usize,
                        dw_dil as usize,
                    ) as u32;
                    schedule.push(Step::Im2ColHost {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        col_byte_off: arena.offset(node.id) as u32,
                        n,
                        c_in,
                        h,
                        w,
                        h_out,
                        w_out,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        dh,
                        dw_dil,
                        use_gpu: im2col_use_gpu(n, exec_mode),
                    });
                }
                Op::Reverse { axes } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let mut rev_mask = vec![false; rank];
                    for &a in axes {
                        if a < rank {
                            rev_mask[a] = true;
                        }
                    }
                    schedule.push(Step::ReverseHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        dims,
                        rev_mask,
                        elem_bytes: in_shape.dtype().size_bytes() as u32,
                    });
                }
                Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let reduced = in_shape.dim(*axis).unwrap_static();
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    schedule.push(Step::ArgReduceHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        reduced: reduced as u32,
                        inner: inner as u32,
                        is_max: matches!(node.op, Op::ArgMax { .. }),
                    });
                }
                Op::AxialRope2d {
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::AxialRope2dHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: in_shape.dim(0).unwrap_static() as u32,
                        seq: in_shape.dim(1).unwrap_static() as u32,
                        hidden: in_shape.dim(2).unwrap_static() as u32,
                        end_x: *end_x as u32,
                        end_y: *end_y as u32,
                        head_dim: *head_dim as u32,
                        num_heads: *num_heads as u32,
                        theta: *theta,
                        repeat_factor: *repeat_factor as u32,
                    });
                }
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-cuda GatedDeltaNet: state_size {state_size} > {}",
                            rlx_cpu::gdn::GDN_MAX_STATE
                        );
                    }
                    let q_id = node.inputs[0];
                    let q_shape = &graph.node(q_id).shape;
                    let state_off = if *carry_state {
                        arena.offset(node.inputs[5])
                    } else {
                        0
                    };
                    schedule.push(Step::GatedDeltaNet {
                        q_byte_off: arena.offset(q_id) as u32,
                        k_byte_off: arena.offset(node.inputs[1]) as u32,
                        v_byte_off: arena.offset(node.inputs[2]) as u32,
                        g_byte_off: arena.offset(node.inputs[3]) as u32,
                        beta_byte_off: arena.offset(node.inputs[4]) as u32,
                        state_byte_off: state_off as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: q_shape.dim(0).unwrap_static() as u32,
                        seq: q_shape.dim(1).unwrap_static() as u32,
                        heads: q_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                        use_carry: *carry_state,
                    });
                }
                Op::Lstm {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let (h0, c0) = if *carry {
                        (
                            arena.offset(node.inputs[4]) as u32,
                            arena.offset(node.inputs[5]) as u32,
                        )
                    } else {
                        (0u32, 0u32)
                    };
                    schedule.push(Step::Lstm {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                        w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                        bias_byte_off: arena.offset(node.inputs[3]) as u32,
                        h0_byte_off: h0,
                        c0_byte_off: c0,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    });
                }
                Op::Scan {
                    body,
                    length,
                    save_trajectory,
                    num_bcast,
                    num_xs,
                    ..
                } => {
                    let nb = *num_bcast as usize;
                    let nx = *num_xs as usize;
                    let plan = rlx_cpu::thunk::compile_scan_body(body, nb, nx);
                    let bcast_outer: Vec<(usize, usize)> = (0..nb)
                        .map(|i| {
                            let id = node.inputs[1 + i];
                            (arena.offset(id), graph.node(id).shape.size_bytes().unwrap())
                        })
                        .collect();
                    let xs_outer: Vec<(usize, usize)> = (0..nx)
                        .map(|i| {
                            let id = node.inputs[1 + nb + i];
                            let total = graph.node(id).shape.size_bytes().unwrap();
                            (arena.offset(id), total / *length as usize)
                        })
                        .collect();
                    schedule.push(Step::ScanHost {
                        plan: std::sync::Arc::new(plan),
                        outer_init_off: arena.offset(node.inputs[0]),
                        outer_final_off: arena.offset(node.id),
                        length: *length,
                        save_trajectory: *save_trajectory,
                        xs_outer,
                        bcast_outer,
                    });
                }
                Op::Custom { name, attrs, .. } => match name.as_str() {
                    "llada2.group_limited_gate" => {
                        let sig_id = node.inputs[0];
                        let route_id = node.inputs[1];
                        let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                        let mut attr_buf = [0u8; 20];
                        let n = attrs.len().min(20);
                        attr_buf[..n].copy_from_slice(&attrs[..n]);
                        schedule.push(Step::Llada2GroupLimitedGate {
                            sig_off: (arena.offset(sig_id) / 4) as u32,
                            route_off: (arena.offset(route_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            n_elems,
                            attrs: attr_buf,
                        });
                    }
                    "umap.knn" => {
                        let pw_id = node.inputs[0];
                        let n = graph.node(pw_id).shape.dims()[0].unwrap_static() as u32;
                        let k = u32::from_le_bytes(attrs[..4].try_into().unwrap());
                        schedule.push(Step::UmapKnn {
                            pairwise_off: (arena.offset(pw_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            n,
                            k,
                        });
                    }
                    "gdino.ms_deform_attn" => {
                        let in_offs: Vec<(u32, u32)> = node
                            .inputs
                            .iter()
                            .map(|&id| {
                                let len = graph.node(id).shape.num_elements().unwrap() as u32;
                                ((arena.offset(id) / 4) as u32, len)
                            })
                            .collect();
                        let out_len = node.shape.num_elements().unwrap() as u32;
                        schedule.push(Step::MsDeformAttnHost {
                            in_offs,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            out_len,
                            attrs: attrs.clone(),
                        });
                    }
                    other => panic!("rlx-cuda: unsupported Op::Custom('{other}')"),
                },

                Op::GaussianSplatRender {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRender {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        dst_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::GaussianSplatRenderBackward {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRenderBackward {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        d_loss_off: arena.offset(node.inputs[7]) as u32,
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_off: arena.offset(node.id) as u32,
                        packed_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                        loss_grad_clip: *loss_grad_clip,
                        sh_band: *sh_band,
                        max_anisotropy: *max_anisotropy,
                    });
                }

                Op::GaussianSplatPrepare {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatPrepare {
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        meta_len: elem_len(node.inputs[6]),
                        prep_off: arena.offset(node.id) as u32,
                        prep_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::GaussianSplatRasterize {
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    let prep_id = node.inputs[0];
                    let count = match &graph.node(prep_id).op {
                        rlx_ir::Op::GaussianSplatPrepare { .. } => {
                            elem_len(graph.node(prep_id).inputs[0]) / 3
                        }
                        _ => 1,
                    };
                    schedule.push(Step::GaussianSplatRasterize {
                        prep_off: arena.offset(prep_id) as u32,
                        prep_len: elem_len(prep_id),
                        meta_off: arena.offset(node.inputs[1]) as u32,
                        meta_len: elem_len(node.inputs[1]),
                        dst_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        count,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let op_id = pool_op_id(*kind);
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => {
                            schedule.push(Step::Pool1d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                l: in_dims[2].unwrap_static() as u32,
                                l_out: out_dims[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride[0] as u32,
                                pl: padding[0] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        2 => {
                            schedule.push(Step::Pool2d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                h: in_dims[2].unwrap_static() as u32,
                                w: in_dims[3].unwrap_static() as u32,
                                h_out: out_dims[2].unwrap_static() as u32,
                                w_out: out_dims[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride[0] as u32,
                                sw: stride[1] as u32,
                                ph: padding[0] as u32,
                                pw: padding[1] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        3 => {
                            schedule.push(Step::Pool3d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                d: in_dims[2].unwrap_static() as u32,
                                h: in_dims[3].unwrap_static() as u32,
                                w: in_dims[4].unwrap_static() as u32,
                                d_out: out_dims[2].unwrap_static() as u32,
                                h_out: out_dims[3].unwrap_static() as u32,
                                w_out: out_dims[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride[0] as u32,
                                sh: stride[1] as u32,
                                sw: stride[2] as u32,
                                pd: padding[0] as u32,
                                ph: padding[1] as u32,
                                pw: padding[2] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        other => panic!("rlx-cuda Pool: unsupported kernel rank {other}"),
                    }
                }
                Op::LayerNorm2d { eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::LayerNorm2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::ConvTranspose2d {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::ConvTranspose2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w_in: in_shape.dim(3).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    });
                }
                Op::GroupNorm { num_groups, eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::GroupNorm {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        num_groups: *num_groups as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::ResizeNearest2x => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2x {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
                }
                Op::Conv {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let w_off = (arena.offset(w_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => {
                            schedule.push(Step::Conv1d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                l: in_dims[2].unwrap_static() as u32,
                                l_out: out_dims[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride[0] as u32,
                                pl: padding[0] as u32,
                                dl: dilation[0] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        2 => {
                            schedule.push(Step::Conv2d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                h: in_dims[2].unwrap_static() as u32,
                                w: in_dims[3].unwrap_static() as u32,
                                h_out: out_dims[2].unwrap_static() as u32,
                                w_out: out_dims[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride[0] as u32,
                                sw: stride[1] as u32,
                                ph: padding[0] as u32,
                                pw: padding[1] as u32,
                                dh: dilation[0] as u32,
                                dw: dilation[1] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        3 => {
                            schedule.push(Step::Conv3d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                d: in_dims[2].unwrap_static() as u32,
                                h: in_dims[3].unwrap_static() as u32,
                                w: in_dims[4].unwrap_static() as u32,
                                d_out: out_dims[2].unwrap_static() as u32,
                                h_out: out_dims[3].unwrap_static() as u32,
                                w_out: out_dims[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride[0] as u32,
                                sh: stride[1] as u32,
                                sw: stride[2] as u32,
                                pd: padding[0] as u32,
                                ph: padding[1] as u32,
                                pw: padding[2] as u32,
                                dd: dilation[0] as u32,
                                dh: dilation[1] as u32,
                                dw: dilation[2] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        other => panic!("rlx-cuda Conv: unsupported kernel rank {other}"),
                    }
                }
                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let is_greedy = *top_k == 0
                        && (*top_p - 1.0).abs() < 1e-6
                        && (*temperature - 1.0).abs() < 1e-6;
                    if is_greedy {
                        schedule.push(Step::Argmax {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    } else {
                        schedule.push(Step::Sample {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            top_k: *top_k as u32,
                            top_p_bits: top_p.to_bits(),
                            temp_bits: temperature.to_bits(),
                            seed_lo: *seed as u32,
                            seed_hi: (*seed >> 32) as u32,
                        });
                    }
                }
                Op::RngNormal {
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0);
                    schedule.push(Step::RngNormal {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len: len as u32,
                        mean: *mean,
                        scale: *scale,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                Op::RngUniform {
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0);
                    schedule.push(Step::RngUniform {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len: len as u32,
                        low: *low,
                        high: *high,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let eps_bits = eps.to_bits();
                    let off = |i: usize| arena.offset(node.inputs[i]) as u32;
                    let common = (off(0), off(1), off(2), off(3), rows, h, eps_bits);
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => {
                            schedule.push(Step::RmsNormBackwardInput {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dx_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardGamma { .. } => {
                            schedule.push(Step::RmsNormBackwardGamma {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dgamma_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardBeta { .. } => {
                            schedule.push(Step::RmsNormBackwardBeta {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dbeta_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                Op::RopeBackward { head_dim, n_rot } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                        (
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                            dy_shape.dim(2).unwrap_static() as u32,
                        )
                    } else {
                        (
                            1,
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                        )
                    };
                    let cos_len = graph.node(node.inputs[1]).shape.num_elements().unwrap() as u32;
                    schedule.push(Step::RopeBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        cos_byte_off: arena.offset(node.inputs[1]) as u32,
                        sin_byte_off: arena.offset(node.inputs[2]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        hidden,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        cos_len,
                    });
                }
                Op::CumsumBackward { exclusive, .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                    schedule.push(Step::CumsumBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        rows,
                        cols,
                        exclusive: *exclusive,
                    });
                }
                Op::GatherBackward { .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    let rank = out_shape.rank();
                    let axis = match &node.op {
                        Op::GatherBackward { axis } => *axis,
                        _ => 0,
                    };
                    let axis_u = if axis < 0 {
                        (rank as i32 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let outer: usize = (0..axis_u)
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let num_idx = idx_shape.dim(axis_u).unwrap_static();
                    let trailing: usize = (axis_u + 1..dy_shape.rank())
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_dim = out_shape.dim(axis_u).unwrap_static();
                    schedule.push(Step::GatherBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        indices_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                    });
                }
                Op::Conv2dBackwardInput {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    if kernel_size.len() == 2 && dy_shape.rank() == 4 && out_shape.rank() == 4 {
                        schedule.push(Step::Conv2dBackwardInput {
                            dy_byte_off: arena.offset(node.inputs[0]) as u32,
                            w_byte_off: arena.offset(node.inputs[1]) as u32,
                            dx_byte_off: arena.offset(node.id) as u32,
                            n: out_shape.dim(0).unwrap_static() as u32,
                            c_in: out_shape.dim(1).unwrap_static() as u32,
                            h: out_shape.dim(2).unwrap_static() as u32,
                            w_in: out_shape.dim(3).unwrap_static() as u32,
                            c_out: dy_shape.dim(1).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            dh: dilation.first().copied().unwrap_or(1) as u32,
                            dw: dilation.get(1).copied().unwrap_or(1) as u32,
                            groups: *groups as u32,
                        });
                    } else {
                        panic!("rlx-cuda: Conv2dBackwardInput expects 2-D conv on NCHW tensors");
                    }
                }
                Op::Conv2dBackwardWeight {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
                        schedule.push(Step::Conv2dBackwardWeight {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            dy_byte_off: arena.offset(node.inputs[1]) as u32,
                            dw_byte_off: arena.offset(node.id) as u32,
                            n: x_shape.dim(0).unwrap_static() as u32,
                            c_in: x_shape.dim(1).unwrap_static() as u32,
                            h: x_shape.dim(2).unwrap_static() as u32,
                            w: x_shape.dim(3).unwrap_static() as u32,
                            c_out: dy_shape.dim(1).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            dh: dilation.first().copied().unwrap_or(1) as u32,
                            dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                            groups: *groups as u32,
                        });
                    } else {
                        panic!("rlx-cuda: Conv2dBackwardWeight expects 2-D conv on NCHW tensors");
                    }
                }
                Op::MaxPool2dBackward {
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
                        schedule.push(Step::MaxPool2dBackward {
                            x_byte_off: arena.offset(node.inputs[0]) as u32,
                            dy_byte_off: arena.offset(node.inputs[1]) as u32,
                            dx_byte_off: arena.offset(node.id) as u32,
                            n: x_shape.dim(0).unwrap_static() as u32,
                            c: x_shape.dim(1).unwrap_static() as u32,
                            h: x_shape.dim(2).unwrap_static() as u32,
                            w: x_shape.dim(3).unwrap_static() as u32,
                            h_out: dy_shape.dim(2).unwrap_static() as u32,
                            w_out: dy_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                        });
                    } else {
                        panic!("rlx-cuda: MaxPool2dBackward expects 2-D pool on NCHW tensors");
                    }
                }
                other => panic!(
                    "rlx-cuda: op {other:?} not yet lowered. \
                     Open a follow-up PR if you hit this — every other op \
                     in the IR is wired."
                ),
            }
        }

        let schedule = fuse_elementwise_chains(schedule);

        let blas = cuda_blas();
        let needs_blas_lt = schedule_needs_blas_lt(&schedule);
        let needs_dnn = schedule_needs_dnn(&schedule);
        let blas_lt = if needs_blas_lt {
            cuda_blas_lt_handle()
        } else {
            None
        };
        let blas_lt_workspace = if needs_blas_lt {
            cuda_blas_lt_workspace()
        } else {
            None
        };
        let dnn = if needs_dnn { cuda_dnn_handle() } else { None };
        let dnn_workspace = if needs_dnn {
            cuda_dnn_workspace()
        } else {
            None
        };

        let streams = match exec_mode {
            ExecMode::MultiStream(n) if n > 1 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if let Ok(s) = ctx.new_stream() {
                        v.push(s);
                    }
                }
                v
            }
            _ => Vec::new(),
        };

        let output_staging: Vec<F32HostSlot> = graph
            .outputs
            .iter()
            .map(|&id| {
                let elems = graph.node(id).shape.num_elements().unwrap_or(0);
                // Cacheable pinned (not write-combined) so the host-read side
                // of the D2H readback runs at full bandwidth.
                F32HostSlot::new_output(&ctx, elems, pinned_output_staging_enabled())
            })
            .collect();

        let mut input_staging = HashMap::new();
        if pinned_input_staging_enabled(exec_mode) {
            for (name, &id) in &input_offsets {
                let elems = graph.node(id).shape.num_elements().unwrap_or(0);
                input_staging.insert(name.clone(), F32HostSlot::new(&ctx, elems, true));
            }
        }

        let replay_event = if exec_mode == ExecMode::Graph {
            ctx.new_event(None).ok()
        } else {
            None
        };

        let mut input_slot_names = Vec::new();
        let mut input_slots = Vec::new();
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op {
                let off = if arena.has(node.id) {
                    arena.offset(node.id)
                } else {
                    0
                };
                let len = node.shape.num_elements().unwrap_or(0);
                input_slot_names.push(name.clone());
                input_slots.push((off, len));
            }
        }

        let mut host_total = 0usize;
        let mut output_slots = Vec::new();
        for &id in &graph.outputs {
            let n = graph.node(id).shape.num_elements().unwrap_or(0);
            output_slots.push((host_total * 4, n));
            host_total += n;
        }
        let host_arena = vec![0.0f32; host_total];

        Self {
            ctx,
            blas,
            blas_lt,
            blas_lt_workspace,
            dnn,
            dnn_workspace,
            half_act_scratch: None,
            dequant_scratch_off,
            graph,
            arena,
            schedule,
            input_offsets,
            param_offsets,
            meta_buffers,
            exec_mode,
            captured_graph: None,
            streams,
            active_extent: None,
            output_staging,
            input_staging,
            #[cfg(feature = "cufft")]
            cufft_state: crate::cufft_dispatch::CufftState::new(),
            replay_event,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            kv_row_feeds: HashMap::new(),
            gpu_handle_resident: std::collections::HashSet::new(),
            pending_read_indices: None,
            readback_plan_buf: Vec::new(),
            captured_readback_plan: None,
            input_slot_names,
            input_slots,
            output_slots,
            host_arena,
            rng: std::sync::Arc::new(std::sync::RwLock::new(rng)),
        }
    }

    /// Full constructor with explicit compile + exec modes (default RNG).
    pub fn compile_with(graph: Graph, compile_mode: CompileMode, exec_mode: ExecMode) -> Self {
        Self::compile_with_rng(
            graph,
            compile_mode,
            exec_mode,
            rlx_ir::RngOptions::default(),
        )
    }

    /// One-shot eager run. Compiles, executes once with the given
    /// inputs, and drops the executable. No persistent state.
    pub fn eager(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut exec = Self::compile_with(graph, CompileMode::Jit, ExecMode::Eager);
        exec.run(inputs)
    }

    /// Host buffer base for reading outputs after [`Self::run_slots`].
    /// Offsets in the returned slot pairs are **byte** offsets into this buffer.
    pub fn arena_ptr(&self) -> *const u8 {
        self.host_arena.as_ptr() as *const u8
    }

    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.output_slots
    }

    fn upload_slot_inputs(&mut self, inputs: &[&[f32]]) {
        let stream = self.ctx.default_stream();
        for (i, data) in inputs.iter().enumerate() {
            let Some(&(byte_off, max_elems)) = self.input_slots.get(i) else {
                break;
            };
            let off_f32 = byte_off / 4;
            let len = data.len().min(max_elems);
            if len == 0 {
                continue;
            }
            let mut slot = self.arena.f32_buf_mut().slice_mut(off_f32..off_f32 + len);
            if let Some(name) = self.input_slot_names.get(i) {
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    let _ = host.htod(&stream, &mut slot, len);
                    continue;
                }
            }
            let _ = stream.memcpy_htod(&data[..len], &mut slot);
        }
    }

    fn pack_host_arena(&mut self) {
        self.prepare_readback_plan();
        for &i in &self.readback_plan_buf {
            if i >= self.output_staging.len() || i >= self.output_slots.len() {
                continue;
            }
            let (byte_off, n) = self.output_slots[i];
            if n == 0 {
                continue;
            }
            let start = byte_off / 4;
            let end = start + n;
            if end <= self.host_arena.len() {
                self.output_staging[i].copy_into(&mut self.host_arena[start..end]);
            }
        }
    }

    /// Fast path: positional inputs, D2H into [`Self::host_arena`], no per-output `Vec`.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.upload_slot_inputs(inputs);
        let _ = self.run_inner(&[]);
        self.pack_host_arena();
        &self.output_slots
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every step in the schedule passes
    /// `Step::safe_for_active_extent`. Bypasses captured CUDA Graph
    /// (recorded at full extent) when active. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    /// Declared graph-output dtypes, in `graph.outputs` order. Used by
    /// the runtime wrapper's `run_typed` to narrow f32 outputs back to
    /// the declared dtype on the way out.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let off_f32 = self.arena.offset(id) / 4;
            let stream = self.ctx.default_stream();
            let mut slot = self
                .arena
                .f32_buf_mut()
                .slice_mut(off_f32..off_f32 + data.len());
            stream
                .memcpy_htod(data, &mut slot)
                .expect("rlx-cuda: param upload failed");
        }
    }

    /// Upload packed U8/I8 GGUF weights into the param slot (byte offset).
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let byte_off = self.arena.offset(id);
            let stream = self.ctx.default_stream();
            crate::gguf_host::upload_param_bytes(&stream, self.arena.f32_buf_mut(), byte_off, data);
        }
    }

    /// Upload a param as packed half-precision bits (`u16` per element).
    /// Caller passes the raw IEEE-754 binary16 (`F16`) or BFloat16
    /// (`Bf16`) bit pattern; the backend stores it in the half-arena
    /// side-buffer and skips the f32 slot entirely. Use cases:
    /// 2× weight-memory savings for inference, plus Tensor Core matmul
    /// via `cublasGemmEx` when both A and B (or just B) are stored
    /// half-precision.
    ///
    /// When the same `name` is also `set_param`'d as f32, the
    /// half-arena entry takes precedence in the matmul dispatch. Use
    /// only one of the two for any given param.
    pub fn set_param_half(&mut self, name: &str, dtype: crate::arena::HalfDtype, bits: &[u16]) {
        let id = match self.param_offsets.get(name) {
            Some(&id) if self.arena.has(id) => id,
            _ => return,
        };
        let f32_off = (self.arena.offset(id) / 4) as u32;
        let off = self
            .arena
            .register_half_param(&self.ctx, id, f32_off, bits.len(), dtype);
        let stream = self.ctx.default_stream();
        if let Some(buf) = self.arena.half_buffer.as_mut() {
            let mut slot = buf.slice_mut(off..off + bits.len());
            stream
                .memcpy_htod(bits, &mut slot)
                .expect("rlx-cuda: half-param upload failed");
        }
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    /// Run and read back only selected outputs (+ GPU handle feed outputs).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        match read_indices {
            None => self.pending_read_indices = None,
            Some(ix) => {
                let buf = self.pending_read_indices.get_or_insert_with(Vec::new);
                buf.clear();
                buf.extend_from_slice(ix);
                normalize_read_indices(buf);
            }
        }
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        outs
    }

    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    /// Upload any bound (non-resident) GPU handles from host mirrors into the arena.
    pub fn stage_bound_gpu_handles_to_arena(&mut self) {
        let stream = self.ctx.default_stream();
        self.stage_gpu_handle_inputs(&stream, &[]);
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Register a row feed for resident KV decode (mirrors rlx-vulkan).
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    #[allow(dead_code)] // kept for manual stream debugging / future multi-stream sync
    fn sync_all_streams(&self) {
        let _ = self.ctx.default_stream().synchronize();
        for s in &self.streams {
            let _ = s.synchronize();
        }
    }

    /// In-arena f32 copy (element offsets into the unified arena buffer).
    fn copy_arena_f32_range(
        ctx: &Arc<CudaContext>,
        stream: &Arc<cudarc::driver::CudaStream>,
        buffer: &mut cudarc::driver::CudaSlice<f32>,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) {
        if n == 0 || src_off == dst_off {
            return;
        }
        let kernel = copy_kernel(ctx);
        let count = n as u32;
        let src = src_off as u32;
        let dst = dst_off as u32;
        let (grid, block) = dispatch_grid_1d(count, 64);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher.arg(buffer).arg(&count).arg(&src).arg(&dst);
        unsafe {
            let _ = launcher.launch(cfg);
        }
    }

    /// D2D copy of one KV row from a decode output into its resident handle input.
    /// Syncs the stream so a subsequent bucket rollover read sees the new row.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        if row_elems == 0 {
            return;
        }
        let stream = self.ctx.default_stream();
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in &feeds {
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if *out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[*out_idx];
            if in_id == out_id {
                continue;
            }
            let base_out = self.arena.offset(out_id) / 4;
            let base_in = self.arena.offset(in_id) / 4;
            let rel_src = src_row * row_elems;
            let rel_dst = dst_row * row_elems;
            let cap_in = self.arena.len_of(in_id) / 4;
            let cap_out = self.arena.len_of(out_id) / 4;
            if rel_src + row_elems > cap_out || rel_dst + row_elems > cap_in {
                continue;
            }
            let src_off = base_out + rel_src;
            let dst_off = base_in + rel_dst;
            Self::copy_arena_f32_range(
                &self.ctx,
                &stream,
                self.arena.f32_buf_mut(),
                src_off,
                dst_off,
                row_elems,
            );
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
        let _ = stream.synchronize();
    }

    /// Read one row from a graph output without full-tensor D2H.
    /// Caller must ensure GPU work is complete (`run` / `run_read_outputs` syncs).
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if row_inner == 0 || out_idx >= self.graph.outputs.len() {
            return None;
        }
        let id = self.graph.outputs[out_idx];
        let shape_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
        if shape_elems == 0 {
            return None;
        }
        let rel = row * row_inner;
        if rel + row_inner > shape_elems {
            return None;
        }
        let base = self.arena.offset(id) / 4;
        let off = base + rel;
        let cap_f32 = self.arena.len_of(id) / 4;
        if off + row_inner > base + cap_f32 {
            return None;
        }
        let stream = self.ctx.default_stream();
        let mut host = vec![0f32; row_inner];
        let src = self.arena.f32_buf().slice(off..off + row_inner);
        stream.memcpy_dtoh(&src, &mut host).ok()?;
        Some(host)
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name) {
            if out_idx < self.graph.outputs.len() {
                let id = self.graph.outputs[out_idx];
                let stream = self.ctx.default_stream();
                let off_f32 = self.arena.offset(id) / 4;
                let n_f32 = self.arena.len_of(id) / 4;
                let mut host = vec![0f32; n_f32];
                let src = self.arena.f32_buf().slice(off_f32..off_f32 + n_f32);
                if stream.memcpy_dtoh(&src, host.as_mut_slice()).is_ok() {
                    return Some(host);
                }
            }
        }
        if self.gpu_handle_resident.contains(name) {
            if let Some(&id) = self.input_offsets.get(name) {
                let stream = self.ctx.default_stream();
                let off_f32 = self.arena.offset(id) / 4;
                let n_f32 = self.arena.len_of(id) / 4;
                let mut host = vec![0f32; n_f32];
                let src = self.arena.f32_buf().slice(off_f32..off_f32 + n_f32);
                if stream.memcpy_dtoh(&src, host.as_mut_slice()).is_ok() {
                    return Some(host);
                }
            }
        }
        self.gpu_handles.get(name).cloned()
    }

    /// Mark a graph input as device-resident without a host mirror or H2D upload.
    pub fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.insert(name.to_string());
        self.gpu_handles.remove(name);
        true
    }

    #[allow(dead_code)] // kept for future cross-stream device-to-device copies
    fn copy_f32_dtod_between(
        stream: &Arc<cudarc::driver::CudaStream>,
        src: &cudarc::driver::CudaSlice<f32>,
        src_off: usize,
        dst: &mut cudarc::driver::CudaSlice<f32>,
        dst_off: usize,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let src_slice = src.slice(src_off..src_off + n);
        let mut dst_slice = dst.slice_mut(dst_off..dst_off + n);
        let _ = stream.memcpy_dtod(&src_slice, &mut dst_slice);
    }

    /// Copy a resident K/V prefix from another executable (bucket rollover).
    ///
    /// Rows below `outgoing_upper` are read from the source resident inputs; the
    /// top-of-bucket row (`g == outgoing_upper` when `to_row > outgoing_upper`) is
    /// read from decode outputs because `feed_kv_row` cannot write into the last
    /// resident slot when `dst_row == bucket upper`.
    ///
    /// Values are staged host-side (D2H then H2D) to match the flush path used in
    /// `rlx-llama32` today. Padding rows `[to_row..cap)` are zeroed. A future fast
    /// path may use pure D2D once parity is proven.
    pub fn copy_resident_kv_rows_from(
        &mut self,
        src: &Self,
        from_row: usize,
        to_row: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if from_row >= to_row || n_layers == 0 || kv_dim == 0 {
            return true;
        }
        let stream = self.ctx.default_stream();
        let need_top = to_row > outgoing_upper;
        let top_global = outgoing_upper;
        if need_top {
            let _ = stream.synchronize();
        }

        for i in 0..n_layers {
            let k_name = format!("past_k_{i}");
            let v_name = format!("past_v_{i}");
            let Some(&dst_k) = self.input_offsets.get(k_name.as_str()) else {
                return false;
            };
            let Some(&dst_v) = self.input_offsets.get(v_name.as_str()) else {
                return false;
            };
            let Some(&src_k) = src.input_offsets.get(k_name.as_str()) else {
                return false;
            };
            let Some(&src_v) = src.input_offsets.get(v_name.as_str()) else {
                return false;
            };
            if !self.arena.has(dst_k)
                || !self.arena.has(dst_v)
                || !src.arena.has(src_k)
                || !src.arena.has(src_v)
            {
                return false;
            }

            self.gpu_handle_resident.insert(k_name.clone());
            self.gpu_handle_resident.insert(v_name.clone());
            self.gpu_handles.remove(&k_name);
            self.gpu_handles.remove(&v_name);

            let dst_k_base = self.arena.offset(dst_k) / 4;
            let dst_v_base = self.arena.offset(dst_v) / 4;
            let k_out = 1 + 2 * i;
            let v_out = 2 + 2 * i;
            if k_out >= src.graph.outputs.len() || v_out >= src.graph.outputs.len() {
                return false;
            }

            for g in from_row..to_row {
                let row_off = g.saturating_mul(kv_dim);
                let from_output = need_top && g == top_global;
                if row_off + kv_dim > self.arena.len_of(dst_k) / 4
                    || row_off + kv_dim > self.arena.len_of(dst_v) / 4
                {
                    return false;
                }
                let (host_k, host_v) = if from_output {
                    let Some(host_k) = src.read_output_row(k_out, top_global, kv_dim) else {
                        return false;
                    };
                    let Some(host_v) = src.read_output_row(v_out, top_global, kv_dim) else {
                        return false;
                    };
                    (host_k, host_v)
                } else {
                    let Some(host_k) = src.read_gpu_handle_row(k_name.as_str(), g, kv_dim) else {
                        return false;
                    };
                    let Some(host_v) = src.read_gpu_handle_row(v_name.as_str(), g, kv_dim) else {
                        return false;
                    };
                    (host_k, host_v)
                };
                let dst_buf = self.arena.f32_buf_mut();
                let mut dst_k_slice =
                    dst_buf.slice_mut(dst_k_base + row_off..dst_k_base + row_off + kv_dim);
                if stream
                    .memcpy_htod(host_k.as_slice(), &mut dst_k_slice)
                    .is_err()
                {
                    return false;
                }
                let dst_buf = self.arena.f32_buf_mut();
                let mut dst_v_slice =
                    dst_buf.slice_mut(dst_v_base + row_off..dst_v_base + row_off + kv_dim);
                if stream
                    .memcpy_htod(host_v.as_slice(), &mut dst_v_slice)
                    .is_err()
                {
                    return false;
                }
            }

            let cap_rows = self.arena.len_of(dst_k) / 4 / kv_dim.max(1);
            if to_row < cap_rows {
                let zeros = vec![0f32; kv_dim];
                for row in to_row..cap_rows {
                    let row_off = row * kv_dim;
                    let dst_buf = self.arena.f32_buf_mut();
                    let mut dst_k_slice =
                        dst_buf.slice_mut(dst_k_base + row_off..dst_k_base + row_off + kv_dim);
                    if stream
                        .memcpy_htod(zeros.as_slice(), &mut dst_k_slice)
                        .is_err()
                    {
                        return false;
                    }
                    let dst_buf = self.arena.f32_buf_mut();
                    let mut dst_v_slice =
                        dst_buf.slice_mut(dst_v_base + row_off..dst_v_base + row_off + kv_dim);
                    if stream
                        .memcpy_htod(zeros.as_slice(), &mut dst_v_slice)
                        .is_err()
                    {
                        return false;
                    }
                }
            }
        }
        let _ = stream.synchronize();
        true
    }

    /// D2D copy of a resident KV prefix from another executable (bucket rollover).
    pub fn seed_resident_kv_prefix_from(
        &mut self,
        src: &Self,
        prefix_tokens: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        self.copy_resident_kv_rows_from(src, 0, prefix_tokens, outgoing_upper, kv_dim, n_layers)
    }

    /// Read one row from a resident GPU input handle without full-tensor D2H.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if row_inner == 0 {
            return None;
        }
        let &id = self.input_offsets.get(name)?;
        let cap_f32 = self.arena.len_of(id) / 4;
        let rel = row * row_inner;
        if rel + row_inner > cap_f32 {
            return None;
        }
        let base = self.arena.offset(id) / 4;
        let off = base + rel;
        let stream = self.ctx.default_stream();
        let mut host = vec![0f32; row_inner];
        let src = self.arena.f32_buf().slice(off..off + row_inner);
        stream.memcpy_dtoh(&src, &mut host).ok()?;
        Some(host)
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_with_rng(
            self.graph.clone(),
            compile_mode_from_env(),
            exec_mode_from_env(),
            self.rng(),
        );
        for (k, v) in &self.gpu_handles {
            exe.bind_gpu_handle(k, v);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        for (k, &idx) in &self.kv_row_feeds {
            exe.register_kv_row_feed(k, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }

    /// Build the sorted output readback plan into [`Self::readback_plan_buf`].
    fn prepare_readback_plan(&mut self) {
        self.readback_plan_buf.clear();
        let n = self.graph.outputs.len();
        if let Some(ref want) = self.pending_read_indices {
            self.readback_plan_buf.extend_from_slice(want);
            normalize_read_indices(&mut self.readback_plan_buf);
            return;
        }
        self.readback_plan_buf.extend(0..n);
    }

    fn propagate_gpu_handle_feeds_d2d(&mut self, stream: &Arc<cudarc::driver::CudaStream>) {
        let extent = self.active_extent;
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_bytes = self.arena.len_of(out_id);
                let copy_bytes = match extent {
                    Some((actual, upper)) if upper > 0 => {
                        let stride = (out_bytes / (upper + 1)).max(4);
                        (actual * stride).min(out_bytes)
                    }
                    _ => out_bytes,
                }
                .min(self.arena.len_of(in_id));
                let src_off = self.arena.offset(out_id) / 4;
                let dst_off = self.arena.offset(in_id) / 4;
                let n_f32 = copy_bytes / 4;
                if n_f32 > 0 && src_off != dst_off {
                    let mut tmp = vec![0.0f32; n_f32];
                    let src = self.arena.f32_buf().slice(src_off..src_off + n_f32);
                    if stream.memcpy_dtoh(&src, &mut tmp).is_ok() {
                        let mut dst = self.arena.f32_buf_mut().slice_mut(dst_off..dst_off + n_f32);
                        let _ = stream.memcpy_htod(&tmp, &mut dst);
                    }
                }
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    fn stage_gpu_handle_inputs(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        inputs: &[(&str, &[f32])],
    ) {
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let mut slot = self
                    .arena
                    .f32_buf_mut()
                    .slice_mut(off_f32..off_f32 + data.len());
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    let _ = host.htod(stream, &mut slot, data.len());
                } else {
                    let _ = stream.memcpy_htod(data.as_slice(), &mut slot);
                }
            }
        }
    }

    fn refresh_gpu_handles_from_staging(&mut self, plan: &[usize]) {
        if self.pending_read_indices.is_some() {
            return;
        }
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if plan.contains(&out_idx) && out_idx < self.output_staging.len() {
                self.gpu_handles
                    .insert(name.clone(), self.output_staging[out_idx].to_vec());
            }
        }
    }

    fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let default_stream = self.ctx.default_stream();
        let stream = default_stream.clone();

        self.stage_gpu_handle_inputs(&stream, inputs);

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into the
        // captured CUDA Graph.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let mut slot = self
                    .arena
                    .f32_buf_mut()
                    .slice_mut(off_f32..off_f32 + data.len());
                if let Some(host) = self.input_staging.get_mut(name) {
                    host.copy_from_host(data);
                    host.htod(&stream, &mut slot, data.len())
                        .expect("rlx-cuda: pinned input upload failed");
                } else {
                    stream
                        .memcpy_htod(data, &mut slot)
                        .expect("rlx-cuda: input upload failed");
                }
            }
        }

        // Active-extent (PLAN L1): when set + every Step safe, bypass
        // captured CUDA Graph (recorded at full extent) and dispatch
        // per-step with scaled launch dims via the normal loop.
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        // Scale a count by actual/upper with ceiling-division, clamped to [0, full].
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // CUDA Graph fast path: replay a previously-captured schedule.
        // The first run with `ExecMode::Graph` falls through to the
        // normal dispatch loop with stream capture turned on; the
        // resulting graph is stashed in `self.captured_graph` and
        // replayed on every subsequent run.
        let graph_eligible = active.is_none()
            && self.exec_mode == ExecMode::Graph
            && schedule_graph_capture_safe(&self.schedule);
        let do_replay = graph_eligible && self.captured_graph.is_some();
        let do_capture = graph_eligible && self.captured_graph.is_none();

        if do_replay {
            self.prepare_readback_plan();
            let plan_ok = self
                .captured_readback_plan
                .as_ref()
                .is_some_and(|p| p.as_slice() == self.readback_plan_buf.as_slice());
            if plan_ok {
                self.captured_graph
                    .as_ref()
                    .unwrap()
                    .launch()
                    .expect("rlx-cuda: graph replay failed");
                if let Some(evt) = &self.replay_event {
                    evt.record(&stream)
                        .expect("rlx-cuda: replay event record failed");
                    evt.synchronize()
                        .expect("rlx-cuda: replay event sync failed");
                } else {
                    stream.synchronize().expect("rlx-cuda: stream sync failed");
                }
                run_tail_host_audio_ops(&self.schedule, &stream, self.arena.f32_buf_mut(), false);
                let plan = self.readback_plan_buf.clone();
                let read_all = plan.len() == self.graph.outputs.len();
                // DtoH must run after every replay — inputs change each run and
                // must not rely on dtoh baked into the captured graph.
                if read_all {
                    self.fill_output_staging(&stream)
                        .expect("rlx-cuda: output dtoh failed after replay");
                } else {
                    self.fill_output_staging_indices(&stream, &plan)
                        .expect("rlx-cuda: partial output dtoh failed after replay");
                }
                self.refresh_gpu_handles_from_staging(&plan);
                return self.outputs_from_staging_plan(&plan);
            }
            // Readback plan changed (e.g. partial grads); drop stale capture and re-dispatch.
            self.captured_graph = None;
            self.captured_readback_plan = None;
        }
        let _ = do_replay;

        let mut capturing = false;
        if do_capture {
            capturing = stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .is_ok();
        }

        // Multi-stream scheduler state. When `exec_mode ==
        // MultiStream(n)`, each Step gets assigned to one of `n` pool
        // streams based on producer-consumer dependencies on arena
        // offsets. Independent ops (e.g. unfused Q/K/V matmuls)
        // parallelise; producer-consumer chains stay on one stream.
        let multi_stream =
            matches!(self.exec_mode, ExecMode::MultiStream(_)) && !self.streams.is_empty();
        let mut producer_of: HashMap<u32, usize> = HashMap::new();
        let mut last_event: HashMap<usize, cudarc::driver::CudaEvent> = HashMap::new();
        let mut rr_cursor: usize = 0;

        // Dispatch each step. Each iteration is wrapped in an NVTX
        // range so nsight-systems traces show step boundaries cleanly.
        // Gated behind the `nvtx` feature because CUDA 13 removed
        // `nvToolsExt.dll`; cudarc panics on first call when the lib
        // isn't loadable.
        for step in &self.schedule {
            #[cfg(feature = "nvtx")]
            let _nvtx = cudarc::nvtx::scoped_range(step_name(step));
            // PLAN L3: cross-backend Perfetto trace; no-op when env
            // var RLX_TRACE_PERFETTO unset.
            let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "cuda");

            // Per-step stream selection. In single-stream mode `stream`
            // shadows to the default stream; in multi-stream mode it
            // shadows to the assigned pool stream (and we cross-stream
            // event-wait on every producer not on the chosen stream).
            let assigned_idx: Option<usize> = if multi_stream {
                let (reads, _) = step_offsets(step);
                let mut producer_streams: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for r in &reads {
                    if let Some(&s) = producer_of.get(r) {
                        producer_streams.insert(s);
                    }
                }
                let chosen = if producer_streams.is_empty() {
                    let s = rr_cursor % self.streams.len();
                    rr_cursor += 1;
                    s
                } else if producer_streams.len() == 1 {
                    *producer_streams.iter().next().unwrap()
                } else {
                    // Multiple producers — keep the chosen one's queue
                    // intact and event-wait on the others.
                    let chosen = *producer_streams.iter().next().unwrap();
                    for s in &producer_streams {
                        if *s != chosen
                            && let Some(evt) = last_event.get(s)
                        {
                            let _ = self.streams[chosen].wait(evt);
                        }
                    }
                    chosen
                };
                Some(chosen)
            } else {
                None
            };
            let stream: Arc<cudarc::driver::CudaStream> = match assigned_idx {
                Some(i) => self.streams[i].clone(),
                None => default_stream.clone(),
            };
            // Re-bind cuBLAS / cuDNN handles to the active stream so
            // their internal kernel launches go to the right queue.
            if multi_stream {
                if let Some(blas) = self.blas.as_ref() {
                    let blas = blas.lock().unwrap();
                    unsafe {
                        let _ = cudarc::cublas::result::set_stream(
                            *blas.handle(),
                            stream.cu_stream() as _,
                        );
                    }
                }
                if let Some(handle) = self.dnn {
                    unsafe {
                        let _ = cudarc::cudnn::result::set_stream(
                            handle,
                            stream.cu_stream() as cudnn_sys::cudaStream_t,
                        );
                    }
                }
            }
            match step {
                Step::ScaledQuantScale {
                    x_off_f32,
                    scale_off_f32,
                    n,
                    max_finite,
                } => {
                    let kernel = crate::kernels::scaled_quant_scale_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_off_f32)
                        .arg(n)
                        .arg(max_finite);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quant_scale launch failed");
                    }
                }
                Step::ScaledQuantizeFp8 {
                    x_off_f32,
                    scale_off_f32,
                    out_byte_off,
                    n,
                    e5m2,
                } => {
                    let kernel = crate::kernels::scaled_quantize_fp8_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_off_f32)
                        .arg(out_byte_off)
                        .arg(n)
                        .arg(e5m2);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quantize_fp8 launch failed");
                    }
                }
                Step::ScaledMatMul {
                    m,
                    k,
                    n,
                    lhs_byte_off,
                    rhs_byte_off,
                    lhs_scale_byte_off,
                    rhs_scale_byte_off,
                    out_byte_off,
                    has_bias,
                    bias_byte_off,
                    lhs_e5m2,
                    rhs_e5m2,
                } => {
                    let lt_handle = self
                        .blas_lt
                        .expect("rlx-cuda ScaledMatMul: cublasLt handle required for FP8 GEMM");
                    let mut workspace = self
                        .blas_lt_workspace
                        .as_ref()
                        .expect("rlx-cuda ScaledMatMul: cublasLt workspace required")
                        .lock()
                        .unwrap();
                    let (workspace_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                    let (arena_ptr, _record) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                    let cu_stream = stream.cu_stream();
                    let r = unsafe {
                        cublaslt_matmul_fp8(
                            lt_handle,
                            workspace_ptr,
                            CUBLASLT_WORKSPACE_BYTES,
                            arena_ptr,
                            *m,
                            *k,
                            *n,
                            *lhs_byte_off as u64,
                            *rhs_byte_off as u64,
                            *lhs_scale_byte_off as u64,
                            *rhs_scale_byte_off as u64,
                            *out_byte_off as u64,
                            *has_bias != 0,
                            *bias_byte_off as u64,
                            *lhs_e5m2 != 0,
                            *rhs_e5m2 != 0,
                            cu_stream,
                        )
                    };
                    r.expect(
                        "rlx-cuda: cublasLt FP8 GEMM failed (needs sm_89+ and 16B-aligned operands)",
                    );
                }
                Step::ScaledQuantScaleGeneral {
                    x_off_f32,
                    scale_byte_off,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let nblk = if *scale_mode == 0 {
                        1
                    } else {
                        cols.div_ceil(*block)
                    };
                    let total = if *scale_mode == 0 { 1 } else { rows * nblk };
                    let kernel = crate::kernels::scaled_quant_scale_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 128);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_byte_off)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quant_scale_general launch failed");
                    }
                }
                Step::ScaledQuantizeGeneral {
                    x_off_f32,
                    scale_byte_off,
                    out_byte_off,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let total = rows * cols;
                    let kernel = crate::kernels::scaled_quantize_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_byte_off)
                        .arg(out_byte_off)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quantize_general launch failed");
                    }
                }
                Step::ScaledDequantizeGeneral {
                    codes_byte_off,
                    scale_byte_off,
                    out_off_f32,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let total = rows * cols;
                    let kernel = crate::kernels::scaled_dequantize_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(codes_byte_off)
                        .arg(scale_byte_off)
                        .arg(out_off_f32)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_dequantize_general launch failed");
                    }
                }
                Step::ScaledMatMulDecode {
                    m,
                    k,
                    n,
                    lhs_byte_off,
                    rhs_byte_off,
                    lhs_scale_byte_off,
                    rhs_scale_byte_off,
                    out_off_f32,
                    lhs_fmt,
                    rhs_fmt,
                    scale_mode,
                    block,
                    has_bias,
                    bias_off_f32,
                } => {
                    let kernel = crate::kernels::scaled_matmul_decode_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(16), (*m).div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(lhs_byte_off)
                        .arg(rhs_byte_off)
                        .arg(lhs_scale_byte_off)
                        .arg(rhs_scale_byte_off)
                        .arg(out_off_f32)
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(lhs_fmt)
                        .arg(rhs_fmt)
                        .arg(scale_mode)
                        .arg(block)
                        .arg(has_bias)
                        .arg(bias_off_f32);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_matmul_decode launch failed");
                    }
                }
                Step::Matmul {
                    m,
                    k,
                    n,
                    a_off_f32,
                    b_off_f32,
                    c_off_f32,
                    batch,
                    a_batch_stride,
                    b_batch_stride,
                    c_batch_stride,
                    has_bias,
                    bias_off_f32,
                    act_id,
                } => {
                    if matmul_parity_mode() {
                        let kernel = matmul_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul (parity) launch failed");
                        }
                        if let Some(idx) = assigned_idx {
                            if let Ok(evt) = stream.record_event(None) {
                                last_event.insert(idx, evt);
                            }
                            let (_, writes) = step_offsets(step);
                            for w in &writes {
                                producer_of.insert(*w, idx);
                            }
                        }
                        continue;
                    }

                    // Tier 0: mixed-precision GemmEx — when B (the weight)
                    // is stored in the half-arena, cast activations to
                    // f16/bf16 in a scratch buffer and call cublasGemmEx
                    // with both inputs half + f32 accumulator. Falls
                    // through to cublasLt on any setup or runtime error.
                    let used_mixed = try_mixed_precision_gemm(
                        &self.ctx,
                        &mut self.arena,
                        &mut self.half_act_scratch,
                        self.blas.as_ref(),
                        &stream,
                        *m,
                        *k,
                        *n,
                        *batch,
                        *a_off_f32,
                        *b_off_f32,
                        *c_off_f32,
                    );
                    if used_mixed {
                        // Optional bias / activation epilogue.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (mixed) failed");
                            }
                        }
                        // Multi-stream tail bookkeeping still runs at end of step.
                        if let Some(idx) = assigned_idx {
                            if let Ok(evt) = stream.record_event(None) {
                                last_event.insert(idx, evt);
                            }
                            let (_, writes) = step_offsets(step);
                            for w in &writes {
                                producer_of.insert(*w, idx);
                            }
                        }
                        continue;
                    }

                    // Tier 1: cublasLt fused (matmul + bias + relu/gelu in
                    // one launch). Only used when the activation is one of
                    // the two cublasLt natively fuses; other acts (silu,
                    // sigmoid, etc.) fall through to the sgemm + epilogue
                    // kernel path.
                    let try_cublaslt = self.blas_lt.is_some()
                        && self.blas_lt_workspace.is_some()
                        && cublaslt_act_supported(*act_id);
                    let used_cublaslt = if try_cublaslt {
                        let lt_handle = self.blas_lt.unwrap();
                        let mut workspace =
                            self.blas_lt_workspace.as_ref().unwrap().lock().unwrap();
                        let (workspace_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _record) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let cu_stream = stream.cu_stream();
                        let act = cublaslt_act_for(*act_id);
                        let r = unsafe {
                            cublaslt_matmul_fused(
                                lt_handle,
                                workspace_ptr,
                                CUBLASLT_WORKSPACE_BYTES,
                                arena_ptr,
                                *m,
                                *k,
                                *n,
                                *a_off_f32,
                                *b_off_f32,
                                *c_off_f32,
                                *has_bias != 0,
                                *bias_off_f32,
                                act,
                                *batch,
                                *a_batch_stride,
                                *b_batch_stride,
                                *c_batch_stride,
                                cu_stream,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("matmul.cublasLt", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cublaslt {
                        continue;
                    }

                    // Tier 2: cuBLAS sgemm via raw pointers (bypasses
                    // the borrow checker's same-buffer aliasing).
                    let used_cublas = if let Some(blas) = self.blas.as_ref() {
                        let blas = blas.lock().unwrap();
                        let (arena_ptr_u64, _record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let a_dev = arena_ptr_u64 + (*a_off_f32 as u64) * 4;
                        let b_dev = arena_ptr_u64 + (*b_off_f32 as u64) * 4;
                        let c_dev = arena_ptr_u64 + (*c_off_f32 as u64) * 4;
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        // cuBLAS is column-major; we have row-major. Trick:
                        // computing C = A·B (row-major) is the same as
                        // computing C^T = B^T · A^T (column-major), and
                        // viewing our row-major arrays as column-major
                        // automatically yields the transpose.
                        let result = unsafe {
                            if *batch == 1 {
                                cudarc::cublas::result::sgemm(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    a_dev as *const f32,
                                    *k as i32,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                )
                            } else {
                                cudarc::cublas::result::sgemm_strided_batched(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    *b_batch_stride as i64,
                                    a_dev as *const f32,
                                    *k as i32,
                                    *a_batch_stride as i64,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                    *c_batch_stride as i64,
                                    *batch as i32,
                                )
                            }
                        };
                        if let Err(ref e) = result {
                            log_fallback("matmul.cublasSgemm", e);
                        }
                        result.is_ok()
                    } else {
                        false
                    };

                    if used_cublas {
                        // Optional fused epilogue (bias + activation) as
                        // a separate element-wise kernel.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue launch failed");
                            }
                        }
                    } else if use_wmma() {
                        // WMMA Tensor Core path: 32×64 block tile, 128 threads/block,
                        // SM 70+ only. Doesn't fuse bias/activation — those go to the
                        // shared epilogue kernel.
                        let kernel = matmul_wmma_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(32), *batch),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul_wmma launch failed");
                        }
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (post-wmma) failed");
                            }
                        }
                    } else {
                        // Custom scalar kernel fallback: 64×64 block tile, 4×4 register tile.
                        let kernel = matmul_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul launch failed");
                        }
                    }
                }
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = binary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: binary launch failed");
                    }
                }
                Step::ElementwiseRegion {
                    len,
                    num_inputs,
                    num_steps,
                    dst_off,
                    input_offs: _,
                    scalar_input_mask,
                    input_modulus,
                    meta_idx,
                    spatial_prologue,
                    prologue_w,
                    prologue_h,
                    prologue_nc,
                } => {
                    let len_s = scale(*len);
                    if len_s == 0 {
                        continue;
                    }
                    let kernel = elementwise_region_kernel(&self.ctx);
                    let ((gx, gy, gz), (bx, by, bz)) = if *spatial_prologue {
                        dispatch_grid_prologue_nchw(*prologue_w, *prologue_h, *prologue_nc)
                    } else {
                        let (grid, block) = dispatch_grid_1d(len_s, 256);
                        ((grid, 1, 1), (block, 1, 1))
                    };
                    let cfg = LaunchConfig {
                        grid_dim: (gx, gy, gz),
                        block_dim: (bx, by, bz),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    // input_modulus is passed by-value as a 64-byte
                    // const param (16 u32s). Could move to meta_buffer
                    // but a constant param keeps the kernel signature
                    // self-describing.
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&len_s)
                        .arg(num_inputs)
                        .arg(num_steps)
                        .arg(dst_off)
                        .arg(&self.meta_buffers[*meta_idx])
                        .arg(scalar_input_mask)
                        .arg(input_modulus);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: elementwise_region launch failed");
                    }
                }
                Step::BatchElementwiseRegion {
                    slice_len,
                    num_batch,
                    num_steps,
                    base_dst_off,
                    slice_elems,
                    batch_offs_idx,
                    meta_idx,
                    scalar_input_mask,
                    input_modulus,
                    ..
                } => {
                    let slice_len_s = scale(*slice_len);
                    let num_batch_s = scale(*num_batch);
                    if slice_len_s == 0 || num_batch_s == 0 {
                        continue;
                    }
                    let kernel = batch_elementwise_region_kernel(&self.ctx);
                    let (grid_x, block_x) = dispatch_grid_1d(slice_len_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid_x, 1, num_batch_s),
                        block_dim: (block_x, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&slice_len_s)
                        .arg(&num_batch_s)
                        .arg(num_steps)
                        .arg(base_dst_off)
                        .arg(slice_elems)
                        .arg(&self.meta_buffers[*batch_offs_idx])
                        .arg(&self.meta_buffers[*meta_idx])
                        .arg(scalar_input_mask)
                        .arg(input_modulus);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_elementwise_region launch failed");
                    }
                }
                Step::FusedBinaryUnary {
                    n,
                    a_off,
                    b_off,
                    out_off,
                    bin_op,
                    un_op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = fused_binary_unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(out_off)
                        .arg(bin_op)
                        .arg(un_op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_binary_unary launch failed");
                    }
                }
                Step::Unary {
                    n,
                    in_off,
                    out_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: unary launch failed");
                    }
                }
                Step::Compare {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = compare_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: compare launch failed");
                    }
                }
                Step::Where {
                    n,
                    cond_off,
                    x_off,
                    y_off,
                    out_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = where_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(cond_off)
                        .arg(x_off)
                        .arg(y_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: where launch failed");
                    }
                }
                Step::Reduce {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = reduce_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: reduce launch failed");
                    }
                }
                Step::Softmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax launch failed");
                    }
                }
                Step::LayerNorm {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    gamma_off,
                    beta_off,
                    eps_bits,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = layernorm_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(eps_bits)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layernorm launch failed");
                    }
                }
                Step::FusedResidualLn {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off,
                    eps_bits,
                    has_bias,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = fused_residual_ln_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(residual_off)
                        .arg(bias_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_residual_ln launch failed");
                    }
                }
                Step::FusedResidualRmsNorm {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off,
                    eps_bits,
                    has_bias,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = fused_residual_rms_norm_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(residual_off)
                        .arg(bias_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_residual_rms_norm launch failed");
                    }
                }
                Step::Gather {
                    n_out,
                    n_idx,
                    dim,
                    vocab,
                    in_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_out, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n_out)
                        .arg(n_idx)
                        .arg(dim)
                        .arg(vocab)
                        .arg(in_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather launch failed");
                    }
                }
                Step::GatherAxis {
                    total,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                    table_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_axis_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(axis_dim)
                        .arg(num_idx)
                        .arg(trailing)
                        .arg(table_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather_axis launch failed");
                    }
                }
                Step::Narrow {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = narrow_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: narrow launch failed");
                    }
                }
                Step::Argmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let kernel = argmax_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: argmax launch failed");
                    }
                }
                Step::Transpose {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = transpose_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: transpose launch failed");
                    }
                }
                Step::Expand {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = expand_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: expand launch failed");
                    }
                }
                Step::Concat {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = concat_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: concat launch failed");
                    }
                }
                Step::Attention {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    softcap_bits,
                    window,
                    seq_q_stride,
                    seq_k_stride,
                    mask_batch_stride,
                    mask_head_stride,
                    q_batch_stride,
                    q_head_stride,
                    q_seq_stride,
                    k_batch_stride,
                    k_head_stride,
                    k_seq_stride,
                    v_batch_stride,
                    v_head_stride,
                    v_seq_stride,
                    o_batch_stride,
                    o_head_stride,
                    o_seq_stride,
                } => {
                    // Tiled flash supports arbitrary Q/K/V strides (BSHD and BHSD).
                    // Row kernel only when head_dim exceeds the flash tile cap or forced.
                    let use_row = rlx_ir::attention_dispatch_use_row(
                        *head_dim,
                        "RLX_CUDA_FORCE_ATTENTION_ROW",
                    );
                    let mut launcher = stream.launch_builder(if use_row {
                        &attention_row_kernel(&self.ctx).function
                    } else {
                        &attention_kernel(&self.ctx).function
                    });
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(heads)
                        .arg(seq_q)
                        .arg(seq_k)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window)
                        .arg(seq_q_stride)
                        .arg(seq_k_stride)
                        .arg(mask_batch_stride)
                        .arg(mask_head_stride)
                        .arg(q_batch_stride)
                        .arg(q_head_stride)
                        .arg(q_seq_stride)
                        .arg(k_batch_stride)
                        .arg(k_head_stride)
                        .arg(k_seq_stride)
                        .arg(v_batch_stride)
                        .arg(v_head_stride)
                        .arg(v_seq_stride)
                        .arg(o_batch_stride)
                        .arg(o_head_stride)
                        .arg(o_seq_stride)
                        .arg(softcap_bits);
                    let cfg = if use_row {
                        let total = batch * heads * seq_q;
                        let block = 256u32;
                        LaunchConfig {
                            grid_dim: (total.div_ceil(block), 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        let q_blocks = (*seq_q).div_ceil(16);
                        LaunchConfig {
                            grid_dim: (q_blocks, batch * heads, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention launch failed");
                    }
                }
                Step::FusedAttn {
                    qkv_off,
                    mask_off,
                    cos_off,
                    sin_off,
                    out_off,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    mask_kind,
                    scale_bits,
                    has_rope,
                } => {
                    let kernel = fused_attn_kernel(&self.ctx);
                    // One block per (batch·head); score matrix [seq·seq] in
                    // dynamic shared memory. The native gate (rlx-cuda unfuse)
                    // keeps `seq` small enough to fit the 48 KB default budget.
                    let cfg = LaunchConfig {
                        grid_dim: (batch * heads, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: seq * seq * 4,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(qkv_off)
                        .arg(mask_off)
                        .arg(cos_off)
                        .arg(sin_off)
                        .arg(out_off)
                        .arg(batch)
                        .arg(seq)
                        .arg(heads)
                        .arg(head_dim)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(has_rope);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_attn launch failed");
                    }
                }
                Step::AttentionBackward {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    dy_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    window,
                    wrt,
                } => {
                    let kernel = attention_bwd_kernel(&self.ctx);
                    let seq_axis = if *wrt == 0 { *seq_q } else { *seq_k };
                    let y_blocks = seq_axis.div_ceil(256);
                    let cfg = LaunchConfig {
                        grid_dim: (batch * heads, y_blocks, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(heads)
                        .arg(seq_q)
                        .arg(seq_k)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window)
                        .arg(wrt);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention_bwd launch failed");
                    }
                }
                Step::Rope {
                    n_total,
                    seq,
                    head_dim,
                    half,
                    rot_half,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off,
                    last_dim,
                    interleaved,
                } => {
                    let kernel = rope_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n_total)
                        .arg(seq)
                        .arg(head_dim)
                        .arg(half)
                        .arg(rot_half)
                        .arg(in_off)
                        .arg(cos_off)
                        .arg(sin_off)
                        .arg(out_off)
                        .arg(last_dim)
                        .arg(interleaved);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: rope launch failed");
                    }
                }
                Step::Cumsum {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    exclusive,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = cumsum_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(outer_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(exclusive);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: cumsum launch failed");
                    }
                }
                Step::TopK {
                    outer,
                    inner,
                    k,
                    in_off,
                    out_off,
                } => {
                    let kernel = topk_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(k)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: topk launch failed");
                    }
                }
                Step::GroupedMatmul {
                    m,
                    k,
                    n,
                    num_experts,
                    in_off,
                    w_off,
                    idx_off,
                    out_off,
                } => {
                    // Tier 1: sorted-batch dispatch via cuBLAS. Reads
                    // the idx buffer back to host, finds runs of
                    // identical consecutive expert ids, and issues one
                    // cublasSgemm per run. Wins big when tokens are
                    // pre-sorted by expert (the standard MoE upstream
                    // convention) — for random idx the run count is
                    // ~m and the launch overhead would negate the win,
                    // so we fall back to the kernel in that case.
                    let used_sorted = if let Some(blas) = self.blas.as_ref() {
                        // Sync first so prior writes to idx are visible.
                        stream
                            .synchronize()
                            .expect("rlx-cuda: stream sync before idx download");
                        let idx_host = {
                            let idx_slot = self
                                .arena
                                .f32_buf()
                                .slice(*idx_off as usize..(idx_off + m) as usize);
                            stream.clone_dtoh(&idx_slot).ok()
                        };
                        match idx_host {
                            Some(idx_vec) => {
                                let mut runs: Vec<(u32, u32, u32)> = Vec::new();
                                let mut i = 0usize;
                                let mn = *m as usize;
                                while i < mn {
                                    let e = idx_vec[i] as u32;
                                    let mut j = i + 1;
                                    while j < mn && (idx_vec[j] as u32) == e {
                                        j += 1;
                                    }
                                    if e < *num_experts {
                                        runs.push((i as u32, j as u32, e));
                                    }
                                    i = j;
                                }
                                // Heuristic: bail when the run count
                                // exceeds m/4 (idx isn't usefully sorted).
                                let threshold = (mn / 4).max(2);
                                if !runs.is_empty() && runs.len() <= threshold {
                                    let blas = blas.lock().unwrap();
                                    let (arena_ptr, _record) =
                                        self.arena.f32_buf_mut().device_ptr_mut(&stream);
                                    let alpha: f32 = 1.0;
                                    let beta: f32 = 0.0;
                                    let mut all_ok = true;
                                    for (lo, hi, e) in &runs {
                                        let rows = hi - lo;
                                        let a_dev = arena_ptr + ((*in_off + lo * k) as u64) * 4;
                                        let b_dev = arena_ptr + ((*w_off + e * k * n) as u64) * 4;
                                        let c_dev = arena_ptr + ((*out_off + lo * n) as u64) * 4;
                                        let r = unsafe {
                                            cudarc::cublas::result::sgemm(
                                                *blas.handle(),
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                                *n as i32,
                                                rows as i32,
                                                *k as i32,
                                                &alpha as *const f32,
                                                b_dev as *const f32,
                                                *n as i32,
                                                a_dev as *const f32,
                                                *k as i32,
                                                &beta as *const f32,
                                                c_dev as *mut f32,
                                                *n as i32,
                                            )
                                        };
                                        if r.is_err() {
                                            all_ok = false;
                                            break;
                                        }
                                    }
                                    all_ok
                                } else {
                                    false
                                }
                            }
                            None => false,
                        }
                    } else {
                        false
                    };
                    if used_sorted {
                        continue;
                    }

                    // Fallback: per-token expert lookup kernel.
                    let kernel = grouped_matmul_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(num_experts)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: grouped_matmul launch failed");
                    }
                }
                Step::ScatterAddZero { out_off, out_total } => {
                    let kernel = scatter_add_zero_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(out_off)
                        .arg(out_total);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_zero launch failed");
                    }
                }
                Step::ScatterAddAcc {
                    out_off,
                    upd_off,
                    idx_off,
                    num_updates,
                    trailing,
                    out_dim,
                } => {
                    let kernel = scatter_add_acc_kernel(&self.ctx);
                    let total = num_updates * trailing;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(out_off)
                        .arg(upd_off)
                        .arg(idx_off)
                        .arg(num_updates)
                        .arg(trailing)
                        .arg(out_dim);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_acc launch failed");
                    }
                }
                Step::DequantMatmul {
                    m,
                    k,
                    n,
                    block_size,
                    scheme_id,
                    x_off,
                    w_off,
                    scale_off,
                    zp_off,
                    out_off,
                } => {
                    let kernel = dequant_matmul_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(block_size)
                        .arg(scheme_id)
                        .arg(x_off)
                        .arg(w_off)
                        .arg(scale_off)
                        .arg(zp_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: dequant_matmul launch failed");
                    }
                }
                Step::DequantMatmulGguf {
                    m,
                    k,
                    n,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    // Decode GEMV (m=1, Q4_K/Q6_K): fused on-device kernel — parity
                    // with rlx-vulkan and rlx-cpu `gguf_matmul_bt`. Prefill (m>1)
                    // uses dequant_gguf + `matmul_bt`.
                    let fused_gemv = crate::gguf_gpu::gguf_fused_gemv_m1_supported(
                        *scheme_id,
                        *m as usize,
                        *k as usize,
                    ) && rlx_ir::env::var("ORPHEUS_CUDA_GGUF_FUSED_M1").as_deref()
                        != Some("0");
                    if fused_gemv {
                        crate::gguf_gpu::run_dequant_matmul_gguf_gemv_m1(
                            &self.ctx,
                            &stream,
                            self.arena.f32_buf_mut(),
                            *n as usize,
                            *k as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    } else {
                        // Keep the dequant+matmul on-device by DEFAULT, including
                        // decode (m=1) for schemes without a fused GEMV (e.g. Q5_0,
                        // Q8_0). Host dequant per decode step is ~6x slower end-to-end
                        // (gemma3-270m: 3.1 → 18.6 tok/s on RTX 3080 Ti). Opt out with
                        // RLX_CUDA_GGUF_HOST=1.
                        let use_gpu = self.dequant_scratch_off > 0
                            && rlx_ir::env::var("RLX_CUDA_GGUF_HOST").as_deref() != Some("1");
                        if use_gpu {
                            crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                *m as usize,
                                *k as usize,
                                *n as usize,
                                *scheme_id,
                                *x_byte_off as usize,
                                *w_byte_off as usize,
                                self.dequant_scratch_off,
                                *out_byte_off as usize,
                            );
                        } else {
                            crate::gguf_host::run_dequant_matmul_gguf(
                                &stream,
                                self.arena.f32_buf_mut(),
                                *m as usize,
                                *k as usize,
                                *n as usize,
                                *scheme_id,
                                *x_byte_off as usize,
                                *w_byte_off as usize,
                                *out_byte_off as usize,
                            );
                        }
                    }
                }
                Step::DequantGroupedMatmulGguf {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    let use_gpu = self.dequant_scratch_off > 0;
                    if use_gpu {
                        crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                            &self.ctx,
                            &stream,
                            self.arena.f32_buf_mut(),
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &stream,
                            self.arena.f32_buf_mut(),
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
                }
                Step::Sample {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    top_k,
                    top_p_bits,
                    temp_bits,
                    seed_lo,
                    seed_hi,
                } => {
                    let kernel = sample_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(top_k)
                        .arg(top_p_bits)
                        .arg(temp_bits)
                        .arg(seed_lo)
                        .arg(seed_hi);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: sample launch failed");
                    }
                }
                Step::RngNormal {
                    dst_byte_off,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_normal(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *dst_byte_off as usize,
                        *len as usize,
                        *mean,
                        *scale,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                Step::RngUniform {
                    dst_byte_off,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_uniform(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *dst_byte_off as usize,
                        *len as usize,
                        *low,
                        *high,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                Step::SelectiveScan {
                    batch,
                    seq,
                    hidden,
                    state_size,
                    x_off,
                    delta_off,
                    a_off,
                    b_off,
                    c_off,
                    out_off,
                } => {
                    let kernel = selective_scan_kernel(&self.ctx);
                    let total = batch * hidden;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(seq)
                        .arg(hidden)
                        .arg(state_size)
                        .arg(x_off)
                        .arg(delta_off)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: selective_scan launch failed");
                    }
                }
                Step::Fft {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype_tag,
                    use_gpu,
                    real_input,
                } => {
                    if *use_gpu {
                        let norm = rlx_ir::fft::FftNorm::from_tag(*norm_tag);
                        let scale = norm.output_scale(*n_complex as usize, *inverse) as f32;
                        // Backend precedence for the GPU FFT op: native Stockham
                        // (n≤4096) → cuFFT → native multi/single-kernel. Each is
                        // behind its own feature; with none on, only the last arm
                        // compiles. `real_input` (the fused real→complex path) is
                        // only the native kernel can read, so it forces this arm.
                        #[allow(unused_mut)]
                        let mut handled = false;
                        let _ = real_input;

                        #[cfg(feature = "native-cuda-fft")]
                        if !handled
                            && (*real_input
                                || (crate::native_fft_dispatch::stockham_enabled()
                                    && crate::native_fft_dispatch::stockham_eligible(*n_complex)))
                        {
                            crate::native_fft_dispatch::run_fft_native_stockham(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                *src_byte_off / 4,
                                *dst_byte_off / 4,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                                *real_input,
                            );
                            handled = true;
                        }

                        #[cfg(feature = "cufft")]
                        if !handled && crate::cufft_dispatch::cufft_should_use(*n_complex) {
                            crate::cufft_dispatch::run_fft_cufft(
                                &self.ctx,
                                &stream,
                                &mut self.cufft_state,
                                self.arena.f32_buf_mut(),
                                *src_byte_off / 4,
                                *dst_byte_off / 4,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                            );
                            handled = true;
                        }

                        if !handled {
                            crate::fft_dispatch::run_fft_gpu(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                *src_byte_off / 4,
                                *dst_byte_off / 4,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                            );
                        }
                    } else {
                        let (buf, arena_size) = self.arena.f32_buf_and_size();
                        crate::fft_host::run_fft1d(
                            &stream,
                            buf,
                            arena_size,
                            *src_byte_off as usize,
                            *dst_byte_off as usize,
                            *outer as usize,
                            *n_complex as usize,
                            *inverse,
                            *norm_tag,
                            fft_dtype_from_tag(*dtype_tag),
                        );
                    }
                }
                Step::WelchPeaksGpu {
                    spec_off,
                    dst_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                    n_bins,
                } => {
                    crate::welch_peaks_dispatch::run_welch_peaks_gpu(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *spec_off,
                        *dst_off,
                        *welch_batch,
                        *n_fft,
                        *n_segments,
                        *k,
                        *n_bins,
                    );
                }
                Step::LogMelHost { .. }
                | Step::LogMelBackwardHost { .. }
                | Step::WelchPeaksHost { .. } => {}
                Step::Im2ColHost {
                    x_byte_off,
                    col_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    use_gpu,
                } => {
                    if *use_gpu {
                        let kernel = im2col_kernel(&self.ctx);
                        let m = *n * *h_out * *w_out;
                        let k = *c_in * *kh * *kw;
                        let total = m * k;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let x_off = *x_byte_off / 4;
                        let col_off = *col_byte_off / 4;
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(n)
                            .arg(c_in)
                            .arg(h)
                            .arg(w)
                            .arg(h_out)
                            .arg(w_out)
                            .arg(kh)
                            .arg(kw)
                            .arg(sh)
                            .arg(sw)
                            .arg(ph)
                            .arg(pw)
                            .arg(dh)
                            .arg(dw_dil)
                            .arg(&x_off)
                            .arg(&col_off);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: im2col launch failed");
                        }
                    } else {
                        crate::im2col_host::run_im2col(
                            &stream,
                            self.arena.f32_buf_mut(),
                            *x_byte_off as usize,
                            *col_byte_off as usize,
                            *n,
                            *c_in,
                            *h,
                            *w,
                            *h_out,
                            *w_out,
                            *kh,
                            *kw,
                            *sh,
                            *sw,
                            *ph,
                            *pw,
                            *dh,
                            *dw_dil,
                        );
                    }
                }
                Step::ReverseHost {
                    src_byte_off,
                    dst_byte_off,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    crate::host_misc::run_reverse(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        dims,
                        rev_mask,
                        *elem_bytes as usize,
                    );
                }
                Step::ArgReduceHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    reduced,
                    inner,
                    is_max,
                } => {
                    crate::host_misc::run_argreduce(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *reduced as usize,
                        *inner as usize,
                        *is_max,
                    );
                }
                Step::AxialRope2dHost {
                    src_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    hidden,
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    crate::host_misc::run_axial_rope2d(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *hidden as usize,
                        *end_x as usize,
                        *end_y as usize,
                        *head_dim as usize,
                        *num_heads as usize,
                        *theta,
                        *repeat_factor as usize,
                    );
                }
                Step::GatedDeltaNet {
                    q_byte_off,
                    k_byte_off,
                    v_byte_off,
                    g_byte_off,
                    beta_byte_off,
                    state_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    heads,
                    state_size,
                    use_carry,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::gdn_host::run_gated_delta_net(
                        &stream,
                        buf,
                        arena_size,
                        *q_byte_off as usize,
                        *k_byte_off as usize,
                        *v_byte_off as usize,
                        *g_byte_off as usize,
                        *beta_byte_off as usize,
                        *state_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *heads as usize,
                        *state_size as usize,
                        *use_carry,
                    );
                }
                Step::Lstm {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    bias_byte_off,
                    h0_byte_off,
                    c0_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::lstm_host::run_lstm(
                        &stream,
                        buf,
                        arena_size,
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *bias_byte_off as usize,
                        *h0_byte_off as usize,
                        *c0_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                    );
                }
                Step::ScanHost {
                    plan,
                    outer_init_off,
                    outer_final_off,
                    length,
                    save_trajectory,
                    xs_outer,
                    bcast_outer,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::scan_host::run_scan(
                        &stream,
                        buf,
                        arena_size,
                        plan,
                        *outer_init_off,
                        *outer_final_off,
                        *length,
                        *save_trajectory,
                        xs_outer,
                        bcast_outer,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_off,
                    route_off,
                    out_off,
                    n_elems,
                    attrs,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &stream,
                        buf,
                        arena_size,
                        *sig_off as usize,
                        *route_off as usize,
                        *out_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::MsDeformAttnHost {
                    in_offs,
                    out_off,
                    out_len,
                    attrs,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::ms_deform_attn_host::run_ms_deform_attn(
                        &stream,
                        buf,
                        arena_size,
                        in_offs,
                        *out_off as usize,
                        *out_len as usize,
                        attrs,
                    );
                }
                Step::UmapKnn {
                    pairwise_off,
                    out_off,
                    n,
                    k,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::umap_knn_host::run_umap_knn(
                        &stream,
                        buf,
                        arena_size,
                        *pairwise_off as usize,
                        *out_off as usize,
                        *n as usize,
                        *k as usize,
                    );
                }
                Step::LayerNorm2d {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    eps_bits,
                } => {
                    let kernel = layer_norm2d_kernel(&self.ctx);
                    let total = n * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layer_norm2d launch failed");
                    }
                }
                Step::ConvTranspose2d {
                    src_off,
                    w_off,
                    dst_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                } => {
                    let kernel = conv_transpose2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(w_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c_in)
                        .arg(h)
                        .arg(w_in)
                        .arg(c_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv_transpose2d launch failed");
                    }
                }
                Step::GroupNorm {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps_bits,
                } => {
                    let kernel = group_norm_kernel(&self.ctx);
                    let grid = n * num_groups;
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(num_groups)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm launch failed");
                    }
                }
                Step::ResizeNearest2x {
                    src_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    let kernel = resize_nearest_2x_kernel(&self.ctx);
                    let total = n * c * h * 2 * w * 2;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: resize_nearest_2x launch failed");
                    }
                }
                Step::GaussianSplatRender {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    dst_off,
                    dst_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    #[cfg(feature = "native-splat")]
                    crate::splat_native::run_gaussian_splat_render_native(
                        &stream,
                        buf,
                        arena_size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                    #[cfg(not(feature = "native-splat"))]
                    crate::splat_host::run_gaussian_splat_render(
                        &stream,
                        buf,
                        arena_size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
                        *dst_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                Step::GaussianSplatPrepare {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    meta_len,
                    prep_off,
                    prep_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_prepare(
                        &stream,
                        buf,
                        arena_size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *prep_off as usize,
                        *prep_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                Step::GaussianSplatRasterize {
                    prep_off,
                    prep_len,
                    meta_off,
                    meta_len,
                    dst_off,
                    dst_len,
                    count,
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_rasterize(
                        &stream,
                        buf,
                        arena_size,
                        *prep_off as usize,
                        *prep_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *dst_off as usize,
                        *dst_len as usize,
                        *count as usize,
                        *width,
                        *height,
                        *tile_size,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                Step::GaussianSplatRenderBackward {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    d_loss_off,
                    d_loss_len,
                    packed_off,
                    packed_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_render_backward(
                        &stream,
                        buf,
                        arena_size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *d_loss_off as usize,
                        *d_loss_len as usize,
                        *packed_off as usize,
                        *packed_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                        *loss_grad_clip,
                        *sh_band,
                        *max_anisotropy,
                    );
                }
                Step::RmsNormBackwardInput {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dx_byte_off / 4,
                        *eps_bits,
                        0,
                    );
                }
                Step::RmsNormBackwardGamma {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dgamma_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dgamma_byte_off / 4,
                        *eps_bits,
                        1,
                    );
                }
                Step::RmsNormBackwardBeta {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dbeta_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dbeta_byte_off / 4,
                        *eps_bits,
                        2,
                    );
                }
                Step::RopeBackward {
                    dy_byte_off,
                    cos_byte_off,
                    sin_byte_off,
                    dx_byte_off,
                    batch,
                    seq,
                    hidden,
                    head_dim,
                    n_rot,
                    cos_len,
                } => {
                    launch_rope_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *batch,
                        *seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
                        *dy_byte_off / 4,
                        *cos_byte_off / 4,
                        *sin_byte_off / 4,
                        *dx_byte_off / 4,
                        *cos_len,
                    );
                }
                Step::CumsumBackward {
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    cols,
                    exclusive,
                } => {
                    launch_cumsum_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *cols,
                        *dy_byte_off / 4,
                        *dx_byte_off / 4,
                        if *exclusive { 1 } else { 0 },
                    );
                }
                Step::GatherBackward {
                    dy_byte_off,
                    indices_byte_off,
                    dst_byte_off,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                } => {
                    launch_gather_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
                        *dy_byte_off / 4,
                        *indices_byte_off / 4,
                        *dst_byte_off / 4,
                    );
                }
                Step::MaxPool2dBackward {
                    x_byte_off,
                    dy_byte_off,
                    dx_byte_off,
                    n,
                    c,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                } => {
                    let kernel = maxpool2d_backward_kernel(&self.ctx);
                    let total = n * c * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let x_o = *x_byte_off / 4;
                    let dy_o = *dy_byte_off / 4;
                    let dx_o = *dx_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(&x_o)
                        .arg(&dy_o)
                        .arg(&dx_o);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: maxpool2d_backward launch failed");
                    }
                }
                Step::Conv2dBackwardInput {
                    dy_byte_off,
                    w_byte_off,
                    dx_byte_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                } => {
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_backward_data(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *h,
                                *w_in,
                                *h_out,
                                *w_out,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *ph,
                                *pw,
                                *dh,
                                *dw,
                                *groups,
                                *dy_byte_off / 4,
                                *w_byte_off / 4,
                                *dx_byte_off / 4,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d_bwd_data.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if !used_cudnn {
                        let buf = self.arena.f32_buf_mut();
                        crate::training_bwd_host::run_conv2d_backward_input(
                            &stream,
                            buf,
                            *dy_byte_off as usize / 4,
                            *w_byte_off as usize / 4,
                            *dx_byte_off as usize / 4,
                            *n,
                            *c_in,
                            *h,
                            *w_in,
                            *c_out,
                            *h_out,
                            *w_out,
                            *kh,
                            *kw,
                            *sh,
                            *sw,
                            *ph,
                            *pw,
                            *dh,
                            *dw,
                            *groups,
                        );
                    }
                }
                Step::Conv2dBackwardWeight {
                    x_byte_off,
                    dy_byte_off,
                    dw_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    groups,
                } => {
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_backward_filter(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *h,
                                *w,
                                *h_out,
                                *w_out,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *ph,
                                *pw,
                                *dh,
                                *dw_dil,
                                *groups,
                                *x_byte_off / 4,
                                *dy_byte_off / 4,
                                *dw_byte_off / 4,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d_bwd_filter.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if !used_cudnn {
                        let buf = self.arena.f32_buf_mut();
                        crate::training_bwd_host::run_conv2d_backward_weight(
                            &stream,
                            buf,
                            *x_byte_off as usize / 4,
                            *dy_byte_off as usize / 4,
                            *dw_byte_off as usize / 4,
                            *n,
                            *c_in,
                            *h,
                            *w,
                            *c_out,
                            *h_out,
                            *w_out,
                            *kh,
                            *kw,
                            *sh,
                            *sw,
                            *ph,
                            *pw,
                            *dh,
                            *dw_dil,
                            *groups,
                        );
                    }
                }
                Step::Pool1d {
                    n,
                    c,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool1d_kernel(&self.ctx);
                    let total = n * c * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool1d launch failed");
                    }
                }
                Step::Pool2d {
                    n,
                    c,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool2d_kernel(&self.ctx);
                    let total = n * c * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool2d launch failed");
                    }
                }
                Step::Pool3d {
                    n,
                    c,
                    d,
                    h,
                    w,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool3d_kernel(&self.ctx);
                    let total = n * c * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool3d launch failed");
                    }
                }
                Step::Conv1d {
                    n,
                    c_in,
                    c_out,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    dl,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN — 1-D conv as a degenerate 2-D conv
                    // with H=1, kh=1, sh=1, ph=0, dh=1. Same descriptors
                    // as conv2d; the H axis just collapses to 1.
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                /*h*/ 1,
                                *l,
                                /*h_out*/ 1,
                                *l_out,
                                /*kh*/ 1,
                                *kl,
                                /*sh*/ 1,
                                *sl,
                                /*ph*/ 0,
                                *pl,
                                /*dh*/ 1,
                                *dl,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv1d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv1d_kernel(&self.ctx);
                    let total = n * c_out * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(dl)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv1d launch failed");
                    }
                }
                Step::Conv2d {
                    n,
                    c_in,
                    c_out,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN — picks the fastest algo via the v7
                    // heuristic for the supplied shape + workspace size.
                    // Matmul parity (RLX_CUDA_PARITY) must not disable cuDNN conv — the
                    // custom conv2d.cu fallback drifts vs CPU on Deep4; cuDNN matches CPU.
                    let try_cudnn = self.dnn.is_some()
                        && self.dnn_workspace.is_some()
                        && !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN");
                    let used_cudnn = if try_cudnn {
                        let handle = self.dnn.expect("dnn handle");
                        let workspace = self.dnn_workspace.as_ref().expect("dnn workspace");
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *h,
                                *w,
                                *h_out,
                                *w_out,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *ph,
                                *pw,
                                *dh,
                                *dw,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel (cuDNN preferred via PATH).
                    let kernel = conv2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv2d launch failed");
                    }
                }
                Step::Conv3d {
                    n,
                    c_in,
                    c_out,
                    d,
                    h,
                    w,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN nd-conv (NCDHW + 3-D pads/strides/dilations).
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv3d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *d,
                                *h,
                                *w,
                                *d_out,
                                *h_out,
                                *w_out,
                                *kd,
                                *kh,
                                *kw,
                                *sd,
                                *sh,
                                *sw,
                                *pd,
                                *ph,
                                *pw,
                                *dd,
                                *dh,
                                *dw,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv3d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(dd)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv3d launch failed");
                    }
                }
            }

            // Multi-stream tail: record an event so future steps can
            // wait on this one, then update producer_of with the
            // offsets this step wrote.
            if let Some(idx) = assigned_idx {
                if let Ok(evt) = stream.record_event(None) {
                    last_event.insert(idx, evt);
                }
                let (_, writes) = step_offsets(step);
                for w in &writes {
                    producer_of.insert(*w, idx);
                }
            }
        }

        // Multi-stream: sync every pool stream so output reads see all
        // produced data.
        if multi_stream {
            for s in &self.streams {
                let _ = s.synchronize();
            }
        }

        self.prepare_readback_plan();
        let plan = self.readback_plan_buf.clone();
        run_tail_host_audio_ops(&self.schedule, &stream, self.arena.f32_buf_mut(), true);
        if !self.gpu_handle_feeds.is_empty() {
            self.propagate_gpu_handle_feeds_d2d(&stream);
        }
        let read_all = plan.len() == self.graph.outputs.len();

        if capturing {
            // End capture before dtoh — the graph records compute kernels only.
            let cu_graph = stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags
                    ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
            ).expect("rlx-cuda: end_capture failed");
            if let Some(g) = cu_graph {
                g.upload().expect("rlx-cuda: graph upload failed");
                g.launch().expect("rlx-cuda: graph first launch failed");
                self.captured_graph = Some(g);
                self.captured_readback_plan = Some(plan.clone());
            }
        }

        if read_all {
            self.fill_output_staging(&stream)
                .expect("rlx-cuda: output dtoh failed");
        } else {
            self.fill_output_staging_indices(&stream, &plan)
                .expect("rlx-cuda: partial output dtoh failed");
        }
        self.refresh_gpu_handles_from_staging(&plan);
        stream.synchronize().expect("rlx-cuda: stream sync failed");
        self.outputs_from_staging_plan(&plan)
    }

    fn fill_output_staging_indices(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        indices: &[usize],
    ) -> Result<(), cudarc::driver::DriverError> {
        for &i in indices {
            let id = self.graph.outputs[i];
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            debug_assert_eq!(self.output_staging[i].len(), elems);
            let slot = self.arena.f32_buf().slice(off_f32..off_f32 + elems);
            self.output_staging[i].dtoh(stream, &slot)?;
        }
        Ok(())
    }

    fn outputs_from_staging_plan(&self, plan: &[usize]) -> Vec<Vec<f32>> {
        if plan.len() == self.graph.outputs.len() {
            return self.outputs_from_staging();
        }
        plan.iter()
            .map(|&i| self.output_staging[i].to_vec())
            .collect()
    }

    fn fill_output_staging(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), cudarc::driver::DriverError> {
        for (i, &id) in self.graph.outputs.iter().enumerate() {
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            debug_assert_eq!(self.output_staging[i].len(), elems);
            let slot = self.arena.f32_buf().slice(off_f32..off_f32 + elems);
            self.output_staging[i].dtoh(stream, &slot)?;
        }
        Ok(())
    }

    fn outputs_from_staging(&self) -> Vec<Vec<f32>> {
        self.output_staging
            .iter()
            .map(F32HostSlot::to_vec)
            .collect()
    }
}

fn launch_cumsum_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    inner: u32,
    dy_off: u32,
    dx_off: u32,
    exclusive: u32,
) {
    let kernel = cumsum_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(outer, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&outer)
        .arg(&inner)
        .arg(&dy_off)
        .arg(&dx_off)
        .arg(&exclusive);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: cumsum_bwd launch failed");
    }
}

fn launch_rope_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dy_off: u32,
    cos_off: u32,
    sin_off: u32,
    dx_off: u32,
    cos_len: u32,
) {
    let total = batch * seq * hidden;
    let kernel = rope_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(total, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&batch)
        .arg(&seq)
        .arg(&hidden)
        .arg(&head_dim)
        .arg(&n_rot)
        .arg(&dy_off)
        .arg(&cos_off)
        .arg(&sin_off)
        .arg(&dx_off)
        .arg(&cos_len);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rope_bwd launch failed");
    }
}

fn launch_gather_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    dy_off: u32,
    idx_off: u32,
    dst_off: u32,
) {
    let total = outer * axis_dim * trailing;
    if total > 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&dst_off).arg(&total);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: gather_bwd zero launch failed");
        }
    }
    let kernel = gather_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (outer, (num_idx * trailing).div_ceil(256), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&outer)
        .arg(&axis_dim)
        .arg(&num_idx)
        .arg(&trailing)
        .arg(&dy_off)
        .arg(&idx_off)
        .arg(&dst_off);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: gather_bwd launch failed");
    }
}

fn launch_rms_norm_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    rows: u32,
    inner: u32,
    x_off: u32,
    gamma_off: u32,
    beta_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    wrt: u32,
) {
    if wrt != 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(inner, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&out_off).arg(&inner);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: rms_norm_bwd zero launch failed");
        }
    }
    let kernel = rms_norm_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&rows)
        .arg(&inner)
        .arg(&x_off)
        .arg(&gamma_off)
        .arg(&beta_off)
        .arg(&dy_off)
        .arg(&out_off)
        .arg(&eps_bits)
        .arg(&wrt);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rms_norm_bwd launch failed");
    }
}

#[cfg(test)]
mod tests {
    //! Pure-function tests for the multi-stream scheduler analysis and
    //! the element-wise fusion pass. Both are pure Rust against
    //! synthesized `Vec<Step>` inputs — no CUDA driver needed, so they
    //! run on Mac.
    use super::*;

    #[test]
    fn normalize_read_indices_dedupes() {
        let mut v = vec![3, 1, 2, 1, 0];
        normalize_read_indices(&mut v);
        assert_eq!(v, vec![0, 1, 2, 3]);
    }

    #[test]
    fn step_offsets_binary() {
        let s = Step::Binary {
            n: 8,
            a_off: 100,
            b_off: 200,
            c_off: 300,
            op: 0,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![100, 200]);
        assert_eq!(w, vec![300]);
    }

    #[test]
    fn step_offsets_matmul_with_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 1,
            bias_off_f32: 40,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20, 40]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_matmul_no_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 0,
            bias_off_f32: 0,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_attention_causal_no_mask_arg() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 1, // causal — mask_off ignored
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(!r.contains(&9999), "causal mask must not consume mask_off");
        assert_eq!(r, vec![0, 100, 200]);
    }

    #[test]
    fn step_offsets_attention_custom_mask_pulls_mask() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 2, // custom mask
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(r.contains(&9999));
    }

    #[test]
    fn step_offsets_scatter_add_acc_marks_out_as_rmw() {
        let s = Step::ScatterAddAcc {
            out_off: 100,
            upd_off: 200,
            idx_off: 300,
            num_updates: 4,
            trailing: 1,
            out_dim: 16,
        };
        let (r, w) = step_offsets(&s);
        // out is read-modify-write, so it appears in BOTH reads and writes
        // — this lets the multi-stream scheduler force the prior
        // ScatterAddZero to complete before the accumulate launches.
        assert!(r.contains(&100));
        assert!(w.contains(&100));
    }

    #[test]
    fn fuse_elementwise_merges_binary_then_unary() {
        let schedule = vec![
            // c = a + b
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            // d = relu(c)
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 1, "expected exactly one fused step");
        match &fused[0] {
            Step::FusedBinaryUnary {
                n,
                a_off,
                b_off,
                out_off,
                bin_op,
                un_op,
            } => {
                assert_eq!(*n, 4);
                assert_eq!(*a_off, 0);
                assert_eq!(*b_off, 4);
                assert_eq!(*out_off, 12);
                assert_eq!(*bin_op, 0);
                assert_eq!(*un_op, 0);
            }
            other => panic!("expected FusedBinaryUnary, got {}", step_name(other)),
        }
    }

    #[test]
    fn fuse_elementwise_skips_when_intermediate_has_two_consumers() {
        // c = a + b
        // d = relu(c)
        // e = c * c   ← second consumer of c, blocks fusion
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 8,
                b_off: 8,
                c_off: 16,
                op: 2,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 3, "no fusion: c has multiple consumers");
        assert!(matches!(&fused[0], Step::Binary { .. }));
        assert!(matches!(&fused[1], Step::Unary { .. }));
        assert!(matches!(&fused[2], Step::Binary { .. }));
    }

    #[test]
    fn fuse_elementwise_skips_when_n_mismatch() {
        // Different element counts → can't fuse (different launch grid).
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 8,
                in_off: 8,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_skips_when_unary_input_isnt_binary_output() {
        // Unary reads a different offset than what Binary wrote.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 99,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_handles_multiple_chains() {
        // Two independent Binary→Unary chains in a row — both should fuse.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 16,
                b_off: 20,
                c_off: 24,
                op: 2,
            },
            Step::Unary {
                n: 4,
                in_off: 24,
                out_off: 28,
                op: 9,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
        assert!(matches!(&fused[0], Step::FusedBinaryUnary { .. }));
        assert!(matches!(&fused[1], Step::FusedBinaryUnary { .. }));
    }
}
