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

//! `RocmExecutable` — sister to `rlx-cuda::CudaExecutable`.
//!
//! Full IR walk, memory plan, Step emission, and HIP kernel dispatch
//! mirroring `rlx-cuda` with `HipBuffer` / hipBLAS / MIOpen types.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{Graph, NodeId};

use std::sync::Mutex;

use crate::arena::{Arena, HalfDtype};
use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;

const MIOPEN_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const HIPBLASLT_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

// ── Step enum ─────────────────────────────────────────────────────────
// Copy of `rlx-cuda::backend::Step` — same variants, same fields.
// Kept private to the crate; the public surface is `RocmExecutable`.

#[derive(Clone)]
pub(crate) enum Step {
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
    /// Native FP8 (FNUZ) tensor-core GEMM via hipBLASLt. TN: lhs[m,k]·rhs[n,k]ᵀ.
    /// All offsets are BYTES (codes u8; scales/out/bias f32).
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
    /// FP6 configs hipBLASLt can't do.
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
    /// Gated-DeltaNet — host scan between GPU segments.
    Fft {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype_tag: u32,
        use_gpu: bool,
    },
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
    /// (D2H → CPU body loop → H2D). Not graph-capture-safe.
    ScanHost {
        plan: std::sync::Arc<rlx_cpu::thunk::ScanBodyPlan>,
        outer_init_off: usize,
        outer_final_off: usize,
        length: u32,
        save_trajectory: bool,
        xs_outer: Vec<(usize, usize)>,
        bcast_outer: Vec<(usize, usize)>,
    },
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
    ResizeNearest2x {
        src_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
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
    /// Kernel source shared with rlx-cuda (`elementwise_region.cu`).
    /// `input_offs` mirrors what's packed in `meta` and is kept in
    /// the Step so the multi-stream scheduler can resolve
    /// producer-consumer dependencies without unpacking metadata.
    ElementwiseRegion {
        len: u32,
        num_inputs: u32,
        num_steps: u32,
        dst_off: u32,
        input_offs: [u32; 16],
        /// PLAN L2 quality fast path: per-input scalar-broadcast bitfield.
        scalar_input_mask: u32,
        /// PLAN L2 quality general broadcast: per-input element count.
        /// `0` ⇒ no broadcast (kernel reads gid); `>0` ⇒ kernel reads
        /// `arena[input_offs[i] + (gid % input_modulus[i])]`.
        input_modulus: [u32; 16],
        meta_idx: usize,
        spatial_prologue: bool,
        prologue_w: u32,
        prologue_h: u32,
        prologue_nc: u32,
    },
    BatchElementwiseRegion {
        slice_len: u32,
        num_batch: u32,
        num_steps: u32,
        base_dst_off: u32,
        slice_elems: u32,
        batch_input_offs: [u32; 64],
        batch_offs_idx: usize,
        meta_idx: usize,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
    },
}

// ── Modes ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    #[default]
    Jit,
    Aot,
}

/// `Stream` (default single-stream dispatch). `Graph` captures the
/// schedule into a hipGraph on first run and replays it on subsequent
/// runs — eliminates per-launch dispatch overhead. `Eager` is a
/// one-shot compile + run + drop helper. `MultiStream(n)` allocates a
/// pool of `n` streams and assigns each Step based on data
/// dependencies (same dep-aware scheduler as rlx-cuda).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    Stream,
    Graph,
    Eager,
    MultiStream(usize),
}

// ── log_fallback (port from rlx-cuda) ────────────────────────────────

pub(crate) fn log_fallback(tier: &str, err: impl std::fmt::Debug) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        rlx_ir::env::var("RLX_ROCM_LOG_FALLBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if enabled {
        eprintln!("rlx-rocm: tier '{tier}' fell back: {err:?}");
    }
}

// ── step_name (port from rlx-cuda) ────────────────────────────────────

fn rocm_fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-rocm Op::Fft: unsupported dtype {other:?}"),
    }
}

fn rocm_fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-rocm Op::Fft: bad dtype tag {other}"),
    }
}

pub(crate) fn step_name(step: &Step) -> &'static str {
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
        Step::Gather { .. } => "rlx::Gather",
        Step::GatherAxis { .. } => "rlx::GatherAxis",
        Step::Narrow { .. } => "rlx::Narrow",
        Step::Concat { .. } => "rlx::Concat",
        Step::Transpose { .. } => "rlx::Transpose",
        Step::Expand { .. } => "rlx::Expand",
        Step::Argmax { .. } => "rlx::Argmax",
        Step::Attention { .. } => "rlx::Attention",
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

// ── step_offsets (port from rlx-cuda) ─────────────────────────────────

pub(crate) fn step_offsets(step: &Step) -> (Vec<u32>, Vec<u32>) {
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
        } => (vec![*upd_off, *idx_off, *out_off], vec![*out_off]),
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
    }
}

// ── fuse_elementwise_chains (port from rlx-cuda) ──────────────────────

pub(crate) fn fuse_elementwise_chains(schedule: Vec<Step>) -> Vec<Step> {
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

// ── Op-id encoders + matmul shape (port from rlx-cuda) ───────────────

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
                "rlx-rocm {op_label}: batched shape mismatch \
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
            "rlx-rocm {op_label}: unsupported shapes a={a_shape:?} b={b_shape:?} out={out_shape:?}"
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

/// Upload a `&[u32]` to a freshly-allocated device buffer (analogue of
/// cudarc's `stream.clone_htod`). Used for transpose / expand meta
/// buffers.
fn upload_meta(ctx: &Arc<RocmContext>, data: &[u32]) -> HipBuffer<u32> {
    let mut buf = HipBuffer::<u32>::alloc_zeros(&ctx.runtime, data.len().max(1))
        .expect("rlx-rocm: meta upload alloc failed");
    buf.copy_from_host(data)
        .expect("rlx-rocm: meta upload htod failed");
    buf
}

/// Upload an arbitrary `&[f32]` slice to a specific arena offset
/// (used for Constant nodes during compile).
fn upload_to_arena(ctx: &Arc<RocmContext>, arena_ptr: HipDeviceptr, off_f32: usize, data: &[f32]) {
    let dst = arena_ptr + (off_f32 as u64) * 4;
    let bytes = std::mem::size_of_val(data);
    unsafe {
        let _ = (ctx.runtime.hip_memcpy_htod)(dst, data.as_ptr() as *const _, bytes);
    }
}

/// Opt-in MFMA / WMMA matrix-core kernel via rocWMMA. Reads
/// `RLX_ROCM_MFMA=1` once at process start. When true and the higher
/// tiers (mixed-precision, hipBLASLt, hipBLAS) all decline, the
/// matmul dispatch picks the matrix-core kernel instead of the
/// scalar fallback. The kernel will fail to compile under hipRTC on
/// archs without rocWMMA support; the cache miss surfaces as a
/// clean fallback through the normal panic path here, so we keep
/// this opt-in.
fn use_mfma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::var("RLX_ROCM_MFMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Mixed-precision matmul tier: when the weight (B input) is stored
/// in the half-arena, cast f32 activations to f16/bf16 in the scratch
/// buffer and run `hipblasGemmEx` with both inputs half + f32
/// accumulator. Returns `true` on success. Same shape as
/// `rlx-cuda::backend::try_mixed_precision_gemm` (free function so the
/// caller can hold `&self.schedule` across the call without violating
/// disjoint-field borrow checks).
fn try_mixed_precision_gemm_rocm(
    ctx: &Arc<RocmContext>,
    arena: &mut Arena,
    half_act_scratch: &mut Option<HipBuffer<u16>>,
    blas: Option<&Arc<Mutex<HipblasContext>>>,
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
    let need_resize = half_act_scratch.as_ref().is_none_or(|s| s.len < act_elems);
    if need_resize {
        *half_act_scratch = HipBuffer::<u16>::alloc_zeros(&ctx.runtime, act_elems.max(4)).ok();
    }
    if half_act_scratch.is_none() {
        return false;
    }

    // Phase 1: cast activations f32 → f16/bf16 into the scratch.
    let n_total = m * k * batch.max(1);
    let dtype_id: u32 = match half_dtype {
        HalfDtype::F16 => 0,
        HalfDtype::Bf16 => 1,
    };
    let stream = ctx.default_stream;
    let kernel = crate::kernels::cast_f32_to_half_kernel(ctx);
    let arena_base = arena.buffer.ptr;
    let scratch_ptr = half_act_scratch.as_ref().unwrap().ptr;
    // The cast kernel takes a `float*` source pointer (already at the
    // input offset) and a `unsigned short*` dest. We use raw pointer
    // values so the kernel reads from a_off + i.
    let src_dev = arena_base + (a_off_f32 as u64) * 4;
    let mut src_pp = src_dev;
    let mut dst_pp = scratch_ptr;
    crate::launch_kernel!(
        kernel,
        stream,
        (n_total.div_ceil(256), 1, 1),
        (256, 1, 1),
        [&mut src_pp, &mut dst_pp, &n_total, &dtype_id]
    );

    // Phase 2: hipblasGemmEx with both inputs half + f32 output.
    let blas = blas.lock().unwrap();
    let half_buf_ptr = match arena.half_buffer.as_ref() {
        Some(b) => b.ptr,
        None => return false,
    };
    let weight_dev = half_buf_ptr + (half_off as u64) * 2; // u16 = 2 bytes
    let act_dev = scratch_ptr;
    let c_dev = arena_base + (c_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let cuda_dt = match half_dtype {
        HalfDtype::F16 => HipblasDatatype::R16F,
        HalfDtype::Bf16 => HipblasDatatype::R16BF,
    };
    let compute_ty = match half_dtype {
        HalfDtype::F16 => HipblasComputeType::F32Fast16F,
        HalfDtype::Bf16 => HipblasComputeType::F32Fast16BF,
    };
    let result = unsafe {
        (blas.runtime.gemm_ex)(
            blas.handle,
            HipblasOperation::N,
            HipblasOperation::N,
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
            HipblasDatatype::R32F,
            n as i32,
            compute_ty,
            hipblas_gemm_default(),
        )
    };
    if let Err(e) = result.ok() {
        log_fallback("matmul.hipblasGemmEx (mixed)", e);
        return false;
    }
    true
}

// ── RocmExecutable ────────────────────────────────────────────────────

pub struct RocmExecutable {
    pub(crate) ctx: Arc<RocmContext>,
    /// hipBLAS handle bound to the same default stream as `ctx`. Used
    /// for plain matmul (no fused bias/activation); falls back to the
    /// custom kernel when libhipblas isn't available.
    pub(crate) blas: Option<Arc<Mutex<HipblasContext>>>,
    /// hipBLASLt handle for fused matmul + bias + relu/gelu. Falls
    /// back to plain sgemm + matmul_epilogue.cu when unavailable.
    pub(crate) blas_lt: Option<Arc<HipblasLtContext>>,
    /// 4 MiB scratch workspace for hipBLASLt heuristic-selected algos.
    pub(crate) blas_lt_workspace: Option<HipBuffer<u8>>,
    /// MIOpen handle for conv2d. Falls back to the custom direct-conv
    /// kernel when libMIOpen isn't available.
    pub(crate) dnn: Option<Arc<MiopenContext>>,
    /// Scratch workspace for MIOpen-selected conv algorithms (32 MiB
    /// — same shape as rlx-cuda's cuDNN workspace).
    pub(crate) dnn_workspace: Option<HipBuffer<u8>>,
    /// Byte offset in the f32 arena for GGUF dequant scratch (0 = none).
    pub(crate) dequant_scratch_off: usize,
    pub(crate) graph: Graph,
    pub(crate) arena: Arena,
    pub(crate) schedule: Vec<Step>,
    pub(crate) input_offsets: HashMap<String, NodeId>,
    pub(crate) param_offsets: HashMap<String, NodeId>,
    pub(crate) meta_buffers: Vec<HipBuffer<u32>>,
    pub(crate) exec_mode: ExecMode,
    pub(crate) half_act_scratch: Option<HipBuffer<u16>>,
    /// Captured hipGraphExec from `ExecMode::Graph`'s first-run
    /// capture; replayed via `hipGraphLaunch` on subsequent runs.
    pub(crate) captured_graph: Option<crate::hip::HipGraphExec>,
    /// Stream pool for `ExecMode::MultiStream(n)`. Empty otherwise.
    /// Each entry was created via `hipStreamCreate` and gets dropped
    /// when this struct is dropped.
    pub(crate) streams: Vec<crate::hip::HipStream>,
    /// Active-extent hint (PLAN L1). Mirrors rlx-cuda — bypasses
    /// hipGraph capture (recorded at full extent) when set + every
    /// step in the safe set.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Pinned or pageable host slots for output download.
    pub(crate) output_staging: Vec<F32HostSlot>,
    /// Pinned input staging when `RLX_ROCM_PINNED_IO=1` or graph mode.
    pub(crate) input_staging: HashMap<String, F32HostSlot>,
    /// Persistent KV inputs (host mirror + device upload each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// When set, only these output indices (+ feed outputs) are read back from device.
    pending_read_indices: Option<Vec<usize>>,
    /// Graph input names in declaration order (parallel to `input_slots`).
    input_slot_names: Vec<String>,
    /// Graph inputs in declaration order: `(arena_byte_offset, max_f32_elems)`.
    input_slots: Vec<(usize, usize)>,
    /// Host readback layout: `(byte_offset_in_host_arena, f32_elems)` per graph output.
    output_slots: Vec<(usize, usize)>,
    /// Pageable host mirror for `run_slots` / `arena_ptr` (not the GPU arena).
    host_arena: Vec<f32>,
    /// Runtime-mutable RNG policy for in-graph random ops.
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

impl RocmExecutable {
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.rng.write().expect("rng lock") = rng;
    }

    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Initial coverage matches rlx-cuda's: simple element-wise +
    /// reductions + softmax + LayerNorm + cumsum. Matmul and the
    /// rest still default to unsafe.
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
            | Step::ScanHost { .. }
            | Step::ReverseHost { .. }
            | Step::ArgReduceHost { .. }
            | Step::AxialRope2dHost { .. }
            | Step::GaussianSplatRender { .. }
            | Step::GaussianSplatRenderBackward { .. }
            | Step::GaussianSplatPrepare { .. }
            | Step::GaussianSplatRasterize { .. } => false,
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

fn im2col_use_gpu(n: u32, exec_mode: ExecMode) -> bool {
    if rlx_ir::env::var("RLX_ROCM_IM2COL_HOST").is_some() {
        return false;
    }
    if matches!(exec_mode, ExecMode::Graph) {
        return n > 0;
    }
    n > 0
}

fn pinned_io_enabled(exec_mode: ExecMode) -> bool {
    if matches!(exec_mode, ExecMode::Graph) {
        return true;
    }
    rlx_ir::env::var("RLX_ROCM_PINNED_IO").is_some_and(|v| !v.eq_ignore_ascii_case("0"))
}

impl Drop for RocmExecutable {
    fn drop(&mut self) {
        unsafe {
            if let Some(g) = self.captured_graph.take() {
                let _ = (self.ctx.runtime.hip_graph_exec_destroy)(g);
            }
            for s in self.streams.drain(..) {
                let _ = (self.ctx.runtime.hip_stream_destroy)(s);
            }
        }
    }
}

mod compile;
mod fill;
mod output;
mod run;
mod set;

impl RocmExecutable {
    /// One-shot eager run.
    pub fn eager(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut exec = Self::compile_with(graph, CompileMode::Jit, ExecMode::Eager);
        exec.run(inputs)
    }


    /// Host buffer base for reading outputs after [`Self::run_slots`].
    /// Offsets in the returned slot pairs are **byte** offsets into this buffer.
    pub fn arena_ptr(&self) -> *const u8 {
        self.host_arena.as_ptr() as *const u8
    }


    pub(crate) fn upload_slot_inputs(&mut self, inputs: &[&[f32]]) {
        let rt = &self.ctx.runtime;
        let arena_base = self.arena.buffer.ptr;
        for (i, data) in inputs.iter().enumerate() {
            let Some(&(byte_off, max_elems)) = self.input_slots.get(i) else {
                break;
            };
            let off_f32 = byte_off / 4;
            let len = data.len().min(max_elems);
            if len == 0 {
                continue;
            }
            let dst = arena_base + (off_f32 as u64) * 4;
            if let Some(name) = self.input_slot_names.get(i) {
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(&data[..len]);
                    host.htod(rt, dst, len)
                        .expect("rlx-rocm: pinned slot input upload failed");
                    continue;
                }
            }
            unsafe {
                let _ = (rt.hip_memcpy_htod)(
                    dst,
                    data.as_ptr() as *const _,
                    len * std::mem::size_of::<f32>(),
                );
            }
        }
    }


    pub(crate) fn pack_host_arena(&mut self) {
        let plan = self.readback_plan();
        for &i in &plan {
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


    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }


    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }


    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }


    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.gpu_handles.get(name).cloned()
    }


    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_rng(self.graph.clone(), self.rng());
        for (k, v) in &self.gpu_handles {
            exe.bind_gpu_handle(k, v);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }


    pub(crate) fn readback_plan(&self) -> Vec<usize> {
        let n = self.graph.outputs.len();
        if self.pending_read_indices.is_none() && self.gpu_handle_feeds.is_empty() {
            return (0..n).collect();
        }
        let mut set = std::collections::HashSet::new();
        if let Some(ref want) = self.pending_read_indices {
            set.extend(want.iter().copied());
        } else {
            return (0..n).collect();
        }
        for &idx in self.gpu_handle_feeds.values() {
            set.insert(idx);
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort_unstable();
        v
    }


    pub(crate) fn stage_gpu_handle_inputs(&mut self, inputs: &[(&str, &[f32])]) {
        let arena_base = self.arena.buffer.ptr;
        for (name, data) in &self.gpu_handles {
            if inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let dst = arena_base + (off_f32 as u64) * 4;
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    host.htod(&self.ctx.runtime, dst, data.len())
                        .expect("rlx-rocm: gpu handle upload failed");
                } else {
                    unsafe {
                        let _ = (self.ctx.runtime.hip_memcpy_htod)(
                            dst,
                            data.as_ptr() as *const _,
                            std::mem::size_of_val(data.as_slice()),
                        );
                    }
                }
            }
        }
    }


    pub(crate) fn refresh_gpu_handles_from_staging(&mut self, plan: &[usize]) {
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if plan.contains(&out_idx) && out_idx < self.output_staging.len() {
                self.gpu_handles
                    .insert(name.clone(), self.output_staging[out_idx].to_vec());
            }
        }
    }


    pub(crate) fn finalize_outputs(&mut self) -> Vec<Vec<f32>> {
        let plan = self.readback_plan();
        if plan.len() == self.graph.outputs.len() {
            self.fill_output_staging_all();
        } else {
            self.fill_output_staging_indices(&plan);
        }
        self.refresh_gpu_handles_from_staging(&plan);
        self.outputs_from_staging_plan(&plan)
    }


    pub(crate) fn outputs_from_staging_plan(&self, plan: &[usize]) -> Vec<Vec<f32>> {
        if self.pending_read_indices.is_none() && plan.len() == self.graph.outputs.len() {
            return self
                .output_staging
                .iter()
                .map(F32HostSlot::to_vec)
                .collect();
        }
        let want = self.pending_read_indices.as_deref().unwrap_or(plan);
        want.iter()
            .map(|&i| self.output_staging[i].to_vec())
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_offsets_binary() {
        let s = Step::Binary {
            n: 4,
            a_off: 0,
            b_off: 4,
            c_off: 8,
            op: 0,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![0, 4]);
        assert_eq!(w, vec![8]);
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
        // out is read-modify-write: present in BOTH reads and writes
        // so multi-stream sees the prior ScatterAddZero as a producer.
        assert!(r.contains(&100));
        assert!(w.contains(&100));
    }

    #[test]
    fn fuse_elementwise_merges_binary_then_unary() {
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
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 1);
        assert!(matches!(&fused[0], Step::FusedBinaryUnary { .. }));
    }

    #[test]
    fn fuse_elementwise_skips_when_intermediate_has_two_consumers() {
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
        assert_eq!(fused.len(), 3);
    }
}

