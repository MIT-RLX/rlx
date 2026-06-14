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

use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};

use std::sync::Mutex;

use crate::arena::{Arena, HalfDtype, plan_f32_uniform};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
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
    Llada2GroupLimitedGate {
        sig_off: u32,
        route_off: u32,
        out_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
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
        Step::GatedDeltaNet { .. } => "rlx::GatedDeltaNet",
        Step::Llada2GroupLimitedGate { .. } => "rlx::Llada2GroupLimitedGate",
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
            | Step::UmapKnn { .. }
            | Step::LogMelHost { .. }
            | Step::LogMelBackwardHost { .. }
            | Step::WelchPeaksHost { .. }
            | Step::RngNormal { .. }
            | Step::RngUniform { .. }
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

impl RocmExecutable {
    /// JIT compile, stream-mode execution. Default entry point.
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with_rng(
            graph,
            CompileMode::Jit,
            ExecMode::Stream,
            rlx_ir::RngOptions::default(),
        )
    }

    pub fn compile_rng(graph: Graph, rng: rlx_ir::RngOptions) -> Self {
        Self::compile_with_rng(graph, CompileMode::Jit, ExecMode::Stream, rng)
    }

    /// Compile with explicit RNG policy (used by [`rlx-runtime`]).
    pub fn compile_with_rng(
        graph: Graph,
        compile_mode: CompileMode,
        exec_mode: ExecMode,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        let ctx = rocm_context().expect("rlx-rocm: no HIP runtime available");

        if compile_mode == CompileMode::Aot {
            crate::kernels::prewarm_all(&ctx);
        }

        // Decompose composed ops we don't yet have native kernels for
        // (FusedMatMulBiasAct, canonical DotGeneral) into primitives
        // before memory planning.
        let graph = crate::unfuse::unfuse(graph);

        let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
        let mut plan = plan_f32_uniform(&graph, 16);
        let dequant_scratch_off = if dequant_scratch > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            plan.arena_size = aligned + dequant_scratch;
            aligned
        } else {
            0
        };
        let mut arena = Arena::from_plan(&ctx, &plan);
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0);
            arena.set_actual_len(node.id, elems * 4);
        }

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
        let arena_ptr = arena.buffer.ptr;
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let bytes_to_write = data.len().min(arena.len_of(node.id));
                let n_f32 = bytes_to_write / 4;
                let f32_view: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_f32) };
                let off_f32 = arena.offset(node.id) / 4;
                upload_to_arena(&ctx, arena_ptr, off_f32, f32_view);
            }
        }

        let mut schedule: Vec<Step> = Vec::new();
        let mut meta_buffers: Vec<HipBuffer<u32>> = Vec::new();
        let mut packed_bshd_attn: HashMap<NodeId, (NodeId, u32)> = HashMap::new();
        if !rlx_ir::env::flag("RLX_ROCM_NO_PACKED_BSHD_ATTN") {
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
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::Reshape { .. } | Op::Cast { .. } => {
                    // No-op: arena planner aliased the slot.
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
                            "rlx-rocm BatchElementwiseRegion: num_batch_inputs={n} steps={}",
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
                        let meta = upload_meta(&ctx, &meta_arr);
                        let meta_idx = meta_buffers.len();
                        meta_buffers.push(meta);
                        let batch_vec: Vec<u32> = batch_input_offs[..n].to_vec();
                        let batch_dev = upload_meta(&ctx, &batch_vec);
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
                            let meta = upload_meta(&ctx, &meta_arr);
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
                    // 149-u32 metadata buffer (16 input offsets + 32 steps *
                    // 4 u32s + prologue tail) uploaded once at compile time;
                    // the kernel walks the chain interpretively in registers.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-rocm ElementwiseRegion: chain too large \
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
                    let meta = upload_meta(&ctx, &meta_arr);
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
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    if axes.len() != 1 || axes[0] != in_dims.len() - 1 {
                        panic!(
                            "rlx-rocm Reduce: only single last-axis supported \
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
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims_u[i + 1];
                    }
                    let out_dims_u: Vec<u32> = perm.iter().map(|&i| in_dims_u[i]).collect();
                    let strides_for_out: Vec<u32> = perm.iter().map(|&i| in_strides[i]).collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims_u);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = upload_meta(&ctx, &meta_data);
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
                    if rank != in_shape.len() {
                        panic!(
                            "rlx-rocm Expand: rank mismatch (in={}, target={})",
                            in_shape.len(),
                            rank
                        );
                    }
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
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
                    let meta = upload_meta(&ctx, &meta_data);
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
                    score_scale: _,
                    attn_logit_softcap: _,
                } => {
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-rocm Attention: unfuse should have promoted to rank-4");
                    }
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let batch = geom.batch as u32;
                    let heads = geom.heads as u32;
                    let seq_q = geom.seq_q as u32;
                    let seq_k = geom.seq_k as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
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
                        panic!("rlx-rocm AttentionBackward: unfuse should have promoted to rank-4");
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
                Op::Rope { head_dim, n_rot: _ } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-rocm Rope: last_dim {} not multiple of head_dim {}",
                            last, head_dim
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-rocm Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    schedule.push(Step::Rope {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        cos_off: (arena.offset(cos_id) / 4) as u32,
                        sin_off: (arena.offset(sin_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        last_dim: last as u32,
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
                    let out_dims = node.shape.dims();
                    let x_dims = graph.node(x_id).shape.dims();
                    let m = out_dims[0].unwrap_static() as u32;
                    let n = out_dims[1].unwrap_static() as u32;
                    let k = x_dims[1].unwrap_static() as u32;
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
                            other => panic!("rlx-rocm DequantMatMul: unsupported scheme {other:?}"),
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
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!("rlx-rocm SelectiveScan: state_size {state_size} > 256 cap");
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
                    schedule.push(Step::Fft {
                        src_byte_off: arena.offset(in_id) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_complex: meta.n_complex as u32,
                        inverse: *inverse,
                        norm_tag: norm.tag(),
                        dtype_tag: rocm_fft_dtype_tag(dtype),
                        use_gpu,
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
                        panic!("rlx-rocm Im2Col: 2D NCHW only");
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
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-rocm GatedDeltaNet: state_size {state_size} > {}",
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
                    other => panic!("rlx-rocm: unsupported Op::Custom('{other}')"),
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

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let op_id = reduce_op_id(*kind);
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => schedule.push(Step::Pool1d {
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
                        }),
                        2 => schedule.push(Step::Pool2d {
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
                        }),
                        3 => schedule.push(Step::Pool3d {
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
                        }),
                        other => panic!("rlx-rocm Pool: unsupported kernel rank {other}"),
                    }
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
                        1 => schedule.push(Step::Conv1d {
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
                        }),
                        2 => schedule.push(Step::Conv2d {
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
                        }),
                        3 => schedule.push(Step::Conv3d {
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
                        }),
                        other => panic!("rlx-rocm Conv: unsupported kernel rank {other}"),
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
                other => panic!(
                    "rlx-rocm: op {other:?} not yet lowered. \
                     Open a follow-up PR if you hit this — every other op \
                     in the IR is wired."
                ),
            }
        }

        let schedule = fuse_elementwise_chains(schedule);
        let blas = rocm_blas();
        let blas_lt = rocm_blas_lt();
        let blas_lt_workspace = if blas_lt.is_some() {
            HipBuffer::<u8>::alloc_zeros(&ctx.runtime, HIPBLASLT_WORKSPACE_BYTES).ok()
        } else {
            None
        };
        let dnn = rocm_dnn();
        let dnn_workspace = if dnn.is_some() {
            HipBuffer::<u8>::alloc_zeros(&ctx.runtime, MIOPEN_WORKSPACE_BYTES).ok()
        } else {
            None
        };

        // Stream pool for MultiStream(n). Allocated up-front so the
        // scheduler doesn't pay creation cost per run().
        let mut streams: Vec<crate::hip::HipStream> = Vec::new();
        if let ExecMode::MultiStream(n) = exec_mode
            && n > 1
        {
            for _ in 0..n {
                let mut s: crate::hip::HipStream = std::ptr::null_mut();
                unsafe {
                    if (ctx.runtime.hip_stream_create)(&mut s).ok().is_ok() {
                        streams.push(s);
                    }
                }
            }
        }

        let output_staging: Vec<F32HostSlot> = graph
            .outputs
            .iter()
            .map(|&id| {
                let elems = graph.node(id).shape.num_elements().unwrap_or(0);
                F32HostSlot::new(&ctx.runtime, elems, pinned_io_enabled(exec_mode))
            })
            .collect();

        let mut input_staging = HashMap::new();
        if pinned_io_enabled(exec_mode) {
            for (name, &id) in &input_offsets {
                let elems = graph.node(id).shape.num_elements().unwrap_or(0);
                input_staging.insert(name.clone(), F32HostSlot::new(&ctx.runtime, elems, true));
            }
        }

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
            dequant_scratch_off,
            graph,
            arena,
            schedule,
            input_offsets,
            param_offsets,
            meta_buffers,
            exec_mode,
            half_act_scratch: None,
            captured_graph: None,
            streams,
            active_extent: None,
            output_staging,
            input_staging,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            pending_read_indices: None,
            input_slot_names,
            input_slots,
            output_slots,
            host_arena,
            rng: std::sync::Arc::new(std::sync::RwLock::new(rng)),
        }
    }

    pub fn compile_with(graph: Graph, compile_mode: CompileMode, exec_mode: ExecMode) -> Self {
        Self::compile_with_rng(
            graph,
            compile_mode,
            exec_mode,
            rlx_ir::RngOptions::default(),
        )
    }

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

    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.output_slots
    }

    fn upload_slot_inputs(&mut self, inputs: &[&[f32]]) {
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

    fn pack_host_arena(&mut self) {
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

    /// Fast path: positional inputs, D2H into [`Self::host_arena`], no per-output `Vec`.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.upload_slot_inputs(inputs);
        let _ = self.run_inner(&[]);
        self.pack_host_arena();
        &self.output_slots
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every step in the schedule is in the safe set.
    /// See PLAN L1.
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
            let bytes = data.len() * 4;
            let dst = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            unsafe {
                let _ = (self.ctx.runtime.hip_memcpy_htod)(dst, data.as_ptr() as *const _, bytes);
            }
        }
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let byte_off = self.arena.offset(id);
            crate::gguf_host::upload_param_bytes(&self.ctx, &mut self.arena.buffer, byte_off, data);
        }
    }

    pub fn set_param_half(&mut self, name: &str, dtype: HalfDtype, bits: &[u16]) {
        let id = match self.param_offsets.get(name) {
            Some(&id) if self.arena.has(id) => id,
            _ => return,
        };
        let f32_off = (self.arena.offset(id) / 4) as u32;
        let off = self
            .arena
            .register_half_param(&self.ctx, id, f32_off, bits.len(), dtype);
        if let Some(buf) = self.arena.half_buffer.as_mut() {
            let bytes = bits.len() * 2;
            let dst = buf.ptr + (off as u64) * 2;
            unsafe {
                let _ = (self.ctx.runtime.hip_memcpy_htod)(dst, bits.as_ptr() as *const _, bytes);
            }
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
        self.pending_read_indices = read_indices.map(|s| s.to_vec());
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        outs
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

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
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

    fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        use crate::kernels::*;

        let stream = self.ctx.default_stream;
        let arena_base = self.arena.buffer.ptr;

        self.stage_gpu_handle_inputs(inputs);

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into a
        // captured hipGraph.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let dst = arena_base + (off_f32 as u64) * 4;
                if let Some(host) = self.input_staging.get_mut(name) {
                    host.copy_from_host(data);
                    host.htod(&self.ctx.runtime, dst, data.len())
                        .expect("rlx-rocm: pinned input upload failed");
                } else {
                    unsafe {
                        let _ = (self.ctx.runtime.hip_memcpy_htod)(
                            dst,
                            data.as_ptr() as *const _,
                            std::mem::size_of_val(data),
                        );
                    }
                }
            }
        }

        // Active-extent (PLAN L1): when set + every Step safe, bypass
        // hipGraph capture/replay (recorded at full extent) and dispatch
        // per-step with scaled launch dims via the normal loop.
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // hipGraph fast path: replay the previously-captured schedule.
        let graph_eligible = active.is_none()
            && self.exec_mode == ExecMode::Graph
            && schedule_graph_capture_safe(&self.schedule);
        let do_replay = graph_eligible && self.captured_graph.is_some();
        let do_capture = graph_eligible && self.captured_graph.is_none();
        if do_replay {
            unsafe {
                let _ = (self.ctx.runtime.hip_graph_launch)(self.captured_graph.unwrap(), stream);
                let _ = (self.ctx.runtime.hip_stream_sync)(stream);
            }
            self.run_tail_host_audio_ops(false);
            return self.finalize_outputs();
        }
        if do_capture {
            // hipStreamCaptureMode_Relaxed = 2 (matches CUDA value).
            unsafe {
                let _ = (self.ctx.runtime.hip_stream_begin_capture)(stream, 2);
            }
        }

        // Multi-stream scheduler state. When `exec_mode ==
        // MultiStream(n)`, each Step gets assigned to one of `n`
        // pool streams based on producer-consumer dependencies on
        // arena offsets. Independent ops parallelise; producer-
        // consumer chains stay on one stream.
        let multi_stream =
            matches!(self.exec_mode, ExecMode::MultiStream(_)) && !self.streams.is_empty();
        let mut producer_of: HashMap<u32, usize> = HashMap::new();
        let mut last_event: HashMap<usize, crate::hip::HipEvent> = HashMap::new();
        let mut rr_cursor: usize = 0;

        // Dispatch each step on the default stream.
        for step in &self.schedule {
            let _roctx = crate::roctx::scoped_range(step_name(step));
            // PLAN L3: cross-backend Perfetto trace; no-op when env
            // var RLX_TRACE_PERFETTO unset.
            let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "rocm");
            let mut arena_ptr = arena_base;

            // Per-step stream selection (multi-stream mode only).
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
                    let chosen = *producer_streams.iter().next().unwrap();
                    for s in &producer_streams {
                        if *s != chosen
                            && let Some(evt) = last_event.get(s)
                        {
                            unsafe {
                                let _ = (self.ctx.runtime.hip_stream_wait_event)(
                                    self.streams[chosen],
                                    *evt,
                                    0,
                                );
                            }
                        }
                    }
                    chosen
                };
                Some(chosen)
            } else {
                None
            };
            // Shadow the outer `stream` with the assigned stream.
            #[allow(unused_assignments)]
            let stream = match assigned_idx {
                Some(i) => self.streams[i],
                None => stream,
            };
            // Re-bind hipBLAS handle to the active stream so the
            // hipblasSgemm path's internal kernel launches go to the
            // right queue.
            if multi_stream && let Some(blas) = self.blas.as_ref() {
                let blas = blas.lock().unwrap();
                unsafe {
                    let _ = blas.set_stream(stream);
                }
            }
            match step {
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, c_off, op]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, c_off, op]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, in_off, out_off, op]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, cond_off, x_off, y_off, out_off]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, out_off, bin_op, un_op]
                    );
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
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (gx, gy, gz),
                        (bx, by, bz),
                        [
                            &mut arena_ptr,
                            &len_s,
                            num_inputs,
                            num_steps,
                            dst_off,
                            &mut meta_ptr,
                            scalar_input_mask,
                            input_modulus
                        ]
                    );
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
                    let mut batch_ptr = self.meta_buffers[*batch_offs_idx].ptr;
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid_x, 1, num_batch_s),
                        (block_x, 1, 1),
                        [
                            &mut arena_ptr,
                            &slice_len_s,
                            &num_batch_s,
                            num_steps,
                            base_dst_off,
                            slice_elems,
                            &mut batch_ptr,
                            &mut meta_ptr,
                            scalar_input_mask,
                            input_modulus
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off, op]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &outer_s,
                            inner,
                            in_off,
                            out_off,
                            gamma_off,
                            beta_off,
                            eps_bits,
                            op
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &outer_s,
                            inner,
                            in_off,
                            residual_off,
                            bias_off,
                            gamma_off,
                            beta_off,
                            out_off,
                            eps_bits,
                            has_bias
                        ]
                    );
                }
                Step::Argmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let kernel = argmax_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, outer, inner, in_off, out_off]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off, exclusive]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, outer, inner, k, in_off, out_off]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            outer,
                            inner,
                            in_off,
                            out_off,
                            top_k,
                            top_p_bits,
                            temp_bits,
                            seed_lo,
                            seed_hi
                        ]
                    );
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
                        &self.ctx,
                        &self.arena.buffer,
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
                        &self.ctx,
                        &self.arena.buffer,
                        *dst_byte_off as usize,
                        *len as usize,
                        *low,
                        *high,
                        *key,
                        *op_seed,
                        opts,
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n_out,
                            n_idx,
                            dim,
                            vocab,
                            in_off,
                            idx_off,
                            out_off
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off,
                            idx_off,
                            out_off
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            inner,
                            axis_in_size,
                            axis_out_size,
                            start,
                            in_off,
                            out_off
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            inner,
                            axis_in_size,
                            axis_out_size,
                            start,
                            in_off,
                            out_off
                        ]
                    );
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
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            rank,
                            out_total,
                            in_off,
                            out_off,
                            &mut meta_ptr
                        ]
                    );
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
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            rank,
                            out_total,
                            in_off,
                            out_off,
                            &mut meta_ptr
                        ]
                    );
                }
                Step::Rope {
                    n_total,
                    seq,
                    head_dim,
                    half,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off,
                    last_dim,
                } => {
                    let kernel = rope_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n_total,
                            seq,
                            head_dim,
                            half,
                            in_off,
                            cos_off,
                            sin_off,
                            out_off,
                            last_dim
                        ]
                    );
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
                    let use_row = rlx_ir::attention_dispatch_use_row(
                        *head_dim,
                        "RLX_ROCM_FORCE_ATTENTION_ROW",
                    );
                    if use_row {
                        let total = batch * heads * seq_q;
                        let block = 256u32;
                        crate::launch_kernel!(
                            attention_row_kernel(&self.ctx),
                            stream,
                            (total.div_ceil(block), 1, 1),
                            (block, 1, 1),
                            [
                                &mut arena_ptr,
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
                                o_seq_stride
                            ]
                        );
                    } else {
                        let q_blocks = (*seq_q).div_ceil(16);
                        crate::launch_kernel!(
                            attention_kernel(&self.ctx),
                            stream,
                            (q_blocks, batch * heads, 1),
                            (128, 1, 1),
                            [
                                &mut arena_ptr,
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
                                o_seq_stride
                            ]
                        );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (batch * heads, y_blocks, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
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
                            wrt
                        ]
                    );
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
                    // Tier 0: mixed-precision GemmEx — when B is in
                    // the half-arena, cast activations to f16/bf16
                    // and call hipblasGemmEx with both inputs half +
                    // f32 accumulator. Bias / activation epilogue
                    // runs through the shared matmul_epilogue kernel.
                    let used_mixed = try_mixed_precision_gemm_rocm(
                        &self.ctx,
                        &mut self.arena,
                        &mut self.half_act_scratch,
                        self.blas.as_ref(),
                        *m,
                        *k,
                        *n,
                        *batch,
                        *a_off_f32,
                        *b_off_f32,
                        *c_off_f32,
                    );
                    if used_mixed {
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 1: hipBLASLt fused (matmul + bias + relu/gelu
                    // in one launch). Only when activation is one of
                    // the two natively fusable; other acts fall through
                    // to plain sgemm + epilogue kernel. Strided-batch
                    // is handled via LAYOUT_ATTR_BATCH_COUNT /
                    // STRIDED_BATCH_OFFSET in matmul_fused.
                    let try_lt = self.blas_lt.is_some()
                        && self.blas_lt_workspace.is_some()
                        && crate::hipblaslt::act_supported(*act_id);
                    let used_lt = if try_lt {
                        let lt = self.blas_lt.as_ref().unwrap();
                        let workspace = self.blas_lt_workspace.as_ref().unwrap();
                        let epilogue = crate::hipblaslt::epilogue_for(*act_id, *has_bias != 0)
                            .expect("rlx-rocm: act_supported lied");
                        let r = unsafe {
                            crate::hipblaslt::matmul_fused(
                                lt,
                                workspace.ptr,
                                HIPBLASLT_WORKSPACE_BYTES,
                                arena_base,
                                *m,
                                *k,
                                *n,
                                *a_off_f32,
                                *b_off_f32,
                                *c_off_f32,
                                *has_bias != 0,
                                *bias_off_f32,
                                epilogue,
                                *batch,
                                *a_batch_stride,
                                *b_batch_stride,
                                *c_batch_stride,
                                stream,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("matmul.hipblaslt", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_lt {
                        continue;
                    }

                    // Tier 2: hipBLAS sgemm via raw pointers. Same A↔B
                    // swap trick as the cuBLAS path in rlx-cuda — we
                    // compute the column-major transpose of our row-
                    // major matmul, which gives the right result back.
                    let used_hipblas = if let Some(blas) = self.blas.as_ref() {
                        let blas = blas.lock().unwrap();
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        let a_dev = arena_base + (*a_off_f32 as u64) * 4;
                        let b_dev = arena_base + (*b_off_f32 as u64) * 4;
                        let c_dev = arena_base + (*c_off_f32 as u64) * 4;
                        let result = unsafe {
                            if *batch == 1 {
                                (blas.runtime.sgemm)(
                                    blas.handle,
                                    HipblasOperation::N,
                                    HipblasOperation::N,
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
                                (blas.runtime.sgemm_strided)(
                                    blas.handle,
                                    HipblasOperation::N,
                                    HipblasOperation::N,
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
                        if let Err(e) = result.ok() {
                            log_fallback("matmul.hipblasSgemm", e);
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    };
                    if used_hipblas {
                        // Optional bias / activation post-pass via the
                        // matmul_epilogue kernel (same shared kernel
                        // as rlx-cuda's cuBLAS path).
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 3: rocWMMA matrix-core kernel. Opt-in via
                    // `RLX_ROCM_MFMA=1`. f16 multiply / f32 accumulate
                    // — bias / activation run through the shared
                    // matmul_epilogue kernel afterward.
                    if use_mfma() {
                        let kernel = matmul_mfma_kernel(&self.ctx);
                        crate::launch_kernel!(
                            kernel,
                            stream,
                            ((*n).div_ceil(32), (*m).div_ceil(32), *batch),
                            (256, 1, 1),
                            [
                                &mut arena_ptr,
                                m,
                                k,
                                n,
                                a_off_f32,
                                b_off_f32,
                                c_off_f32,
                                batch,
                                a_batch_stride,
                                b_batch_stride,
                                c_batch_stride
                            ]
                        );
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 4: custom 64×64 + 4×4 register-tile kernel.
                    let kernel = matmul_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                        (16, 16, 1),
                        [
                            &mut arena_ptr,
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
                            act_id
                        ]
                    );
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
                    // Tier 1: sorted-batch dispatch via hipBLAS. Direct
                    // port from rlx-cuda — sync the stream so prior
                    // writes to idx are visible, dtoh-copy the idx
                    // buffer, walk it for runs, issue one
                    // hipblasSgemm per run when run count <= m/4.
                    // Random idx falls back to the per-token kernel.
                    let used_sorted = if let Some(blas) = self.blas.as_ref() {
                        unsafe {
                            let _ = (self.ctx.runtime.hip_stream_sync)(stream);
                        }
                        let mn = *m as usize;
                        let mut idx_host = vec![0.0_f32; mn];
                        let idx_dev = arena_base + (*idx_off as u64) * 4;
                        let dtoh_ok = unsafe {
                            (self.ctx.runtime.hip_memcpy_dtoh)(
                                idx_host.as_mut_ptr() as *mut _,
                                idx_dev,
                                mn * 4,
                            )
                            .ok()
                            .is_ok()
                        };
                        if dtoh_ok {
                            let mut runs: Vec<(u32, u32, u32)> = Vec::new();
                            let mut i = 0usize;
                            while i < mn {
                                let e = idx_host[i] as u32;
                                let mut j = i + 1;
                                while j < mn && (idx_host[j] as u32) == e {
                                    j += 1;
                                }
                                if e < *num_experts {
                                    runs.push((i as u32, j as u32, e));
                                }
                                i = j;
                            }
                            let threshold = (mn / 4).max(2);
                            if !runs.is_empty() && runs.len() <= threshold {
                                let blas = blas.lock().unwrap();
                                let alpha: f32 = 1.0;
                                let beta: f32 = 0.0;
                                let mut all_ok = true;
                                for (lo, hi, e) in &runs {
                                    let rows = hi - lo;
                                    let a_dev = arena_base + ((*in_off + lo * k) as u64) * 4;
                                    let b_dev = arena_base + ((*w_off + e * k * n) as u64) * 4;
                                    let c_dev = arena_base + ((*out_off + lo * n) as u64) * 4;
                                    let r = unsafe {
                                        (blas.runtime.sgemm)(
                                            blas.handle,
                                            HipblasOperation::N,
                                            HipblasOperation::N,
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
                                    if r.ok().is_err() {
                                        log_fallback("grouped_matmul.hipblas", r);
                                        all_ok = false;
                                        break;
                                    }
                                }
                                all_ok
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if used_sorted {
                        continue;
                    }

                    // Fallback: per-token expert-lookup kernel.
                    let kernel = grouped_matmul_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        (8, 8, 1),
                        [
                            &mut arena_ptr,
                            m,
                            k,
                            n,
                            num_experts,
                            in_off,
                            w_off,
                            idx_off,
                            out_off
                        ]
                    );
                }
                Step::ScatterAddZero { out_off, out_total } => {
                    let kernel = scatter_add_zero_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, out_off, out_total]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            out_off,
                            upd_off,
                            idx_off,
                            num_updates,
                            trailing,
                            out_dim
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        (8, 8, 1),
                        [
                            &mut arena_ptr,
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off,
                            w_off,
                            scale_off,
                            zp_off,
                            out_off
                        ]
                    );
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                            &self.ctx,
                            stream,
                            &self.arena.buffer,
                            blas,
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
                            &self.ctx,
                            &self.arena.buffer,
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        unsafe {
                            crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                                &self.ctx,
                                stream,
                                &self.arena.buffer,
                                blas,
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
                        }
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &self.ctx,
                            &self.arena.buffer,
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            batch,
                            seq,
                            hidden,
                            state_size,
                            x_off,
                            delta_off,
                            a_off,
                            b_off,
                            c_off,
                            out_off
                        ]
                    );
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
                } => {
                    if *use_gpu {
                        let norm = rlx_ir::fft::FftNorm::from_tag(*norm_tag);
                        let scale = norm.output_scale(*n_complex as usize, *inverse) as f32;
                        crate::fft_dispatch::run_fft_gpu(
                            &self.ctx,
                            stream,
                            arena_ptr,
                            *src_byte_off / 4,
                            *dst_byte_off / 4,
                            *outer,
                            *n_complex,
                            *inverse,
                            scale,
                        );
                    } else {
                        crate::fft_host::run_fft1d(
                            &self.ctx,
                            &self.arena.buffer,
                            self.arena.size,
                            *src_byte_off as usize,
                            *dst_byte_off as usize,
                            *outer as usize,
                            *n_complex as usize,
                            *inverse,
                            *norm_tag,
                            rocm_fft_dtype_from_tag(*dtype_tag),
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
                        stream,
                        arena_ptr,
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
                        let m_dim = *n * *h_out * *w_out;
                        let k_dim = *c_in * *kh * *kw;
                        let total = m_dim * k_dim;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let x_off = *x_byte_off / 4;
                        let col_off = *col_byte_off / 4;
                        crate::launch_kernel!(
                            kernel,
                            stream,
                            (grid, 1, 1),
                            (block, 1, 1),
                            [
                                &mut arena_ptr,
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
                                &x_off,
                                &col_off
                            ]
                        );
                    } else {
                        crate::im2col_host::run_im2col(
                            &self.ctx,
                            &self.arena.buffer,
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
                    crate::gdn_host::run_gated_delta_net(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                Step::Llada2GroupLimitedGate {
                    sig_off,
                    route_off,
                    out_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *sig_off as usize,
                        *route_off as usize,
                        *out_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::UmapKnn {
                    pairwise_off,
                    out_off,
                    n,
                    k,
                } => {
                    crate::umap_knn_host::run_umap_knn(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *pairwise_off as usize,
                        *out_off as usize,
                        *n as usize,
                        *k as usize,
                    );
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
                    #[cfg(feature = "native-splat")]
                    crate::splat_native::run_gaussian_splat_render_native(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                    crate::splat_host::run_gaussian_splat_render_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                    crate::splat_host::run_gaussian_splat_prepare(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                    crate::splat_host::run_gaussian_splat_rasterize(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let wrt = 0u32;
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dx_off,
                            eps_bits,
                            &wrt
                        ]
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
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dgamma_off = *dgamma_byte_off / 4;
                    let wrt = 1u32;
                    let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                    let (zgrid, zblock) = dispatch_grid_1d(*h, 256);
                    crate::launch_kernel!(
                        zk,
                        stream,
                        (zgrid, 1, 1),
                        (zblock, 1, 1),
                        [&mut arena_ptr, &dgamma_off, h]
                    );
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dgamma_off,
                            eps_bits,
                            &wrt
                        ]
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
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dbeta_off = *dbeta_byte_off / 4;
                    let wrt = 2u32;
                    let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                    let (zgrid, zblock) = dispatch_grid_1d(*h, 256);
                    crate::launch_kernel!(
                        zk,
                        stream,
                        (zgrid, 1, 1),
                        (zblock, 1, 1),
                        [&mut arena_ptr, &dbeta_off, h]
                    );
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dbeta_off,
                            eps_bits,
                            &wrt
                        ]
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
                    let dy_off = *dy_byte_off / 4;
                    let cos_off = *cos_byte_off / 4;
                    let sin_off = *sin_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let kernel = rope_backward_kernel(&self.ctx);
                    let total = batch * seq * hidden;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            batch,
                            seq,
                            hidden,
                            head_dim,
                            n_rot,
                            &dy_off,
                            &cos_off,
                            &sin_off,
                            &dx_off,
                            cos_len
                        ]
                    );
                }
                Step::CumsumBackward {
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    cols,
                    exclusive,
                } => {
                    let dy_off = *dy_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let excl = if *exclusive { 1u32 } else { 0u32 };
                    let kernel = cumsum_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*rows, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, rows, cols, &dy_off, &dx_off, &excl]
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
                    let dy_off = *dy_byte_off / 4;
                    let idx_off = *indices_byte_off / 4;
                    let dst_off = *dst_byte_off / 4;
                    let total = *outer * *axis_dim * *trailing;
                    if total > 0 {
                        let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                        let (zgrid, zblock) = dispatch_grid_1d(total, 256);
                        crate::launch_kernel!(
                            zk,
                            stream,
                            (zgrid, 1, 1),
                            (zblock, 1, 1),
                            [&mut arena_ptr, &dst_off, &total]
                        );
                    }
                    let kernel = gather_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*outer, (num_idx * trailing).div_ceil(256), 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            &dy_off,
                            &idx_off,
                            &dst_off
                        ]
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            src_off,
                            g_off,
                            b_off,
                            dst_off,
                            n,
                            c,
                            h,
                            w,
                            eps_bits
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            groups
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            src_off,
                            g_off,
                            b_off,
                            dst_off,
                            n,
                            c,
                            h,
                            w,
                            num_groups,
                            eps_bits
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, src_off, dst_off, n, c, h, w]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c,
                            l,
                            l_out,
                            kl,
                            sl,
                            pl,
                            op,
                            in_off,
                            out_off
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            out_off
                        ]
                    );
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
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            out_off
                        ]
                    );
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
                    // Tier 1: MIOpen forward conv as a degenerate 2-D
                    // conv (H=kh=sh=1, ph=0, dh=1). Same trick rlx-cuda
                    // uses in conv1d.
                    let used_miopen = if let (Some(dnn), Some(workspace)) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref())
                    {
                        let r = unsafe {
                            crate::miopen::conv2d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
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
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv1d.miopen", e);
                        }
                        r.is_ok() && *dl == 1
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    let kernel = conv1d_kernel(&self.ctx);
                    let total = n * c_out * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            out_off
                        ]
                    );
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
                    // Tier 1: MIOpen forward conv. Bounded to dilation=1
                    // for now since MIOpen's miopenInitConvolutionDescriptor
                    // takes a dilation_h/dilation_w pair (no nd version
                    // here); when dh != 1 || dw != 1 we fall through.
                    let used_miopen = if let (Some(dnn), Some(workspace), 1, 1) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref(), *dh, *dw)
                    {
                        let r = unsafe {
                            crate::miopen::conv2d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
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
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d.miopen", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            out_off
                        ]
                    );
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
                    // Tier 1: MIOpen nd-conv. NCDHW input/output, 5-D
                    // KCDHW filter, 3-D pads/strides/dilations.
                    let used_miopen = if let (Some(dnn), Some(workspace)) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref())
                    {
                        let r = unsafe {
                            crate::miopen::conv3d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
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
                            log_fallback("conv3d.miopen", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    let kernel = conv3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            out_off
                        ]
                    );
                }
            }

            // Multi-stream tail: record an event so future steps can
            // wait on this one, then update producer_of with the
            // offsets this step wrote.
            if let Some(idx) = assigned_idx {
                let mut evt: crate::hip::HipEvent = std::ptr::null_mut();
                unsafe {
                    if (self.ctx.runtime.hip_event_create)(&mut evt, 0)
                        .ok()
                        .is_ok()
                    {
                        let _ = (self.ctx.runtime.hip_event_record)(evt, stream);
                        // Replace any older event for this stream.
                        if let Some(prev) = last_event.insert(idx, evt) {
                            let _ = (self.ctx.runtime.hip_event_destroy)(prev);
                        }
                    }
                }
                let (_, writes) = step_offsets(step);
                for w in &writes {
                    producer_of.insert(*w, idx);
                }
            }
        }

        // Multi-stream: sync every pool stream + clean up events so
        // output reads see all produced data.
        if multi_stream {
            for s in &self.streams {
                unsafe {
                    let _ = (self.ctx.runtime.hip_stream_sync)(*s);
                }
            }
            for (_, evt) in last_event.drain() {
                unsafe {
                    let _ = (self.ctx.runtime.hip_event_destroy)(evt);
                }
            }
        }

        if do_capture {
            unsafe {
                let mut graph: crate::hip::HipGraph = std::ptr::null_mut();
                let mut graph_exec: crate::hip::HipGraphExec = std::ptr::null_mut();
                if (self.ctx.runtime.hip_stream_end_capture)(stream, &mut graph)
                    .ok()
                    .is_ok()
                    && !graph.is_null()
                {
                    let r = (self.ctx.runtime.hip_graph_instantiate)(
                        &mut graph_exec,
                        graph,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                    );
                    let _ = (self.ctx.runtime.hip_graph_destroy)(graph);
                    if r.ok().is_ok() {
                        // First-run launch: actually compute outputs.
                        let _ = (self.ctx.runtime.hip_graph_launch)(graph_exec, stream);
                        self.captured_graph = Some(graph_exec);
                    }
                }
            }
        }

        // Sync stream + read outputs.
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(stream);
        }
        self.run_tail_host_audio_ops(false);
        self.finalize_outputs()
    }

    fn run_tail_host_audio_ops(&self, pre_sync: bool) {
        if !self.schedule.iter().any(step_is_tail_host) {
            return;
        }
        if pre_sync {
            unsafe {
                let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
            }
        }
        for step in &self.schedule {
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
                        &self.ctx,
                        &self.arena.buffer,
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
                        &self.ctx,
                        &self.arena.buffer,
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
                        &self.ctx,
                        &self.arena.buffer,
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

    fn readback_plan(&self) -> Vec<usize> {
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

    fn stage_gpu_handle_inputs(&mut self, inputs: &[(&str, &[f32])]) {
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

    fn refresh_gpu_handles_from_staging(&mut self, plan: &[usize]) {
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if plan.contains(&out_idx) && out_idx < self.output_staging.len() {
                self.gpu_handles
                    .insert(name.clone(), self.output_staging[out_idx].to_vec());
            }
        }
    }

    fn finalize_outputs(&mut self) -> Vec<Vec<f32>> {
        let plan = self.readback_plan();
        if plan.len() == self.graph.outputs.len() {
            self.fill_output_staging_all();
        } else {
            self.fill_output_staging_indices(&plan);
        }
        self.refresh_gpu_handles_from_staging(&plan);
        self.outputs_from_staging_plan(&plan)
    }

    fn fill_output_staging_indices(&mut self, indices: &[usize]) {
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
        }
        for &i in indices {
            let id = self.graph.outputs[i];
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            let src = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            debug_assert_eq!(self.output_staging[i].len(), elems);
            self.output_staging[i]
                .dtoh(&self.ctx.runtime, src, elems)
                .expect("rlx-rocm: partial output download failed");
        }
    }

    fn fill_output_staging_all(&mut self) {
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
        }
        for (i, &id) in self.graph.outputs.iter().enumerate() {
            let off_f32 = self.arena.offset(id) / 4;
            let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
            let src = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            debug_assert_eq!(self.output_staging[i].len(), elems);
            self.output_staging[i]
                .dtoh(&self.ctx.runtime, src, elems)
                .expect("rlx-rocm: output download failed");
        }
    }

    fn outputs_from_staging_plan(&self, plan: &[usize]) -> Vec<Vec<f32>> {
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
