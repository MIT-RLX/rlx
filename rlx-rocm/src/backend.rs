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
//! Currently lands the **structural foundation**: full `Step` enum,
//! `CompileMode` / `ExecMode` (Stream + Eager — Graph + MultiStream
//! deferred), `RocmExecutable` struct + lifecycle, `set_param` /
//! `set_param_half` / output read, and the pure-Rust helpers
//! (`step_offsets`, `fuse_elementwise_chains`, `step_name`,
//! `prewarm_all`). The two remaining pieces are **mechanical ports**
//! from `rlx-cuda` that need to be done with care since we can't
//! validate on Mac:
//!
//!   1. `lower_graph()` — IR walk that builds `Vec<Step>` from a
//!      `Graph`. Pure IR-level code; copy from `rlx-cuda::compile_with`
//!      with `cudarc` type swaps where applicable. ~700 lines.
//!
//!   2. `dispatch_step()` — match-arm dispatch that maps each Step
//!      to a kernel launch via the HIP shim. Custom kernels only —
//!      no hipBLAS / hipBLASLt / MIOpen tiers (those land tier-by-
//!      tier in subsequent commits). ~600 lines.
//!
//! Until those land, `compile_with` and `run` panic with a clear
//! pointer to where the work picks up.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};

use std::sync::Mutex;

use crate::arena::{Arena, HalfDtype, plan_f32_uniform};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
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
        Step::SelectiveScan { .. } => "rlx::SelectiveScan",
        Step::GatedDeltaNet { .. } => "rlx::GatedDeltaNet",
        Step::Llada2GroupLimitedGate { .. } => "rlx::Llada2GroupLimitedGate",
        Step::GaussianSplatRender { .. } => "rlx::GaussianSplatRender",
        Step::GaussianSplatRenderBackward { .. } => "rlx::GaussianSplatRenderBackward",
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
        Step::FusedBinaryUnary { .. } => "rlx::FusedBinaryUnary",
        Step::ElementwiseRegion { .. } => "rlx::ElementwiseRegion",
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
        Step::Llada2GroupLimitedGate {
            sig_off,
            route_off,
            out_off,
            ..
        } => (vec![*sig_off, *route_off], vec![*out_off]),
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
        )
    }
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
        Self::compile_with(graph, CompileMode::Jit, ExecMode::Stream)
    }

    /// One-shot eager run.
    pub fn eager(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut exec = Self::compile_with(graph, CompileMode::Jit, ExecMode::Eager);
        exec.run(inputs)
    }

    /// Full constructor with explicit compile + exec modes. Mirrors
    /// `rlx-cuda::backend::CudaExecutable::compile_with` — the IR
    /// walk + memory plan + Step emission is identical; only the
    /// device-buffer types differ (`HipBuffer` vs `CudaSlice`).
    pub fn compile_with(graph: Graph, compile_mode: CompileMode, exec_mode: ExecMode) -> Self {
        let ctx = rocm_context().expect("rlx-rocm: no HIP runtime available");

        if compile_mode == CompileMode::Aot {
            crate::kernels::prewarm_all(&ctx);
        }

        // Decompose composed ops we don't yet have native kernels for
        // (FusedMatMulBiasAct, canonical DotGeneral) into primitives
        // before memory planning.
        let graph = crate::unfuse::unfuse(graph);

        let plan = plan_f32_uniform(&graph, 16);
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
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // 72-u32 metadata buffer (8 input offsets + 16 steps *
                    // 4 u32s) uploaded once at compile time; the kernel
                    // walks the chain interpretively in registers. Caps
                    // and op_sub numbering match the cross-backend
                    // Metal MSL / wgpu WGSL / rlx-cuda encoders so the
                    // shared `elementwise_region.cu` kernel interprets
                    // the byte stream identically.
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
                    let encode_operand = |op: &ChainOperand| -> u32 {
                        match *op {
                            ChainOperand::Input(i) => i & 0x7FFF_FFFFu32,
                            ChainOperand::Step(i) => 0x8000_0000u32 | (i & 0x7FFF_FFFFu32),
                        }
                    };
                    let act_sub = |a: Activation| match a {
                        Activation::Gelu => 0u32,
                        Activation::GeluApprox => 1,
                        Activation::Silu => 2,
                        Activation::Relu => 3,
                        Activation::Sigmoid => 4,
                        Activation::Tanh => 5,
                        Activation::Exp => 6,
                        Activation::Log => 7,
                        Activation::Sqrt => 8,
                        Activation::Rsqrt => 9,
                        Activation::Neg => 10,
                        Activation::Abs => 11,
                        Activation::Round => 12,
                        Activation::Sin => 13,
                        Activation::Cos => 14,
                        Activation::Tan => 15,
                        Activation::Atan => 16,
                    };
                    let bin_sub = |b: BinaryOp| match b {
                        BinaryOp::Add => 0u32,
                        BinaryOp::Sub => 1,
                        BinaryOp::Mul => 2,
                        BinaryOp::Div => 3,
                        BinaryOp::Max => 4,
                        BinaryOp::Min => 5,
                        BinaryOp::Pow => 6,
                    };
                    let cmp_sub = |c: CmpOp| match c {
                        CmpOp::Eq => 0u32,
                        CmpOp::Ne => 1,
                        CmpOp::Lt => 2,
                        CmpOp::Le => 3,
                        CmpOp::Gt => 4,
                        CmpOp::Ge => 5,
                    };
                    // meta layout: 16 input offsets + 32 steps × 4 u32s = 144 words.
                    let mut meta_data: Vec<u32> = Vec::with_capacity(144);
                    meta_data.extend_from_slice(&input_offs);
                    let mut chain_enc = [0u32; 128];
                    for (k, step) in chain.iter().enumerate() {
                        let base = k * 4;
                        let (kind, sub, lhs, rhs) = match step {
                            ChainStep::Activation(a, src) => {
                                (0u32, act_sub(*a), encode_operand(src), 0u32)
                            }
                            ChainStep::Cast(_, src) => (1u32, 0, encode_operand(src), 0u32),
                            ChainStep::Binary(op, l, r) => {
                                (2u32, bin_sub(*op), encode_operand(l), encode_operand(r))
                            }
                            ChainStep::Compare(op, l, r) => {
                                (3u32, cmp_sub(*op), encode_operand(l), encode_operand(r))
                            }
                            ChainStep::Where(c, t, f) =>
                            // Pack 3 operands into the 4-u32 step:
                            // op_sub=cond, lhs=on_true, rhs=on_false.
                            {
                                (
                                    4u32,
                                    encode_operand(c),
                                    encode_operand(t),
                                    encode_operand(f),
                                )
                            }
                        };
                        chain_enc[base] = kind;
                        chain_enc[base + 1] = sub;
                        chain_enc[base + 2] = lhs;
                        chain_enc[base + 3] = rhs;
                    }
                    meta_data.extend_from_slice(&chain_enc);
                    let meta = upload_meta(&ctx, &meta_data);
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::ElementwiseRegion {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        input_offs,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                        meta_idx,
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
                } => {
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-rocm Attention: unfuse should have promoted to rank-4");
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
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, 0u32, 0u32),
                    };
                    let _ = num_heads;
                    schedule.push(Step::Attention {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
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
                Op::Custom { name, attrs, .. } => {
                    if name != "llada2.group_limited_gate" {
                        panic!("rlx-rocm: unsupported Op::Custom('{name}')");
                    }
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

        Self {
            ctx,
            blas,
            blas_lt,
            blas_lt_workspace,
            dnn,
            dnn_workspace,
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
        }
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
        use crate::kernels::*;

        let stream = self.ctx.default_stream;
        let arena_base = self.arena.buffer.ptr;

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into a
        // captured hipGraph.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let dst = arena_base + (off_f32 as u64) * 4;
                unsafe {
                    let _ = (self.ctx.runtime.hip_memcpy_htod)(
                        dst,
                        data.as_ptr() as *const _,
                        std::mem::size_of_val(data),
                    );
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
        let do_replay =
            active.is_none() && self.exec_mode == ExecMode::Graph && self.captured_graph.is_some();
        let do_capture =
            active.is_none() && self.exec_mode == ExecMode::Graph && self.captured_graph.is_none();
        if do_replay {
            unsafe {
                let _ = (self.ctx.runtime.hip_graph_launch)(self.captured_graph.unwrap(), stream);
                let _ = (self.ctx.runtime.hip_stream_sync)(stream);
            }
            return self.read_outputs();
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
                } => {
                    let len_s = scale(*len);
                    if len_s == 0 {
                        continue;
                    }
                    let kernel = elementwise_region_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(len_s, 256);
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
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
                } => {
                    // FlashAttention-1: BR=16 q-rows per block, 128 threads/block.
                    let kernel = attention_kernel(&self.ctx);
                    let q_blocks = (*seq_q).div_ceil(16);
                    crate::launch_kernel!(
                        kernel,
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
                            window
                        ]
                    );
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
                    crate::training_bwd_host::run_rms_norm_backward_input(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *x_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dy_byte_off as usize,
                        *dx_byte_off as usize,
                        *rows,
                        *h,
                        f32::from_bits(*eps_bits),
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
                    crate::training_bwd_host::run_rms_norm_backward_gamma(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *x_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dy_byte_off as usize,
                        *dgamma_byte_off as usize,
                        *rows,
                        *h,
                        f32::from_bits(*eps_bits),
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
                    crate::training_bwd_host::run_rms_norm_backward_beta(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *x_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dy_byte_off as usize,
                        *dbeta_byte_off as usize,
                        *rows,
                        *h,
                        f32::from_bits(*eps_bits),
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
                    crate::training_bwd_host::run_rope_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *dy_byte_off as usize,
                        *cos_byte_off as usize,
                        *sin_byte_off as usize,
                        *dx_byte_off as usize,
                        *batch,
                        *seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
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
                    crate::training_bwd_host::run_cumsum_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *dy_byte_off as usize,
                        *dx_byte_off as usize,
                        *rows,
                        *cols,
                        *exclusive,
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
                    crate::training_bwd_host::run_gather_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *dy_byte_off as usize,
                        *indices_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
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
        self.read_outputs()
    }

    pub(crate) fn read_outputs(&self) -> Vec<Vec<f32>> {
        self.graph
            .outputs
            .iter()
            .map(|&id| {
                let off_f32 = self.arena.offset(id) / 4;
                let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                let mut host = vec![0.0_f32; elems];
                let src = self.arena.buffer.ptr + (off_f32 as u64) * 4;
                unsafe {
                    let _ = (self.ctx.runtime.hip_memcpy_dtoh)(
                        host.as_mut_ptr() as *mut _,
                        src,
                        elems * 4,
                    );
                }
                host
            })
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
        };
        let (r, _) = step_offsets(&s);
        assert!(!r.contains(&9999), "causal mask must not consume mask_off");
        assert_eq!(r, vec![0, 100, 200]);
    }

    #[test]
    fn step_offsets_attention_custom_mask_pulls_mask() {
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
