// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin `DeviceArena` host-fallback adapters for this GPU backend.
//!
//! Each function builds the backend arena wrapper and forwards to
//! [`rlx_gpu_host`]. Prefer adding new host-staged ops there first, then
//! a one-liner here (or via [`rlx_gpu_host::forward_arena_op!`]).

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use rlx_ir::DType;
use std::sync::Arc;

pub fn run_fft1d(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    src_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    dtype: DType,
) {
    let _ = arena_size_bytes;
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_fft1d(
        &mut arena,
        src_byte_off,
        dst_byte_off,
        outer,
        n_complex,
        inverse,
        norm_tag,
        dtype,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gated_delta_net(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    q_byte_off: usize,
    k_byte_off: usize,
    v_byte_off: usize,
    g_byte_off: usize,
    beta_byte_off: usize,
    state_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    state_size: usize,
    use_carry: bool,
    gate_per_channel: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gated_delta_net(
        &mut arena,
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
        gate_per_channel,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_im2col(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    x_byte_off: usize,
    col_byte_off: usize,
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
) {
    // im2col addresses a sub-span, so `size_bytes` is unused by the shared fn;
    // pass 0 rather than plumbing the (here-irrelevant) full arena size.
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_im2col(
        &mut arena,
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
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_llada2_group_limited_gate(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    sig_f32_off: usize,
    route_f32_off: usize,
    out_f32_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_llada2_group_limited_gate(
        &mut arena,
        sig_f32_off,
        route_f32_off,
        out_f32_off,
        n_elems,
        attrs,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_log_mel_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dy_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    pre_sync: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_log_mel_backward(
        &mut arena,
        spec_byte_off,
        filt_byte_off,
        dy_byte_off,
        dst_byte_off,
        outer,
        n_fft,
        n_bins,
        n_mels,
        pre_sync,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_log_mel(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    pre_sync: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_log_mel(
        &mut arena,
        spec_byte_off,
        filt_byte_off,
        dst_byte_off,
        outer,
        n_fft,
        n_bins,
        n_mels,
        pre_sync,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_lstm(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x_byte_off: usize,
    w_ih_byte_off: usize,
    w_hh_byte_off: usize,
    bias_byte_off: usize,
    h0_byte_off: usize,
    c0_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_lstm(
        &mut arena,
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
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gru(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x_byte_off: usize,
    w_ih_byte_off: usize,
    w_hh_byte_off: usize,
    b_ih_byte_off: usize,
    b_hh_byte_off: usize,
    h0_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gru(
        &mut arena,
        x_byte_off,
        w_ih_byte_off,
        w_hh_byte_off,
        b_ih_byte_off,
        b_hh_byte_off,
        h0_byte_off,
        dst_byte_off,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_rnn(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x_byte_off: usize,
    w_ih_byte_off: usize,
    w_hh_byte_off: usize,
    bias_byte_off: usize,
    h0_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    relu: bool,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_rnn(
        &mut arena,
        x_byte_off,
        w_ih_byte_off,
        w_hh_byte_off,
        bias_byte_off,
        h0_byte_off,
        dst_byte_off,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
        relu,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_mamba2(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    x_byte_off: usize,
    dt_byte_off: usize,
    a_byte_off: usize,
    b_byte_off: usize,
    c_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_mamba2(
        &mut arena,
        x_byte_off,
        dt_byte_off,
        a_byte_off,
        b_byte_off,
        c_byte_off,
        dst_byte_off,
        batch,
        seq,
        heads,
        head_dim,
        state_size,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_ms_deform_attn(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    in_offs: &[(u32, u32)],
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_ms_deform_attn(&mut arena, in_offs, out_off, out_len, attrs);
}

rlx_gpu_host::forward_arena_op! {
    pub fn run_umap_knn(
        stream: &Arc<CudaStream>,
        buffer: &mut CudaSlice<f32>,
        _arena_size_bytes: usize
        ;
        pairwise_f32_off: usize,
        out_f32_off: usize,
        n: usize,
        k: usize,
    ) {
        CudaArena {
            stream,
            buffer,
            size_bytes: 0,
        }
    }
}

rlx_gpu_host::forward_arena_op! {
    pub fn run_welch_peaks(
        stream: &Arc<CudaStream>,
        buffer: &mut CudaSlice<f32>
        ;
        spec_byte_off: usize,
        dst_byte_off: usize,
        welch_batch: usize,
        n_fft: usize,
        n_segments: usize,
        k: usize,
        pre_sync: bool,
    ) {
        CudaArena {
            stream,
            buffer,
            size_bytes: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    positions_off: usize,
    positions_len: usize,
    scales_off: usize,
    scales_len: usize,
    rotations_off: usize,
    rotations_len: usize,
    opacities_off: usize,
    opacities_len: usize,
    colors_off: usize,
    colors_len: usize,
    sh_coeffs_off: usize,
    sh_coeffs_len: usize,
    meta_off: usize,
    dst_off: usize,
    dst_len: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gaussian_splat_render(
        &mut arena,
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
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_render_backward(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    positions_off: usize,
    positions_len: usize,
    scales_off: usize,
    scales_len: usize,
    rotations_off: usize,
    rotations_len: usize,
    opacities_off: usize,
    opacities_len: usize,
    colors_off: usize,
    colors_len: usize,
    sh_coeffs_off: usize,
    sh_coeffs_len: usize,
    meta_off: usize,
    d_loss_off: usize,
    d_loss_len: usize,
    packed_off: usize,
    packed_len: usize,
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
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gaussian_splat_render_backward(
        &mut arena,
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
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_prepare(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    positions_off: usize,
    positions_len: usize,
    scales_off: usize,
    scales_len: usize,
    rotations_off: usize,
    rotations_len: usize,
    opacities_off: usize,
    opacities_len: usize,
    colors_off: usize,
    colors_len: usize,
    sh_coeffs_off: usize,
    sh_coeffs_len: usize,
    meta_off: usize,
    meta_len: usize,
    prep_off: usize,
    prep_len: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gaussian_splat_prepare(
        &mut arena,
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
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_gaussian_splat_rasterize(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    prep_off: usize,
    prep_len: usize,
    meta_off: usize,
    meta_len: usize,
    dst_off: usize,
    dst_len: usize,
    count: usize,
    width: u32,
    height: u32,
    tile_size: u32,
    alpha_cutoff: f32,
    max_splat_steps: u32,
    transmittance_threshold: f32,
    max_list_entries: u32,
) {
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: arena_size_bytes,
    };
    rlx_gpu_host::run_gaussian_splat_rasterize(
        &mut arena,
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
    );
}
