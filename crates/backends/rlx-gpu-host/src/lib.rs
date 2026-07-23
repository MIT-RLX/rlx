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

//! Backend-agnostic host-fallback kernels for RLX GPU backends.
//!
//! Ops without a native GPU kernel run on the host: copy the device arena
//! down, run the shared [`rlx_cpu`] implementation, copy back
//! (`D2H → CPU → H2D`). The only per-backend variation is the memcpy itself,
//! captured by the [`DeviceArena`] trait — so the staging logic lives here
//! once instead of being copy-pasted into `rlx-cuda`, `rlx-rocm`, …
//!
//! All offsets are exactly as the callers passed them historically: some ops
//! address the arena in **bytes** (byte-pointer CPU kernels), others in
//! **f32 elements** (typed CPU kernels). This crate is layout-neutral — it
//! just forwards those offsets to the same `rlx_cpu` function the backends
//! already called.

#![allow(clippy::too_many_arguments)]

mod collective;
mod custom;
mod gguf;
mod rng;
mod scan;
mod spd;
mod training_bwd;
mod vision;

pub use collective::{COLLECTIVE_OPS, run_collective_bytes, run_collective_f32};
pub use custom::{
    clear_custom_param_cache, dtype_bytes_to_f32, f32_slots_to_dtype, has_host_kernel,
    run_custom_host_bytes, run_custom_host_f32,
};
pub use gguf::{
    gguf_scheme_id, run_dequant_grouped_matmul_gguf, run_dequant_matmul_gguf, scheme_from_id,
    upload_param_bytes,
};
pub use rng::{run_rng_normal, run_rng_uniform};
pub use scan::{
    HostTensorCache, run_host_op, run_host_op_packed, run_host_op_packed_cached, run_host_op_span,
    run_indexing, run_scan, run_scan_span,
};
pub use spd::{SpdInput, eval as eval_spd, is_spd_host, run_spd, run_spd_spans};
pub use training_bwd::{
    run_conv2d_backward_input, run_conv2d_backward_weight, run_conv2d_forward,
    run_maxpool2d_backward,
};
pub use vision::{
    run_conv_transpose2d_nchw, run_conv_transpose3d_ncdhw, run_group_norm_nchw, run_gru,
    run_layer_norm2d_nchw, run_mamba2, run_resize_nearest_2x, run_rnn,
};

pub mod vmath;

/// Forward a host-fallback op through a backend-specific [`DeviceArena`] wrapper.
///
/// Parameters before `;` build the arena; parameters after are forwarded to the
/// shared `$crate::$name` kernel unchanged.
///
/// ```ignore
/// rlx_gpu_host::forward_arena_op! {
///     pub fn run_umap_knn(
///         stream: &Arc<CudaStream>,
///         buffer: &mut CudaSlice<f32>,
///         _arena_size_bytes: usize
///         ;
///         pairwise_f32_off: usize,
///         out_f32_off: usize,
///         n: usize,
///         k: usize,
///     ) {
///         CudaArena { stream, buffer, size_bytes: 0 }
///     }
/// }
/// ```
#[macro_export]
macro_rules! forward_arena_op {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $($ctor_arg:ident : $ctor_ty:ty),+ $(,)?
            ; $($fwd_arg:ident : $fwd_ty:ty),* $(,)?
        ) {
            $arena:expr
        }
    ) => {
        $(#[$meta])*
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($ctor_arg : $ctor_ty,)+ $($fwd_arg : $fwd_ty),*) {
            let mut arena = $arena;
            $crate::$name(&mut arena, $($fwd_arg),*);
        }
    };
}

/// Staging interface a GPU backend implements over its device arena so the
/// shared host-fallback kernels below can move bytes to/from the host.
///
/// Implementations wrap the backend's stream/queue + arena buffer. Offsets are
/// **byte** offsets into the arena; `dtoh`/`htod` lengths are in bytes and are
/// expected to be 4-byte (f32) aligned.
pub trait DeviceArena {
    /// Total arena size in bytes (for whole-arena mirror ops).
    fn arena_bytes(&self) -> usize;
    /// Block until all previously-enqueued device work has completed.
    fn sync(&mut self);
    /// Copy `dst.len()` bytes from the arena at `byte_off` into `dst`.
    fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]);
    /// Copy `src` into the arena at `byte_off`.
    fn htod(&mut self, byte_off: usize, src: &[u8]);
}

/// Mirror the entire arena to a host byte buffer, run `f` against it (using the
/// arena's original byte offsets), then copy the whole buffer back. This is the
/// staging pattern for ops that touch scattered, hard-to-bound regions.
#[inline]
pub(crate) fn with_whole_arena<A: DeviceArena>(a: &mut A, f: impl FnOnce(&mut [u8])) {
    let n = a.arena_bytes();
    a.sync();
    let mut host = vec![0u8; n];
    a.dtoh(0, &mut host);
    f(&mut host);
    a.htod(0, &host);
}

// ---------------------------------------------------------------------------
// Whole-arena mirror ops
// ---------------------------------------------------------------------------

/// Host-side `Op::GatedDeltaNet`.
pub fn run_gated_delta_net<A: DeviceArena>(
    a: &mut A,
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
) {
    assert!(
        state_size <= rlx_cpu::gdn::GDN_MAX_STATE,
        "GatedDeltaNet: state_size {state_size} > {}",
        rlx_cpu::gdn::GDN_MAX_STATE
    );
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_gated_delta_net_f32(
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            if use_carry { state_byte_off } else { 0 },
            dst_byte_off,
            batch,
            seq,
            heads,
            state_size,
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::Lstm`.
pub fn run_lstm<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::thunk::execute_lstm_f32(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::Custom("gdino.ms_deform_attn")`. `in_offs`/`out_off` are
/// f32-element offsets into the arena.
pub fn run_ms_deform_attn<A: DeviceArena>(
    a: &mut A,
    in_offs: &[(u32, u32)],
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    let offs: Vec<(usize, usize)> = in_offs
        .iter()
        .map(|&(o, l)| (o as usize, l as usize))
        .collect();
    with_whole_arena(a, |host| {
        let f32s: &mut [f32] = bytemuck::cast_slice_mut(host);
        rlx_cpu::ms_deform_attn::execute_in_arena(f32s, &offs, out_off, out_len, attrs)
            .expect("ms_deform_attn host execute failed");
    });
}

/// Host-side `Op::Custom("llada2.group_limited_gate")`. Offsets are
/// f32-element offsets into the arena.
pub fn run_llada2_group_limited_gate<A: DeviceArena>(
    a: &mut A,
    sig_f32_off: usize,
    route_f32_off: usize,
    out_f32_off: usize,
    n_elems: usize,
    attrs: &[u8],
) {
    with_whole_arena(a, |host| {
        let f32s: &mut [f32] = bytemuck::cast_slice_mut(host);
        rlx_cpu::llada2_gate::execute_gate_in_f32_arena(
            f32s,
            sig_f32_off,
            route_f32_off,
            out_f32_off,
            n_elems,
            attrs,
        )
        .expect("llada2 group-limited gate host execute failed");
    });
}

// ---------------------------------------------------------------------------
// Sub-range / span ops (copy only the touched region)
// ---------------------------------------------------------------------------

/// Host-side `Op::Im2Col`. Stages only the span covering the `x` and `col`
/// tensors, runs against span-relative offsets, and copies the span back.
pub fn run_im2col<A: DeviceArena>(
    a: &mut A,
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
    let per_batch = (c_in as usize) * (h as usize) * (w as usize);
    let n_eff = if n == 0 { 0 } else { n as usize };
    let m = n_eff * h_out as usize * w_out as usize;
    let k = (c_in as usize) * (kh as usize) * (kw as usize);
    let x_len = if n == 0 {
        per_batch.max(1)
    } else {
        n_eff * per_batch
    };
    let col_len = if n == 0 { k.max(1) } else { m * k };
    let span_start = x_byte_off.min(col_byte_off);
    let span_end = (x_byte_off + x_len * 4).max(col_byte_off + col_len * 4);
    let span_len = span_end.saturating_sub(span_start);

    a.sync();
    let mut host = vec![0u8; span_len];
    a.dtoh(span_start, &mut host);
    unsafe {
        rlx_cpu::im2col::execute_im2col_rows_layout(
            x_byte_off - span_start,
            col_byte_off - span_start,
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
            host.as_mut_ptr(),
        );
    }
    a.htod(span_start, &host);
}

/// Host-side `Op::Custom("umap.knn")`. Reads only the `pairwise` matrix and
/// writes only the packed kNN output. Offsets are f32-element offsets.
pub fn run_umap_knn<A: DeviceArena>(
    a: &mut A,
    pairwise_f32_off: usize,
    out_f32_off: usize,
    n: usize,
    k: usize,
) {
    let pw_len = n * n;
    let out_len = n * 2 * k;

    a.sync();
    let mut pairwise_bytes = vec![0u8; pw_len * 4];
    a.dtoh(pairwise_f32_off * 4, &mut pairwise_bytes);
    let pairwise: &[f32] = bytemuck::cast_slice(&pairwise_bytes);

    let mut packed = vec![0f32; out_len];
    rlx_cpu::umap_knn::knn_forward_packed(pairwise, n, k, &mut packed);

    a.htod(out_f32_off * 4, bytemuck::cast_slice(&packed));
}

/// Host-side `Op::WelchPeaks`. Stages the span covering `spec` and `dst`, runs
/// the CPU reference against span-relative byte offsets, copies the span back.
/// `pre_sync` gates the device sync (some call sites already synced).
pub fn run_welch_peaks<A: DeviceArena>(
    a: &mut A,
    spec_byte_off: usize,
    dst_byte_off: usize,
    welch_batch: usize,
    n_fft: usize,
    n_segments: usize,
    k: usize,
    pre_sync: bool,
) {
    let spec_len = welch_batch * n_segments * n_fft * 2;
    let dst_len = welch_batch * k * 2;
    let span_off = spec_byte_off.min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4).max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;
    assert_eq!(
        span_off % 4,
        0,
        "welch_peaks_host: span_off must be f32-aligned"
    );
    assert_eq!(
        span_len % 4,
        0,
        "welch_peaks_host: span_len must be f32-aligned"
    );

    if pre_sync {
        a.sync();
    }

    let mut host = vec![0u8; span_len];
    a.dtoh(span_off, &mut host);

    unsafe {
        rlx_cpu::thunk::execute_welch_peaks_f32(
            spec_byte_off - span_off,
            dst_byte_off - span_off,
            welch_batch,
            n_fft,
            n_segments,
            k,
            host.as_mut_ptr(),
        );
    }

    a.htod(span_off, &host);
}

/// Host-side `Op::LogMel`. Stages the span covering `spec`, `filt` and `dst`.
pub fn run_log_mel<A: DeviceArena>(
    a: &mut A,
    spec_byte_off: usize,
    filt_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    pre_sync: bool,
) {
    let spec_len = outer * n_fft * 2;
    let filt_len = n_mels * n_bins;
    let dst_len = outer * n_mels;
    let span_off = spec_byte_off.min(filt_byte_off).min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4)
        .max(filt_byte_off + filt_len * 4)
        .max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;
    assert_eq!(
        span_off % 4,
        0,
        "log_mel_host: span_off must be f32-aligned"
    );
    assert_eq!(
        span_len % 4,
        0,
        "log_mel_host: span_len must be f32-aligned"
    );

    if pre_sync {
        a.sync();
    }

    let mut host = vec![0u8; span_len];
    a.dtoh(span_off, &mut host);

    unsafe {
        rlx_cpu::thunk::execute_log_mel_f32(
            spec_byte_off - span_off,
            filt_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_fft,
            n_bins,
            n_mels,
            host.as_mut_ptr(),
        );
    }

    a.htod(span_off, &host);
}

/// Host-side `Op::LogMel` backward. Stages the span covering `spec`, `filt`,
/// `dy` and `dst`.
pub fn run_log_mel_backward<A: DeviceArena>(
    a: &mut A,
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
    let spec_len = outer * n_fft * 2;
    let filt_len = n_mels * n_bins;
    let dy_len = outer * n_mels;
    let dst_len = outer * n_fft * 2;
    let span_off = spec_byte_off
        .min(filt_byte_off)
        .min(dy_byte_off)
        .min(dst_byte_off);
    let span_end = (spec_byte_off + spec_len * 4)
        .max(filt_byte_off + filt_len * 4)
        .max(dy_byte_off + dy_len * 4)
        .max(dst_byte_off + dst_len * 4);
    let span_len = span_end - span_off;
    assert_eq!(
        span_off % 4,
        0,
        "log_mel_backward_host: span_off must be f32-aligned"
    );
    assert_eq!(
        span_len % 4,
        0,
        "log_mel_backward_host: span_len must be f32-aligned"
    );

    if pre_sync {
        a.sync();
    }

    let mut host = vec![0u8; span_len];
    a.dtoh(span_off, &mut host);

    unsafe {
        rlx_cpu::thunk::execute_log_mel_backward_f32(
            spec_byte_off - span_off,
            filt_byte_off - span_off,
            dy_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_fft,
            n_bins,
            n_mels,
            host.as_mut_ptr(),
        );
    }

    a.htod(span_off, &host);
}

/// Host-side `Op::Fft` (1-D). Uses [`rlx_ir::fft`] to compute the touched
/// arena byte span, then stages just that span.
pub fn run_fft1d<A: DeviceArena>(
    a: &mut A,
    src_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    dtype: rlx_ir::DType,
) {
    let meta = rlx_ir::fft::FftMeta {
        outer,
        n_complex,
        axis_extent: match dtype {
            rlx_ir::DType::C64 => n_complex,
            rlx_ir::DType::F32 | rlx_ir::DType::F64 => n_complex * 2,
            other => panic!("fft_host: unsupported dtype {other:?}"),
        },
    };
    let row_bytes = meta.row_bytes(dtype);
    let (span_off, span_len) =
        rlx_ir::fft::fft_arena_byte_span(src_byte_off, dst_byte_off, row_bytes, outer);
    assert_eq!(span_off % 4, 0, "fft_host: span_off must be f32-aligned");
    assert_eq!(span_len % 4, 0, "fft_host: span_len must be f32-aligned");

    a.sync();

    let mut host = vec![0u8; span_len];
    a.dtoh(span_off, &mut host);

    unsafe {
        rlx_cpu::thunk::execute_fft1d(
            src_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_complex,
            inverse,
            norm_tag,
            dtype,
            host.as_mut_ptr(),
        );
    }

    a.htod(span_off, &host);
}

/// Sync, copy `[span_start, span_end)` device→host (rounded out to f32
/// boundaries so byte-oriented spans stay 4-aligned for the memcpy), run `body`
/// against the host base (offsets must be span-relative), copy host→device.
/// `span_start` is f32-aligned (arena allocations are f32).
#[inline]
fn stage_span<A: DeviceArena>(
    a: &mut A,
    span_start: usize,
    span_end: usize,
    body: impl FnOnce(*mut u8),
) {
    if span_end <= span_start {
        return;
    }
    a.sync();
    let span_start_f32 = span_start / 4;
    let span_end_f32 = span_end.div_ceil(4);
    let mut host = vec![0u8; (span_end_f32 - span_start_f32) * 4];
    a.dtoh(span_start_f32 * 4, &mut host);
    body(host.as_mut_ptr());
    a.htod(span_start_f32 * 4, &host);
}

/// Host-side `Op::Reverse` (batch-general reverse/flip, dtype-agnostic).
pub fn run_reverse<A: DeviceArena>(
    a: &mut A,
    src: usize,
    dst: usize,
    dims: &[u32],
    rev_mask: &[bool],
    elem_bytes: usize,
) {
    let total: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
    let bytes = total * elem_bytes;
    let span_start = src.min(dst);
    let span_end = (src + bytes).max(dst + bytes);
    stage_span(a, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_reverse(
            src - span_start,
            dst - span_start,
            dims,
            rev_mask,
            elem_bytes,
            base,
        );
    });
}

/// Host-side `Op::ArgMax`/`Op::ArgMin` (f32-encoded indices) over the middle
/// `reduced` axis.
pub fn run_argreduce<A: DeviceArena>(
    a: &mut A,
    src: usize,
    dst: usize,
    outer: usize,
    reduced: usize,
    inner: usize,
    is_max: bool,
) {
    let in_bytes = outer * reduced * inner * 4;
    let out_bytes = outer * inner * 4;
    let span_start = src.min(dst);
    let span_end = (src + in_bytes).max(dst + out_bytes);
    stage_span(a, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_argreduce_f32(
            src - span_start,
            dst - span_start,
            outer,
            reduced,
            inner,
            is_max,
            base,
        );
    });
}

/// Host-side `Op::AxialRope2d` on `[batch, seq, hidden]`.
pub fn run_axial_rope2d<A: DeviceArena>(
    a: &mut A,
    src: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
    end_x: usize,
    end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
) {
    let bytes = batch * seq * hidden * 4;
    let span_start = src.min(dst);
    let span_end = (src + bytes).max(dst + bytes);
    stage_span(a, span_start, span_end, |base| unsafe {
        rlx_cpu::thunk::execute_axial_rope2d_f32(
            src - span_start,
            dst - span_start,
            batch,
            seq,
            hidden,
            end_x,
            end_y,
            head_dim,
            num_heads,
            theta,
            repeat_factor,
            base,
        );
    });
}

// ---------------------------------------------------------------------------
// Gaussian-splat whole-arena ops
// ---------------------------------------------------------------------------

/// Host-side `Op::GaussianSplatRender`. Byte offsets.
pub fn run_gaussian_splat_render<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::splat::execute_gaussian_splat_render(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::GaussianSplatRender` backward. Byte offsets.
pub fn run_gaussian_splat_render_backward<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::splat::execute_gaussian_splat_render_backward(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::GaussianSplatRender` prepare stage. Byte offsets.
pub fn run_gaussian_splat_prepare<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::splat::execute_gaussian_splat_prepare(
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
            host.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::GaussianSplatRender` rasterize stage. Byte offsets.
pub fn run_gaussian_splat_rasterize<A: DeviceArena>(
    a: &mut A,
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
    with_whole_arena(a, |host| unsafe {
        rlx_cpu::splat::execute_gaussian_splat_rasterize(
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
            host.as_mut_ptr(),
        );
    });
}

// ---------------------------------------------------------------------------
// Training-backward whole-arena ops
// ---------------------------------------------------------------------------

/// Host-side `Op::RmsNorm` backward (input gradient). Byte offsets.
pub fn run_rms_norm_backward_input<A: DeviceArena>(
    a: &mut A,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_input_f32(
            x,
            gamma,
            beta,
            dy,
            dx,
            rows,
            h,
            eps,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::RmsNorm` backward (gamma gradient). Byte offsets.
pub fn run_rms_norm_backward_gamma<A: DeviceArena>(
    a: &mut A,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dgamma: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_gamma_f32(
            x,
            gamma,
            beta,
            dy,
            dgamma,
            rows,
            h,
            eps,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::RmsNorm` backward (beta gradient). Byte offsets.
pub fn run_rms_norm_backward_beta<A: DeviceArena>(
    a: &mut A,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dbeta: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_rms_norm_backward_beta_f32(
            x,
            gamma,
            beta,
            dy,
            dbeta,
            rows,
            h,
            eps,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::Rope` backward. Byte offsets.
pub fn run_rope_backward<A: DeviceArena>(
    a: &mut A,
    dy: usize,
    cos: usize,
    sin: usize,
    dx: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    cos_len: u32,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_rope_backward_f32(
            dy,
            cos,
            sin,
            dx,
            batch,
            seq,
            hidden,
            head_dim,
            n_rot,
            cos_len,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::CumSum` backward. Byte offsets.
pub fn run_cumsum_backward<A: DeviceArena>(
    a: &mut A,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_cumsum_backward_f32(
            dy,
            dx,
            rows,
            cols,
            exclusive,
            base.as_mut_ptr(),
        );
    });
}

/// Host-side `Op::Gather` backward. Byte offsets.
pub fn run_gather_backward<A: DeviceArena>(
    a: &mut A,
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    with_whole_arena(a, |base| unsafe {
        rlx_cpu::thunk::execute_gather_backward_f32(
            dy,
            indices,
            dst,
            outer,
            axis_dim,
            num_idx,
            trailing,
            base.as_mut_ptr(),
        );
    });
}

#[cfg(test)]
mod tests {
    //! Equivalence guards for the staging math: running an op *through* the
    //! [`DeviceArena`] staging path must produce byte-identical results to
    //! calling the underlying `rlx_cpu` kernel directly, in-place, at the same
    //! absolute offsets. These catch offset-arithmetic bugs (span extraction,
    //! relative-offset rebase, f32-element ↔ byte scaling) with no GPU.
    use super::*;

    /// A host-memory stand-in for a device arena: `dtoh`/`htod` are plain
    /// memcpys against a `Vec<u8>` that plays the role of device memory.
    struct VecArena {
        data: Vec<u8>,
    }
    impl DeviceArena for VecArena {
        fn arena_bytes(&self) -> usize {
            self.data.len()
        }
        fn sync(&mut self) {}
        fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
            dst.copy_from_slice(&self.data[byte_off..byte_off + dst.len()]);
        }
        fn htod(&mut self, byte_off: usize, src: &[u8]) {
            self.data[byte_off..byte_off + src.len()].copy_from_slice(src);
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(v).to_vec()
    }

    #[test]
    fn im2col_staged_matches_inplace() {
        // 1×1×3×3 input, 2×2 kernel, stride 1, no pad/dil → 2×2 output.
        let (n, c_in, h, w) = (1u32, 1u32, 3u32, 3u32);
        let (kh, kw) = (2u32, 2u32);
        let (h_out, w_out) = (2u32, 2u32);
        let x_off = 0usize; // 9 f32
        let col_off = 9 * 4usize; // 16 f32, f32-aligned after x
        let arena_bytes = col_off + 16 * 4;

        // Seed the arena: x = 1..=9, col region zeroed.
        let mut init = vec![0u8; arena_bytes];
        init[x_off..x_off + 9 * 4]
            .copy_from_slice(&f32_bytes(&(1..=9).map(|i| i as f32).collect::<Vec<_>>()));

        // Direct in-place reference.
        let mut direct = init.clone();
        unsafe {
            rlx_cpu::im2col::execute_im2col_rows_layout(
                x_off,
                col_off,
                n,
                c_in,
                h,
                w,
                h_out,
                w_out,
                kh,
                kw,
                1,
                1,
                0,
                0,
                1,
                1,
                direct.as_mut_ptr(),
            );
        }

        // Staged through the arena.
        let mut arena = VecArena { data: init };
        run_im2col(
            &mut arena, x_off, col_off, n, c_in, h, w, h_out, w_out, kh, kw, 1, 1, 0, 0, 1, 1,
        );

        assert_eq!(arena.data, direct, "im2col staging diverged from in-place");
        // Sanity: the col region is actually populated (not all-zero).
        assert!(arena.data[col_off..].iter().any(|&b| b != 0));
    }

    #[test]
    fn umap_knn_staged_matches_direct() {
        // 4 points, k=2. Pairwise distance matrix at offset 0, output after it.
        let (n, k) = (4usize, 2usize);
        let pairwise: Vec<f32> = vec![
            0.0, 1.0, 4.0, 9.0, //
            1.0, 0.0, 1.0, 4.0, //
            4.0, 1.0, 0.0, 1.0, //
            9.0, 4.0, 1.0, 0.0,
        ];
        let out_len = n * 2 * k;
        let pw_off_f32 = 0usize;
        let out_off_f32 = n * n; // f32 elements
        let arena_bytes = (out_off_f32 + out_len) * 4;

        let mut init = vec![0u8; arena_bytes];
        init[pw_off_f32 * 4..pw_off_f32 * 4 + n * n * 4].copy_from_slice(&f32_bytes(&pairwise));

        // Direct reference.
        let mut expected = vec![0f32; out_len];
        rlx_cpu::umap_knn::knn_forward_packed(&pairwise, n, k, &mut expected);

        // Staged.
        let mut arena = VecArena { data: init };
        run_umap_knn(&mut arena, pw_off_f32, out_off_f32, n, k);

        let got: &[f32] =
            bytemuck::cast_slice(&arena.data[out_off_f32 * 4..out_off_f32 * 4 + out_len * 4]);
        assert_eq!(got, expected.as_slice(), "umap.knn staging diverged");
    }

    #[test]
    fn gguf_packed_upload_rmw_and_aligned() {
        let mut arena = VecArena {
            data: vec![0u8; 16],
        };
        // Aligned write of 8 bytes.
        upload_param_bytes(&mut arena, 0, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&arena.data[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        // Unaligned RMW splice of 3 bytes at offset 1.
        upload_param_bytes(&mut arena, 1, &[0xAA, 0xBB, 0xCC]);
        assert_eq!(arena.data[0], 1);
        assert_eq!(&arena.data[1..4], &[0xAA, 0xBB, 0xCC]);
        assert_eq!(&arena.data[4..8], &[5, 6, 7, 8]);
    }

    #[test]
    fn rng_normal_writes_htod_only() {
        let mut arena = VecArena {
            data: vec![0u8; 16],
        };
        run_rng_normal(
            &mut arena,
            0,
            4,
            0.0,
            1.0,
            42,
            None,
            rlx_ir::RngOptions::default(),
        );
        let got: &[f32] = bytemuck::cast_slice(&arena.data);
        assert!(got.iter().any(|&x| x != 0.0), "rng fill left zeros");
    }

    #[test]
    fn custom_f32_slot_dtype_roundtrip_i64() {
        let slots = [1.0f32, -2.0, 3.0];
        let bytes = f32_slots_to_dtype(&slots, rlx_ir::DType::I64);
        let back = dtype_bytes_to_f32(&bytes, rlx_ir::DType::I64);
        assert_eq!(back, slots);
    }
}
