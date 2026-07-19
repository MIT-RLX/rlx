#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_fft(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Fft { inverse, norm } = &node.op else {
        unreachable!()
    };
    {
        let shape = &node.shape;
        let meta = rlx_ir::fft::fft_meta(shape);
        let dtype = shape.dtype();
        assert!(
            matches!(
                dtype,
                rlx_ir::DType::F32 | rlx_ir::DType::F64 | rlx_ir::DType::C64
            ),
            "Op::Fft on CPU requires F32, F64, or C64, got {dtype:?}"
        );
        Thunk::Fft1d {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            outer: meta.outer as u32,
            n_complex: meta.n_complex as u32,
            inverse: *inverse,
            norm_tag: norm.tag(),
            dtype,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fft_butterfly_stage(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FftButterflyStage { stage, n_fft } = &node.op else {
        unreachable!()
    };
    {
        let state_shape = graph.node(node.inputs[0]).shape.clone();
        assert_eq!(
            state_shape.dtype(),
            rlx_ir::DType::F32,
            "Op::FftButterflyStage requires F32 state"
        );
        let batch = state_shape.dim(0).unwrap_static() as u32;
        Thunk::FftButterflyStage {
            state_src: node_offset(arena, node.inputs[0]),
            state_dst: node_offset(arena, node.id),
            gate_src: node_offset(arena, node.inputs[1]),
            rev_src: node_offset(arena, node.inputs[2]),
            tw_re_src: node_offset(arena, node.inputs[3]),
            tw_im_src: node_offset(arena, node.inputs[4]),
            batch,
            n_fft: *n_fft,
            stage: *stage,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_log_mel(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LogMel = &node.op else { unreachable!() };
    {
        let spec_shape = graph.node(node.inputs[0]).shape.clone();
        let filt_shape = graph.node(node.inputs[1]).shape.clone();
        let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
            .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
        Thunk::LogMel {
            spec: node_offset(arena, node.inputs[0]),
            filters: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            outer: meta.outer as u32,
            n_fft: meta.n_fft as u32,
            n_bins: meta.n_bins as u32,
            n_mels: meta.n_mels as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_log_mel_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LogMelBackward = &node.op else {
        unreachable!()
    };
    {
        let spec_shape = graph.node(node.inputs[0]).shape.clone();
        let filt_shape = graph.node(node.inputs[1]).shape.clone();
        let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
            .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
        Thunk::LogMelBackward {
            spec: node_offset(arena, node.inputs[0]),
            filters: node_offset(arena, node.inputs[1]),
            dy: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            outer: meta.outer as u32,
            n_fft: meta.n_fft as u32,
            n_bins: meta.n_bins as u32,
            n_mels: meta.n_mels as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_welch_peaks(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::WelchPeaks { k, n_segments } = &node.op else {
        unreachable!()
    };
    {
        let spec_shape = graph.node(node.inputs[0]).shape.clone();
        let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
            .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
        Thunk::WelchPeaks {
            spec: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            welch_batch: meta.welch_batch as u32,
            n_fft: meta.n_fft as u32,
            n_segments: meta.n_segments as u32,
            k: meta.k as u32,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fft1d(t: &Thunk, base: *mut u8) {
    let Thunk::Fft1d {
        src,
        dst,
        outer,
        n_complex,
        inverse,
        norm_tag,
        dtype,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        match dtype {
            rlx_ir::DType::F64 => execute_fft1d_f64(
                *src,
                *dst,
                *outer as usize,
                *n_complex as usize,
                *inverse,
                *norm_tag,
                base,
            ),
            rlx_ir::DType::F32 => execute_fft1d_f32(
                *src,
                *dst,
                *outer as usize,
                *n_complex as usize,
                *inverse,
                *norm_tag,
                base,
            ),
            rlx_ir::DType::C64 => execute_fft1d_c64(
                *src,
                *dst,
                *outer as usize,
                *n_complex as usize,
                *inverse,
                *norm_tag,
                base,
            ),
            other => panic!("Op::Fft on CPU requires F32/F64/C64, got {other:?}"),
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fft_butterfly_stage(t: &Thunk, base: *mut u8) {
    let Thunk::FftButterflyStage {
        state_src,
        state_dst,
        gate_src,
        rev_src,
        tw_re_src,
        tw_im_src,
        batch,
        n_fft,
        stage,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_fft_butterfly_stage_f32(
            *state_src,
            *state_dst,
            *gate_src,
            *rev_src,
            *tw_re_src,
            *tw_im_src,
            *batch as usize,
            *n_fft as usize,
            *stage as usize,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_log_mel(t: &Thunk, base: *mut u8) {
    let Thunk::LogMel {
        spec,
        filters,
        dst,
        outer,
        n_fft,
        n_bins,
        n_mels,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_log_mel_f32(
            *spec,
            *filters,
            *dst,
            *outer as usize,
            *n_fft as usize,
            *n_bins as usize,
            *n_mels as usize,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_log_mel_backward(t: &Thunk, base: *mut u8) {
    let Thunk::LogMelBackward {
        spec,
        filters,
        dy,
        dst,
        outer,
        n_fft,
        n_bins,
        n_mels,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_log_mel_backward_f32(
            *spec,
            *filters,
            *dy,
            *dst,
            *outer as usize,
            *n_fft as usize,
            *n_bins as usize,
            *n_mels as usize,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_welch_peaks(t: &Thunk, base: *mut u8) {
    let Thunk::WelchPeaks {
        spec,
        dst,
        welch_batch,
        n_fft,
        n_segments,
        k,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_welch_peaks_f32(
            *spec,
            *dst,
            *welch_batch as usize,
            *n_fft as usize,
            *n_segments as usize,
            *k as usize,
            base,
        );
    }
}

/// Execute a batched 1D FFT in the f64 2N-real-block layout.
/// Each "row" is `2N` f64 elements: first `N` real, then `N` imag.
/// The `outer` rows are independent and processed sequentially.
///
/// Both forward and inverse use the same Cooley-Tukey radix-2 DIT
/// kernel — only the twiddle-factor sign differs. Power-of-2 only
/// (the IR builder rejects non-power-of-2 sizes at graph-build time).
/// Batched 1D FFT on the f64 2N-real-block layout. Public so other
/// backend crates can invoke this as a host fallback against a
/// unified-memory arena (e.g. rlx-metal: sync the command buffer,
/// pass the Metal `Buffer::contents()` pointer as `base`, restart the
/// command buffer). Self-contained — no rlx-cpu state required.
///
/// Safety: `base + src` and `base + dst` must be valid for the
/// `outer * 2 * n_complex * sizeof::<f64>()` byte range and stay
/// alive for the duration of the call.
pub unsafe fn execute_fft1d_f64(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    base: *mut u8,
) {
    let row_elems = 2 * n_complex;
    let norm = rlx_ir::fft::FftNorm::from_tag(norm_tag);
    let scale = norm.output_scale(n_complex, inverse);
    let is_pow2 = n_complex.is_power_of_two();
    let use_naive = !is_pow2 && n_complex <= 16;
    let scratch_template = if is_pow2 || use_naive {
        BluesteinScratchF64::empty()
    } else {
        BluesteinScratchF64::build(n_complex, inverse)
    };

    let run_row = |bp: *mut u8,
                   re: &mut [f64],
                   im: &mut [f64],
                   scratch: &mut BluesteinScratchF64,
                   o: usize| {
        let row_offset = src + o * row_elems * std::mem::size_of::<f64>();
        let s = unsafe { sl_f64(row_offset, bp, row_elems) };
        re.copy_from_slice(&s[..n_complex]);
        im.copy_from_slice(&s[n_complex..]);
        if is_pow2 {
            fft_radix2_inplace_f64(re, im, inverse);
        } else if use_naive {
            fft_naive_inplace_f64(re, im, inverse);
        } else {
            fft_bluestein_inplace_f64(re, im, inverse, scratch);
        }
        if scale != 1.0 {
            re.iter_mut().for_each(|v| *v *= scale);
            im.iter_mut().for_each(|v| *v *= scale);
        }
        let dst_offset = dst + o * row_elems * std::mem::size_of::<f64>();
        let d = unsafe { sl_mut_f64(dst_offset, bp, row_elems) };
        d[..n_complex].copy_from_slice(re);
        d[n_complex..].copy_from_slice(im);
    };

    let parallel = outer >= 8
        && (outer as u64) * (n_complex as u64) >= (1 << 15)
        && cpu_fft_parallel_enabled();
    if parallel {
        use rayon::prelude::*;
        let p = FftArenaPtr(base);
        (0..outer).into_par_iter().for_each_init(
            || {
                (
                    vec![0f64; n_complex],
                    vec![0f64; n_complex],
                    scratch_template.clone(),
                )
            },
            move |(re, im, scratch), o| run_row(p.ptr(), re, im, scratch, o),
        );
    } else {
        let mut re = vec![0f64; n_complex];
        let mut im = vec![0f64; n_complex];
        let mut scratch = scratch_template;
        for o in 0..outer {
            run_row(base, &mut re, &mut im, &mut scratch, o);
        }
    }
}

/// Ternary pruned radix-2 butterfly stage on `[batch, n_fft, 2]` interleaved state.
pub unsafe fn execute_fft_butterfly_stage_f32(
    state_src: usize,
    state_dst: usize,
    gate_src: usize,
    rev_src: usize,
    tw_re_src: usize,
    tw_im_src: usize,
    batch: usize,
    n_fft: usize,
    stage: usize,
    base: *mut u8,
) {
    let half = n_fft / 2;
    let stride = 1usize << stage;
    let gate = unsafe { sl(gate_src, base, half) };
    let rev = unsafe { sl(rev_src, base, half) };
    let tw_re = unsafe { sl(tw_re_src, base, half) };
    let tw_im = unsafe { sl(tw_im_src, base, half) };
    let row_elems = n_fft * 2;
    for b in 0..batch {
        let in_off = state_src + b * row_elems * std::mem::size_of::<f32>();
        let out_off = state_dst + b * row_elems * std::mem::size_of::<f32>();
        let inp = unsafe { sl(in_off, base, row_elems) };
        let out = unsafe { sl_mut(out_off, base, row_elems) };
        out.copy_from_slice(inp);
        for bf in 0..half {
            if gate[bf] == 0.0 {
                continue;
            }
            let group = bf / stride;
            let k = bf % stride;
            let i0 = group * 2 * stride + k;
            let i1 = i0 + stride;
            let w_re = tw_re[bf];
            let w_im = tw_im[bf];
            let in_a_re = inp[i0 * 2];
            let in_a_im = inp[i0 * 2 + 1];
            let in_b_re = inp[i1 * 2];
            let in_b_im = inp[i1 * 2 + 1];
            let (b_re, b_im) = (
                in_b_re * w_re - in_b_im * w_im,
                in_b_re * w_im + in_b_im * w_re,
            );
            let (top_re, top_im) = (in_a_re + b_re, in_a_im + b_im);
            let (bot_re, bot_im) = (in_a_re - b_re, in_a_im - b_im);
            let (oa_re, oa_im, ob_re, ob_im) = if rev[bf] >= 0.5 {
                (bot_re, bot_im, top_re, top_im)
            } else {
                (top_re, top_im, bot_re, bot_im)
            };
            out[i0 * 2] = oa_re;
            out[i0 * 2 + 1] = oa_im;
            out[i1 * 2] = ob_re;
            out[i1 * 2 + 1] = ob_im;
        }
    }
}

/// f32 mirror of `execute_fft1d_f64`. Same public-host-fallback role.
/// native FFT batch loop: rayon-parallelize rows unless `RLX_FFT_CPU_PARALLEL`
/// is `0`/`off`/`false`. Default on.
pub(crate) fn cpu_fft_parallel_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_CPU_PARALLEL").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Radix-4 CPU FFT for pure powers of four unless `RLX_FFT_RADIX4` is
/// `0`/`off`/`false`. Default on.
pub(crate) fn cpu_fft_radix4_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_RADIX4").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Arena base pointer wrapper that is `Send`/`Sync` so the FFT batch loop can
/// hand it to rayon workers. Safe because every row `o` reads and writes a
/// disjoint `row_elems`-wide slice of the arena (`src`/`dst` are either equal or
/// non-overlapping regions), so concurrent rows never touch the same bytes.
#[derive(Clone, Copy)]
pub(crate) struct FftArenaPtr(*mut u8);
// SAFETY: see FftArenaPtr doc — per-row slices are disjoint.
unsafe impl Send for FftArenaPtr {}
unsafe impl Sync for FftArenaPtr {}
impl FftArenaPtr {
    /// Take by `self` so closures capture the whole (Send/Sync) wrapper rather
    /// than the bare `*mut u8` field (edition-2021 disjoint closure capture).
    #[inline]
    fn ptr(self) -> *mut u8 {
        self.0
    }
}

pub unsafe fn execute_fft1d_f32(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    base: *mut u8,
) {
    let row_elems = 2 * n_complex;
    let norm = rlx_ir::fft::FftNorm::from_tag(norm_tag);
    let scale = norm.output_scale(n_complex, inverse) as f32;
    let is_pow2 = n_complex.is_power_of_two();
    let use_naive = !is_pow2 && n_complex <= 16;
    // Pure powers of four take the radix-4 path (fewer stages); `RLX_FFT_RADIX4=0`
    // forces radix-2 (A/B benchmarking). n<16 stays radix-2 (win negligible).
    let use_radix4 = is_pow2 && n_complex >= 16 && is_pow4(n_complex) && cpu_fft_radix4_enabled();
    let scratch_template = if is_pow2 || use_naive {
        BluesteinScratchF32::empty()
    } else {
        BluesteinScratchF32::build(n_complex, inverse)
    };

    // One independent FFT per row. Writes to `re`/`im` scratch then to the row's
    // arena slice; safe to run rows concurrently (see FftArenaPtr).
    let run_row = |bp: *mut u8,
                   re: &mut [f32],
                   im: &mut [f32],
                   scratch: &mut BluesteinScratchF32,
                   o: usize| {
        let row_offset = src + o * row_elems * std::mem::size_of::<f32>();
        let s = unsafe { sl(row_offset, bp, row_elems) };
        re.copy_from_slice(&s[..n_complex]);
        im.copy_from_slice(&s[n_complex..]);
        if use_radix4 {
            fft_radix4_inplace_f32(re, im, inverse);
        } else if is_pow2 {
            fft_radix2_inplace_f32(re, im, inverse);
        } else if use_naive {
            fft_naive_inplace_f32(re, im, inverse);
        } else {
            fft_bluestein_inplace_f32(re, im, inverse, scratch);
        }
        if scale != 1.0 {
            re.iter_mut().for_each(|v| *v *= scale);
            im.iter_mut().for_each(|v| *v *= scale);
        }
        let dst_offset = dst + o * row_elems * std::mem::size_of::<f32>();
        let d = unsafe { sl_mut(dst_offset, bp, row_elems) };
        d[..n_complex].copy_from_slice(re);
        d[n_complex..].copy_from_slice(im);
    };

    // Rows are independent — parallelize across the batch once there's enough
    // work to amortize rayon's dispatch overhead. `for_each_init` builds the
    // re/im scratch and a scratch-template clone once per worker, not per row.
    // `RLX_FFT_CPU_PARALLEL=0` forces serial (A/B benchmarking / escape hatch).
    // Threshold: measured break-even is ~outer·n ≈ 2^15 (bench_fft_cpu_parallel);
    // below it rayon dispatch dominates and parallel regresses (e.g. 0.5× at
    // n=256 batch=16), so gate on both a row floor and total work.
    let parallel = outer >= 8
        && (outer as u64) * (n_complex as u64) >= (1 << 15)
        && cpu_fft_parallel_enabled();
    if parallel {
        use rayon::prelude::*;
        let p = FftArenaPtr(base);
        (0..outer).into_par_iter().for_each_init(
            || {
                (
                    vec![0f32; n_complex],
                    vec![0f32; n_complex],
                    scratch_template.clone(),
                )
            },
            move |(re, im, scratch), o| run_row(p.ptr(), re, im, scratch, o),
        );
    } else {
        let mut re = vec![0f32; n_complex];
        let mut im = vec![0f32; n_complex];
        let mut scratch = scratch_template;
        for o in 0..outer {
            run_row(base, &mut re, &mut im, &mut scratch, o);
        }
    }
}

/// C64 interleaved layout: each complex element is `[re: f32, im: f32]`.
pub unsafe fn execute_fft1d_c64(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    base: *mut u8,
) {
    let row_bytes = n_complex * 8;
    let norm = rlx_ir::fft::FftNorm::from_tag(norm_tag);
    let scale = norm.output_scale(n_complex, inverse) as f32;
    let is_pow2 = n_complex.is_power_of_two();
    let use_naive = !is_pow2 && n_complex <= 16;
    let scratch_template = if is_pow2 || use_naive {
        BluesteinScratchF32::empty()
    } else {
        BluesteinScratchF32::build(n_complex, inverse)
    };

    // One independent FFT per row (interleaved [re,im] complex layout).
    let run_row = |bp: *mut u8,
                   re: &mut [f32],
                   im: &mut [f32],
                   scratch: &mut BluesteinScratchF32,
                   o: usize| {
        let row_offset = src + o * row_bytes;
        for i in 0..n_complex {
            let elem_off = row_offset + i * 8;
            re[i] = f32::from_le_bytes([
                unsafe { *bp.add(elem_off) },
                unsafe { *bp.add(elem_off + 1) },
                unsafe { *bp.add(elem_off + 2) },
                unsafe { *bp.add(elem_off + 3) },
            ]);
            im[i] = f32::from_le_bytes([
                unsafe { *bp.add(elem_off + 4) },
                unsafe { *bp.add(elem_off + 5) },
                unsafe { *bp.add(elem_off + 6) },
                unsafe { *bp.add(elem_off + 7) },
            ]);
        }
        if is_pow2 {
            fft_radix2_inplace_f32(re, im, inverse);
        } else if use_naive {
            fft_naive_inplace_f32(re, im, inverse);
        } else {
            fft_bluestein_inplace_f32(re, im, inverse, scratch);
        }
        if scale != 1.0 {
            re.iter_mut().for_each(|v| *v *= scale);
            im.iter_mut().for_each(|v| *v *= scale);
        }
        let dst_row = dst + o * row_bytes;
        for i in 0..n_complex {
            let elem_off = dst_row + i * 8;
            let re_b = re[i].to_le_bytes();
            let im_b = im[i].to_le_bytes();
            for j in 0..4 {
                unsafe { *bp.add(elem_off + j) = re_b[j] };
                unsafe { *bp.add(elem_off + 4 + j) = im_b[j] };
            }
        }
    };

    let parallel = outer >= 8
        && (outer as u64) * (n_complex as u64) >= (1 << 15)
        && cpu_fft_parallel_enabled();
    if parallel {
        use rayon::prelude::*;
        let p = FftArenaPtr(base);
        (0..outer).into_par_iter().for_each_init(
            || {
                (
                    vec![0f32; n_complex],
                    vec![0f32; n_complex],
                    scratch_template.clone(),
                )
            },
            move |(re, im, scratch), o| run_row(p.ptr(), re, im, scratch, o),
        );
    } else {
        let mut re = vec![0f32; n_complex];
        let mut im = vec![0f32; n_complex];
        let mut scratch = scratch_template;
        for o in 0..outer {
            run_row(base, &mut re, &mut im, &mut scratch, o);
        }
    }
}

/// Dtype-dispatching host entry for `Op::LogMel` (shared by GPU host fallbacks).
pub unsafe fn execute_log_mel(
    spec: usize,
    filters: usize,
    dst: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    base: *mut u8,
) {
    execute_log_mel_f32(spec, filters, dst, outer, n_fft, n_bins, n_mels, base);
}

pub unsafe fn execute_log_mel_f32(
    spec: usize,
    filters: usize,
    dst: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    base: *mut u8,
) {
    let spec_ptr = base.add(spec) as *const f32;
    let filt_ptr = base.add(filters) as *const f32;
    let dst_ptr = base.add(dst) as *mut f32;
    let spec = std::slice::from_raw_parts(spec_ptr, outer * n_fft * 2);
    let filters = std::slice::from_raw_parts(filt_ptr, n_mels * n_bins);
    let out = std::slice::from_raw_parts_mut(dst_ptr, outer * n_mels);
    rlx_ir::audio::log_mel_block_f32(spec, filters, outer, n_fft, n_bins, n_mels, out);
}

pub unsafe fn execute_welch_peaks_f32(
    spec: usize,
    dst: usize,
    welch_batch: usize,
    n_fft: usize,
    n_segments: usize,
    k: usize,
    base: *mut u8,
) {
    let spec_ptr = base.add(spec) as *const f32;
    let dst_ptr = base.add(dst) as *mut f32;
    let outer = welch_batch * n_segments;
    let spec = std::slice::from_raw_parts(spec_ptr, outer * n_fft * 2);
    let out = std::slice::from_raw_parts_mut(dst_ptr, welch_batch * k * 2);
    rlx_ir::audio::welch_peaks_block_f32(spec, welch_batch, n_fft, n_segments, k, out);
}

pub unsafe fn execute_log_mel_backward_f32(
    spec: usize,
    filters: usize,
    dy: usize,
    dst: usize,
    outer: usize,
    n_fft: usize,
    n_bins: usize,
    n_mels: usize,
    base: *mut u8,
) {
    let spec_ptr = base.add(spec) as *const f32;
    let filt_ptr = base.add(filters) as *const f32;
    let dy_ptr = base.add(dy) as *const f32;
    let dst_ptr = base.add(dst) as *mut f32;
    let spec = std::slice::from_raw_parts(spec_ptr, outer * n_fft * 2);
    let filters = std::slice::from_raw_parts(filt_ptr, n_mels * n_bins);
    let dy = std::slice::from_raw_parts(dy_ptr, outer * n_mels);
    let d_spec = std::slice::from_raw_parts_mut(dst_ptr, outer * n_fft * 2);
    d_spec.fill(0.0);
    rlx_ir::audio::log_mel_block_vjp(spec, filters, dy, outer, n_fft, n_bins, n_mels, d_spec);
}

/// Dtype-dispatching host entry for `Op::Fft` (shared by GPU host fallbacks).
pub unsafe fn execute_fft1d(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    dtype: rlx_ir::DType,
    base: *mut u8,
) {
    match dtype {
        rlx_ir::DType::F32 => {
            execute_fft1d_f32(src, dst, outer, n_complex, inverse, norm_tag, base)
        }
        rlx_ir::DType::F64 => {
            execute_fft1d_f64(src, dst, outer, n_complex, inverse, norm_tag, base)
        }
        rlx_ir::DType::C64 => {
            execute_fft1d_c64(src, dst, outer, n_complex, inverse, norm_tag, base)
        }
        other => panic!("execute_fft1d: unsupported dtype {other:?}"),
    }
}

/// f32 in-place radix-2 DIT Cooley-Tukey. Structurally identical to
/// the f64 path; twiddle recurrence is kept in f64 so accumulated
/// rotation drift doesn't dominate the per-stage error budget at
/// larger N.
pub(crate) fn fft_radix2_inplace_f32(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert!(
        n.is_power_of_two(),
        "fft_radix2_f32: n={n} must be a power of two"
    );
    if n <= 1 {
        return;
    }

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let w_re_step = theta.cos();
        let w_im_step = theta.sin();
        let mut i = 0usize;
        while i < n {
            let mut wre = 1.0_f64;
            let mut wim = 0.0_f64;
            for k in 0..half {
                let wre_f = wre as f32;
                let wim_f = wim as f32;
                let t_re = wre_f * re[i + k + half] - wim_f * im[i + k + half];
                let t_im = wre_f * im[i + k + half] + wim_f * re[i + k + half];
                let u_re = re[i + k];
                let u_im = im[i + k];
                re[i + k] = u_re + t_re;
                im[i + k] = u_im + t_im;
                re[i + k + half] = u_re - t_re;
                im[i + k + half] = u_im - t_im;
                let new_wre = wre * w_re_step - wim * w_im_step;
                let new_wim = wre * w_im_step + wim * w_re_step;
                wre = new_wre;
                wim = new_wim;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// True for a *pure* power of four (log2 even): 4, 16, 64, 256, 1024, 4096, …
/// These take the radix-4 path; other pow-2 sizes (2·4^k) stay on radix-2.
#[inline]
pub(crate) fn is_pow4(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros().is_multiple_of(2)
}

/// In-place radix-4 DIT FFT on split (re, im) f32 arrays for `n = 4^m`.
/// Half the stages of radix-2 (log4 vs log2) → half the array sweeps and
/// twiddle-recurrence iterations, a per-row win that compounds with the
/// rayon batch parallelism. Twiddles run in f64 (as in the radix-2 path).
/// Forward ω = exp(-2πi/n); inverse ω = exp(+2πi/n), no 1/N scale.
pub(crate) fn fft_radix4_inplace_f32(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert!(is_pow4(n), "fft_radix4_f32: n={n} must be a power of four");
    if n <= 1 {
        return;
    }
    let m = n.trailing_zeros() / 2; // number of base-4 digits

    // Base-4 digit-reversal permutation.
    for i in 0..n {
        let mut x = i;
        let mut r = 0usize;
        for _ in 0..m {
            r = (r << 2) | (x & 3);
            x >>= 2;
        }
        if i < r {
            re.swap(i, r);
            im.swap(i, r);
        }
    }

    // `-j` multiply for the odd radix-4 outputs (sign flips for inverse).
    // forward: (-j)(a+bi) = b - ai ; inverse: (+j)(a+bi) = -b + ai.
    let jsign = if inverse { 1.0_f32 } else { -1.0_f32 };
    let tw_sign = if inverse { 1.0_f64 } else { -1.0_f64 };

    let mut len = 4usize;
    while len <= n {
        let q = len / 4;
        let theta = tw_sign * 2.0 * std::f64::consts::PI / (len as f64);
        let wstep_re = theta.cos();
        let wstep_im = theta.sin();
        let mut base = 0usize;
        while base < n {
            let mut w1_re = 1.0_f64;
            let mut w1_im = 0.0_f64;
            for k in 0..q {
                // w2 = w1², w3 = w1³.
                let w2_re = w1_re * w1_re - w1_im * w1_im;
                let w2_im = 2.0 * w1_re * w1_im;
                let w3_re = w2_re * w1_re - w2_im * w1_im;
                let w3_im = w2_re * w1_im + w2_im * w1_re;

                let (i0, i1, i2, i3) = (base + k, base + k + q, base + k + 2 * q, base + k + 3 * q);
                let a_re = re[i0];
                let a_im = im[i0];
                let (w1r, w1i) = (w1_re as f32, w1_im as f32);
                let (w2r, w2i) = (w2_re as f32, w2_im as f32);
                let (w3r, w3i) = (w3_re as f32, w3_im as f32);
                let b_re = w1r * re[i1] - w1i * im[i1];
                let b_im = w1r * im[i1] + w1i * re[i1];
                let c_re = w2r * re[i2] - w2i * im[i2];
                let c_im = w2r * im[i2] + w2i * re[i2];
                let d_re = w3r * re[i3] - w3i * im[i3];
                let d_im = w3r * im[i3] + w3i * re[i3];

                let t0_re = a_re + c_re;
                let t0_im = a_im + c_im;
                let t1_re = a_re - c_re;
                let t1_im = a_im - c_im;
                let t2_re = b_re + d_re;
                let t2_im = b_im + d_im;
                let t3_re = b_re - d_re;
                let t3_im = b_im - d_im;
                // Forward: X[i1] = t1 + (-j·t3), X[i3] = t1 - (-j·t3), where
                // (-j·t3) = (t3_im, -t3_re). jsign=-1 forward / +1 inverse.
                let jt3_re = -jsign * t3_im;
                let jt3_im = jsign * t3_re;

                re[i0] = t0_re + t2_re;
                im[i0] = t0_im + t2_im;
                re[i2] = t0_re - t2_re;
                im[i2] = t0_im - t2_im;
                re[i1] = t1_re + jt3_re;
                im[i1] = t1_im + jt3_im;
                re[i3] = t1_re - jt3_re;
                im[i3] = t1_im - jt3_im;

                let nw_re = w1_re * wstep_re - w1_im * wstep_im;
                let nw_im = w1_re * wstep_im + w1_im * wstep_re;
                w1_re = nw_re;
                w1_im = nw_im;
            }
            base += len;
        }
        len <<= 2;
    }
}

/// In-place radix-2 DIT Cooley-Tukey FFT on split (real, imag) f64
/// arrays. `n = re.len() = im.len()` must be a power of two. Forward
/// uses ω = exp(-2πi/n); inverse uses ω = exp(+2πi/n) (no 1/N scale).
pub(crate) fn fft_radix2_inplace_f64(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert!(
        n.is_power_of_two(),
        "fft_radix2: n={n} must be a power of two"
    );
    if n <= 1 {
        return;
    }

    // Bit-reverse permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies: ω_len = exp(±2πi/len).
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let w_re_step = theta.cos();
        let w_im_step = theta.sin();
        let mut i = 0usize;
        while i < n {
            // Twiddle starts at 1+0i for each segment.
            let mut wre = 1.0_f64;
            let mut wim = 0.0_f64;
            for k in 0..half {
                let t_re = wre * re[i + k + half] - wim * im[i + k + half];
                let t_im = wre * im[i + k + half] + wim * re[i + k + half];
                let u_re = re[i + k];
                let u_im = im[i + k];
                re[i + k] = u_re + t_re;
                im[i + k] = u_im + t_im;
                re[i + k + half] = u_re - t_re;
                im[i + k + half] = u_im - t_im;
                let new_wre = wre * w_re_step - wim * w_im_step;
                let new_wim = wre * w_im_step + wim * w_re_step;
                wre = new_wre;
                wim = new_wim;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Pre-computed chirp + filter-spectrum for one (N, direction) pair.
/// Built once per call to `execute_fft1d_f64` and reused across rows
/// when `outer > 1` — the chirp and FFT(b) don't depend on the input.
#[derive(Clone)]
pub(crate) struct BluesteinScratchF64 {
    /// Power-of-two convolution length, ≥ 2N - 1.
    m: usize,
    /// `w[k] = exp(sign · iπ · k² / N)` for k=0..N, where sign matches
    /// the requested direction. Forward chirp on the way in, output
    /// chirp on the way out.
    w_re: Vec<f64>,
    w_im: Vec<f64>,
    /// FFT of the embedded filter `b[k] = conj(w[|k|])` in length-M.
    /// Doesn't depend on the input — precomputed once.
    bf_re: Vec<f64>,
    bf_im: Vec<f64>,
    /// Workspace reused per row (avoids per-row allocation).
    ar: Vec<f64>,
    ai: Vec<f64>,
}

impl BluesteinScratchF64 {
    fn empty() -> Self {
        Self {
            m: 0,
            w_re: Vec::new(),
            w_im: Vec::new(),
            bf_re: Vec::new(),
            bf_im: Vec::new(),
            ar: Vec::new(),
            ai: Vec::new(),
        }
    }

    fn build(n: usize, inverse: bool) -> Self {
        // M = next power of two ≥ 2N - 1 keeps the inner FFT on the
        // fast radix-2 path. For N=1 fall back to M=1 (no-op convolution).
        let m = if n <= 1 {
            1
        } else {
            (2 * n - 1).next_power_of_two()
        };

        // Chirp arg reduced via k² mod 2N — without this, large N
        // bleeds precision into the trig call (n² grows quadratically).
        let mod_2n = (2 * n) as u64;
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut w_re = vec![0.0_f64; n];
        let mut w_im = vec![0.0_f64; n];
        for k in 0..n {
            let k2 = (k as u64).wrapping_mul(k as u64) % mod_2n;
            let theta = sign * std::f64::consts::PI * (k2 as f64) / (n as f64);
            w_re[k] = theta.cos();
            w_im[k] = theta.sin();
        }

        // Embed b[k] = conj(w[|k|]) into length M with the negative
        // indices wrapping to the tail: b[-j] → B[M-j] for j=1..N-1.
        let mut bf_re = vec![0.0_f64; m];
        let mut bf_im = vec![0.0_f64; m];
        if n > 0 {
            bf_re[0] = w_re[0];
            bf_im[0] = -w_im[0];
            for k in 1..n {
                bf_re[k] = w_re[k];
                bf_im[k] = -w_im[k];
                bf_re[m - k] = w_re[k];
                bf_im[m - k] = -w_im[k];
            }
        }
        if m > 1 {
            fft_radix2_inplace_f64(&mut bf_re, &mut bf_im, false);
        }

        Self {
            m,
            w_re,
            w_im,
            bf_re,
            bf_im,
            ar: vec![0.0_f64; m],
            ai: vec![0.0_f64; m],
        }
    }
}

/// Direct O(N²) DFT for small non-pow2 N (faster than Bluestein setup).
pub(crate) fn fft_naive_inplace_f64(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    for k in 0..n {
        for nn in 0..n {
            let theta = sign * 2.0 * std::f64::consts::PI * (nn as f64) * (k as f64) / (n as f64);
            let c = theta.cos();
            let s = theta.sin();
            out_re[k] += re[nn] * c - im[nn] * s;
            out_im[k] += re[nn] * s + im[nn] * c;
        }
    }
    re.copy_from_slice(&out_re);
    im.copy_from_slice(&out_im);
}

pub(crate) fn fft_naive_inplace_f32(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    let sign = if inverse { 1.0f32 } else { -1.0f32 };
    let mut out_re = vec![0.0_f32; n];
    let mut out_im = vec![0.0_f32; n];
    for k in 0..n {
        for nn in 0..n {
            let theta = sign * 2.0 * std::f32::consts::PI * (nn as f32) * (k as f32) / (n as f32);
            let c = theta.cos();
            let s = theta.sin();
            out_re[k] += re[nn] * c - im[nn] * s;
            out_im[k] += re[nn] * s + im[nn] * c;
        }
    }
    re.copy_from_slice(&out_re);
    im.copy_from_slice(&out_im);
}

/// Bluestein (chirp-z) FFT for arbitrary N. Identity used:
///   `n·k = (n² + k² - (k-n)²) / 2`
/// which lets the DFT be written as a linear convolution sandwiched
/// between two chirp multiplies:
///   `X[k] = w[k] · ((x·w) ⊛ conj(w))[k]`   where `w[n] = exp(±iπ·n²/N)`.
/// The convolution is computed via a length-M radix-2 FFT (M ≥ 2N-1).
/// Both directions stay unnormalized to match the radix-2 path, so the
/// chain rule keeps working without scaling.
pub(crate) fn fft_bluestein_inplace_f64(
    re: &mut [f64],
    im: &mut [f64],
    _inverse: bool,
    s: &mut BluesteinScratchF64,
) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert_eq!(s.w_re.len(), n);
    if n <= 1 {
        return;
    }
    let m = s.m;

    // Pre-chirp: a[k] = x[k] · w[k], zero-padded to M.
    for k in 0..m {
        s.ar[k] = 0.0;
        s.ai[k] = 0.0;
    }
    for k in 0..n {
        s.ar[k] = re[k] * s.w_re[k] - im[k] * s.w_im[k];
        s.ai[k] = re[k] * s.w_im[k] + im[k] * s.w_re[k];
    }

    // Length-M forward FFT of the padded chirped input.
    fft_radix2_inplace_f64(&mut s.ar, &mut s.ai, false);

    // Pointwise product with FFT(b). Stored back into (ar, ai).
    for k in 0..m {
        let ar = s.ar[k];
        let ai = s.ai[k];
        let br = s.bf_re[k];
        let bi = s.bf_im[k];
        s.ar[k] = ar * br - ai * bi;
        s.ai[k] = ar * bi + ai * br;
    }

    // Inverse FFT — radix-2 here is the unnormalized inverse, so we
    // divide by M to recover the true circular convolution.
    fft_radix2_inplace_f64(&mut s.ar, &mut s.ai, true);
    let inv_m = 1.0 / (m as f64);

    // Post-chirp: X[k] = w[k] · Y[k] / M for k = 0..N.
    for k in 0..n {
        let yr = s.ar[k] * inv_m;
        let yi = s.ai[k] * inv_m;
        re[k] = yr * s.w_re[k] - yi * s.w_im[k];
        im[k] = yr * s.w_im[k] + yi * s.w_re[k];
    }
}

/// f32 mirror of `BluesteinScratchF64`. Chirp is computed in f64 for
/// precision (same justification as the radix-2 f32 path: twiddles in
/// f64, butterflies in f32). The actual conv buffers are f32.
///
/// `Clone` lets each rayon worker own a copy (read-only `w`/`bf` tables plus
/// its own `ar`/`ai` convolution workspace) when the batch loop is parallel.
#[derive(Clone)]
pub(crate) struct BluesteinScratchF32 {
    m: usize,
    w_re: Vec<f32>,
    w_im: Vec<f32>,
    bf_re: Vec<f32>,
    bf_im: Vec<f32>,
    ar: Vec<f32>,
    ai: Vec<f32>,
}

impl BluesteinScratchF32 {
    fn empty() -> Self {
        Self {
            m: 0,
            w_re: Vec::new(),
            w_im: Vec::new(),
            bf_re: Vec::new(),
            bf_im: Vec::new(),
            ar: Vec::new(),
            ai: Vec::new(),
        }
    }

    fn build(n: usize, inverse: bool) -> Self {
        let m = if n <= 1 {
            1
        } else {
            (2 * n - 1).next_power_of_two()
        };

        let mod_2n = (2 * n) as u64;
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut w_re = vec![0.0_f32; n];
        let mut w_im = vec![0.0_f32; n];
        for k in 0..n {
            let k2 = (k as u64).wrapping_mul(k as u64) % mod_2n;
            let theta = sign * std::f64::consts::PI * (k2 as f64) / (n as f64);
            w_re[k] = theta.cos() as f32;
            w_im[k] = theta.sin() as f32;
        }

        let mut bf_re = vec![0.0_f32; m];
        let mut bf_im = vec![0.0_f32; m];
        if n > 0 {
            bf_re[0] = w_re[0];
            bf_im[0] = -w_im[0];
            for k in 1..n {
                bf_re[k] = w_re[k];
                bf_im[k] = -w_im[k];
                bf_re[m - k] = w_re[k];
                bf_im[m - k] = -w_im[k];
            }
        }
        if m > 1 {
            fft_radix2_inplace_f32(&mut bf_re, &mut bf_im, false);
        }

        Self {
            m,
            w_re,
            w_im,
            bf_re,
            bf_im,
            ar: vec![0.0_f32; m],
            ai: vec![0.0_f32; m],
        }
    }
}

pub(crate) fn fft_bluestein_inplace_f32(
    re: &mut [f32],
    im: &mut [f32],
    _inverse: bool,
    s: &mut BluesteinScratchF32,
) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert_eq!(s.w_re.len(), n);
    if n <= 1 {
        return;
    }
    let m = s.m;

    for k in 0..m {
        s.ar[k] = 0.0;
        s.ai[k] = 0.0;
    }
    for k in 0..n {
        s.ar[k] = re[k] * s.w_re[k] - im[k] * s.w_im[k];
        s.ai[k] = re[k] * s.w_im[k] + im[k] * s.w_re[k];
    }

    fft_radix2_inplace_f32(&mut s.ar, &mut s.ai, false);

    for k in 0..m {
        let ar = s.ar[k];
        let ai = s.ai[k];
        let br = s.bf_re[k];
        let bi = s.bf_im[k];
        s.ar[k] = ar * br - ai * bi;
        s.ai[k] = ar * bi + ai * br;
    }

    fft_radix2_inplace_f32(&mut s.ar, &mut s.ai, true);
    let inv_m = 1.0_f32 / (m as f32);

    for k in 0..n {
        let yr = s.ar[k] * inv_m;
        let yi = s.ai[k] * inv_m;
        re[k] = yr * s.w_re[k] - yi * s.w_im[k];
        im[k] = yr * s.w_im[k] + yi * s.w_re[k];
    }
}
