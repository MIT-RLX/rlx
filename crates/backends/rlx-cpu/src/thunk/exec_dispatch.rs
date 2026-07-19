#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Execute a thunk schedule on a raw arena buffer.
/// Fastest executor: call pre-compiled closures sequentially.
/// Zero match dispatch — each closure is a direct kernel call.
pub fn execute_compiled(schedule: &ThunkSchedule, arena_buf: &mut [u8]) {
    let base = arena_buf.as_mut_ptr();
    for f in &schedule.compiled_fns {
        f(base);
    }
}

/// Active-extent execution stub. The runtime calls this when it has an
/// active-extent hint set. CPU doesn't implement per-thunk active-extent
/// scaling yet — return false so the caller falls back to the full
/// `execute_thunks` path.
pub fn execute_thunks_active(
    schedule: &ThunkSchedule,
    _arena_buf: &mut [u8],
    _actual: usize,
    _upper: usize,
) -> bool {
    let _ = schedule;
    false
}

/// Match-based executor (fallback, used by tests).
pub(crate) struct MoeResidencyGuard;
impl Drop for MoeResidencyGuard {
    fn drop(&mut self) {
        if let Some(stats) = crate::moe_residency::take_stats() {
            crate::moe_residency::stash_last_forward_stats(stats);
        } else {
            crate::moe_residency::clear_mask();
        }
    }
}

/// Contiguous AVX2 Add/Mul/Sub. Large buffers are split across Rayon so we
/// keep SIMD *and* multi-core (serial AVX2 alone lost to 16-wide scalar rayon
/// on F5-TTS DiT residuals).
#[cfg(target_arch = "x86_64")]
#[inline]
fn binary_contig_f32(l: &[f32], r: &[f32], o: &mut [f32], op: BinaryOp) -> bool {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return false;
    }
    if !matches!(op, BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub) {
        return false;
    }
    let len = o.len();
    let l_ptr = l.as_ptr() as usize;
    let r_ptr = r.as_ptr() as usize;
    let o_ptr = o.as_mut_ptr() as usize;
    let run = |i0: usize, i1: usize| unsafe {
        use std::arch::x86_64::*;
        let l = l_ptr as *const f32;
        let r = r_ptr as *const f32;
        let o = o_ptr as *mut f32;
        let mut i = i0;
        // Align to 8-wide when possible within the chunk.
        while i + 8 <= i1 {
            let a = _mm256_loadu_ps(l.add(i));
            let b = _mm256_loadu_ps(r.add(i));
            let res = match op {
                BinaryOp::Add => _mm256_add_ps(a, b),
                BinaryOp::Sub => _mm256_sub_ps(a, b),
                BinaryOp::Mul => _mm256_mul_ps(a, b),
                _ => unreachable!(),
            };
            _mm256_storeu_ps(o.add(i), res);
            i += 8;
        }
        while i < i1 {
            let a = *l.add(i);
            let b = *r.add(i);
            *o.add(i) = match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                _ => unreachable!(),
            };
            i += 1;
        }
    };
    if len >= 8192 && crate::pool::num_threads() > 1 {
        crate::pool::par_for(len, crate::pool::chunk_floor(len), &|off, cnt| {
            run(off, off + cnt);
        });
    } else {
        run(0, len);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
#[allow(dead_code)]
fn binary_contig_f32(_l: &[f32], _r: &[f32], _o: &mut [f32], _op: BinaryOp) -> bool {
    false
}

/// Row-tiled rhs broadcast: `o[i] = op(l[i], r[i % rl])` with optional AVX2.
#[inline]
fn binary_row_bcast_f32(l: &[f32], r: &[f32], o: &mut [f32], op: BinaryOp, rl: usize) -> bool {
    let len = o.len();
    if rl == 0 || rl >= len || !len.is_multiple_of(rl) || l.len() < len || r.len() < rl {
        return false;
    }
    let rows = len / rl;
    let l_ptr = l.as_ptr() as usize;
    let r_ptr = r.as_ptr() as usize;
    let o_ptr = o.as_mut_ptr() as usize;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = rl >= 8
        && rl.is_multiple_of(8)
        && matches!(op, BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub)
        && std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let _use_avx2 = false;

    let run_rows = |row0: usize, row1: usize| unsafe {
        let l = l_ptr as *const f32;
        let r = r_ptr as *const f32;
        let o = o_ptr as *mut f32;
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            use std::arch::x86_64::*;
            let chunks = rl / 8;
            for row in row0..row1 {
                let base = row * rl;
                for c in 0..chunks {
                    let off = base + c * 8;
                    let roff = c * 8;
                    let a = _mm256_loadu_ps(l.add(off));
                    let b = _mm256_loadu_ps(r.add(roff));
                    let res = match op {
                        BinaryOp::Add => _mm256_add_ps(a, b),
                        BinaryOp::Sub => _mm256_sub_ps(a, b),
                        BinaryOp::Mul => _mm256_mul_ps(a, b),
                        _ => unreachable!(),
                    };
                    _mm256_storeu_ps(o.add(off), res);
                }
            }
            return;
        }
        for row in row0..row1 {
            let base = row * rl;
            for j in 0..rl {
                let i = base + j;
                let a = *l.add(i);
                let b = *r.add(j);
                *o.add(i) = match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Max => a.max(b),
                    BinaryOp::Min => a.min(b),
                    BinaryOp::Pow => a.powf(b),
                };
            }
        }
    };
    if rows >= 4 && crate::pool::num_threads() > 1 && len >= 8192 {
        crate::pool::par_for(rows, 1, &|off, cnt| run_rows(off, off + cnt));
    } else {
        run_rows(0, rows);
    }
    true
}

pub(crate) fn thunk_kind_name(t: &Thunk) -> &'static str {
    match t {
        Thunk::Nop => "Nop",
        Thunk::Gather { .. } => "Gather",
        Thunk::GatherAxis { .. } => "GatherAxis",
        Thunk::TopK { .. } => "TopK",
        Thunk::Copy { .. } => "Copy",
        Thunk::CopyF64 { .. } => "CopyF64",
        Thunk::CopyI64 { .. } => "CopyI64",
        Thunk::CastF32ToI64 { .. } => "CastF32ToI64",
        Thunk::CastI64ToF32 { .. } => "CastI64ToF32",
        Thunk::CastBoolToI32 { .. } => "CastBoolToI32",
        Thunk::CastBoolToF32 { .. } => "CastBoolToF32",
        Thunk::CastF32ToBool { .. } => "CastF32ToBool",
        Thunk::CastI32ToF32 { .. } => "CastI32ToF32",
        Thunk::CastI32ToI64 { .. } => "CastI32ToI64",
        Thunk::CastI32ToBool { .. } => "CastI32ToBool",
        Thunk::CastI64ToBool { .. } => "CastI64ToBool",
        Thunk::CastBoolToI64 { .. } => "CastBoolToI64",
        Thunk::Transpose { .. } => "Transpose",
        Thunk::TransposeF64 { .. } => "TransposeF64",
        Thunk::Where { .. } => "Where",
        Thunk::Fma { .. } => "Fma",
        Thunk::Compare { .. } => "Compare",
        Thunk::BinaryFull { .. } => "BinaryFull",
        Thunk::BinaryFullF64 { .. } => "BinaryFullF64",
        Thunk::Sgemm { .. } => "Sgemm",
        Thunk::SgemmT { .. } => "SgemmT",
        Thunk::SgdMomentum { .. } => "SgdMomentum",
        Thunk::Dgemm { .. } => "Dgemm",
        Thunk::FusedMmBiasAct { .. } => "FusedMmBiasAct",
        Thunk::BiasAdd { .. } => "BiasAdd",
        Thunk::LayerNorm { .. } => "LayerNorm",
        Thunk::Softmax { .. } => "Softmax",
        Thunk::Conv2D { .. } => "Conv2D",
        Thunk::Conv2D1x1 { .. } => "Conv2D1x1",
        Thunk::Conv3d { .. } => "Conv3d",
        Thunk::ConvTranspose3d { .. } => "ConvTranspose3d",
        Thunk::CustomOp { .. } => "CustomOp",
        Thunk::ActivationInPlace { .. } => "ActivationInPlace",
        Thunk::Narrow { .. } => "Narrow",
        Thunk::Cumsum { .. } => "Cumsum",
        Thunk::Reduce { .. } => "Reduce",
        Thunk::BatchedSgemm { .. } => "BatchedSgemm",
        Thunk::DequantMatMul { .. } => "DequantMatMul",
        Thunk::Quantize { .. } => "Quantize",
        Thunk::Dequantize { .. } => "Dequantize",
        Thunk::ConvTranspose2d { .. } => "ConvTranspose2d",
        Thunk::ResizeNearest2x { .. } => "ResizeNearest2x",
        Thunk::ElementwiseRegion { .. } => "ElementwiseRegion",
        Thunk::Conv2dBackwardInput { .. } => "Conv2dBackwardInput",
        Thunk::Conv2dBackwardWeight { .. } => "Conv2dBackwardWeight",
        Thunk::Pool2D { .. } => "Pool2D",
        Thunk::MaxPool2dBackward { .. } => "MaxPool2dBackward",
        Thunk::ReluBackward { .. } => "ReluBackward",
        Thunk::ActivationBackward { .. } => "ActivationBackward",
        Thunk::Im2Col { .. } => "Im2Col",
        Thunk::SoftmaxCrossEntropyDense { .. } => "SoftmaxCrossEntropyDense",
        Thunk::SoftmaxCrossEntropy { .. } => "SoftmaxCrossEntropy",
        Thunk::SoftmaxCrossEntropyBackward { .. } => "SoftmaxCrossEntropyBackward",
        Thunk::Attention { .. } => "Attention",
        Thunk::AdaLayerNorm { .. } => "AdaLayerNorm",
        Thunk::GatedResidual { .. } => "GatedResidual",
        Thunk::Rope { .. } => "Rope",
        Thunk::Concat { .. } => "Concat",
        Thunk::RmsNorm { .. } => "RmsNorm",
        Thunk::FusedResidualLN { .. } => "FusedResidualLN",
        Thunk::FusedSwiGLU { .. } => "FusedSwiGLU",
        Thunk::AxialRope2d { .. } => "AxialRope2d",
        _ => "Other",
    }
}

/// Per-thunk-kind wall-time accumulator, populated only when the env var
/// `RLX_PROFILE_THUNKS` is set. Used to see which ops dominate a step so the
/// optimizer/kernels can target the real hotspots rather than guesses.
pub(crate) static THUNK_PROFILE: std::sync::Mutex<
    Option<std::collections::BTreeMap<&'static str, (u128, u64)>>,
> = std::sync::Mutex::new(None);

#[inline]
pub(crate) fn profile_record(name: &'static str, d: std::time::Duration) {
    let mut g = THUNK_PROFILE.lock().unwrap();
    let map = g.get_or_insert_with(std::collections::BTreeMap::new);
    let e = map.entry(name).or_insert((0, 0));
    e.0 += d.as_nanos();
    e.1 += 1;
}

/// Print and clear the per-thunk-kind time profile gathered under
/// `RLX_PROFILE_THUNKS`. Call after a run to see where the time went.
pub fn dump_thunk_profile() {
    let mut g = THUNK_PROFILE.lock().unwrap();
    if let Some(map) = g.take() {
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by_key(|b| std::cmp::Reverse(b.1.0));
        let total: u128 = v.iter().map(|(_, (ns, _))| *ns).sum();
        eprintln!(
            "[thunk-profile] total {:.1}ms across kinds:",
            total as f64 / 1e6
        );
        for (name, (ns, c)) in v.iter().take(25) {
            eprintln!("  {name:<28} {:>8.1}ms  ({c} calls)", *ns as f64 / 1e6);
        }
    }
}

pub fn execute_thunks(schedule: &ThunkSchedule, arena_buf: &mut [u8]) {
    crate::moe_residency::reset_gmm_counters();
    if let Some(layers) = schedule.moe_resident_layers.clone() {
        crate::moe_residency::set_per_layer_masks(Some(layers));
    } else {
        crate::moe_residency::set_mask(schedule.moe_resident.clone());
    }
    if let Some(cap) = schedule.moe_topk_capture.as_ref() {
        cap.clear();
    }
    let _moe_guard = MoeResidencyGuard;
    let base = arena_buf.as_mut_ptr();
    let mask_thr = schedule.mask_threshold;
    let mask_neg = schedule.mask_neg_inf;
    let score_thr = schedule.score_skip;
    let thunks = &schedule.thunks;
    let len = thunks.len();

    // Pre-allocate ALL reusable buffers once (zero per-call allocation)
    let max_h = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::FusedResidualLN { h, .. }
            | Thunk::FusedResidualRmsNorm { h, .. }
            | Thunk::LayerNorm { h, .. } => Some(*h as usize),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let zero_bias = vec![0f32; max_h];

    // Pre-allocate per-(batch,head) score buffers for parallel SDPA.
    // Q/K/V/out are accessed via strided BLAS — no deinterleave copy needed.
    let max_sdpa = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::Attention {
                batch,
                seq,
                kv_seq,
                heads,
                head_dim,
                ..
            } => Some((
                *batch as usize,
                (*seq as usize).max(*kv_seq as usize),
                *heads as usize,
                *head_dim as usize,
            )),
            _ => None,
        })
        .fold((0, 0, 0, 0), |(mb, ms, mh, md), (b, s, h, d)| {
            (mb.max(b), ms.max(s), mh.max(h), md.max(d))
        });
    let (max_batch, max_seq, max_heads, _max_dh) = max_sdpa;
    let max_units = max_batch * max_heads;
    let mut sdpa_scores = vec![0f32; max_units * max_seq * max_seq];

    // Pre-allocate fused layer buffers (reused across all 12+ layers — zero malloc per layer)
    let fl = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::FusedBertLayer {
                batch,
                seq,
                hs,
                int_dim,
                ..
            } => {
                let m = (*batch as usize) * (*seq as usize);
                let h = *hs as usize;
                let id = *int_dim as usize;
                Some((m, h, id, m * (*seq as usize)))
            }
            Thunk::FusedNomicLayer {
                batch,
                seq,
                hs,
                int_dim,
                ..
            } => {
                let m = (*batch as usize) * (*seq as usize);
                let h = *hs as usize;
                let id = *int_dim as usize;
                Some((m, h, id, m * (*seq as usize)))
            }
            _ => None,
        })
        .fold((0, 0, 0, 0), |(mm, mh, mi, ms), (m, h, id, ss)| {
            (mm.max(m), mh.max(h), mi.max(id), ms.max(ss))
        });
    let (fl_m, fl_h, fl_int, fl_ss) = fl;
    let mut fl_qkv = vec![0f32; fl_m * 3 * fl_h];
    let mut fl_attn = vec![0f32; fl_m * fl_h];
    let mut fl_res = vec![0f32; fl_m * fl_h];
    let mut fl_normed = vec![0f32; fl_m * fl_h];
    let mut fl_ffn = vec![0f32; fl_m * fl_int.max(2 * fl_int)]; // Nomic needs 2×int for fused fc11+fc12
    let mut fl_sc = vec![0f32; fl_ss.max(1)];

    let trace_thunks = std::env::var_os("RLX_TRACE_THUNK").is_some();
    if trace_thunks {
        eprintln!(
            "[thunk] prealloc max_h={max_h} sdpa={} fl_m={fl_m} fl_h={fl_h} fl_int={fl_int}",
            max_units * max_seq * max_seq
        );
    }
    let profile = std::env::var_os("RLX_PROFILE_THUNKS").is_some();
    // Time the previous thunk at the top of each iteration (avoids touching the
    // giant match's many arms). The last thunk's tail is folded into the next
    // step's first sample — negligible over a training run.
    let mut prof_prev: Option<(&'static str, std::time::Instant)> = None;
    for i in 0..len {
        if profile {
            if let Some((pn, pt)) = prof_prev.take() {
                profile_record(pn, pt.elapsed());
            }
        }
        let thunk = unsafe { thunks.get_unchecked(i) };
        if trace_thunks && (i < 120 || i % 200 == 0 || i + 1 == len) {
            eprintln!("[thunk {i}/{len}] {}", thunk_kind_name(thunk));
        }
        let trace_done = trace_thunks && i < 120;
        if profile {
            prof_prev = Some((thunk_kind_name(thunk), std::time::Instant::now()));
        }
        match thunk {
            Thunk::Nop => exec_nop(thunk),
            Thunk::ElementwiseRegion { .. } => exec_elementwise_region(thunk, base),
            Thunk::GaussianSplatRender { .. } => exec_gaussian_splat_render(thunk, base),
            Thunk::GaussianSplatRenderBackward { .. } => {
                exec_gaussian_splat_render_backward(thunk, base)
            }
            Thunk::GaussianSplatPrepare { .. } => exec_gaussian_splat_prepare(thunk, base),
            Thunk::GaussianSplatRasterize { .. } => exec_gaussian_splat_rasterize(thunk, base),
            Thunk::Fft1d { .. } => exec_fft1d(thunk, base),
            Thunk::FftButterflyStage { .. } => exec_fft_butterfly_stage(thunk, base),
            Thunk::LogMel { .. } => exec_log_mel(thunk, base),
            Thunk::LogMelBackward { .. } => exec_log_mel_backward(thunk, base),
            Thunk::WelchPeaks { .. } => exec_welch_peaks(thunk, base),
            Thunk::CustomFn { .. } => exec_custom_fn(thunk, base),
            Thunk::Sgemm { a, b, c, m, k, n } => {
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                if trace_thunks {
                    eprintln!("[sgemm] m={m} k={k} n={n} a={} b={} c={}", *a, *b, *c);
                }
                let c_len = m.saturating_mul(n);
                let a_len = m.saturating_mul(k);
                let b_len = k.saturating_mul(n);
                let arena_len = arena_buf.len();
                let max_a = (arena_len.saturating_sub(*a)) / 4;
                let max_b = (arena_len.saturating_sub(*b)) / 4;
                let max_c = (arena_len.saturating_sub(*c)) / 4;
                let a_len = a_len.min(max_a);
                let b_len = b_len.min(max_b);
                let c_len = c_len.min(max_c);
                unsafe {
                    let a_sl = sl(*a, base, a_len);
                    let b_sl = sl(*b, base, b_len);
                    let c_sl = sl_mut(*c, base, c_len);
                    if std::ptr::eq(a_sl.as_ptr(), c_sl.as_ptr())
                        || std::ptr::eq(b_sl.as_ptr(), c_sl.as_ptr())
                    {
                        let mut tmp = vec![0.0f32; c_len];
                        crate::blas::sgemm_auto(a_sl, b_sl, &mut tmp, m, k, n);
                        c_sl.copy_from_slice(&tmp);
                    } else {
                        crate::blas::sgemm_auto(a_sl, b_sl, c_sl, m, k, n);
                    }
                }
            }

            Thunk::SgemmT {
                a,
                b,
                c,
                m,
                k,
                n,
                ta,
                tb,
            } => {
                // C[m,n] = op(A) @ op(B). RowMajor cblas: lda/ldb = stored
                // row-length of each operand → m if A is transposed else k;
                // k if B is transposed else n. Element counts are m*k / k*n
                // regardless of layout.
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                let lda = if *ta { m } else { k };
                let ldb = if *tb { k } else { n };
                let arena_len = arena_buf.len();
                let a_len = (m * k).min((arena_len.saturating_sub(*a)) / 4);
                let b_len = (k * n).min((arena_len.saturating_sub(*b)) / 4);
                let c_len = (m * n).min((arena_len.saturating_sub(*c)) / 4);
                unsafe {
                    let a_sl = sl(*a, base, a_len);
                    let b_sl = sl(*b, base, b_len);
                    let c_sl = sl_mut(*c, base, c_len);
                    let (ap, bp) = (a_sl.as_ptr(), b_sl.as_ptr());
                    if std::ptr::eq(ap, c_sl.as_ptr()) || std::ptr::eq(bp, c_sl.as_ptr()) {
                        let mut tmp = vec![0.0f32; c_len];
                        crate::blas::sgemm_general(
                            ap,
                            bp,
                            tmp.as_mut_ptr(),
                            m,
                            n,
                            k,
                            1.0,
                            0.0,
                            lda,
                            ldb,
                            n,
                            *ta,
                            *tb,
                        );
                        c_sl.copy_from_slice(&tmp);
                    } else {
                        crate::blas::sgemm_general(
                            ap,
                            bp,
                            c_sl.as_mut_ptr(),
                            m,
                            n,
                            k,
                            1.0,
                            0.0,
                            lda,
                            ldb,
                            n,
                            *ta,
                            *tb,
                        );
                    }
                }
            }

            Thunk::SgdMomentum { .. } => exec_sgd_momentum(thunk, base),
            Thunk::CgemmC64 { .. } => exec_cgemm_c64(thunk, base),
            Thunk::DenseSolveF64 { .. } => exec_dense_solve_f64(thunk, base),
            Thunk::DenseSolveF32 { .. } => exec_dense_solve_f32(thunk, base),
            Thunk::BatchedDenseSolveF64 { .. } => exec_batched_dense_solve_f64(thunk, base),
            Thunk::BatchedDenseSolveF32 { .. } => exec_batched_dense_solve_f32(thunk, base),
            Thunk::BatchedDgemmF64 { .. } => exec_batched_dgemm_f64(thunk, base),
            Thunk::BatchedSgemm {
                a,
                b,
                c,
                batch,
                m,
                k,
                n,
                a_bcast,
                b_bcast,
            } => {
                let (b_, m_, k_, n_) = (*batch as usize, *m as usize, *k as usize, *n as usize);
                if trace_thunks {
                    eprintln!(
                        "[batched-sgemm] batch={b_} m={m_} k={k_} n={n_} a_bcast={a_bcast} b_bcast={b_bcast} a={} b={} c={}",
                        *a, *b, *c
                    );
                }
                let a_mat = m_.saturating_mul(k_); // per-matrix element count
                let b_mat = k_.saturating_mul(n_);
                let c_mat = m_.saturating_mul(n_);
                // A broadcast operand (batch dim 1) has batch stride 0 → reuse
                // matrix 0 for every output batch; else stride by its matrix size.
                let a_bstride = if *a_bcast { 0 } else { a_mat };
                let b_bstride = if *b_bcast { 0 } else { b_mat };
                let arena_len = arena_buf.len();
                let a_cap = (arena_len.saturating_sub(*a)) / 4;
                let b_cap = (arena_len.saturating_sub(*b)) / 4;
                let c_cap = (arena_len.saturating_sub(*c)) / 4;
                let a_count = if *a_bcast { 1 } else { b_ };
                let b_count = if *b_bcast { 1 } else { b_ };
                let a_elems = (a_count * a_mat).min(a_cap);
                let b_elems = (b_count * b_mat).min(b_cap);
                let c_elems = (b_ * c_mat).min(c_cap);
                unsafe {
                    let a_full = sl(*a, base, a_elems);
                    let b_full = sl(*b, base, b_elems);
                    let c_full = sl_mut(*c, base, c_elems);
                    // Parallelize independent batch GEMMs with Rayon (BLAS
                    // stays 1-thread per worker). Serial for tiny batches.
                    if b_ >= 2 && crate::pool::num_threads() > 1 {
                        let a_ptr = a_full.as_ptr() as usize;
                        let b_ptr = b_full.as_ptr() as usize;
                        let c_ptr = c_full.as_mut_ptr() as usize;
                        let a_len = a_full.len();
                        let b_len = b_full.len();
                        let c_len = c_full.len();
                        crate::pool::par_for(b_, 1, &|off, cnt| {
                            for bi in off..off + cnt {
                                let a0 = bi * a_bstride;
                                let b0 = bi * b_bstride;
                                let c0 = bi * c_mat;
                                if a0 + a_mat > a_len || b0 + b_mat > b_len || c0 + c_mat > c_len {
                                    break;
                                }
                                // SAFETY: pointers from disjoint batch slices; c
                                // ranges don't overlap across workers.
                                let a_slice = std::slice::from_raw_parts(
                                    (a_ptr as *const f32).add(a0),
                                    a_mat,
                                );
                                let b_slice = std::slice::from_raw_parts(
                                    (b_ptr as *const f32).add(b0),
                                    b_mat,
                                );
                                let c_slice = std::slice::from_raw_parts_mut(
                                    (c_ptr as *mut f32).add(c0),
                                    c_mat,
                                );
                                if std::ptr::eq(a_slice.as_ptr(), c_slice.as_mut_ptr())
                                    || std::ptr::eq(b_slice.as_ptr(), c_slice.as_mut_ptr())
                                {
                                    let mut tmp = vec![0.0f32; c_mat];
                                    crate::blas::sgemm(a_slice, b_slice, &mut tmp, m_, k_, n_);
                                    c_slice.copy_from_slice(&tmp);
                                } else {
                                    crate::blas::sgemm(a_slice, b_slice, c_slice, m_, k_, n_);
                                }
                            }
                        });
                    } else {
                        for bi in 0..b_ {
                            let a0 = bi * a_bstride;
                            let b0 = bi * b_bstride;
                            let c0 = bi * c_mat;
                            if a0 + a_mat > a_full.len()
                                || b0 + b_mat > b_full.len()
                                || c0 + c_mat > c_full.len()
                            {
                                break;
                            }
                            let a_slice = &a_full[a0..a0 + a_mat];
                            let b_slice = &b_full[b0..b0 + b_mat];
                            let c_slice = &mut c_full[c0..c0 + c_mat];
                            if std::ptr::eq(a_slice.as_ptr(), c_slice.as_mut_ptr())
                                || std::ptr::eq(b_slice.as_ptr(), c_slice.as_mut_ptr())
                            {
                                let mut tmp = vec![0.0f32; c_mat];
                                crate::blas::sgemm_auto(a_slice, b_slice, &mut tmp, m_, k_, n_);
                                c_slice.copy_from_slice(&tmp);
                            } else {
                                crate::blas::sgemm_auto(a_slice, b_slice, c_slice, m_, k_, n_);
                            }
                        }
                    }
                }
            }

            Thunk::Dgemm { .. } => exec_dgemm(thunk, base),
            Thunk::TransposeF64 { .. } => exec_transpose_f64(thunk, base),
            Thunk::ActivationF64 { .. } => exec_activation_f64(thunk, base),
            Thunk::ReduceSumF64 { .. } => exec_reduce_sum_f64(thunk, base),
            Thunk::CopyF64 { src, dst, len } => {
                let mut len = *len as usize;
                if *src == *dst || len == 0 {
                    continue;
                }
                let arena_len = arena_buf.len();
                let max_from_src = (arena_len.saturating_sub(*src)) / 8;
                let max_from_dst = (arena_len.saturating_sub(*dst)) / 8;
                len = len.min(max_from_src).min(max_from_dst);
                if len == 0 {
                    continue;
                }
                let byte_len = len.saturating_mul(8);
                unsafe {
                    std::ptr::copy(base.add(*src), base.add(*dst), byte_len);
                }
            }

            Thunk::CopyI64 { src, dst, len } => {
                let mut len = *len as usize;
                if *src == *dst || len == 0 {
                    continue;
                }
                let arena_len = arena_buf.len();
                let max_from_src = (arena_len.saturating_sub(*src)) / 8;
                let max_from_dst = (arena_len.saturating_sub(*dst)) / 8;
                len = len.min(max_from_src).min(max_from_dst);
                if len == 0 {
                    continue;
                }
                let byte_len = len.saturating_mul(8);
                unsafe {
                    std::ptr::copy(base.add(*src), base.add(*dst), byte_len);
                }
            }

            Thunk::CastF32ToI64 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl(*src, base, len);
                    let out = sl_mut_i64(*dst, base, len);
                    // ONNX Cast float→int truncates toward zero (not round-half).
                    for i in 0..len {
                        out[i] = inp[i] as i64;
                    }
                }
            }

            Thunk::CastF32ToF64 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl(*src, base, len);
                    let out = sl_mut_f64(*dst, base, len);
                    for i in 0..len {
                        out[i] = inp[i] as f64;
                    }
                }
            }

            Thunk::CastF32ToI32 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl(*src, base, len);
                    let out = sl_mut_i32(*dst, base, len);
                    // ONNX Cast float→int truncates toward zero (not round-half).
                    for i in 0..len {
                        out[i] = inp[i] as i32;
                    }
                }
            }

            Thunk::CastI64ToF32 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl_i64(*src, base, len);
                    let out = sl_mut(*dst, base, len);
                    for i in 0..len {
                        out[i] = inp[i] as f32;
                    }
                }
            }

            Thunk::CastBoolToI32 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = &arena_buf[*src..*src + len];
                    let out = sl_mut_i32(*dst, base, len);
                    for i in 0..len {
                        out[i] = i32::from(inp[i] != 0);
                    }
                }
            }

            Thunk::CastI32ToF32 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl_i32(*src, base, len);
                    let out = sl_mut(*dst, base, len);
                    for i in 0..len {
                        out[i] = inp[i] as f32;
                    }
                }
            }

            Thunk::CastI32ToI64 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl_i32(*src, base, len);
                    let out = sl_mut_i64(*dst, base, len);
                    for i in 0..len {
                        out[i] = inp[i] as i64;
                    }
                }
            }

            Thunk::CastI32ToBool { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                // src/dst are byte offsets; i32 = 4 bytes, bool = 1 byte. Copy the
                // input first to avoid aliasing the arena while writing the output.
                let bytes: Vec<u8> = arena_buf[*src..*src + len * 4].to_vec();
                for i in 0..len {
                    let v = i32::from_le_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]);
                    arena_buf[*dst + i] = u8::from(v != 0);
                }
            }

            Thunk::CastI64ToBool { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                // i64 = 8 bytes, bool = 1 byte — copy first to avoid aliasing.
                let bytes: Vec<u8> = arena_buf[*src..*src + len * 8].to_vec();
                for i in 0..len {
                    let v = i64::from_le_bytes([
                        bytes[i * 8],
                        bytes[i * 8 + 1],
                        bytes[i * 8 + 2],
                        bytes[i * 8 + 3],
                        bytes[i * 8 + 4],
                        bytes[i * 8 + 5],
                        bytes[i * 8 + 6],
                        bytes[i * 8 + 7],
                    ]);
                    arena_buf[*dst + i] = u8::from(v != 0);
                }
            }

            Thunk::CastBoolToI64 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                let bools: Vec<u8> = arena_buf[*src..*src + len].to_vec();
                for i in 0..len {
                    let v = (bools[i] != 0) as i64;
                    arena_buf[*dst + i * 8..*dst + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
            }

            Thunk::CastBoolToF32 { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = &arena_buf[*src..*src + len];
                    let out = sl_mut(*dst, base, len);
                    for i in 0..len {
                        out[i] = if inp[i] != 0 { 1.0 } else { 0.0 };
                    }
                }
            }

            Thunk::CastF32ToBool { src, dst, len } => {
                let len = *len as usize;
                if len == 0 {
                    continue;
                }
                unsafe {
                    let inp = sl(*src, base, len).to_vec();
                    let out = &mut arena_buf[*dst..*dst + len];
                    for i in 0..len {
                        out[i] = u8::from(inp[i] != 0.0);
                    }
                }
            }

            Thunk::BinaryFullF64 { .. } => exec_binary_full_f64(thunk, base),
            Thunk::BinaryFullC64 { .. } => exec_binary_full_c64(thunk, base),
            Thunk::ComplexNormSqF32 { .. } => exec_complex_norm_sq_f32(thunk, base),
            Thunk::ComplexNormSqBackwardF32 { .. } => {
                exec_complex_norm_sq_backward_f32(thunk, base)
            }
            Thunk::ConjugateC64 { .. } => exec_conjugate_c64(thunk, base),
            Thunk::ActivationC64 { .. } => exec_activation_c64(thunk, base),
            Thunk::Scan { .. } => exec_scan(thunk, base),
            Thunk::ScanBackward {
                body_vjp,
                body_init,
                body_carry_in_off,
                body_x_offs,
                body_d_output_off,
                body_dcarry_out_off,
                outer_init_off,
                outer_traj_off,
                outer_upstream_off,
                outer_xs_offs,
                outer_dinit_off,
                length,
                carry_bytes,
                save_trajectory,
                num_checkpoints,
                forward_body,
                forward_body_init,
                forward_body_carry_in_off,
                forward_body_output_off,
                forward_body_x_offs,
                carry_elem_size,
            } => {
                // Two backward paths share the same per-iteration body
                // (body_vjp run + dcarry threading). The "All" path
                // reads the carry directly from the saved trajectory
                // each step. The "Recursive checkpointing" path stores
                // only K saved checkpoints and reconstructs intermediate
                // carries via Griewank-style recursive subdivision —
                // see [`griewank_process_segment`]. Auxiliary memory
                // is `O(log(segment_size) · carry_bytes)` for the
                // recursion stack, vs the old segment-cache scheme's
                // `O(segment_size · carry_bytes)`. Total recompute work
                // grows from `O(length)` to `O(length · log)`, which
                // is the canonical Griewank trade.
                let cb = *carry_bytes as usize;
                let n_steps = *length as usize;
                let k_total = *num_checkpoints as usize;
                let is_recursive = k_total != 0 && k_total != n_steps;
                let checkpoint_t_for_k = |k: usize| -> usize {
                    ((k + 1) * n_steps)
                        .div_ceil(k_total)
                        .saturating_sub(1)
                        .min(n_steps - 1)
                };

                let mut fwd_buf: Vec<u8> = if is_recursive {
                    (**forward_body_init.as_ref().unwrap()).clone()
                } else {
                    Vec::new()
                };

                let mut dcarry: Vec<u8> = vec![0u8; cb];
                if !*save_trajectory {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(*outer_upstream_off),
                            dcarry.as_mut_ptr(),
                            cb,
                        );
                    }
                }

                let mut body_buf: Vec<u8> = (**body_init).clone();

                // Per-iteration backward action — shared between the
                // direct-trajectory (All) and Griewank (Recursive) paths.
                // Both feed the same body_vjp run with carry-at-t,
                // x_t_i, and d_output, then thread dcarry backward.
                let process_iter =
                    |t: usize, carry_in: &[u8], dcarry: &mut Vec<u8>, body_buf: &mut Vec<u8>| {
                        if *save_trajectory {
                            unsafe {
                                let up_off = *outer_upstream_off + t * cb;
                                match *carry_elem_size {
                                    4 => {
                                        let up_ptr = base.add(up_off) as *const f32;
                                        let dc_ptr = dcarry.as_mut_ptr() as *mut f32;
                                        let n_elems = cb / 4;
                                        for i in 0..n_elems {
                                            *dc_ptr.add(i) += *up_ptr.add(i);
                                        }
                                    }
                                    8 => {
                                        let up_ptr = base.add(up_off) as *const f64;
                                        let dc_ptr = dcarry.as_mut_ptr() as *mut f64;
                                        let n_elems = cb / 8;
                                        for i in 0..n_elems {
                                            *dc_ptr.add(i) += *up_ptr.add(i);
                                        }
                                    }
                                    other => panic!(
                                        "ScanBackward: unsupported carry elem size {other} \
                                     (only f32/f64 carries are supported today)"
                                    ),
                                }
                            }
                        }
                        body_buf[*body_carry_in_off..*body_carry_in_off + cb]
                            .copy_from_slice(carry_in);
                        unsafe {
                            for (i, body_x_off) in body_x_offs.iter().enumerate() {
                                let (outer_xs_off, per_step_bytes) = outer_xs_offs[i];
                                let psb = per_step_bytes as usize;
                                std::ptr::copy_nonoverlapping(
                                    base.add(outer_xs_off + t * psb),
                                    body_buf.as_mut_ptr().add(*body_x_off),
                                    psb,
                                );
                            }
                            std::ptr::copy_nonoverlapping(
                                dcarry.as_ptr(),
                                body_buf.as_mut_ptr().add(*body_d_output_off),
                                cb,
                            );
                        }
                        execute_thunks(body_vjp, body_buf);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                body_buf.as_ptr().add(*body_dcarry_out_off),
                                dcarry.as_mut_ptr(),
                                cb,
                            );
                        }
                    };

                if is_recursive {
                    // Griewank treeverse path. Process saved-checkpoint
                    // segments from highest-t to lowest-t; within each,
                    // recursive binary subdivision via
                    // `griewank_process_segment`. Auxiliary memory:
                    // O(log(seg_size) · cb) for the recursion stack
                    // (vs O(seg_size · cb) for the older segment-cache
                    // scheme); recompute work: O(seg_size · log).
                    let leaf_threshold = 4usize;
                    let fb_sched = forward_body.as_ref().unwrap();
                    let fb_init = forward_body_init.as_ref().unwrap().as_slice();
                    let mut segment_end = n_steps - 1;
                    for seg_k in (0..k_total).rev() {
                        let segment_start = if seg_k == 0 {
                            0
                        } else {
                            checkpoint_t_for_k(seg_k - 1) + 1
                        };
                        let mut anchor: Vec<u8> = vec![0u8; cb];
                        unsafe {
                            let src = if seg_k == 0 {
                                base.add(*outer_init_off)
                            } else {
                                base.add(*outer_traj_off + (seg_k - 1) * cb)
                            };
                            std::ptr::copy_nonoverlapping(src, anchor.as_mut_ptr(), cb);
                        }
                        // Closure adapter for the helper's signature
                        // (mutably re-borrows dcarry / body_buf each call).
                        let mut leaf_action = |t: usize, carry_in: &[u8]| {
                            process_iter(t, carry_in, &mut dcarry, &mut body_buf);
                        };
                        unsafe {
                            griewank_process_segment(
                                segment_start,
                                segment_end,
                                &anchor,
                                cb,
                                fb_sched,
                                fb_init,
                                *forward_body_carry_in_off,
                                *forward_body_output_off,
                                forward_body_x_offs,
                                base,
                                outer_xs_offs,
                                &mut fwd_buf,
                                leaf_threshold,
                                &mut leaf_action,
                            );
                        }
                        if seg_k == 0 {
                            break;
                        }
                        segment_end = segment_start - 1;
                    }
                } else {
                    // All-trajectory path: read each carry directly
                    // from the saved trajectory buffer.
                    let mut carry_buf: Vec<u8> = vec![0u8; cb];
                    for t in (0..n_steps).rev() {
                        unsafe {
                            let src = if t == 0 {
                                base.add(*outer_init_off)
                            } else {
                                base.add(*outer_traj_off + (t - 1) * cb)
                            };
                            std::ptr::copy_nonoverlapping(src, carry_buf.as_mut_ptr(), cb);
                        }
                        process_iter(t, &carry_buf, &mut dcarry, &mut body_buf);
                    }
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(dcarry.as_ptr(), base.add(*outer_dinit_off), cb);
                }
            }

            Thunk::ScanBackwardXs { .. } => exec_scan_backward_xs(thunk, base),
            Thunk::FusedMmBiasAct { .. } => exec_fused_mm_bias_act(thunk, base),
            Thunk::FusedResidualLN {
                x,
                res,
                bias,
                g,
                b,
                out,
                rows,
                h,
                eps,
                has_bias,
            } => {
                let (rows, h) = (*rows as usize, *h as usize);
                unsafe {
                    let zero = &zero_bias[..h];
                    let bi = if *has_bias { sl(*bias, base, h) } else { zero };
                    let x_ptr = sl(*x, base, rows * h).as_ptr() as usize;
                    let r_ptr = sl(*res, base, rows * h).as_ptr() as usize;
                    let o_ptr = sl_mut(*out, base, rows * h).as_mut_ptr() as usize;
                    let bi_ptr = bi.as_ptr() as usize;
                    let g_ptr = sl(*g, base, h).as_ptr() as usize;
                    let b_ptr = sl(*b, base, h).as_ptr() as usize;
                    let e = *eps;
                    crate::pool::par_for(rows, 4, &|off, cnt| {
                        let xs =
                            std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                        let rs =
                            std::slice::from_raw_parts((r_ptr as *const f32).add(off * h), cnt * h);
                        let os = std::slice::from_raw_parts_mut(
                            (o_ptr as *mut f32).add(off * h),
                            cnt * h,
                        );
                        let bi = std::slice::from_raw_parts(bi_ptr as *const f32, h);
                        let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                        let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                        crate::kernels::residual_bias_layer_norm(xs, rs, bi, g, b, os, cnt, h, e);
                    });
                }
            }

            Thunk::FusedResidualRmsNorm {
                x,
                res,
                bias,
                g,
                b,
                out,
                rows,
                h,
                eps,
                has_bias,
            } => {
                let (rows, h) = (*rows as usize, *h as usize);
                unsafe {
                    let zero = &zero_bias[..h];
                    let bi = if *has_bias { sl(*bias, base, h) } else { zero };
                    let x_ptr = sl(*x, base, rows * h).as_ptr() as usize;
                    let r_ptr = sl(*res, base, rows * h).as_ptr() as usize;
                    let o_ptr = sl_mut(*out, base, rows * h).as_mut_ptr() as usize;
                    let bi_ptr = bi.as_ptr() as usize;
                    let g_ptr = sl(*g, base, h).as_ptr() as usize;
                    let b_ptr = sl(*b, base, h).as_ptr() as usize;
                    let e = *eps;
                    crate::pool::par_for(rows, 4, &|off, cnt| {
                        let xs =
                            std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                        let rs =
                            std::slice::from_raw_parts((r_ptr as *const f32).add(off * h), cnt * h);
                        let os = std::slice::from_raw_parts_mut(
                            (o_ptr as *mut f32).add(off * h),
                            cnt * h,
                        );
                        let bi = std::slice::from_raw_parts(bi_ptr as *const f32, h);
                        let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                        let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                        crate::kernels::residual_bias_rms_norm(xs, rs, bi, g, b, os, cnt, h, e);
                    });
                }
            }

            Thunk::BiasAdd { .. } => exec_bias_add(thunk, base),
            Thunk::BinaryFull {
                lhs,
                rhs,
                dst,
                len,
                lhs_len,
                rhs_len,
                op,
                out_dims_bcast,
                bcast_lhs_strides,
                bcast_rhs_strides,
                elem_bytes,
            } => {
                let len = *len as usize;
                let ll = (*lhs_len as usize).max(1);
                let rl = (*rhs_len as usize).max(1);
                let eb = (*elem_bytes).max(1) as usize;
                let arena_len = arena_buf.len();
                let ll = ll.min((arena_len.saturating_sub(*lhs)) / eb);
                let rl = rl.min((arena_len.saturating_sub(*rhs)) / eb);
                let len = len.min((arena_len.saturating_sub(*dst)) / eb);
                unsafe {
                    if eb == 8 {
                        let l = sl_i64(*lhs, base, ll);
                        let r = sl_i64(*rhs, base, rl);
                        let o = sl_mut_i64(*dst, base, len);
                        // Same fused-index + hoisted-op + parallel treatment as
                        // the f32 path below (bit-exact; elements independent).
                        let rank = out_dims_bcast.len();
                        let odb = &out_dims_bcast[..];
                        let lstr = &bcast_lhs_strides[..];
                        let rstr = &bcast_rhs_strides[..];
                        let idx = |i: usize| -> (usize, usize) {
                            if rank == 0 {
                                let li = if ll == 1 { 0 } else { i % ll };
                                let ri = if rl == 1 { 0 } else { i % rl };
                                (li, ri)
                            } else {
                                let mut rem = i;
                                let (mut li, mut ri) = (0usize, 0usize);
                                for ax in (0..rank).rev() {
                                    let sz = odb[ax] as usize;
                                    let c = rem % sz;
                                    rem /= sz;
                                    li += c * lstr[ax] as usize;
                                    ri += c * rstr[ax] as usize;
                                }
                                (li, ri)
                            }
                        };
                        macro_rules! bini64 {
                            ($f:expr) => {{
                                let f = $f;
                                if len >= 8192 {
                                    use rayon::prelude::*;
                                    o.par_iter_mut().enumerate().for_each(|(i, out)| {
                                        let (li, ri) = idx(i);
                                        *out = f(l[li], r[ri]);
                                    });
                                } else {
                                    for i in 0..len {
                                        let (li, ri) = idx(i);
                                        o[i] = f(l[li], r[ri]);
                                    }
                                }
                            }};
                        }
                        match op {
                            BinaryOp::Add => bini64!(|a: i64, b: i64| a.wrapping_add(b)),
                            BinaryOp::Sub => bini64!(|a: i64, b: i64| a.wrapping_sub(b)),
                            BinaryOp::Mul => bini64!(|a: i64, b: i64| a.wrapping_mul(b)),
                            BinaryOp::Div => {
                                bini64!(|a: i64, b: i64| if b == 0 { 0 } else { a / b })
                            }
                            BinaryOp::Max => bini64!(|a: i64, b: i64| a.max(b)),
                            BinaryOp::Min => bini64!(|a: i64, b: i64| a.min(b)),
                            BinaryOp::Pow => bini64!(|a: i64, b: i64| a.pow(b.max(0) as u32)),
                        }
                    } else {
                        let l = sl(*lhs, base, ll);
                        let r = sl(*rhs, base, rl);
                        let o = sl_mut(*dst, base, len);
                        if ll == len && rl == len {
                            #[cfg(target_arch = "aarch64")]
                            if matches!(op, BinaryOp::Add | BinaryOp::Mul) {
                                use std::arch::aarch64::*;
                                let chunks = len / 4;
                                for c in 0..chunks {
                                    let off = c * 4;
                                    let vl = vld1q_f32(l.as_ptr().add(off));
                                    let vr = vld1q_f32(r.as_ptr().add(off));
                                    let res = match op {
                                        BinaryOp::Add => vaddq_f32(vl, vr),
                                        BinaryOp::Mul => vmulq_f32(vl, vr),
                                        _ => unreachable!(),
                                    };
                                    vst1q_f32(o.as_mut_ptr().add(off), res);
                                }
                                for i in (chunks * 4)..len {
                                    o[i] = match op {
                                        BinaryOp::Add => l[i] + r[i],
                                        BinaryOp::Mul => l[i] * r[i],
                                        _ => unreachable!(),
                                    };
                                }
                                continue;
                            }
                            // x86: contiguous same-shape Add/Mul/Sub — AVX2 when
                            // available, else a plain parallel loop (no N-D index walk).
                            #[cfg(target_arch = "x86_64")]
                            if matches!(op, BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub) {
                                let used = binary_contig_f32(l, r, o, *op);
                                if used {
                                    continue;
                                }
                            }
                            // Contiguous fallback for other ops / arches: skip
                            // broadcast index math entirely.
                            macro_rules! bin_contig {
                                ($f:expr) => {{
                                    let f = $f;
                                    if len >= 8192 {
                                        use rayon::prelude::*;
                                        o.par_iter_mut()
                                            .zip(l.par_iter())
                                            .zip(r.par_iter())
                                            .for_each(|((out, a), b)| *out = f(*a, *b));
                                    } else {
                                        for i in 0..len {
                                            o[i] = f(l[i], r[i]);
                                        }
                                    }
                                }};
                            }
                            match op {
                                BinaryOp::Add => bin_contig!(|a: f32, b: f32| a + b),
                                BinaryOp::Sub => bin_contig!(|a: f32, b: f32| a - b),
                                BinaryOp::Mul => bin_contig!(|a: f32, b: f32| a * b),
                                BinaryOp::Div => bin_contig!(|a: f32, b: f32| a / b),
                                BinaryOp::Max => bin_contig!(|a: f32, b: f32| a.max(b)),
                                BinaryOp::Min => bin_contig!(|a: f32, b: f32| a.min(b)),
                                BinaryOp::Pow => bin_contig!(|a: f32, b: f32| a.powf(b)),
                            }
                            continue;
                        }
                        // Trailing-row broadcast: rhs tiles every `rl` elements
                        // (bias/scale along last dim). Covers empty dims and the
                        // common `[…, D]` rhs with leading broadcast strides 0.
                        if ll == len && rl > 0 && rl < len && len.is_multiple_of(rl) {
                            let rhs_tile = out_dims_bcast.is_empty()
                                || (bcast_rhs_strides.len() == out_dims_bcast.len()
                                    && bcast_rhs_strides.last() == Some(&1)
                                    && bcast_rhs_strides
                                        [..bcast_rhs_strides.len().saturating_sub(1)]
                                        .iter()
                                        .all(|&s| s == 0));
                            if rhs_tile {
                                let used = binary_row_bcast_f32(l, r, o, *op, rl);
                                if used {
                                    continue;
                                }
                            }
                        }
                        // Broadcast / small-operand path. This scalar loop was
                        // ~88% of a supertonic subgraph's CPU time. Three fixes,
                        // all bit-exact (each output element is independent):
                        //   1) fuse the coord decomposition with the stride
                        //      dot-product — one pass, no per-call `coords` Vec,
                        //      no second loop;
                        //   2) hoist the op-match out of the per-element loop
                        //      (one branch per call, not per element);
                        //   3) parallelize across cores for large outputs.
                        let rank = out_dims_bcast.len();
                        let odb = &out_dims_bcast[..];
                        let lstr = &bcast_lhs_strides[..];
                        let rstr = &bcast_rhs_strides[..];
                        let idx = |i: usize| -> (usize, usize) {
                            if rank == 0 {
                                let li = if ll == 1 { 0 } else { i % ll };
                                let ri = if rl == 1 { 0 } else { i % rl };
                                (li, ri)
                            } else {
                                let mut rem = i;
                                let (mut li, mut ri) = (0usize, 0usize);
                                for ax in (0..rank).rev() {
                                    let sz = odb[ax] as usize;
                                    let c = rem % sz;
                                    rem /= sz;
                                    li += c * lstr[ax] as usize;
                                    ri += c * rstr[ax] as usize;
                                }
                                (li, ri)
                            }
                        };
                        macro_rules! binf32 {
                            ($f:expr) => {{
                                let f = $f;
                                if len >= 8192 {
                                    use rayon::prelude::*;
                                    o.par_iter_mut().enumerate().for_each(|(i, out)| {
                                        let (li, ri) = idx(i);
                                        *out = f(l[li], r[ri]);
                                    });
                                } else {
                                    for i in 0..len {
                                        let (li, ri) = idx(i);
                                        o[i] = f(l[li], r[ri]);
                                    }
                                }
                            }};
                        }
                        match op {
                            BinaryOp::Add => binf32!(|a: f32, b: f32| a + b),
                            BinaryOp::Sub => binf32!(|a: f32, b: f32| a - b),
                            BinaryOp::Mul => binf32!(|a: f32, b: f32| a * b),
                            BinaryOp::Div => binf32!(|a: f32, b: f32| a / b),
                            BinaryOp::Max => binf32!(|a: f32, b: f32| a.max(b)),
                            BinaryOp::Min => binf32!(|a: f32, b: f32| a.min(b)),
                            BinaryOp::Pow => binf32!(|a: f32, b: f32| a.powf(b)),
                        }
                    }
                }
            }

            Thunk::Gather { .. } => exec_gather(thunk, base),
            Thunk::Narrow {
                src,
                dst,
                outer,
                src_stride,
                dst_stride,
                inner,
                elem_bytes,
            } => {
                let (outer, ss, ds, inner, eb) = (
                    *outer as usize,
                    *src_stride as usize,
                    *dst_stride as usize,
                    *inner as usize,
                    *elem_bytes as usize,
                );
                let row_bytes = inner.saturating_mul(eb);
                let src_row_stride = ss.saturating_mul(eb);
                let dst_row_stride = ds.saturating_mul(eb);
                if trace_thunks {
                    eprintln!(
                        "[narrow] src={} dst={} outer={outer} ss={ss} ds={ds} inner={inner} eb={eb} row={row_bytes} arena={}",
                        *src,
                        *dst,
                        arena_buf.len()
                    );
                }
                if row_bytes > 0 && *src != *dst {
                    let arena_len = arena_buf.len();
                    // Parallelize independent row copies for large narrows
                    // (attention head splits, DiT reshape slices).
                    if outer >= 4
                        && row_bytes >= 64
                        && crate::pool::num_threads() > 1
                        && crate::pool::should_parallelize(outer.saturating_mul(row_bytes / 4))
                    {
                        let base_addr = base as usize;
                        let src0 = *src;
                        let dst0 = *dst;
                        crate::pool::par_for(outer, 1, &|off, cnt| {
                            for o in off..off + cnt {
                                let s_off = src0 + o * src_row_stride;
                                let d_off = dst0 + o * dst_row_stride;
                                if s_off == d_off {
                                    continue;
                                }
                                if s_off.saturating_add(row_bytes) > arena_len
                                    || d_off.saturating_add(row_bytes) > arena_len
                                {
                                    break;
                                }
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        (base_addr as *const u8).add(s_off),
                                        (base_addr as *mut u8).add(d_off),
                                        row_bytes,
                                    );
                                }
                            }
                        });
                    } else {
                        for o in 0..outer {
                            let s_off = *src + o * src_row_stride;
                            let d_off = *dst + o * dst_row_stride;
                            if s_off == d_off {
                                continue;
                            }
                            if s_off.saturating_add(row_bytes) > arena_len
                                || d_off.saturating_add(row_bytes) > arena_len
                            {
                                break;
                            }
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    base.add(s_off),
                                    base.add(d_off),
                                    row_bytes,
                                );
                            }
                        }
                    }
                }
            }

            Thunk::Copy { src, dst, len } => {
                let mut len = *len as usize;
                if *src == *dst || len == 0 {
                    continue;
                }
                let arena_len = arena_buf.len();
                let max_from_src = (arena_len.saturating_sub(*src)) / 4;
                let max_from_dst = (arena_len.saturating_sub(*dst)) / 4;
                len = len.min(max_from_src).min(max_from_dst);
                if len == 0 {
                    continue;
                }
                let byte_len = len.saturating_mul(4);
                // Parallel memcpy for huge arena moves (DiT residual buffers).
                if len >= 262_144 && crate::pool::num_threads() > 1 {
                    let base_addr = base as usize;
                    let src0 = *src;
                    let dst0 = *dst;
                    crate::pool::par_for(len, crate::pool::chunk_floor(len), &|off, cnt| {
                        let n = cnt.saturating_mul(4);
                        unsafe {
                            std::ptr::copy(
                                (base_addr as *const u8).add(src0 + off * 4),
                                (base_addr as *mut u8).add(dst0 + off * 4),
                                n,
                            );
                        }
                    });
                } else {
                    unsafe {
                        std::ptr::copy(base.add(*src), base.add(*dst), byte_len);
                    }
                }
            }

            Thunk::LayerNorm { .. } => exec_layer_norm(thunk, base),
            Thunk::GroupNorm { .. } => exec_group_norm(thunk, base),
            Thunk::BatchNormInference { .. } => exec_batch_norm_inference(thunk, base),
            Thunk::LayerNorm2d { .. } => exec_layer_norm2d(thunk, base),
            Thunk::ConvTranspose2d { .. } => exec_conv_transpose2d(thunk, base),
            Thunk::ResizeNearest2x { .. } => exec_resize_nearest2x(thunk, base),
            Thunk::AxialRope2d { .. } => exec_axial_rope2d(thunk, base),
            Thunk::RmsNorm { .. } => exec_rms_norm(thunk, base),
            Thunk::AdaLayerNorm { .. } => exec_ada_layer_norm(thunk, base),
            Thunk::GatedResidual { .. } => exec_gated_residual(thunk, base),
            Thunk::AdaLayerNormBackward { .. } => exec_ada_layer_norm_backward(thunk, base),
            Thunk::GatedResidualBackward { .. } => exec_gated_residual_backward(thunk, base),
            Thunk::Softmax { .. } => exec_softmax(thunk, base),
            Thunk::Cumsum { .. } => exec_cumsum(thunk, base),
            Thunk::Sample { .. } => exec_sample(thunk, base),
            Thunk::RngNormal {
                dst,
                len,
                mean,
                scale,
                key,
                op_seed,
            } => {
                let n = *len as usize;
                unsafe {
                    let out = sl_mut(*dst, base, n);
                    let opts = *schedule.rng.read().unwrap();
                    rlx_ir::fill_normal_like(out, *mean, *scale, opts, *key, *op_seed);
                }
            }

            Thunk::RngUniform {
                dst,
                len,
                low,
                high,
                key,
                op_seed,
            } => {
                let n = *len as usize;
                unsafe {
                    let out = sl_mut(*dst, base, n);
                    let opts = *schedule.rng.read().unwrap();
                    rlx_ir::fill_uniform_like(out, *low, *high, opts, *key, *op_seed);
                }
            }

            Thunk::GatedDeltaNet { .. } => exec_gated_delta_net(thunk, base),
            Thunk::Lstm { .. } => exec_lstm(thunk, base),
            Thunk::Gru { .. } => exec_gru(thunk, base),
            Thunk::Rnn { .. } => exec_rnn(thunk, base),
            Thunk::Mamba2 { .. } => exec_mamba2(thunk, base),
            Thunk::SelectiveScan { .. } => exec_selective_scan(thunk, base),
            Thunk::DequantMatMul { .. } => exec_dequant_mat_mul(thunk, base),
            Thunk::DequantMatMulGguf { .. } => exec_dequant_mat_mul_gguf(thunk, base),
            Thunk::DequantMatMulInt4 { .. } => exec_dequant_mat_mul_int4(thunk, base),
            Thunk::DequantMatMulFp8 { .. } => exec_dequant_mat_mul_fp8(thunk, base),
            Thunk::DequantMatMulNvfp4 { .. } => exec_dequant_mat_mul_nvfp4(thunk, base),
            Thunk::ScaledMatMul { .. } => exec_scaled_mat_mul(thunk, base),
            Thunk::ScaledQuantize { .. } => exec_scaled_quantize(thunk, base),
            Thunk::ScaledQuantScale { .. } => exec_scaled_quant_scale(thunk, base),
            Thunk::ScaledDequantize { .. } => exec_scaled_dequantize(thunk, base),
            Thunk::LoraMatMul { .. } => exec_lora_mat_mul(thunk, base),
            Thunk::Attention {
                q,
                k,
                v,
                mask,
                out,
                batch,
                seq,
                kv_seq,
                heads,
                head_dim,
                mask_kind,
                scale,
                softcap,
                q_row_stride,
                k_row_stride,
                v_row_stride,
                bhsd,
                kv_heads,
            } => {
                let (b, q_s, k_s, nh, dh) = (
                    *batch as usize,
                    *seq as usize,
                    *kv_seq as usize,
                    *heads as usize,
                    *head_dim as usize,
                );
                let nkv = (*kv_heads as usize).max(1);
                let group = (nh / nkv).max(1); // query heads per KV head (GQA/MQA)
                let hs = nh * dh;
                // For [B, H, S, D] layout each (b, h) tile is dense
                // contiguous; the qrs/krs/vrs strides are not used.
                let (qrs, krs, vrs) = if *bhsd {
                    (dh, dh, dh)
                } else {
                    (
                        *q_row_stride as usize,
                        *k_row_stride as usize,
                        *v_row_stride as usize,
                    )
                };
                let bhsd = *bhsd;
                let _ = (q_row_stride, k_row_stride, v_row_stride);
                let scale = *scale;
                let ss = q_s * k_s;
                let cfg = crate::config::RuntimeConfig::global();
                unsafe {
                    // Slice lengths cover the strided span. When Q/K/V
                    // alias the parent QKV (post-#46-fusion), the same
                    // bytes back all three slices — compiler bounds
                    // checks see the right size. For [B, H, S, D] the
                    // buffer is densely B*H*S*D elements; the row
                    // strides aren't used.
                    let q_len = if bhsd {
                        b * nh * q_s * dh
                    } else {
                        b * q_s * qrs
                    };
                    let k_len = if bhsd {
                        b * nkv * k_s * dh
                    } else {
                        b * k_s * krs
                    };
                    let v_len = if bhsd {
                        b * nkv * k_s * dh
                    } else {
                        b * k_s * vrs
                    };
                    let q_data = sl(*q, base, q_len);
                    let k_data = sl(*k, base, k_len);
                    let v_data = sl(*v, base, v_len);
                    let mask_data: &[f32] = match mask_kind {
                        rlx_ir::op::MaskKind::Custom => sl(*mask, base, b * k_s),
                        rlx_ir::op::MaskKind::Bias => sl(*mask, base, b * nh * q_s * k_s),
                        _ => &[],
                    };
                    let out_len = if bhsd {
                        b * nh * q_s * dh
                    } else {
                        b * q_s * hs
                    };
                    let out_data = sl_mut(*out, base, out_len);

                    // ── [B, H, S, D] fallback ──────────────────────
                    // The NEON / strided-BLAS specializations below
                    // are written for the [B, S, H, D] layout. When
                    // the input is head-major ([B, H, S, D] —
                    // matching rlx-cuda / rlx-rocm / rlx-tpu), bypass
                    // them and run a simple (correct but slower)
                    // scalar implementation. Production-CPU inference
                    // graphs use [B, S, H, D] so they still hit the
                    // hot path; cross-backend parity tests use
                    // [B, H, S, D] and land here.
                    if bhsd {
                        let scores = &mut sdpa_scores[..ss];
                        for bi in 0..b {
                            for hi in 0..nh {
                                let kv_hi = hi / group; // GQA/MQA: shared KV head
                                let q_head_base = bi * nh * q_s * dh + hi * q_s * dh;
                                let k_head_base = bi * nkv * k_s * dh + kv_hi * k_s * dh;
                                // Q@K^T
                                for qi in 0..q_s {
                                    let q_base = q_head_base + qi * dh;
                                    for ki in 0..k_s {
                                        let k_base = k_head_base + ki * dh;
                                        let mut dot = 0f32;
                                        for d in 0..dh {
                                            dot += q_data[q_base + d] * k_data[k_base + d];
                                        }
                                        scores[qi * k_s + ki] = dot * scale;
                                        if matches!(mask_kind, rlx_ir::op::MaskKind::Custom)
                                            && !mask_data.is_empty()
                                            && mask_data[bi * k_s + ki] < mask_thr
                                        {
                                            scores[qi * k_s + ki] = mask_neg;
                                        }
                                    }
                                }
                                if matches!(mask_kind, rlx_ir::op::MaskKind::Bias) {
                                    let off = (bi * nh + hi) * q_s * k_s;
                                    for i in 0..q_s * k_s {
                                        scores[i] += mask_data[off + i];
                                    }
                                }
                                apply_synthetic_mask(scores, q_s, k_s, *mask_kind);
                                // Gemma 2 attention logit soft-cap (post-mask, pre-softmax).
                                if *softcap > 0.0 {
                                    for s in scores.iter_mut() {
                                        *s = *softcap * (*s / *softcap).tanh();
                                    }
                                }
                                crate::kernels::neon_softmax(scores, q_s, k_s);
                                // score @ V
                                for qi in 0..q_s {
                                    let o_base = q_head_base + qi * dh;
                                    for d in 0..dh {
                                        out_data[o_base + d] = 0.0;
                                    }
                                    for ki in 0..k_s {
                                        let sc = scores[qi * k_s + ki];
                                        if sc > score_thr {
                                            let v_base = k_head_base + ki * dh;
                                            for d in 0..dh {
                                                out_data[o_base + d] += sc * v_data[v_base + d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // ── Auto-select kernel: NEON dots vs strided BLAS ───
                    // For tiny inputs (batch=1, short seq), per-head BLAS call
                    // overhead (~0.5µs × 2 calls × num_heads × num_layers)
                    // exceeds the NEON compute cost. Use direct strided NEON
                    // with zero dispatch overhead.
                    // For batch≥2: always BLAS + par_for (parallelism wins).
                    if b == 1 && q_s.max(k_s) <= cfg.sdpa_seq_threshold {
                        // ── Sequential NEON path (zero overhead) ──
                        let scores = &mut sdpa_scores[..ss];
                        #[cfg(target_arch = "aarch64")]
                        let neon_chunks = dh / 4;

                        for bi in 0..b {
                            for hi in 0..nh {
                                let kv_hi = hi / group; // GQA/MQA: shared KV head
                                // Q@K^T via strided NEON dot products
                                for qi in 0..q_s {
                                    let q_off = bi * q_s * qrs + qi * qrs + hi * dh;
                                    for ki in 0..k_s {
                                        let k_off = bi * k_s * krs + ki * krs + kv_hi * dh;
                                        #[cfg(target_arch = "aarch64")]
                                        let mut dot;
                                        #[cfg(not(target_arch = "aarch64"))]
                                        let mut dot = 0f32;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let mut acc = vdupq_n_f32(0.0);
                                            for c in 0..neon_chunks {
                                                let vq =
                                                    vld1q_f32(q_data.as_ptr().add(q_off + c * 4));
                                                let vk =
                                                    vld1q_f32(k_data.as_ptr().add(k_off + c * 4));
                                                acc = vfmaq_f32(acc, vq, vk);
                                            }
                                            dot = vaddvq_f32(acc);
                                            for d in (neon_chunks * 4)..dh {
                                                dot += q_data[q_off + d] * k_data[k_off + d];
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..dh {
                                            dot += q_data[q_off + d] * k_data[k_off + d];
                                        }
                                        scores[qi * k_s + ki] = dot * scale;
                                        // Inner-loop Custom mask check —
                                        // Causal / SlidingWindow / None
                                        // apply outside the loop below.
                                        // Skip for Bias — that mask is a
                                        // per-head additive tensor, not a
                                        // 0/1 key-padding mask.
                                        if matches!(mask_kind, rlx_ir::op::MaskKind::Custom)
                                            && !mask_data.is_empty()
                                            && mask_data[bi * k_s + ki] < mask_thr
                                        {
                                            scores[qi * k_s + ki] = mask_neg;
                                        }
                                    }
                                }

                                if matches!(mask_kind, rlx_ir::op::MaskKind::Bias) {
                                    let off = (bi * nh + hi) * q_s * k_s;
                                    for i in 0..q_s * k_s {
                                        scores[i] += mask_data[off + i];
                                    }
                                }
                                apply_synthetic_mask(scores, q_s, k_s, *mask_kind);
                                crate::kernels::neon_softmax(scores, q_s, k_s);

                                // Score@V via strided NEON accumulation (zero-copy)
                                for qi in 0..q_s {
                                    let o_off = bi * q_s * hs + qi * hs + hi * dh;
                                    // Zero output for this head position
                                    for d in 0..dh {
                                        out_data[o_off + d] = 0.0;
                                    }
                                    for ki in 0..k_s {
                                        let sc = scores[qi * k_s + ki];
                                        if sc > score_thr {
                                            let v_off = bi * k_s * vrs + ki * vrs + kv_hi * dh;
                                            #[cfg(target_arch = "aarch64")]
                                            {
                                                use std::arch::aarch64::*;
                                                let vsc = vdupq_n_f32(sc);
                                                for c in 0..neon_chunks {
                                                    let off = c * 4;
                                                    let vo = vld1q_f32(
                                                        out_data.as_ptr().add(o_off + off),
                                                    );
                                                    let vv =
                                                        vld1q_f32(v_data.as_ptr().add(v_off + off));
                                                    vst1q_f32(
                                                        out_data.as_mut_ptr().add(o_off + off),
                                                        vfmaq_f32(vo, vsc, vv),
                                                    );
                                                }
                                            }
                                            #[cfg(not(target_arch = "aarch64"))]
                                            for d in 0..dh {
                                                out_data[o_off + d] += sc * v_data[v_off + d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // ── Parallel strided BLAS path (high throughput) ──
                        let total_work = b * nh;
                        let q_addr = q_data.as_ptr() as usize;
                        let k_addr = k_data.as_ptr() as usize;
                        let v_addr = v_data.as_ptr() as usize;
                        let m_addr = mask_data.as_ptr() as usize;
                        let o_addr = out_data.as_mut_ptr() as usize;
                        let sc_addr = sdpa_scores.as_mut_ptr() as usize;

                        crate::pool::par_for(total_work, 1, &|off, cnt| {
                            for idx in off..off + cnt {
                                let bi = idx / nh;
                                let hi = idx % nh;
                                let kv_hi = hi / group; // GQA/MQA: shared KV head

                                let q_start = (q_addr as *const f32).add(bi * q_s * qrs + hi * dh);
                                let k_start =
                                    (k_addr as *const f32).add(bi * k_s * krs + kv_hi * dh);
                                let v_start =
                                    (v_addr as *const f32).add(bi * k_s * vrs + kv_hi * dh);
                                let o_start = (o_addr as *mut f32).add(bi * q_s * hs + hi * dh);
                                let sc = std::slice::from_raw_parts_mut(
                                    (sc_addr as *mut f32).add(idx * ss),
                                    ss,
                                );

                                // LDA = qrs, LDB = krs (parent row strides
                                // when fused; hs otherwise).
                                crate::blas::sgemm_general(
                                    q_start,
                                    k_start,
                                    sc.as_mut_ptr(),
                                    q_s,
                                    k_s,
                                    dh,
                                    scale,
                                    0.0,
                                    qrs,
                                    krs,
                                    k_s,
                                    false,
                                    true,
                                );

                                match mask_kind {
                                    rlx_ir::op::MaskKind::Custom => {
                                        let mask_bi = std::slice::from_raw_parts(
                                            (m_addr as *const f32).add(bi * k_s),
                                            k_s,
                                        );
                                        for ki in 0..k_s {
                                            if mask_bi[ki] < mask_thr {
                                                for qi in 0..q_s {
                                                    sc[qi * k_s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Bias => {
                                        // Per-head additive bias slice.
                                        let bias = std::slice::from_raw_parts(
                                            (m_addr as *const f32).add((bi * nh + hi) * q_s * k_s),
                                            q_s * k_s,
                                        );
                                        for i in 0..q_s * k_s {
                                            sc[i] += bias[i];
                                        }
                                    }
                                    _ => apply_synthetic_mask(sc, q_s, k_s, *mask_kind),
                                }

                                crate::kernels::neon_softmax(sc, q_s, k_s);

                                // LDB = vrs (parent row stride when
                                // fused; hs otherwise). LDC stays hs —
                                // output is its own contiguous buffer.
                                crate::blas::sgemm_general(
                                    sc.as_ptr(),
                                    v_start,
                                    o_start,
                                    q_s,
                                    dh,
                                    k_s,
                                    1.0,
                                    0.0,
                                    k_s,
                                    vrs,
                                    hs,
                                    false,
                                    false,
                                );
                            }
                        });
                    }
                }
            }

            Thunk::AttentionBackward { .. } => exec_attention_backward(thunk, base),
            Thunk::ActivationInPlace { .. } => exec_activation_in_place(thunk, base),
            Thunk::FusedAttnBlock {
                hidden,
                qkv_w,
                out_w,
                mask,
                mask_kind,
                out,
                qkv_b,
                out_b,
                cos,
                sin,
                cos_len,
                batch,
                seq,
                hs,
                nh,
                dh,
                has_bias,
                has_rope,
                interleaved,
            } => {
                let (b, s) = (*batch as usize, *seq as usize);
                let (h, n_h, d_h) = (*hs as usize, *nh as usize, *dh as usize);
                let interleaved = *interleaved;
                let m = b * s;
                let scale = (d_h as f32).powf(-0.5);
                let half = d_h / 2;
                // Only `Custom` consumes the per-key padding buffer; `Causal` /
                // `SlidingWindow` are synthesized from (qi, ki) below, and have
                // no mask buffer (so reading one would touch unrelated arena
                // bytes). q_seq == kv_seq here (guaranteed at fusion time), so
                // the absolute query position is just `qi`.
                let use_custom_mask = matches!(mask_kind, rlx_ir::op::MaskKind::Custom);
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let wq = sl(*qkv_w, base, h * 3 * h);
                    let wo = sl(*out_w, base, h * h);
                    let mk = if use_custom_mask {
                        sl(*mask, base, b * s)
                    } else {
                        &[]
                    };
                    let dst = sl_mut(*out, base, m * h);

                    // Stack-allocated intermediates — all fit in L1 cache for small batch
                    let mut qkv = vec![0f32; m * 3 * h];
                    let mut attn_out = vec![0f32; m * h];
                    let mut scores_buf = vec![0f32; s * s]; // one head at a time

                    // 1. QKV projection: [m, h] @ [h, 3h] → [m, 3h]
                    crate::blas::sgemm(inp, wq, &mut qkv, m, h, 3 * h);
                    if *has_bias {
                        let bias = sl(*qkv_b, base, 3 * h);
                        crate::blas::bias_add(&mut qkv, bias, m, 3 * h);
                    }

                    // 2. Multi-head SDPA (Q/K/V are views into qkv at offsets 0, h, 2h)
                    //    Process heads sequentially with inline RoPE — zero copy.
                    #[cfg(target_arch = "aarch64")]
                    let neon_chunks = d_h / 4;
                    #[cfg(target_arch = "aarch64")]
                    let _rope_chunks = half / 4;

                    for bi in 0..b {
                        for hi in 0..n_h {
                            // For each (query_pos, key_pos): compute Q@K^T with inline RoPE
                            for qi in 0..s {
                                let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                for ki in 0..s {
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    let mut dot = 0f32;

                                    if *has_rope {
                                        // Apply RoPE inline during dot product
                                        let q_cos = qi * half;
                                        let k_cos = ki * half;
                                        let cos_tab = sl(*cos, base, *cos_len as usize);
                                        let sin_tab = sl(*sin, base, *cos_len as usize);
                                        // Rotate per pair, then dot. The q·k sum
                                        // is layout-independent, so only the pair
                                        // element offsets differ by style:
                                        //   NeoX:  (i, i+half)   GPT-J: (2i, 2i+1)
                                        // angle index is the pair index `i` for both.
                                        for i in 0..half {
                                            let (qo1, qo2, ko1, ko2) = if interleaved {
                                                (2 * i, 2 * i + 1, 2 * i, 2 * i + 1)
                                            } else {
                                                (i, half + i, i, half + i)
                                            };
                                            let q1 = qkv[q_base + qo1];
                                            let q2 = qkv[q_base + qo2];
                                            let k1 = qkv[k_base + ko1];
                                            let k2 = qkv[k_base + ko2];
                                            let c_q = cos_tab[q_cos + i];
                                            let s_q = sin_tab[q_cos + i];
                                            let c_k = cos_tab[k_cos + i];
                                            let s_k = sin_tab[k_cos + i];
                                            let qr1 = q1 * c_q - q2 * s_q;
                                            let kr1 = k1 * c_k - k2 * s_k;
                                            let qr2 = q2 * c_q + q1 * s_q;
                                            let kr2 = k2 * c_k + k1 * s_k;
                                            dot += qr1 * kr1 + qr2 * kr2;
                                        }
                                    } else {
                                        // Standard dot product
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let mut acc = vdupq_n_f32(0.0);
                                            for c in 0..neon_chunks {
                                                let vq =
                                                    vld1q_f32(qkv.as_ptr().add(q_base + c * 4));
                                                let vk =
                                                    vld1q_f32(qkv.as_ptr().add(k_base + c * 4));
                                                acc = vfmaq_f32(acc, vq, vk);
                                            }
                                            dot = vaddvq_f32(acc);
                                            for d in (neon_chunks * 4)..d_h {
                                                dot += qkv[q_base + d] * qkv[k_base + d];
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            dot += qkv[q_base + d] * qkv[k_base + d];
                                        }
                                    }

                                    scores_buf[qi * s + ki] = dot * scale;
                                    // Synthesized position masks (q_offset == 0):
                                    //   Causal         → mask future keys ki > qi
                                    //   SlidingWindow  → also mask ki + w < qi
                                    let pos_masked = match mask_kind {
                                        rlx_ir::op::MaskKind::Causal => ki > qi,
                                        rlx_ir::op::MaskKind::SlidingWindow(w) => {
                                            ki > qi || ki + *w < qi
                                        }
                                        _ => false,
                                    };
                                    if pos_masked || (use_custom_mask && mk[bi * s + ki] < mask_thr)
                                    {
                                        scores_buf[qi * s + ki] = mask_neg;
                                    }
                                }
                            }

                            // Softmax
                            crate::kernels::neon_softmax(&mut scores_buf[..s * s], s, s);

                            // Score @ V accumulation (V at offset 2h in QKV)
                            for qi in 0..s {
                                let o_base = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn_out[o_base + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let sc = scores_buf[qi * s + ki];
                                    if sc > score_thr {
                                        let v_base = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vsc = vdupq_n_f32(sc);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                let vo =
                                                    vld1q_f32(attn_out.as_ptr().add(o_base + off));
                                                let vv = vld1q_f32(qkv.as_ptr().add(v_base + off));
                                                vst1q_f32(
                                                    attn_out.as_mut_ptr().add(o_base + off),
                                                    vfmaq_f32(vo, vsc, vv),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn_out[o_base + d] += sc * qkv[v_base + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // 3. Output projection: [m, h] @ [h, h] → dst
                    crate::blas::sgemm(&attn_out, wo, dst, m, h, h);
                    if *has_bias {
                        let bias = sl(*out_b, base, h);
                        crate::blas::bias_add(dst, bias, m, h);
                    }
                }
            }

            Thunk::Rope { .. } => exec_rope(thunk, base),
            Thunk::FusedBertLayer {
                hidden,
                qkv_w,
                qkv_b,
                out_w,
                out_b,
                mask,
                ln1_g,
                ln1_b,
                eps1,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                ln2_g,
                ln2_b,
                eps2,
                out,
                batch,
                seq,
                hs,
                nh,
                dh,
                int_dim,
            } => {
                let (b, s, h, n_h, d_h) = (
                    *batch as usize,
                    *seq as usize,
                    *hs as usize,
                    *nh as usize,
                    *dh as usize,
                );
                let m = b * s;
                let id = *int_dim as usize;
                let scale = (d_h as f32).powf(-0.5);
                let _half = d_h / 2;
                #[cfg(target_arch = "aarch64")]
                let neon_chunks = d_h / 4;
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let dst = sl_mut(*out, base, m * h);
                    let mk = sl(*mask, base, b * s);

                    // Pre-allocated buffers (zero malloc per layer — allocated once before thunk loop)
                    let qkv = std::slice::from_raw_parts_mut(fl_qkv.as_mut_ptr(), m * 3 * h);
                    let attn = std::slice::from_raw_parts_mut(fl_attn.as_mut_ptr(), m * h);
                    let res = std::slice::from_raw_parts_mut(fl_res.as_mut_ptr(), m * h);
                    let normed = std::slice::from_raw_parts_mut(fl_normed.as_mut_ptr(), m * h);
                    let ffn = std::slice::from_raw_parts_mut(fl_ffn.as_mut_ptr(), m * id);
                    let sc = std::slice::from_raw_parts_mut(fl_sc.as_mut_ptr(), s * s);

                    // QKV (parallelized across cores — multiple AMX coprocessors)
                    crate::blas::par_sgemm_bias(
                        inp,
                        sl(*qkv_w, base, h * 3 * h),
                        sl(*qkv_b, base, 3 * h),
                        qkv,
                        m,
                        h,
                        3 * h,
                    );

                    // SDPA per head (sequential NEON, inline — zero overhead)
                    for bi in 0..b {
                        for hi in 0..n_h {
                            for qi in 0..s {
                                for ki in 0..s {
                                    let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    #[cfg(target_arch = "aarch64")]
                                    let dot;
                                    #[cfg(not(target_arch = "aarch64"))]
                                    let mut dot = 0f32;
                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        use std::arch::aarch64::*;
                                        let mut acc = vdupq_n_f32(0.0);
                                        for c in 0..neon_chunks {
                                            acc = vfmaq_f32(
                                                acc,
                                                vld1q_f32(qkv.as_ptr().add(q_base + c * 4)),
                                                vld1q_f32(qkv.as_ptr().add(k_base + c * 4)),
                                            );
                                        }
                                        dot = vaddvq_f32(acc);
                                    }
                                    #[cfg(not(target_arch = "aarch64"))]
                                    for d in 0..d_h {
                                        dot += qkv[q_base + d] * qkv[k_base + d];
                                    }
                                    sc[qi * s + ki] = dot * scale;
                                    if mk[bi * s + ki] < mask_thr {
                                        sc[qi * s + ki] = mask_neg;
                                    }
                                }
                            }
                            crate::kernels::neon_softmax(&mut sc[..s * s], s, s);
                            for qi in 0..s {
                                let o = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn[o + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let w = sc[qi * s + ki];
                                    if w > score_thr {
                                        let v = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vw = vdupq_n_f32(w);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                vst1q_f32(
                                                    attn.as_mut_ptr().add(o + off),
                                                    vfmaq_f32(
                                                        vld1q_f32(attn.as_ptr().add(o + off)),
                                                        vw,
                                                        vld1q_f32(qkv.as_ptr().add(v + off)),
                                                    ),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn[o + d] += w * qkv[v + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Out proj (sgemm + bias fused) + residual add with NEON
                    crate::blas::sgemm_bias(
                        attn,
                        sl(*out_w, base, h * h),
                        sl(*out_b, base, h),
                        res,
                        m,
                        h,
                        h,
                    );
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        let chunks_h = (m * h) / 4;
                        for c in 0..chunks_h {
                            let off = c * 4;
                            vst1q_f32(
                                res.as_mut_ptr().add(off),
                                vaddq_f32(
                                    vld1q_f32(res.as_ptr().add(off)),
                                    vld1q_f32(inp.as_ptr().add(off)),
                                ),
                            );
                        }
                        for i in (chunks_h * 4)..(m * h) {
                            res[i] += inp[i];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..m * h {
                        res[i] += inp[i];
                    }

                    // LN1 (fused residual already done above — just normalize)
                    let g1 = sl(*ln1_g, base, h);
                    let b1 = sl(*ln1_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g1,
                            b1,
                            &mut normed[r * h..(r + 1) * h],
                            h,
                            *eps1,
                        );
                    }

                    // FFN: fc1 (parallel across cores) + GELU
                    crate::blas::par_sgemm_bias(
                        normed,
                        sl(*fc1_w, base, h * id),
                        sl(*fc1_b, base, id),
                        ffn,
                        m,
                        h,
                        id,
                    );
                    crate::kernels::par_gelu_inplace(ffn);

                    // fc2 + bias (parallel across cores) + residual with NEON
                    crate::blas::par_sgemm_bias(
                        ffn,
                        sl(*fc2_w, base, id * h),
                        sl(*fc2_b, base, h),
                        res,
                        m,
                        id,
                        h,
                    );
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        let chunks_h = (m * h) / 4;
                        for c in 0..chunks_h {
                            let off = c * 4;
                            vst1q_f32(
                                res.as_mut_ptr().add(off),
                                vaddq_f32(
                                    vld1q_f32(res.as_ptr().add(off)),
                                    vld1q_f32(normed.as_ptr().add(off)),
                                ),
                            );
                        }
                        for i in (chunks_h * 4)..(m * h) {
                            res[i] += normed[i];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..m * h {
                        res[i] += normed[i];
                    }

                    // LN2 → output
                    let g2 = sl(*ln2_g, base, h);
                    let b2 = sl(*ln2_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g2,
                            b2,
                            &mut dst[r * h..(r + 1) * h],
                            h,
                            *eps2,
                        );
                    }
                }
            }

            Thunk::FusedNomicLayer {
                hidden,
                qkv_w,
                out_w,
                mask,
                cos,
                sin,
                cos_len,
                ln1_g,
                ln1_b,
                eps1,
                fc11_w,
                fc12_w: _,
                fc2_w,
                ln2_g,
                ln2_b,
                eps2,
                out,
                batch,
                seq,
                hs,
                nh,
                dh,
                int_dim,
                interleaved,
            } => {
                let interleaved = *interleaved;
                let (b, s, h, n_h, d_h) = (
                    *batch as usize,
                    *seq as usize,
                    *hs as usize,
                    *nh as usize,
                    *dh as usize,
                );
                let m = b * s;
                let id = *int_dim as usize;
                let scale = (d_h as f32).powf(-0.5);
                let half_dh = d_h / 2;
                #[cfg(target_arch = "aarch64")]
                let neon_chunks = d_h / 4;
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let dst = sl_mut(*out, base, m * h);
                    let mk = sl(*mask, base, b * s);
                    let cos_tab = sl(*cos, base, *cos_len as usize);
                    let sin_tab = sl(*sin, base, *cos_len as usize);
                    // fc11_w is the fused [h, 2*int_dim] weight (fc11 || fc12 concatenated)
                    let fused_fc_w = sl(*fc11_w, base, h * 2 * id);

                    let mut qkv = vec![0f32; m * 3 * h];
                    let mut attn = vec![0f32; m * h];
                    let mut res = vec![0f32; m * h];
                    let mut normed = vec![0f32; m * h];
                    let mut ffn_concat = vec![0f32; m * 2 * id]; // fc11||fc12 output
                    let mut sc = vec![0f32; s * s];

                    // QKV (no bias)
                    crate::blas::sgemm(inp, sl(*qkv_w, base, h * 3 * h), &mut qkv, m, h, 3 * h);

                    // SDPA with inline RoPE
                    for bi in 0..b {
                        for hi in 0..n_h {
                            for qi in 0..s {
                                for ki in 0..s {
                                    let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    let mut dot = 0f32;
                                    for i in 0..half_dh {
                                        // NeoX pairs (i, i+half); GPT-J pairs (2i, 2i+1).
                                        let (o1, o2) = if interleaved {
                                            (2 * i, 2 * i + 1)
                                        } else {
                                            (i, half_dh + i)
                                        };
                                        let q1 = qkv[q_base + o1];
                                        let q2 = qkv[q_base + o2];
                                        let k1 = qkv[k_base + o1];
                                        let k2 = qkv[k_base + o2];
                                        let cq = cos_tab[qi * half_dh + i];
                                        let sq = sin_tab[qi * half_dh + i];
                                        let ck = cos_tab[ki * half_dh + i];
                                        let sk = sin_tab[ki * half_dh + i];
                                        dot += (q1 * cq - q2 * sq) * (k1 * ck - k2 * sk)
                                            + (q2 * cq + q1 * sq) * (k2 * ck + k1 * sk);
                                    }
                                    sc[qi * s + ki] = dot * scale;
                                    if mk[bi * s + ki] < mask_thr {
                                        sc[qi * s + ki] = mask_neg;
                                    }
                                }
                            }
                            crate::kernels::neon_softmax(&mut sc[..s * s], s, s);
                            for qi in 0..s {
                                let o = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn[o + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let w = sc[qi * s + ki];
                                    if w > score_thr {
                                        let v = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vw = vdupq_n_f32(w);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                vst1q_f32(
                                                    attn.as_mut_ptr().add(o + off),
                                                    vfmaq_f32(
                                                        vld1q_f32(attn.as_ptr().add(o + off)),
                                                        vw,
                                                        vld1q_f32(qkv.as_ptr().add(v + off)),
                                                    ),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn[o + d] += w * qkv[v + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Out proj (no bias) + residual
                    crate::blas::sgemm(&attn, sl(*out_w, base, h * h), &mut res, m, h, h);
                    for i in 0..m * h {
                        res[i] += inp[i];
                    }

                    // LN1
                    let g1 = sl(*ln1_g, base, h);
                    let b1 = sl(*ln1_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g1,
                            b1,
                            &mut normed[r * h..(r + 1) * h],
                            h,
                            *eps1,
                        );
                    }

                    // SwiGLU: fused fc11+fc12 sgemm, then split, silu, mul
                    crate::blas::sgemm(&normed, fused_fc_w, &mut ffn_concat, m, h, 2 * id);
                    // Split: first id cols = fc11 (up), second id cols = fc12 (gate)
                    // SiLU on gate, then multiply up * gate → store in up region
                    for row in 0..m {
                        let bo = row * 2 * id;
                        // SiLU in-place on gate portion
                        for j in 0..id {
                            let x = ffn_concat[bo + id + j];
                            ffn_concat[bo + id + j] = x / (1.0 + (-x).exp());
                        }
                        // Multiply: up[j] *= gate[j]
                        for j in 0..id {
                            ffn_concat[bo + j] *= ffn_concat[bo + id + j];
                        }
                    }

                    // fc2 (no bias) + residual. The up*silu(gate) product lives in
                    // the FIRST `id` cols of each 2*id-wide ffn_concat row; gather
                    // it contiguous and run the SAME `sgemm` dispatch the unfused
                    // path uses (a strided `sgemm_general` here would force the
                    // BLAS/scalar path and diverge from the unfused NEON sgemm).
                    let mut swiglu_contig = vec![0f32; m * id];
                    for row in 0..m {
                        let bo = row * 2 * id;
                        swiglu_contig[row * id..(row + 1) * id]
                            .copy_from_slice(&ffn_concat[bo..bo + id]);
                    }
                    crate::blas::sgemm(
                        &swiglu_contig,
                        sl(*fc2_w, base, id * h),
                        &mut res,
                        m,
                        id,
                        h,
                    );
                    for i in 0..m * h {
                        res[i] += normed[i];
                    }

                    // LN2 → output
                    let g2 = sl(*ln2_g, base, h);
                    let b2 = sl(*ln2_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g2,
                            b2,
                            &mut dst[r * h..(r + 1) * h],
                            h,
                            *eps2,
                        );
                    }
                }
            }

            Thunk::FusedSwiGLU { .. } => exec_fused_swi_g_l_u(thunk, base),
            Thunk::Concat { .. } => exec_concat(thunk, base),
            Thunk::ConcatF64 { .. } => exec_concat_f64(thunk, base),
            Thunk::Compare {
                lhs,
                rhs,
                dst,
                len,
                op,
                inputs_i64,
                inputs_elem_bytes,
                dst_elem_bytes,
                lhs_scalar,
                rhs_scalar,
            } => {
                let len = *len as usize;
                let arena_len = arena_buf.len();
                let elem = (*inputs_elem_bytes).max(1) as usize;
                let dst_eb = (*dst_elem_bytes).max(1) as usize;
                let l_n = if *lhs_scalar { 1 } else { len };
                let r_n = if *rhs_scalar { 1 } else { len };
                let max_l = (arena_len.saturating_sub(*lhs)) / elem;
                let max_r = (arena_len.saturating_sub(*rhs)) / elem;
                let max_d = (arena_len.saturating_sub(*dst)) / dst_eb;
                // Keep full `len` when broadcasting scalars — only the
                // non-scalar operands (and dst) may shrink the loop.
                let mut len = len.min(max_d);
                if *lhs_scalar {
                    if max_l < 1 {
                        len = 0;
                    }
                } else {
                    len = len.min(max_l);
                }
                if *rhs_scalar {
                    if max_r < 1 {
                        len = 0;
                    }
                } else {
                    len = len.min(max_r);
                }
                if trace_thunks && len > 0 {
                    eprintln!(
                        "[compare] len={len} lhs={} rhs={} dst={} ls={} rs={}",
                        *lhs, *rhs, *dst, *lhs_scalar, *rhs_scalar
                    );
                }
                if elem == 1 {
                    let l = arena_buf[*lhs..*lhs + l_n.min(max_l).max(1)].to_vec();
                    let r = arena_buf[*rhs..*rhs + r_n.min(max_r).max(1)].to_vec();
                    for i in 0..len {
                        let li = if *lhs_scalar { 0 } else { i };
                        let ri = if *rhs_scalar { 0 } else { i };
                        let v = match op {
                            CmpOp::Eq => l[li] == r[ri],
                            CmpOp::Ne => l[li] != r[ri],
                            CmpOp::Lt => l[li] < r[ri],
                            CmpOp::Le => l[li] <= r[ri],
                            CmpOp::Gt => l[li] > r[ri],
                            CmpOp::Ge => l[li] >= r[ri],
                        };
                        if *dst_elem_bytes == 1 {
                            arena_buf[*dst + i] = u8::from(v);
                        } else {
                            unsafe {
                                let o = sl_mut(*dst, base, len);
                                o[i] = if v { 1.0 } else { 0.0 };
                            }
                        }
                    }
                } else if *inputs_i64 != 0 {
                    unsafe {
                        let l = sl_i64(*lhs, base, l_n.min(max_l).max(1));
                        let r = sl_i64(*rhs, base, r_n.min(max_r).max(1));
                        for i in 0..len {
                            let li = if *lhs_scalar { 0 } else { i };
                            let ri = if *rhs_scalar { 0 } else { i };
                            let v = match op {
                                CmpOp::Eq => l[li] == r[ri],
                                CmpOp::Ne => l[li] != r[ri],
                                CmpOp::Lt => l[li] < r[ri],
                                CmpOp::Le => l[li] <= r[ri],
                                CmpOp::Gt => l[li] > r[ri],
                                CmpOp::Ge => l[li] >= r[ri],
                            };
                            if *dst_elem_bytes == 1 {
                                arena_buf[*dst + i] = u8::from(v);
                            } else {
                                let o = sl_mut(*dst, base, len);
                                o[i] = if v { 1.0 } else { 0.0 };
                            }
                        }
                    }
                } else {
                    unsafe {
                        let l = sl(*lhs, base, l_n.min(max_l).max(1));
                        let r = sl(*rhs, base, r_n.min(max_r).max(1));
                        for i in 0..len {
                            let li = if *lhs_scalar { 0 } else { i };
                            let ri = if *rhs_scalar { 0 } else { i };
                            let v = match op {
                                CmpOp::Eq => l[li] == r[ri],
                                CmpOp::Ne => l[li] != r[ri],
                                CmpOp::Lt => l[li] < r[ri],
                                CmpOp::Le => l[li] <= r[ri],
                                CmpOp::Gt => l[li] > r[ri],
                                CmpOp::Ge => l[li] >= r[ri],
                            };
                            if *dst_elem_bytes == 1 {
                                arena_buf[*dst + i] = u8::from(v);
                            } else {
                                let o = sl_mut(*dst, base, len);
                                o[i] = if v { 1.0 } else { 0.0 };
                            }
                        }
                    }
                }
            }

            Thunk::Where {
                cond,
                on_true,
                on_false,
                dst,
                len,
                elem_bytes,
                cond_elem_bytes,
                cond_scalar,
                true_scalar,
                false_scalar,
            } => {
                let len = *len as usize;
                let eb = *elem_bytes as usize;
                let cond_eb = (*cond_elem_bytes).max(1) as usize;
                let arena_len = arena_buf.len();
                let c_n = if *cond_scalar { 1 } else { len };
                let t_n = if *true_scalar { 1 } else { len };
                let f_n = if *false_scalar { 1 } else { len };
                let max_c = (arena_len.saturating_sub(*cond)) / cond_eb;
                let max_t = (arena_len.saturating_sub(*on_true)) / eb;
                let max_f = (arena_len.saturating_sub(*on_false)) / eb;
                let max_d = (arena_len.saturating_sub(*dst)) / eb;
                let mut len = len.min(max_d);
                if *cond_scalar {
                    if max_c < 1 {
                        len = 0;
                    }
                } else {
                    len = len.min(max_c);
                }
                if *true_scalar {
                    if max_t < 1 {
                        len = 0;
                    }
                } else {
                    len = len.min(max_t);
                }
                if *false_scalar {
                    if max_f < 1 {
                        len = 0;
                    }
                } else {
                    len = len.min(max_f);
                }
                unsafe {
                    if *elem_bytes == 8 {
                        let t = sl_i64(*on_true, base, t_n.min(max_t).max(1));
                        let e = sl_i64(*on_false, base, f_n.min(max_f).max(1));
                        let o = sl_mut_i64(*dst, base, len);
                        if *cond_elem_bytes == 1 {
                            let c = &arena_buf[*cond..*cond + c_n.min(max_c).max(1)];
                            for i in 0..len {
                                let ci = if *cond_scalar { 0 } else { i };
                                let ti = if *true_scalar { 0 } else { i };
                                let ei = if *false_scalar { 0 } else { i };
                                o[i] = if c[ci] != 0 { t[ti] } else { e[ei] };
                            }
                        } else if *cond_elem_bytes == 4 {
                            // Bool-as-f32 masks (f32-uniform arena).
                            let c = sl(*cond, base, c_n.min(max_c).max(1));
                            for i in 0..len {
                                let ci = if *cond_scalar { 0 } else { i };
                                let ti = if *true_scalar { 0 } else { i };
                                let ei = if *false_scalar { 0 } else { i };
                                o[i] = if c[ci] != 0.0 { t[ti] } else { e[ei] };
                            }
                        } else {
                            let c = sl_i64(*cond, base, c_n.min(max_c).max(1));
                            for i in 0..len {
                                let ci = if *cond_scalar { 0 } else { i };
                                let ti = if *true_scalar { 0 } else { i };
                                let ei = if *false_scalar { 0 } else { i };
                                o[i] = if c[ci] != 0 { t[ti] } else { e[ei] };
                            }
                        }
                    } else if *cond_elem_bytes == 1 {
                        let c = &arena_buf[*cond..*cond + c_n.min(max_c).max(1)];
                        let t = sl(*on_true, base, t_n.min(max_t).max(1));
                        let e = sl(*on_false, base, f_n.min(max_f).max(1));
                        let o = sl_mut(*dst, base, len);
                        for i in 0..len {
                            let ci = if *cond_scalar { 0 } else { i };
                            let ti = if *true_scalar { 0 } else { i };
                            let ei = if *false_scalar { 0 } else { i };
                            o[i] = if c[ci] != 0 { t[ti] } else { e[ei] };
                        }
                    } else {
                        let c = sl(*cond, base, c_n.min(max_c).max(1));
                        let t = sl(*on_true, base, t_n.min(max_t).max(1));
                        let e = sl(*on_false, base, f_n.min(max_f).max(1));
                        let o = sl_mut(*dst, base, len);
                        for i in 0..len {
                            let ci = if *cond_scalar { 0 } else { i };
                            let ti = if *true_scalar { 0 } else { i };
                            let ei = if *false_scalar { 0 } else { i };
                            o[i] = if c[ci] != 0.0 { t[ti] } else { e[ei] };
                        }
                    }
                }
            }

            Thunk::Fma {
                a,
                b,
                c,
                dst,
                len,
                elem_bytes,
            } => {
                let len = *len as usize;
                let eb = (*elem_bytes).max(1) as usize;
                let arena_len = arena_buf.len();
                let len = len
                    .min(arena_len.saturating_sub(*a) / eb)
                    .min(arena_len.saturating_sub(*b) / eb)
                    .min(arena_len.saturating_sub(*c) / eb)
                    .min(arena_len.saturating_sub(*dst) / eb);
                unsafe {
                    if *elem_bytes == 8 {
                        let av = sl_f64(*a, base, len);
                        let bv = sl_f64(*b, base, len);
                        let cv = sl_f64(*c, base, len);
                        let o = sl_mut_f64(*dst, base, len);
                        for i in 0..len {
                            o[i] = av[i].mul_add(bv[i], cv[i]);
                        }
                    } else {
                        let av = sl(*a, base, len);
                        let bv = sl(*b, base, len);
                        let cv = sl(*c, base, len);
                        let o = sl_mut(*dst, base, len);
                        for i in 0..len {
                            o[i] = av[i].mul_add(bv[i], cv[i]);
                        }
                    }
                }
            }

            Thunk::ScatterAdd { .. } => exec_scatter_add(thunk, base),
            Thunk::ScatterNd { .. } => exec_scatter_nd(thunk, base),
            Thunk::ScatterElements { .. } => exec_scatter_elements(thunk, base),
            Thunk::GatherNd { .. } => exec_gather_nd(thunk, base),
            Thunk::GatherElements { .. } => exec_gather_elements(thunk, base),
            Thunk::GroupedMatMul {
                input,
                weight,
                expert_idx,
                dst,
                m,
                k_dim,
                n,
                num_experts,
            } => {
                let m = *m as usize;
                let k_dim = *k_dim as usize;
                let n = *n as usize;
                let num_experts = *num_experts as usize;
                unsafe {
                    let inp = sl(*input, base, m * k_dim);
                    let wt = sl(*weight, base, num_experts * k_dim * n);
                    let ids = sl(*expert_idx, base, m);
                    let out = sl_mut(*dst, base, m * n);

                    // Counting-sort tokens by their assigned expert.
                    // counts[e] = how many tokens routed to expert e.
                    let mut counts = vec![0usize; num_experts];
                    for i in 0..m {
                        let e = ids[i] as usize;
                        debug_assert!(
                            e < num_experts,
                            "expert_idx out of range: {e} >= {num_experts}"
                        );
                        counts[e] += 1;
                    }
                    // Cumulative offsets into the packed buffer.
                    let mut offsets = vec![0usize; num_experts + 1];
                    for e in 0..num_experts {
                        offsets[e + 1] = offsets[e] + counts[e];
                    }
                    // Pack: each expert's rows land contiguously in `packed_in`.
                    // `original_pos[packed_idx] = original_token_idx` for the
                    // unpermute step at the end.
                    let mut packed_in = vec![0f32; m * k_dim];
                    let mut original_pos = vec![0usize; m];
                    let mut write_idx = vec![0usize; num_experts];
                    for i in 0..m {
                        let e = ids[i] as usize;
                        let dst_row = offsets[e] + write_idx[e];
                        packed_in[dst_row * k_dim..(dst_row + 1) * k_dim]
                            .copy_from_slice(&inp[i * k_dim..(i + 1) * k_dim]);
                        original_pos[dst_row] = i;
                        write_idx[e] += 1;
                    }

                    // One BLAS sgemm per expert. Skip experts with no
                    // tokens — common at the tail when M is much smaller
                    // than num_experts × k.
                    let mut packed_out = vec![0f32; m * n];
                    let expert_stride = k_dim * n;
                    let gmm_ord = crate::moe_residency::next_gmm_ord();
                    let moe_layer = gmm_ord / 3;
                    for e in 0..num_experts {
                        let count = counts[e];
                        if count == 0 {
                            continue;
                        }
                        crate::moe_residency::record_expert_tokens(moe_layer, e, count);
                        let in_start = offsets[e];
                        let in_slice = &packed_in[in_start * k_dim..(in_start + count) * k_dim];
                        let w_slab: &[f32] =
                            if !crate::moe_residency::expert_on_device_for_layer(moe_layer, e) {
                                if let Some(ptr) =
                                    crate::moe_residency::host_expert_weight_ptr(gmm_ord, e)
                                {
                                    std::slice::from_raw_parts(ptr, expert_stride)
                                } else {
                                    &wt[e * expert_stride..(e + 1) * expert_stride]
                                }
                            } else {
                                &wt[e * expert_stride..(e + 1) * expert_stride]
                            };
                        let out_slice = &mut packed_out[in_start * n..(in_start + count) * n];
                        crate::blas::sgemm(in_slice, w_slab, out_slice, count, k_dim, n);
                    }

                    // Unpermute back to original token order.
                    for packed_idx in 0..m {
                        let i = original_pos[packed_idx];
                        out[i * n..(i + 1) * n]
                            .copy_from_slice(&packed_out[packed_idx * n..(packed_idx + 1) * n]);
                    }
                }
            }

            Thunk::DequantGroupedMatMulGguf { .. } => {
                exec_dequant_grouped_mat_mul_gguf(thunk, base)
            }
            Thunk::DequantMoEWeightsGguf { .. } => exec_dequant_mo_e_weights_gguf(thunk, base),
            Thunk::TopK {
                src,
                dst,
                outer,
                axis_dim,
                k,
                indices_i64,
            } => {
                let outer = *outer as usize;
                let axis_dim = *axis_dim as usize;
                let k = *k as usize;
                unsafe {
                    let inp = sl(*src, base, outer * axis_dim);
                    // Repeated argmax with masking. O(k * axis_dim) per row;
                    // good enough for small k (MoE typical k=2–8). For larger
                    // k a partial heap would win.
                    let mut row_buf: Vec<f32> = vec![0.0; axis_dim];
                    if *indices_i64 != 0 {
                        let out = sl_mut_i64(*dst, base, outer * k);
                        for o in 0..outer {
                            row_buf.copy_from_slice(&inp[o * axis_dim..(o + 1) * axis_dim]);
                            for ki in 0..k {
                                let mut best_i = 0usize;
                                let mut best_v = row_buf[0];
                                for i in 1..axis_dim {
                                    let v = row_buf[i];
                                    if v > best_v {
                                        best_v = v;
                                        best_i = i;
                                    }
                                }
                                out[o * k + ki] = best_i as i64;
                                row_buf[best_i] = f32::NEG_INFINITY;
                            }
                        }
                    } else {
                        let out = sl_mut(*dst, base, outer * k);
                        for o in 0..outer {
                            row_buf.copy_from_slice(&inp[o * axis_dim..(o + 1) * axis_dim]);
                            for ki in 0..k {
                                let mut best_i = 0usize;
                                let mut best_v = row_buf[0];
                                for i in 1..axis_dim {
                                    let v = row_buf[i];
                                    if v > best_v {
                                        best_v = v;
                                        best_i = i;
                                    }
                                }
                                out[o * k + ki] = best_i as f32;
                                row_buf[best_i] = f32::NEG_INFINITY;
                            }
                        }
                        if let Some(cap) = schedule.moe_topk_capture.as_ref() {
                            cap.push_topk_f32(&out[..outer * k], axis_dim);
                        }
                    }
                }
            }

            Thunk::Reduce { .. } => exec_reduce(thunk, base),
            Thunk::ArgReduce { .. } => exec_arg_reduce(thunk, base),
            Thunk::Conv2D1x1 { .. } => exec_conv2_d1x1(thunk, base),
            Thunk::Conv2D { .. } => exec_conv2_d(thunk, base),
            Thunk::Conv3d { .. } => exec_conv3d(thunk, base),
            Thunk::ConvTranspose3d { .. } => exec_conv_transpose3d(thunk, base),
            Thunk::Pool2D {
                src,
                dst,
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
                kind,
            } => {
                let n = *n as usize;
                let c = *c as usize;
                let h = *h as usize;
                let w = *w as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let kernel_area = (kh * kw) as f32;
                unsafe {
                    let inp = sl(*src, base, n * c * h * w);
                    let out = sl_mut(*dst, base, n * c * h_out * w_out);
                    // Each (n, c) plane is independent and writes a disjoint
                    // output region, so pooling fans out over the channel-batch
                    // when RLX_FAST_CONV is set.
                    let out_addr = out.as_mut_ptr() as usize;
                    let is_max = matches!(kind, ReduceOp::Max);
                    let is_mean = matches!(kind, ReduceOp::Mean);
                    // No-padding windows (the conv-net case) are always fully
                    // in-bounds, so the hot path drops the per-element bounds
                    // branches and hoists the reduce-op choice out of the loop.
                    let nopad = ph == 0 && pw == 0;
                    let pool_plane = |nc: usize| {
                        let ni = nc / c;
                        let ci = nc % c;
                        let in_chan = ni * c * h * w + ci * h * w;
                        let out_chan = ni * c * h_out * w_out + ci * h_out * w_out;
                        let op = out_addr as *mut f32;
                        for ho in 0..h_out {
                            for wo in 0..w_out {
                                let acc = if nopad {
                                    let row0 = in_chan + (ho * sh) * w + wo * sw;
                                    let mut a = if is_max { f32::NEG_INFINITY } else { 0.0 };
                                    for ki in 0..kh {
                                        let row = row0 + ki * w;
                                        if is_max {
                                            for kj in 0..kw {
                                                a = a.max(inp[row + kj]);
                                            }
                                        } else {
                                            for kj in 0..kw {
                                                a += inp[row + kj];
                                            }
                                        }
                                    }
                                    a
                                } else {
                                    let mut a = if is_max { f32::NEG_INFINITY } else { 0.0 };
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let hi = ho * sh + ki;
                                            let wi = wo * sw + kj;
                                            if hi < ph || wi < pw {
                                                continue;
                                            }
                                            let hi = hi - ph;
                                            let wi = wi - pw;
                                            if hi >= h || wi >= w {
                                                continue;
                                            }
                                            let v = inp[in_chan + hi * w + wi];
                                            if is_max {
                                                a = a.max(v);
                                            } else {
                                                a += v;
                                            }
                                        }
                                    }
                                    a
                                };
                                let acc = if is_mean { acc / kernel_area } else { acc };
                                *op.add(out_chan + ho * w_out + wo) = acc;
                            }
                        }
                    };
                    if fast_conv_enabled() && crate::pool::should_parallelize(n * c * h_out * w_out)
                    {
                        crate::pool::par_for(
                            n * c,
                            crate::pool::outer_chunk(n * c),
                            &|off, cnt| {
                                for nc in off..off + cnt {
                                    pool_plane(nc);
                                }
                            },
                        );
                    } else {
                        for nc in 0..n * c {
                            pool_plane(nc);
                        }
                    }
                }
            }

            Thunk::ReluBackward { .. } => exec_relu_backward(thunk, base),
            Thunk::ReluBackwardF64 { .. } => exec_relu_backward_f64(thunk, base),
            Thunk::QMatMul { .. } => exec_q_mat_mul(thunk, base),
            Thunk::QConv2d {
                x,
                w,
                bias,
                out,
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
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let n = *n as usize;
                let c_in = *c_in as usize;
                let h = *h as usize;
                let w_in = *w_in as usize;
                let c_out = *c_out as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let dh = *dh as usize;
                let dw = *dw as usize;
                let groups = *groups as usize;
                let c_in_per_g = c_in / groups;
                let c_out_per_g = c_out / groups;
                unsafe {
                    let x_ptr = base.add(*x) as *const i8;
                    let w_ptr = base.add(*w) as *const i8;
                    let bias_ptr = base.add(*bias) as *const i32;
                    let out_ptr = base.add(*out) as *mut i8;
                    for ni in 0..n {
                        for co in 0..c_out {
                            let g = co / c_out_per_g;
                            let ci_start = g * c_in_per_g;
                            for ho in 0..h_out {
                                for wo in 0..w_out {
                                    let mut acc: i32 = *bias_ptr.add(co);
                                    for ci_off in 0..c_in_per_g {
                                        let ci = ci_start + ci_off;
                                        let in_chan = ((ni * c_in) + ci) * h * w_in;
                                        let wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                let hi = ho * sh + ki * dh;
                                                let wi = wo * sw + kj * dw;
                                                if hi < ph || wi < pw {
                                                    continue;
                                                }
                                                let hi = hi - ph;
                                                let wi = wi - pw;
                                                if hi >= h || wi >= w_in {
                                                    continue;
                                                }
                                                let xv = *x_ptr.add(in_chan + hi * w_in + wi)
                                                    as i32
                                                    - *x_zp;
                                                let wv = *w_ptr.add(wt_chan + ki * kw + kj) as i32
                                                    - *w_zp;
                                                acc += xv * wv;
                                            }
                                        }
                                    }
                                    let r = (acc as f32 * *mult).round() as i32 + *out_zp;
                                    let r = r.clamp(-128, 127) as i8;
                                    let dst = ((ni * c_out) + co) * h_out * w_out + ho * w_out + wo;
                                    *out_ptr.add(dst) = r;
                                }
                            }
                        }
                    }
                }
            }

            Thunk::Quantize { .. } => exec_quantize(thunk, base),
            Thunk::Dequantize { .. } => exec_dequantize(thunk, base),
            Thunk::FakeQuantize { .. } => exec_fake_quantize(thunk, base),
            Thunk::ActivationBackward { .. } => exec_activation_backward(thunk, base),
            Thunk::ActivationBackwardF64 { .. } => exec_activation_backward_f64(thunk, base),
            Thunk::FakeQuantizeLSQ { .. } => exec_fake_quantize_l_s_q(thunk, base),
            Thunk::FakeQuantizeLSQBackwardX { .. } => {
                exec_fake_quantize_l_s_q_backward_x(thunk, base)
            }
            Thunk::FakeQuantizeLSQBackwardScale { .. } => {
                exec_fake_quantize_l_s_q_backward_scale(thunk, base)
            }
            Thunk::FakeQuantizeBackward { .. } => exec_fake_quantize_backward(thunk, base),
            Thunk::LayerNormBackwardInput { .. } => exec_layer_norm_backward_input(thunk, base),
            Thunk::BatchNormInferenceBackwardInput { .. } => {
                exec_batch_norm_inference_backward_input(thunk, base)
            }
            Thunk::BatchNormInferenceBackwardGamma { .. } => {
                exec_batch_norm_inference_backward_gamma(thunk, base)
            }
            Thunk::BatchNormInferenceBackwardBeta { .. } => {
                exec_batch_norm_inference_backward_beta(thunk, base)
            }
            Thunk::LayerNormBackwardGamma { .. } => exec_layer_norm_backward_gamma(thunk, base),
            Thunk::RmsNormBackwardInput { .. } => exec_rms_norm_backward_input(thunk, base),
            Thunk::RmsNormBackwardGamma { .. } => exec_rms_norm_backward_gamma(thunk, base),
            Thunk::RmsNormBackwardBeta { .. } => exec_rms_norm_backward_beta(thunk, base),
            Thunk::RopeBackward { .. } => exec_rope_backward(thunk, base),
            Thunk::CumsumBackward { .. } => exec_cumsum_backward(thunk, base),
            Thunk::GroupNormBackwardInput { .. } => exec_group_norm_backward_input(thunk, base),
            Thunk::GroupNormBackwardGamma { .. } => exec_group_norm_backward_gamma(thunk, base),
            Thunk::GroupNormBackwardBeta { .. } => exec_group_norm_backward_beta(thunk, base),
            Thunk::GatherBackward { .. } => exec_gather_backward(thunk, base),
            Thunk::MaxPool2dBackward { .. } => exec_max_pool2d_backward(thunk, base),
            Thunk::Conv2dBackwardInput { .. } => exec_conv2d_backward_input(thunk, base),
            Thunk::Conv2dBackwardWeight { .. } => exec_conv2d_backward_weight(thunk, base),
            Thunk::Im2Col { .. } => exec_im2_col(thunk, base),
            Thunk::SoftmaxCrossEntropyDense { .. } => exec_softmax_cross_entropy_dense(thunk, base),
            Thunk::SoftmaxCrossEntropy { .. } => exec_softmax_cross_entropy(thunk, base),
            Thunk::SoftmaxCrossEntropyBackward { .. } => {
                exec_softmax_cross_entropy_backward(thunk, base)
            }
            Thunk::GatherAxis { .. } => exec_gather_axis(thunk, base),
            Thunk::Transpose {
                src,
                dst,
                in_total,
                out_dims,
                in_strides,
                elem_bytes,
            } => {
                // N-D index walk: for each output flat index, decompose into
                // multi-dim coords using out_dims, then dot with in_strides
                // to find the source flat index. Stride 0 = broadcast (read
                // the same input element repeatedly along that dim).
                let rank = out_dims.len();
                let total: usize = out_dims.iter().map(|&d| d as usize).product();
                // Empty output (e.g. Zipformer downsample pad `Expand` → `[0,1,C]`
                // when T is already a multiple of the stride) — nothing to write.
                if total == 0 {
                    // fall through
                } else {
                    let in_total = *in_total as usize;
                    unsafe {
                        if *elem_bytes == 1 {
                            // 1-byte dtypes (Bool / I8 / U8). Without this branch the
                            // `else` path below reads/writes 4 bytes per element via the
                            // f32 slice, corrupting e.g. a broadcast of the VITS attention
                            // mask (Bool, expanded over heads) — masking wrong positions.
                            let inp = arena_buf[*src..*src + in_total].to_vec();
                            let out = &mut arena_buf[*dst..*dst + total];
                            let mut idx = vec![0usize; rank];
                            for o in 0..total {
                                let mut src_idx = 0usize;
                                for d in 0..rank {
                                    src_idx += idx[d] * in_strides[d] as usize;
                                }
                                out[o] = inp[broadcast_src_index(src_idx, in_total)];
                                for d in (0..rank).rev() {
                                    idx[d] += 1;
                                    if idx[d] < out_dims[d] as usize {
                                        break;
                                    }
                                    idx[d] = 0;
                                }
                            }
                        } else if *elem_bytes == 8 {
                            let inp = sl_i64(*src, base, in_total);
                            let out = sl_mut_i64(*dst, base, total);
                            let mut idx = vec![0usize; rank];
                            for o in 0..total {
                                let mut src_idx = 0usize;
                                for d in 0..rank {
                                    src_idx += idx[d] * in_strides[d] as usize;
                                }
                                out[o] = inp[broadcast_src_index(src_idx, in_total)];
                                for d in (0..rank).rev() {
                                    idx[d] += 1;
                                    if idx[d] < out_dims[d] as usize {
                                        break;
                                    }
                                    idx[d] = 0;
                                }
                            }
                        } else {
                            let inp = sl(*src, base, in_total);
                            let out = sl_mut(*dst, base, total);
                            if rank == 4
                                && in_strides[0] == 0
                                && in_strides[2] == 0
                                && in_strides[3] == 0
                                && in_strides[1] != 0
                            {
                                // Per-channel broadcast out[n,c,h,w] = in[c*sc] — how
                                // a conv bias `[C]` reaches `[N,C,H,W]` (and its
                                // recompute in the backward graph). Fill each (n,c)
                                // plane with its scalar instead of an N-D index walk
                                // over every element; parallel over the channel-batch.
                                let d1 = out_dims[1] as usize;
                                let sc = in_strides[1] as usize;
                                let plane = (out_dims[2] as usize) * (out_dims[3] as usize);
                                let nc_total = (out_dims[0] as usize) * d1;
                                let out_addr = out.as_mut_ptr() as usize;
                                let fill = |nc0: usize, nc1: usize| {
                                    let op = out_addr as *mut f32;
                                    for nc in nc0..nc1 {
                                        let v = inp[(nc % d1) * sc];
                                        let base_off = nc * plane;
                                        for k in 0..plane {
                                            *op.add(base_off + k) = v;
                                        }
                                    }
                                };
                                if fast_conv_enabled() && crate::pool::should_parallelize(total) {
                                    crate::pool::par_for(
                                        nc_total,
                                        crate::pool::outer_chunk(nc_total),
                                        &|off, cnt| fill(off, off + cnt),
                                    );
                                } else {
                                    fill(0, nc_total);
                                }
                            } else if rank == 2 && in_strides[0] != 0 && in_strides[1] != 0 {
                                // Fast 2D transpose (the common matmul-backward case:
                                // xᵀ, wᵀ). out[i,j] = in[i*s0 + j*s1]; tiled over
                                // columns for write-locality, parallel over rows. Far
                                // cheaper than the general per-element index walk.
                                let d0 = out_dims[0] as usize;
                                let d1 = out_dims[1] as usize;
                                let s0 = in_strides[0] as usize;
                                let s1 = in_strides[1] as usize;
                                let out_addr = out.as_mut_ptr() as usize;
                                let tile = |i0: usize, i1: usize| {
                                    let op = out_addr as *mut f32;
                                    const T: usize = 32;
                                    let mut j0 = 0;
                                    while j0 < d1 {
                                        let j1 = (j0 + T).min(d1);
                                        for i in i0..i1 {
                                            let inb = i * s0;
                                            let outb = i * d1;
                                            for j in j0..j1 {
                                                *op.add(outb + j) = inp[inb + j * s1];
                                            }
                                        }
                                        j0 = j1;
                                    }
                                };
                                if fast_conv_enabled() && crate::pool::should_parallelize(total) {
                                    crate::pool::par_for(
                                        d0,
                                        crate::pool::outer_chunk(d0),
                                        &|off, cnt| tile(off, off + cnt),
                                    );
                                } else {
                                    tile(0, d0);
                                }
                            } else if rank >= 3
                                && *in_strides.last().unwrap_or(&0) == 1
                                && out_dims[rank - 1] as usize >= 8
                            {
                                // Innermost dim contiguous in both layouts: copy one
                                // row at a time (memcpy) instead of per-element index
                                // walk. Covers attention BSHD↔BHSD and most Expand/
                                // permute patterns in DiT.
                                let row = out_dims[rank - 1] as usize;
                                let planes = total / row;
                                let s_last = in_strides[rank - 1] as usize; // 1
                                let _ = s_last;
                                let out_addr = out.as_mut_ptr() as usize;
                                let in_addr = inp.as_ptr() as usize;
                                let dims = out_dims.to_vec();
                                let strides = in_strides.to_vec();
                                let copy_planes = |p0: usize, p1: usize| {
                                    let mut idx = vec![0usize; rank];
                                    let mut rem = p0;
                                    // Decode plane index into leading coords (exclude last dim).
                                    for d in (0..rank - 1).rev() {
                                        let dim = dims[d] as usize;
                                        idx[d] = rem % dim;
                                        rem /= dim;
                                    }
                                    for p in p0..p1 {
                                        let mut src = 0usize;
                                        for d in 0..rank - 1 {
                                            src += idx[d] * strides[d] as usize;
                                        }
                                        std::ptr::copy_nonoverlapping(
                                            (in_addr as *const f32).add(src),
                                            (out_addr as *mut f32).add(p * row),
                                            row,
                                        );
                                        for d in (0..rank - 1).rev() {
                                            idx[d] += 1;
                                            if idx[d] < dims[d] as usize {
                                                break;
                                            }
                                            idx[d] = 0;
                                        }
                                    }
                                };
                                if crate::pool::should_parallelize(total) && planes >= 4 {
                                    crate::pool::par_for(
                                        planes,
                                        crate::pool::outer_chunk(planes),
                                        &|off, cnt| copy_planes(off, off + cnt),
                                    );
                                } else {
                                    copy_planes(0, planes);
                                }
                            } else if fast_conv_enabled() && crate::pool::should_parallelize(total)
                            {
                                // Parallel: each chunk seeds its starting multi-index
                                // from `off`, then walks incrementally. Output writes
                                // are disjoint per `o`.
                                let out_addr = out.as_mut_ptr() as usize;
                                crate::pool::par_for(
                                    total,
                                    crate::pool::chunk_floor(total),
                                    &|off, cnt| {
                                        let mut idx = vec![0usize; rank];
                                        let mut rem = off;
                                        for d in (0..rank).rev() {
                                            let dim = out_dims[d] as usize;
                                            idx[d] = rem % dim;
                                            rem /= dim;
                                        }
                                        for o in off..off + cnt {
                                            let mut src_idx = 0usize;
                                            for d in 0..rank {
                                                src_idx += idx[d] * in_strides[d] as usize;
                                            }
                                            let v = inp[broadcast_src_index(src_idx, in_total)];
                                            *((out_addr as *mut f32).add(o)) = v;
                                            for d in (0..rank).rev() {
                                                idx[d] += 1;
                                                if idx[d] < out_dims[d] as usize {
                                                    break;
                                                }
                                                idx[d] = 0;
                                            }
                                        }
                                    },
                                );
                            } else {
                                let mut idx = vec![0usize; rank];
                                for o in 0..total {
                                    let mut src_idx = 0usize;
                                    for d in 0..rank {
                                        src_idx += idx[d] * in_strides[d] as usize;
                                    }
                                    out[o] = inp[broadcast_src_index(src_idx, in_total)];
                                    for d in (0..rank).rev() {
                                        idx[d] += 1;
                                        if idx[d] < out_dims[d] as usize {
                                            break;
                                        }
                                        idx[d] = 0;
                                    }
                                }
                            }
                        }
                    }
                } // total != 0
            }

            Thunk::CustomOp { .. } => exec_custom_op(thunk, base),
            Thunk::Reverse { .. } => exec_reverse(thunk, base),
        }
        if trace_done {
            eprintln!("[thunk {i} done]");
        }
    }
    if profile {
        if let Some((pn, pt)) = prof_prev.take() {
            profile_record(pn, pt.elapsed());
        }
        // Auto-dump under RLX_PROFILE_THUNKS so any binary shows where the run's
        // time went (per execute_thunks call — the hot subgraph stands out).
        dump_thunk_profile();
    }
}

#[inline(always)]
pub(crate) fn exec_nop(t: &Thunk) {
    let Thunk::Nop = t else { unreachable!() };
    {}
}
