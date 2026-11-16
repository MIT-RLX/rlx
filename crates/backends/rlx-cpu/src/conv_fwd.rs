// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NCHW conv2d forward (matches `Thunk::Conv2D` / PyTorch cross-correlation).

#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward_nchw_f32(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    debug_assert_eq!(inp.len(), n * c_in * h * w);
    debug_assert_eq!(wt.len(), c_out * c_in_per_g * kh * kw);
    debug_assert_eq!(out.len(), n * c_out * h_out * w_out);

    // im2col + BLAS GEMM. Per (batch, group) we gather the receptive fields into
    // a `[K, N]` column matrix (K = c_in_per_g·kh·kw, N = h_out·w_out) and compute
    // `out_g[M,N] = W_g[M,K] @ col[K,N]` via sgemm (M = c_out_per_g). The weight's
    // inner layout `[c_in_per_g, kh, kw]` matches the col row order, and the
    // output block for (batch, group) is contiguous `[c_out_per_g, N]` row-major,
    // so sgemm writes straight into `out`. Replaces a naive 6-deep loop that made
    // conv-heavy models (the deform host kernel had the same class of bug).
    let k = c_in_per_g * kh * kw;
    let hw_out = h_out * w_out;
    let mut col = vec![0f32; k * hw_out];
    for ni in 0..n {
        for g in 0..groups {
            let ci_start = g * c_in_per_g;
            // Build col[row, p] where row = (ci_off·kh + ki)·kw + kj, p = ho·w_out + wo.
            for ci_off in 0..c_in_per_g {
                let in_chan = (ni * c_in + ci_start + ci_off) * h * w;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let row = (ci_off * kh + ki) * kw + kj;
                        let col_row = &mut col[row * hw_out..(row + 1) * hw_out];
                        for ho in 0..h_out {
                            let hi = ho * sh + ki * dh;
                            let in_y = hi.wrapping_sub(ph);
                            let y_ok = hi >= ph && in_y < h;
                            for wo in 0..w_out {
                                let wi = wo * sw + kj * dw;
                                let in_x = wi.wrapping_sub(pw);
                                col_row[ho * w_out + wo] = if y_ok && wi >= pw && in_x < w {
                                    inp[in_chan + in_y * w + in_x]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                }
            }
            let wt_g = &wt[g * c_out_per_g * k..(g * c_out_per_g + c_out_per_g) * k];
            let out_off = (ni * c_out + g * c_out_per_g) * hw_out;
            let out_g = &mut out[out_off..out_off + c_out_per_g * hw_out];
            crate::blas::sgemm(wt_g, &col, out_g, c_out_per_g, k, hw_out);
        }
    }
}

/// NCDHW conv3d forward (PyTorch cross-correlation; the depth-axis analogue of
/// [`conv2d_forward_nchw_f32`]). Weight `[C_out, C_in/g, kD, kH, kW]`.
///
/// im2col + BLAS GEMM like the 2-D kernel, but **tiled**, because the 3-D
/// column matrix cannot be materialised whole: a 160³ volume with 24 channels
/// and a 3×3×3 kernel gives `K·N = 648 × 4.096M` floats — 10.6 GB. So the
/// output is walked one depth slice at a time, and each slice in position tiles
/// sized to keep the column buffer near `COL_BUDGET_F32`. Positions within a
/// depth slice are contiguous in the output, so each tile's GEMM result copies
/// back as one run per output channel.
///
/// This replaced a naive nine-deep scalar loop. The loop was correct — it is
/// still the reference the tests below check against — but a volumetric U-Net
/// is a few hundred GMAC, which at scalar speed is tens of minutes per scan.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_forward_ncdhw_f32(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    bias: Option<&[f32]>,
) {
    conv3d_forward_ncdhw_f32_budgeted(
        inp,
        wt,
        out,
        n,
        c_in,
        d,
        h,
        w,
        c_out,
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
        COL_BUDGET_F32,
        bias,
    )
}

/// [`conv3d_forward_ncdhw_f32`] with an explicit column budget.
///
/// Exists so the tiling can be tested on small inputs: with the production
/// budget a test-sized volume is always one tile, which would leave the tile
/// boundaries — the part most likely to be wrong — unexercised.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_forward_ncdhw_f32_budgeted(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    col_budget: usize,
    bias: Option<&[f32]>,
) {
    conv3d_forward_impl(
        inp,
        wt,
        out,
        n,
        c_in,
        d,
        h,
        w,
        c_out,
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
        col_budget,
        bias,
        Epilogue::default(),
    )
}

/// [`conv3d_forward_ncdhw_f32`] with `epi` applied to every output value after the bias: an
/// activation and a batch norm fused into the store, applied while the stored run is still in
/// cache.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_forward_ncdhw_f32_fused(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    bias: Option<&[f32]>,
    epi: Epilogue<'_>,
) {
    conv3d_forward_impl(
        inp,
        wt,
        out,
        n,
        c_in,
        d,
        h,
        w,
        c_out,
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
        COL_BUDGET_F32,
        bias,
        epi,
    )
}

/// What a convolution applies to each output value after its bias, in this order: an
/// activation (`act`, or LeakyReLU as `max(x, slope · x)`), then a per-channel `x · scale +
/// shift`. Each step is the same f32 operation the unfused graph's thunk performs, so applying
/// it in the store changes no bit of the result.
#[derive(Clone, Copy, Default)]
pub struct Epilogue<'a> {
    pub act: Option<fn(f32) -> f32>,
    pub leaky: Option<f32>,
    pub affine: Option<(&'a [f32], &'a [f32])>,
}

impl Epilogue<'_> {
    /// Apply to one stored run of output channel `co`.
    #[inline]
    fn finish(&self, run: &mut [f32], co: usize) {
        if let Some(f) = self.act {
            run.iter_mut().for_each(|v| *v = f(*v));
        }
        if let Some(a) = self.leaky {
            run.iter_mut().for_each(|v| *v = v.max(*v * a));
        }
        if let Some((s, t)) = self.affine {
            let (s, t) = (s[co], t[co]);
            run.iter_mut().for_each(|v| *v = *v * s + t);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv3d_forward_impl(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    dd: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    col_budget: usize,
    bias: Option<&[f32]>,
    epi: Epilogue<'_>,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    debug_assert_eq!(inp.len(), n * c_in * d * h * w);
    debug_assert_eq!(wt.len(), c_out * c_in_per_g * kd * kh * kw);
    debug_assert_eq!(out.len(), n * c_out * d_out * h_out * w_out);
    debug_assert!(bias.is_none_or(|b| b.len() == c_out));
    let k = c_in_per_g * kd * kh * kw;
    let hw_out = h_out * w_out;
    let dhw_out = d_out * hw_out;
    if k == 0 || hw_out == 0 || d_out == 0 {
        return;
    }
    if groups == 1
        && [sd, sh, sw, dd, dh, dw] == [1; 6]
        && d_out + kd == d + 2 * pd + 1
        && h_out + kh == h + 2 * ph + 1
        && w_out + kw == w + 2 * pw + 1
        && !crate::blas::f64_accum_enabled()
    {
        return conv3d_unit_stride(
            inp,
            wt,
            out,
            n,
            c_in,
            [d, h, w],
            c_out,
            [d_out, h_out, w_out],
            [kd, kh, kw],
            [pd, ph, pw],
            col_budget,
            bias,
            epi,
        );
    }

    // Work splits by output depth slice: each (batch, group, slice) is one
    // unit, and units write disjoint runs of `out`. Each task gets its own
    // column buffer, sized to its share of the budget, so the peak stays at
    // `col_budget` however many threads run. With the row-wise gather below,
    // MeshNet's 9 dilated 5-channel layers over a ~170×200×170 crop went from
    // 20 s to 2 s on an M-series (5 rayon workers), output bit-identical.
    let units = n * groups * d_out;
    let workers = crate::pool::num_threads().min(units).max(1);
    // Positions per tile: as many as fit the share, never zero, never more
    // than one depth slice.
    let tile = (col_budget / workers / k).clamp(1, hw_out);
    let out_at = SendPtr(out.as_mut_ptr());
    let out_len = out.len();
    let work_per_slice = k * c_out_per_g * hw_out;
    crate::pool::par_for(
        units,
        crate::pool::grain_for(work_per_slice),
        &|off, cnt| {
            let mut col = vec![0f32; k * tile];
            let mut acc = vec![0f32; c_out_per_g * tile];
            // SAFETY: every unit writes only its own depth slice of each of its
            // output channels; units are disjoint across tasks.
            let out = unsafe { std::slice::from_raw_parts_mut(out_at.get(), out_len) };
            for u in off..off + cnt {
                let od = u % d_out;
                let g = (u / d_out) % groups;
                let ni = u / (d_out * groups);
                let ci_start = g * c_in_per_g;
                let wt_g = &wt[g * c_out_per_g * k..(g * c_out_per_g + c_out_per_g) * k];
                let mut p0 = 0;
                while p0 < hw_out {
                    let len = tile.min(hw_out - p0);
                    // col[row, p] with row = ((ci_off·kD + kdi)·kH + ki)·kW + kj,
                    // matching the weight's inner layout so the GEMM lines up.
                    for ci_off in 0..c_in_per_g {
                        let in_chan = (ni * c_in + ci_start + ci_off) * d * h * w;
                        for kdi in 0..kd {
                            let di = od * sd + kdi * dd;
                            let in_z = di.wrapping_sub(pd);
                            let z_ok = di >= pd && in_z < d;
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let row = ((ci_off * kd + kdi) * kh + ki) * kw + kj;
                                    let dst = &mut col[row * tile..row * tile + len];
                                    if !z_ok {
                                        dst.fill(0.0);
                                        continue;
                                    }
                                    let plane = in_chan + in_z * h * w;
                                    // An output row at a time: one bounds check
                                    // per row, and at stride 1 the in-bounds
                                    // stretch is one copy of an input row. (Per
                                    // element, the divide and the checks cost more
                                    // than the GEMM for a narrow layer.)
                                    let mut t = 0;
                                    while t < len {
                                        let p = p0 + t;
                                        let (ho, wo0) = (p / w_out, p % w_out);
                                        let run = (w_out - wo0).min(len - t);
                                        let seg = &mut dst[t..t + run];
                                        t += run;
                                        let hi = ho * sh + ki * dh;
                                        let in_y = hi.wrapping_sub(ph);
                                        if hi < ph || in_y >= h {
                                            seg.fill(0.0);
                                            continue;
                                        }
                                        let row = &inp[plane + in_y * w..plane + in_y * w + w];
                                        if sw == 1 {
                                            // in_x = wo + shift, in bounds for wo in [a, b)
                                            let shift = (kj * dw) as isize - pw as isize;
                                            let end = wo0 + run;
                                            let a = ((-shift).max(0) as usize).clamp(wo0, end);
                                            let b = ((w as isize - shift).max(0) as usize)
                                                .clamp(a, end);
                                            seg[..a - wo0].fill(0.0);
                                            // (empty when the whole run is padding,
                                            // where `a + shift` may be negative)
                                            if a < b {
                                                seg[a - wo0..b - wo0].copy_from_slice(
                                                    &row[(a as isize + shift) as usize
                                                        ..(b as isize + shift) as usize],
                                                );
                                            }
                                            seg[b - wo0..].fill(0.0);
                                        } else {
                                            for (o, slot) in seg.iter_mut().enumerate() {
                                                let wi = (wo0 + o) * sw + kj * dw;
                                                let in_x = wi.wrapping_sub(pw);
                                                *slot = if wi >= pw && in_x < w {
                                                    row[in_x]
                                                } else {
                                                    0.0
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let acc_used = &mut acc[..c_out_per_g * len];
                    if len == tile {
                        crate::blas::sgemm(wt_g, &col, acc_used, c_out_per_g, k, len);
                    } else {
                        // The last tile is short, so the column buffer's stride
                        // no longer equals its logical width. Compact it rather
                        // than let the GEMM read across rows.
                        for row in 0..k {
                            col.copy_within(row * tile..row * tile + len, row * len);
                        }
                        crate::blas::sgemm(wt_g, &col[..k * len], acc_used, c_out_per_g, k, len);
                    }
                    // One contiguous run per output channel: a depth slice is
                    // contiguous in NCDHW, and so is a position range within it.
                    for co_off in 0..c_out_per_g {
                        let co = g * c_out_per_g + co_off;
                        let base = (ni * c_out + co) * dhw_out + od * hw_out + p0;
                        let run = &mut out[base..base + len];
                        run.copy_from_slice(&acc_used[co_off * len..(co_off + 1) * len]);
                        // a folded conv bias, added in the store: no bias-sized
                        // tensor, no separate add over the output
                        if let Some(b) = bias {
                            run.iter_mut().for_each(|v| *v += b[co]);
                        }
                        epi.finish(run, co);
                    }
                    p0 += len;
                }
            }
        },
    );
}

/// Unit-stride, undilated, ungrouped 3-D convolution with the in-plane taps as pointer offsets.
///
/// The general path gathers a column row per `(c_in, kd, kh, kw)` — 27 copies of the input for a
/// 3×3×3 kernel — and that gather, not the GEMM, was most of a SynthSeg layer: 0.5 TFLOP/s on
/// an M4 Pro against the 2 TFLOP/s its BLAS reaches on the same shapes. Here the buffer holds
/// one zero-padded row per `(c_in, kd)` covering the tile's output rows and the `kh − 1` below
/// them, `wp = w + 2·pw` wide. An output at `(ho, wo)` sits at `ho·wp + wo` on that padded grid,
/// and its tap `(kh, kw)` is the same buffer read `kh·wp + kw` further on, so each of the
/// `kh·kw` taps is one accumulating GEMM over a shifted view with `K = c_in·kd`: a ninth of the
/// copying. The `wp − w_out` columns past each output row are computed and dropped.
///
/// With one or two input channels that `K` is too short for the GEMM to pay, so only `kw` is
/// folded (`kh` joins the buffer's rows): `kw` GEMMs with `K = c_in·kd·kh`.
#[allow(clippy::too_many_arguments)]
fn conv3d_unit_stride(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    [d, h, w]: [usize; 3],
    c_out: usize,
    [d_out, h_out, w_out]: [usize; 3],
    [kd, kh, kw]: [usize; 3],
    [pd, ph, pw]: [usize; 3],
    col_budget: usize,
    bias: Option<&[f32]>,
    epi: Epilogue<'_>,
) {
    let fold_h = c_in * kd >= 8;
    let taps: Vec<(usize, usize)> = if fold_h {
        (0..kh).flat_map(|a| (0..kw).map(move |b| (a, b))).collect()
    } else {
        (0..kw).map(|b| (0, b)).collect()
    };
    // Buffer rows per tap, `(ci, z)` or `(ci, z, y)`, matching the weight rows below.
    let rows_per_ci = if fold_h { kd } else { kd * kh };
    let kr = c_in * rows_per_ci;
    let k = c_in * kd * kh * kw;
    // `[tap][c_out][kr]`: the weight's columns for one tap, contiguous for the GEMM.
    let mut wts = vec![0f32; taps.len() * c_out * kr];
    let wts_at = SendPtr(wts.as_mut_ptr());
    let wts_len = wts.len();
    let taps_ref = &taps;
    crate::pool::par_for(
        c_out,
        crate::pool::grain_for(k * taps.len()),
        &|off, cnt| {
            // SAFETY: each task fills only the rows of its own output channels.
            let wts = unsafe { std::slice::from_raw_parts_mut(wts_at.get(), wts_len) };
            for o in off..off + cnt {
                for (t, &(a, b)) in taps_ref.iter().enumerate() {
                    let dst = &mut wts[(t * c_out + o) * kr..(t * c_out + o + 1) * kr];
                    for (r, v) in dst.iter_mut().enumerate() {
                        let (ci, rest) = (r / rows_per_ci, r % rows_per_ci);
                        let (z, y) = if fold_h {
                            (rest, a)
                        } else {
                            (rest / kh, rest % kh)
                        };
                        *v = wt[(((o * c_in + ci) * kd + z) * kh + y) * kw + b];
                    }
                }
            }
        },
    );
    let wts = &wts;

    let wp = w + 2 * pw;
    let span_extra = if fold_h { kh - 1 } else { 0 };
    // Output rows per tile: the buffer, `kr` rows of `(rows + span_extra)·wp`, sized to a
    // worker's share of the budget. Capped well under the general path's budget: the
    // buffer is read once per tap, so it pays to keep it in cache.
    let workers = crate::pool::num_threads().max(1);
    let share = col_budget.min(UNIT_STRIDE_BUDGET_F32) / workers;
    let rows = (share / (kr * wp))
        .saturating_sub(span_extra)
        .clamp(1, h_out);
    let chunks = h_out.div_ceil(rows);
    let units = n * d_out * chunks;
    let hw_in = h * w;
    let dhw_out = d_out * h_out * w_out;
    let out_at = SendPtr(out.as_mut_ptr());
    let out_len = out.len();
    // Units are handed out one at a time rather than split up front: with performance and
    // efficiency cores (or a busy machine) a static split waits on the slowest share.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = k * c_out * rows * wp;
    let tasks = if units * work < crate::pool::TASK_WORK_FLOOR {
        1
    } else {
        workers.min(units)
    };
    crate::pool::par_for(tasks, 1, &|_, _| {
        let len = (rows + span_extra) * wp + kw;
        let mut buf = vec![0f32; kr * len];
        let mut acc = vec![0f32; c_out * rows * wp];
        // SAFETY: a unit writes only output rows `[ho0, ho0 + nr)` of its own depth slice,
        // and each unit is taken by exactly one task.
        let out = unsafe { std::slice::from_raw_parts_mut(out_at.get(), out_len) };
        loop {
            let u = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if u >= units {
                break;
            }
            let ch = u % chunks;
            let od = (u / chunks) % d_out;
            let ni = u / (chunks * d_out);
            let ho0 = ch * rows;
            let nr = rows.min(h_out - ho0);
            let lines = nr + span_extra;
            for (r, row) in buf.chunks_exact_mut(len).enumerate() {
                let (ci, rest) = (r / rows_per_ci, r % rows_per_ci);
                let (z, y) = if fold_h {
                    (rest, 0)
                } else {
                    (rest / kh, rest % kh)
                };
                // Out-of-range (padding) coordinates wrap to huge values and fail `< d`.
                let iz = (od + z).wrapping_sub(pd);
                for (l, seg) in row[..lines * wp].chunks_exact_mut(wp).enumerate() {
                    let iy = (ho0 + l + y).wrapping_sub(ph);
                    if iz >= d || iy >= h {
                        seg.fill(0.0);
                    } else {
                        let at = ((ni * c_in + ci) * d + iz) * hw_in + iy * w;
                        let src = &inp[at..at + w];
                        seg[..pw].fill(0.0);
                        seg[pw..pw + w].copy_from_slice(src);
                        seg[pw + w..].fill(0.0);
                    }
                }
                row[lines * wp..].fill(0.0);
            }
            let cols = nr * wp;
            for (t, &(a, b)) in taps.iter().enumerate() {
                crate::blas::sgemm_ld(
                    &wts[t * c_out * kr..(t + 1) * c_out * kr],
                    kr,
                    &buf[a * wp + b..],
                    len,
                    &mut acc,
                    cols,
                    c_out,
                    kr,
                    cols,
                    if t == 0 { 0.0 } else { 1.0 },
                );
            }
            for co in 0..c_out {
                let base = (ni * c_out + co) * dhw_out + (od * h_out + ho0) * w_out;
                let b = bias.map_or(0.0, |b| b[co]);
                for l in 0..nr {
                    let src = &acc[co * cols + l * wp..co * cols + l * wp + w_out];
                    let dst = &mut out[base + l * w_out..base + (l + 1) * w_out];
                    if bias.is_some() {
                        for (o, v) in dst.iter_mut().zip(src) {
                            *o = v + b;
                        }
                    } else {
                        dst.copy_from_slice(src);
                    }
                    epi.finish(dst, co);
                }
            }
        }
    });
}

/// [`conv3d_unit_stride`]'s buffer cap, in f32 elements (16 MB) across all workers.
const UNIT_STRIDE_BUDGET_F32: usize = 4 * 1024 * 1024;

/// `out`'s base pointer, handed to the depth-slice tasks.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: the tasks it is shared with write disjoint ranges.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    // a method, so a closure captures the wrapper (Sync), not the raw field
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Column-buffer target, in f32 elements (64 MB).
///
/// Large enough that the GEMM's `N` stays in the thousands — where BLAS is
/// efficient — and small enough to stay out of the territory where a 3-D
/// im2col allocates more memory than the machine has.
const COL_BUDGET_F32: usize = 16 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    /// The nine-deep scalar loop the tiled GEMM kernel replaced, kept verbatim
    /// as the reference. Every case below asserts the two agree, so the
    /// rewrite is checked against the behaviour it was meant to preserve
    /// rather than against my own re-derivation of the same indices.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        inp: &[f32],
        wt: &[f32],
        out: &mut [f32],
        n: usize,
        c_in: usize,
        d: usize,
        h: usize,
        w: usize,
        c_out: usize,
        d_out: usize,
        h_out: usize,
        w_out: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        sd: usize,
        sh: usize,
        sw: usize,
        pd: usize,
        ph: usize,
        pw: usize,
        dd: usize,
        dh: usize,
        dw: usize,
        groups: usize,
    ) {
        let c_in_per_g = c_in / groups;
        let c_out_per_g = c_out / groups;
        for ni in 0..n {
            for co in 0..c_out {
                let ci_start = (co / c_out_per_g) * c_in_per_g;
                for od in 0..d_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let mut acc = 0f32;
                            for ci_off in 0..c_in_per_g {
                                let in_chan = ((ni * c_in) + ci_start + ci_off) * d * h * w;
                                let wt_chan = ((co * c_in_per_g) + ci_off) * kd * kh * kw;
                                for kdi in 0..kd {
                                    let di = od * sd + kdi * dd;
                                    if di < pd {
                                        continue;
                                    }
                                    let di = di - pd;
                                    if di >= d {
                                        continue;
                                    }
                                    for ki in 0..kh {
                                        let hi = ho * sh + ki * dh;
                                        if hi < ph {
                                            continue;
                                        }
                                        let hi = hi - ph;
                                        if hi >= h {
                                            continue;
                                        }
                                        for kj in 0..kw {
                                            let wi = wo * sw + kj * dw;
                                            if wi < pw {
                                                continue;
                                            }
                                            let wi = wi - pw;
                                            if wi >= w {
                                                continue;
                                            }
                                            acc += inp[in_chan + (di * h + hi) * w + wi]
                                                * wt[wt_chan + (kdi * kh + ki) * kw + kj];
                                        }
                                    }
                                }
                            }
                            out[((ni * c_out) + co) * d_out * h_out * w_out
                                + (od * h_out + ho) * w_out
                                + wo] = acc;
                        }
                    }
                }
            }
        }
    }

    /// Deterministic, non-repeating and both-signed: a ramp would let index
    /// mistakes cancel, and all-positive data hides sign errors in the weights.
    fn ramp(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i * 2654435761 + seed * 40503) % 1013) as f32 / 1013.0;
                x * 2.0 - 1.0
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn check(
        name: &str,
        n: usize,
        c_in: usize,
        d: usize,
        h: usize,
        w: usize,
        c_out: usize,
        kd: usize,
        kh: usize,
        kw: usize,
        stride: [usize; 3],
        pad: [usize; 3],
        dil: [usize; 3],
        groups: usize,
        budget: usize,
    ) {
        let out_dim = |sz, k, s: usize, p, dl: usize| (sz + 2 * p - dl * (k - 1) - 1) / s + 1;
        let d_out = out_dim(d, kd, stride[0], pad[0], dil[0]);
        let h_out = out_dim(h, kh, stride[1], pad[1], dil[1]);
        let w_out = out_dim(w, kw, stride[2], pad[2], dil[2]);
        let inp = ramp(n * c_in * d * h * w, 1);
        let wt = ramp(c_out * (c_in / groups) * kd * kh * kw, 7);
        let mut got = vec![0f32; n * c_out * d_out * h_out * w_out];
        let mut want = vec![0f32; got.len()];

        reference(
            &inp, &wt, &mut want, n, c_in, d, h, w, c_out, d_out, h_out, w_out, kd, kh, kw,
            stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1], dil[2],
            groups,
        );
        conv3d_forward_ncdhw_f32_budgeted(
            &inp, &wt, &mut got, n, c_in, d, h, w, c_out, d_out, h_out, w_out, kd, kh, kw,
            stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1], dil[2],
            groups, budget, None,
        );

        assert!(
            want.iter().any(|v| v.abs() > 1e-3),
            "{name}: reference is all zeros"
        );
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            // f32 GEMM sums in a different order than the scalar loop, so the
            // tolerance is accumulation slack, not a correctness allowance.
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "{name}: element {i}: {a} vs reference {b}"
            );
        }
    }

    const BIG: usize = 16 * 1024 * 1024;

    #[test]
    fn same_padded_3x3x3_matches_the_reference() {
        // The SynthSeg / U-Net shape: odd kernel, unit stride, same padding.
        check(
            "same3", 1, 4, 7, 6, 5, 6, 3, 3, 3, [1; 3], [1; 3], [1; 3], 1, BIG,
        );
    }

    #[test]
    fn unpadded_and_anisotropic_kernels_match() {
        check(
            "valid", 1, 3, 6, 5, 7, 4, 3, 1, 2, [1; 3], [0; 3], [1; 3], 1, BIG,
        );
    }

    #[test]
    fn strided_and_dilated_match() {
        check(
            "stride",
            1,
            2,
            9,
            8,
            7,
            4,
            3,
            3,
            3,
            [2, 1, 3],
            [1, 0, 2],
            [1; 3],
            1,
            BIG,
        );
        check(
            "dilate",
            1,
            2,
            9,
            8,
            9,
            4,
            2,
            3,
            2,
            [1; 3],
            [2, 1, 0],
            [2, 1, 3],
            1,
            BIG,
        );
    }

    #[test]
    fn strided_and_dilated_together_match() {
        // stride and dilation on the same axes (the per-element W gather, its
        // taps `sw` apart per output and `dw` apart per kernel column), also
        // in tiles that start mid-row
        for (name, budget) in [("stride+dilate", BIG), ("stride+dilate, tiled", 5)] {
            check(
                name,
                2,
                3,
                11,
                9,
                13,
                5,
                3,
                2,
                3,
                [2, 3, 2],
                [2, 1, 3],
                [2, 1, 2],
                1,
                budget,
            );
        }
    }

    #[test]
    fn padding_wider_than_a_row_matches() {
        // a dilated "same" conv on a narrow volume: whole output rows fall in
        // the padding, which the row-wise gather once indexed below zero
        check(
            "pad > row",
            1,
            2,
            6,
            5,
            3,
            3,
            3,
            3,
            3,
            [1; 3],
            [4; 3],
            [4; 3],
            1,
            BIG,
        );
        check(
            "pad > row, tiled",
            1,
            2,
            6,
            5,
            3,
            3,
            3,
            3,
            3,
            [1; 3],
            [4; 3],
            [4; 3],
            1,
            7,
        );
    }

    #[test]
    fn grouped_and_depthwise_match() {
        check(
            "groups2", 1, 4, 5, 5, 5, 6, 3, 3, 3, [1; 3], [1; 3], [1; 3], 2, BIG,
        );
        check(
            "depthwise",
            1,
            4,
            5,
            6,
            5,
            4,
            3,
            3,
            3,
            [1; 3],
            [1; 3],
            [1; 3],
            4,
            BIG,
        );
    }

    #[test]
    fn a_batch_of_more_than_one_matches() {
        // A per-batch offset error is invisible at n=1, which is how it ships.
        check(
            "batch3", 3, 2, 5, 4, 6, 4, 3, 3, 3, [1; 3], [1; 3], [1; 3], 1, BIG,
        );
    }

    #[test]
    fn tiling_does_not_change_the_result() {
        // Budgets chosen so `hw_out` splits into several tiles with a short one
        // at the end — the compaction path, which the production budget never
        // reaches on a test-sized volume.
        for budget in [1, 7, 64, 199, 1024] {
            check(
                &format!("tile{budget}"),
                2,
                4,
                5,
                6,
                7,
                6,
                3,
                3,
                3,
                [1; 3],
                [1; 3],
                [1; 3],
                2,
                budget,
            );
        }
    }

    #[test]
    fn the_unit_stride_path_matches_at_every_tiling() {
        // (n, c_in, d, h, w, c_out, kd, kh, kw, pad): one input channel folds only kW,
        // the rest fold kH too; valid, same and over-padded; 1×1 and anisotropic kernels.
        let cases = [
            (1, 1, 5, 6, 7, 3, 3, 3, 3, [1, 1, 1]),
            (2, 3, 4, 5, 6, 4, 3, 3, 3, [1, 1, 1]),
            (1, 4, 6, 5, 7, 2, 3, 3, 3, [0, 0, 0]),
            (1, 5, 3, 4, 5, 6, 1, 1, 1, [0, 0, 0]),
            (1, 3, 5, 6, 7, 2, 3, 1, 5, [1, 0, 2]),
            (1, 2, 6, 7, 5, 3, 5, 5, 5, [2, 2, 2]),
            (1, 3, 4, 4, 4, 2, 3, 3, 3, [2, 2, 2]),
        ];
        for (n, c_in, d, h, w, c_out, kd, kh, kw, pad) in cases {
            for budget in [1, 50, 300, BIG] {
                check(
                    &format!("unit{c_in}x{kd}{kh}{kw}p{pad:?}b{budget}"),
                    n,
                    c_in,
                    d,
                    h,
                    w,
                    c_out,
                    kd,
                    kh,
                    kw,
                    [1; 3],
                    pad,
                    [1; 3],
                    1,
                    budget,
                );
            }
        }
    }

    #[test]
    fn a_bias_in_the_store_is_a_bias_added_after() {
        let (n, c_in, d, h, w, c_out) = (2, 3, 5, 4, 6, 4);
        let inp = ramp(n * c_in * d * h * w, 3);
        let wt = ramp(c_out * c_in * 27, 11);
        let bias = [0.5f32, -1.0, 2.0, 0.25];
        let mut plain = vec![0f32; n * c_out * d * h * w];
        let mut biased = plain.clone();
        for (out, b) in [(&mut plain, None), (&mut biased, Some(&bias[..]))] {
            conv3d_forward_ncdhw_f32_budgeted(
                &inp, &wt, out, n, c_in, d, h, w, c_out, d, h, w, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2,
                2, 1, 37, b,
            );
        }
        for (i, (p, q)) in plain.iter().zip(&biased).enumerate() {
            let co = (i / (d * h * w)) % c_out;
            assert_eq!(*q, p + bias[co], "element {i}");
        }
    }

    #[test]
    fn every_tile_size_agrees_with_the_single_tile_result() {
        // Same input, every budget: the tiling must be invisible in the output
        // to within accumulation slack. Not bit-exact — the tile width is the
        // GEMM's `N`, so changing it changes BLAS's blocking and the order it
        // sums in. Demanding equality here would be asserting something about
        // the BLAS rather than about this kernel.
        let (n, c_in, d, h, w, c_out) = (1, 3, 4, 5, 6, 3);
        let (d_out, h_out, w_out) = (4, 5, 6);
        let inp = ramp(n * c_in * d * h * w, 3);
        let wt = ramp(c_out * c_in * 27, 11);
        let mut single = vec![0f32; n * c_out * d_out * h_out * w_out];
        conv3d_forward_ncdhw_f32_budgeted(
            &inp,
            &wt,
            &mut single,
            n,
            c_in,
            d,
            h,
            w,
            c_out,
            d_out,
            h_out,
            w_out,
            3,
            3,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            BIG,
            None,
        );
        for budget in 1..40 {
            let mut tiled = vec![0f32; single.len()];
            conv3d_forward_ncdhw_f32_budgeted(
                &inp, &wt, &mut tiled, n, c_in, d, h, w, c_out, d_out, h_out, w_out, 3, 3, 3, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, budget, None,
            );
            for (i, (a, b)) in tiled.iter().zip(&single).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "budget {budget} changed element {i}: {a} vs {b}"
                );
            }
        }
    }
}
