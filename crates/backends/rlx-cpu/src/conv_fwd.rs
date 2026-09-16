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
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    debug_assert_eq!(inp.len(), n * c_in * d * h * w);
    debug_assert_eq!(wt.len(), c_out * c_in_per_g * kd * kh * kw);
    debug_assert_eq!(out.len(), n * c_out * d_out * h_out * w_out);
    let k = c_in_per_g * kd * kh * kw;
    let hw_out = h_out * w_out;
    let dhw_out = d_out * hw_out;
    if k == 0 || hw_out == 0 || d_out == 0 {
        return;
    }

    // Positions per tile: as many as fit the budget, never zero, never more
    // than one depth slice.
    let tile = (col_budget / k).clamp(1, hw_out);
    let mut col = vec![0f32; k * tile];
    let mut acc = vec![0f32; c_out_per_g * tile];

    for ni in 0..n {
        for g in 0..groups {
            let ci_start = g * c_in_per_g;
            let wt_g = &wt[g * c_out_per_g * k..(g * c_out_per_g + c_out_per_g) * k];
            for od in 0..d_out {
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
                                    for (t, slot) in dst.iter_mut().enumerate() {
                                        let p = p0 + t;
                                        let (ho, wo) = (p / w_out, p % w_out);
                                        let hi = ho * sh + ki * dh;
                                        let in_y = hi.wrapping_sub(ph);
                                        let wi = wo * sw + kj * dw;
                                        let in_x = wi.wrapping_sub(pw);
                                        *slot = if hi >= ph && in_y < h && wi >= pw && in_x < w {
                                            inp[plane + in_y * w + in_x]
                                        } else {
                                            0.0
                                        };
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
                        out[base..base + len]
                            .copy_from_slice(&acc_used[co_off * len..(co_off + 1) * len]);
                    }
                    p0 += len;
                }
            }
        }
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
            groups, budget,
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
        );
        for budget in 1..40 {
            let mut tiled = vec![0f32; single.len()];
            conv3d_forward_ncdhw_f32_budgeted(
                &inp, &wt, &mut tiled, n, c_in, d, h, w, c_out, d_out, h_out, w_out, 3, 3, 3, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, budget,
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
