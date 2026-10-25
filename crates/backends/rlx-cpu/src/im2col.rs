// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NCHW im2col in `[M, C·kH·kW]` row layout (`M = N · H_out · W_out`).

/// im2col in `[M, C·kH·kW]` row layout (`M = N · H_out · W_out`).
#[allow(clippy::too_many_arguments)]
pub fn im2col_rows_layout(
    x: &[f32],
    col: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    // Wide kernels (temporal convs) copy each kernel row's in-bounds taps as
    // one slice; narrow ones (EEGNet's spatial conv, 1×1) are fastest element
    // by element, a bounds test per value. See `gather_timing`.
    let k = c_in * kh * kw;
    if kw == 1 {
        // 1-wide kernels (EEGNet's spatial conv, 1×1): one input column per
        // output position, so check it once and read the kernel rows' offsets
        // worked out per output row
        let mut rows = vec![None; kh];
        for ni in 0..n {
            let x_n = &x[ni * c_in * h * w..(ni + 1) * c_in * h * w];
            for ho in 0..h_out {
                for (ki, r) in rows.iter_mut().enumerate() {
                    let hi = (ho * sh + ki * dh) as isize - ph as isize;
                    *r = (hi >= 0 && hi < h as isize).then(|| hi as usize * w);
                }
                for wo in 0..w_out {
                    let out = &mut col[(ni * h_out * w_out + ho * w_out + wo) * k..][..k];
                    let wi = (wo * sw) as isize - pw as isize;
                    if wi < 0 || wi >= w as isize {
                        out.fill(0.0);
                        continue;
                    }
                    let wi = wi as usize;
                    for (ci, seg) in out.chunks_exact_mut(kh).enumerate() {
                        let x_c = &x_n[ci * h * w..(ci + 1) * h * w];
                        for (v, r) in seg.iter_mut().zip(&rows) {
                            *v = match r {
                                Some(r) => x_c[r + wi],
                                None => 0.0,
                            };
                        }
                    }
                }
            }
        }
        return;
    }
    if kw < 16 || dw_dil != 1 {
        for ni in 0..n {
            let x_base = ni * c_in * h * w;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let row = (ni * h_out * w_out + ho * w_out + wo) * k;
                    let mut elem = 0usize;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = (ho * sh + ki * dh) as isize - ph as isize;
                                let wi = (wo * sw + kj * dw_dil) as isize - pw as isize;
                                col[row + elem] =
                                    if hi < 0 || hi >= h as isize || wi < 0 || wi >= w as isize {
                                        0.0
                                    } else {
                                        x[x_base + (ci * h + hi as usize) * w + wi as usize]
                                    };
                                elem += 1;
                            }
                        }
                    }
                }
            }
        }
        return;
    }
    for ni in 0..n {
        let x_n = &x[ni * c_in * h * w..(ni + 1) * c_in * h * w];
        for ho in 0..h_out {
            for wo in 0..w_out {
                let out = &mut col[(ni * h_out * w_out + ho * w_out + wo) * k..][..k];
                let w0 = (wo * sw) as isize - pw as isize;
                let (lo, hi) = tap_range(w0, kw, w);
                for (c, seg) in out.chunks_exact_mut(kw).enumerate() {
                    let (ci, ki) = (c / kh, c % kh);
                    let hr = (ho * sh + ki * dh) as isize - ph as isize;
                    if hr < 0 || hr >= h as isize || lo == hi {
                        seg.fill(0.0);
                        continue;
                    }
                    let xrow = &x_n[(ci * h + hr as usize) * w..][..w];
                    let first = (w0 + lo as isize) as usize;
                    seg[..lo].fill(0.0);
                    seg[hi..].fill(0.0);
                    seg[lo..hi].copy_from_slice(&xrow[first..first + (hi - lo)]);
                }
            }
        }
    }
}

/// The taps `kj` in `0..kw` whose input column `w0 + kj` lies in `0..w`, as
/// `lo..hi` (empty when none does).
fn tap_range(w0: isize, kw: usize, w: usize) -> (usize, usize) {
    let lo = (-w0).clamp(0, kw as isize) as usize;
    let hi = (w as isize - w0).clamp(lo as isize, kw as isize) as usize;
    (lo, hi)
}

/// Run im2col against a contiguous f32 arena (`base` is byte offset 0).
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_im2col_rows_layout(
    x_off: usize,
    col_off: usize,
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
    base: *mut u8,
) {
    unsafe {
        let c_in = c_in as usize;
        let h = h as usize;
        let w = w as usize;
        let per_batch = c_in * h * w;
        let n_eff = if n == 0 { 0usize } else { n as usize };
        let x_floats = if n_eff == 0 {
            per_batch.max(1)
        } else {
            n_eff * per_batch
        };
        let xs = std::slice::from_raw_parts(base.add(x_off) as *const f32, x_floats);
        let n = if n == 0 {
            xs.len() / per_batch.max(1)
        } else {
            n_eff
        };
        let m = n * h_out as usize * w_out as usize;
        let k = c_in * kh as usize * kw as usize;
        let cols = std::slice::from_raw_parts_mut(base.add(col_off) as *mut f32, m * k);
        im2col_rows_layout(
            xs,
            cols,
            n,
            c_in,
            h,
            w,
            h_out as usize,
            w_out as usize,
            kh as usize,
            kw as usize,
            sh as usize,
            sw as usize,
            ph as usize,
            pw as usize,
            dh as usize,
            dw_dil as usize,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original per-element loop, kept verbatim as the definition.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        x: &[f32],
        col: &mut [f32],
        n: usize,
        c_in: usize,
        h: usize,
        w: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw_dil: usize,
    ) {
        let k = c_in * kh * kw;
        for ni in 0..n {
            let x_base = ni * c_in * h * w;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let row = (ni * h_out * w_out + ho * w_out + wo) * k;
                    let mut elem = 0usize;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = (ho * sh + ki * dh) as isize - ph as isize;
                                let wi = (wo * sw + kj * dw_dil) as isize - pw as isize;
                                col[row + elem] =
                                    if hi < 0 || hi >= h as isize || wi < 0 || wi >= w as isize {
                                        0.0
                                    } else {
                                        x[x_base + (ci * h + hi as usize) * w + wi as usize]
                                    };
                                elem += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bit for bit against the original loop over strides, padding (wider
    /// than the kernel's reach too), dilation and kernels wider than the input.
    #[test]
    fn rows_layout_matches_the_original_loop() {
        let mut cases = 0;
        for &(h, w) in &[(1usize, 1usize), (1, 7), (3, 5), (4, 9), (2, 16)] {
            for &(kh, kw) in &[(1usize, 1usize), (1, 3), (2, 2), (1, 9), (3, 4), (1, 20)] {
                for &(sh, sw) in &[(1usize, 1usize), (2, 3), (1, 2)] {
                    for &(ph, pw) in &[(0usize, 0usize), (1, 2), (0, 5), (2, 9)] {
                        for &(dh, dw) in &[(1usize, 1usize), (2, 3), (1, 2)] {
                            let (reach_h, reach_w) = ((kh - 1) * dh + 1, (kw - 1) * dw + 1);
                            if h + 2 * ph < reach_h || w + 2 * pw < reach_w {
                                continue;
                            }
                            let h_out = (h + 2 * ph - reach_h) / sh + 1;
                            let w_out = (w + 2 * pw - reach_w) / sw + 1;
                            let (n, c_in) = (2, 3);
                            let x: Vec<f32> = (0..n * c_in * h * w)
                                .map(|i| i as f32 * 0.5 - 7.0)
                                .collect();
                            let len = n * h_out * w_out * c_in * kh * kw;
                            let (mut got, mut want) = (vec![f32::NAN; len], vec![f32::NAN; len]);
                            im2col_rows_layout(
                                &x, &mut got, n, c_in, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
                                dh, dw,
                            );
                            naive(
                                &x, &mut want, n, c_in, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
                                dh, dw,
                            );
                            let same = got
                                .iter()
                                .zip(&want)
                                .all(|(a, b)| a.to_bits() == b.to_bits());
                            assert!(
                                same,
                                "h{h} w{w} k{kh}x{kw} s{sh},{sw} p{ph},{pw} d{dh},{dw}"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 500, "{cases} cases");
    }

    /// Timing, not correctness: against the original loop on EEG decoder conv
    /// shapes, alternating which runs first.
    /// `cargo test --release -p rlx-cpu --lib -- --ignored gather_timing --nocapture`
    #[test]
    #[ignore]
    fn gather_timing() {
        let shapes = [
            ("eegnet temporal", 1, 8, 256, 1, 64, 32),
            ("eegnet spatial", 16, 8, 256, 8, 1, 0),
            ("shallow temporal", 1, 22, 512, 1, 25, 0),
            ("pointwise", 16, 1, 256, 1, 1, 0),
        ];
        for (name, c_in, h, w, kh, kw, pw) in shapes {
            let h_out = h - kh + 1;
            let w_out = w + 2 * pw - kw + 1;
            let x: Vec<f32> = (0..c_in * h * w).map(|i| i as f32).collect();
            let mut col = vec![0f32; h_out * w_out * c_in * kh * kw];
            let reps = (2_000_000 / col.len()).max(3);
            let mut t = [0f64; 2];
            for round in 0..6 {
                for which in [round % 2, 1 - round % 2] {
                    let t0 = std::time::Instant::now();
                    for _ in 0..reps {
                        if which == 0 {
                            naive(
                                &x, &mut col, 1, c_in, h, w, h_out, w_out, kh, kw, 1, 1, 0, pw, 1,
                                1,
                            );
                        } else {
                            im2col_rows_layout(
                                &x, &mut col, 1, c_in, h, w, h_out, w_out, kh, kw, 1, 1, 0, pw, 1,
                                1,
                            );
                        }
                        std::hint::black_box(&col);
                    }
                    t[which] += t0.elapsed().as_secs_f64();
                }
            }
            println!(
                "rows_layout {name:18}: old {:.3}s  new {:.3}s  new/old {:.2}",
                t[0],
                t[1],
                t[1] / t[0]
            );
        }
    }
}
