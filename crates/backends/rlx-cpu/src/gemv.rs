// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Few-row matmuls against a row-major `[n, k]` weight at memory bandwidth.
//!
//! `y = x · Wᵀ` for a handful of rows `x` (a decode step, a few beams) is a
//! pure stream over `W`. A single vendor GEMV runs on one core — with
//! Accelerate on the AMX unit, ~90 GB/s — so a 1 GB LM-head table costs
//! ~11 ms. Splitting `W`'s rows over the pool and dotting each row with every
//! `x` (one load of the row, `m` dots) reads it once at the bandwidth of all
//! the cores: ~175 GB/s on an M4 Pro.
//!
//! Each output is one thread's dot product in a fixed order, so results do not
//! depend on the thread count or the split.

/// `y[m, n] = x[m, k] · w[n, k]ᵀ` (overwrites `y`). Row-parallel over `n`.
pub fn matmul_nt_rows(x: &[f32], w: &[f32], y: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(
        x.len() >= m * k && w.len() >= n * k && y.len() >= m * n,
        "matmul_nt_rows: x [{m}, {k}], w [{n}, {k}], y [{m}, {n}]"
    );
    if m == 0 || n == 0 {
        return;
    }
    // 256 rows × 4 KB (k = 1024) is 1 MB of W per task: enough to amortize
    // the fork, small enough to balance a loaded machine.
    const GRAIN_ROWS: usize = 256;
    let (xa, wa, ya) = (
        x.as_ptr() as usize,
        w.as_ptr() as usize,
        y.as_mut_ptr() as usize,
    );
    let tasks = n.div_ceil(GRAIN_ROWS);
    crate::pool::par_for(tasks, 1, &|off, cnt| {
        for t in off..off + cnt {
            let r0 = t * GRAIN_ROWS;
            let r1 = (r0 + GRAIN_ROWS).min(n);
            for o in r0..r1 {
                // SAFETY: bounds asserted above; tasks write disjoint `o`
                // columns of `y`, and `x` / `w` are only read.
                let row = unsafe { std::slice::from_raw_parts((wa as *const f32).add(o * k), k) };
                for b in 0..m {
                    let xb =
                        unsafe { std::slice::from_raw_parts((xa as *const f32).add(b * k), k) };
                    let v = dot(row, xb);
                    unsafe { *(ya as *mut f32).add(b * n + o) = v };
                }
            }
        }
    });
}

/// `Σ a[i]·b[i]` with four 4-lane accumulators (16 per iteration).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let k = a.len();
        // SAFETY: NEON is baseline on aarch64; every load is within `k`.
        unsafe {
            let (pa, pb) = (a.as_ptr(), b.as_ptr());
            let mut acc = [vdupq_n_f32(0.0); 4];
            let mut i = 0;
            while i + 16 <= k {
                for (j, s) in acc.iter_mut().enumerate() {
                    *s = vfmaq_f32(
                        *s,
                        vld1q_f32(pa.add(i + 4 * j)),
                        vld1q_f32(pb.add(i + 4 * j)),
                    );
                }
                i += 16;
            }
            let mut sum = vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ));
            while i < k {
                sum += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            sum
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = [0f32; 16];
        let ((ca, ra), (cb, rb)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
        for (x, y) in ca.iter().zip(cb) {
            for j in 0..16 {
                acc[j] += x[j] * y[j];
            }
        }
        acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_definition() {
        for (m, n, k) in [(1, 1, 1), (1, 300, 1024), (4, 513, 37), (3, 1000, 64)] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6)
                .collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 11) % 17) as f32 * 0.05 - 0.4)
                .collect();
            let mut y = vec![f32::NAN; m * n];
            matmul_nt_rows(&x, &w, &mut y, m, n, k);
            for b in 0..m {
                for o in 0..n {
                    let want: f64 = (0..k)
                        .map(|i| x[b * k + i] as f64 * w[o * k + i] as f64)
                        .sum();
                    let got = y[b * n + o] as f64;
                    assert!(
                        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "m={m} n={n} k={k} [{b},{o}]: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn independent_of_the_split() {
        // Same values whatever the pool does: each output is one fixed-order dot.
        let (m, n, k) = (2, 2000, 300);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut a = vec![0f32; m * n];
        matmul_nt_rows(&x, &w, &mut a, m, n, k);
        for o in [0, 255, 256, 1999] {
            let mut one = vec![0f32; m];
            matmul_nt_rows(&x, &w[o * k..(o + 1) * k], &mut one, m, 1, k);
            assert_eq!([a[o], a[n + o]], [one[0], one[1]]);
        }
    }
}
