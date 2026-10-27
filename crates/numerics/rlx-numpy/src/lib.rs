// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side numpy numerics reproduced bit-for-bit.
//!
//! A port that is "mathematically right" is routinely 1 ULP off numpy because numpy
//! fixes an evaluation order: `sum` is a blocked pairwise reduction, `round` is
//! half-to-even, `linspace` multiplies by a precomputed step and pins the endpoint,
//! and `sin`/`cos` are separate libm calls that LLVM would otherwise merge into one
//! `sincos`. These are the spellings the exg reference ports converged on, gathered
//! here so each port stops carrying its own copy.
//!
//! For numpy/torch *random streams* see `rlx-rng`.

/// numpy's internal `pairwise_sum` over a contiguous f64 run — the loop `np.sum` and
/// `np.mean` run along a contiguous axis. Like numpy's C routine it starts short runs
/// from `-0.0`, so an empty or all-`-0.0` input gives `-0.0`; [`sum`] adds the
/// reduction identity on top, as `np.sum` does.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0_f64;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0_f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum(a)` for a contiguous f64 array: [`pairwise_sum`] plus the identity `+0.0`,
/// which is where `np.sum([])` and `np.sum([-0.0])` get their `+0.0`.
pub fn sum(a: &[f64]) -> f64 {
    pairwise_sum(a) + 0.0
}

/// Numpy/`pairwise_sum`-style recursive pairwise summation in f32: base case
/// is an 8-way-unrolled accumulator loop (enables the same balanced-tree
/// error cancellation a vectorized reduction gets), the recursive case splits
/// at the middle (rounded down to a multiple of 8). A naive left-to-right
/// sum drifts from torch's actual (SIMD tree-reduced) `GroupNorm`/`LayerNorm`
/// CPU-kernel sums by 1–2 orders of magnitude more ULPs than this does for
/// the ~300–10k element reductions here (BENDR's `GroupNorm(EH/2, EH)` has
/// only 2 channels/group, so almost the whole per-block time axis lands in
/// one reduction) — empirically verified against the real checkpoint's
/// actual `GroupNorm` output down to 1–2 ULP.
pub fn pairwise_sum_f32(x: &[f32]) -> f32 {
    let n = x.len();
    if n <= 128 {
        let mut acc = [0f32; 8];
        let mut i = 0;
        while i + 8 <= n {
            for (l, a) in acc.iter_mut().enumerate() {
                *a += x[i + l];
            }
            i += 8;
        }
        let mut sum =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        while i < n {
            sum += x[i];
            i += 1;
        }
        sum
    } else {
        let half = (n / 2) - (n / 2) % 8;
        pairwise_sum_f32(&x[..half]) + pairwise_sum_f32(&x[half..])
    }
}

/// `np.round(x)` — half to even, including numpy's sign of zero (`-0.5` → `-0.0`).
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        // `copysign`: `np.round(-0.5)` is `-0.0`, which `-1.0 - (-1.0)` is not.
        (r - x.signum()).copysign(x)
    } else {
        r
    }
}

/// `np.linspace(start, stop, num)`.
///
/// `i · step + start` with `step = (stop − start)/(num − 1)` formed once, and the final
/// element **replaced** by `stop` rather than computed — so the sequence is exactly
/// symmetric at the ends but not in between.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    out[num - 1] = stop;
    out
}

/// `(x.sin(), x.cos())` — **separately**, which is not what the optimiser would do.
///
/// LLVM merges a `sin` and a `cos` of the same argument into the platform's `sincos`
/// (`__sincos_stret` on Darwin), which is **not** bit-identical to two separate calls:
/// measured on this host, it disagrees on 71 of 200 000 arguments. numpy calls them
/// separately, so a port that lets the merge happen matches the reference in a debug build
/// and drifts in a release one. Both arguments take a barrier, not just one — with a barrier
/// on only the cosine, a nearby `sin(x)` lets LLVM common-subexpression the unguarded call
/// against the merged result and the problem returns.
#[inline]
pub fn sin_cos_separate(x: f64) -> (f64, f64) {
    let s = std::hint::black_box(x).sin();
    let c = std::hint::black_box(x).cos();
    (s, c)
}

pub mod accelerate;
pub mod npy;
mod tanh_table;
pub mod torch;
/// `np.argsort` on the Linux x86-64 reference (x86-simd-sort AVX2).
pub use rlx_linux_ref::xss;

/// `np.tanh(x)` for `float64` on a SIMD-with-FMA build (NumPy ≥ 2.0 on arm64 and
/// AVX-512 x86): NumPy does **not** call libm here but its own SVML-derived kernel
/// (`loops_hyperbolic.dispatch.cpp.src::simd_tanh_f64`) — measured to differ from
/// libm `tanh` on 18 % of arguments. Per element: pick one of 16 subintervals from
/// the exponent bits, `y = |x| - b`, evaluate a degree-16 polynomial by FMA Horner,
/// saturate to ±1 past `|x| > HUGE_THRESHOLD`, restore the sign.
pub fn tanh(x: f64) -> f64 {
    use tanh_table::LUT18X16;
    let xb = x.to_bits() as i64;
    let ndnan = xb & 0x7ff8_0000_0000_0000;
    let special = ndnan <= 0x7fe0_0000_0000_0000; // `Le`, signed
    let idxs = ndnan.wrapping_sub(0x3fc0_0000_0000_0000);
    // 32-bit lanewise max(·, 0) / min(·, 0x780000) on both halves, then >> 51
    let lo = ((idxs as u64) & 0xffff_ffff) as u32 as i32;
    let hi = ((idxs as u64) >> 32) as u32 as i32;
    let clamp = |v: i32| v.clamp(0, 0x78_0000);
    let packed = ((clamp(hi) as u32 as u64) << 32) | (clamp(lo) as u32 as u64);
    let idx = (packed >> 51) as usize;
    let row = &LUT18X16[idx];
    let c = |k: usize| f64::from_bits(row[1 + k]);
    let b = f64::from_bits(row[0]);
    let sign = x.to_bits() & 0x8000_0000_0000_0000;
    let y = x.abs() - b;
    let mut r = c(16).mul_add(y, c(15));
    for k in (0..15).rev() {
        r = r.mul_add(y, c(k));
    }
    if !special {
        r = 1.0;
    }
    let r = f64::from_bits(r.to_bits() | sign);
    if x.is_nan() { f64::NAN } else { r }
}
