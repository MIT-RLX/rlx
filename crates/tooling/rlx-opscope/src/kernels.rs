// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-pattern-specialized matmul kernels — the actuation for the exploits the
//! probe finds. Each is a **decomposition** of the dense matmul that the mined
//! structure makes cheap:
//! - [`sparse_skip_matmul`] — skip zero activations (exact; ~1/density speedup);
//! - [`factorize`] + [`factored_matmul`] — **low-rank** `W ≈ U·V`, so
//!   `x·W = (x·U)·V` with `min(k,n)/r` fewer FLOPs (bounded rank-r error);
//! - [`quant_matmul`] — per-output-channel int8 (4× smaller weights, bounded
//!   error);
//! - [`monarch_project`] + [`monarch_matmul`] — **Monarch** block-diagonal
//!   factorization `M = L·P·R`: a real sub-quadratic (`2·n^{1.5}`) matmul kernel;
//! - [`tucker_hosvd`] / [`tucker_reconstruct`] — **Tucker** (HOSVD) core+factors
//!   for ≥3-way tensors (conv weights, reshaped GEMMs) — a compressor;
//! - [`tt_svd`] / [`tt_reconstruct`] — **Tensor-Train** (TT-SVD) chain of 3-way
//!   cores — the deepest compressor, param count grows ~linearly in the modes.
//! Gate each behind the probe's recommendation (density / stable-rank / q-error /
//! multilinear-rank). The last three are chained by [`crate::layers`] and the
//! DADO allocator to pick a per-layer decomposition under a global budget.
//!
//! # Quantized matmul kernels (int8)
//!
//! [`matmul_f32`] / [`matmul_w8a16`] / [`matmul_w8a8`] are weight-stationary GEMM
//! kernels (`y[m,n] = x[m,k]·W`) at three precisions — the actuation for the
//! "quantize" verdict. NEON-vectorized on aarch64 (the W8A8 path emits `SDOT` via
//! inline asm). See `docs/quant-kernels.md` for the full measured speed/precision
//! story and the decision guide.
//!
//! **Honest scope (read before trusting a speedup):** these are *portable
//! demonstration* kernels, single-threaded, no cache tiling. On Apple Silicon
//! **Accelerate's AMX-backed `sgemm` beats every kernel here by 10–50×** — even
//! its *f32* beats the hand-written *int8*. So on Apple, int8's value is memory
//! footprint, not speed (route f32 to Accelerate/AMX; int8-on-AMX needs BNNS).
//! The hand kernels are the right path only where no vendor BLAS/matrix unit is
//! available. And every rel-error here is *per-matmul*, not model-output quality
//! — always end-to-end bench a quant recipe before shipping (`qwen_quant_bench`).

use crate::guard::dense_matmul;
use crate::svd::truncated_svd;

/// `x[m,k]·W[k,n]` skipping zero elements of `x` (per-row). Bit-exact vs dense;
/// runtime ∝ nnz(x), so ~`1/density` faster on sparse activations.
pub fn sparse_skip_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        let orow = &mut o[i * n..(i + 1) * n];
        for p in 0..k {
            let xv = x[i * k + p];
            if xv == 0.0 {
                continue;
            }
            let wrow = &w[p * n..(p + 1) * n];
            for j in 0..n {
                orow[j] += xv * wrow[j];
            }
        }
    }
    o
}

/// Randomized rank-`r` factorization `W[k,n] ≈ U[k,r]·V[r,n]` (range finder:
/// `Y=W·Ω`, orthonormalize → `U`, `V=Uᵀ·W`). One-time, on the stationary weight.
pub fn factorize(w: &[f32], k: usize, n: usize, r: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let r = r.min(k).min(n).max(1);
    // Deterministic random Ω [n,r] via xorshift.
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f64 / i64::MAX as f64) as f32
    };
    let omega: Vec<f32> = (0..n * r).map(|_| next()).collect();
    // Y = W·Ω  [k,r]
    let mut u = vec![0f32; k * r];
    for i in 0..k {
        for c in 0..r {
            let mut acc = 0f32;
            for p in 0..n {
                acc += w[i * n + p] * omega[p * r + c];
            }
            u[i * r + c] = acc;
        }
    }
    // Subspace (power) iterations: Y = W·(Wᵀ·Y) — sharpens the captured range so
    // the rank-r span is the dominant singular subspace (robust randomized SVD).
    for _ in 0..2 {
        let mut z = vec![0f32; n * r]; // Wᵀ·Y  [n,r]
        for p in 0..n {
            for c in 0..r {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += w[i * n + p] * u[i * r + c];
                }
                z[p * r + c] = acc;
            }
        }
        for i in 0..k {
            for c in 0..r {
                let mut acc = 0f32;
                for p in 0..n {
                    acc += w[i * n + p] * z[p * r + c];
                }
                u[i * r + c] = acc;
            }
        }
    }
    // Orthonormalize U columns (modified Gram-Schmidt) with a RELATIVE rank-
    // reveal: a redundant column collapses to a ~1e-6 numerical residual after
    // orthogonalization — normalizing that to a unit vector would inject noise
    // into the basis, so instead zero it out (it spans nothing).
    let ref_norm = (0..r)
        .map(|c| (0..k).map(|i| u[i * r + c].powi(2)).sum::<f32>().sqrt())
        .fold(0f32, f32::max)
        .max(1e-20);
    for c in 0..r {
        let mut nrm = 0f32;
        for i in 0..k {
            nrm += u[i * r + c] * u[i * r + c];
        }
        nrm = nrm.sqrt();
        if nrm < 1e-4 * ref_norm {
            for i in 0..k {
                u[i * r + c] = 0.0; // rank-deficient direction → contributes nothing
            }
            continue;
        }
        for i in 0..k {
            u[i * r + c] /= nrm;
        }
        for c2 in c + 1..r {
            let mut dot = 0f32;
            for i in 0..k {
                dot += u[i * r + c] * u[i * r + c2];
            }
            for i in 0..k {
                u[i * r + c2] -= dot * u[i * r + c];
            }
        }
    }
    // V = Uᵀ·W  [r,n]
    let mut v = vec![0f32; r * n];
    for c in 0..r {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += u[i * r + c] * w[i * n + j];
            }
            v[c * n + j] = acc;
        }
    }
    (u, v, r)
}

/// Factored matmul `x·(U·V) = (x·U)·V`.
pub fn factored_matmul(
    x: &[f32],
    u: &[f32],
    v: &[f32],
    m: usize,
    k: usize,
    r: usize,
    n: usize,
) -> Vec<f32> {
    let xu = dense_matmul(x, u, m, k, r); // [m,r]
    dense_matmul(&xu, v, m, r, n) // [m,n]
}

/// Per-output-channel symmetric int8 matmul (dequantized to f32). 4× smaller
/// weights; error bounded by the per-channel quant step.
pub fn quant_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut wq = vec![0i8; k * n];
    let mut scale = vec![0f32; n];
    for j in 0..n {
        let mut amax = 0f32;
        for p in 0..k {
            amax = amax.max(w[p * n + j].abs());
        }
        let sc = if amax < 1e-20 { 1.0 } else { amax / 127.0 };
        scale[j] = sc;
        for p in 0..k {
            wq[p * n + j] = (w[p * n + j] / sc).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * wq[p * n + j] as f32;
            }
            o[i * n + j] = acc * scale[j];
        }
    }
    o
}

// ─────────────── weight-stationary decode GEMV (int8 vs f32) ───────────────
//
// The quant *latency* win is memory bandwidth, and it only shows in the right
// shape: token-by-token decode is a GEMV where each weight is read once. The
// naive weight-*row* (SAXPY) form is dominated by f32 accumulator traffic and
// hides the win — so use the **dot-product** form with the weight transposed to
// output-major `[n,k]` (each output's weights contiguous): activations stay in
// cache, the accumulator is a register, and weight traffic dominates. Then int8
// weights (1 byte) stream 4× less than f32 (4 bytes). On aarch64 the i8→f32
// widen is vectorized with NEON so the bandwidth saving isn't eaten by the cast.

/// Transpose `W[k,n]` (row-major) → `Wt[n,k]` — the output-major layout a
/// weight-stationary GEMV wants. One-time, at load.
pub fn transpose(w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut wt = vec![0f32; n * k];
    for p in 0..k {
        for j in 0..n {
            wt[j * k + p] = w[p * n + j];
        }
    }
    wt
}

/// Quantize `W[k,n]` per output channel to int8 in output-major `[n,k]` layout
/// (transpose + quantize in one pass). Returns `(codes[n*k], scale[n])`.
pub fn quantize_cols_t(w: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; n * k];
    let mut sc = vec![0f32; n];
    for j in 0..n {
        let mut amax = 0f32;
        for p in 0..k {
            amax = amax.max(w[p * n + j].abs());
        }
        let s = if amax < 1e-20 { 1.0 } else { amax / 127.0 };
        sc[j] = s;
        for p in 0..k {
            q[j * k + p] = (w[p * n + j] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, sc)
}

// Portable fallback (also the parity reference in tests). Used whenever the NEON
// path isn't compiled — non-aarch64, or aarch64 without the `neon` feature.
#[cfg_attr(all(target_arch = "aarch64", feature = "neon"), allow(dead_code))]
fn gemv_f32_dot_scalar(x: &[f32], wt: &[f32], k: usize, n: usize, y: &mut [f32]) {
    for j in 0..n {
        let wr = &wt[j * k..(j + 1) * k];
        let mut s = 0f32;
        for p in 0..k {
            s += x[p] * wr[p];
        }
        y[j] = s;
    }
}

#[cfg_attr(all(target_arch = "aarch64", feature = "neon"), allow(dead_code))]
fn gemv_i8_dot_scalar(x: &[f32], wtq: &[i8], sc: &[f32], k: usize, n: usize, y: &mut [f32]) {
    for j in 0..n {
        let wr = &wtq[j * k..(j + 1) * k];
        let mut s = 0f32;
        for p in 0..k {
            s += x[p] * wr[p] as f32;
        }
        y[j] = s * sc[j];
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn gemv_f32_dot_neon(x: &[f32], wt: &[f32], k: usize, n: usize, y: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        for j in 0..n {
            let wr = wt.as_ptr().add(j * k);
            // 4 independent accumulators break the FMA dependency chain so the
            // core issues loads at bandwidth instead of stalling on FMA latency.
            let (mut a0, mut a1, mut a2, mut a3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            let mut p = 0;
            while p + 16 <= k {
                a0 = vfmaq_f32(a0, vld1q_f32(x.as_ptr().add(p)), vld1q_f32(wr.add(p)));
                a1 = vfmaq_f32(
                    a1,
                    vld1q_f32(x.as_ptr().add(p + 4)),
                    vld1q_f32(wr.add(p + 4)),
                );
                a2 = vfmaq_f32(
                    a2,
                    vld1q_f32(x.as_ptr().add(p + 8)),
                    vld1q_f32(wr.add(p + 8)),
                );
                a3 = vfmaq_f32(
                    a3,
                    vld1q_f32(x.as_ptr().add(p + 12)),
                    vld1q_f32(wr.add(p + 12)),
                );
                p += 16;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            while p < k {
                s += x[p] * *wr.add(p);
                p += 1;
            }
            y[j] = s;
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
unsafe fn gemv_i8_dot_neon(x: &[f32], wtq: &[i8], sc: &[f32], k: usize, n: usize, y: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        for j in 0..n {
            let wr = wtq.as_ptr().add(j * k);
            // One 16-byte int8 load feeds four independent f32 accumulators — the
            // widen (vmovl) is off the critical path, so this is bandwidth-bound.
            let (mut a0, mut a1, mut a2, mut a3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            let mut p = 0;
            while p + 16 <= k {
                let w8 = vld1q_s8(wr.add(p)); // 16×i8, one 16-byte load
                let lo = vmovl_s8(vget_low_s8(w8)); // →8×i16
                let hi = vmovl_s8(vget_high_s8(w8));
                let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))); // →4×f32
                let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
                let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
                let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
                a0 = vfmaq_f32(a0, vld1q_f32(x.as_ptr().add(p)), w0);
                a1 = vfmaq_f32(a1, vld1q_f32(x.as_ptr().add(p + 4)), w1);
                a2 = vfmaq_f32(a2, vld1q_f32(x.as_ptr().add(p + 8)), w2);
                a3 = vfmaq_f32(a3, vld1q_f32(x.as_ptr().add(p + 12)), w3);
                p += 16;
            }
            let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            while p < k {
                s += x[p] * (*wr.add(p)) as f32;
                p += 1;
            }
            y[j] = s * sc[j];
        }
    }
}

/// Weight-stationary f32 GEMV `y[n] = x[k]·Wtᵀ` (Wt is `[n,k]`). NEON on aarch64.
pub fn gemv_f32_dot(x: &[f32], wt: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    unsafe {
        gemv_f32_dot_neon(x, wt, k, n, &mut y);
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    gemv_f32_dot_scalar(x, wt, k, n, &mut y);
    y
}

/// Weight-stationary int8 GEMV, **W8A16** (int8 weights, f32 activations). `wtq`/
/// `sc` come from [`quantize_cols_t`]. NEON i8→f32 widening on aarch64. The widen
/// is pure overhead once the weight is cache-resident, so this rarely beats f32.
pub fn gemv_i8_dot(x: &[f32], wtq: &[i8], sc: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    unsafe {
        gemv_i8_dot_neon(x, wtq, sc, k, n, &mut y);
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    gemv_i8_dot_scalar(x, wtq, sc, k, n, &mut y);
    y
}

/// Per-tensor symmetric int8 quantization of an activation vector → `(codes, scale)`.
pub fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let s = if amax < 1e-20 { 1.0 } else { amax / 127.0 };
    (
        x.iter()
            .map(|&v| (v / s).round().clamp(-127.0, 127.0) as i8)
            .collect(),
        s,
    )
}

fn gemv_i8i8_dot_scalar(
    xq: &[i8],
    wtq: &[i8],
    sx: f32,
    sc: &[f32],
    k: usize,
    n: usize,
    y: &mut [f32],
) {
    for j in 0..n {
        let wr = &wtq[j * k..(j + 1) * k];
        let mut s = 0i32;
        for p in 0..k {
            s += xq[p] as i32 * wr[p] as i32;
        }
        y[j] = s as f32 * sx * sc[j];
    }
}

// int8×int8 dot with the ARMv8.2 DotProd `SDOT` instruction: one `vdotq_s32`
// does 16 int8 MACs into 4 int32 lanes — ~4× the f32 FMA MAC throughput. This is
// the actual compute win (what llama.cpp's Q8_0 uses); needs W8A8 (quantized
// activations too). Detected at runtime; falls back to the scalar dot otherwise.
// `vdotq_s32` is still unstable (`stdarch_neon_dotprod`), so emit the `SDOT`
// instruction via inline asm instead — stable, and identical codegen. Enabling
// `dotprod` for this fn lets the assembler accept `sdot`; the caller runtime-
// detects the feature before dispatching here.
#[cfg(all(target_arch = "aarch64", feature = "dotprod"))]
#[target_feature(enable = "dotprod")]
unsafe fn gemv_i8i8_dot_dotprod(
    xq: &[i8],
    wtq: &[i8],
    sx: f32,
    sc: &[f32],
    k: usize,
    n: usize,
    y: &mut [f32],
) {
    use std::arch::aarch64::*;
    use std::arch::asm;
    unsafe {
        for j in 0..n {
            let wr = wtq.as_ptr().add(j * k);
            let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0)); // 2 accumulators for ILP
            let mut p = 0;
            while p + 32 <= k {
                let (xa, wa) = (vld1q_s8(xq.as_ptr().add(p)), vld1q_s8(wr.add(p)));
                let (xb, wb) = (vld1q_s8(xq.as_ptr().add(p + 16)), vld1q_s8(wr.add(p + 16)));
                // sdot Vacc.4s, Vx.16b, Vw.16b — 16 int8 MACs into 4 i32 lanes.
                asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) a0, in(vreg) xa, in(vreg) wa, options(pure, nomem, nostack, preserves_flags));
                asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) a1, in(vreg) xb, in(vreg) wb, options(pure, nomem, nostack, preserves_flags));
                p += 32;
            }
            let mut acc = vaddq_s32(a0, a1);
            while p + 16 <= k {
                let (xa, wa) = (vld1q_s8(xq.as_ptr().add(p)), vld1q_s8(wr.add(p)));
                asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) xa, in(vreg) wa, options(pure, nomem, nostack, preserves_flags));
                p += 16;
            }
            let mut s = vaddvq_s32(acc);
            while p < k {
                s += xq[p] as i32 * (*wr.add(p)) as i32;
                p += 1;
            }
            y[j] = s as f32 * sx * sc[j];
        }
    }
}

/// Weight-stationary int8 GEMV, **W8A8** (int8 weights *and* activations). `xq`/
/// `sx` from [`quantize_row_i8`], `wtq`/`sc` from [`quantize_cols_t`]. Uses the
/// hardware `SDOT` int8 dot product on aarch64+dotprod — the real compute win.
pub fn gemv_i8i8_dot(xq: &[i8], wtq: &[i8], sx: f32, sc: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    #[cfg(all(target_arch = "aarch64", feature = "dotprod"))]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            unsafe { gemv_i8i8_dot_dotprod(xq, wtq, sx, sc, k, n, &mut y) };
            return y;
        }
    }
    gemv_i8i8_dot_scalar(xq, wtq, sx, sc, k, n, &mut y);
    y
}

// ─── matmul-level kernels: y[m,n] = x[m,k]·W (weight pre-transposed to [n,k]) ───
//
// The decode GEMV generalizes to a GEMM by looping rows — for the small `m` of
// LLM inference (decode m=1, prefill m=seq) this reuses the validated NEON dot
// cores and writes straight into each output row (no per-row alloc). Three
// precisions: f32, **W8A16** (int8 weights, f32 activations), **W8A8** (both
// int8, hardware SDOT). Weights come from [`quantize_cols_t`]; W8A8 activations
// from [`quantize_rows_i8`].

/// f32 GEMM `y[m,n] = x[m,k]·Wtᵀ`, `Wt` is `[n,k]` (from [`transpose`]).
pub fn matmul_f32(x: &[f32], wt: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        let (xr, yr) = (&x[i * k..(i + 1) * k], &mut y[i * n..(i + 1) * n]);
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        unsafe {
            gemv_f32_dot_neon(xr, wt, k, n, yr);
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        gemv_f32_dot_scalar(xr, wt, k, n, yr);
    }
    y
}

/// W8A16 GEMM: int8 weights (`wtq`/`sc` from [`quantize_cols_t`]), f32 activations.
pub fn matmul_w8a16(x: &[f32], wtq: &[i8], sc: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        let (xr, yr) = (&x[i * k..(i + 1) * k], &mut y[i * n..(i + 1) * n]);
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        unsafe {
            gemv_i8_dot_neon(xr, wtq, sc, k, n, yr);
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        gemv_i8_dot_scalar(xr, wtq, sc, k, n, yr);
    }
    y
}

/// Per-row symmetric int8 quantization of activations `x[m,k]` → `(codes, scale[m])`.
pub fn quantize_rows_i8(x: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; m * k];
    let mut s = vec![0f32; m];
    for i in 0..m {
        let (qr, sr) = quantize_row_i8(&x[i * k..(i + 1) * k]);
        q[i * k..(i + 1) * k].copy_from_slice(&qr);
        s[i] = sr;
    }
    (q, s)
}

/// W8A8 GEMM: int8 weights AND int8 activations (`xq`/`sx` from
/// [`quantize_rows_i8`]). Hardware SDOT on aarch64+dotprod, scalar otherwise.
pub fn matmul_w8a8(
    xq: &[i8],
    sx: &[f32],
    wtq: &[i8],
    sc: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    #[cfg(all(target_arch = "aarch64", feature = "dotprod"))]
    let has_dot = std::arch::is_aarch64_feature_detected!("dotprod");
    for i in 0..m {
        let (xr, yr) = (&xq[i * k..(i + 1) * k], &mut y[i * n..(i + 1) * n]);
        #[cfg(all(target_arch = "aarch64", feature = "dotprod"))]
        {
            if has_dot {
                unsafe { gemv_i8i8_dot_dotprod(xr, wtq, sx[i], sc, k, n, yr) };
            } else {
                gemv_i8i8_dot_scalar(xr, wtq, sx[i], sc, k, n, yr);
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "dotprod")))]
        gemv_i8i8_dot_scalar(xr, wtq, sx[i], sc, k, n, yr);
    }
    y
}

/// Relative L2 error `‖a-b‖/‖a‖`.
pub fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-9;
    num / den
}

// ───────────────────────────── Monarch ─────────────────────────────
//
// A Monarch matrix `M ∈ R^{n×n}` (n = m²) factors into two block-diagonal
// factors with a transpose (permutation) between — `M = L·P·R`. The dense
// entry factorizes as a *product* of the two block factors:
//
//     M[(l,s),(ri,j)] = w1[ri,j,l] · w2[l,s,ri]        (l,s,ri,j ∈ 0..m)
//
// so projecting a dense `W` onto the Monarch class is exactly `m²` independent
// **rank-1** SVDs — one per `(l,ri)` block — and the matmul is two batched
// block-diagonal contractions costing `2·m³ = 2·n^{1.5}` vs dense `n²`.
// (Dao et al. 2022, "Monarch: Expressive Structured Matrices".)

/// Largest `m` with `m² ≤ n` and `m² == n` — Monarch needs a perfect square.
pub fn monarch_blocks(n: usize) -> Option<usize> {
    let m = (n as f64).sqrt().round() as usize;
    if m * m == n && m > 0 { Some(m) } else { None }
}

/// Project a square dense `W[n,n]` (n = m²) onto the Monarch class via `m²`
/// rank-1 SVDs. `W` is in the matmul weight layout `[in, out]` (`w[in*n+out]`,
/// as [`dense_matmul`] consumes). Returns the two block-diagonal factors, each
/// flat `[m,m,m]`: `w1` indexed `[ri,j,l]`, `w2` indexed `[l,s,ri]`. Exact for
/// true-Monarch `W`.
pub fn monarch_project(w: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    let n = m * m;
    let mut w1 = vec![0f32; m * m * m]; // [ri, j, l]
    let mut w2 = vec![0f32; m * m * m]; // [l, s, ri]
    let mut blk = vec![0f32; m * m];
    for l in 0..m {
        for ri in 0..m {
            // B[s,j] = W[in=(ri,j), out=(l,s)] — the block coupling in-group ri
            // (index j) to out-group l (index s). Rank-1 in (s,j) ⇒ the factors.
            for s in 0..m {
                for j in 0..m {
                    blk[s * m + j] = w[(ri * m + j) * n + (l * m + s)];
                }
            }
            let (u, sv, v) = truncated_svd(&blk, m, m, 1); // rank-1: u[s], σ, v[j]
            for s in 0..m {
                w2[l * m * m + s * m + ri] = sv[0] * u[s];
            }
            for j in 0..m {
                w1[ri * m * m + j * m + l] = v[j];
            }
        }
    }
    (w1, w2)
}

/// Monarch matmul `y[batch,n] = x[batch,n]·M` via two block-diagonal
/// contractions (no dense `M` ever formed). `n = m²`; cost `2·batch·m³`.
pub fn monarch_matmul(x: &[f32], w1: &[f32], w2: &[f32], batch: usize, m: usize) -> Vec<f32> {
    let n = m * m;
    let mut y = vec![0f32; batch * n];
    let mut t = vec![0f32; m * m]; // t[ri, l]
    for b in 0..batch {
        // Step 1: per input-group `ri`, an m×m contraction j→l.
        for ri in 0..m {
            for l in 0..m {
                let mut acc = 0f32;
                for j in 0..m {
                    acc += x[b * n + ri * m + j] * w1[ri * m * m + j * m + l];
                }
                t[ri * m + l] = acc;
            }
        }
        // Step 2: per output-group `l`, an m×m contraction ri→s.
        for l in 0..m {
            for s in 0..m {
                let mut acc = 0f32;
                for ri in 0..m {
                    acc += t[ri * m + l] * w2[l * m * m + s * m + ri];
                }
                y[b * n + l * m + s] = acc;
            }
        }
    }
    y
}

/// Parameter count of a Monarch factorization of an `n×n` matrix (`2n^{1.5}`).
pub fn monarch_params(m: usize) -> usize {
    2 * m * m * m
}

// ───────────────────────────── Tucker ─────────────────────────────
//
// Tucker (HOSVD) compresses an N-way tensor as a small dense **core** contracted
// with a **factor matrix per mode**: `X ≈ G ×₀ U₀ ×₁ U₁ ×₂ U₂`. For a matrix
// this collapses to plain low-rank; it earns its keep on ≥3-way tensors — conv
// weights `[out,in,kh,kw]`, or a GEMM weight reshaped `[k, n₁, n₂]`. Factors are
// the leading left-singular vectors of each mode unfolding; the core is the
// original projected onto them. (Implemented for the 3-way case.)

/// Mode-`mode` unfolding of a 3-way tensor `dims=[d0,d1,d2]` (row-major) into a
/// matrix `[dims[mode], numel/dims[mode]]`, columns in natural order of the rest.
fn unfold3(x: &[f32], d: [usize; 3], mode: usize) -> (Vec<f32>, usize, usize) {
    let (d0, d1, d2) = (d[0], d[1], d[2]);
    let rows = d[mode];
    let cols = d0 * d1 * d2 / rows;
    let mut out = vec![0f32; rows * cols];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let v = x[(i0 * d1 + i1) * d2 + i2];
                let (r, c) = match mode {
                    0 => (i0, i1 * d2 + i2),
                    1 => (i1, i0 * d2 + i2),
                    _ => (i2, i0 * d1 + i1),
                };
                out[r * cols + c] = v;
            }
        }
    }
    (out, rows, cols)
}

/// HOSVD of a 3-way tensor `dims` to Tucker ranks `ranks`. Returns
/// `(core[r0*r1*r2], [U0,U1,U2])` with `Ui` flat `[dims[i]*ranks[i]]`. The core
/// is formed by **sequential mode products** `G = X ×₀ U0ᵀ ×₁ U1ᵀ ×₂ U2ᵀ` —
/// `O(r·numel)`, not the `O(numel·∏r)` full contraction.
pub fn tucker_hosvd(x: &[f32], dims: [usize; 3], ranks: [usize; 3]) -> (Vec<f32>, [Vec<f32>; 3]) {
    let r: [usize; 3] = [
        ranks[0].min(dims[0]).max(1),
        ranks[1].min(dims[1]).max(1),
        ranks[2].min(dims[2]).max(1),
    ];
    let (d0, d1, d2) = (dims[0], dims[1], dims[2]);
    let mut factors: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for mode in 0..3 {
        let (uf, rows, cols) = unfold3(x, dims, mode);
        let (u, _s, _v) = truncated_svd(&uf, rows, cols, r[mode]); // U: [rows*r]
        factors[mode] = u;
    }
    // ×₀ U0ᵀ : T0[j0, i1, i2] = Σ_i0 U0[i0,j0]·X[i0,i1,i2]   → [r0, d1, d2]
    let mut t0 = vec![0f32; r[0] * d1 * d2];
    for i0 in 0..d0 {
        for j0 in 0..r[0] {
            let u = factors[0][i0 * r[0] + j0];
            if u == 0.0 {
                continue;
            }
            let src = &x[i0 * d1 * d2..(i0 + 1) * d1 * d2];
            let dst = &mut t0[j0 * d1 * d2..(j0 + 1) * d1 * d2];
            for e in 0..d1 * d2 {
                dst[e] += u * src[e];
            }
        }
    }
    // ×₁ U1ᵀ : T1[j0, j1, i2] = Σ_i1 U1[i1,j1]·T0[j0,i1,i2]  → [r0, r1, d2]
    let mut t1 = vec![0f32; r[0] * r[1] * d2];
    for j0 in 0..r[0] {
        for i1 in 0..d1 {
            for j1 in 0..r[1] {
                let u = factors[1][i1 * r[1] + j1];
                if u == 0.0 {
                    continue;
                }
                let src = &t0[(j0 * d1 + i1) * d2..(j0 * d1 + i1 + 1) * d2];
                let dst = &mut t1[(j0 * r[1] + j1) * d2..(j0 * r[1] + j1 + 1) * d2];
                for i2 in 0..d2 {
                    dst[i2] += u * src[i2];
                }
            }
        }
    }
    // ×₂ U2ᵀ : G[j0,j1,j2] = Σ_i2 U2[i2,j2]·T1[j0,j1,i2]     → [r0, r1, r2]
    let mut core = vec![0f32; r[0] * r[1] * r[2]];
    for a in 0..r[0] * r[1] {
        for i2 in 0..d2 {
            let v = t1[a * d2 + i2];
            if v == 0.0 {
                continue;
            }
            for j2 in 0..r[2] {
                core[a * r[2] + j2] += v * factors[2][i2 * r[2] + j2];
            }
        }
    }
    (core, factors)
}

/// Reconstruct `X̂ = G ×₀ U0 ×₁ U1 ×₂ U2` by sequential mode products.
pub fn tucker_reconstruct(
    core: &[f32],
    dims: [usize; 3],
    ranks: [usize; 3],
    factors: &[Vec<f32>; 3],
) -> Vec<f32> {
    let r = [
        ranks[0].min(dims[0]).max(1),
        ranks[1].min(dims[1]).max(1),
        ranks[2].min(dims[2]).max(1),
    ];
    let (d0, d1, d2) = (dims[0], dims[1], dims[2]);
    // ×₀ U0 : R0[i0,j1,j2] = Σ_j0 U0[i0,j0]·G[j0,j1,j2]  → [d0, r1, r2]
    let mut r0 = vec![0f32; d0 * r[1] * r[2]];
    for i0 in 0..d0 {
        for j0 in 0..r[0] {
            let u = factors[0][i0 * r[0] + j0];
            if u == 0.0 {
                continue;
            }
            let src = &core[j0 * r[1] * r[2]..(j0 + 1) * r[1] * r[2]];
            let dst = &mut r0[i0 * r[1] * r[2]..(i0 + 1) * r[1] * r[2]];
            for e in 0..r[1] * r[2] {
                dst[e] += u * src[e];
            }
        }
    }
    // ×₁ U1 : R1[i0,i1,j2] = Σ_j1 U1[i1,j1]·R0[i0,j1,j2]  → [d0, d1, r2]
    let mut r1 = vec![0f32; d0 * d1 * r[2]];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for j1 in 0..r[1] {
                let u = factors[1][i1 * r[1] + j1];
                if u == 0.0 {
                    continue;
                }
                let src = &r0[(i0 * r[1] + j1) * r[2]..(i0 * r[1] + j1 + 1) * r[2]];
                let dst = &mut r1[(i0 * d1 + i1) * r[2]..(i0 * d1 + i1 + 1) * r[2]];
                for j2 in 0..r[2] {
                    dst[j2] += u * src[j2];
                }
            }
        }
    }
    // ×₂ U2 : X̂[i0,i1,i2] = Σ_j2 U2[i2,j2]·R1[i0,i1,j2]  → [d0, d1, d2]
    let mut out = vec![0f32; d0 * d1 * d2];
    for a in 0..d0 * d1 {
        for j2 in 0..r[2] {
            let v = r1[a * r[2] + j2];
            if v == 0.0 {
                continue;
            }
            for i2 in 0..d2 {
                out[a * d2 + i2] += v * factors[2][i2 * r[2] + j2];
            }
        }
    }
    out
}

/// Parameter count of a 3-way Tucker decomposition (core + three factors).
pub fn tucker_params(dims: [usize; 3], ranks: [usize; 3]) -> usize {
    ranks[0] * ranks[1] * ranks[2] + dims[0] * ranks[0] + dims[1] * ranks[1] + dims[2] * ranks[2]
}

// ─────────────────────────── Tensor-Train ───────────────────────────
//
// TT (Oseledets 2011) writes an N-way tensor as a chain of 3-way **cores**
// `Gₖ[rₖ₋₁, dₖ, rₖ]` (with `r₀ = r_N = 1`): `X[i₀..i_{N-1}] = G₀[i₀]·G₁[i₁]···`.
// TT-SVD builds it by a left-to-right sweep of truncated SVDs on the sequential
// reshape. Param count is `Σ rₖ₋₁·dₖ·rₖ` — linear in the number of modes for
// fixed bond rank, so it compresses hardest when a big dim is factored into many
// small modes (the "tensorizing neural nets" trick, Novikov et al. 2015).

/// One TT core: `(r_prev, d, r_next, flat[r_prev*d*r_next])`, `[rp,ik,rn]`-major.
pub type TtCore = (usize, usize, usize, Vec<f32>);

/// TT-SVD of an N-way tensor `dims` (row-major) with bond rank capped at
/// `max_rank`. Returns the core chain (`r0 = rlast = 1`).
pub fn tt_svd(x: &[f32], dims: &[usize], max_rank: usize) -> Vec<TtCore> {
    let nd = dims.len();
    let numel: usize = dims.iter().product();
    let mut cores: Vec<TtCore> = Vec::with_capacity(nd);
    let mut c = x.to_vec();
    let mut r_prev = 1usize;
    let mut elems = numel; // elements currently in `c`
    for k in 0..nd {
        let dk = dims[k];
        let rows = r_prev * dk;
        let cols = elems / rows;
        if k == nd - 1 {
            // Last core: bond rank collapses to 1; `c` is already [r_prev*dk, 1].
            cores.push((r_prev, dk, 1, c.clone()));
            break;
        }
        let rk = max_rank.min(rows).min(cols).max(1);
        let (u, sv, v) = truncated_svd(&c, rows, cols, rk); // U:[rows*rk], V:[cols*rk]
        cores.push((r_prev, dk, rk, u)); // U reshaped [r_prev, dk, rk]
        // c_next = diag(σ)·Vᵀ  →  [rk, cols]
        let mut cn = vec![0f32; rk * cols];
        for a in 0..rk {
            let s = sv[a];
            for j in 0..cols {
                cn[a * cols + j] = s * v[j * rk + a];
            }
        }
        c = cn;
        r_prev = rk;
        elems = rk * cols;
    }
    cores
}

/// Reconstruct the full tensor (flat, row-major) from a TT core chain.
pub fn tt_reconstruct(cores: &[TtCore]) -> Vec<f32> {
    // acc: [prod(d0..dk), r_k]; start from core 0 (r_prev = 1).
    let (_, d0, r1, ref g0) = cores[0];
    let mut acc = g0.clone(); // [d0, r1]
    let mut p = d0;
    let mut rank = r1;
    for core in &cores[1..] {
        let (rp, dk, rn, ref g) = *core;
        debug_assert_eq!(rp, rank);
        let mut next = vec![0f32; p * dk * rn];
        for pi in 0..p {
            for ik in 0..dk {
                for rnn in 0..rn {
                    let mut s = 0f32;
                    for r in 0..rank {
                        s += acc[pi * rank + r] * g[(r * dk + ik) * rn + rnn];
                    }
                    next[(pi * dk + ik) * rn + rnn] = s;
                }
            }
        }
        acc = next;
        p *= dk;
        rank = rn;
    }
    acc // [numel, 1]
}

/// Parameter count of a TT core chain (`Σ rₖ₋₁·dₖ·rₖ`).
pub fn tt_params(cores: &[TtCore]) -> usize {
    cores.iter().map(|&(rp, d, rn, _)| rp * d * rn).sum()
}

// ─────────────────────────── attention kernels ───────────────────────────
//
// The fused-op IO analysis (`shapes::fused_io_report`) showed attention's scores
// matrix is O(s²) on-chip traffic. The NAIVE kernel materializes the full [s,s]
// scores (and its exp) per head; FLASH tiles it with an online softmax so only a
// [bq,bk] tile is ever live — bounding the on-chip term, the whole reason
// flash-attention exists. Both are single-precision references (q,k [bh,s,d];
// v,out [bh,s,dv]); `bh` = batch·heads folded, `causal` masks j>i.

/// SIMD dot product `Σ a·b` over `n` (NEON on aarch64+neon, scalar else).
#[inline]
fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    unsafe {
        use std::arch::aarch64::*;
        let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let chunks = n / 8;
        for c in 0..chunks {
            let p = c * 8;
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(a.as_ptr().add(p)),
                vld1q_f32(b.as_ptr().add(p)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(a.as_ptr().add(p + 4)),
                vld1q_f32(b.as_ptr().add(p + 4)),
            );
        }
        let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
        for i in chunks * 8..n {
            s += a[i] * b[i];
        }
        s
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        (0..n).map(|i| a[i] * b[i]).sum()
    }
}

/// SIMD `acc[c] = acc[c]·corr + p·v[c]` over `n` (the flash rescale-accumulate;
/// `corr=1` is the no-rescale / naive `acc += p·v` case).
#[inline]
fn axpy_corr(acc: &mut [f32], corr: f32, p: f32, v: &[f32], n: usize) {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    unsafe {
        use std::arch::aarch64::*;
        let (vc, vp) = (vdupq_n_f32(corr), vdupq_n_f32(p));
        let chunks = n / 4;
        for c in 0..chunks {
            let off = c * 4;
            let va = vld1q_f32(acc.as_ptr().add(off));
            let vv = vld1q_f32(v.as_ptr().add(off));
            vst1q_f32(
                acc.as_mut_ptr().add(off),
                vfmaq_f32(vmulq_f32(va, vc), vp, vv),
            );
        }
        for i in chunks * 4..n {
            acc[i] = acc[i] * corr + p * v[i];
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    for i in 0..n {
        acc[i] = acc[i] * corr + p * v[i];
    }
}

/// Naive SDPA: materialize the full `[s,s]` scores + softmax per head, then `·V` —
/// the standard reference (what frameworks do, and what OOMs at long context). The
/// O(s²) on-chip term made explicit, the baseline flash beats. NEON dot + AV so the
/// speed comparison isolates the algorithm, not the vectorization.
pub fn attention_naive(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    s: usize,
    d: usize,
    dv: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; bh * s * dv];
    let mut scores = vec![0f32; s * s]; // the [s,s] flash never allocates
    for h in 0..bh {
        let (qb, kb) = (
            &q[h * s * d..(h + 1) * s * d],
            &k[h * s * d..(h + 1) * s * d],
        );
        let (vb, ob) = (
            &v[h * s * dv..(h + 1) * s * dv],
            &mut out[h * s * dv..(h + 1) * s * dv],
        );
        for i in 0..s {
            let qrow = &qb[i * d..(i + 1) * d];
            let kmax = if causal { i } else { s - 1 };
            for j in 0..=kmax {
                scores[i * s + j] = dot_f32(qrow, &kb[j * d..(j + 1) * d], d) * scale;
            }
            let m = scores[i * s..i * s + kmax + 1]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for sj in scores[i * s..i * s + kmax + 1].iter_mut() {
                let p = (*sj - m).exp();
                *sj = p;
                sum += p;
            }
            let orow = &mut ob[i * dv..(i + 1) * dv];
            let inv = 1.0 / sum;
            for j in 0..=kmax {
                axpy_corr(
                    orow,
                    1.0,
                    scores[i * s + j] * inv,
                    &vb[j * dv..(j + 1) * dv],
                    dv,
                );
            }
        }
    }
    out
}

/// Flash SDPA: tile queries `[bq]` × keys `[bk]` with an ONLINE softmax (running
/// max/sum/accumulator per query row), so the full `[s,s]` scores are never
/// materialized — only a `[bq,bk]` tile is live. Mathematically equal to
/// [`attention_naive`] (to fp rounding). This is the kernel that bounds the O(s²)
/// on-chip traffic `fused_io_report` flagged.
pub fn attention_flash(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    s: usize,
    d: usize,
    dv: usize,
    causal: bool,
    scale: f32,
    bq: usize,
    bk: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; bh * s * dv];
    for h in 0..bh {
        let (qb, kb) = (
            &q[h * s * d..(h + 1) * s * d],
            &k[h * s * d..(h + 1) * s * d],
        );
        let (vb, ob) = (
            &v[h * s * dv..(h + 1) * s * dv],
            &mut out[h * s * dv..(h + 1) * s * dv],
        );
        let mut qi = 0;
        while qi < s {
            let qe = (qi + bq).min(s);
            let nq = qe - qi;
            let mut m = vec![f32::NEG_INFINITY; nq]; // running max per query row
            let mut l = vec![0f32; nq]; // running softmax denom
            let mut acc = vec![0f32; nq * dv]; // running output
            let mut kj = 0;
            while kj < s {
                let ke = (kj + bk).min(s);
                if causal && kj > qe - 1 {
                    break; // whole key tile is in the future for every query in this block
                }
                for ii in 0..nq {
                    let qrow = &qb[(qi + ii) * d..(qi + ii + 1) * d];
                    let arow = &mut acc[ii * dv..(ii + 1) * dv];
                    for j in kj..ke {
                        if causal && j > qi + ii {
                            continue;
                        }
                        let sij = dot_f32(qrow, &kb[j * d..(j + 1) * d], d) * scale; // SIMD dot
                        let vrow = &vb[j * dv..(j + 1) * dv];
                        if sij > m[ii] {
                            // new running max → rescale the accumulator (the only exp-heavy path)
                            let corr = (m[ii] - sij).exp(); // exp(-inf)=0 on the first key
                            l[ii] = l[ii] * corr + 1.0; // p = exp(sij - sij) = 1
                            axpy_corr(arow, corr, 1.0, vrow, dv);
                            m[ii] = sij;
                        } else {
                            // max unchanged → no rescale (corr=1); skip the correction exp
                            let p = (sij - m[ii]).exp();
                            l[ii] += p;
                            axpy_corr(arow, 1.0, p, vrow, dv);
                        }
                    }
                }
                kj = ke;
            }
            for ii in 0..nq {
                let li = l[ii].max(1e-20);
                let (arow, orow) = (
                    &acc[ii * dv..(ii + 1) * dv],
                    &mut ob[(qi + ii) * dv..(qi + ii + 1) * dv],
                );
                for c in 0..dv {
                    orow[c] = arow[c] / li;
                }
            }
            qi = qe;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_recovers_low_rank() {
        let (k, n) = (64usize, 48usize);
        let a: Vec<f32> = (0..k * 2).map(|i| (i * 7 % 13) as f32 - 6.0).collect();
        let b: Vec<f32> = (0..2 * n).map(|i| (i * 5 % 11) as f32 - 5.0).collect();
        let mut w = vec![0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut s = 0f32;
                for p in 0..2 {
                    s += a[i * 2 + p] * b[p * n + j];
                }
                w[i * n + j] = s;
            }
        }
        let (u, v, r) = factorize(&w, k, n, 4);
        let recon = dense_matmul(&u, &v, k, r, n);
        assert!(
            rel_err(&w, &recon) < 1e-3,
            "rank-2 recon err {}",
            rel_err(&w, &recon)
        );
    }

    // Deterministic pseudo-random fill in [-1,1].
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as i64 as f64 / i64::MAX as f64) as f32
            })
            .collect()
    }

    #[test]
    fn monarch_operator_matches_dense() {
        // The block factors imply a dense M; the fast operator must equal x·M.
        let m = 4;
        let n = m * m;
        let (w1, w2) = (fill(m * m * m, 11), fill(m * m * m, 22));
        let mut dense = vec![0f32; n * n];
        for l in 0..m {
            for s in 0..m {
                for ri in 0..m {
                    for j in 0..m {
                        // weight layout [in=(ri,j), out=(l,s)] to match dense_matmul.
                        dense[(ri * m + j) * n + (l * m + s)] =
                            w1[ri * m * m + j * m + l] * w2[l * m * m + s * m + ri];
                    }
                }
            }
        }
        let batch = 3;
        let x = fill(batch * n, 33);
        let fast = monarch_matmul(&x, &w1, &w2, batch, m);
        let slow = dense_matmul(&x, &dense, batch, n, n);
        assert!(
            rel_err(&slow, &fast) < 1e-5,
            "monarch operator err {}",
            rel_err(&slow, &fast)
        );
    }

    #[test]
    fn monarch_projects_true_monarch_exactly() {
        // A genuinely-Monarch W → projection + operator reproduce x·W to ~0.
        let m = 4;
        let n = m * m;
        let (w1, w2) = (fill(m * m * m, 44), fill(m * m * m, 55));
        let mut dense = vec![0f32; n * n];
        for l in 0..m {
            for s in 0..m {
                for ri in 0..m {
                    for j in 0..m {
                        // weight layout [in=(ri,j), out=(l,s)] to match dense_matmul.
                        dense[(ri * m + j) * n + (l * m + s)] =
                            w1[ri * m * m + j * m + l] * w2[l * m * m + s * m + ri];
                    }
                }
            }
        }
        let (p1, p2) = monarch_project(&dense, m);
        let x = fill(2 * n, 66);
        let got = monarch_matmul(&x, &p1, &p2, 2, m);
        let want = dense_matmul(&x, &dense, 2, n, n);
        assert!(
            rel_err(&want, &got) < 1e-3,
            "monarch projection err {}",
            rel_err(&want, &got)
        );
    }

    #[test]
    fn tucker_recovers_low_multilinear_rank() {
        // Build X with multilinear rank (2,2,2); HOSVD at ranks ≥ that is exact.
        let dims = [6usize, 5, 4];
        let mr = [2usize, 2, 2];
        let core = fill(mr[0] * mr[1] * mr[2], 7);
        let f: [Vec<f32>; 3] = [
            fill(dims[0] * mr[0], 8),
            fill(dims[1] * mr[1], 9),
            fill(dims[2] * mr[2], 10),
        ];
        let x = tucker_reconstruct(&core, dims, mr, &f);
        let (g, u) = tucker_hosvd(&x, dims, [3, 3, 3]);
        let recon = tucker_reconstruct(&g, dims, [3, 3, 3], &u);
        assert!(
            rel_err(&x, &recon) < 1e-3,
            "tucker recon err {}",
            rel_err(&x, &recon)
        );
    }

    #[test]
    fn gemv_dot_kernels_match_dense() {
        let (k, n) = (256usize, 192usize);
        let w = fill(k * n, 71); // [k,n]
        let x = fill(k, 72);
        // Reference: y[j] = Σ_p x[p]·W[p,j]  (a 1×k · k×n dense matmul).
        let want = dense_matmul(&x, &w, 1, k, n);
        // f32 weight-stationary dot GEMV on the transposed weight → exact.
        let wt = transpose(&w, k, n);
        let gf = gemv_f32_dot(&x, &wt, k, n);
        assert!(
            rel_err(&want, &gf) < 1e-4,
            "f32 GEMV err {}",
            rel_err(&want, &gf)
        );
        let mut gf_scalar = vec![0f32; n];
        gemv_f32_dot_scalar(&x, &wt, k, n, &mut gf_scalar);
        assert!(
            rel_err(&gf_scalar, &gf) < 1e-5,
            "neon vs scalar f32 {}",
            rel_err(&gf_scalar, &gf)
        );
        // int8 kernel → within per-channel quant error; and NEON == scalar.
        let (wtq, sc) = quantize_cols_t(&w, k, n);
        let gi = gemv_i8_dot(&x, &wtq, &sc, k, n);
        assert!(
            rel_err(&want, &gi) < 0.05,
            "int8 W8A16 GEMV err {}",
            rel_err(&want, &gi)
        );
        let mut scalar = vec![0f32; n];
        gemv_i8_dot_scalar(&x, &wtq, &sc, k, n, &mut scalar);
        assert!(
            rel_err(&scalar, &gi) < 1e-5,
            "neon vs scalar int8 {}",
            rel_err(&scalar, &gi)
        );
        // W8A8 (SDOT path) — both operands int8; within combined quant error, and
        // the hardware dotprod path must match its own scalar reference exactly.
        let (xq, sx) = quantize_row_i8(&x);
        let g88 = gemv_i8i8_dot(&xq, &wtq, sx, &sc, k, n);
        assert!(
            rel_err(&want, &g88) < 0.08,
            "int8 W8A8 GEMV err {}",
            rel_err(&want, &g88)
        );
        let mut s88 = vec![0f32; n];
        gemv_i8i8_dot_scalar(&xq, &wtq, sx, &sc, k, n, &mut s88);
        assert!(
            rel_err(&s88, &g88) < 1e-6,
            "sdot vs scalar W8A8 {}",
            rel_err(&s88, &g88)
        );
    }

    #[test]
    fn matmul_kernels_match_dense() {
        let (m, k, n) = (4usize, 128usize, 96usize);
        let w = fill(k * n, 81);
        let x = fill(m * k, 82);
        let want = dense_matmul(&x, &w, m, k, n); // [m,n] f32 reference
        let wt = transpose(&w, k, n);
        let (wtq, sc) = quantize_cols_t(&w, k, n);
        // f32 matmul via the kernel path → exact.
        assert!(rel_err(&want, &matmul_f32(&x, &wt, m, k, n)) < 1e-4);
        // W8A16 → weight-quant error only.
        let y16 = matmul_w8a16(&x, &wtq, &sc, m, k, n);
        assert!(
            rel_err(&want, &y16) < 0.05,
            "W8A16 err {}",
            rel_err(&want, &y16)
        );
        // W8A8 → weight + activation quant error.
        let (xq, sx) = quantize_rows_i8(&x, m, k);
        let y8 = matmul_w8a8(&xq, &sx, &wtq, &sc, m, k, n);
        assert!(
            rel_err(&want, &y8) < 0.08,
            "W8A8 err {}",
            rel_err(&want, &y8)
        );
    }

    #[test]
    fn flash_attention_matches_naive() {
        // Random Q/K/V; flash (tiled online softmax) must equal naive (materialized
        // scores) to fp rounding — both causal and full, non-square d≠dv, tiles that
        // don't divide s.
        let (bh, s, d, dv) = (2usize, 20usize, 8usize, 6usize);
        let q = fill(bh * s * d, 1);
        let k = fill(bh * s * d, 2);
        let v = fill(bh * s * dv, 3);
        let scale = 1.0 / (d as f32).sqrt();
        for causal in [false, true] {
            let want = attention_naive(&q, &k, &v, bh, s, d, dv, causal, scale);
            for (bq, bk) in [(4usize, 4usize), (7, 5), (s, s)] {
                let got = attention_flash(&q, &k, &v, bh, s, d, dv, causal, scale, bq, bk);
                assert!(
                    rel_err(&want, &got) < 1e-5,
                    "flash≠naive causal={causal} bq={bq} bk={bk} err {}",
                    rel_err(&want, &got)
                );
            }
        }
    }

    #[test]
    fn tt_recovers_low_bond_rank() {
        // A tensor generated from bond-rank-2 cores; TT-SVD at rank ≥2 recovers it.
        let dims = [4usize, 5, 3, 4];
        let cores: Vec<TtCore> = vec![
            (1, 4, 2, fill(1 * 4 * 2, 1)),
            (2, 5, 2, fill(2 * 5 * 2, 2)),
            (2, 3, 2, fill(2 * 3 * 2, 3)),
            (2, 4, 1, fill(2 * 4 * 1, 4)),
        ];
        let x = tt_reconstruct(&cores);
        let got = tt_svd(&x, &dims, 3);
        let recon = tt_reconstruct(&got);
        assert!(
            rel_err(&x, &recon) < 1e-3,
            "tt recon err {}",
            rel_err(&x, &recon)
        );
    }
}
