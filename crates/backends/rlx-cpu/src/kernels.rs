// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SIMD kernels for fused operations.
//!
//! These are the production fused CPU kernels.
//! Each kernel processes data in-place or into a pre-allocated output buffer
//! (from the arena). No allocation.

use crate::pool;

// ── NEON vectorized exp ─────────────────────────────────────────────────

/// NEON vectorized exp(x) for 4 floats. Range reduction + 6th-order Taylor.
/// Max relative error: ~2e-7 across [-87, 88].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn neon_exp4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vmaxq_f32(x, vdupq_n_f32(-87.3));
    let x = vminq_f32(x, vdupq_n_f32(88.7));
    let inv_ln2 = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693_145_75);
    let ln2_lo = vdupq_n_f32(1.428_606_8e-6);
    let n = vrndnq_f32(vmulq_f32(x, inv_ln2));
    let r = vfmsq_f32(vfmsq_f32(x, n, ln2_hi), n, ln2_lo);
    let c1 = vdupq_n_f32(1.0);
    let mut p = vdupq_n_f32(0.001_388_888_9);
    p = vfmaq_f32(vdupq_n_f32(0.008_333_334), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.041_666_668), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.166_666_67), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(c1, p, r);
    let ni = vcvtq_s32_f32(n);
    vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(ni, 23)))
}

/// AVX2+FMA vectorised exp(x) for 8 floats. Same range reduction +
/// 6th-order Taylor polynomial as `neon_exp4`. Max relative error
/// stays in the ~2e-7 range. Runtime-dispatch via `is_x86_feature_detected`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// Cody–Waite ln2 split (`ln2_hi`/`ln2_lo`) and the Taylor coefficients below are
// written at full precision on purpose — each rounds to the intended nearest-f32,
// and truncating the literals would move the hand-tuned polynomial. Silence the
// pedantic precision lint rather than degrade the approximation.
#[allow(unsafe_op_in_unsafe_fn, clippy::excessive_precision)]
pub unsafe fn avx2_exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.3));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.7));
    let inv_ln2 = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693145751953125);
    let ln2_lo = _mm256_set1_ps(1.428606765330187e-6);
    // n = round(x / ln2)  (round-to-nearest-even)
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x, inv_ln2,
    ));
    // r = x − n·ln2_hi − n·ln2_lo
    let r = _mm256_fnmadd_ps(n, ln2_lo, _mm256_fnmadd_ps(n, ln2_hi, x));
    let c1 = _mm256_set1_ps(1.0);
    let mut p = _mm256_set1_ps(0.001388888888888889);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.008333333333333333));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.041666666666666664));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.16666666666666666));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c1);
    // 2^n via integer-bias trick on the f32 exponent field.
    let ni = _mm256_cvtps_epi32(n);
    let shifted = _mm256_slli_epi32::<23>(ni);
    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), shifted))
}

// ── Fused bias + GELU ───────────────────────────────────────────────────

/// Fused bias addition + GELU activation on a [m, n] buffer.
/// Uses Abramowitz & Stegun erf approximation with NEON exp.
#[cfg(target_arch = "aarch64")]
pub fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    use std::arch::aarch64::*;
    let chunks = n / 4;
    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let inv_sqrt2 = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);
        let p = vdupq_n_f32(0.3275911);
        let a1 = vdupq_n_f32(0.254_829_6);
        let a2 = vdupq_n_f32(-0.284_496_72);
        let a3 = vdupq_n_f32(1.421_413_8);
        let a4 = vdupq_n_f32(-1.453_152_1);
        let a5 = vdupq_n_f32(1.061_405_4);
        let neg_one = vdupq_n_f32(-1.0);
        let zero = vdupq_n_f32(0.0);

        for row in 0..m {
            let base = row * n;
            for c in 0..chunks {
                let off = base + c * 4;
                let ptr = data.as_mut_ptr().add(off);
                let x = vaddq_f32(vld1q_f32(ptr), vld1q_f32(bias.as_ptr().add(c * 4)));
                let erf_arg = vmulq_f32(x, inv_sqrt2);
                let xa = vabsq_f32(erf_arg);
                let sign = vbslq_f32(vcgeq_f32(erf_arg, zero), one, neg_one);
                let denom = vfmaq_f32(one, p, xa);
                let t = vdivq_f32(one, denom);
                let mut y = a5;
                y = vfmaq_f32(a4, y, t);
                y = vfmaq_f32(a3, y, t);
                y = vfmaq_f32(a2, y, t);
                y = vfmaq_f32(a1, y, t);
                y = vmulq_f32(y, t);
                let exp_val = neon_exp4(vnegq_f32(vmulq_f32(xa, xa)));
                let erf_val = vmulq_f32(sign, vfmsq_f32(one, y, exp_val));
                vst1q_f32(ptr, vmulq_f32(x, vmulq_f32(half, vaddq_f32(one, erf_val))));
            }
            for i in (chunks * 4)..n {
                let x = data[base + i] + bias[i];
                data[base + i] = scalar_gelu(x);
            }
        }
    }
}

/// Erf-GELU + bias via AVX2+FMA.
///
/// # Safety
/// Caller has checked `is_x86_feature_detected!("avx2")` and `"fma"`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::excessive_precision)]
unsafe fn bias_gelu_avx2(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 8;
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let inv_sqrt2 = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
        let p = _mm256_set1_ps(0.3275911);
        let a1 = _mm256_set1_ps(0.254829592);
        let a2 = _mm256_set1_ps(-0.284496736);
        let a3 = _mm256_set1_ps(1.421413741);
        let a4 = _mm256_set1_ps(-1.453152027);
        let a5 = _mm256_set1_ps(1.061405429);
        let neg_one = _mm256_set1_ps(-1.0);
        let zero = _mm256_set1_ps(0.0);
        // Sign bit mask for fabs via AND with 0x7fffffff.
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));

        for row in 0..m {
            let base = row * n;
            for c in 0..chunks {
                let off = base + c * 8;
                let ptr = data.as_mut_ptr().add(off);
                let x = _mm256_add_ps(
                    _mm256_loadu_ps(ptr),
                    _mm256_loadu_ps(bias.as_ptr().add(c * 8)),
                );
                let erf_arg = _mm256_mul_ps(x, inv_sqrt2);
                let xa = _mm256_and_ps(erf_arg, abs_mask);
                // sign = (erf_arg >= 0) ? 1 : -1
                let ge0 = _mm256_cmp_ps::<_CMP_GE_OQ>(erf_arg, zero);
                let sign = _mm256_blendv_ps(neg_one, one, ge0);
                let denom = _mm256_fmadd_ps(p, xa, one);
                let t = _mm256_div_ps(one, denom);
                let mut y = a5;
                y = _mm256_fmadd_ps(y, t, a4);
                y = _mm256_fmadd_ps(y, t, a3);
                y = _mm256_fmadd_ps(y, t, a2);
                y = _mm256_fmadd_ps(y, t, a1);
                y = _mm256_mul_ps(y, t);
                let exp_val = avx2_exp8(_mm256_sub_ps(zero, _mm256_mul_ps(xa, xa)));
                // erf = sign * (1 - y*exp(-xa^2))
                let erf_val = _mm256_mul_ps(sign, _mm256_fnmadd_ps(y, exp_val, one));
                _mm256_storeu_ps(
                    ptr,
                    _mm256_mul_ps(x, _mm256_mul_ps(half, _mm256_add_ps(one, erf_val))),
                );
            }
            for i in (chunks * 8)..n {
                let x = data[base + i] + bias[i];
                data[base + i] = scalar_gelu(x);
            }
        }
    }
}

/// Portable bias + erf-GELU. Shared by the non-SIMD targets and by x86-64's
/// runtime fallback, so one binary covers AVX2 hosts and pre-AVX ones
/// (every Atom through Tremont) without recompiling.
#[cfg(not(target_arch = "aarch64"))]
fn bias_gelu_scalar(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for row in 0..m {
        let base = row * n;
        for i in 0..n {
            let x = data[base + i] + bias[i];
            data[base + i] = scalar_gelu(x);
        }
    }
}

/// x86-64 bias + GELU. Dispatched on CPUID at runtime rather than on
/// `cfg(target_feature)`: a compile-time gate meant the portable build lost
/// the AVX2 kernel entirely and the only way to get it was a
/// `-C target-cpu=native` artifact that SIGILLs on an Atom.
#[cfg(target_arch = "x86_64")]
pub fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both feature bits just checked.
        unsafe { bias_gelu_avx2(data, bias, m, n) };
        return;
    }
    bias_gelu_scalar(data, bias, m, n);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    bias_gelu_scalar(data, bias, m, n);
}

/// SwiGLU over `outer` rows: `out[i] = up[i] · silu(gate[i])`, where each input
/// row holds the two halves concatenated (`gate_first` says which comes first).
///
/// Exists because both `Thunk::FusedSwiGLU` execution paths had this loop
/// inlined, byte-identically, with a libm `expf` CALL PER ELEMENT — on every
/// architecture, not just the ones without SIMD. This is the FFN activation of
/// every modern transformer, so it was scalar on Apple Silicon and AVX2 x86
/// too. Hoisting the `gate_first` branch out of the inner loop leaves two
/// contiguous runs and a body of pure float ops over
/// `vmath::exp_poly` (crate-private), which auto-vectorizes to NEON on
/// baseline SSE2 on x86.
///
/// Matches `Activation::Silu`, which already uses a polynomial sigmoid on
/// every arch, so this makes the fused and unfused paths agree rather than
/// introducing a new approximation.
pub fn swiglu_rows(inp: &[f32], out: &mut [f32], outer: usize, n: usize, gate_first: bool) {
    debug_assert!(inp.len() >= outer * 2 * n && out.len() >= outer * n);
    for o in 0..outer {
        let in_row = &inp[o * 2 * n..(o + 1) * 2 * n];
        let out_row = &mut out[o * n..(o + 1) * n];
        let (gate, up) = if gate_first {
            (&in_row[..n], &in_row[n..])
        } else {
            (&in_row[n..], &in_row[..n])
        };
        for i in 0..n {
            let g = gate[i];
            out_row[i] = up[i] * (g / (1.0 + crate::vmath::exp_poly(-g)));
        }
    }
}

/// Parallel bias + GELU across thread pool.
pub fn par_bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let cfg = crate::config::RuntimeConfig::global();
    if m * n < cfg.par_threshold || m < cfg.min_rows_per_thread {
        bias_gelu(data, bias, m, n);
        return;
    }
    let data_ptr = data.as_mut_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    pool::par_for(m, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let d = std::slice::from_raw_parts_mut((data_ptr as *mut f32).add(off * n), cnt * n);
        let b = std::slice::from_raw_parts(bias_ptr as *const f32, n);
        bias_gelu(d, b, cnt, n);
    });
}

// ── Fused SiLU ──────────────────────────────────────────────────────────

/// SiLU (Swish) in-place: x / (1 + exp(-x))
#[cfg(target_arch = "aarch64")]
pub fn silu_inplace(data: &mut [f32]) {
    use std::arch::aarch64::*;
    let chunks = data.len() / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for c in 0..chunks {
            let ptr = data.as_mut_ptr().add(c * 4);
            let x = vld1q_f32(ptr);
            let exp_neg = neon_exp4(vnegq_f32(x));
            let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg));
            vst1q_f32(ptr, vmulq_f32(x, sigmoid));
        }
    }
    for i in (chunks * 4)..data.len() {
        let x = data[i];
        data[i] = x / (1.0 + (-x).exp());
    }
}

/// SiLU via AVX2+FMA. Caller must have checked `is_x86_feature_detected`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silu_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = data.len() / 8;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_set1_ps(0.0);
    for c in 0..chunks {
        let off = c * 8;
        let ptr = data.as_mut_ptr().add(off);
        let x = _mm256_loadu_ps(ptr);
        // silu(x) = x / (1 + exp(-x))
        let neg_x = _mm256_sub_ps(zero, x);
        let denom = _mm256_add_ps(one, avx2_exp8(neg_x));
        _mm256_storeu_ps(ptr, _mm256_div_ps(x, denom));
    }
    for i in (chunks * 8)..data.len() {
        let x = data[i];
        data[i] = x / (1.0 + (-x).exp());
    }
}

#[cfg(target_arch = "x86_64")]
pub fn silu_inplace(data: &mut [f32]) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { silu_inplace_avx2(data) };
        return;
    }
    // Same sigmoid form as `silu_inplace_avx2`, over `exp_poly` so the loop
    // vectorizes to baseline SSE2 instead of calling libm per element.
    for v in data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + crate::vmath::exp_poly(-x));
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn silu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + crate::vmath::exp_poly(-x));
    }
}

// ── LayerNorm (2-pass) ──────────────────────────────────────────────────

/// Single-row LayerNorm: out = (x - mean) * inv_std * gamma + beta.
/// 2-pass: compute mean+variance (E\[x²\]-E\[x\]²), then normalize.
#[cfg(target_arch = "aarch64")]
pub fn layer_norm_row(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;
    let inv_hf = 1.0 / h as f32;
    let chunks = h / 4;
    unsafe {
        // Pass 1: mean.
        let mut vsum = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let x = vld1q_f32(input.as_ptr().add(c * 4));
            vsum = vaddq_f32(vsum, x);
        }
        let mut sum = vaddvq_f32(vsum);
        for i in (chunks * 4)..h {
            sum += input[i];
        }
        let mean = sum * inv_hf;
        let vmean = vdupq_n_f32(mean);
        // Pass 2: var = mean((x − mean)²). Two-pass avoids the one-pass
        // E[x²]−mean² cancellation that corrupts rows with a large DC offset.
        let mut vdev = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let d = vsubq_f32(vld1q_f32(input.as_ptr().add(c * 4)), vmean);
            vdev = vfmaq_f32(vdev, d, d);
        }
        let mut dev = vaddvq_f32(vdev);
        for i in (chunks * 4)..h {
            let d = input[i] - mean;
            dev += d * d;
        }
        let var = (dev * inv_hf).max(0.0);
        let inv = 1.0 / (var + eps).sqrt();
        let vinv = vdupq_n_f32(inv);
        for c in 0..chunks {
            let off = c * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let norm = vmulq_f32(vsubq_f32(x, vmean), vinv);
            vst1q_f32(
                output.as_mut_ptr().add(off),
                vfmaq_f32(
                    vld1q_f32(beta.as_ptr().add(off)),
                    norm,
                    vld1q_f32(gamma.as_ptr().add(off)),
                ),
            );
        }
        for i in (chunks * 4)..h {
            output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

/// Two-pass LayerNorm over one row via AVX2+FMA.
///
/// # Safety
/// Caller has checked `is_x86_feature_detected!("avx2")` and `"fma"`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn layer_norm_row_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;
    let inv_hf = 1.0 / h as f32;
    let chunks = h / 8;
    unsafe {
        // Pass 1: mean.
        let mut vsum = _mm256_setzero_ps();
        for c in 0..chunks {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(input.as_ptr().add(c * 8)));
        }
        // Horizontal reduce: 8 lanes → 1.
        let hsum = {
            let lo = _mm256_castps256_ps128(vsum);
            let hi = _mm256_extractf128_ps::<1>(vsum);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
            _mm_cvtss_f32(s1)
        };
        let mut sum = hsum;
        for i in (chunks * 8)..h {
            sum += input[i];
        }
        let mean = sum * inv_hf;
        let vmean = _mm256_set1_ps(mean);
        // Pass 2: var = mean((x − mean)²). Two-pass avoids one-pass cancellation.
        let mut vdev = _mm256_setzero_ps();
        for c in 0..chunks {
            let d = _mm256_sub_ps(_mm256_loadu_ps(input.as_ptr().add(c * 8)), vmean);
            vdev = _mm256_fmadd_ps(d, d, vdev);
        }
        let hdev = {
            let lo = _mm256_castps256_ps128(vdev);
            let hi = _mm256_extractf128_ps::<1>(vdev);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
            _mm_cvtss_f32(s1)
        };
        let mut dev = hdev;
        for i in (chunks * 8)..h {
            let d = input[i] - mean;
            dev += d * d;
        }
        let var = (dev * inv_hf).max(0.0);
        let inv = 1.0 / (var + eps).sqrt();
        let vinv = _mm256_set1_ps(inv);
        for c in 0..chunks {
            let off = c * 8;
            let x = _mm256_loadu_ps(input.as_ptr().add(off));
            let norm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinv);
            let g = _mm256_loadu_ps(gamma.as_ptr().add(off));
            let b = _mm256_loadu_ps(beta.as_ptr().add(off));
            _mm256_storeu_ps(output.as_mut_ptr().add(off), _mm256_fmadd_ps(norm, g, b));
        }
        for i in (chunks * 8)..h {
            output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

/// Portable two-pass LayerNorm over one row. Shared by the non-SIMD targets
/// and by x86-64's runtime fallback.
#[cfg(not(target_arch = "aarch64"))]
fn layer_norm_row_scalar(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    let inv_hf = 1.0 / h as f32;
    // Two-pass: var = mean((x − mean)²) — avoids one-pass E[x²]−mean² cancellation.
    let mut sum = 0f32;
    for i in 0..h {
        sum += input[i];
    }
    let mean = sum * inv_hf;
    let mut dev = 0f32;
    for i in 0..h {
        let d = input[i] - mean;
        dev += d * d;
    }
    let var = (dev * inv_hf).max(0.0);
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..h {
        output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
    }
}

/// x86-64 LayerNorm row. CPUID-dispatched at runtime so the same artifact
/// takes the AVX2 kernel on a capable host and the portable two-pass loop on
/// a pre-AVX one (Atom Bonnell..Tremont) — see [`bias_gelu`].
#[cfg(target_arch = "x86_64")]
pub fn layer_norm_row(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both feature bits just checked.
        unsafe { layer_norm_row_avx2(input, gamma, beta, output, h, eps) };
        return;
    }
    layer_norm_row_scalar(input, gamma, beta, output, h, eps);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn layer_norm_row(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    layer_norm_row_scalar(input, gamma, beta, output, h, eps);
}

/// LayerNorm over `rows` contiguous rows of width `h`.
///
/// Exists to hoist the CPUID dispatch out of the caller's row loop. Every
/// transformer-block path here calls `layer_norm_row` inside `for r in 0..m`,
/// which ran the `is_x86_feature_detected!` pair once per ROW. That check is a
/// cached relaxed load, so it reads as free — measured, it is **13.2% of the
/// work at h = 32**, 2.4% at h = 128, and nothing by h = 1024. Short rows are
/// not exotic, so the estimate was worth checking rather than trusting.
///
/// No behaviour change: same kernel, same order, one check per op.
pub fn layer_norm_rows(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    rows: usize,
    h: usize,
    eps: f32,
) {
    debug_assert!(input.len() >= rows * h && output.len() >= rows * h);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            for r in 0..rows {
                // SAFETY: both feature bits checked once, above the loop.
                unsafe {
                    layer_norm_row_avx2(
                        &input[r * h..(r + 1) * h],
                        gamma,
                        beta,
                        &mut output[r * h..(r + 1) * h],
                        h,
                        eps,
                    )
                };
            }
            return;
        }
        for r in 0..rows {
            layer_norm_row_scalar(
                &input[r * h..(r + 1) * h],
                gamma,
                beta,
                &mut output[r * h..(r + 1) * h],
                h,
                eps,
            );
        }
    }
    // Other targets dispatch at compile time, so there is nothing to hoist.
    #[cfg(not(target_arch = "x86_64"))]
    for r in 0..rows {
        layer_norm_row(
            &input[r * h..(r + 1) * h],
            gamma,
            beta,
            &mut output[r * h..(r + 1) * h],
            h,
            eps,
        );
    }
}

/// Inference BatchNorm with frozen running statistics (PyTorch `BatchNorm*d` eval).
///
/// `x` is row-major with feature dimension `channels` on the last axis
/// (`[B, C]`, `[B, P, C]`, …). `gamma`, `beta`, `mean`, `var` are length `C`.
pub fn batch_norm_inference(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    out: &mut [f32],
    channels: usize,
    eps: f32,
) {
    let n = x.len() / channels.max(1);
    for i in 0..n {
        for c in 0..channels {
            let idx = i * channels + c;
            let inv = 1.0 / (var[c] + eps).sqrt();
            let xhat = (x[idx] - mean[c]) * inv;
            out[idx] = gamma[c] * xhat + beta[c];
        }
    }
}

/// `d_x` for [`batch_norm_inference`] (mean/var treated as constants).
pub fn batch_norm_inference_backward_input(
    x: &[f32],
    gamma: &[f32],
    _mean: &[f32],
    var: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    channels: usize,
    eps: f32,
) {
    let n = x.len() / channels.max(1);
    for i in 0..n {
        for c in 0..channels {
            let idx = i * channels + c;
            let inv = 1.0 / (var[c] + eps).sqrt();
            dx[idx] = dy[idx] * gamma[c] * inv;
        }
    }
}

/// `d_gamma` for [`batch_norm_inference`].
pub fn batch_norm_inference_backward_gamma(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    dy: &[f32],
    dgamma: &mut [f32],
    channels: usize,
    eps: f32,
) {
    dgamma.fill(0.0);
    let n = x.len() / channels.max(1);
    for i in 0..n {
        for c in 0..channels {
            let idx = i * channels + c;
            let inv = 1.0 / (var[c] + eps).sqrt();
            let xhat = (x[idx] - mean[c]) * inv;
            dgamma[c] += dy[idx] * xhat;
        }
    }
}

/// `d_beta` for [`batch_norm_inference`].
pub fn batch_norm_inference_backward_beta(dy: &[f32], dbeta: &mut [f32], channels: usize) {
    dbeta.fill(0.0);
    let n = dy.len() / channels.max(1);
    for i in 0..n {
        for c in 0..channels {
            dbeta[c] += dy[i * channels + c];
        }
    }
}

/// Fused residual + bias + LayerNorm on [n, h] buffers.
/// Computes: output\[row\] = LN(a\[row\] + b\[row\] + bias, gamma, beta)
pub fn residual_bias_layer_norm(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    n: usize,
    h: usize,
    eps: f32,
) {
    // Temporary per-row buffer for a+b+bias (stack allocated for small h)
    let mut tmp = vec![0f32; h];
    for row in 0..n {
        let base = row * h;
        for i in 0..h {
            tmp[i] = a[base + i] + b[base + i] + bias[i];
        }
        layer_norm_row(&tmp, gamma, beta, &mut output[base..base + h], h, eps);
    }
}

/// Fused residual + bias + RMSNorm on [n, h] buffers.
/// Computes: output`row` = RmsNorm(a`row` + b`row` + bias, gamma, beta)
pub fn residual_bias_rms_norm(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    n: usize,
    h: usize,
    eps: f32,
) {
    let inv_h = 1.0 / h as f32;
    for row in 0..n {
        let base = row * h;
        let mut sumsq = 0f32;
        for i in 0..h {
            let v = a[base + i] + b[base + i] + bias[i];
            sumsq += v * v;
        }
        let inv_rms = (sumsq * inv_h + eps).sqrt().recip();
        for i in 0..h {
            let v = a[base + i] + b[base + i] + bias[i];
            output[base + i] = v * inv_rms * gamma[i] + beta[i];
        }
    }
}

/// Parallel residual + bias + LayerNorm.
pub fn par_residual_bias_ln(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    n: usize,
    h: usize,
    eps: f32,
) {
    let cfg = crate::config::RuntimeConfig::global();
    if n * h < cfg.par_threshold || n < cfg.min_rows_per_thread {
        residual_bias_layer_norm(a, b, bias, gamma, beta, output, n, h, eps);
        return;
    }
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let o_ptr = output.as_mut_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    let gamma_ptr = gamma.as_ptr() as usize;
    let beta_ptr = beta.as_ptr() as usize;
    pool::par_for(n, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let a_s = std::slice::from_raw_parts((a_ptr as *const f32).add(off * h), cnt * h);
        let b_s = std::slice::from_raw_parts((b_ptr as *const f32).add(off * h), cnt * h);
        let o_s = std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
        let bi = std::slice::from_raw_parts(bias_ptr as *const f32, h);
        let g = std::slice::from_raw_parts(gamma_ptr as *const f32, h);
        let be = std::slice::from_raw_parts(beta_ptr as *const f32, h);
        residual_bias_layer_norm(a_s, b_s, bi, g, be, o_s, cnt, h, eps);
    });
}

// ── Softmax (NEON / runtime AVX2 / parallel rows) ───────────────────────

/// Row-parallel wrapper: each row is independent so Rayon can split the outer
/// loop when there is enough work to amortize hand-off.
#[inline]
fn par_softmax_rows<F: Fn(&mut [f32], usize, usize) + Sync>(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    kernel: &F,
) {
    if rows >= 4 && pool::should_parallelize(rows * cols) {
        let base = data.as_mut_ptr() as usize;
        pool::par_for(rows, 1, &|off, cnt| {
            for r in off..off + cnt {
                let row = unsafe {
                    std::slice::from_raw_parts_mut((base as *mut f32).add(r * cols), cols)
                };
                kernel(row, 1, cols);
            }
        });
    } else {
        kernel(data, rows, cols);
    }
}

/// NEON-vectorized softmax: 3-pass (max, exp+sum, normalize).
#[cfg(target_arch = "aarch64")]
fn softmax_rows_neon(data: &mut [f32], rows: usize, cols: usize) {
    use std::arch::aarch64::*;
    let chunks = cols / 4;
    unsafe {
        for row in 0..rows {
            let base = row * cols;
            let ptr = data.as_mut_ptr().add(base);

            // Pass 1: find row max
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            for c in 0..chunks {
                vmax = vmaxq_f32(vmax, vld1q_f32(ptr.add(c * 4)));
            }
            let mut max_val = vmaxvq_f32(vmax);
            for i in (chunks * 4)..cols {
                max_val = max_val.max(*ptr.add(i));
            }

            // Pass 2: exp(x - max) and accumulate sum
            let vmx = vdupq_n_f32(max_val);
            let mut vsum = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let off = c * 4;
                let e = neon_exp4(vsubq_f32(vld1q_f32(ptr.add(off)), vmx));
                vst1q_f32(ptr.add(off), e);
                vsum = vaddq_f32(vsum, e);
            }
            let mut sum = vaddvq_f32(vsum);
            for i in (chunks * 4)..cols {
                let e = (*ptr.add(i) - max_val).exp();
                *ptr.add(i) = e;
                sum += e;
            }

            // Pass 3: normalize
            let vinv = vdupq_n_f32(1.0 / sum);
            for c in 0..chunks {
                let off = c * 4;
                vst1q_f32(ptr.add(off), vmulq_f32(vld1q_f32(ptr.add(off)), vinv));
            }
            let inv = 1.0 / sum;
            for i in (chunks * 4)..cols {
                *ptr.add(i) *= inv;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
    par_softmax_rows(data, rows, cols, &softmax_rows_neon);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn softmax_rows_avx2(data: &mut [f32], rows: usize, cols: usize) {
    use std::arch::x86_64::*;
    let chunks = cols / 8;
    for r in 0..rows {
        let row = data.as_mut_ptr().add(r * cols);
        // 1) Vector max for stability.
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in 0..chunks {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row.add(c * 8)));
        }
        let mut max_v = {
            let lo = _mm256_castps256_ps128(vmax);
            let hi = _mm256_extractf128_ps::<1>(vmax);
            let s4 = _mm_max_ps(lo, hi);
            let s2 = _mm_max_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_max_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
            _mm_cvtss_f32(s1)
        };
        for i in (chunks * 8)..cols {
            let v = *row.add(i);
            if v > max_v {
                max_v = v;
            }
        }
        // 2) exp(x − max) and sum.
        let vmax = _mm256_set1_ps(max_v);
        let mut vsum = _mm256_setzero_ps();
        for c in 0..chunks {
            let off = c * 8;
            let e = avx2_exp8(_mm256_sub_ps(_mm256_loadu_ps(row.add(off)), vmax));
            _mm256_storeu_ps(row.add(off), e);
            vsum = _mm256_add_ps(vsum, e);
        }
        let mut sum_v = {
            let lo = _mm256_castps256_ps128(vsum);
            let hi = _mm256_extractf128_ps::<1>(vsum);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
            _mm_cvtss_f32(s1)
        };
        for i in (chunks * 8)..cols {
            let v = (*row.add(i) - max_v).exp();
            *row.add(i) = v;
            sum_v += v;
        }
        // 3) Normalize.
        let vinv = _mm256_set1_ps(1.0 / sum_v);
        for c in 0..chunks {
            let off = c * 8;
            _mm256_storeu_ps(
                row.add(off),
                _mm256_mul_ps(_mm256_loadu_ps(row.add(off)), vinv),
            );
        }
        let inv_sum = 1.0 / sum_v;
        for i in (chunks * 8)..cols {
            *row.add(i) *= inv_sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
    let avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    if avx2 {
        par_softmax_rows(data, rows, cols, &|d, r, c| unsafe {
            softmax_rows_avx2(d, r, c);
        });
    } else {
        par_softmax_rows(data, rows, cols, &softmax_rows_poly);
    }
}

pub fn softmax_rows_poly(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        softmax_row_poly(&mut data[r * cols..(r + 1) * cols]);
    }
}

/// Horizontal max over four independent accumulators.
///
/// `iter().fold(NEG_INFINITY, f32::max)` does NOT vectorize: a float reduction
/// is not reassociable without fast-math, which Rust never enables, so LLVM is
/// obliged to keep one serial dependency chain. Four lanes that only combine at
/// the end give it four chains it can fuse into `maxps` / `fmaxnm`.
///
/// This DOES change the order of operations, and for `max` that is exactly
/// equivalent (max is associative and commutative over non-NaN floats, and the
/// NaN behaviour of `f32::max` — return the non-NaN operand — is unchanged by
/// regrouping).
#[inline]
fn reduce_max4(row: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 4];
    let mut it = row.chunks_exact(4);
    for ch in &mut it {
        for i in 0..4 {
            acc[i] = acc[i].max(ch[i]);
        }
    }
    let mut m = acc[0].max(acc[1]).max(acc[2].max(acc[3]));
    for &v in it.remainder() {
        m = m.max(v);
    }
    m
}

/// Horizontal sum over four independent accumulators — see [`reduce_max4`] for
/// why `iter().sum()` cannot vectorize.
///
/// Unlike `max`, regrouping a float *sum* does change the result. It changes it
/// for the better: pairing partial sums is a (shallow) tree reduction, whose
/// error grows like log(n) rather than n, so this is more accurate than the
/// sequential sum it replaces, not less.
#[inline]
fn reduce_sum4(row: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    let mut it = row.chunks_exact(4);
    for ch in &mut it {
        for i in 0..4 {
            acc[i] += ch[i];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for &v in it.remainder() {
        s += v;
    }
    s
}

/// Row softmax over the portable [`crate::vmath::exp_poly`] — the fast arm for
/// hosts without AVX2 (every Atom through Tremont, plus wasm/riscv).
///
/// Was `crate::naive::softmax`, which is the *accuracy reference* parity tests
/// compare kernels against and so deliberately calls libm per element; that
/// left pre-AVX2 x86 with no SIMD on the hottest op in attention. `naive`
/// stays untouched as the reference, and this arm uses the same polynomial as
/// `softmax_rows_avx2`, so the two dispatch arms agree.
///
/// Three loops rather than one on purpose: the exp and scale passes vectorize,
/// while the max and sum reductions cannot (float reductions are not
/// reassociable without fast-math, which Rust never enables). Fusing them
/// would sink the whole thing back to scalar.
// Dead on aarch64 by design (NEON arm always wins); kept compiled so the
// guard test runs on an Apple dev machine.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn softmax_row_poly(row: &mut [f32]) {
    let max_v = reduce_max4(row);
    if !max_v.is_finite() {
        // All -inf (a fully masked row) or a NaN: defer to the reference,
        // which has the careful handling for these.
        let n = row.len();
        crate::naive::softmax(row, 1, n);
        return;
    }
    for v in row.iter_mut() {
        *v = crate::vmath::exp_poly(*v - max_v);
    }
    let sum = reduce_sum4(row);
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
    par_softmax_rows(data, rows, cols, &softmax_rows_poly);
}

// ── GELU in-place (no bias) ────────────────────────────────────────────

/// NEON GELU activation in-place (without bias addition).
#[cfg(target_arch = "aarch64")]
pub fn gelu_inplace(data: &mut [f32]) {
    use std::arch::aarch64::*;
    let len = data.len();
    let chunks = len / 4;
    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let inv_sqrt2 = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);
        let p = vdupq_n_f32(0.3275911);
        let a1 = vdupq_n_f32(0.254_829_6);
        let a2 = vdupq_n_f32(-0.284_496_72);
        let a3 = vdupq_n_f32(1.421_413_8);
        let a4 = vdupq_n_f32(-1.453_152_1);
        let a5 = vdupq_n_f32(1.061_405_4);
        let neg_one = vdupq_n_f32(-1.0);
        let zero = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let ptr = data.as_mut_ptr().add(c * 4);
            let x = vld1q_f32(ptr);
            let erf_arg = vmulq_f32(x, inv_sqrt2);
            let xa = vabsq_f32(erf_arg);
            let sign = vbslq_f32(vcgeq_f32(erf_arg, zero), one, neg_one);
            let denom = vfmaq_f32(one, p, xa);
            let t = vdivq_f32(one, denom);
            let mut y = a5;
            y = vfmaq_f32(a4, y, t);
            y = vfmaq_f32(a3, y, t);
            y = vfmaq_f32(a2, y, t);
            y = vfmaq_f32(a1, y, t);
            y = vmulq_f32(y, t);
            let exp_val = neon_exp4(vnegq_f32(vmulq_f32(xa, xa)));
            let erf_val = vmulq_f32(sign, vfmsq_f32(one, y, exp_val));
            vst1q_f32(ptr, vmulq_f32(x, vmulq_f32(half, vaddq_f32(one, erf_val))));
        }
        for i in (chunks * 4)..len {
            data[i] = scalar_gelu(data[i]);
        }
    }
}

/// Erf-GELU via AVX2+FMA. Caller must have checked feature bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
// a1..a5 are the Abramowitz & Stegun 7.1.26 erf coefficients — published
// full-precision values whose nearest-f32 is exactly what we want. Keep the
// literals verbatim; the precision lint is a false positive here.
#[allow(unsafe_op_in_unsafe_fn, clippy::excessive_precision)]
unsafe fn gelu_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = data.len() / 8;
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let inv_sqrt2 = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
    let p = _mm256_set1_ps(0.3275911);
    let a1 = _mm256_set1_ps(0.254829592);
    let a2 = _mm256_set1_ps(-0.284496736);
    let a3 = _mm256_set1_ps(1.421413741);
    let a4 = _mm256_set1_ps(-1.453152027);
    let a5 = _mm256_set1_ps(1.061405429);
    let neg_one = _mm256_set1_ps(-1.0);
    let zero = _mm256_set1_ps(0.0);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    for c in 0..chunks {
        let off = c * 8;
        let ptr = data.as_mut_ptr().add(off);
        let x = _mm256_loadu_ps(ptr);
        let erf_arg = _mm256_mul_ps(x, inv_sqrt2);
        let xa = _mm256_and_ps(erf_arg, abs_mask);
        let ge0 = _mm256_cmp_ps::<_CMP_GE_OQ>(erf_arg, zero);
        let sign = _mm256_blendv_ps(neg_one, one, ge0);
        let denom = _mm256_fmadd_ps(p, xa, one);
        let t = _mm256_div_ps(one, denom);
        let mut y = a5;
        y = _mm256_fmadd_ps(y, t, a4);
        y = _mm256_fmadd_ps(y, t, a3);
        y = _mm256_fmadd_ps(y, t, a2);
        y = _mm256_fmadd_ps(y, t, a1);
        y = _mm256_mul_ps(y, t);
        let exp_val = avx2_exp8(_mm256_sub_ps(zero, _mm256_mul_ps(xa, xa)));
        let erf_val = _mm256_mul_ps(sign, _mm256_fnmadd_ps(y, exp_val, one));
        _mm256_storeu_ps(
            ptr,
            _mm256_mul_ps(x, _mm256_mul_ps(half, _mm256_add_ps(one, erf_val))),
        );
    }
    for i in (chunks * 8)..data.len() {
        data[i] = scalar_gelu(data[i]);
    }
}

#[cfg(target_arch = "x86_64")]
pub fn gelu_inplace(data: &mut [f32]) {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe { gelu_inplace_avx2(data) };
        return;
    }
    for v in data.iter_mut() {
        *v = scalar_gelu(*v);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn gelu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = scalar_gelu(*v);
    }
}

/// Parallel GELU in-place (splits work across thread pool).
///
/// Activation kernels are O(n) with very low per-element cost
/// (~10 NEON cycles on aarch64). Pool dispatch overhead — even
/// with the parked design — is in the multi-µs range under
/// container scheduling, which dwarfs the actual compute for any
/// reasonable activation size. Threshold here is 1 Mi elements:
/// only crossed by very large activation tensors (e.g. an
/// H=4096, FFN=14336, S=1024 LLM up-projection at ~14M
/// elements). Single-thread NEON is the clear win below that.
const ACTIVATION_PAR_MIN: usize = 1 << 20;

/// Tanh-approximation GELU (matches PyTorch/candle `Tensor::gelu`):
///   y = 0.5 x (1 + tanh(√(2/π) · (x + 0.044715 x³)))
///
/// Scalar-only for now; the erf-based `gelu_inplace` above is SIMD.
/// Routed from `Activation::GeluApprox` so models that need
/// numerical parity with PyTorch's default GELU (e.g. DINOv2,
/// many ViTs) get the right formula. Use `Activation::Gelu` for the
/// erf form (also PyTorch-default in some newer builds).
#[inline]
pub fn scalar_gelu_approx(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // √(2/π)
    const A: f32 = 0.044_715;
    0.5 * x * (1.0 + (C * (x + A * x * x * x)).tanh())
}

pub fn gelu_approx_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = scalar_gelu_approx(*v);
    }
}

pub fn par_gelu_approx_inplace(data: &mut [f32]) {
    let len = data.len();
    if len < ACTIVATION_PAR_MIN {
        gelu_approx_inplace(data);
        return;
    }
    let cfg = crate::config::RuntimeConfig::global();
    let chunk = 512;
    let rows = len / chunk;
    if rows < 2 {
        gelu_approx_inplace(data);
        return;
    }
    let data_ptr = data.as_mut_ptr() as usize;
    pool::par_for(rows, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let start = off * chunk;
        let end = if off + cnt >= rows {
            len
        } else {
            (off + cnt) * chunk
        };
        let s = std::slice::from_raw_parts_mut((data_ptr as *mut f32).add(start), end - start);
        gelu_approx_inplace(s);
    });
    // No trailing pass for `len % chunk`: the final `par_for` chunk already
    // extends its `end` to `len` (see `off + cnt >= rows` above), so a second
    // pass over `data[rows * chunk..]` would apply the activation TWICE to the
    // last `len % 512` elements. In place that is silently wrong — and only for
    // the tail, so whole-tensor summaries still look right.
}

pub fn gelu_approx_out(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = scalar_gelu_approx(*s);
    }
}

pub fn par_gelu_approx_out(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let len = src.len();
    if len < ACTIVATION_PAR_MIN {
        gelu_approx_out(src, dst);
        return;
    }
    let cfg = crate::config::RuntimeConfig::global();
    let chunk = 512;
    let rows = len / chunk;
    if rows < 2 {
        gelu_approx_out(src, dst);
        return;
    }
    let src_ptr = src.as_ptr() as usize;
    let dst_ptr = dst.as_mut_ptr() as usize;
    pool::par_for(rows, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let start = off * chunk;
        let end = if off + cnt >= rows {
            len
        } else {
            (off + cnt) * chunk
        };
        let n = end - start;
        let s = std::slice::from_raw_parts((src_ptr as *const f32).add(start), n);
        let d = std::slice::from_raw_parts_mut((dst_ptr as *mut f32).add(start), n);
        gelu_approx_out(s, d);
    });
    // No trailing pass for `len % chunk`: the final `par_for` chunk already
    // extends its `end` to `len` (see `off + cnt >= rows` above), so a second
    // pass over `data[rows * chunk..]` would apply the activation TWICE to the
    // last `len % 512` elements. In place that is silently wrong — and only for
    // the tail, so whole-tensor summaries still look right.
}

pub fn par_gelu_inplace(data: &mut [f32]) {
    let len = data.len();
    if len < ACTIVATION_PAR_MIN {
        gelu_inplace(data);
        return;
    }
    let cfg = crate::config::RuntimeConfig::global();
    let chunk = 512;
    let rows = len / chunk;
    if rows < 2 {
        gelu_inplace(data);
        return;
    }
    let data_ptr = data.as_mut_ptr() as usize;
    pool::par_for(rows, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let start = off * chunk;
        let end = if off + cnt >= rows {
            len
        } else {
            (off + cnt) * chunk
        };
        let s = std::slice::from_raw_parts_mut((data_ptr as *mut f32).add(start), end - start);
        gelu_inplace(s);
    });
    // No trailing pass for `len % chunk`: the final `par_for` chunk already
    // extends its `end` to `len` (see `off + cnt >= rows` above), so a second
    // pass over `data[rows * chunk..]` would apply the activation TWICE to the
    // last `len % 512` elements. In place that is silently wrong — and only for
    // the tail, so whole-tensor summaries still look right.
}

/// Parallel SiLU in-place. Same threshold reasoning as `par_gelu_inplace`.
pub fn par_silu_inplace(data: &mut [f32]) {
    let len = data.len();
    if len < ACTIVATION_PAR_MIN {
        silu_inplace(data);
        return;
    }
    let cfg = crate::config::RuntimeConfig::global();
    let chunk = 512;
    let rows = len / chunk;
    if rows < 2 {
        silu_inplace(data);
        return;
    }
    let data_ptr = data.as_mut_ptr() as usize;
    pool::par_for(rows, cfg.min_rows_per_thread, &|off, cnt| unsafe {
        let start = off * chunk;
        let end = if off + cnt >= rows {
            len
        } else {
            (off + cnt) * chunk
        };
        let s = std::slice::from_raw_parts_mut((data_ptr as *mut f32).add(start), end - start);
        silu_inplace(s);
    });
    // No trailing pass for `len % chunk`: the final `par_for` chunk already
    // extends its `end` to `len` (see `off + cnt >= rows` above), so a second
    // pass over `data[rows * chunk..]` would apply the activation TWICE to the
    // last `len % 512` elements. In place that is silently wrong — and only for
    // the tail, so whole-tensor summaries still look right.
}

// ── Small-m NEON matmul ─────────────────────────────────────────────────

/// NEON matmul for tiny m (1-8 rows). Avoids BLAS call overhead.
/// C = A @ B where A=\[m,k\], B=\[k,n\], C=\[m,n\], all row-major.
/// For m≤8 with small k×n (under ~16K elements), this beats cblas_sgemm
/// by avoiding AMX setup cost and function call overhead.
#[cfg(target_arch = "aarch64")]
pub fn neon_sgemm_small(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::aarch64::*;
    let n4 = n / 4;
    unsafe {
        for j4 in 0..n4 {
            let j = j4 * 4;
            // m accumulators (one per output row, 4-wide)
            let mut acc = [vdupq_n_f32(0.0); 8];
            for kk in 0..k {
                let bv = vld1q_f32(b.as_ptr().add(kk * n + j));
                for i in 0..m {
                    let av = vdupq_n_f32(*a.as_ptr().add(i * k + kk));
                    acc[i] = vfmaq_f32(acc[i], av, bv);
                }
            }
            for i in 0..m {
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc[i]);
            }
        }
        // Remainder columns
        for j in (n4 * 4)..n {
            for i in 0..m {
                let mut sum = 0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sgemm_small(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    crate::naive::matmul(a, b, c, m, k, n);
}

/// NEON sgemm_bias for tiny m: C = A @ B + bias.
#[cfg(target_arch = "aarch64")]
pub fn neon_sgemm_bias_small(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    neon_sgemm_small(a, b, c, m, k, n);
    crate::blas::bias_add(c, bias, m, n);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sgemm_bias_small(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    crate::naive::matmul(a, b, c, m, k, n);
    crate::naive::bias_add(c, bias, m, n);
}

// ── Scalar fallbacks ────────────────────────────────────────────────────

fn scalar_gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + scalar_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn scalar_erf(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    // `exp_poly`, not libm: this is the scalar arm of an approximation whose
    // vector arms already use the same polynomial (`avx2_exp8` / `neon_exp4`),
    // so matching it both removes a libm call per element and makes the arms
    // agree. A&S 7.1.26 itself is only ~1.5e-7 accurate, so the polynomial's
    // 2.5e-7 on the inner exp is well inside the erf approximation's own error.
    sign * (1.0 - y * crate::vmath::exp_poly(-xa * xa))
}

/// NCHW LayerNorm2d (candle / SAM semantics): normalize across channels at
/// each spatial position. `gamma`/`beta` are per-channel `[C]`.
pub fn layer_norm2d_nchw(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    eps: f32,
) {
    let spatial = h * w;
    for b in 0..batch {
        for i in 0..spatial {
            let mut mean = 0.0f32;
            for c in 0..channels {
                mean += input[((b * channels + c) * spatial) + i];
            }
            mean /= channels as f32;
            let mut var = 0.0f32;
            for c in 0..channels {
                let d = input[((b * channels + c) * spatial) + i] - mean;
                var += d * d;
            }
            var /= channels as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..channels {
                let v = (input[((b * channels + c) * spatial) + i] - mean) * inv;
                output[((b * channels + c) * spatial) + i] = v * gamma[c] + beta[c];
            }
        }
    }
}

/// NCHW transposed convolution (PyTorch `ConvTranspose2d`, no bias).
/// Weight layout `[C_in, C_out/groups, kH, kW]`.
pub fn conv_transpose2d_nchw(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
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
    output.fill(0.0);
    // Guard degenerate grouping: `groups == 0` divides by zero at `c_in / groups`,
    // and `groups > c_in` makes `c_in_per_g == 0` so `ic / c_in_per_g` panics. Such
    // a conv-transpose has nothing to accumulate — leave the zeroed output as-is.
    if groups == 0 || c_in < groups {
        return;
    }
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    for ni in 0..n {
        for ic in 0..c_in {
            let g = ic / c_in_per_g;
            let _ic_off = ic % c_in_per_g;
            for iy in 0..h {
                for ix in 0..w {
                    let v = input[((ni * c_in + ic) * h + iy) * w + ix];
                    if v == 0.0 {
                        continue;
                    }
                    for ky in 0..kh {
                        let oy = iy * sh + ky * dh;
                        if oy < ph || oy >= h_out + ph {
                            continue;
                        }
                        let oy = oy - ph;
                        if oy >= h_out {
                            continue;
                        }
                        for kx in 0..kw {
                            let ox = ix * sw + kx * dw;
                            if ox < pw || ox >= w_out + pw {
                                continue;
                            }
                            let ox = ox - pw;
                            if ox >= w_out {
                                continue;
                            }
                            for oc_off in 0..c_out_per_g {
                                let oc = g * c_out_per_g + oc_off;
                                let w_idx = ((ic * c_out_per_g + oc_off) * kh + ky) * kw + kx;
                                let wt = weight[w_idx];
                                output[((ni * c_out + oc) * h_out + oy) * w_out + ox] += v * wt;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// NCDHW transposed convolution (the depth-axis analogue of
/// [`conv_transpose2d_nchw`]). Scatter form: each input voxel adds its
/// weighted kernel into the output. Weight `[C_in, C_out/g, kD, kH, kW]`.
/// `output_padding` is assumed folded into `d_out/h_out/w_out` by the caller.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d_ncdhw(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
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
    output.fill(0.0);
    // Guard degenerate grouping: `groups == 0` divides by zero at `c_in / groups`,
    // and `groups > c_in` makes `c_in_per_g == 0` so `ic / c_in_per_g` panics. Such
    // a conv-transpose has nothing to accumulate — leave the zeroed output as-is.
    if groups == 0 || c_in < groups {
        return;
    }
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    for ni in 0..n {
        for ic in 0..c_in {
            let g = ic / c_in_per_g;
            for id in 0..d {
                for iy in 0..h {
                    for ix in 0..w {
                        let v = input[(((ni * c_in + ic) * d + id) * h + iy) * w + ix];
                        if v == 0.0 {
                            continue;
                        }
                        for kz in 0..kd {
                            let oz = id * sd + kz * dd;
                            if oz < pd || oz >= d_out + pd {
                                continue;
                            }
                            let oz = oz - pd;
                            if oz >= d_out {
                                continue;
                            }
                            for ky in 0..kh {
                                let oy = iy * sh + ky * dh;
                                if oy < ph || oy >= h_out + ph {
                                    continue;
                                }
                                let oy = oy - ph;
                                if oy >= h_out {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ox = ix * sw + kx * dw;
                                    if ox < pw || ox >= w_out + pw {
                                        continue;
                                    }
                                    let ox = ox - pw;
                                    if ox >= w_out {
                                        continue;
                                    }
                                    for oc_off in 0..c_out_per_g {
                                        let oc = g * c_out_per_g + oc_off;
                                        let w_idx = (((ic * c_out_per_g + oc_off) * kd + kz) * kh
                                            + ky)
                                            * kw
                                            + kx;
                                        let wt = weight[w_idx];
                                        output[(((ni * c_out + oc) * d_out + oz) * h_out + oy)
                                            * w_out
                                            + ox] += v * wt;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// NCHW group normalization: normalizes each `(C/G)×H×W` group.
pub fn group_norm_nchw(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    eps: f32,
) {
    let cpg = channels / num_groups;
    let spatial = h * w;
    let n = (cpg * spatial) as f32;
    for b in 0..batch {
        for g in 0..num_groups {
            let c0 = g * cpg;
            let mut mean = 0.0f32;
            for c in 0..cpg {
                let plane = &input
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                mean += plane.iter().sum::<f32>();
            }
            mean /= n;
            let mut var = 0.0f32;
            for c in 0..cpg {
                let plane = &input
                    [((b * channels + c0 + c) * spatial)..((b * channels + c0 + c + 1) * spatial)];
                for &v in plane {
                    let d = v - mean;
                    var += d * d;
                }
            }
            var /= n;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..cpg {
                let gi = c0 + c;
                let gamm = gamma[gi];
                let bet = beta[gi];
                let src =
                    &input[((b * channels + gi) * spatial)..((b * channels + gi + 1) * spatial)];
                let dst = &mut output
                    [((b * channels + gi) * spatial)..((b * channels + gi + 1) * spatial)];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = (s - mean) * inv * gamm + bet;
                }
            }
        }
    }
}

/// Nearest-neighbor 2× upsample on planar NCHW.
pub fn resize_nearest_2x_nchw(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    h: usize,
    w: usize,
) {
    let h2 = h * 2;
    let w2 = w * 2;
    for c in 0..channels {
        let plane = &input[c * h * w..(c + 1) * h * w];
        let dst = &mut output[c * h2 * w2..(c + 1) * h2 * w2];
        for y in 0..h {
            for x in 0..w {
                let v = plane[y * w + x];
                for dy in 0..2 {
                    for dx in 0..2 {
                        dst[(y * 2 + dy) * w2 + (x * 2 + dx)] = v;
                    }
                }
            }
        }
    }
}

/// Nearest-neighbor NCDHW resample to `[d_out, h_out, w_out]` spatial size.
/// Mapping: `src = min(floor(dst * in / out), in - 1)`.
pub fn interpolate3d_ncdhw(
    input: &[f32],
    output: &mut [f32],
    n: usize,
    c: usize,
    d_in: usize,
    h_in: usize,
    w_in: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
) {
    for bn in 0..n {
        for ch in 0..c {
            let in_base = (bn * c + ch) * d_in * h_in * w_in;
            let out_base = (bn * c + ch) * d_out * h_out * w_out;
            for do_ in 0..d_out {
                let di = ((do_ * d_in) / d_out).min(d_in.saturating_sub(1));
                for ho in 0..h_out {
                    let hi = ((ho * h_in) / h_out).min(h_in.saturating_sub(1));
                    for wo in 0..w_out {
                        let wi = ((wo * w_in) / w_out).min(w_in.saturating_sub(1));
                        let src = in_base + (di * h_in + hi) * w_in + wi;
                        let dst = out_base + (do_ * h_out + ho) * w_out + wo;
                        output[dst] = input[src];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_correctness() {
        let x = 1.5f32;
        let g = scalar_gelu(x);
        // Reference: gelu(1.5) ≈ 1.3990
        assert!((g - 1.3990).abs() < 0.01, "gelu(1.5) = {g}");
    }

    /// The two CPUID-dispatched arms of `bias_gelu` / `layer_norm_row` must
    /// agree. One artifact now takes the AVX2 kernel on a capable host and the
    /// portable one on a pre-AVX host (every Atom through Tremont), so a
    /// divergence here is a numerics difference that depends on which machine
    /// runs the binary — the hardest kind to reproduce. The arms use the same
    /// Abramowitz & Stegun 7.1.26 erf and the same two-pass LayerNorm; only
    /// `exp` (fast polynomial vs libm) and the summation order differ.
    ///
    /// Returns early rather than failing where AVX2 is absent: there the
    /// scalar arm is the only one reachable, and `just check-isa` exercises
    /// exactly that leg on an emulated Atom.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_dispatch_arms_agree() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let (m, n) = (3usize, 64usize);
        let data: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.37 - 35.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32 * 0.11 - 3.5).collect();
        let mut simd = data.clone();
        let mut scalar = data.clone();
        // SAFETY: both feature bits checked above.
        unsafe { bias_gelu_avx2(&mut simd, &bias, m, n) };
        bias_gelu_scalar(&mut scalar, &bias, m, n);
        let worst = simd
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // Measured 1.19e-7 — exactly one f32 ULP at this magnitude. The
        // bound leaves ~16 ULP of headroom; anything near it means the two
        // arms stopped computing the same function.
        assert!(worst < 2e-6, "bias_gelu arms diverge by {worst}");

        let h = 128usize;
        let input: Vec<f32> = (0..h)
            .map(|i| ((i * 37) % 19) as f32 * 0.25 - 2.0)
            .collect();
        let gamma: Vec<f32> = (0..h).map(|i| 1.0 + i as f32 * 0.01).collect();
        let beta: Vec<f32> = (0..h).map(|i| i as f32 * -0.02).collect();
        let mut simd = vec![0f32; h];
        let mut scalar = vec![0f32; h];
        // SAFETY: both feature bits checked above.
        unsafe { layer_norm_row_avx2(&input, &gamma, &beta, &mut simd, h, 1e-5) };
        layer_norm_row_scalar(&input, &gamma, &beta, &mut scalar, h, 1e-5);
        let worst = simd
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // Measured 4.77e-7 — 4 ULP, from vector vs sequential summation
        // order in the two reduction passes, not a formulation difference.
        assert!(worst < 4e-6, "layer_norm_row arms diverge by {worst}");
    }

    /// `softmax_rows_poly` is the arm that runs wherever AVX2 is absent — i.e.
    /// on an Atom, never on the machine this is usually developed on. Nothing
    /// covered it, so a defect there would have been invisible locally and
    /// wrong in production. Pin it against `naive::softmax`, the reference.
    #[test]
    fn softmax_poly_arm_matches_the_reference() {
        let cases: Vec<Vec<f32>> = vec![
            (0..64)
                .map(|i| ((i * 37) % 23) as f32 * 0.3 - 3.0)
                .collect(),
            // Large magnitudes: the max-subtraction has to keep this finite.
            (0..48).map(|i| i as f32 * 4.0 - 90.0).collect(),
            // Already-uniform row, and a single dominant element.
            vec![1.0; 33],
            {
                let mut v = vec![-50.0f32; 17];
                v[9] = 50.0;
                v
            },
            // Fully masked row — exercises the non-finite early return.
            vec![f32::NEG_INFINITY; 8],
        ];
        for (ci, src) in cases.iter().enumerate() {
            let cols = src.len();
            let mut poly = src.clone();
            let mut want = src.clone();
            softmax_rows_poly(&mut poly, 1, cols);
            crate::naive::softmax(&mut want, 1, cols);
            let worst = poly
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // Measured 4.66e-10 on the realistic rows: the polynomial's
            // relative error largely cancels in the normalization, since it
            // scales numerator and denominator alike.
            assert!(worst < 1e-6, "case {ci}: softmax poly arm off by {worst:e}");
            if src.iter().any(|v| v.is_finite()) {
                let sum: f32 = poly.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "case {ci}: rows sum to {sum}");
            }
        }
    }

    /// `swiglu_rows` is the FFN activation of every modern transformer and had
    /// no CPU-side test at all — its only coverage was GPU parity tests in
    /// which the CPU *is* the reference, so a CPU regression would have
    /// silently moved the thing everything else is checked against.
    #[test]
    fn swiglu_rows_matches_a_libm_reference() {
        for &gate_first in &[true, false] {
            let (outer, n) = (5usize, 37usize);
            let inp: Vec<f32> = (0..outer * 2 * n)
                .map(|i| ((i * 29) % 61) as f32 * 0.4 - 12.0)
                .collect();
            let mut got = vec![0f32; outer * n];
            swiglu_rows(&inp, &mut got, outer, n, gate_first);

            let mut want = vec![0f32; outer * n];
            for o in 0..outer {
                for i in 0..n {
                    let row = &inp[o * 2 * n..(o + 1) * 2 * n];
                    let (up, gate) = if gate_first {
                        (row[n + i], row[i])
                    } else {
                        (row[i], row[n + i])
                    };
                    want[o * n + i] = up * (gate / (1.0 + (-gate).exp()));
                }
            }
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("swiglu gate_first={gate_first} worst abs diff {worst:e}");
            // Measured 2.38e-7 / 1.19e-7, i.e. 1-2 ULP at these magnitudes.
            assert!(
                worst < 1e-5,
                "swiglu off by {worst:e} (gate_first={gate_first})"
            );
        }
    }

    #[test]
    fn bias_gelu_works() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        bias_gelu(&mut data, &bias, 2, 4);
        // After bias+gelu, values should be > 0 (all inputs positive)
        for &v in &data {
            assert!(v > 0.0, "bias_gelu produced {v}");
        }
    }

    #[test]
    fn batch_norm_inference_roundtrip() {
        let c = 4usize;
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; c];
        let beta = vec![0.0; c];
        let mean = vec![2.5, 2.5, 2.5, 2.5];
        let var = vec![1.0; c];
        let mut y = vec![0.0; 8];
        batch_norm_inference(&x, &gamma, &beta, &mean, &var, &mut y, c, 1e-5);
        let mut dx = vec![0.0; 8];
        let dy = vec![1.0; 8];
        let mut dgamma = vec![0.0; c];
        let mut dbeta = vec![0.0; c];
        batch_norm_inference_backward_input(&x, &gamma, &mean, &var, &dy, &mut dx, c, 1e-5);
        batch_norm_inference_backward_gamma(&x, &mean, &var, &dy, &mut dgamma, c, 1e-5);
        batch_norm_inference_backward_beta(&dy, &mut dbeta, c);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(dx.iter().all(|v| v.is_finite()));
        assert!(dgamma.iter().any(|&v| v.abs() > 1e-6));
        assert_eq!(dbeta, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn layer_norm_unit_test() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        layer_norm_row(&input, &gamma, &beta, &mut output, 4, 1e-5);
        // Mean=2.5, std≈1.118. output ≈ [-1.342, -0.447, 0.447, 1.342]
        assert!((output[0] - -1.342).abs() < 0.01);
        assert!((output[3] - 1.342).abs() < 0.01);
        // Sum should be ~0 (normalized)
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 0.01, "LN sum should be ~0, got {sum}");
    }

    #[test]
    fn par_bias_gelu_matches_sequential() {
        let n = 100;
        let m = 64;
        let mut data_par = vec![0.5f32; n * m];
        let mut data_seq = data_par.clone();
        let bias = vec![0.1f32; m];

        bias_gelu(&mut data_seq, &bias, n, m);
        par_bias_gelu(&mut data_par, &bias, n, m);

        let max_diff: f32 = data_par
            .iter()
            .zip(data_seq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-6, "par vs seq diff: {max_diff}");
    }
}

#[cfg(test)]
mod par_activation_tail_tests {
    use super::*;

    /// The parallel activation kernels split the buffer into 512-element chunks
    /// and let the last worker absorb the remainder. A trailing "finish the
    /// tail" pass on top of that applied the activation TWICE to the final
    /// `len % 512` elements — wrong only in the tail, so tensor-wide summaries
    /// (max, mean) still looked right.
    ///
    /// `rlx-tinymyo` hit this on an MLP up-projection of `[1, 1387, 768]`:
    /// 1387·768 = 1_065_216 is over `ACTIVATION_PAR_MIN` and leaves 256
    /// elements over, so the last 256 values of the final token were
    /// `gelu(gelu(x))` — an 18% error confined to one row, while every
    /// accelerator (which does not share this kernel) was correct.
    fn assert_matches_serial(len: usize) {
        let mk = || -> Vec<f32> { (0..len).map(|i| ((i % 97) as f32 / 12.0) - 4.0).collect() };
        for (name, par, serial) in [
            (
                "gelu",
                par_gelu_inplace as fn(&mut [f32]),
                gelu_inplace as fn(&mut [f32]),
            ),
            ("gelu_approx", par_gelu_approx_inplace, gelu_approx_inplace),
            ("silu", par_silu_inplace, silu_inplace),
        ] {
            let (mut a, mut b) = (mk(), mk());
            par(&mut a);
            serial(&mut b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-6,
                    "{name} len={len} index {i}: parallel {x} != serial {y}"
                );
            }
        }
        let (src, mut d_par, mut d_ser) = (mk(), mk(), mk());
        par_gelu_approx_out(&src, &mut d_par);
        gelu_approx_out(&src, &mut d_ser);
        assert_eq!(d_par, d_ser, "gelu_approx_out len={len}");
    }

    #[test]
    fn a_tail_shorter_than_the_chunk_is_not_activated_twice() {
        // 1387 * 768 — the shape that exposed it; 256 elements over a chunk.
        assert_matches_serial(1_065_216);
    }

    #[test]
    fn tail_lengths_around_the_parallel_threshold_all_match_serial() {
        for extra in [0usize, 1, 255, 256, 511] {
            assert_matches_serial(ACTIVATION_PAR_MIN + extra);
        }
    }
}
