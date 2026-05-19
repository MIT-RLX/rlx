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

//! SIMD kernels for fused operations.
//!
//! These are the production kernels extracted from burnembed's ndarray_fused.rs.
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
/// stays in the ~2e-7 range. Requires `+avx2,+fma` codegen.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn avx2_exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.3));
    let x = _mm256_min_ps(x, _mm256_set1_ps(88.7));
    let inv_ln2 = _mm256_set1_ps(1.442695040888963);
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
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

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub fn bias_gelu(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for row in 0..m {
        let base = row * n;
        for i in 0..n {
            let x = data[base + i] + bias[i];
            data[base + i] = scalar_gelu(x);
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn silu_inplace(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = data.len() / 8;
    unsafe {
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
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub fn silu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
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
        let mut vsum = vdupq_n_f32(0.0);
        let mut vsumsq = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let x = vld1q_f32(input.as_ptr().add(c * 4));
            vsum = vaddq_f32(vsum, x);
            vsumsq = vfmaq_f32(vsumsq, x, x);
        }
        let mut sum = vaddvq_f32(vsum);
        let mut sumsq = vaddvq_f32(vsumsq);
        for i in (chunks * 4)..h {
            sum += input[i];
            sumsq += input[i] * input[i];
        }
        let mean = sum * inv_hf;
        let var = sumsq * inv_hf - mean * mean;
        let inv = 1.0 / (var + eps).sqrt();
        let vmean = vdupq_n_f32(mean);
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn layer_norm_row(
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
        let mut vsum = _mm256_setzero_ps();
        let mut vsumsq = _mm256_setzero_ps();
        for c in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(c * 8));
            vsum = _mm256_add_ps(vsum, x);
            vsumsq = _mm256_fmadd_ps(x, x, vsumsq);
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
        let hsumsq = {
            let lo = _mm256_castps256_ps128(vsumsq);
            let hi = _mm256_extractf128_ps::<1>(vsumsq);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
            _mm_cvtss_f32(s1)
        };
        let mut sum = hsum;
        let mut sumsq = hsumsq;
        for i in (chunks * 8)..h {
            sum += input[i];
            sumsq += input[i] * input[i];
        }
        let mean = sum * inv_hf;
        let var = sumsq * inv_hf - mean * mean;
        let inv = 1.0 / (var + eps).sqrt();
        let vmean = _mm256_set1_ps(mean);
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

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub fn layer_norm_row(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    let inv_hf = 1.0 / h as f32;
    let mut sum = 0f32;
    let mut sumsq = 0f32;
    for i in 0..h {
        sum += input[i];
        sumsq += input[i] * input[i];
    }
    let mean = sum * inv_hf;
    let var = sumsq * inv_hf - mean * mean;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..h {
        output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
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

// ── NEON softmax ────────────────────────────────────────────────────────

/// NEON-vectorized softmax: 3-pass (max, exp+sum, normalize).
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
    use std::arch::x86_64::*;
    let chunks = cols / 8;
    unsafe {
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
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub fn neon_softmax(data: &mut [f32], rows: usize, cols: usize) {
    crate::naive::softmax(data, rows, cols);
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub fn gelu_inplace(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = data.len() / 8;
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
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
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
    const C: f32 = 0.797_884_56; // √(2/π)
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
    let done = rows * chunk;
    if done < len {
        gelu_approx_inplace(&mut data[done..]);
    }
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
    let done = rows * chunk;
    if done < len {
        gelu_inplace(&mut data[done..]);
    }
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
    let done = rows * chunk;
    if done < len {
        silu_inplace(&mut data[done..]);
    }
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
    sign * (1.0 - y * (-xa * xa).exp())
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
