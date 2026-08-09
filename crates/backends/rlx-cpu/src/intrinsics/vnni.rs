// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! x86-64 VNNI int8 dot kernels for low-bit quant matmul.
//!
//! Measured on VNNI-capable CPUs, keeping 2-bit weights and int8-quantized
//! activations in their packed byte form and running `VPDPBUSD`
//! (`u8 × i8 → i32` fused multiply-accumulate) directly on them avoids
//! materializing an f32 weight slab — the dominant cost of the
//! dequant→f32→BLAS path for low-bit models.
//!
//! Thicker than the pure ISA wrappers in `neon.rs`/`apple_amx.rs`: for a
//! 2-bit dot the code unpack and the `dpbusd` accumulate are inseparable, so
//! the whole per-block kernel lives here where a reader expects SIMD. The
//! `Q2_0` block layout is defined in `rlx_gguf::q2_dequant`; the activation
//! block is `d_x`(f32) · 128×i8 · `xsum`(i32).
//!
//! Two encodings are selected at runtime:
//!   - AVX-512-VNNI + VL  → `_mm256_dpbusd_epi32`
//!   - AVX-VNNI (no 512)  → `_mm256_dpbusd_avx_epi32`
//! Neither saturates, so results are bit-identical to the scalar reference.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;
use rlx_gguf::q2_dequant::QK2_0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    None,
    Avx,
    Avx512,
}

#[inline]
fn detect() -> Kind {
    // `is_x86_feature_detected!` caches after first call, so this is a cheap
    // relaxed load in the row loop.
    if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
        Kind::Avx512
    } else if is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") {
        Kind::Avx
    } else {
        Kind::None
    }
}

/// True when this CPU can run the VNNI dot (else callers use the scalar dot).
#[inline]
pub fn has_vnni() -> bool {
    detect() != Kind::None
}

/// Dot one packed `Q2_0` weight block (34 B) with one packed int8 activation
/// block. Dispatches to the best available VNNI encoding, or the scalar
/// reference when the CPU has no VNNI.
#[inline]
pub fn q2_0_dot_q8_g128(w: &[u8], a: &[u8]) -> f32 {
    match detect() {
        // SAFETY: gated by runtime feature detection matching each fn's
        // `target_feature`; slice lengths are the fixed Q2_0 / Q8_0_G128 blocks.
        Kind::Avx512 => unsafe { dot_avx512(w, a) },
        Kind::Avx => unsafe { dot_avx(w, a) },
        Kind::None => rlx_gguf::q2_dequant::q2_0_dot_q8_g128(w, a),
    }
}

/// Expand 8 packed bytes (32 LSB-first 2-bit codes) into 32 `u8` lanes in
/// natural order, each in `0..=3`. Matches `(qs[j/4] >> ((j%4)*2)) & 3`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack32(p: __m128i) -> __m256i {
    // Replicate each of the 8 source bytes across 4 output lanes: lane
    // (4i+r) holds source byte i (r = 0..3).
    let bcast = _mm256_broadcastsi128_si256(p);
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, // lo 128: bytes 0..3
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, // hi 128: bytes 4..7
    );
    let rep = _mm256_shuffle_epi8(bcast, shuf);
    // Extract the r-th 2-bit field per lane: (byte * 2^(6-2r)) >> 6 & 3.
    // Done in 16-bit lanes to keep the intermediate product exact.
    let mul = _mm256_setr_epi16(64, 16, 4, 1, 64, 16, 4, 1, 64, 16, 4, 1, 64, 16, 4, 1);
    let three = _mm256_set1_epi16(3);
    let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep)); // lanes 0..15
    let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(rep)); // lanes 16..31
    let lo = _mm256_and_si256(_mm256_srli_epi16::<6>(_mm256_mullo_epi16(lo, mul)), three);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<6>(_mm256_mullo_epi16(hi, mul)), three);
    // packus interleaves the two 128-bit halves; permute 64-bit groups back
    // to natural order [0,1,2,3] from the packed [0,2,1,3].
    let packed = _mm256_packus_epi16(lo, hi);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot_avx(w: &[u8], a: &[u8]) -> f32 {
    let d = half::f16::from_le_bytes([w[0], w[1]]).to_f32();
    let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
    let xsum = i32::from_le_bytes([a[4 + QK2_0], a[5 + QK2_0], a[6 + QK2_0], a[7 + QK2_0]]);
    unsafe {
        let qs = w.as_ptr().add(2);
        let acts = a.as_ptr().add(4);
        let mut acc = _mm256_setzero_si256();
        for g in 0..(QK2_0 / 32) {
            let codes = unpack32(_mm_loadl_epi64(qs.add(g * 8) as *const __m128i));
            let act = _mm256_loadu_si256(acts.add(g * 32) as *const __m256i);
            acc = _mm256_dpbusd_avx_epi32(acc, codes, act); // u8 codes × i8 acts
        }
        d * dx * (hsum_i32(acc) - xsum) as f32
    }
}

#[inline]
#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
unsafe fn dot_avx512(w: &[u8], a: &[u8]) -> f32 {
    let d = half::f16::from_le_bytes([w[0], w[1]]).to_f32();
    let dx = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
    let xsum = i32::from_le_bytes([a[4 + QK2_0], a[5 + QK2_0], a[6 + QK2_0], a[7 + QK2_0]]);
    unsafe {
        let qs = w.as_ptr().add(2);
        let acts = a.as_ptr().add(4);
        let mut acc = _mm256_setzero_si256();
        for g in 0..(QK2_0 / 32) {
            let codes = unpack32(_mm_loadl_epi64(qs.add(g * 8) as *const __m128i));
            let act = _mm256_loadu_si256(acts.add(g * 32) as *const __m256i);
            acc = _mm256_dpbusd_epi32(acc, codes, act); // u8 codes × i8 acts
        }
        d * dx * (hsum_i32(acc) - xsum) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_gguf::q2_dequant::{Q8_0_G128_BYTES, quantize_q2_0, quantize_q8_0_g128_row};

    #[test]
    fn vnni_matches_scalar() {
        if !has_vnni() {
            eprintln!("skip: no VNNI on this CPU");
            return;
        }
        // Two 128-groups so the row helper / multi-block path is exercised.
        let mut wf = vec![0f32; 2 * QK2_0];
        let mut xf = vec![0f32; 2 * QK2_0];
        for j in 0..wf.len() {
            wf[j] = (((j * 5) % 4) as i32 - 1) as f32 * 0.25; // codes 0..3 → -.25..+.5
            xf[j] = ((j as f32) * 0.021).cos() * 2.5;
        }
        let w = quantize_q2_0(&wf).unwrap();
        let mut a = vec![0u8; 2 * Q8_0_G128_BYTES];
        quantize_q8_0_g128_row(&xf, &mut a);
        for b in 0..2 {
            let wb = &w[b * 34..(b + 1) * 34];
            let ab = &a[b * Q8_0_G128_BYTES..(b + 1) * Q8_0_G128_BYTES];
            let simd = q2_0_dot_q8_g128(wb, ab);
            let scal = rlx_gguf::q2_dequant::q2_0_dot_q8_g128(wb, ab);
            assert!(
                (simd - scal).abs() <= 1e-4 * (1.0 + scal.abs()),
                "{simd} vs {scal}"
            );
        }
    }

    #[test]
    #[ignore = "perf microbench; run with --ignored --nocapture on VNNI hw"]
    fn bench_q2_0_dot() {
        use std::time::Instant;
        if !has_vnni() {
            eprintln!("skip: no VNNI on this CPU");
            return;
        }
        let (k, n, iters) = (4096usize, 4096usize, 50usize);
        let bpr = k / QK2_0;
        let wf: Vec<f32> = (0..k * n)
            .map(|i| (((i * 7) % 4) as i32 - 1) as f32 * 0.1)
            .collect();
        let w = quantize_q2_0(&wf).unwrap();
        let xf: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut aq = vec![0u8; bpr * Q8_0_G128_BYTES];
        quantize_q8_0_g128_row(&xf, &mut aq);
        let row_bytes = bpr * 34;

        let run = |dot: &dyn Fn(&[u8], &[u8]) -> f32| {
            let mut sink = 0f32;
            let t = Instant::now();
            for _ in 0..iters {
                for j in 0..n {
                    let row = &w[j * row_bytes..(j + 1) * row_bytes];
                    for b in 0..bpr {
                        sink += dot(
                            &row[b * 34..b * 34 + 34],
                            &aq[b * Q8_0_G128_BYTES..(b + 1) * Q8_0_G128_BYTES],
                        );
                    }
                }
            }
            (t.elapsed().as_secs_f64(), sink)
        };
        let (vnni, s0) = run(&q2_0_dot_q8_g128);
        let (scal, s1) = run(&rlx_gguf::q2_dequant::q2_0_dot_q8_g128);
        eprintln!(
            "q2_0 GEMV {k}x{n} x{iters} (1 thread): VNNI {vnni:.3}s  scalar {scal:.3}s  \
             speedup {:.2}x  (sink {s0:.1}/{s1:.1})",
            scal / vnni
        );
    }
}
