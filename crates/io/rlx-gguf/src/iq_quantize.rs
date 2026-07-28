// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IQ-family GGUF encoders. IQ4NL / IQ4XS follow `quantize_row_iq4_nl_impl`
//! from llama.cpp; IQ2/3/1 use kmap + sign-extraction (llama.cpp-style,
//! uniform weights — no imatrix). Output round-trips through
//! [`crate::iq_dequant`].

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::QK_K;
use crate::iq_dequant::QK4_NL;
use crate::iq_encode_common::{GROUP_MAX_EPS, f16_bytes, nearest_int};
use crate::iq_grids::KVALUES_IQ4NL;

const IQ4_NL_BLOCK_BYTES: usize = 2 + QK4_NL / 2;
const IQ4XS_BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
const IQ2XXS_BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2;
const IQ2XS_BLOCK_BYTES: usize = 2 + (QK_K / 8) * 2 + QK_K / 32;
const IQ2S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32;
const IQ3XXS_BLOCK_BYTES: usize = 2 + 3 * (QK_K / 8);
const IQ3S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64;
const IQ1S_BLOCK_BYTES: usize = 2 + QK_K / 8 + (QK_K / 32) * 2;
const IQ1M_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 16 + QK_K / 32;

fn best_index_kvalues(values: &[i8; 16], x: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = (x - values[0] as f32).abs();
    for i in 1..16 {
        let err = (x - values[i] as f32).abs();
        if err < best_err {
            best = i as u8;
            best_err = err;
        }
    }
    best
}

/// Shared IQ4NL / IQ4XS block quantizer (`quantize_row_iq4_nl_impl`).
fn quantize_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],
    ntry: i32,
) -> (Vec<u8>, [u8; 2], u16, [u8; 4]) {
    let mut q4 = vec![0u8; super_block_size / 2];
    let mut dh = [0u8; 2];
    let mut scales_h = 0u16;
    let mut scales_l = [0u8; 4];
    let mut l_buf = vec![0u8; super_block_size];
    let mut scales = vec![0f32; super_block_size / block_size];
    let mut weight = vec![0f32; block_size];

    let mut sigma2 = 0f32;
    for &v in &x[..super_block_size] {
        sigma2 += v * v;
    }
    sigma2 *= 2.0 / super_block_size as f32;

    let nblocks = super_block_size / block_size;
    let mut max_scale = 0f32;
    let mut amax_scale = 0f32;

    for ib in 0..nblocks {
        let xb = &x[ib * block_size..(ib + 1) * block_size];
        let lb = &mut l_buf[ib * block_size..(ib + 1) * block_size];
        for j in 0..block_size {
            weight[j] = sigma2 + xb[j] * xb[j];
        }
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let mut d = if ntry > 0 {
            -max / KVALUES_IQ4NL[0] as f32
        } else {
            max / KVALUES_IQ4NL[0] as f32
        };
        let mut id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut sumqx = 0f32;
        let mut sumq2 = 0f32;
        for j in 0..block_size {
            let l = best_index_kvalues(&KVALUES_IQ4NL, id * xb[j]);
            lb[j] = l;
            let q = KVALUES_IQ4NL[l as usize] as f32;
            let w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        if ntry > 0 {
            for itry in -ntry..=ntry {
                id = (itry as f32 + KVALUES_IQ4NL[0] as f32) / max;
                sumqx = 0.0;
                sumq2 = 0.0;
                for j in 0..block_size {
                    let l = best_index_kvalues(&KVALUES_IQ4NL, id * xb[j]);
                    let q = KVALUES_IQ4NL[l as usize] as f32;
                    let w = weight[j];
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    d = sumqx / sumq2;
                    best = d * sumqx;
                    for j in 0..block_size {
                        lb[j] = best_index_kvalues(
                            &KVALUES_IQ4NL,
                            (if d != 0.0 { 1.0 / d } else { 0.0 }) * xb[j],
                        );
                    }
                }
            }
        }
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if nblocks > 1 {
        let nb = nblocks;
        let d = -max_scale / 32.0;
        dh.copy_from_slice(&f16_bytes(d));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..nb {
            let mut l = nearest_int(id * scales[ib]);
            l = l.clamp(-32, 31);
            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let lb = &mut l_buf[ib * block_size..(ib + 1) * block_size];
            let xb = &x[ib * block_size..(ib + 1) * block_size];
            for j in 0..block_size {
                lb[j] = best_index_kvalues(&KVALUES_IQ4NL, idl * xb[j]);
            }
            l += 32;
            let l_l = (l & 0xf) as u8;
            let l_h = (l >> 4) as u8;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h |= (l_h as u16) << (2 * (ib % 8));
        }
    } else {
        dh.copy_from_slice(&f16_bytes(scales[0]));
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..super_block_size {
                l_buf[j] = best_index_kvalues(&KVALUES_IQ4NL, id * x[j]);
            }
        }
    }

    for i in 0..super_block_size / 32 {
        for j in 0..16 {
            q4[16 * i + j] = l_buf[32 * i + j] | (l_buf[32 * i + 16 + j] << 4);
        }
    }
    (q4, dh, scales_h, scales_l)
}

pub fn quantize_iq4_nl_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK4_NL && out.len() >= IQ4_NL_BLOCK_BYTES);
    let (q4, dh, _, _) = quantize_iq4_nl_impl(QK4_NL, 32, src, -1);
    out[0..2].copy_from_slice(&dh);
    out[2..].copy_from_slice(&q4);
}

pub fn quantize_iq4_xs_block(src: &[f32], out: &mut [u8]) {
    assert!(src.len() >= QK_K && out.len() >= IQ4XS_BLOCK_BYTES);
    let (q4, dh, scales_h, scales_l) = quantize_iq4_nl_impl(QK_K, 32, src, 7);
    out[0..2].copy_from_slice(&dh);
    out[2..4].copy_from_slice(&scales_h.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l);
    out[8..].copy_from_slice(&q4);
}

macro_rules! block_quantize_iq {
    ($name:ident, $block_fn:path, $block_bytes:expr) => {
        pub fn $name(src: &[f32]) -> Result<Vec<u8>> {
            if !src.len().is_multiple_of(QK_K) {
                bail!(
                    "{}: n={} not divisible by {QK_K}",
                    stringify!($name),
                    src.len()
                );
            }
            let nb = src.len() / QK_K;
            let blk = $block_bytes;
            let mut out = vec![0u8; nb * blk];
            out.par_chunks_mut(blk)
                .zip(src.par_chunks(QK_K))
                .for_each(|(ob, sb)| $block_fn(sb, ob));
            Ok(out)
        }
    };
}

macro_rules! block_quantize_iq4 {
    ($name:ident, $block_fn:ident, $block_size:expr, $block_bytes:expr) => {
        pub fn $name(src: &[f32]) -> Result<Vec<u8>> {
            if !src.len().is_multiple_of($block_size) {
                bail!(
                    "{}: n={} not divisible by {}",
                    stringify!($name),
                    src.len(),
                    $block_size
                );
            }
            let nb = src.len() / $block_size;
            let blk = $block_bytes;
            let mut out = vec![0u8; nb * blk];
            out.par_chunks_mut(blk)
                .zip(src.par_chunks($block_size))
                .for_each(|(ob, sb)| $block_fn(sb, ob));
            Ok(out)
        }
    };
}

block_quantize_iq4!(
    quantize_iq4_nl,
    quantize_iq4_nl_block,
    QK4_NL,
    IQ4_NL_BLOCK_BYTES
);
block_quantize_iq4!(
    quantize_iq4_xs,
    quantize_iq4_xs_block,
    QK_K,
    IQ4XS_BLOCK_BYTES
);

block_quantize_iq!(
    quantize_iq2_xxs,
    crate::iq2_encode::quantize_iq2_xxs_block,
    IQ2XXS_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq2_xs,
    crate::iq2_encode::quantize_iq2_xs_block,
    IQ2XS_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq2_s,
    crate::iq2_encode::quantize_iq2_s_block,
    IQ2S_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq3_xxs,
    crate::iq3_encode::quantize_iq3_xxs_block,
    IQ3XXS_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq3_s,
    crate::iq3_encode::quantize_iq3_s_block,
    IQ3S_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq1_s,
    crate::iq1_encode::quantize_iq1_s_block,
    IQ1S_BLOCK_BYTES
);
block_quantize_iq!(
    quantize_iq1_m,
    crate::iq1_encode::quantize_iq1_m_block,
    IQ1M_BLOCK_BYTES
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iq_dequant::{dequant_iq2_xxs, dequant_iq4_nl, dequant_iq4_xs};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn iq4_nl_roundtrip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).sin()).collect();
        let q = quantize_iq4_nl(&x).unwrap();
        let out = dequant_iq4_nl(&q, 64).unwrap();
        assert!(cosine(&x, &out) > 0.98, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn iq4_xs_roundtrip() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.04).cos()).collect();
        let q = quantize_iq4_xs(&x).unwrap();
        let out = dequant_iq4_xs(&q, 256).unwrap();
        assert!(cosine(&x, &out) > 0.95, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn iq2_xxs_roundtrip_smoke() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let q = quantize_iq2_xxs(&x).unwrap();
        let out = dequant_iq2_xxs(&q, 256).unwrap();
        assert!(cosine(&x, &out) > 0.83, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn iq3_xxs_roundtrip_smoke() {
        use crate::iq_dequant::dequant_iq3_xxs;
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.025).cos() * 0.4).collect();
        let q = quantize_iq3_xxs(&x).unwrap();
        let out = dequant_iq3_xxs(&q, 256).unwrap();
        assert!(cosine(&x, &out) > 0.80, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn iq1_m_roundtrip_smoke() {
        use crate::iq_dequant::dequant_iq1_m;
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();
        let q = quantize_iq1_m(&x).unwrap();
        let out = dequant_iq1_m(&q, 256).unwrap();
        assert!(cosine(&x, &out) > 0.75, "cos={}", cosine(&x, &out));
    }

    #[test]
    fn iq1_s_roundtrip_smoke() {
        use crate::iq_dequant::dequant_iq1_s;
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin() * 0.3).collect();
        let q = quantize_iq1_s(&x).unwrap();
        let out = dequant_iq1_s(&q, 256).unwrap();
        assert!(cosine(&x, &out) > 0.80, "cos={}", cosine(&x, &out));
    }
}
