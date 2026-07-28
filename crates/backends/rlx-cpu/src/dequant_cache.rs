// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cache dequantized GGUF weight bytes for static params.
//!
//! Qwen3.5 decode with `--packed` was re-dequantizing every K-quant
//! weight on every matmul (hundreds of times per token). Keys are
//! `(w_off, len, k, n, scheme, sample)` — arena byte offset plus a cheap
//! head/mid/tail sample so distinct tensors that share an offset across
//! graphs do not collide. Do **not** key on the absolute data pointer:
//! the CPU arena base can move between forwards, which would treat every
//! token as a cache miss and OOM after duplicating f32 weights.

use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct DequantKey {
    w_off: u64,
    len: u64,
    k: u32,
    n: u32,
    scheme: u8,
    /// Cheap content sample (not a full-slab hash).
    sample: u64,
}

fn weight_bytes_sample(w_bytes: &[u8]) -> u64 {
    // O(1) content sample — full-slab DefaultHasher on every lookup made
    // decode dominate on large GGUF mats (lm_head alone is hundreds of MiB).
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let n = w_bytes.len();
    n.hash(&mut hasher);
    if n == 0 {
        return hasher.finish();
    }
    const CHUNK: usize = 64;
    let head = CHUNK.min(n);
    w_bytes[..head].hash(&mut hasher);
    if n > CHUNK * 2 {
        let mid = n / 2;
        let half = CHUNK / 2;
        w_bytes[mid - half..mid + half].hash(&mut hasher);
        w_bytes[n - CHUNK..].hash(&mut hasher);
    } else if n > CHUNK {
        w_bytes[n - head..].hash(&mut hasher);
    }
    hasher.finish()
}

fn scheme_tag(scheme: QuantScheme) -> u8 {
    match scheme {
        QuantScheme::GgufQ4K => 1,
        QuantScheme::GgufQ5K => 2,
        QuantScheme::GgufQ6K => 3,
        QuantScheme::GgufQ8K => 4,
        QuantScheme::GgufQ4_0 => 5,
        QuantScheme::GgufQ8_0 => 6,
        QuantScheme::GgufQ2K => 7,
        QuantScheme::GgufQ3K => 8,
        QuantScheme::GgufIQ4NL => 9,
        QuantScheme::GgufIQ4XS => 10,
        QuantScheme::GgufIQ2XXS => 11,
        QuantScheme::GgufIQ2XS => 12,
        QuantScheme::GgufIQ2S => 13,
        QuantScheme::GgufIQ3XXS => 14,
        QuantScheme::GgufIQ3S => 15,
        QuantScheme::GgufIQ1S => 16,
        QuantScheme::GgufIQ1M => 17,
        QuantScheme::GgufTQ1_0 => 18,
        QuantScheme::GgufTQ2_0 => 19,
        QuantScheme::GgufMXFP4 => 20,
        QuantScheme::GgufNVFP4 => 21,
        QuantScheme::GgufQ4_1 => 22,
        QuantScheme::GgufQ5_0 => 23,
        QuantScheme::GgufQ5_1 => 24,
        QuantScheme::GgufQ1_0 => 25,
        QuantScheme::GgufQ2_0 => 26,
        _ => 255,
    }
}

fn dequant_gguf(w_bytes: &[u8], k: usize, n: usize, scheme: QuantScheme) -> Vec<f32> {
    let n_elems = k * n;
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, n_elems),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(w_bytes, n_elems),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(w_bytes, n_elems),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(w_bytes, n_elems),
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k(w_bytes, n_elems),
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k(w_bytes, n_elems),
        QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0(w_bytes, n_elems),
        QuantScheme::GgufQ4_1 => rlx_gguf::dequant_q4_1(w_bytes, n_elems),
        QuantScheme::GgufQ5_0 => rlx_gguf::dequant_q5_0(w_bytes, n_elems),
        QuantScheme::GgufQ5_1 => rlx_gguf::dequant_q5_1(w_bytes, n_elems),
        QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0(w_bytes, n_elems),
        QuantScheme::GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(w_bytes, n_elems),
        QuantScheme::GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(w_bytes, n_elems),
        QuantScheme::GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(w_bytes, n_elems),
        QuantScheme::GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(w_bytes, n_elems),
        QuantScheme::GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(w_bytes, n_elems),
        QuantScheme::GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(w_bytes, n_elems),
        QuantScheme::GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(w_bytes, n_elems),
        QuantScheme::GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(w_bytes, n_elems),
        QuantScheme::GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(w_bytes, n_elems),
        QuantScheme::GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(w_bytes, n_elems),
        QuantScheme::GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(w_bytes, n_elems),
        QuantScheme::GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(w_bytes, n_elems),
        QuantScheme::GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(w_bytes, n_elems),
        QuantScheme::GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(w_bytes, n_elems),
        QuantScheme::GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(w_bytes, n_elems),
        other => panic!("dequant_cache: unsupported GGUF scheme {other:?}"),
    }
    .expect("GGUF dequant failed")
}

static CACHE: OnceLock<RwLock<HashMap<DequantKey, Arc<[f32]>>>> = OnceLock::new();

fn cache_enabled() -> bool {
    !matches!(
        rlx_ir::env::var("RLX_DEQUANT_CACHE").as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Return dense `[k×n]` weights (GGUF row-major `[n,k]` layout) for `w_bytes`.
///
/// `w_off` is the arena byte offset of the quantized slab (stable across
/// arena reallocations that preserve layout).
pub fn gguf_weight_f32(
    w_off: usize,
    w_bytes: &[u8],
    k: usize,
    n: usize,
    scheme: QuantScheme,
) -> Arc<[f32]> {
    if !cache_enabled() {
        return Arc::from(dequant_gguf(w_bytes, k, n, scheme).into_boxed_slice());
    }
    let key = DequantKey {
        w_off: w_off as u64,
        len: w_bytes.len() as u64,
        k: k as u32,
        n: n as u32,
        scheme: scheme_tag(scheme),
        sample: weight_bytes_sample(w_bytes),
    };
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(hit) = cache.read().expect("dequant cache poisoned").get(&key) {
        return Arc::clone(hit);
    }
    let dense = dequant_gguf(w_bytes, k, n, scheme);
    // Cap individual entries so a pathological shape cannot ask for multi-GiB
    // slabs; Qwen3-0.6B lm_head is ~622 MiB f32 and should still cache.
    const MAX_CACHED_ELEMS: usize = 256 * 1024 * 1024; // 256M f32 ≈ 1 GiB
    if dense.len() > MAX_CACHED_ELEMS {
        return Arc::from(dense.into_boxed_slice());
    }
    let arc: Arc<[f32]> = Arc::from(dense.into_boxed_slice());
    cache
        .write()
        .expect("dequant cache poisoned")
        .insert(key, Arc::clone(&arc));
    arc
}

/// Drop cached dequantized weights (e.g. between model loads in tests).
pub fn clear_dequant_cache() {
    if let Some(c) = CACHE.get() {
        c.write().expect("dequant cache poisoned").clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_dequant_cache_hits_on_second_lookup() {
        clear_dequant_cache();
        const QK_K: usize = 256;
        let mut packed = Vec::new();
        packed.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        packed.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut scales = [0u8; 12];
        for s in &mut scales[0..4] {
            *s = 0x01;
        }
        packed.extend_from_slice(&scales);
        packed.extend(std::iter::repeat_n(0x77u8, QK_K / 2));
        let k = 256;
        let n = 1;
        let w_off = 4096;
        let a = gguf_weight_f32(w_off, &packed, k, n, QuantScheme::GgufQ4K);
        let b = gguf_weight_f32(w_off, &packed, k, n, QuantScheme::GgufQ4K);
        assert!(Arc::ptr_eq(&a, &b), "same offset + bytes should hit");
        // Different arena offsets are distinct slots (base can move; offset is stable).
        let other_off = gguf_weight_f32(w_off + 999, &packed, k, n, QuantScheme::GgufQ4K);
        assert!(
            !Arc::ptr_eq(&a, &other_off),
            "different w_off should miss even with identical bytes"
        );
        let mut other = packed.clone();
        other[0] ^= 0x01;
        let c = gguf_weight_f32(w_off, &other, k, n, QuantScheme::GgufQ4K);
        assert!(
            !Arc::ptr_eq(&a, &c),
            "different bytes at same offset should miss"
        );
    }

    #[test]
    fn gguf_dequant_cache_q4_1_q5_hits() {
        clear_dequant_cache();
        let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        for (scheme, ggml) in [
            (QuantScheme::GgufQ4_1, rlx_gguf::GgmlType::Q4_1),
            (QuantScheme::GgufQ5_0, rlx_gguf::GgmlType::Q5_0),
            (QuantScheme::GgufQ5_1, rlx_gguf::GgmlType::Q5_1),
        ] {
            let packed = rlx_gguf::quantize(&w, ggml).unwrap();
            let a = gguf_weight_f32(0, &packed, 32, 16, scheme);
            let b = gguf_weight_f32(0, &packed, 32, 16, scheme);
            assert!(Arc::ptr_eq(&a, &b), "cache hit expected for {scheme:?}");
        }
    }
}
