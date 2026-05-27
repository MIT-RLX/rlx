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
//! Cache dequantized GGUF weight bytes for static params.
//!
//! Qwen3.5 decode with `--packed` was re-dequantizing every K-quant
//! weight on every matmul (hundreds of times per token). Keys are
//! `(k, n, scheme, bytes_hash)` — stable for identical GGUF bytes regardless
//! of arena offset (multiple compiled graphs reuse offsets).

use rlx_ir::quant::QuantScheme;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct DequantKey {
    k: u32,
    n: u32,
    scheme: u8,
    /// Content hash — `w_off` alone collides across compiled graphs.
    bytes_hash: u64,
}

fn weight_bytes_hash(w_bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    w_bytes.hash(&mut hasher);
    hasher.finish()
}

fn scheme_tag(scheme: QuantScheme) -> u8 {
    match scheme {
        QuantScheme::GgufQ4K => 1,
        QuantScheme::GgufQ5K => 2,
        QuantScheme::GgufQ6K => 3,
        QuantScheme::GgufQ8K => 4,
        _ => 255,
    }
}

fn dequant_gguf(w_bytes: &[u8], k: usize, n: usize, scheme: QuantScheme) -> Vec<f32> {
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, k * n),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(w_bytes, k * n),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(w_bytes, k * n),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(w_bytes, k * n),
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
pub fn gguf_weight_f32(
    _w_off: usize,
    w_bytes: &[u8],
    k: usize,
    n: usize,
    scheme: QuantScheme,
) -> Arc<[f32]> {
    if !cache_enabled() {
        return Arc::from(dequant_gguf(w_bytes, k, n, scheme).into_boxed_slice());
    }
    let key = DequantKey {
        k: k as u32,
        n: n as u32,
        scheme: scheme_tag(scheme),
        bytes_hash: weight_bytes_hash(w_bytes),
    };
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(hit) = cache.read().expect("dequant cache poisoned").get(&key) {
        return Arc::clone(hit);
    }
    let dense = dequant_gguf(w_bytes, k, n, scheme);
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
        let hash = weight_bytes_hash(&packed);
        let a = gguf_weight_f32(w_off, &packed, k, n, QuantScheme::GgufQ4K);
        let b = gguf_weight_f32(w_off + 999, &packed, k, n, QuantScheme::GgufQ4K);
        assert!(
            Arc::ptr_eq(&a, &b),
            "same bytes at different offsets should hit"
        );
        let mut other = packed.clone();
        other[0] ^= 0x01;
        let c = gguf_weight_f32(w_off, &other, k, n, QuantScheme::GgufQ4K);
        assert!(!Arc::ptr_eq(&a, &c), "different bytes should miss: {hash}");
    }
}
