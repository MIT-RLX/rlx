// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every one-argument `f32` function rlx-libm exports, by its glibc name, and
//! the exhaustive hash both `tools/glibc-exhaustive` (of glibc's outputs) and
//! `tests/f32_exhaustive.rs` (of the ports') compute over all 2^32 inputs.

use rlx_libm as l;

/// One f32 port, by glibc name.
pub type F32Fn = (&'static str, fn(f32) -> f32);

/// The functions of the glibc build named: `aarch64` swaps in
/// `rlx_libm::aarch64::expf`, the one f32 function whose builds differ.
pub fn f32_unary(aarch64: bool) -> Vec<F32Fn> {
    let mut v = F32_UNARY.to_vec();
    if aarch64 {
        for e in &mut v {
            if e.0 == "expf" {
                e.1 = l::aarch64::expf;
            }
        }
    }
    v
}

const F32_UNARY: &[F32Fn] = &[
    ("acosf", l::acosf),
    ("asinf", l::asinf),
    ("atanf", l::atanf),
    ("acoshf", l::acoshf),
    ("asinhf", l::asinhf),
    ("atanhf", l::atanhf),
    ("cbrtf", l::cbrtf),
    ("coshf", l::coshf),
    ("sinhf", l::sinhf),
    ("tanhf", l::tanhf),
    ("tanf", l::tanf),
    ("sinf", |x| l::sincosf(x).0),
    ("cosf", |x| l::sincosf(x).1),
    ("expf", l::expf),
    ("exp2f", l::exp2f),
    ("exp10f", l::exp10f),
    ("expm1f", l::expm1f),
    ("logf", l::logf),
    ("log2f", l::log2f),
    ("log10f", l::log10f),
    ("log1pf", l::log1pf),
    ("erff", l::erff),
    ("erfcf", l::erfcf),
    ("lgammaf", l::lgammaf),
    ("tgammaf", l::tgammaf),
    ("sqrtf", l::sqrtf),
    ("floorf", l::floorf),
    ("ceilf", l::ceilf),
    ("truncf", l::truncf),
    ("roundf", l::roundf),
    ("rintf", l::rintf),
    ("roundevenf", l::rintf),
];

/// Output bits for hashing, any NaN as the canonical quiet NaN.
pub fn canon(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
}

/// 256 chunks of 2^24 inputs (chunk `c` = bit patterns `c << 24 ..`).
pub const CHUNKS: u32 = 256;

pub fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}
pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over the outputs of one chunk.
pub fn chunk_hash(c: u32, f: &dyn Fn(f32) -> f32) -> u64 {
    let mut h = FNV0;
    for i in (c << 24)..=((c << 24) | 0x00ff_ffff) {
        h = fnv(h, &canon(f(f32::from_bits(i))).to_le_bytes());
    }
    h
}

/// The exhaustive hash: FNV-1a over the chunk hashes in order, the chunks
/// computed on every core.
pub fn exhaustive_hash(f: &(dyn Fn(f32) -> f32 + Sync)) -> u64 {
    use std::sync::atomic::{AtomicU32, Ordering};
    let next = AtomicU32::new(0);
    let mut hs = vec![0u64; CHUNKS as usize];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let parts: Vec<Vec<(u32, u64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= CHUNKS {
                            return v;
                        }
                        v.push((c, chunk_hash(c, f)));
                    }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (c, h) in parts.into_iter().flatten() {
        hs[c as usize] = h;
    }
    hs.iter().fold(FNV0, |h, c| fnv(h, &c.to_le_bytes()))
}
