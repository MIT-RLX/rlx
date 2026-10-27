// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Both [`engine::Contraction`] choices against the pocketfft header compiled each
//! way. Outputs compare as FNV-1a hashes of their IEEE-754 bits (zero tolerance).

use rlx_pocketfft::engine::{self, Cx};

fn sig(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn fnv(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// `tests/vectors/cpp_contract_{on,off}.txt`: the pinned header built by clang with
/// `-ffp-contract={on,off}` on macOS arm64 (`tools/contraction_ref.cpp`), n = 1..=600.
#[test]
fn contraction_options_match_cpp_builds() {
    // The fixtures carry Apple's merged sincos twiddles: macOS with
    // rlx-libm's `libm` feature (`--features libm`) only.
    if !cfg!(all(target_os = "macos", feature = "libm")) {
        return;
    }
    for (file, contraction) in [
        ("on", engine::Contraction::Clang),
        ("off", engine::Contraction::None),
    ] {
        let text = std::fs::read_to_string(format!(
            "{}/tests/vectors/cpp_contract_{file}.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let o = engine::Options {
            contraction,
            twiddles: engine::Twiddles::AppleSinCos,
        };
        let mut bad = Vec::new();
        for line in text.lines() {
            let t: Vec<&str> = line.split(' ').collect();
            let n: usize = t[0].parse().unwrap();
            let h = |i: usize| u64::from_str_radix(t[i], 16).unwrap();
            let (x, y) = (sig(n, n as u64), sig(n, 7 + n as u64));
            let xs = engine::r2c(&x, true, 1.0, o);
            let flat = |v: &[Cx<f64>]| v.iter().flat_map(|c| [c.r, c.i]).collect::<Vec<_>>();
            let xr = engine::c2r(&xs, n, false, 1.0 / n as f64, o);
            let z: Vec<Cx<f64>> = x.iter().zip(&y).map(|(&r, &i)| Cx::new(r, i)).collect();
            let oz = engine::c2c(&z, true, 1.0, o);
            for (k, got) in [fnv(&flat(&xs)), fnv(&xr), fnv(&flat(&oz))]
                .into_iter()
                .enumerate()
            {
                if got != h(k + 1) {
                    bad.push(format!("{} {n}", ["r2c", "c2r", "c2c"][k]));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{file}: {} mismatches {:?}",
            bad.len(),
            &bad[..bad.len().min(30)]
        );
    }
}
