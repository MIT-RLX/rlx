// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-exact parity against `scipy.fft` (SciPy 1.17.1; `tests/vectors/scipy_fft.txt`
//! from `tools/gen_vectors.py`). Tolerance is **zero**: every output is compared
//! as its IEEE-754 bit pattern (long outputs through an FNV-1a hash of those bits).

/// SciPy's outputs on the reference for this target: macOS arm64 on Apple targets,
/// the Linux x86-64 wheel (msi) elsewhere.
#[cfg(target_vendor = "apple")]
const VECTORS: &str = include_str!("vectors/scipy_fft.txt");
#[cfg(not(target_vendor = "apple"))]
const VECTORS: &str = include_str!("vectors/linux-x86_64/scipy_fft.txt");

use rlx_pocketfft::*;

fn lcg(n: usize, seed: u64) -> Vec<f64> {
    (0..n as u64)
        .map(|k| {
            ((k.wrapping_mul(2654435761).wrapping_add(seed)) % (1u64 << 32)) as f64 / 4294967296.0
                - 0.5
        })
        .collect()
}
fn cplx(v: &[f64]) -> Vec<C64> {
    v.chunks(2).map(|c| [c[0], c[1]]).collect()
}
fn flat(v: &[C64]) -> Vec<f64> {
    v.iter().flat_map(|c| [c[0], c[1]]).collect()
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
fn norm(s: &str) -> Norm {
    match s {
        "backward" => Norm::Backward,
        "ortho" => Norm::Ortho,
        "forward" => Norm::Forward,
        o => panic!("norm {o}"),
    }
}
/// `Err` describes the first difference.
fn check(got: &[f64], want: &[&str]) -> Result<(), String> {
    if want[0] == "fnv" {
        let (h, len) = (
            u64::from_str_radix(want[1], 16).unwrap(),
            want[2].parse::<usize>().unwrap(),
        );
        if got.len() != len {
            return Err(format!("length {} != {len}", got.len()));
        }
        return if fnv(got) == h {
            Ok(())
        } else {
            Err("hash differs".into())
        };
    }
    if got.len() != want.len() {
        return Err(format!("length {} != {}", got.len(), want.len()));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = u64::from_str_radix(w, 16).unwrap();
        if g.to_bits() != w {
            return Err(format!("[{i}] {g:e} != {:e}", f64::from_bits(w)));
        }
    }
    Ok(())
}

#[test]
fn matches_scipy_fft_bit_for_bit() {
    let text = VECTORS;
    let (mut n_ok, mut fails) = (0usize, Vec::new());
    let mut last_rfft: Option<Vec<C64>> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        match t[0] {
            "good_real" | "good_cmplx" => {
                for kv in &t[1..] {
                    let (n, g) = kv.split_once(':').unwrap();
                    let (n, g): (usize, usize) = (n.parse().unwrap(), g.parse().unwrap());
                    let got = if t[0] == "good_real" {
                        next_fast_len_real(n)
                    } else {
                        next_fast_len_cmplx(n)
                    };
                    assert_eq!(got, g, "{} {n}", t[0]);
                }
                continue;
            }
            _ => {}
        }
        let (len, n, nm) = (
            t[1].parse::<usize>().unwrap(),
            t[2].parse::<usize>().unwrap(),
            norm(t[3]),
        );
        let want = &t[4..];
        let x = lcg(len, len as u64);
        let got: Vec<f64> = match t[0] {
            "rfft" => {
                let r = rfft(&x, Some(n), nm);
                last_rfft = Some(r.clone());
                flat(&r)
            }
            // the input is the rfft line just before, at the same norm
            "irfft" => irfft(last_rfft.as_ref().unwrap(), Some(n), nm),
            "irfft_n" => irfft(&rfft(&x, None, Norm::Backward), Some(n), nm),
            "fft" => flat(&fft(&cplx(&lcg(2 * len, 7 * len as u64)), None, nm)),
            "ifft" => flat(&ifft(&cplx(&lcg(2 * len, 7 * len as u64)), None, nm)),
            o => panic!("op {o}"),
        };
        match check(&got, want) {
            Ok(()) => n_ok += 1,
            Err(e) => fails.push(format!("{} len={len} n={n} {}: {e}", t[0], t[3])),
        }
    }
    assert!(
        fails.is_empty(),
        "{} of {} differ:\n{}",
        fails.len(),
        fails.len() + n_ok,
        fails.join("\n")
    );
    assert!(n_ok > 1200, "only {n_ok} cases ran");
}
