// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every one-argument f32 port on all 2^32 inputs, against glibc 2.43, on any
//! OS: `tests/vectors/glibc_*_2_43_f32_exhaustive.txt` hold hashes of glibc's
//! outputs (`tools/glibc-exhaustive`, which also lists any mismatch). Slow
//! (minutes on every core), so ignored by default: `just test-libm-exhaustive`.

#[path = "common/f32_unary.rs"]
mod f32_unary;

use f32_unary::*;

fn check(file: &str, text: &str, aarch64: bool) {
    let want: std::collections::HashMap<&str, &str> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_once(' ').unwrap())
        .collect();
    let mut bad = Vec::new();
    for (name, port) in f32_unary(aarch64) {
        let w = want
            .get(name)
            .unwrap_or_else(|| panic!("{file}: no line for {name}"));
        let h = format!("{:016x}", exhaustive_hash(&port));
        eprintln!("{name:12} {}", if h == *w { "ok" } else { "DIFFERS" });
        if h != *w {
            bad.push(name);
        }
    }
    assert!(
        bad.is_empty(),
        "{file}: differ from glibc on some f32 input: {bad:?}"
    );
}

#[test]
#[ignore = "all 2^32 inputs of every f32 function: `just test-libm-exhaustive`"]
fn every_f32_port_matches_glibc_on_every_input_x86_64() {
    check(
        "glibc_x86_64_2_43_f32_exhaustive.txt",
        include_str!("vectors/glibc_x86_64_2_43_f32_exhaustive.txt"),
        false,
    );
}

#[test]
#[ignore = "all 2^32 inputs of every f32 function: `just test-libm-exhaustive`"]
fn every_f32_port_matches_glibc_on_every_input_aarch64() {
    check(
        "glibc_aarch64_2_43_f32_exhaustive.txt",
        include_str!("vectors/glibc_aarch64_2_43_f32_exhaustive.txt"),
        true,
    );
}
