// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_numpy::tanh` against `np.tanh` (NumPy 2.3.5, arm64), bit-for-bit, on
//! 25 000 random arguments plus the special values (±0, tiny, saturation, ±inf, NaN).

#[test]
fn tanh_matches_numpy() {
    let text = include_str!("vectors/tanh.txt");
    let mut n = 0;
    let mut bad = Vec::new();
    for l in text.lines() {
        let mut t = l
            .split_whitespace()
            .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()));
        let (x, want) = (t.next().unwrap(), t.next().unwrap());
        let got = rlx_numpy::tanh(x);
        n += 1;
        if !(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())) {
            bad.push(format!("tanh({x:e}) = {got:e}, numpy {want:e}"));
        }
    }
    assert!(n > 20000);
    assert!(
        bad.is_empty(),
        "{} of {n} differ, e.g. {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
    // and it is not libm's
    let libm_diff = text
        .lines()
        .filter(|l| {
            let x = f64::from_bits(
                u64::from_str_radix(l.split_whitespace().next().unwrap(), 16).unwrap(),
            );
            x.tanh().to_bits() != rlx_numpy::tanh(x).to_bits() && !x.is_nan()
        })
        .count();
    assert!(
        libm_diff > 1000,
        "expected the kernel to differ from libm tanh (got {libm_diff})"
    );
}
