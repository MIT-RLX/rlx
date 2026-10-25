// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tensor.bernoulli_(p)` / `F.dropout` keep masks against torch 2.11 (arm64,
//! no MKL): `torch.manual_seed(3); torch.empty(40).bernoulli_(0.9)`.
use rlx_rng::torch::RandomState;

#[test]
fn bernoulli_matches_torch() {
    let want = "1111111111111101110111111111111111111110";
    let mut r = RandomState::new(3);
    let got: String = (0..40)
        .map(|_| if r.bernoulli(0.9) { '1' } else { '0' })
        .collect();
    assert_eq!(got, want);
}
