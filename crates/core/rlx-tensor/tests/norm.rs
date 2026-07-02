// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! LLM normalization + rotary-embedding ops, validated end-to-end:
//! `rms_norm`/`layer_norm` against their naive decompositions, and `rope`
//! against the GPT-NeoX rotation (identity + a hand-computed angle).
//! Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn rms_norm_matches_naive() {
    // out = x / sqrt(mean(x^2, -1) + eps) * gamma + beta
    let eps: f64 = 1e-5;
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], [2, 3]);
    let gamma = Tensor::from_vec(vec![1.0, 2.0, 0.5], [3]);
    let beta = Tensor::from_vec(vec![0.1, -0.2, 0.3], [3]);

    let fused = x.rms_norm(&gamma, &beta, eps as f32).to_vec();

    let ms = (&x * &x).mean([1], true); // [2,1]
    let denom = (&ms + eps).sqrt();
    let normed = &x / &denom;
    let naive = (&(&normed * &gamma) + &beta).to_vec();

    approx(&fused, &naive);
}

#[test]
fn layer_norm_matches_naive() {
    // out = (x - mean) / sqrt(var + eps) * gamma + beta
    let eps: f64 = 1e-5;
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0], [2, 3]);
    let gamma = Tensor::from_vec(vec![1.0, 2.0, 0.5], [3]);
    let beta = Tensor::from_vec(vec![0.1, -0.2, 0.3], [3]);

    let fused = x.layer_norm(&gamma, &beta, eps as f32).to_vec();

    let mu = x.mean([1], true); // [2,1]
    let xc = &x - &mu;
    let var = (&xc * &xc).mean([1], true);
    let denom = (&var + eps).sqrt();
    let normed = &xc / &denom;
    let naive = (&(&normed * &gamma) + &beta).to_vec();

    approx(&fused, &naive);
}

#[test]
fn rope_identity_when_cos1_sin0() {
    // cos=1, sin=0 -> rotation is the identity for any layout.
    // x: [b=1, seq=2, hidden=4] (nh=1, head_dim=4). cos/sin len = seq*(dh/2)=4.
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4]);
    let cos = Tensor::from_vec(vec![1.0; 4], [4]);
    let sin = Tensor::from_vec(vec![0.0; 4], [4]);
    let out = x.rope(&cos, &sin, 4).to_vec();
    approx(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn rope_rotates_by_known_angle() {
    // Single position, head_dim=2 (one rotation pair), cos=0.6, sin=0.8.
    // x=[x1,x2]=[1,2] -> [x1*c - x2*s, x2*c + x1*s] = [-1.0, 2.0].
    let x = Tensor::from_vec(vec![1.0, 2.0], [1, 1, 2]);
    let cos = Tensor::from_vec(vec![0.6], [1]);
    let sin = Tensor::from_vec(vec![0.8], [1]);
    let out = x.rope(&cos, &sin, 2).to_vec();
    approx(&out, &[-1.0, 2.0]);
}
