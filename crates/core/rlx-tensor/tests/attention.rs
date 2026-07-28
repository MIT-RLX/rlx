// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validate the fused `Op::Attention` against the naive decomposition
//! `softmax(Q·Kᵀ · d^-0.5) · V`, both run through the lazy `Tensor`/`Func`
//! pipeline. The headline op for rlx's LLM domain. Run:
//! `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::{Func, MaskKind, shape};

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
    }
}

#[test]
fn fused_attention_matches_naive() {
    // [b=1, n=4, nh=1, dh=4], fed as named inputs (the layout the backends use).
    let n = 4usize;
    let dh = 4usize;
    let qd: Vec<f32> = (0..n * dh).map(|i| (i as f32 * 0.13).sin()).collect();
    let kd: Vec<f32> = (0..n * dh).map(|i| (i as f32 * 0.17).cos()).collect();
    let vd: Vec<f32> = (0..n * dh).map(|i| i as f32 * 0.05 - 0.3).collect();
    let feed: &[(&str, &[f32])] = &[("q", &qd), ("k", &kd), ("v", &vd)];

    let fused = Func::new("attn", |s| {
        let q = s.input("q", shape![1, 4, 1, 4]);
        let k = s.input("k", shape![1, 4, 1, 4]);
        let v = s.input("v", shape![1, 4, 1, 4]);
        q.attention(&k, &v, 1, 4, MaskKind::None)
    })
    .run(feed);

    let naive = Func::new("naive", |s| {
        let q = s.input("q", shape![1, 4, 1, 4]).reshape(vec![4_i64, 4]);
        let k = s.input("k", shape![1, 4, 1, 4]).reshape(vec![4_i64, 4]);
        let v = s.input("v", shape![1, 4, 1, 4]).reshape(vec![4_i64, 4]);
        let scale = (4.0_f32).powf(-0.5);
        let scores = &q.matmul(&k.t()) * scale; // [4,4]
        scores.softmax(1).matmul(&v)
    })
    .run(feed);

    approx(&fused[0], &naive[0]);
}

#[test]
fn fused_attention_nonzero_realistic() {
    // seq != nh to avoid layout mis-disambiguation; check the op produces
    // non-zero output through our pipeline.
    let (b, s, nh, dh) = (1usize, 4usize, 2usize, 4usize);
    let n = b * s * nh * dh;
    let qd: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
    let kd: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
    let vd: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let out = Func::new("a", |g| {
        let q = g.input("q", shape![1, 4, 2, 4]);
        let k = g.input("k", shape![1, 4, 2, 4]);
        let v = g.input("v", shape![1, 4, 2, 4]);
        q.attention(&k, &v, 2, 4, MaskKind::None)
    })
    .run(&[("q", &qd), ("k", &kd), ("v", &vd)]);
    let nonzero = out[0].iter().filter(|x| x.abs() > 1e-9).count();
    eprintln!("ATTN_NONZERO={nonzero}/{}", out[0].len());
    assert!(nonzero > 0, "fused attention all zeros: {:?}", out[0]);
}
