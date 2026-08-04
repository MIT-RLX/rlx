// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Muon for the SynthMatMul codebook + KAN spline coefficients. Both are 2-D
//! (`[num_entries, entry_dim]`, `[channels, num_basis]`), so Muon must
//! orthogonalize them via Newton-Schulz — NOT fall back to SGD-with-momentum
//! (which is what happens silently if a param is presented with a non-2-D
//! shape). This pins that the 2-D path fires for our shapes.

use rlx_optim::{Muon, Optimizer};

#[test]
fn muon_orthogonalizes_codebook_and_coeff_shapes() {
    let lr = 0.1f32;
    let grad: Vec<f32> = (0..8 * 4).map(|i| (i as f32 * 0.13).sin()).collect();

    // Codebook [8, 4] — 2-D → Newton-Schulz orthogonalized update.
    let mut m2 = Muon::new(lr);
    let mut p2 = vec![0f32; 8 * 4];
    m2.step("cb", &[8, 4], &mut p2, &grad);

    // Same values as a flat [32] param — 1-D → SGD-with-momentum fallback.
    let mut m1 = Muon::new(lr);
    let mut p1 = vec![0f32; 8 * 4];
    m1.step("cb", &[32], &mut p1, &grad);

    assert!(p2.iter().all(|v| v.is_finite()));
    // The orthogonalized 2-D update must differ from the raw-gradient 1-D one —
    // proving the Newton-Schulz (matrix) path was taken for the codebook shape.
    let diff: f32 = p2.iter().zip(&p1).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-2,
        "2-D Muon (Newton-Schulz) must differ from 1-D SGD fallback (diff={diff})"
    );

    // KAN coeff [16, 6] — also 2-D → orthogonalized, finite, non-trivial.
    let cg: Vec<f32> = (0..16 * 6).map(|i| (i as f32 * 0.07).cos()).collect();
    let mut mc = Muon::new(lr);
    let mut pc = vec![0f32; 16 * 6];
    mc.step("coeff", &[16, 6], &mut pc, &cg);
    assert!(pc.iter().all(|v| v.is_finite()));
    assert!(
        pc.iter().any(|v| v.abs() > 1e-6),
        "coeff update should be non-trivial"
    );
}
