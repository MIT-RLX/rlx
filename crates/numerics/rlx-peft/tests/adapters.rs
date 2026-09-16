// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The properties each adapter exists for.
//!
//! HuggingFace PEFT is not installed here, so these do not diff against its
//! output. Instead they assert the mathematical guarantees the methods are
//! *chosen* for — orthogonality, identity-at-initialisation, the parameter
//! budget — which is the stronger check anyway: an implementation can match a
//! reference on one input and still violate the property that makes the method
//! worth using.

use rlx_peft::{
    LoraConfig, adalora_delta, cayley_orthogonal, dora_weight, ia3_apply, lora_delta,
    lora_param_count,
    oft::{oft_weight, skew},
};

fn mat(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s >> 30;
            s = s.wrapping_mul(0xBF58476D1CE4E5B9);
            s ^= s >> 27;
            s = s.wrapping_mul(0x94D049BB133111EB);
            s ^= s >> 31;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// PEFT initialises `B` to zero, so an untrained LoRA layer reproduces the base
/// model EXACTLY. A nonzero init perturbs a pretrained network before any
/// training — the usual cause of "PEFT made my model worse immediately".
#[test]
fn lora_is_the_identity_at_initialisation() {
    let (i, o, r) = (6, 4, 2);
    let a = mat(r * i, 1);
    let b = vec![0f64; o * r];
    let d = lora_delta(&a, &b, i, o, LoraConfig { r, alpha: 16.0 }).unwrap();
    assert!(d.iter().all(|v| *v == 0.0), "B=0 must give a zero update");
}

#[test]
fn lora_scaling_is_alpha_over_r() {
    let (i, o, r) = (5, 3, 2);
    let (a, b) = (mat(r * i, 2), mat(o * r, 3));
    let d1 = lora_delta(&a, &b, i, o, LoraConfig { r, alpha: r as f64 }).unwrap();
    let d2 = lora_delta(
        &a,
        &b,
        i,
        o,
        LoraConfig {
            r,
            alpha: 4.0 * r as f64,
        },
    )
    .unwrap();
    for (x, y) in d1.iter().zip(&d2) {
        assert!(
            (4.0 * x - y).abs() < 1e-12,
            "alpha must scale the update linearly"
        );
    }
}

#[test]
fn lora_saves_parameters_only_below_the_break_even_rank() {
    // r(in+out) < in*out. For 64x64 the break-even is r = 32.
    assert_eq!(lora_param_count(64, 64, 8), 8 * 128);
    assert!(rlx_peft::lora_is_economical(64, 64, 8));
    assert!(
        !rlx_peft::lora_is_economical(64, 64, 32),
        "at r=32 the factorisation costs as much"
    );
    assert!(!rlx_peft::lora_is_economical(64, 64, 40));
}

/// IA3 starts as the identity too, and adds no weight update at all.
#[test]
fn ia3_with_unit_scales_is_a_no_op() {
    let (rows, out) = (3, 5);
    let y = mat(rows * out, 4);
    let l = vec![1f64; out];
    let got = ia3_apply(&y, &l, rows, out).unwrap();
    assert_eq!(got, y);
    // It rescales per OUTPUT feature, broadcast across rows.
    let mut l2 = vec![1f64; out];
    l2[2] = 3.0;
    let got = ia3_apply(&y, &l2, rows, out).unwrap();
    for r in 0..rows {
        assert!((got[r * out + 2] - 3.0 * y[r * out + 2]).abs() < 1e-12);
        assert!((got[r * out + 1] - y[r * out + 1]).abs() < 1e-12);
    }
}

/// AdaLoRA prunes singular values during training, so zeros in Λ are the normal
/// operating state, not a degenerate input.
#[test]
fn adalora_zeroed_singular_values_drop_their_directions() {
    let (i, o, r) = (6, 4, 3);
    let (p, q) = (mat(o * r, 5), mat(r * i, 6));
    let full = adalora_delta(&p, &[1.0, 1.0, 1.0], &q, i, o, LoraConfig { r, alpha: 3.0 }).unwrap();
    let pruned =
        adalora_delta(&p, &[1.0, 0.0, 1.0], &q, i, o, LoraConfig { r, alpha: 3.0 }).unwrap();
    assert_ne!(
        full, pruned,
        "zeroing a singular value must change the update"
    );
    let none = adalora_delta(&p, &[0.0; 3], &q, i, o, LoraConfig { r, alpha: 3.0 }).unwrap();
    assert!(
        none.iter().all(|v| v.abs() < 1e-15),
        "fully pruned means no update"
    );
}

/// DoRA initialised with the base weight's own row norms is exactly the base
/// weight — the decomposition is lossless before training.
#[test]
fn dora_reproduces_the_base_weight_at_initialisation() {
    let (i, o) = (7, 5);
    let w = mat(o * i, 7);
    let m = rlx_peft::adapters::dora_init_magnitude(&w, i, o).unwrap();
    let zero = vec![0f64; o * i];
    let got = dora_weight(&w, &zero, &m, i, o).unwrap();
    let worst = got
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(worst < 1e-12, "DoRA at init differs from W by {worst:.3e}");
}

/// The whole point of OFT: `R` is orthogonal for ANY parameter value, so
/// training never leaves the manifold and no re-orthogonalisation is needed.
#[test]
fn the_cayley_transform_is_orthogonal_for_any_parameter() {
    for seed in [11u64, 12, 13] {
        for n in [2usize, 4, 8] {
            let q = mat(n * n, seed + n as u64);
            let r = cayley_orthogonal(&q, n).unwrap();
            // RᵀR = I
            for a in 0..n {
                for b in 0..n {
                    let dot: f64 = (0..n).map(|t| r[t * n + a] * r[t * n + b]).sum();
                    let want = f64::from(a == b);
                    assert!(
                        (dot - want).abs() < 1e-10,
                        "RᵀR[{a},{b}] = {dot}, expected {want} (n={n}, seed={seed})"
                    );
                }
            }
        }
    }
}

#[test]
fn oft_is_the_identity_at_zero_and_preserves_norms() {
    let (i, o, bs) = (6, 4, 2);
    let w = mat(o * i, 21);
    let zeros: Vec<Vec<f64>> = (0..o / bs).map(|_| vec![0f64; bs * bs]).collect();
    let got = oft_weight(&w, &zeros, i, o, bs).unwrap();
    let worst = got
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(worst < 1e-12, "Q=0 must give R=I; differs by {worst:.3e}");

    // A rotation preserves the norm of each block's columns — the geometry the
    // pretrained model learned survives adaptation. That is why OFT exists.
    let blocks: Vec<Vec<f64>> = (0..o / bs).map(|b| mat(bs * bs, 30 + b as u64)).collect();
    let rot = oft_weight(&w, &blocks, i, o, bs).unwrap();
    for blk in 0..o / bs {
        for j in 0..i {
            let before: f64 = (0..bs).map(|t| w[(blk * bs + t) * i + j].powi(2)).sum();
            let after: f64 = (0..bs).map(|t| rot[(blk * bs + t) * i + j].powi(2)).sum();
            assert!(
                (before - after).abs() < 1e-9,
                "block {blk} column {j}: norm {before} -> {after}"
            );
        }
    }
}

/// A non-skew parameter is skew-symmetrised rather than trusted: a symmetric
/// component silently breaks orthogonality while leaving `R` plausible.
#[test]
fn a_non_skew_parameter_is_corrected_not_trusted() {
    let n = 4;
    let q = mat(n * n, 41); // arbitrary, certainly not skew
    let s = skew(&q, n).unwrap();
    for a in 0..n {
        assert!(s[a * n + a].abs() < 1e-15, "diagonal must be zero");
        for b in 0..n {
            assert!((s[a * n + b] + s[b * n + a]).abs() < 1e-15, "must be skew");
        }
    }
    // And the resulting R is still orthogonal.
    let r = cayley_orthogonal(&q, n).unwrap();
    for a in 0..n {
        let norm: f64 = (0..n).map(|t| r[a * n + t].powi(2)).sum();
        assert!((norm - 1.0).abs() < 1e-10, "row {a} norm {norm}");
    }
}
