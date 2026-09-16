// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibration evidence for [`TileCostModel`], carried in-tree.
//!
//! The model's doc comment claims its *ordering* transfers to shapes it was not
//! fitted on. That claim is only worth what it can be re-checked against, so
//! the measurements behind it live here rather than in a commit message: six
//! tiles across three shapes on an RTX 3080 Ti, median of 30 L2-flushed samples
//! per point, taken with the GPU verified idle (min-of-5 NVML utilization under
//! 15%) by `rlx-cuda`'s `tune_dispatch`.
//!
//! The gate is leave-one-shape-out. Fit on two shapes, rank the third, and
//! require the *ordering* to hold — not the magnitudes, which are off by up to
//! 122% at the smallest shape and are excluded from every claim the model
//! makes.
//!
//! # Why rank correlation and not "picks the winner"
//!
//! `32x32x32_t2x2_b16x16` is fastest at all three measured shapes. A model
//! hardcoded to answer `32x32x32` would therefore score 3/3 on winners while
//! knowing nothing. Spearman ρ over the full six-tile ordering is the test that
//! such a model fails, so that is the gate. `winner_survives_the_prefilter`
//! is kept as a *separate, weaker* check of the property the tuner actually
//! depends on, and is labelled as weak where it sits.

use rlx_gpu_dispatch::cost::{
    Bottleneck, CostProvenance, SYNC_EVENT_MAC_EQUIVALENT, TileCostModel,
};
use rlx_gpu_dispatch::dispatch::Workload;
use rlx_gpu_dispatch::tiles::TileParams;

/// `(bm, bn, bk, tm, tn, bdx, bdy)` for each measured candidate.
const TILES: &[(&str, TileParams)] = &[
    (
        "64x64x16",
        TileParams {
            bm: 64,
            bn: 64,
            bk: 16,
            tm: 4,
            tn: 4,
            bdx: 16,
            bdy: 16,
        },
    ),
    (
        "32x32x16",
        TileParams {
            bm: 32,
            bn: 32,
            bk: 16,
            tm: 4,
            tn: 4,
            bdx: 8,
            bdy: 8,
        },
    ),
    (
        "32x32x32",
        TileParams {
            bm: 32,
            bn: 32,
            bk: 32,
            tm: 2,
            tn: 2,
            bdx: 16,
            bdy: 16,
        },
    ),
    (
        "64x64x32",
        TileParams {
            bm: 64,
            bn: 64,
            bk: 32,
            tm: 4,
            tn: 4,
            bdx: 16,
            bdy: 16,
        },
    ),
    (
        "128x128x8",
        TileParams {
            bm: 128,
            bn: 128,
            bk: 8,
            tm: 8,
            tn: 8,
            bdx: 16,
            bdy: 16,
        },
    ),
    (
        "128x128x16",
        TileParams {
            bm: 128,
            bn: 128,
            bk: 16,
            tm: 8,
            tn: 8,
            bdx: 16,
            bdy: 16,
        },
    ),
];

/// `(m, k, n, [median ms per tile, in TILES order])`, RTX 3080 Ti, idle.
const MEASURED: &[(usize, usize, usize, [f64; 6])] = &[
    (1, 1024, 1024, [0.109, 0.092, 0.055, 0.083, 0.247, 0.204]),
    (1, 4096, 4096, [0.540, 0.370, 0.313, 0.439, 0.811, 0.632]),
    (32, 4096, 4096, [0.636, 0.466, 0.464, 0.570, 1.037, 0.860]),
];

/// Worst held-out ordering correlation the model is allowed to show.
///
/// Set from the observed worst case (0.943) with headroom, so a coefficient
/// edit that degrades ranking trips this rather than passing quietly.
const MIN_HOLDOUT_RHO: f64 = 0.90;

fn ranks(order: &[usize]) -> Vec<usize> {
    let mut r = vec![0; order.len()];
    for (rank, &i) in order.iter().enumerate() {
        r[i] = rank;
    }
    r
}

/// Spearman's ρ over two orderings of the same items. No ties are possible
/// here: measured times are distinct and scores are distinct.
fn spearman(a: &[usize], b: &[usize]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let d2: f64 = ra
        .iter()
        .zip(&rb)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

/// Least squares for `t = c0·padded + c1·sync` via the 2×2 normal equations.
/// Two unknowns, so an explicit solve avoids a linear-algebra dependency in a
/// crate that deliberately has none.
///
/// No intercept: see the module docs on `cost` for why a fixed term was fitted,
/// found to exceed the fastest measured time, and removed.
fn fit(rows: &[([f64; 2], f64)]) -> [f64; 2] {
    let (mut a, mut b, mut c, mut p, mut q) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in rows {
        a += x[0] * x[0];
        b += x[0] * x[1];
        c += x[1] * x[1];
        p += x[0] * y;
        q += x[1] * y;
    }
    let det = a * c - b * b;
    assert!(
        det.abs() > 1e-12,
        "singular normal equations — features are collinear"
    );
    [(p * c - q * b) / det, (a * q - b * p) / det]
}

/// The model's two features.
fn features(t: &TileParams, m: usize, k: usize, n: usize) -> [f64; 2] {
    let (bm, bn, bk) = (t.bm as u64, t.bn as u64, t.bk as u64);
    let (mu, ku, nu) = (m as u64, k as u64, n as u64);
    let (tiles_m, tiles_n) = (mu.div_ceil(bm), nu.div_ceil(bn));
    let padded = (tiles_m * bm * tiles_n * bn * ku) as f64;
    let sync = (tiles_m * tiles_n * ku.div_ceil(bk)) as f64;
    // Scaled so the normal equations stay well conditioned; the ratio the model
    // ships is rescaled back below.
    [padded / 1e9, sync / 1e6]
}

fn measured_order(times: &[f64; 6]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..6).collect();
    idx.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).expect("finite"));
    idx
}

#[test]
fn holdout_ranking_meets_the_documented_rho() {
    let mut worst = f64::INFINITY;
    let mut worst_shape = (0, 0, 0);

    for held in MEASURED {
        let train: Vec<([f64; 2], f64)> = MEASURED
            .iter()
            .filter(|s| (s.0, s.1, s.2) != (held.0, held.1, held.2))
            .flat_map(|&(m, k, n, times)| {
                TILES
                    .iter()
                    .enumerate()
                    .map(move |(i, (_, t))| (features(t, m, k, n), times[i]))
            })
            .collect();
        let c = fit(&train);

        let (m, k, n, times) = *held;
        let mut pred: Vec<(usize, f64)> = TILES
            .iter()
            .enumerate()
            .map(|(i, (_, t))| {
                let f = features(t, m, k, n);
                (i, c[0] * f[0] + c[1] * f[1])
            })
            .collect();
        pred.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
        let pred_order: Vec<usize> = pred.iter().map(|(i, _)| *i).collect();

        let rho = spearman(&measured_order(&times), &pred_order);
        if rho < worst {
            worst = rho;
            worst_shape = (m, k, n);
        }
    }

    assert!(
        worst >= MIN_HOLDOUT_RHO,
        "held-out ordering degraded: worst rho {worst:.3} at {worst_shape:?}, floor {MIN_HOLDOUT_RHO}"
    );

    // The shipped model advertises a rho; it must not overstate what the data
    // supports.
    if let CostProvenance::Measured { holdout_rho, .. } = TileCostModel::sm86().provenance() {
        assert!(
            (holdout_rho as f64) <= worst + 1e-3,
            "sm86() advertises rho {holdout_rho} but the data gives {worst:.3}"
        );
    } else {
        panic!("sm86() must carry Measured provenance");
    }
}

#[test]
fn shipped_coefficients_reproduce_the_measured_ordering() {
    // Not held out — this checks the coefficients actually shipped, rather than
    // ones refitted inside the test, still order the data they came from.
    let model = TileCostModel::sm86();
    for &(m, k, n, times) in MEASURED {
        let w = Workload::Matmul { m, k, n };
        let mut pred: Vec<(usize, f64)> = TILES
            .iter()
            .enumerate()
            .map(|(i, (_, t))| (i, model.estimate(t, &w).expect("matmul workload").score))
            .collect();
        pred.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
        let rho = spearman(
            &measured_order(&times),
            &pred.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        );
        assert!(
            rho >= MIN_HOLDOUT_RHO,
            "shipped coefficients rank {m}x{k}x{n} at rho {rho:.3}"
        );
    }
}

#[test]
fn winner_survives_the_prefilter() {
    // WEAK by construction: the same tile wins all three measured shapes, so
    // this passes for any model biased toward it. It is here because it is the
    // property `tune_dispatch` depends on, not because it discriminates.
    // `holdout_ranking_meets_the_documented_rho` is the real gate.
    let model = TileCostModel::sm86();
    let all: Vec<TileParams> = TILES.iter().map(|(_, t)| *t).collect();
    for &(m, k, n, times) in MEASURED {
        let best = measured_order(&times)[0];
        let pf = model.prefilter(&Workload::Matmul { m, k, n }, &all, 3);
        assert!(
            pf.tiles().contains(&TILES[best].1),
            "prefilter dropped the measured winner {} at {m}x{k}x{n}",
            TILES[best].0
        );
    }
}

#[test]
fn shipped_constant_is_what_the_data_says() {
    // `SYNC_EVENT_MAC_EQUIVALENT` is the whole model, so it is re-derived here
    // from the timings rather than trusted as a literal. The features are
    // scaled by 1e9 and 1e6, so the ratio rescales by 1e3.
    let rows: Vec<([f64; 2], f64)> = MEASURED
        .iter()
        .flat_map(|&(m, k, n, times)| {
            TILES
                .iter()
                .enumerate()
                .map(move |(i, (_, t))| (features(t, m, k, n), times[i]))
        })
        .collect();
    let c = fit(&rows);
    let derived = (c[1] / c[0]) * 1e3;
    let rel = (derived - SYNC_EVENT_MAC_EQUIVALENT).abs() / SYNC_EVENT_MAC_EQUIVALENT;
    assert!(
        rel < 0.01,
        "shipped constant {SYNC_EVENT_MAC_EQUIVALENT} disagrees with the data ({derived:.0})"
    );
}

#[test]
fn the_removed_fixed_term_was_not_physical() {
    // Pins the reason the model has two terms and not three. Refitting WITH an
    // intercept puts more in it than the fastest time ever measured, which no
    // per-dispatch cost can be. Kept as a test so a future "let's add back a
    // launch-overhead term" has to confront the same number.
    let mut rows: Vec<([f64; 3], f64)> = Vec::new();
    for &(m, k, n, times) in MEASURED {
        for (i, (_, t)) in TILES.iter().enumerate() {
            let f = features(t, m, k, n);
            rows.push(([1.0, f[0], f[1]], times[i]));
        }
    }
    // 3x3 normal equations, Gauss-Jordan with partial pivoting.
    let mut aug = [[0.0f64; 4]; 3];
    for (x, y) in &rows {
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] += x[i] * x[j];
            }
            aug[i][3] += x[i] * y;
        }
    }
    for col in 0..3 {
        let piv = (col..3)
            .max_by(|&a, &b| {
                aug[a][col]
                    .abs()
                    .partial_cmp(&aug[b][col].abs())
                    .expect("finite")
            })
            .expect("nonempty");
        aug.swap(col, piv);
        let d = aug[col][col];
        for v in aug[col].iter_mut() {
            *v /= d;
        }
        for row in 0..3 {
            if row != col {
                let f = aug[row][col];
                for j in 0..4 {
                    aug[row][j] -= f * aug[col][j];
                }
            }
        }
    }
    let intercept = aug[0][3];
    let fastest = MEASURED
        .iter()
        .flat_map(|(_, _, _, t)| t.iter().copied())
        .fold(f64::INFINITY, f64::min);
    assert!(
        intercept > fastest,
        "the intercept ({intercept:.4} ms) no longer exceeds the fastest measured time \
         ({fastest:.4} ms) — the fixed term may now be identifiable, so revisit its removal \
         deliberately rather than leaving this test as stale prose"
    );
}

#[test]
fn bottleneck_attribution_matches_the_measured_regimes() {
    let model = TileCostModel::sm86();
    // Decode against the largest block tile: the measured worst performer, and
    // the model should say why rather than just scoring it last.
    let big = TILES[4].1; // 128x128x8
    let c = model
        .estimate(
            &big,
            &Workload::Matmul {
                m: 1,
                k: 4096,
                n: 4096,
            },
        )
        .expect("matmul");
    assert!(
        matches!(c.bottleneck, Bottleneck::PaddingWaste { .. }),
        "128x128 at m=1 should attribute to padding, got {}",
        c.bottleneck
    );
    // A shape that divides the tile exactly has nothing left to blame but the
    // compute itself.
    let c = model
        .estimate(
            &TILES[0].1,
            &Workload::Matmul {
                m: 4096,
                k: 4096,
                n: 4096,
            },
        )
        .expect("matmul");
    assert_eq!(c.useful_fraction(), 1.0);
    assert!(
        matches!(c.bottleneck, Bottleneck::ComputeBound),
        "got {}",
        c.bottleneck
    );
}
