// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! o-means against e-means on the host.
//!
//! ```text
//! cargo test -p rlx-vq --release --features omeans --test omeans -- --nocapture
//! ```

#![cfg(feature = "omeans")]

use rlx_vq::emeans::{BatchLaw, Emeans, EmeansConfig, assign_l2};
use rlx_vq::omeans::{Omeans, OmeansConfig};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// `n` rows around `blobs` centres with unequal spreads and sizes.
fn blobs(n: usize, d: usize, blobs: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let centres: Vec<f32> = (0..blobs * d)
        .map(|_| uniform(&mut s) * 20.0 - 10.0)
        .collect();
    let mut x = Vec::with_capacity(n * d);
    for _ in 0..n {
        let u = uniform(&mut s);
        let b = ((u * u) * blobs as f32) as usize % blobs;
        let spread = 0.3 + (b % 5) as f32 * 0.2;
        for t in 0..d {
            let noise = (uniform(&mut s) + uniform(&mut s) + uniform(&mut s) - 1.5) * spread;
            x.push(centres[b * d + t] + noise);
        }
    }
    x
}

fn mse(x: &[f32], c: &[f32], d: usize) -> f64 {
    let (dist, _) = assign_l2(x, c, d);
    dist.iter().map(|&v| v as f64).sum::<f64>() / (x.len() / d) as f64
}

fn emeans_cfg(nepochs: f64) -> EmeansConfig {
    EmeansConfig {
        nepochs,
        // o-means' Auto picks Throughput below 2M rows; use it for both.
        batch_law: BatchLaw::Throughput,
        ..Default::default()
    }
}

/// Probing every cell is exact search, and without revival (`τ = 0`) both
/// trainers draw nothing after the initial rows. From 2 epochs up o-means
/// closes with e-means' last-pass mean, so the two runs must be identical.
#[test]
fn exhaustive_search_without_revival_is_emeans() {
    let (n, d, k) = (4096, 16, 64);
    let x = blobs(n, d, 24, 3);
    let mut e = emeans_cfg(2.0);
    e.unused_threshold = Some(0.0);
    let (ce, le) = Emeans::new(d, k, e.clone()).train(&x, None);
    let mut om = Omeans::new(
        d,
        k,
        OmeansConfig {
            emeans: e,
            nlist: Some(8),
            nprobe: Some(8),
        },
    );
    let (co, lo) = om.train(&x, None);
    assert_eq!(co, ce, "centroids differ");
    assert_eq!(lo, le, "labels differ");
    let r = om.report();
    assert_eq!(r.n_splits, 0);
    println!(
        "exhaustive: mse {:.6}, {} fits, evals {:.3} of exact",
        mse(&x, &co, d),
        r.rebuilds,
        r.distance_evals as f64 / r.exact_distance_evals as f64
    );
}

/// With a few probes out of many cells, o-means must stay within 2% of
/// e-means' error while computing a fraction of its distances, and its final
/// index must find the exact nearest centroid for nearly every row.
#[test]
fn two_level_search_keeps_emeans_error() {
    let (n, d, k) = (16384, 16, 1024);
    let x = blobs(n, d, 96, 5);
    for nepochs in [1.0, 2.0] {
        let e = emeans_cfg(nepochs);
        let t0 = std::time::Instant::now();
        let (ce, _) = Emeans::new(d, k, e.clone()).train(&x, None);
        let te = t0.elapsed();
        let mut om = Omeans::new(
            d,
            k,
            OmeansConfig {
                emeans: e,
                nlist: Some(64),
                nprobe: Some(6),
            },
        );
        let t0 = std::time::Instant::now();
        let (co, _) = om.train(&x, None);
        let to = t0.elapsed();
        let (me, mo) = (mse(&x, &ce, d), mse(&x, &co, d));
        let r = om.report();
        let share = r.distance_evals as f64 / r.exact_distance_evals as f64;
        let (_, exact) = om.assign(&x);
        let (_, approx, _) = om.assign_approx(&x);
        let missed = exact.iter().zip(&approx).filter(|(a, b)| a != b).count();
        let miss = missed as f64 / n as f64;
        println!(
            "T={nepochs}: e-means {me:.5} ({te:.2?}), o-means {mo:.5} ({to:.2?}) ratio {:.4}; \
             splits {}, fits {}, evals {share:.3} of exact, final-index miss {miss:.4}",
            mo / me,
            r.n_splits,
            r.rebuilds
        );
        assert!(mo < 1.02 * me, "T={nepochs}: o-means {mo} vs e-means {me}");
        assert!(share < 0.3, "T={nepochs}: {share} of exact distances");
        assert!(miss < 0.05, "T={nepochs}: final index missed {miss}");
        assert!(r.n_splits > 0, "revival never fired");
    }
}

/// Labels from training are each visited row's last-pass label; a run below
/// one epoch leaves the rows it never visited at `u32::MAX`.
#[test]
fn fractional_run_labels_only_visited_rows() {
    let (n, d, k) = (2000, 8, 32);
    let x = blobs(n, d, 10, 9);
    let mut om = Omeans::new(
        d,
        k,
        OmeansConfig {
            emeans: emeans_cfg(0.5),
            ..Default::default()
        },
    );
    let (c, labels) = om.train(&x, None);
    let labels = labels.expect("capture_assignments is on");
    let visited = labels.iter().filter(|&&l| l != u32::MAX).count();
    assert_eq!(visited, 1000);
    assert!(labels.iter().all(|&l| l == u32::MAX || (l as usize) < k));
    assert!(c.iter().all(|v| v.is_finite()));
    assert!(om.schedule().unwrap().recency_close);
}

/// The device assignment on every available backend: the run must reach the
/// host trainer's error. The two differ only in f32 summation order and the
/// shifted frame, but a flipped near-tie changes later revivals, so the runs
/// are compared by error, not label by label (`omeans::device`'s unit test
/// compares one batch).
#[cfg(feature = "omeans-device")]
#[test]
fn device_assignment_matches_host_on_every_backend() {
    use rlx_vq::omeans::device::DeviceOmeans;
    let (n, d, k) = (16384, 16, 1024);
    let x = blobs(n, d, 96, 5);
    let cfg = OmeansConfig {
        emeans: emeans_cfg(2.0),
        nlist: Some(64),
        nprobe: Some(6),
    };
    let mut host = Omeans::new(d, k, cfg.clone());
    let t0 = std::time::Instant::now();
    let (ch, _) = host.train(&x, None);
    let th = t0.elapsed();
    let mh = mse(&x, &ch, d);
    println!("host: mse {mh:.5} ({th:.2?})");
    for dev in rlx_runtime::available_devices() {
        let mut om = DeviceOmeans::new(dev, d, k, cfg.clone());
        let t0 = std::time::Instant::now();
        let (cd, _) = om.train(&x, None);
        let td = t0.elapsed();
        let md = mse(&x, &cd, d);
        let r = om.device_report();
        println!(
            "{dev:?}: mse {md:.5} ratio {:.4} ({td:.2?}); \
             {} probe passes, {} compiles, U {} M {}, resident {}",
            md / mh,
            r.probe_passes,
            r.step_compiles,
            r.units,
            r.candidates,
            r.resident
        );
        assert!(md < 1.02 * mh, "{dev:?}: mse {md} vs host {mh}");
    }
}

/// Large-k timing, not run by default:
///
/// ```text
/// OMEANS_BENCH_N=200000 OMEANS_BENCH_K=8192 OMEANS_BENCH_D=96 OMEANS_BENCH_T=1 \
///   cargo test -p rlx-vq --release --features omeans-device --test omeans \
///   bench_large_k -- --ignored --nocapture
/// ```
///
/// Each available device (or `OMEANS_BENCH_DEVICES=Metal,Cuda`) trains dense
/// e-means (`DeviceEmeans`) and o-means (`DeviceOmeans`) with the same batch
/// law and epochs; the error is measured on a 16384-row subsample.
#[cfg(feature = "omeans-device")]
#[test]
#[ignore]
fn bench_large_k() {
    use rlx_vq::emeans::device::DeviceEmeans;
    use rlx_vq::omeans::device::DeviceOmeans;
    let env = |name: &str, default: f64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let n = env("OMEANS_BENCH_N", 200_000.0) as usize;
    let k = env("OMEANS_BENCH_K", 8192.0) as usize;
    let d = env("OMEANS_BENCH_D", 96.0) as usize;
    let nepochs = env("OMEANS_BENCH_T", 1.0);
    let want = std::env::var("OMEANS_BENCH_DEVICES").ok();
    let x = blobs(n, d, (k / 4).max(8), 21);
    let probe = &x[..n.min(16384) * d];
    let cfg = OmeansConfig {
        emeans: emeans_cfg(nepochs),
        ..Default::default()
    };
    println!("n={n} k={k} d={d} T={nepochs}");
    for dev in rlx_runtime::available_devices() {
        if want
            .as_ref()
            .is_some_and(|w| !w.split(',').any(|s| s == format!("{dev:?}")))
        {
            continue;
        }
        let t0 = std::time::Instant::now();
        let (ce, _) = DeviceEmeans::new(dev, d, k, cfg.emeans.clone()).train(&x, None);
        let te = t0.elapsed().as_secs_f64();
        let mut om = DeviceOmeans::new(dev, d, k, cfg.clone());
        let t0 = std::time::Instant::now();
        let (co, _) = om.train(&x, None);
        let to = t0.elapsed().as_secs_f64();
        let (me, mo) = (mse(probe, &ce, d), mse(probe, &co, d));
        let r = om.device_report();
        let s = om.omeans().schedule().unwrap();
        let rep = om.omeans().report();
        println!(
            "{dev:?}: e-means {te:.2}s mse {me:.4} | o-means {to:.2}s mse {mo:.4} \
             (x{:.2} faster, error {:+.2}%) | B {} nlist {} nprobe {} U {} M {} | \
             probe {:.2}s step {:.2}s tiles {:.2}s cells {:.2}s revival {:.2}s, {} probe passes, {} compiles, \
             evals {:.3} of exact, unit work {:.3} of exact",
            te / to,
            100.0 * (mo / me - 1.0),
            s.recipe.batch_size,
            s.nlist,
            s.nprobe,
            r.units,
            r.candidates,
            r.probe_seconds,
            r.step_seconds,
            r.tile_seconds,
            r.cell_seconds,
            r.revival_seconds,
            r.probe_passes,
            r.step_compiles,
            rep.distance_evals as f64 / rep.exact_distance_evals as f64,
            r.unit_evals as f64 / rep.exact_distance_evals as f64
        );
    }
}

/// Lloyd's k-means: `iters` full passes from `init`.
fn lloyd(x: &[f32], init: &[f32], d: usize, iters: usize) -> Vec<f32> {
    let k = init.len() / d;
    let mut c = init.to_vec();
    for _ in 0..iters {
        let (_, labels) = assign_l2(x, &c, d);
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (row, &l) in x.chunks_exact(d).zip(&labels) {
            counts[l as usize] += 1;
            for (s, v) in sums[l as usize * d..][..d].iter_mut().zip(row) {
                *s += *v as f64;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for t in 0..d {
                    c[j * d + t] = (sums[j * d + t] / counts[j] as f64) as f32;
                }
            }
        }
    }
    c
}

/// k-means (Lloyd, 25 iterations) against e-means and o-means on the host,
/// all from the same random rows, not run by default:
///
/// ```text
/// cargo test -p rlx-vq --release --features omeans --test omeans \
///   bench_kmeans_emeans_omeans -- --ignored --nocapture
/// ```
///
/// Prints training MSE and wall time; o-means uses its default cells.
#[test]
#[ignore]
fn bench_kmeans_emeans_omeans() {
    let (n, d) = (100_000, 32);
    for k in [1024, 4096] {
        let x = blobs(n, d, k / 4, 17);
        let init: Vec<f32> = {
            let perm = rlx_rng::numpy::RandomState::new(1234).permutation(n);
            perm[..k]
                .iter()
                .flat_map(|&i| x[i * d..(i + 1) * d].iter().copied())
                .collect()
        };
        let t0 = std::time::Instant::now();
        let cl = lloyd(&x, &init, d, 25);
        let tl = t0.elapsed().as_secs_f64();
        let ml = mse(&x, &cl, d);
        println!("n={n} d={d} k={k}: Lloyd-25 mse {ml:.4} in {tl:.2}s");
        for nepochs in [1.0, 2.0, 5.0] {
            let e = emeans_cfg(nepochs);
            let t0 = std::time::Instant::now();
            let (ce, _) = Emeans::new(d, k, e.clone()).train(&x, Some(&init));
            let te = t0.elapsed().as_secs_f64();
            let mut om = Omeans::new(
                d,
                k,
                OmeansConfig {
                    emeans: e,
                    ..Default::default()
                },
            );
            let t0 = std::time::Instant::now();
            let (co, _) = om.train(&x, Some(&init));
            let to = t0.elapsed().as_secs_f64();
            let (me, mo) = (mse(&x, &ce, d), mse(&x, &co, d));
            let s = om.schedule().unwrap();
            println!(
                "  T={nepochs}: e-means mse {me:.4} ({:+.2}% vs Lloyd) in {te:.2}s | \
                 o-means mse {mo:.4} ({:+.2}%) in {to:.2}s, nlist {} nprobe {}, evals {:.3} of exact",
                100.0 * (me / ml - 1.0),
                100.0 * (mo / ml - 1.0),
                s.nlist,
                s.nprobe,
                om.report().distance_evals as f64 / om.report().exact_distance_evals as f64
            );
        }
    }
}
