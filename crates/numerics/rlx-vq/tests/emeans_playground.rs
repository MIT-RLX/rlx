// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! e-means on the datasets of the paper's browser playground
//! (<https://e-means.github.io/playground.html>, arXiv:2611.09501).
//!
//! The generators are ported from the page (same mulberry32 streams and
//! shapes, `d = 2`); the Meta-logo bitmap is left out. Each dataset is split
//! and initialized as the page does (train/test split seeded 1234, both
//! methods from the same seed-0 sample), then raced as the page races them:
//! 5 e-means epochs against 25 Lloyd iterations, scored on the held-out rows.
//! Every available device also runs the device-resident trainer with the
//! page's bf16 assignment (data and centroids rounded to bf16, f32 compute).
//!
//! The page's size is n = 100 000, k = 500. The default here is smaller so
//! the test stays a test; set `EMEANS_PLAYGROUND_N` / `EMEANS_PLAYGROUND_K`
//! to run the page's size, and `EMEANS_PLAYGROUND_DEVICES=Metal,Cuda` to limit
//! the device runs:
//!
//! ```sh
//! EMEANS_PLAYGROUND_N=100000 EMEANS_PLAYGROUND_K=500 cargo test -p rlx-vq --release \
//!     --features emeans-device,metal --test emeans_playground -- --nocapture
//! ```

#![cfg(feature = "emeans-device")]

use std::f64::consts::PI;

use rlx_ir::DType;
use rlx_runtime::{CompileOptions, Device, HalfCasts};
use rlx_vq::emeans::device::{DeviceEmeans, device_options};
use rlx_vq::emeans::{Emeans, EmeansConfig, assign_l2};

// ── Ported generators ───────────────────────────────────────────

/// The page's `mulberry32`.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6D2B_79F5);
        let a = self.0;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        (t ^ (t >> 14)) as f64 / 4_294_967_296.0
    }
    fn gaussian(&mut self) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.next();
        }
        let mut v = 0.0;
        while v == 0.0 {
            v = self.next();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64).floor() as usize
    }
}

fn shuffled_indices(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut a: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next() * (i + 1) as f64).floor() as usize;
        a.swap(i, j);
    }
    a
}

fn multinomial_sizes(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    let mut sizes = vec![0; k];
    for _ in 0..n {
        sizes[rng.below(k)] += 1;
    }
    sizes
}

fn weighted_multinomial_sizes(rng: &mut Rng, n: usize, weights: &[f64]) -> Vec<usize> {
    let k = weights.len();
    let wsum: f64 = weights.iter().sum();
    let mut cum = vec![0.0; k];
    let mut acc = 0.0;
    for i in 0..k {
        acc += weights[i] / wsum;
        cum[i] = acc;
    }
    let mut sizes = vec![0; k];
    for _ in 0..n {
        let r = rng.next();
        let (mut lo, mut hi) = (0, k - 1);
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if cum[mid] < r {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        sizes[lo] += 1;
    }
    sizes
}

type Rows = Vec<Vec<f64>>;

fn pad_rows(rng: &mut Rng, rows: Rows, from: usize, d: usize) -> Rows {
    if from >= d {
        return rows.into_iter().map(|r| r[..d].to_vec()).collect();
    }
    rows.into_iter()
        .map(|mut r| {
            for _ in from..d {
                r.push(rng.gaussian() * 0.05);
            }
            r
        })
        .collect()
}

fn centers(rng: &mut Rng, k: usize, d: usize, lo: f64, span: f64) -> Rows {
    (0..k)
        .map(|_| (0..d).map(|_| rng.next() * span + lo).collect())
        .collect()
}

fn gaussian_blobs(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = centers(&mut rng, k, d, -10.0, 20.0);
    let sizes = multinomial_sizes(&mut rng, n, k);
    let mut rows = Vec::with_capacity(n);
    for i in 0..k {
        for _ in 0..sizes[i] {
            rows.push(c[i].iter().map(|&m| m + rng.gaussian()).collect());
        }
    }
    rows
}

fn gaussian_blobs_close(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = centers(&mut rng, k, d, -2.0, 4.0);
    let sizes = multinomial_sizes(&mut rng, n, k);
    let mut rows = Vec::with_capacity(n);
    for i in 0..k {
        for _ in 0..sizes[i] {
            rows.push(c[i].iter().map(|&m| m + rng.gaussian()).collect());
        }
    }
    rows
}

fn anisotropic_blobs(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = centers(&mut rng, k, d, -10.0, 20.0);
    let sizes = multinomial_sizes(&mut rng, n, k);
    let mut rows = Vec::with_capacity(n);
    for i in 0..k {
        let scale: Vec<f64> = (0..d).map(|_| rng.next() * 2.8 + 0.2).collect();
        for _ in 0..sizes[i] {
            rows.push(
                c[i].iter()
                    .zip(&scale)
                    .map(|(&m, &s)| m + rng.gaussian() * s)
                    .collect(),
            );
        }
    }
    rows
}

fn uniform_cubes(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = centers(&mut rng, k, d, -10.0, 20.0);
    let sizes = multinomial_sizes(&mut rng, n, k);
    let mut rows = Vec::with_capacity(n);
    for i in 0..k {
        for _ in 0..sizes[i] {
            rows.push(
                c[i].iter()
                    .map(|&m| m + (rng.next() * 2.0 - 1.0) * 2.0)
                    .collect(),
            );
        }
    }
    rows
}

fn moons_tiled(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let noise = 0.08;
    let mut rng = Rng::new(seed);
    let pairs = (k / 2).max(1);
    let per_pair = n / pairs;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..pairs {
        let cx = rng.next() * 30.0 - 15.0;
        let cy = rng.next() * 30.0 - 15.0;
        let half = per_pair / 2;
        for i in 0..per_pair {
            let t = rng.next() * PI;
            let (mut x, mut y) = if i < half {
                (t.cos(), t.sin())
            } else {
                (1.0 - t.cos(), 1.0 - t.sin() - 0.5)
            };
            x = x * 6.0 + cx + rng.gaussian() * noise * 6.0;
            y = y * 6.0 + cy + rng.gaussian() * noise * 6.0;
            let mut row = vec![x, y];
            for _ in 2..d {
                row.push(rng.gaussian() * 0.05);
            }
            rows.push(row);
        }
    }
    rows
}

fn rings(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let n_rings = (k / 4).max(2);
    let per_ring = n / n_rings;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n_rings {
        let r = 2.0 + i as f64 * 3.0;
        for _ in 0..per_ring {
            let theta = rng.next() * 2.0 * PI;
            let x = r * theta.cos() + rng.gaussian() * 0.3;
            let y = r * theta.sin() + rng.gaussian() * 0.3;
            rows.push(vec![x, y]);
        }
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn duplicate_heavy(n: usize, k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let n_unique = k.max((n as f64 * 0.7).round() as usize);
    let mut base = gaussian_blobs(n_unique, k, d, seed);
    for _ in 0..n - n_unique {
        let j = rng.below(n_unique);
        base.push(base[j].clone());
    }
    base
}

fn syn_unif_cube(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let half = 3f64.sqrt();
    (0..n)
        .map(|_| (0..d).map(|_| (rng.next() * 2.0 - 1.0) * half).collect())
        .collect()
}

fn syn_gauss_pow(alpha: f64) -> impl Fn(usize, usize, usize, u32) -> Rows {
    move |n, _k, d, seed| {
        let mut rng = Rng::new(seed);
        let lam: Vec<f64> = (0..d).map(|i| ((i + 1) as f64).powf(-alpha)).collect();
        let sum: f64 = lam.iter().sum();
        let sd: Vec<f64> = lam.iter().map(|v| (v * d as f64 / sum).sqrt()).collect();
        (0..n)
            .map(|_| sd.iter().map(|s| rng.gaussian() * s).collect())
            .collect()
    }
}

fn syn_gmm(zipf: f64) -> impl Fn(usize, usize, usize, u32) -> Rows {
    move |n, k, d, seed| {
        let mut rng = Rng::new(seed);
        let (sw, smu) = ((1.0f64 / 3.0).sqrt(), (2.0f64 / 3.0).sqrt());
        let means: Rows = (0..k)
            .map(|_| (0..d).map(|_| rng.gaussian() * smu).collect())
            .collect();
        let sizes = if zipf > 0.0 {
            let w: Vec<f64> = (0..k).map(|g| ((g + 1) as f64).powf(-zipf)).collect();
            weighted_multinomial_sizes(&mut rng, n, &w)
        } else {
            multinomial_sizes(&mut rng, n, k)
        };
        let mut rows = Vec::with_capacity(n);
        for g in 0..k {
            for _ in 0..sizes[g] {
                rows.push(means[g].iter().map(|m| m + rng.gaussian() * sw).collect());
            }
        }
        rows
    }
}

fn split_counts(n: usize, weights: &[f64]) -> Vec<usize> {
    let s: f64 = weights.iter().sum();
    let mut c: Vec<usize> = weights
        .iter()
        .map(|w| (n as f64 * w / s).floor() as usize)
        .collect();
    let total: usize = c.iter().sum();
    c[0] += n - total;
    c
}

fn vo_toy(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = split_counts(n, &[7000.0, 1000.0, 500.0, 500.0]);
    let mut rows = Vec::with_capacity(n);
    for (m, cx, cy) in [(c[0], -1.0, -1.0), (c[1], 1.0, -1.0), (c[2], 0.0, 1.0)] {
        for _ in 0..m {
            let x = cx + rng.gaussian() / 2.0;
            let y = cy + rng.gaussian() / 2.0;
            rows.push(vec![x, y]);
        }
    }
    for _ in 0..c[3] {
        let x = (rng.next() - 0.5) * 6.0;
        let y = (rng.next() - 0.5) * 6.0;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn rejection(
    n: usize,
    rng: &mut Rng,
    draw: impl Fn(&mut Rng) -> [f64; 2],
    keep: impl Fn([f64; 2]) -> bool,
) -> Rows {
    let mut rows = Vec::with_capacity(n);
    while rows.len() < n {
        let p = draw(rng);
        if keep(p) {
            rows.push(p.to_vec());
        }
    }
    rows
}

fn spirals(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let t = rng.next().sqrt() * 3.0 * PI;
        let a = t + (rng.next() * 3.0).floor() * 2.0 * PI / 3.0;
        let r = t / (3.0 * PI);
        let x = r * a.cos() + rng.gaussian() * 0.05;
        let y = r * a.sin() + rng.gaussian() * 0.05;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn heart(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let rows = rejection(
        n,
        &mut rng,
        |r| {
            let x = r.next() * 3.0 - 1.5;
            [x, r.next() * 3.0 - 1.3]
        },
        |[x, y]| (x * x + y * y - 1.0).powi(3) - x * x * y * y * y <= 0.0,
    );
    pad_rows(&mut rng, rows, 2, d)
}

fn star(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let v: Vec<[f64; 2]> = (0..10)
        .map(|i| {
            let a = PI / 2.0 + i as f64 * PI / 5.0;
            let r = if i % 2 == 0 { 1.0 } else { 0.4 };
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let e = rng.below(10);
        let t = rng.next();
        let (a, b) = (v[e], v[(e + 1) % 10]);
        let x = a[0] * (1.0 - t) + b[0] * t + rng.gaussian() * 0.015;
        let y = a[1] * (1.0 - t) + b[1] * t + rng.gaussian() * 0.015;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn checkerboard(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let rows = rejection(
        n,
        &mut rng,
        |r| {
            let x = r.next() * 8.0;
            [x, r.next() * 8.0]
        },
        |[x, y]| (x.floor() as i64 + y.floor() as i64) % 2 == 0,
    );
    pad_rows(&mut rng, rows, 2, d)
}

fn sierpinski(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let v = [[0.0, 0.0], [1.0, 0.0], [0.5, 3f64.sqrt() / 2.0]];
    let x0 = rng.next();
    let mut p = [x0, rng.next()];
    let mut rows = Vec::with_capacity(n);
    for i in 0..n + 50 {
        let c = v[rng.below(3)];
        p = [(p[0] + c[0]) / 2.0, (p[1] + c[1]) / 2.0];
        if i >= 50 {
            rows.push(p.to_vec());
        }
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn olympic_rings(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = [
        [-2.2, 0.0],
        [0.0, 0.0],
        [2.2, 0.0],
        [-1.1, -1.0],
        [1.1, -1.0],
    ];
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let r = c[rng.below(5)];
        let t = rng.next() * 2.0 * PI;
        let x = r[0] + t.cos() + rng.gaussian() * 0.04;
        let y = r[1] + t.sin() + rng.gaussian() * 0.04;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn lissajous(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let t = rng.next() * 2.0 * PI;
        let x = (3.0 * t + PI / 2.0).sin() + rng.gaussian() * 0.02;
        let y = (2.0 * t).sin() + rng.gaussian() * 0.02;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn galaxy(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let c = split_counts(n, &[3.0, 7.0]);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..c[0] {
        let x = rng.gaussian() * 0.15;
        rows.push(vec![x, rng.gaussian() * 0.15]);
    }
    for _ in 0..c[1] {
        let t = rng.next() * 2.5 * PI;
        let r = 0.15 * (0.25 * t).exp();
        let a = t + (rng.next() * 2.0).floor() * PI;
        let s = 0.03 + 0.04 * r;
        let x = r * a.cos() + rng.gaussian() * s;
        let y = r * a.sin() + rng.gaussian() * s;
        rows.push(vec![x, y]);
    }
    pad_rows(&mut rng, rows, 2, d)
}

fn flower(n: usize, _k: usize, d: usize, seed: u32) -> Rows {
    let mut rng = Rng::new(seed);
    let rows = rejection(
        n,
        &mut rng,
        |r| {
            let x = r.next() * 2.0 - 1.0;
            [x, r.next() * 2.0 - 1.0]
        },
        |[x, y]| x.hypot(y) <= (5.0 * y.atan2(x) / 2.0).cos().abs(),
    );
    pad_rows(&mut rng, rows, 2, d)
}

type Generator = Box<dyn Fn(usize, usize, usize, u32) -> Rows>;

/// The page's datasets in its menu order, without the Meta logo.
fn datasets() -> Vec<(&'static str, Generator)> {
    vec![
        ("vo_toy", Box::new(vo_toy)),
        ("spirals", Box::new(spirals)),
        ("heart", Box::new(heart)),
        ("star", Box::new(star)),
        ("checkerboard", Box::new(checkerboard)),
        ("sierpinski", Box::new(sierpinski)),
        ("olympic_rings", Box::new(olympic_rings)),
        ("lissajous", Box::new(lissajous)),
        ("galaxy", Box::new(galaxy)),
        ("flower", Box::new(flower)),
        ("gaussian_blobs", Box::new(gaussian_blobs)),
        ("gaussian_blobs_close", Box::new(gaussian_blobs_close)),
        ("anisotropic_blobs", Box::new(anisotropic_blobs)),
        ("uniform_cubes", Box::new(uniform_cubes)),
        ("moons_tiled", Box::new(moons_tiled)),
        ("rings", Box::new(rings)),
        ("duplicate_heavy", Box::new(duplicate_heavy)),
        ("syn_unif_cube", Box::new(syn_unif_cube)),
        ("syn_gauss_iso", Box::new(syn_gauss_pow(0.0))),
        ("syn_gauss_pow_a05", Box::new(syn_gauss_pow(0.5))),
        ("syn_gauss_pow_a1", Box::new(syn_gauss_pow(1.0))),
        ("syn_gauss_pow_a2", Box::new(syn_gauss_pow(2.0))),
        ("syn_gmm", Box::new(syn_gmm(0.0))),
        ("syn_gmm_zipf", Box::new(syn_gmm(1.0))),
    ]
}

// ── The page's protocol ─────────────────────────────────────────

const D: usize = 2;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name}={v}: not a count"))
        })
        .unwrap_or(default)
}

fn flat_f32(rows: &[Vec<f64>]) -> Vec<f32> {
    rows.iter().flatten().map(|&v| v as f32).collect()
}

struct Split {
    train: Vec<f32>,
    test: Vec<f32>,
    init: Vec<f32>,
}

/// `n` training rows plus `max(2000, 0.2 n)` held out, split and initialized
/// as the page does.
fn prepare(generator: &Generator, n: usize, k: usize) -> Option<Split> {
    let held = 2000.max((0.2 * n as f64).round() as usize);
    let x = generator(n + held, k, D, 0);
    if x.len() <= n {
        return None; // the page refuses this shape at this k
    }
    let idx = shuffled_indices(&mut Rng::new(1234), x.len());
    let train: Rows = idx[..n].iter().map(|&i| x[i].clone()).collect();
    let test: Rows = idx[n..].iter().map(|&i| x[i].clone()).collect();
    let pick = shuffled_indices(&mut Rng::new(0), n);
    let init: Rows = pick[..k].iter().map(|&i| train[i].clone()).collect();
    Some(Split {
        train: flat_f32(&train),
        test: flat_f32(&test),
        init: flat_f32(&init),
    })
}

fn mse(x: &[f32], c: &[f32]) -> f64 {
    let (dist, _) = assign_l2(x, c, D);
    dist.iter().map(|&v| v as f64).sum::<f64>() / (x.len() / D) as f64
}

fn lloyd(x: &[f32], init: &[f32], iters: usize) -> Vec<f32> {
    let k = init.len() / D;
    let mut c = init.to_vec();
    for _ in 0..iters {
        let (_, labels) = assign_l2(x, &c, D);
        let mut sums = vec![0f64; k * D];
        let mut counts = vec![0usize; k];
        for (row, &l) in x.as_chunks::<D>().0.iter().zip(&labels) {
            counts[l as usize] += 1;
            for t in 0..D {
                sums[l as usize * D + t] += row[t] as f64;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for t in 0..D {
                    c[j * D + t] = (sums[j * D + t] / counts[j] as f64) as f32;
                }
            }
        }
    }
    c
}

/// The page's bf16 assignment: casts that round (data and centroids rounded
/// once, so `‖c‖²` and `x·c` see the same values) and the assignment in bf16.
/// On top of [`device_options`], so the XDNA NPU's matmul takes those bf16
/// operands in its bf16 kernel (exact for them) instead of INT8.
fn bf16_options(dev: Device) -> (CompileOptions, DType) {
    let mut o = device_options(dev);
    o.half_casts = HalfCasts::Round;
    (o, DType::BF16)
}

#[test]
fn playground_datasets_emeans_vs_lloyd() {
    let n = env_usize("EMEANS_PLAYGROUND_N", 20_000);
    let k = env_usize("EMEANS_PLAYGROUND_K", 100);
    // `EMEANS_PLAYGROUND_DEVICES=Metal,Cuda` limits the device runs (by name).
    let only = std::env::var("EMEANS_PLAYGROUND_DEVICES").ok();
    let devices: Vec<Device> = rlx_runtime::available_devices()
        .into_iter()
        .filter(|d| {
            only.as_deref().is_none_or(|o| {
                o.split(',')
                    .any(|n| n.trim().eq_ignore_ascii_case(&format!("{d:?}")))
            })
        })
        .collect();
    eprintln!(
        "\nplayground protocol: n={n} k={k} d={D}, held-out MSE, 5 e-means epochs vs 25 Lloyd \
         iterations; devices run bf16 assignment ({devices:?})"
    );
    eprintln!(
        "{:<22} {:>11} {:>11} {:>8}  device bf16 / host f32",
        "dataset", "e-means", "Lloyd-25", "ratio"
    );
    let mut ratios = Vec::new();
    let mut failures = Vec::new();
    for (name, generator) in datasets() {
        let Some(s) = prepare(&generator, n, k) else {
            eprintln!("{name:<22} skipped: the page refuses this shape at k={k}");
            continue;
        };
        let mut em = Emeans::new(D, k, EmeansConfig::default());
        let (c, _) = em.train(&s.train, Some(&s.init));
        let e = mse(&s.test, &c);
        let l = mse(&s.test, &lloyd(&s.train, &s.init, 25));
        let ratio = e / l;
        ratios.push(ratio);
        let mut line = format!("{name:<22} {e:>11.5e} {l:>11.5e} {ratio:>8.4}");
        for &dev in &devices {
            let (options, assign) = bf16_options(dev);
            let mut de = DeviceEmeans::with_options(
                dev,
                D,
                k,
                EmeansConfig::default(),
                options.clone(),
                assign,
            );
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                de.train(&s.train, Some(&s.init)).0
            }));
            match got {
                Ok(dc) => {
                    let rel = mse(&s.test, &dc) / e;
                    line += &format!("  {dev:?} {rel:.4}");
                    // bf16 keeps 8 bits: a few percent of loss is its price.
                    if rel.is_nan() || rel >= 1.05 {
                        failures.push(format!("{name} {dev:?}: bf16 device MSE {rel:.4}× host"));
                    }
                }
                Err(_) => {
                    line += &format!("  {dev:?} PANIC");
                    failures.push(format!("{name} {dev:?}: panicked"));
                }
            }
        }
        eprintln!("{line}");
        // The page's claim, per dataset: e-means at 5 epochs is not worse
        // than Lloyd at 25 (a little slack for the hardest overlapping sets).
        if ratio > 1.05 {
            failures.push(format!(
                "{name}: e-means {e:.5e} vs Lloyd {l:.5e} ({ratio:.4})"
            ));
        }
    }
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    let wins = ratios.iter().filter(|&&r| r <= 1.0).count();
    eprintln!(
        "e-means beats Lloyd-25 on {wins}/{} datasets; median held-out ratio {median:.4}",
        ratios.len()
    );
    assert!(median < 1.0, "median e-means/Lloyd ratio {median}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
