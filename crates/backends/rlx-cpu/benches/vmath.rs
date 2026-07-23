// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Host vector-math micro-bench: accurate vector math vs portable SIMD
//! (`*_fast`) vs scalar libm.
//!
//! On Apple, `vvexpf` / `vvtanhf` / `vvrecf` call Accelerate directly.
//! Run: `just micro vmath` or `cargo bench -p rlx-cpu --bench vmath`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_cpu::vmath::{vvexpf, vvexpf_fast, vvlogf, vvrecf, vvsqrtf, vvtanhf, vvtanhf_fast};

fn random_buf(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = rlx_ir::Philox4x32::new(seed);
    let mut v = vec![0f32; len];
    rng.fill_normal(&mut v);
    for x in &mut v {
        *x *= scale;
    }
    v
}

fn libm_exp(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi.exp();
    }
}

fn libm_tanh(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi.tanh();
    }
}

fn libm_rec(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = 1.0 / xi;
    }
}

fn libm_log(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi.ln();
    }
}

fn libm_sqrt(y: &mut [f32], x: &[f32]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi.sqrt();
    }
}

fn bench_vmath(c: &mut Criterion) {
    // Lengths: L1-ish, L2-ish, and streaming.
    let lens = [1 << 10, 1 << 16, 1 << 20];

    let mut group = c.benchmark_group("vmath_exp");
    for &n in &lens {
        let x = random_buf(n, 1, 2.0);
        let mut y = vec![0f32; n];
        let label = format!("n={n}");
        group.bench_function(format!("{label}/vvexpf_accurate"), |b| {
            b.iter(|| vvexpf(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/vvexpf_fast"), |b| {
            b.iter(|| vvexpf_fast(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/libm_scalar"), |b| {
            b.iter(|| libm_exp(black_box(&mut y), black_box(&x)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vmath_tanh");
    for &n in &lens {
        let x = random_buf(n, 2, 1.5);
        let mut y = vec![0f32; n];
        let label = format!("n={n}");
        group.bench_function(format!("{label}/vvtanhf_accurate"), |b| {
            b.iter(|| vvtanhf(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/vvtanhf_fast"), |b| {
            b.iter(|| vvtanhf_fast(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/libm_scalar"), |b| {
            b.iter(|| libm_tanh(black_box(&mut y), black_box(&x)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vmath_log");
    for &n in &lens {
        let x = random_buf(n, 4, 2.0)
            .into_iter()
            .map(|v| v.abs() + 1e-3)
            .collect::<Vec<_>>();
        let mut y = vec![0f32; n];
        let label = format!("n={n}");
        group.bench_function(format!("{label}/vvlogf_accurate"), |b| {
            b.iter(|| vvlogf(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/libm_scalar"), |b| {
            b.iter(|| libm_log(black_box(&mut y), black_box(&x)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vmath_sqrt");
    for &n in &lens {
        let x = random_buf(n, 5, 2.0)
            .into_iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>();
        let mut y = vec![0f32; n];
        let label = format!("n={n}");
        group.bench_function(format!("{label}/vvsqrtf_hardware"), |b| {
            b.iter(|| vvsqrtf(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/libm_scalar"), |b| {
            b.iter(|| libm_sqrt(black_box(&mut y), black_box(&x)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("vmath_rec");
    for &n in &lens {
        // Avoid exact zeros for reciprocal.
        let mut x = random_buf(n, 3, 2.0);
        for xi in &mut x {
            if xi.abs() < 1e-3 {
                *xi = 1e-3;
            }
        }
        let mut y = vec![0f32; n];
        let label = format!("n={n}");
        group.bench_function(format!("{label}/vvrecf_accel"), |b| {
            b.iter(|| vvrecf(black_box(&mut y), black_box(&x)));
        });
        group.bench_function(format!("{label}/libm_scalar"), |b| {
            b.iter(|| libm_rec(black_box(&mut y), black_box(&x)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_vmath);
criterion_main!(benches);
