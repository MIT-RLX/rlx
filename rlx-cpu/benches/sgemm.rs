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

//! Sgemm micro-bench (plan #52).
//!
//! Sweeps a few representative shapes through the rlx-cpu blas
//! path. Run with `cargo bench --bench sgemm`. Per the snippet
//! pattern: one .rs per kernel, criterion handles iteration +
//! statistics + comparing against a baseline.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_cpu::blas::{sgemm, sgemm_epilogue};

fn random_buf(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = rlx_ir::Philox4x32::new(seed);
    let mut v = vec![0f32; len];
    rng.fill_normal(&mut v);
    v
}

fn bench_sgemm(c: &mut Criterion) {
    // Shape coverage: tiny (matches batch=1 attention QK^T),
    // medium (BERT FFN-up batch≈8), large (BGE batch=32).
    let shapes = [
        ("tiny_64x64x64", (64usize, 64, 64)),
        ("medium_256x768x3072", (256, 768, 3072)),
        ("attn_8x512x512", (8, 512, 512)),
    ];

    let mut group = c.benchmark_group("sgemm");
    for (label, (m, k, n)) in shapes {
        let a = random_buf(m * k, 1);
        let b = random_buf(k * n, 2);
        let mut c_buf = vec![0f32; m * n];
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                sgemm(black_box(&a), black_box(&b), black_box(&mut c_buf), m, k, n);
            });
        });
    }
    group.finish();
}

fn bench_sgemm_epilogue(c: &mut Criterion) {
    // Same shapes via the epilogue closure path (plan #1) — should
    // be within ~1% of plain sgemm + a separate elementwise pass.
    let shapes = [
        ("tiny_relu", (64usize, 64, 64)),
        ("medium_relu", (256, 768, 3072)),
    ];
    let mut group = c.benchmark_group("sgemm_epilogue");
    for (label, (m, k, n)) in shapes {
        let a = random_buf(m * k, 3);
        let b = random_buf(k * n, 4);
        let mut c_buf = vec![0f32; m * n];
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                sgemm_epilogue(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_buf),
                    m,
                    k,
                    n,
                    |x: f32| x.max(0.0),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sgemm, bench_sgemm_epilogue);
criterion_main!(benches);
