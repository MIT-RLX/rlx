// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Cholesky` on CPU (LAPACK `potrf`): output is lower-triangular and
//! reconstructs the SPD input (`L·Lᵀ = A`).

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 5;

// A = M·Mᵀ + N·I — symmetric positive-definite and well-conditioned.
fn spd() -> Vec<f32> {
    let m: Vec<f32> = (0..N * N)
        .map(|i| ((i as f32) * 0.31 + 0.2).sin() * 0.5)
        .collect();
    let mut a = vec![0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..N {
                s += m[i * N + k] * m[j * N + k];
            }
            a[i * N + j] = s + if i == j { N as f32 } else { 0.0 };
        }
    }
    a
}

#[test]
fn cholesky_cpu_reconstructs_spd() {
    let a = spd();
    let mut g = Graph::new("chol");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let l = g.cholesky(ai, Shape::new(&[N, N], DType::F32));
    g.set_outputs(vec![l]);
    let out = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("a", &a)])
        .pop()
        .unwrap();

    // Strict upper triangle is zero.
    for i in 0..N {
        for j in (i + 1)..N {
            assert!(
                out[i * N + j].abs() < 1e-5,
                "L[{i}][{j}] should be 0, got {}",
                out[i * N + j]
            );
        }
    }
    // L·Lᵀ == A.
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..N {
                s += out[i * N + k] * out[j * N + k];
            }
            assert!(
                (s - a[i * N + j]).abs() < 1e-4,
                "LLᵀ[{i}][{j}]={s} vs A={}",
                a[i * N + j]
            );
        }
    }
}
