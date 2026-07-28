// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::TriangularSolve` on CPU (BLAS `trsm`): `op(A)·X = B` for both `A` and
//! `Aᵀ` — verified by reconstructing `B`.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 4;

fn run(l: &[f32], b: &[f32], nrhs: usize, lower: bool, transpose: bool) -> Vec<f32> {
    let mut g = Graph::new("trisolve");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let bi = g.input("b", Shape::new(&[N, nrhs], DType::F32));
    let x = g.triangular_solve(ai, bi, lower, transpose, Shape::new(&[N, nrhs], DType::F32));
    g.set_outputs(vec![x]);
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("a", l), ("b", b)])
        .pop()
        .unwrap()
}

#[test]
fn trisolve_lower_and_transpose() {
    // Lower-triangular L with a well-away-from-zero diagonal.
    let mut l = vec![0f32; N * N];
    for i in 0..N {
        for j in 0..=i {
            l[i * N + j] = if i == j {
                (i as f32) + 2.0
            } else {
                0.3 * ((i + j) as f32) - 0.2
            };
        }
    }
    let nrhs = 2;
    let b: Vec<f32> = (0..N * nrhs)
        .map(|k| ((k as f32) * 0.5 + 0.1).sin())
        .collect();

    // Solve L·X = B → check L·X == B.
    let x = run(&l, &b, nrhs, true, false);
    for i in 0..N {
        for c in 0..nrhs {
            let mut s = 0.0f32;
            for k in 0..N {
                s += l[i * N + k] * x[k * nrhs + c];
            }
            assert!((s - b[i * nrhs + c]).abs() < 1e-4, "L·X != B [{i}][{c}]");
        }
    }

    // Solve Lᵀ·X = B → check Lᵀ·X == B  ((Lᵀ)[i][k] = L[k][i]).
    let xt = run(&l, &b, nrhs, true, true);
    for i in 0..N {
        for c in 0..nrhs {
            let mut s = 0.0f32;
            for k in 0..N {
                s += l[k * N + i] * xt[k * nrhs + c];
            }
            assert!((s - b[i * nrhs + c]).abs() < 1e-4, "Lᵀ·X != B [{i}][{c}]");
        }
    }
}
