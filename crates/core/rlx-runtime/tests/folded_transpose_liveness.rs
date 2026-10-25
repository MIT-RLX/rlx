// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a transpose the CPU backend folds into its matmul (a GEMM
//! flag) is never executed — the matmul reads the transpose's input — and
//! the memory plan must keep that input alive until the matmul, not only
//! until the transpose. It ended the input's life at the transpose, so with
//! two `transpose(x·x) @ w` branches the second branch's product took the
//! first's freed slot before the first matmul read it: off by hundreds,
//! silently (an ISTFT head, `re @ cos + im @ sin`, came out wrong).

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn a_folded_transposes_input_lives_until_the_matmul() {
    let (k, t, n) = (9usize, 5usize, 16usize);
    let a: Vec<f32> = (0..k * t)
        .map(|i| ((i * 7 % 13) as f32) * 0.3 + 0.1)
        .collect();
    let p: Vec<f32> = (0..k * t)
        .map(|i| ((i * 5 % 11) as f32) * 0.7 - 3.0)
        .collect();
    let wa: Vec<f32> = (0..k * n).map(|i| ((i * 3 % 7) as f32) - 3.0).collect();
    let wp: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 9) as f32) - 4.0).collect();
    let f = DType::F32;
    let mut g = Graph::new("two_branches");
    let ai = g.input("a", Shape::new(&[k, t], f));
    let pi = g.input("p", Shape::new(&[k, t], f));
    let wai = g.input("wa", Shape::new(&[k, n], f));
    let wpi = g.input("wp", Shape::new(&[k, n], f));
    let x = g.mul(ai, ai);
    let y = g.mul(pi, pi);
    let x = g.transpose_(x, vec![1, 0]);
    let y = g.transpose_(y, vec![1, 0]);
    let x = g.mm(x, wai);
    let y = g.mm(y, wpi);
    g.set_outputs(vec![x, y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.finalize_params();
    let out = c.run(&[("a", &a), ("p", &p), ("wa", &wa), ("wp", &wp)]);
    let mut worst = 0f32;
    for j in 0..t {
        for m in 0..n {
            let xs: f32 = (0..k)
                .map(|r| a[r * t + j] * a[r * t + j] * wa[r * n + m])
                .sum();
            let ys: f32 = (0..k)
                .map(|r| p[r * t + j] * p[r * t + j] * wp[r * n + m])
                .sum();
            worst = worst
                .max((out[0][j * n + m] - xs).abs())
                .max((out[1][j * n + m] - ys).abs());
        }
    }
    assert!(worst < 1e-4, "off by {worst}");
}
