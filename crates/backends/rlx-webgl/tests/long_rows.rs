// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A long 1-D tensor is one `1 x n` row, and a training graph's flat gradient
//! vector (every parameter's gradient concatenated) is millions long: past
//! `MAX_TEXTURE_SIZE` it failed to plan, so no trained decoder ran on WebGL.
//! Such a slot is laid out wide like a tall one, and the result is unchanged.

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, GraphExt, Shape};

#[test]
fn a_row_longer_than_a_texture_is_laid_out_wide() {
    let (n0, n1) = (1_200_000usize, 1_175_680usize);
    let n = n0 + n1;
    let f = DType::F32;
    let mut g = Graph::new("flat_grad");
    let a = g.input("a", Shape::new(&[n0], f));
    let b = g.input("b", Shape::new(&[n1], f));
    let flat = g.concat(vec![a, b], 0, Shape::new(&[n], f));
    let r = g.activation(Activation::Tanh, flat, Shape::new(&[n], f));
    let sq = g.mul(r, flat);
    g.set_outputs(vec![sq]);

    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let (origin, rows, cols) = plan.largest_texture().unwrap();
    assert!(rows.max(cols) <= 16384, "{origin} is {rows} x {cols}");

    let xa: Vec<f32> = (0..n0)
        .map(|i| ((i % 977) as f32 - 488.0) * 0.003)
        .collect();
    let xb: Vec<f32> = (0..n1)
        .map(|i| ((i % 613) as f32 - 306.0) * -0.005)
        .collect();
    let out = rlx_webgl::run_cpu(&plan, &[("a", &xa), ("b", &xb)]).expect("run");
    let want: Vec<f32> = xa.iter().chain(&xb).map(|&x| x.tanh() * x).collect();
    assert_eq!(out[0].len(), n);
    let bad = (0..n)
        .filter(|&i| out[0][i].to_bits() != want[i].to_bits())
        .count();
    assert_eq!(bad, 0, "{bad} of {n} differ");
}
