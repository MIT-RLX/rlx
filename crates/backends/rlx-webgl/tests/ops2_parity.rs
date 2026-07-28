// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity for Im2Col, ArgMax/ArgMin, runtime Gather, and RoPE vs RLX's CPU
// backend.

use rlx_ir::{DType, Graph, GraphExt, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");
    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(
            r.len(),
            gv.len(),
            "output {oi} length: {} vs {}",
            r.len(),
            gv.len()
        );
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

#[test]
fn im2col() {
    let xs = f32s(2 * 4 * 4, |i| (i as f32 * 0.3).sin());
    let mut g = Graph::new("im2col");
    let x = g.input("x", sh(&[1, 2, 4, 4]));
    let y = g.im2col(x, [2, 2], [1, 1], [0, 0], [1, 1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn im2col_padded() {
    let xs = f32s(9, |i| i as f32 - 4.0);
    let mut g = Graph::new("im2col_p");
    let x = g.input("x", sh(&[1, 1, 3, 3]));
    let y = g.im2col(x, [3, 3], [1, 1], [1, 1], [1, 1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn argmax_argmin() {
    // distinct per-row extremes
    let xs = vec![1.0f32, 5.0, 2.0, 3.0, 9.0, 1.0, 1.0, 1.0];
    let mut g = Graph::new("argmax");
    let x = g.input("x", sh(&[2, 4]));
    let mx = g.argmax(x, 1, false, sh(&[2]));
    let mn = g.argmin(x, 1, false, sh(&[2]));
    g.set_outputs(vec![mx, mn]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn gather_embedding() {
    // table [4,3], indices [2] along axis 0 → [2,3].
    let table = f32s(4 * 3, |i| i as f32);
    let idx = vec![2.0f32, 0.0];
    let mut g = Graph::new("gather");
    let t = g.input("t", sh(&[4, 3]));
    let i = g.input("i", sh(&[2]));
    let y = g.gather(t, i, 0, sh(&[2, 3]));
    g.set_outputs(vec![y]);
    parity(g, &[("t", &table), ("i", &idx)]);
}

#[test]
fn rope_full_and_partial() {
    // x [seq=2, head_dim=4], cos/sin [seq=2, head_dim/2=2].
    let xs = f32s(8, |i| i as f32 * 0.3 - 1.0);
    let cs = f32s(4, |i| (0.2 * i as f32).cos());
    let ss = f32s(4, |i| (0.2 * i as f32).sin());

    // Full rotation.
    let mut g = Graph::new("rope");
    let x = g.input("x", sh(&[2, 4]));
    let cos = g.input("c", sh(&[2, 2]));
    let sin = g.input("s", sh(&[2, 2]));
    let y = g.rope(x, cos, sin, 4);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("c", &cs), ("s", &ss)]);

    // Partial rotation (n_rot = 2).
    let mut g = Graph::new("rope_n");
    let x = g.input("x", sh(&[2, 4]));
    let cos = g.input("c", sh(&[2, 2]));
    let sin = g.input("s", sh(&[2, 2]));
    let y = g.rope_n(x, cos, sin, 4, 2);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("c", &cs), ("s", &ss)]);
}
