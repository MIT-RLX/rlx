// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.
//
// Parity for Cumsum / Concat / Pool — lowered to existing kernels (triangular
// matmul, masked-gather blend, reduce-with-window-groups) — vs RLX's CPU
// backend.

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};

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
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

#[test]
fn cumsum_inclusive_and_exclusive() {
    let xs = f32s(8, |i| i as f32 * 0.5 - 1.0);
    for exclusive in [false, true] {
        let mut g = Graph::new("cs");
        let x = g.input("x", sh(&[2, 4]));
        let y = g.cumsum(x, -1, exclusive, sh(&[2, 4]));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
}

#[test]
fn concat_axis1() {
    let a = f32s(2 * 2, |i| i as f32);
    let b = f32s(2 * 3, |i| 10.0 + i as f32);
    let mut g = Graph::new("cat");
    let ai = g.input("a", sh(&[2, 2]));
    let bi = g.input("b", sh(&[2, 3]));
    let y = g.concat(vec![ai, bi], 1, sh(&[2, 5]));
    g.set_outputs(vec![y]);
    parity(g, &[("a", &a), ("b", &b)]);
}

#[test]
fn concat_axis0_three_inputs() {
    let a = f32s(3, |i| i as f32);
    let b = f32s(6, |i| 10.0 + i as f32);
    let c = f32s(3, |i| 100.0 + i as f32);
    let mut g = Graph::new("cat3");
    let ai = g.input("a", sh(&[1, 3]));
    let bi = g.input("b", sh(&[2, 3]));
    let ci = g.input("c", sh(&[1, 3]));
    let y = g.concat(vec![ai, bi, ci], 0, sh(&[4, 3]));
    g.set_outputs(vec![y]);
    parity(g, &[("a", &a), ("b", &b), ("c", &c)]);
}

fn pool(kind: ReduceOp) -> (Graph, Vec<f32>) {
    // NCHW [1,1,4,4], 2x2 stride 2 → [1,1,2,2].
    let xs = f32s(16, |i| (i as f32 * 0.7).sin() + 0.1 * i as f32);
    let mut g = Graph::new("pool");
    let x = g.input("x", sh(&[1, 1, 4, 4]));
    let y = g.add_node(
        Op::Pool {
            kind,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        sh(&[1, 1, 2, 2]),
    );
    g.set_outputs(vec![y]);
    (g, xs)
}

#[test]
fn max_pool2d() {
    let (g, xs) = pool(ReduceOp::Max);
    parity(g, &[("x", &xs)]);
}

#[test]
fn avg_pool2d() {
    let (g, xs) = pool(ReduceOp::Mean);
    parity(g, &[("x", &xs)]);
}

fn conv(pad: usize) -> (Graph, Vec<f32>, Vec<f32>) {
    // input [1,2,4,4], weight [3,2,2,2], stride 1, dilation 1, groups 1.
    let (cin, h, w) = (2usize, 4usize, 4usize);
    let (cout, kh, kw) = (3usize, 2usize, 2usize);
    let xs = f32s(cin * h * w, |i| (i as f32 * 0.5).sin());
    let ws = f32s(cout * cin * kh * kw, |i| 0.1 * (i as f32 % 7.0) - 0.3);
    let mut g = Graph::new("conv");
    let x = g.input("x", sh(&[1, cin, h, w]));
    let weight = g.input("w", sh(&[cout, cin, kh, kw]));
    let y = g.conv2d(x, weight, [kh, kw], [1, 1], [pad, pad], [1, 1], 1);
    g.set_outputs(vec![y]);
    (g, xs, ws)
}

#[test]
fn conv2d_no_pad() {
    let (g, xs, ws) = conv(0);
    parity(g, &[("x", &xs), ("w", &ws)]);
}

#[test]
fn conv2d_padded() {
    let (g, xs, ws) = conv(1);
    parity(g, &[("x", &xs), ("w", &ws)]);
}
