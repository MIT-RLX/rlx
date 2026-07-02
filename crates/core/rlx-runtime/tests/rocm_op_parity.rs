// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! CPU vs ROCm parity through [`Session`] for native Session-path ops.

#![cfg(all(feature = "cpu", feature = "rocm"))]

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session, is_available};

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} len");
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max < tol, "{label}: max_abs_diff={max} tol={tol}");
}

fn run_pair(g: Graph, inputs: &[(&str, &[f32])], tol: f32, label: &str) {
    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_op_parity {label} (unavailable)");
        return;
    }
    let opts = CompileOptions::default();
    let cpu = Session::new(Device::Cpu)
        .compile_with(g.clone(), &opts)
        .run(inputs);
    let gpu = Session::new(Device::Rocm)
        .compile_with(g, &opts)
        .run(inputs);
    assert_eq!(cpu.len(), gpu.len(), "{label} output count");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_close(c, g, tol, &format!("{label}[{i}]"));
    }
}

#[test]
fn rocm_binary_add_parity() {
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);
    run_pair(
        g,
        &[
            ("x", &[1.0, 2.0, 3.0, 4.0]),
            ("y", &[10.0, 20.0, 30.0, 40.0]),
        ],
        1e-5,
        "binary_add",
    );
}

#[test]
fn rocm_relu_parity() {
    let mut g = Graph::new("relu");
    let x = g.input("x", Shape::new(&[5], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[5], DType::F32));
    g.set_outputs(vec![y]);
    run_pair(g, &[("x", &[-2.0, -0.5, 0.0, 1.0, 3.0])], 1e-5, "relu");
}

#[test]
fn rocm_softmax_parity() {
    let mut g = Graph::new("softmax");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.softmax(x, -1, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);
    let data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25).collect();
    run_pair(g, &[("x", &data)], 1e-4, "softmax");
}

#[test]
fn rocm_reduce_sum_parity() {
    let mut g = Graph::new("sum");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.reduce(
        x,
        ReduceOp::Sum,
        vec![1],
        false,
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![y]);
    run_pair(
        g,
        &[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        1e-4,
        "reduce_sum",
    );
}

#[test]
fn rocm_group_norm_parity() {
    let n = 1usize;
    let c = 8usize;
    let h = 4usize;
    let w = 4usize;
    let num_groups = 2usize;
    let x: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.01 - 0.2)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + 0.02 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.01 * i as f32).collect();

    let mut g = Graph::new("gn");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[c], DType::F32));
    let b_p = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x_in, g_p, b_p, num_groups, 1e-5);
    g.set_outputs(vec![y]);

    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_op_parity group_norm (unavailable)");
        return;
    }
    let opts = CompileOptions::default();
    let mut cpu_sess = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    cpu_sess.set_param("gamma", &gamma);
    cpu_sess.set_param("beta", &beta);
    let cpu = cpu_sess.run(&[("x", &x)]);

    let mut rocm_sess = Session::new(Device::Rocm).compile_with(g, &opts);
    rocm_sess.set_param("gamma", &gamma);
    rocm_sess.set_param("beta", &beta);
    let gpu = rocm_sess.run(&[("x", &x)]);

    assert_close(&cpu[0], &gpu[0], 1e-4, "group_norm");
}

#[test]
fn rocm_resize_nearest_2x_parity() {
    let n = 1usize;
    let c = 3usize;
    let h = 5usize;
    let w = 7usize;
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.003).collect();

    let mut g = Graph::new("up2");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let y = g.add_node(
        Op::ResizeNearest2x,
        vec![x_in],
        Shape::new(&[n, c, h * 2, w * 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    run_pair(g, &[("x", &x)], 1e-6, "resize_nearest_2x");
}
