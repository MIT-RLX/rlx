// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `assoc-scan`: on Metal an affine `Scan` is rewritten to a log-depth prefix
//! scan; its output must match the CPU, which keeps the sequential loop. The
//! other GPU backends keep their Scan path and are checked alongside.
#![cfg(feature = "assoc-scan")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

mod common;

const T: usize = 300;
const D: usize = 48;

/// `h = sigmoid(x)·h + y`, full trajectory.
fn gated_recurrence() -> Graph {
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let mut body = Graph::new("body");
    let h = body.input("h", sh(&[D]));
    let x = body.input("x", sh(&[D]));
    let y = body.input("y", sh(&[D]));
    let a = body.activation(Activation::Sigmoid, x, sh(&[D]));
    let ah = body.binary(BinaryOp::Mul, a, h, sh(&[D]));
    let next = body.binary(BinaryOp::Add, ah, y, sh(&[D]));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("outer");
    let h0 = g.input("h0", sh(&[D]));
    let xs = g.input("xs", sh(&[T, D]));
    let ys = g.input("ys", sh(&[T, D]));
    let scan = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: T as u32,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 2,
            num_checkpoints: 0,
        },
        vec![h0, xs, ys],
        sh(&[T, D]),
    );
    g.set_outputs(vec![scan]);
    g
}

fn run(dev: Device) -> Vec<f32> {
    let wave = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * s).sin()).collect() };
    let (h0, xs, ys) = (wave(D, 0.37), wave(T * D, 0.011), wave(T * D, 0.023));
    let opts = CompileOptions::new().scan_unroll_max_length(0);
    let mut sess = Session::new(dev).compile_with(gated_recurrence(), &opts);
    sess.run(&[("h0", &h0[..]), ("xs", &xs[..]), ("ys", &ys[..])])
        .remove(0)
}

fn parity_vs_cpu(dev: Device) {
    if common::skip_unless(dev) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(dev);
    let want = run(Device::Cpu);
    let got = run(dev);
    let err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(err < 1e-4, "{dev:?}: max rel err {err}");
}

#[test]
fn metal() {
    parity_vs_cpu(Device::Metal);
}

#[test]
fn wgpu() {
    parity_vs_cpu(Device::Gpu);
}

#[test]
fn mlx() {
    parity_vs_cpu(Device::Mlx);
}

#[test]
fn cuda() {
    parity_vs_cpu(Device::Cuda);
}

#[test]
fn rocm() {
    parity_vs_cpu(Device::Rocm);
}

#[test]
fn vulkan() {
    parity_vs_cpu(Device::Vulkan);
}
