// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-trailing-axis `Op::Reduce` parity against CPU.
//!
//! `SUPPORTED_OPS` is `OpKind`-granular, so a backend that claims `Reduce`
//! cannot also declare "…but only over a trailing axis block". Legalization
//! sees the claimed kind, calls the graph legal, and returns before
//! `rewrite_for_backend` can dispatch its own non-trailing-reduce lowering.
//! Both rlx-cuda and rlx-rocm therefore run `LowerNonLastAxisReduce`
//! themselves, at the entrance to their own compile.
//!
//! These cases are easy to lose: reverse-mode AD introduces `Reduce{axes:[0]}`
//! (a broadcast/bias gradient) only during the fusion and training stages, so
//! a forward-only smoke test never produces one. `reduce_trailing_control` is
//! the negative control — if it fails too, the harness is broken rather than
//! the lowering.
//!
//! Defaults to the compiled-in GPU backend; override with
//! `RLX_PARITY_DEVICE=rocm|cuda|metal|wgpu|…`.

use rlx_ir::op::{Op, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

/// Default to whichever GPU backend this binary was built with.
fn target() -> Device {
    if let Ok(s) = std::env::var("RLX_PARITY_DEVICE") {
        if let Ok(d) = rlx_runtime::parse_device(&s) {
            return d;
        }
    }
    for dev in [
        Device::Cuda,
        Device::Rocm,
        Device::Metal,
        Device::Gpu,
        Device::Vulkan,
    ] {
        if is_available(dev) {
            return dev;
        }
    }
    Device::Cpu
}

fn reduce_graph(dims: &[usize], axes: Vec<usize>, out: &[usize]) -> Graph {
    let mut g = Graph::new("nontrailing_reduce");
    let x = g.input("x", Shape::new(dims, F));
    let y = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes,
            keep_dim: false,
        },
        vec![x],
        Shape::new(out, F),
    );
    g.set_outputs(vec![y]);
    g
}

fn check(name: &str, dims: &[usize], axes: Vec<usize>, out: &[usize]) {
    let dev = target();
    if dev == Device::Cpu || !is_available(dev) {
        eprintln!("{name}: no GPU backend available — skipping");
        return;
    }
    let n: usize = dims.iter().product();
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let g = reduce_graph(dims, axes.clone(), out);

    let want = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &x)])[0]
        .clone();
    let got = Session::new(dev).compile(g).run(&[("x", &x)])[0].clone();

    assert_eq!(want.len(), got.len(), "{name}: output length differs");
    let err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "{name} [{dev:?}]: axes={axes:?} rank={} max|Δ|={err:.3e}",
        dims.len()
    );
    assert!(err < 1e-5, "{name}: {dev:?} != CPU (max|Δ|={err:.3e})");
}

#[test]
fn reduce_axis0_rank2() {
    // The shape reverse-mode AD emits for a bias gradient.
    check("axis0_rank2", &[4, 3], vec![0], &[3]);
}

#[test]
fn reduce_mid_axis_rank3() {
    check("mid_axis_rank3", &[2, 4, 3], vec![1], &[2, 3]);
}

#[test]
fn reduce_leading_axis_block_rank3() {
    check("leading_block_rank3", &[2, 4, 3], vec![0, 1], &[3]);
}

#[test]
fn reduce_trailing_control() {
    // Negative control: the shape the kernels natively implement.
    check("trailing_control", &[4, 3], vec![1], &[4]);
}
