// SPDX-License-Identifier: MIT OR Apache-2.0
//! Param-invariant hoisting on CUDA. CUDA has no persistent `bind_handle`, so
//! the staging uses the feed-each-forward fallback (prepared boundary fed as an
//! ordinary input). Validates it matches plain CUDA + CPU and is stable.

#![cfg(feature = "cuda")]

use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use rlx_runtime::{CompileOptions, Device, Session, is_available};

/// `y = x @ (w · scale)` — `w·scale` is param-invariant (hoistable into prepare).
fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("hoist");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let w = g.param("w", Shape::new(&[4, 3], f));
    let scale = g.param("scale", Shape::new(&[4, 3], f));
    let w_scaled = g.binary(BinaryOp::Mul, w, scale, Shape::new(&[4, 3], f));
    let y = g.matmul(x, w_scaled, Shape::new(&[2, 3], f));
    g.set_outputs(vec![y]);
    g
}

fn feeds() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = (0..12).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let scale: Vec<f32> = (0..12).map(|i| 0.5 + (i % 4) as f32 * 0.25).collect();
    (x, w, scale)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn cuda_param_hoisting_matches_and_is_stable() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (x, w, scale) = feeds();

    let mut cpu = Session::new(Device::Cpu).compile(build());
    cpu.set_param("w", &w);
    cpu.set_param("scale", &scale);
    let y_cpu = cpu.run(&[("x", x.as_slice())]).pop().unwrap();

    let mut plain = Session::new(Device::Cuda).compile(build());
    plain.set_param("w", &w);
    plain.set_param("scale", &scale);
    let y_plain = plain.run(&[("x", x.as_slice())]).pop().unwrap();

    let mut opts = CompileOptions::new();
    opts.cache_param_invariant = true;
    let mut hoisted = Session::new(Device::Cuda).compile_with(build(), &opts);
    hoisted.set_param("w", &w);
    hoisted.set_param("scale", &scale);
    let y_hoist = hoisted.run(&[("x", x.as_slice())]).pop().unwrap();

    eprintln!(
        "[param-hoist cuda] cpu-vs-cuda {:.6}  cuda-vs-hoisted {:.6}",
        max_diff(&y_cpu, &y_plain),
        max_diff(&y_plain, &y_hoist)
    );
    assert!(max_diff(&y_cpu, &y_plain) <= 1e-4, "cpu vs cuda");
    assert!(
        max_diff(&y_plain, &y_hoist) <= 1e-4,
        "cuda plain vs hoisted"
    );

    // prepare runs once; feed-mode keeps results stable across forwards.
    for _ in 0..4 {
        let yn = hoisted.run(&[("x", x.as_slice())]).pop().unwrap();
        assert!(max_diff(&y_hoist, &yn) <= 1e-6, "hoisted run unstable");
    }
}
