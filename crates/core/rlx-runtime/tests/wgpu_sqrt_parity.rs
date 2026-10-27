// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu `Sqrt` and `Rsqrt` must equal rlx-cpu bit for bit: IEEE `sqrt` is
//! correctly rounded, and CPU `Rsqrt` is `1 / x.sqrt()`, two correctly-rounded
//! steps. WGSL's `sqrt` and `inverseSqrt` promise neither, and the gap differs
//! by driver: Metal's fast-math `sqrt` is approximate, while on NVIDIA a Newton
//! step meant to repair Metal's moved an already exact root by 1 ULP.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

/// Normal positive operands across `2^-100 ..= 2^100`, with zero, infinity and
/// a few perfect squares (whose roots are exact) at the end.
fn operands() -> Vec<f32> {
    let mut s = 0x1234_5678_u32;
    let mut v: Vec<f32> = (0..1 << 16)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            f32::from_bits((s & 0x007f_ffff) | ((s >> 9) % 200 + 27) << 23)
        })
        .collect();
    v.extend([0.0, f32::INFINITY, 1.0, 4.0, 9.0, 0.25, 1e-30, 3.0e38]);
    v
}

fn same_bits(act: Activation) {
    let Some(_gpu) = common::serialize_gpu() else {
        return;
    };
    let x = operands();
    let n = x.len();
    let build = || {
        let mut g = Graph::new("sqrt");
        let i = g.input("x", Shape::new(&[n], DType::F32));
        let y = g.activation(act, i, Shape::new(&[n], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let run = |d: Device| Session::new(d).compile(build()).run(&[("x", &x)]).remove(0);
    let (gpu, cpu) = (run(Device::Gpu), run(Device::Cpu));
    let bad: Vec<String> = (0..n)
        .filter(|&i| gpu[i].to_bits() != cpu[i].to_bits())
        .map(|i| format!("{act:?}({:e}) = cpu {:e} wgpu {:e}", x[i], cpu[i], gpu[i]))
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {n} differ, e.g. {:?}",
        bad.len(),
        &bad[..bad.len().min(6)]
    );
}

#[test]
fn sqrt_is_correctly_rounded() {
    same_bits(Activation::Sqrt);
}

#[test]
fn rsqrt_is_one_over_a_correctly_rounded_sqrt() {
    same_bits(Activation::Rsqrt);
}
