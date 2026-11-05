// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated activation backward kernels stay finite on extreme finite
//! inputs on every backend. SiLU's differentiated form returned NaN for
//! x ∈ [−88, −85] (and Sigmoid's below −88): a single such pre-activation —
//! a Qwen3 MLP has them — turned every gradient of a LoRA backward to NaN on
//! Metal while the CPU's hand-written kernel stayed finite.

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn activation_gradients_are_finite_and_agree_on_extreme_inputs() {
    // The old NaN window, the clamp edges, and the far tails.
    let mut xs: Vec<f32> = (0..=64).map(|i| -88.5 + i as f32 * 0.0625).collect();
    xs.extend([
        -200.0, -120.0, -89.0, -60.0, -1.0, 0.0, 1.0, 60.0, 88.0, 89.0, 120.0, 200.0,
    ]);
    let n = xs.len();
    let dy = vec![1.0f32; n];
    let devices: Vec<Device> = rlx_runtime::available_devices()
        .into_iter()
        .filter(|d| !matches!(d, Device::Ane | Device::Xdna))
        .collect();
    for act in [
        Activation::Silu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Gelu,
    ] {
        let mut g = Graph::new("act_bwd");
        let sh = Shape::new(&[n], DType::F32);
        let x = g.input("x", sh.clone());
        let d = g.input("dy", sh.clone());
        let y = g.activation_backward(act, x, d);
        g.set_outputs(vec![y]);
        let feed = [("x", &xs[..]), ("dy", &dy[..])];
        let cpu = Session::new(Device::Cpu)
            .compile(g.clone())
            .run(&feed)
            .remove(0);
        for &dev in &devices {
            let got = Session::new(dev).compile(g.clone()).run(&feed).remove(0);
            for ((&x, &a), &b) in xs.iter().zip(&got).zip(&cpu) {
                assert!(a.is_finite(), "{dev:?} {act:?}'({x}) = {a}");
                assert!(
                    (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
                    "{dev:?} {act:?}'({x}) = {a}, CPU {b}"
                );
            }
        }
    }
}
