// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU backend folds `conv3d → Add(per-channel bias) → Activation [→ Mul(scale) →
//! Add(shift)]` (or LeakyReLU as `Max(y, Mul(y, slope))` in place of the activation) into one
//! `Conv3d` thunk that applies all of it as it stores
//! (`fold_conv3d_bias_add`, `conv3d_activation_fusion`). That must be invisible: the fused graph's output equals,
//! bit for bit, the same layer kept unfused by giving the convolution a second reader.

#![cfg(feature = "cpu")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 1;
const CI: usize = 3;
const CO: usize = 4;
const S: usize = 6;

fn ramp(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

/// `act(conv3d(x, w) + b)`, then `· scale + shift` per channel with `norm` (a batch norm at
/// inference); with `keep_conv`, the raw convolution is a second output, which stops the folds.
fn layer(act: Option<Activation>, kernel: usize, norm: bool, keep_conv: bool) -> Graph {
    let mut g = Graph::new("conv3d_epilogue");
    let x = g.input("x", Shape::new(&[N, CI, S, S, S], DType::F32));
    let w = g.param(
        "w",
        Shape::new(&[CO, CI, kernel, kernel, kernel], DType::F32),
    );
    let b = g.param("b", Shape::new(&[CO], DType::F32));
    let p = kernel / 2;
    let y = g.conv3d(x, w, [1, 1, 1], [p, p, p], [1, 1, 1], 1);
    let out = Shape::new(&[N, CO, S, S, S], DType::F32);
    let br = g.add_node(
        Op::Reshape {
            new_shape: vec![1, CO as i64, 1, 1, 1],
        },
        vec![b],
        Shape::new(&[1, CO, 1, 1, 1], DType::F32),
    );
    let z = g.binary(BinaryOp::Add, y, br, out.clone());
    // `None`: LeakyReLU as FreeSurfer's SynthMorph and SynthStrip graphs spell it.
    let mut a = match act {
        Some(act) => g.activation(act, z, out.clone()),
        None => {
            let slope = g.param("slope", Shape::new(&[1], DType::F32));
            let slope = g.add_node(
                Op::Reshape {
                    new_shape: vec![1; 5],
                },
                vec![slope],
                Shape::new(&[1; 5], DType::F32),
            );
            let scaled = g.binary(BinaryOp::Mul, z, slope, out.clone());
            g.binary(BinaryOp::Max, z, scaled, out.clone())
        }
    };
    if norm {
        let per_channel = |g: &mut Graph, name: &str| {
            let p = g.param(name, Shape::new(&[CO], DType::F32));
            g.add_node(
                Op::Reshape {
                    new_shape: vec![1, CO as i64, 1, 1, 1],
                },
                vec![p],
                Shape::new(&[1, CO, 1, 1, 1], DType::F32),
            )
        };
        let s = per_channel(&mut g, "scale");
        a = g.binary(BinaryOp::Mul, a, s, out.clone());
        let t = per_channel(&mut g, "shift");
        a = g.binary(BinaryOp::Add, a, t, out);
    }
    g.set_outputs(if keep_conv { vec![a, y] } else { vec![a] });
    g
}

fn run(act: Option<Activation>, kernel: usize, norm: bool, keep_conv: bool) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(layer(act, kernel, norm, keep_conv));
    c.set_param("w", &ramp(CO * CI * kernel.pow(3), 1));
    c.set_param("b", &ramp(CO, 2));
    if act.is_none() {
        c.set_param("slope", &[0.2]);
    }
    if norm {
        c.set_param("scale", &ramp(CO, 4));
        c.set_param("shift", &ramp(CO, 5));
    }
    c.run(&[("x", &ramp(N * CI * S * S * S, 3))]).swap_remove(0)
}

#[test]
fn a_fused_bias_and_activation_equals_the_unfused_layer() {
    for act in [
        Some(Activation::Elu),
        Some(Activation::Relu),
        Some(Activation::Sigmoid),
        Some(Activation::Tanh),
        None,
    ] {
        for kernel in [1, 3, 5] {
            for norm in [false, true] {
                let fused = run(act, kernel, norm, false);
                let unfused = run(act, kernel, norm, true);
                assert_eq!(fused.len(), unfused.len());
                for (i, (a, b)) in fused.iter().zip(&unfused).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{act:?} k{kernel} norm {norm} element {i}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
