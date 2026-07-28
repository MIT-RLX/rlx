// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Forward parity for the scalar activation batch (Floor/Ceil/Sign/Softplus/
//! Elu) on CPU/CUDA/WGPU. The `*_chained` variant places the activation in an
//! Activation→Binary chain that `mark_elementwise` would normally fuse — it
//! guards that `Activation::region_fusable()` keeps these unfused so the
//! standalone kernel (not the fused-region kernel) evaluates them.

#![cfg(feature = "cpu")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const ACTS: [Activation; 11] = [
    Activation::Floor,
    Activation::Ceil,
    Activation::Sign,
    Activation::Softplus,
    Activation::Elu,
    Activation::Erf,
    Activation::HardSwish,
    Activation::HardSigmoid,
    Activation::Mish,
    Activation::Softsign,
    Activation::LogSigmoid,
];

fn erf_ref(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t)
            * (-x * x).exp();
    s * y
}

fn eval(x: f32, a: Activation) -> f32 {
    match a {
        Activation::Floor => x.floor(),
        Activation::Ceil => x.ceil(),
        Activation::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Activation::Softplus => x.max(0.0) + (-(x.abs())).exp().ln_1p(),
        Activation::Elu => {
            if x > 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        Activation::Erf => erf_ref(x),
        Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
        Activation::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        Activation::Mish => x * (x.max(0.0) + (-(x.abs())).exp().ln_1p()).tanh(),
        Activation::Softsign => x / (1.0 + x.abs()),
        Activation::LogSigmoid => x.min(0.0) - (-(x.abs())).exp().ln_1p(),
        _ => unreachable!(),
    }
}

fn run(device: Device, a: Activation, chained: bool, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("act");
    let dims = [x.len()];
    let inp = g.input("x", Shape::new(&dims, DType::F32));
    let s = Shape::new(&dims, DType::F32);
    let y = g.add_node(Op::Activation(a), vec![inp], s.clone());
    // Optionally follow with a Mul (an Activation→Binary chain the region-fusion
    // pass would fold — must stay unfused for these activations).
    let out = if chained {
        let two = g.add_node(
            Op::Constant {
                data: 2.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        g.add_node(Op::Binary(BinaryOp::Mul), vec![y, two], s)
    } else {
        y
    };
    g.set_outputs(vec![out]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn xs() -> Vec<f32> {
    vec![-2.6, -1.0, -0.4, 0.0, 0.4, 1.0, 2.6, 3.9, -3.3, 0.7]
}

#[test]
fn activation_batch_cpu_matches_reference() {
    let x = xs();
    for a in ACTS {
        let want: Vec<f32> = x.iter().map(|&v| eval(v, a)).collect();
        let got = run(Device::Cpu, a, false, &x);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-5, "{a:?}[{i}]: got {g} want {w}");
        }
        // Chained (fusion-gate stress): result must be 2·act(x).
        let want2: Vec<f32> = want.iter().map(|v| v * 2.0).collect();
        let got2 = run(Device::Cpu, a, true, &x);
        for (i, (g, w)) in got2.iter().zip(&want2).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5,
                "{a:?} chained[{i}]: got {g} want {w}"
            );
        }
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_device(device: Device, label: &str) {
    let x = xs();
    for a in ACTS {
        for chained in [false, true] {
            // Floor/Ceil/Sign are exact integers; the rest use transcendental /
            // division ops (erf polynomial, ln_1p vs log(1+…), tanh, /), so a
            // small relative tolerance covers cross-backend last-digit drift.
            let tol = match a {
                Activation::Floor | Activation::Ceil | Activation::Sign => 0.0,
                _ => 1e-4,
            };
            let got = run(device, a, chained, &x);
            let want = run(Device::Cpu, a, chained, &x);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= tol * (1.0 + w.abs()),
                    "{label} {a:?} chained={chained} [{i}]: got {g} want {w}"
                );
            }
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn activation_batch_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn activation_batch_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        return;
    }
    check_device(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn activation_batch_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn activation_batch_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}
