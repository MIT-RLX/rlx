// SPDX-License-Identifier: MIT OR Apache-2.0
//! CUDA `FusedConvBiasAct` (cuDNN fused conv-bias-activation) vs CPU parity.
//!
//! CUDA is the only backend that claims `Op::FusedConvBiasAct`; the fusion
//! `Conv → Reshape(bias) → Expand → Add → [Activation]` collapses to it, and
//! the runtime folds bias + activation either via
//! `cudnnConvolutionBiasActivationForward` (cuDNN-friendly `groups==1, k>1`
//! shapes) or via the direct-conv kernel + `conv_bias_act_epilogue` (1×1 /
//! depthwise, or when libcudnn is absent). This exercises BOTH paths and both
//! the identity (bias-only) and relu/sigmoid/tanh epilogues.
//!
//! Runs on a CUDA rig. The `FuseConvBiasAct` matcher + decompose
//! are separately validated GPU-free in `conv_bias_act_fusion.rs`.

#![cfg(feature = "cuda")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_runtime::{Device, Session, is_available};

#[derive(Clone, Copy)]
struct Cfg {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
    groups: usize,
    act: Option<Activation>,
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_bias_act");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let bias = g.input("b", Shape::new(&[cfg.c_out], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let bias4 = g.reshape(
        bias,
        vec![1, cfg.c_out as i64, 1, 1],
        Shape::new(&[1, cfg.c_out, 1, 1], f),
    );
    let exp = g.add_node(
        Op::Expand {
            target_shape: vec![1, cfg.c_out as i64, cfg.h as i64, cfg.w as i64],
        },
        vec![bias4],
        out_s.clone(),
    );
    let out = g.binary(BinaryOp::Add, y, exp, out_s.clone());
    let out = match cfg.act {
        Some(a) => g.add_node(Op::Activation(a), vec![out], out_s),
        None => out,
    };
    g.set_outputs(vec![out]);
    g
}

fn run(cfg: &Cfg, device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..cfg.c_in * cfg.h * cfg.w)
        .map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k * cfg.k)
        .map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..cfg.c_out).map(|i| i as f32 * 0.05 - 0.2).collect();
    Session::new(device)
        .compile(build(cfg))
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("b", b.as_slice()),
        ])
        .pop()
        .unwrap()
}

fn check(name: &str, cfg: Cfg) {
    let cpu = run(&cfg, Device::Cpu);
    let cuda = run(&cfg, Device::Cuda);
    let maxd = cpu
        .iter()
        .zip(&cuda)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[conv+bias+act] {name}: max abs diff {maxd:.6}");
    assert!(maxd <= 1e-4, "{name}: cuda vs cpu diff {maxd}");
}

#[test]
fn cuda_conv_bias_act_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    // cuDNN-friendly (groups==1, k>1) → cudnnConvolutionBiasActivationForward.
    let friendly = [
        ("3x3-bias", None),
        ("3x3-relu", Some(Activation::Relu)),
        ("3x3-sigmoid", Some(Activation::Sigmoid)),
        ("3x3-tanh", Some(Activation::Tanh)),
    ];
    for (name, act) in friendly {
        check(
            name,
            Cfg {
                c_in: 4,
                c_out: 8,
                h: 16,
                w: 16,
                k: 3,
                p: 1,
                groups: 1,
                act,
            },
        );
    }

    // 1×1 pointwise + depthwise → direct-conv kernel + conv_bias_act_epilogue.
    check(
        "1x1-relu",
        Cfg {
            c_in: 5,
            c_out: 8,
            h: 12,
            w: 12,
            k: 1,
            p: 0,
            groups: 1,
            act: Some(Activation::Relu),
        },
    );
    check(
        "depthwise-relu",
        Cfg {
            c_in: 8,
            c_out: 8,
            h: 12,
            w: 12,
            k: 3,
            p: 1,
            groups: 8,
            act: Some(Activation::Relu),
        },
    );
    check(
        "depthwise-bias",
        Cfg {
            c_in: 8,
            c_out: 8,
            h: 12,
            w: 12,
            k: 3,
            p: 1,
            groups: 8,
            act: None,
        },
    );
}

/// funasr CAM++ frozen-BN block: `conv(no bias) → Mul(scale) → Add(shift) →
/// Relu`. `FuseConvAffineAct` folds the per-channel scale into the weights and
/// routes it through the same cuDNN fused path. Validates that fold on CUDA.
fn build_affine(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_affine");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let scale = g.input("s", Shape::new(&[cfg.c_out], f));
    let shift = g.input("b", Shape::new(&[cfg.c_out], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let cc = Shape::new(&[1, cfg.c_out, 1, 1], f);
    let scale_r = g.reshape(scale, vec![1, cfg.c_out as i64, 1, 1], cc.clone());
    let mul = g.binary(BinaryOp::Mul, y, scale_r, out_s.clone());
    let shift_r = g.reshape(shift, vec![1, cfg.c_out as i64, 1, 1], cc);
    let add = g.binary(BinaryOp::Add, mul, shift_r, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![add], out_s);
    g.set_outputs(vec![out]);
    g
}

fn run_affine(cfg: &Cfg, device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..cfg.c_in * cfg.h * cfg.w)
        .map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k * cfg.k)
        .map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5)
        .collect();
    let s: Vec<f32> = (0..cfg.c_out).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..cfg.c_out).map(|i| i as f32 * 0.05 - 0.2).collect();
    Session::new(device)
        .compile(build_affine(cfg))
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("s", s.as_slice()),
            ("b", b.as_slice()),
        ])
        .pop()
        .unwrap()
}

#[test]
fn cuda_conv_affine_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    // 3×3 friendly → FuseConvAffineAct folds scale into weights → cuDNN fused.
    for (name, cfg) in [
        (
            "affine-3x3-a",
            Cfg {
                c_in: 4,
                c_out: 8,
                h: 16,
                w: 16,
                k: 3,
                p: 1,
                groups: 1,
                act: None,
            },
        ),
        (
            "affine-3x3-b",
            Cfg {
                c_in: 3,
                c_out: 6,
                h: 12,
                w: 12,
                k: 3,
                p: 1,
                groups: 1,
                act: None,
            },
        ),
    ] {
        let cpu = run_affine(&cfg, Device::Cpu);
        let cuda = run_affine(&cfg, Device::Cuda);
        let maxd = cpu
            .iter()
            .zip(&cuda)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[conv+affine] {name}: max abs diff {maxd:.6}");
        assert!(maxd <= 1e-4, "{name}: cuda vs cpu diff {maxd}");
    }
}

/// funasr CAM++ residual block: `conv → Mul → Add(shift) → Add(residual) →
/// Relu`. `FuseConvAffineAct` folds it into `FusedConvBiasAct{has_residual}` →
/// cuDNN's `z`-operand fused call. Validates that residual fold on CUDA.
fn build_affine_residual(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_affine_res");
    let x = g.input("x", Shape::new(&[1, cfg.c_in, cfg.h, cfg.w], f));
    let weight = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let scale = g.input("s", Shape::new(&[cfg.c_out], f));
    let shift = g.input("b", Shape::new(&[cfg.c_out], f));
    let residual = g.input("r", Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f));
    let y = g.conv2d(
        x,
        weight,
        [cfg.k, cfg.k],
        [1, 1],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    let out_s = Shape::new(&[1, cfg.c_out, cfg.h, cfg.w], f);
    let cc = Shape::new(&[1, cfg.c_out, 1, 1], f);
    let scale_r = g.reshape(scale, vec![1, cfg.c_out as i64, 1, 1], cc.clone());
    let mul = g.binary(BinaryOp::Mul, y, scale_r, out_s.clone());
    let shift_r = g.reshape(shift, vec![1, cfg.c_out as i64, 1, 1], cc);
    let add = g.binary(BinaryOp::Add, mul, shift_r, out_s.clone());
    let res_add = g.binary(BinaryOp::Add, add, residual, out_s.clone());
    let out = g.add_node(Op::Activation(Activation::Relu), vec![res_add], out_s);
    g.set_outputs(vec![out]);
    g
}

fn run_affine_residual(cfg: &Cfg, device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..cfg.c_in * cfg.h * cfg.w)
        .map(|i| (i * 7 % 23) as f32 / 23.0 - 0.5)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k * cfg.k)
        .map(|i| (i * 5 % 17) as f32 / 17.0 - 0.5)
        .collect();
    let s: Vec<f32> = (0..cfg.c_out).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..cfg.c_out).map(|i| i as f32 * 0.05 - 0.2).collect();
    let r: Vec<f32> = (0..cfg.c_out * cfg.h * cfg.w)
        .map(|i| (i * 3 % 19) as f32 / 19.0 - 0.5)
        .collect();
    Session::new(device)
        .compile(build_affine_residual(cfg))
        .run(&[
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("s", s.as_slice()),
            ("b", b.as_slice()),
            ("r", r.as_slice()),
        ])
        .pop()
        .unwrap()
}

#[test]
fn cuda_conv_affine_residual_matches_cpu() {
    if !is_available(Device::Cuda) {
        eprintln!("skip: no CUDA device");
        return;
    }
    // ResNet block: conv+affine+residual+relu → cuDNN z-operand fused call.
    let cfg = Cfg {
        c_in: 8,
        c_out: 8,
        h: 16,
        w: 16,
        k: 3,
        p: 1,
        groups: 1,
        act: None,
    };
    let cpu = run_affine_residual(&cfg, Device::Cpu);
    let cuda = run_affine_residual(&cfg, Device::Cuda);
    let maxd = cpu
        .iter()
        .zip(&cuda)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[conv+affine+residual] max abs diff {maxd:.6}");
    assert!(maxd <= 1e-4, "residual: cuda vs cpu diff {maxd}");
}
