// SPDX-License-Identifier: MIT OR Apache-2.0
//! `conv2d → Expand(bias) → add` cross-backend parity (the detection
//! `conv2d_bias` sequence). conv and Expand pass standalone on Metal, but the
//! detection diverges at full resolution — this tests the combination.

#![allow(dead_code)]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_runtime::{Device, Session};

#[derive(Clone, Copy)]
struct Cfg {
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    p: usize,
    groups: usize,
    relu: bool,
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_bias");
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
    let out = if cfg.relu {
        g.add_node(Op::Activation(Activation::Relu), vec![out], out_s)
    } else {
        out
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
    let mut c = Session::new(device).compile(build(cfg));
    c.run(&[
        ("x", x.as_slice()),
        ("w", w.as_slice()),
        ("b", b.as_slice()),
    ])
    .pop()
    .unwrap()
}

fn check(name: &str, cfg: Cfg) {
    let cpu = run(&cfg, Device::Cpu);
    let dev = run(&cfg, Device::Metal);
    let maxd = cpu
        .iter()
        .zip(&dev)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[conv+bias] {name} {}x{}: max abs diff {maxd:.5}",
        cfg.h, cfg.w
    );
    assert!(maxd <= 1e-4, "{name}: conv+bias metal vs cpu diff {maxd}");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn conv_bias_metal_matches_cpu() {
    check(
        "pw-hires",
        Cfg {
            c_in: 1,
            c_out: 8,
            h: 800,
            w: 600,
            k: 1,
            p: 0,
            groups: 1,
            relu: false,
        },
    );
    check(
        "pw-relu-hires",
        Cfg {
            c_in: 1,
            c_out: 8,
            h: 800,
            w: 600,
            k: 1,
            p: 0,
            groups: 1,
            relu: true,
        },
    );
    check(
        "dw-relu-hires",
        Cfg {
            c_in: 8,
            c_out: 8,
            h: 800,
            w: 600,
            k: 3,
            p: 1,
            groups: 8,
            relu: true,
        },
    );
    check(
        "relu-small",
        Cfg {
            c_in: 1,
            c_out: 8,
            h: 6,
            w: 6,
            k: 1,
            p: 0,
            groups: 1,
            relu: true,
        },
    );
}
