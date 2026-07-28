// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity for `Op::Im2Col` on MLX (newly added, primitive-composed) vs the CPU
//! reference. NCHW input → rows layout `[N·H_out·W_out, C·kH·kW]`.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
}

fn build(cfg: &Cfg) -> Graph {
    let mut g = Graph::new("im2col");
    let x = g.input("x", Shape::new(&[cfg.n, cfg.c, cfg.h, cfg.w], DType::F32));
    let y = g.im2col(
        x,
        [cfg.k, cfg.k],
        [cfg.s, cfg.s],
        [cfg.p, cfg.p],
        [cfg.d, cfg.d],
    );
    g.set_outputs(vec![y]);
    g
}

fn run_on(cfg: &Cfg, device: Device) -> Vec<f32> {
    let x: Vec<f32> = (0..cfg.n * cfg.c * cfg.h * cfg.w)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.05)
        .collect();
    let mut exe = Session::new(device).compile(build(cfg));
    exe.run(&[("x", x.as_slice())]).pop().unwrap()
}

fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "3x3s1p1",
            Cfg {
                n: 1,
                c: 4,
                h: 8,
                w: 8,
                k: 3,
                s: 1,
                p: 1,
                d: 1,
            },
        ),
        (
            "3x3s2p1",
            Cfg {
                n: 2,
                c: 8,
                h: 10,
                w: 12,
                k: 3,
                s: 2,
                p: 1,
                d: 1,
            },
        ),
        (
            "1x1s1p0",
            Cfg {
                n: 1,
                c: 16,
                h: 5,
                w: 5,
                k: 1,
                s: 1,
                p: 0,
                d: 1,
            },
        ),
        (
            "3x3d2p2",
            Cfg {
                n: 1,
                c: 3,
                h: 9,
                w: 9,
                k: 3,
                s: 1,
                p: 2,
                d: 2,
            },
        ),
        (
            "2x2s2p0",
            Cfg {
                n: 2,
                c: 6,
                h: 8,
                w: 8,
                k: 2,
                s: 2,
                p: 0,
                d: 1,
            },
        ),
    ]
}

#[allow(dead_code)] // used only by the `mlx`-gated parity test below
fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-5, "{what}: max abs diff {max:e} > 1e-5");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", a.len());
}

#[test]
fn im2col_cpu_runs() {
    for (name, cfg) in cfgs() {
        let out = run_on(&cfg, Device::Cpu);
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn im2col_mlx_matches_cpu() {
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("im2col mlx {name}"),
            &run_on(&cfg, Device::Mlx),
            &run_on(&cfg, Device::Cpu),
        );
    }
}
