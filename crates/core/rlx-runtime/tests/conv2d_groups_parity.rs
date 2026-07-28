// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Conv2d` grouped/depthwise cross-backend parity — the detection U-Net is
//! depthwise-separable (`conv2d_bias_groups` with groups = channels), a classic
//! GPU-backend trap. CPU is the reference (matches rten); every GPU backend must
//! match it.

#![allow(dead_code)]

use rlx_ir::*;
use rlx_runtime::{Device, Session};

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
    groups: usize,
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv2d");
    let x = g.input("x", Shape::new(&[cfg.n, cfg.c_in, cfg.h, cfg.w], f));
    // Conv2d weight layout: [C_out, C_in/groups, kH, kW].
    let wt = g.input(
        "w",
        Shape::new(&[cfg.c_out, cfg.c_in / cfg.groups, cfg.k, cfg.k], f),
    );
    let y = g.conv2d(
        x,
        wt,
        [cfg.k, cfg.k],
        [cfg.s, cfg.s],
        [cfg.p, cfg.p],
        [1, 1],
        cfg.groups,
    );
    g.set_outputs(vec![y]);
    g
}

fn inputs(cfg: &Cfg) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.n * cfg.c_in * cfg.h * cfg.w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..cfg.c_out * (cfg.c_in / cfg.groups) * cfg.k * cfg.k)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.03)
        .collect();
    (x, w)
}

fn run_on(cfg: &Cfg, device: Device) -> Vec<f32> {
    let (x, w) = inputs(cfg);
    let mut c = Session::new(device).compile(build(cfg));
    c.run(&[("x", x.as_slice()), ("w", w.as_slice())])
        .pop()
        .unwrap()
}

fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "regular",
            Cfg {
                n: 1,
                c_in: 4,
                c_out: 6,
                h: 6,
                w: 6,
                k: 3,
                s: 1,
                p: 1,
                groups: 1,
            },
        ),
        (
            "depthwise-3x3",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 6,
                w: 6,
                k: 3,
                s: 1,
                p: 1,
                groups: 8,
            },
        ),
        (
            "depthwise-s2",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 7,
                w: 7,
                k: 3,
                s: 2,
                p: 1,
                groups: 8,
            },
        ),
        (
            "grouped-2",
            Cfg {
                n: 2,
                c_in: 8,
                c_out: 8,
                h: 5,
                w: 5,
                k: 3,
                s: 1,
                p: 1,
                groups: 2,
            },
        ),
        // U-Net scale — exposes size/channel-dependent GPU tiling bugs.
        (
            "large-regular",
            Cfg {
                n: 1,
                c_in: 64,
                c_out: 64,
                h: 64,
                w: 64,
                k: 3,
                s: 1,
                p: 1,
                groups: 1,
            },
        ),
        (
            "large-depthwise",
            Cfg {
                n: 1,
                c_in: 128,
                c_out: 128,
                h: 50,
                w: 50,
                k: 3,
                s: 1,
                p: 1,
                groups: 128,
            },
        ),
        (
            "large-1x1",
            Cfg {
                n: 1,
                c_in: 128,
                c_out: 64,
                h: 50,
                w: 50,
                k: 1,
                s: 1,
                p: 0,
                groups: 1,
            },
        ),
        // Detection in_conv resolution (800×600) — exposes spatial-size kernel bugs.
        (
            "hires-3x3",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: 3,
                s: 1,
                p: 1,
                groups: 1,
            },
        ),
        (
            "hires-dw",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: 3,
                s: 1,
                p: 1,
                groups: 8,
            },
        ),
        (
            "hires-1x1",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 8,
                h: 800,
                w: 600,
                k: 1,
                s: 1,
                p: 0,
                groups: 1,
            },
        ),
        (
            "hires-1ch",
            Cfg {
                n: 1,
                c_in: 1,
                c_out: 8,
                h: 800,
                w: 600,
                k: 1,
                s: 1,
                p: 0,
                groups: 1,
            },
        ),
    ]
}

fn assert_close(what: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{what}: length {} vs {}",
        actual.len(),
        reference.len()
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max <= 1e-4, "{what}: max abs diff {max} > 1e-4");
    eprintln!("{what}: max abs diff {max:.2e} (n={})", actual.len());
}

#[test]
fn conv2d_groups_cpu_runs() {
    for (name, cfg) in cfgs() {
        let out = run_on(&cfg, Device::Cpu);
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
        eprintln!("cpu {name}: {} elems ok", out.len());
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn conv2d_groups_metal_matches_cpu() {
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("metal {name}"),
            &run_on(&cfg, Device::Metal),
            &run_on(&cfg, Device::Cpu),
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn conv2d_groups_mlx_matches_cpu() {
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("mlx {name}"),
            &run_on(&cfg, Device::Mlx),
            &run_on(&cfg, Device::Cpu),
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn conv2d_groups_wgpu_matches_cpu() {
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("wgpu {name}"),
            &run_on(&cfg, Device::Gpu),
            &run_on(&cfg, Device::Cpu),
        );
    }
}
