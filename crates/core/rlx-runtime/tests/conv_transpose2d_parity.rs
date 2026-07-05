// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//! `Op::ConvTranspose2d` cross-backend parity. The CPU kernel matches the
//! reference OCR (ocrs/rten) detection U-Net, so it is the reference here; every
//! GPU backend must reproduce it. This is the regression scaffold for the
//! Metal (wrong mask) / MLX (NHWC channel mismatch) / wgpu (op unsupported)
//! bugs surfaced by the OCR detection decoder.

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
}

fn build(cfg: &Cfg) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_transpose2d");
    let x = g.input("x", Shape::new(&[cfg.n, cfg.c_in, cfg.h, cfg.w], f));
    // ConvTranspose2d weight layout: [C_in, C_out, kH, kW].
    let wt = g.input("w", Shape::new(&[cfg.c_in, cfg.c_out, cfg.k, cfg.k], f));
    let y = g.conv_transpose2d(
        x,
        wt,
        [cfg.k, cfg.k],
        [cfg.s, cfg.s],
        [cfg.p, cfg.p],
        [1, 1],
        [0, 0],
        1,
    );
    g.set_outputs(vec![y]);
    g
}

fn inputs(cfg: &Cfg) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..cfg.n * cfg.c_in * cfg.h * cfg.w)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..cfg.c_in * cfg.c_out * cfg.k * cfg.k)
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

/// U-Net-style upsamples (stride 2) plus a no-stride case.
fn cfgs() -> Vec<(&'static str, Cfg)> {
    vec![
        (
            "up2-k2",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 4,
                h: 6,
                w: 6,
                k: 2,
                s: 2,
                p: 0,
            },
        ),
        (
            "up2-k4-p1",
            Cfg {
                n: 1,
                c_in: 6,
                c_out: 5,
                h: 5,
                w: 7,
                k: 4,
                s: 2,
                p: 1,
            },
        ),
        (
            "s1-k3-p1",
            Cfg {
                n: 2,
                c_in: 4,
                c_out: 3,
                h: 5,
                w: 5,
                k: 3,
                s: 1,
                p: 1,
            },
        ),
        // The detection U-Net decoder's actual upsample: kernel 3, stride 2, pad 1.
        (
            "up2-k3s2-p1",
            Cfg {
                n: 1,
                c_in: 8,
                c_out: 4,
                h: 6,
                w: 6,
                k: 3,
                s: 2,
                p: 1,
            },
        ),
        (
            "up2-k3s2-p1-odd",
            Cfg {
                n: 1,
                c_in: 5,
                c_out: 3,
                h: 7,
                w: 5,
                k: 3,
                s: 2,
                p: 1,
            },
        ),
        // U-Net scale.
        (
            "large-up2-k3s2",
            Cfg {
                n: 1,
                c_in: 64,
                c_out: 32,
                h: 32,
                w: 32,
                k: 3,
                s: 2,
                p: 1,
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
fn conv_transpose2d_cpu_runs() {
    for (name, cfg) in cfgs() {
        let out = run_on(&cfg, Device::Cpu);
        let expect = cfg.n
            * cfg.c_out
            * ((cfg.h - 1) * cfg.s + cfg.k - 2 * cfg.p)
            * ((cfg.w - 1) * cfg.s + cfg.k - 2 * cfg.p);
        assert_eq!(
            out.len(),
            expect,
            "cpu {name}: output len {} != {expect}",
            out.len()
        );
        assert!(out.iter().all(|x| x.is_finite()), "cpu {name}: non-finite");
        eprintln!("cpu {name}: {} elems ok", out.len());
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn conv_transpose2d_metal_matches_cpu() {
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
fn conv_transpose2d_mlx_matches_cpu() {
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
fn conv_transpose2d_wgpu_matches_cpu() {
    for (name, cfg) in cfgs() {
        assert_close(
            &format!("wgpu {name}"),
            &run_on(&cfg, Device::Gpu),
            &run_on(&cfg, Device::Cpu),
        );
    }
}
