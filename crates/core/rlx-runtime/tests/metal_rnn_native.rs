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

//! Native Metal MSL kernels for `Op::Gru`/`Op::Rnn`/`Op::Mamba2` (single-layer
//! / unidir / no-carry → native path; `state_size ≤ 128` for Mamba2) vs CPU.
//! With the op claimed in `METAL_SUPPORTED_OPS` the legalizer keeps the fused
//! op (no decomposition) and dispatches the on-GPU kernel.

#![cfg(all(target_os = "macos", feature = "metal"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn ramp(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 23) as f32 - 11.0) * scale)
        .collect()
}

fn close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    let m = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(m <= 1e-3, "{what}: max abs diff {m:e} > 1e-3");
    eprintln!("{what}: max abs diff {m:.2e}");
}

#[test]
fn gru_native_metal_matches_cpu() {
    let (b, s, inp, h) = (2usize, 5, 6, 8);
    let inputs = [
        ("x", ramp(b * s * inp, 1, 0.1)),
        ("wih", ramp(3 * h * inp, 2, 0.05)),
        ("whh", ramp(3 * h * h, 3, 0.05)),
        ("bih", ramp(3 * h, 4, 0.02)),
        ("bhh", ramp(3 * h, 5, 0.02)),
    ];
    let build = || {
        let mut g = Graph::new("gru");
        let x = g.input("x", Shape::new(&[b, s, inp], DType::F32));
        let wih = g.input("wih", Shape::new(&[3 * h, inp], DType::F32));
        let whh = g.input("whh", Shape::new(&[3 * h, h], DType::F32));
        let bih = g.input("bih", Shape::new(&[3 * h], DType::F32));
        let bhh = g.input("bhh", Shape::new(&[3 * h], DType::F32));
        let y = g.gru(
            x,
            wih,
            whh,
            bih,
            bhh,
            None,
            h,
            1,
            false,
            Shape::new(&[b, s, h], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |d| {
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Session::new(d).compile(build()).run(&refs).pop().unwrap()
    };
    close("gru native", &run(Device::Metal), &run(Device::Cpu));
}

#[test]
fn rnn_native_metal_matches_cpu() {
    let (b, s, inp, h) = (2usize, 5, 6, 8);
    for relu in [false, true] {
        let inputs = [
            ("x", ramp(b * s * inp, 1, 0.1)),
            ("wih", ramp(h * inp, 2, 0.05)),
            ("whh", ramp(h * h, 3, 0.05)),
            ("bias", ramp(h, 4, 0.02)),
        ];
        let build = || {
            let mut g = Graph::new("rnn");
            let x = g.input("x", Shape::new(&[b, s, inp], DType::F32));
            let wih = g.input("wih", Shape::new(&[h, inp], DType::F32));
            let whh = g.input("whh", Shape::new(&[h, h], DType::F32));
            let bias = g.input("bias", Shape::new(&[h], DType::F32));
            let y = g.rnn(
                x,
                wih,
                whh,
                bias,
                None,
                h,
                1,
                false,
                relu,
                Shape::new(&[b, s, h], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        };
        let run = |d| {
            let refs: Vec<(&str, &[f32])> =
                inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
            Session::new(d).compile(build()).run(&refs).pop().unwrap()
        };
        close(
            &format!("rnn native relu={relu}"),
            &run(Device::Metal),
            &run(Device::Cpu),
        );
    }
}

#[test]
fn mamba2_native_metal_matches_cpu() {
    let (b, s, h, p, n) = (2usize, 4, 3, 8, 16);
    let inputs = [
        ("x", ramp(b * s * h * p, 1, 0.06)),
        (
            "dt",
            (0..b * s * h)
                .map(|i| 0.2 + 0.05 * (i % 5) as f32)
                .collect(),
        ),
        ("a", (0..h).map(|i| -0.5 - 0.3 * i as f32).collect()),
        ("b", ramp(b * s * h * n, 3, 0.05)),
        ("c", ramp(b * s * h * n, 7, 0.05)),
    ];
    let build = || {
        let mut g = Graph::new("mamba2");
        let x = g.input("x", Shape::new(&[b, s, h, p], DType::F32));
        let dt = g.input("dt", Shape::new(&[b, s, h], DType::F32));
        let a = g.input("a", Shape::new(&[h], DType::F32));
        let bb = g.input("b", Shape::new(&[b, s, h, n], DType::F32));
        let cc = g.input("c", Shape::new(&[b, s, h, n], DType::F32));
        let y = g.mamba2(
            x,
            dt,
            a,
            bb,
            cc,
            p,
            n,
            Shape::new(&[b, s, h, p], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };
    let run = |d| {
        let refs: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Session::new(d).compile(build()).run(&refs).pop().unwrap()
    };
    close("mamba2 native", &run(Device::Metal), &run(Device::Cpu));
}
