// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::FftQ` on Metal vs CPU, with the fixed-point data produced *inside* the
//! graph.
//!
//! That is the realistic shape: a frontend casts its windowed samples to i32,
//! transforms, and casts the magnitudes back. It also sidesteps a separate
//! limitation — the runtime's typed-IO wrapper widens integer *inputs* to f32
//! before they reach the arena, so an i32 tensor handed in from outside does
//! not survive on these backends.
//!
//! Both devices run the same integer kernel, so the requirement is
//! bit-identical, and the result is also checked against the kernel directly —
//! two devices agreeing proves nothing if both mangle the input the same way.

#![cfg(target_os = "macos")]

use rlx_ir::fft::{FftNorm, FftQScale};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// f32 in → Cast to i32 → FftQ → Cast back to f32.
fn build(n: usize, inverse: bool, scale: FftQScale) -> Graph {
    let mut g = Graph::new("fft_q_ingraph");
    let x = g.input("x", Shape::new(&[2 * n], DType::F32));
    let xi = g.add_node(
        Op::Cast { to: DType::I32 },
        vec![x],
        Shape::new(&[2 * n], DType::I32),
    );
    let y = g.fft_q(xi, inverse, FftNorm::Backward, scale);
    let yf = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![y],
        Shape::new(&[2 * n], DType::F32),
    );
    g.set_outputs(vec![yf]);
    g
}

#[test]
fn metal_fft_q_matches_cpu_and_the_kernel() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    for (n, inverse, scale) in [
        (256usize, false, FftQScale::None),
        (256, true, FftQScale::None),
        (128, false, FftQScale::EveryOther),
        (128, false, FftQScale::PerStage),
    ] {
        let mut xi = vec![0i32; 2 * n];
        for t in 0..n {
            xi[t] = ((t as f64 * 0.23).sin() * 15000.0) as i32;
        }
        let xf: Vec<f32> = xi.iter().map(|&v| v as f32).collect();

        let g = build(n, inverse, scale);
        let dev_out = Session::new(Device::Metal)
            .compile(g.clone())
            .run(&[("x", &xf)])
            .remove(0);
        let cpu_out = Session::new(Device::Cpu)
            .compile(g)
            .run(&[("x", &xf)])
            .remove(0);

        let mut want = xi.clone();
        rlx_ir::fft::fft1d_q32_block(&mut want, 1, n, inverse, FftNorm::Backward, scale).unwrap();
        let want_f: Vec<f32> = want.iter().map(|&v| v as f32).collect();

        assert_eq!(
            cpu_out, want_f,
            "n={n} inverse={inverse} scale={scale:?}: the CPU graph path does not match the kernel"
        );
        assert_eq!(
            dev_out, cpu_out,
            "n={n} inverse={inverse} scale={scale:?}: Metal and CPU disagree"
        );
    }
}
