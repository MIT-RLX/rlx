// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::FftQ` on Metal vs CPU.
//!
//! Both run the same integer kernel — Metal against its unified-memory arena,
//! CPU against its own — so the requirement is **bit-identical**, not close.
//! A fixed-point transform that only nearly agreed across devices would defeat
//! the reason to use one.

#![cfg(target_os = "macos")]

use rlx_ir::fft::{FftNorm, FftQScale};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build(n: usize, inverse: bool, scale: FftQScale) -> Graph {
    let mut g = Graph::new("fft_q");
    let x = g.input("x", Shape::new(&[2 * n], DType::I32));
    let y = g.fft_q(x, inverse, FftNorm::Backward, scale);
    g.set_outputs(vec![y]);
    g
}

fn bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(v: &[u8]) -> Vec<i32> {
    v.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn metal_fft_q_matches_cpu_exactly() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    for (n, inverse, scale) in [
        (256usize, false, FftQScale::None),
        (256, true, FftQScale::None),
        (512, false, FftQScale::Saturating),
        (128, false, FftQScale::EveryOther),
        (128, false, FftQScale::PerStage),
    ] {
        let mut x = vec![0i32; 2 * n];
        for t in 0..n {
            x[t] = ((t as f64 * 0.23).sin() * 15000.0) as i32;
        }
        let raw = bytes(&x);
        let feed: &[(&str, &[u8], DType)] = &[("x", &raw, DType::I32)];

        let g = build(n, inverse, scale);
        let metal = ints(
            &Session::new(Device::Metal)
                .compile(g.clone())
                .run_typed(feed)[0]
                .0,
        );
        let cpu = ints(&Session::new(Device::Cpu).compile(g).run_typed(feed)[0].0);

        // Agreement between two devices proves nothing if both mangle the
        // input the same way. Check against the kernel itself first.
        let mut want = x.clone();
        rlx_ir::fft::fft1d_q32_block(&mut want, 1, n, inverse, FftNorm::Backward, scale).unwrap();
        assert_eq!(
            cpu, want,
            "n={n} inverse={inverse} scale={scale:?}: the CPU graph path does not match \
             the kernel — the typed-IO boundary is not carrying i32"
        );
        assert_eq!(
            metal, cpu,
            "n={n} inverse={inverse} scale={scale:?}: Metal and CPU disagree"
        );
    }
}
