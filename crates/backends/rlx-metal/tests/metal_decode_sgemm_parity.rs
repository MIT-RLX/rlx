// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: GPL-3.0-only

//! Decode-shaped sgemm (m=2 CFG) on Metal vs CPU — exercises SimdPadded /
//! MPS routing that replaced Naive for large k,n with m < 32.

#![cfg(target_os = "macos")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_decode_m2_sgemm_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // Prefer the padded simd path (disable MPS so we exercise MSL).
    rlx_ir::env::set("RLX_DISABLE_MPS", "1");
    rlx_ir::env::unset("RLX_METAL_SGEMM_PRECISE");

    let m = 2usize;
    let k = 256usize;
    let n = 256usize;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();

    let mut g = Graph::new("decode_m2_mm");
    let a_in = g.input("a", Shape::new(&[m, k], DType::F32));
    let b_p = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.mm(a_in, b_p);
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param("w", &b);
        c.run(&[("a", a.as_slice())]).remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), m * n);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let dot: f32 = metal.iter().zip(&cpu).map(|(x, y)| x * y).sum();
    let na = metal.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = cpu.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb + 1e-20);
    eprintln!("decode m=2 sgemm: max_abs={max_abs:.3e} cos={cos:.6}");
    assert!(
        max_abs < 2e-3 && cos > 0.9999,
        "Metal decode sgemm drifted: max_abs={max_abs} cos={cos}"
    );

    rlx_ir::env::unset("RLX_DISABLE_MPS");
}
