// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An `m <= 4` matmul (`matmul_gemv`) must round exactly like the same rows of
//! a larger one (`matmul.wgsl`): K summed in order.
//!
//! `matmul_gemv` used to split K eight ways and add the partial sums, so a
//! batch of up to four rows came out a few ULP away from the same rows in a
//! batch of five. On Metal, CoopF32 takes these shapes and sums in order;
//! a browser has no cooperative matrix and fell to the split gemv, so the
//! same model gave different features in Chrome and natively (bcgfm's
//! squeeze-excite `[N, 42]·[42, 10]`). The split form is fast math only now.
//!
//! Cooperative matrices are switched off here so the gemv path is the one
//! measured on every adapter. One test per file: the switch is process-wide.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// `a [m, k] · b [k, n] + bias` on wgpu, `b` and `bias` params.
fn mm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("gemv_order");
    let x = g.input("a", Shape::new(&[m, k], DType::F32));
    let w = g.param("b", Shape::new(&[k, n], DType::F32));
    let c = g.param("c", Shape::new(&[1, n], DType::F32));
    let y = g.mm(x, w);
    let y = g.add(y, c);
    g.set_outputs(vec![y]);
    let mut cg = Session::new(Device::Gpu).compile(g);
    cg.set_param("b", b);
    cg.set_param("c", bias);
    cg.run(&[("a", &a[..m * k])]).remove(0)
}

#[test]
fn small_m_matmul_rounds_like_the_tiled_kernel() {
    // SAFETY: set before any kernel is built; this file runs one test.
    unsafe {
        std::env::set_var("RLX_WGPU_NO_COOP_F32", "1");
    }
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    const BIG: usize = 8; // above the gemv cut-off, below `matmul_wide`'s
    let mut differing = Vec::new();
    for (k, n) in [
        (42usize, 10usize),
        (10, 42),
        (448, 16),
        (448, 32),
        (64, 64),
        (1000, 33),
    ] {
        let a = noise(BIG * k, 1);
        let b = noise(k * n, 2);
        let bias = noise(n, 3);
        let tiled = mm(BIG, k, n, &a, &b, &bias);
        for m in 1..=4 {
            let small = mm(m, k, n, &a, &b, &bias);
            let off = small
                .iter()
                .zip(&tiled)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            if off > 0 {
                differing.push(format!("[{m}x{k}]·[{k}x{n}]: {off}/{} differ", m * n));
            }
        }
    }
    assert!(
        differing.is_empty(),
        "an m <= 4 matmul rounds differently from the same rows at m = {BIG}:\n{}",
        differing.join("\n")
    );
}
