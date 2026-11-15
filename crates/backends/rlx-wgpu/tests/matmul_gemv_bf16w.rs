// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An `m <= 4` matmul against a packed-BF16 weight (`matmul_gemv_bf16w`)
//! must round exactly like the same rows of a larger one (`matmul_bf16w`'s
//! tiles): K summed in order, the weight widened exactly. Odd widths take
//! its unpaired path; widths past 128 span workgroups.
//!
//! Cooperative matrices are switched off so these are the kernels measured
//! on every adapter. One test per file: the switch is process-wide.

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

/// `a [m, k] · b [k, n]` on wgpu, `b` a BF16 param.
fn mm(m: usize, k: usize, n: usize, a: &[f32], b: &[half::bf16]) -> Vec<f32> {
    let mut g = Graph::new("gemv_bf16w");
    let x = g.input("a", Shape::new(&[m, k], DType::F32));
    let w = g.param("b", Shape::new(&[k, n], DType::BF16));
    let y = g.mm(x, w);
    g.set_outputs(vec![y]);
    let mut cg = Session::new(Device::Gpu).compile(g);
    let bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    cg.set_param_typed("b", &bytes, DType::BF16);
    cg.run(&[("a", &a[..m * k])]).remove(0)
}

#[test]
fn small_m_bf16_matmul_rounds_like_the_tiled_kernel() {
    // SAFETY: set before any kernel is built; this file runs one test.
    unsafe {
        std::env::set_var("RLX_WGPU_NO_COOP_F32", "1");
    }
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    const BIG: usize = 8; // above the gemv cut-off, below `matmul_wide`'s
    let before = rlx_wgpu::bf16_packed_dispatch_count();
    let mut differing = Vec::new();
    for (k, n) in [
        (42usize, 10usize),
        (10, 42),
        (64, 33),
        (448, 130),
        (1000, 256),
        (7, 1),
    ] {
        let a = noise(BIG * k, 1);
        let b: Vec<half::bf16> = noise(k * n, 2)
            .into_iter()
            .map(half::bf16::from_f32)
            .collect();
        let tiled = mm(BIG, k, n, &a, &b);
        // Against the exact product too: widening a bf16 is exact, so only
        // f32 rounding of the sums separates them.
        for r in 0..BIG {
            for c in 0..n {
                let want: f64 = (0..k)
                    .map(|i| a[r * k + i] as f64 * b[i * n + c].to_f64())
                    .sum();
                let got = tiled[r * n + c] as f64;
                assert!(
                    (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "[{k}x{n}] ({r},{c}): {got} vs {want}"
                );
            }
        }
        for m in 1..=4 {
            let small = mm(m, k, n, &a, &b);
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
        rlx_wgpu::bf16_packed_dispatch_count() > before,
        "the BF16 weight was not kept packed"
    );
    assert!(
        differing.is_empty(),
        "an m <= 4 bf16 matmul rounds differently from the same rows at m = {BIG}:\n{}",
        differing.join("\n")
    );
}
