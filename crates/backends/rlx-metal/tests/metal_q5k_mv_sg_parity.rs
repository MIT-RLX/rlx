// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Simdgroup Q5_K decode GEMV parity — `q5k_mv_f32_sg` vs `q5k_mv_f32` vs CPU.
//!
//! Q5_K was the only hot K-quant without a simdgroup GEMV; the
//! one-thread-per-row kernel is occupancy-starved (measured 90 GB/s at
//! n=17408 and 19 GB/s at n=1024, vs ~200/~120 for Q4_K and Q6_K). The new
//! kernel has 32 threads cooperate per output row via `simd_sum`.
//!
//! Both kernels dequantize identically — the only difference is which thread
//! accumulates what, so they differ solely by summation order. The scalar
//! kernel walks a row start-to-end; the simdgroup one sums 8-element strided
//! chunks and reduces across lanes. Hence a relative tolerance, not equality.
//!
//! Shapes cover both regimes: a wide FFN-like `n` and a narrow projection `n`
//! (Qwen3.5/3.8 put Q5_K on `attn_qkv` and `ssm_out`, which are narrow — the
//! case the old kernel handled worst). `n` is also checked when it is NOT a
//! multiple of the rows-per-threadgroup, to exercise the tail guard.
//!
//! One `#[test]`: the off-switch is a process-global env var.

#![cfg(target_os = "macos")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const K: usize = 512; // must be a multiple of 256 (QK_K)

fn weight(n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..K * n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5 + ((i % 11) as f32) * 0.01)
        .collect();
    rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q5K).expect("quantize")
}

fn build(n: usize, packed_len: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("q5k_mv");
    let x = g.input("x", Shape::new(&[1, K], f));
    let w = g.param("w", Shape::new(&[packed_len], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ5K,
        },
        vec![x, w],
        Shape::new(&[1, n], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-3);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[test]
fn q5k_mv_sg_matches_scalar_and_cpu() {
    // 1024/2048: aligned. 1030: n % (NSG*NR0 = 8) != 0 → tail guard.
    for n in [1024usize, 2048, 1030] {
        let packed = weight(n);
        let g = build(n, packed.len());
        let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.03).sin()).collect();

        let run = |device: Device| -> Vec<f32> {
            let mut s = Session::new(device).compile(g.clone());
            s.set_param_typed("w", &packed, DType::U8);
            s.run(&[("x", &x)]).remove(0)
        };

        let cpu = run(Device::Cpu);

        rlx_ir::env::unset("RLX_METAL_Q5K_SG_DISABLE"); // default = simdgroup
        let sg = run(Device::Metal);

        rlx_ir::env::set("RLX_METAL_Q5K_SG_DISABLE", "1");
        let scalar = run(Device::Metal);
        rlx_ir::env::unset("RLX_METAL_Q5K_SG_DISABLE");

        let ss = max_rel(&sg, &scalar);
        let sc = max_rel(&sg, &cpu);
        eprintln!("[q5k n={n}] sg-vs-scalar={ss:.3e} sg-vs-cpu={sc:.3e}");
        assert!(ss < 1e-5, "n={n}: sg vs scalar kernel {ss}");
        assert!(sc < 1e-5, "n={n}: sg vs cpu {sc}");
        assert_eq!(sg.len(), n, "n={n}: output length");
        assert!(
            sg.iter().all(|v| v.is_finite()),
            "n={n}: non-finite output (tail rows unwritten?)"
        );
    }
}
