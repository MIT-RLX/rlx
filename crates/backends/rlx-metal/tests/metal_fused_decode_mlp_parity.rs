// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fused decode-layer MLP (SwiGLU) parity — Metal fused vs unfused vs CPU.
//!
//! Builds the single-token (m == 1) packed-GGUF MLP op chain
//!   rms_norm → gate(DequantMatMul) → up(DequantMatMul) → silu → mul →
//!   down(DequantMatMul) → add(residual)
//! and runs it three ways:
//!   * Metal with `fuse_decode_mlp` ON  (RLX_METAL_FUSE_DECODE unset/1)
//!   * Metal with `fuse_decode_mlp` OFF (RLX_METAL_FUSE_DECODE=0)
//!   * CPU reference
//! The fused path collapses the six matmul/elementwise dispatches into two
//! (`FusedMlpGateUpSwiGLU` + `FusedMlpDownResidual`) reusing the exact
//! `q4k_mv_f32` / `q6k_mm_f32` dequant math. `fused_decode_mlp_blocks()` proves
//! the fused thunks were actually emitted.
//!
//! All cases run in ONE `#[test]` so they execute serially on a single thread:
//! the off-switch is a process-global env var and the m > 1 fallback shares the
//! global MPS matrix cache (see the prefill parity test for the same rationale).

#![cfg(target_os = "macos")]

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const H: usize = 256; // hidden dim  (gate/up k, down n) — Q4_K needs % 256
const I: usize = 512; // intermediate (gate/up n, down k) — Q6_K needs % 256

/// Row-major `[n, k]` weight: output column c owns k contiguous values.
fn weight(seed: f32, k: usize, n: usize, ggml: rlx_gguf::GgmlType) -> Vec<u8> {
    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * seed).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w_row, ggml).expect("quantize")
}

fn build(down_scheme: QuantScheme, packed_lens: (usize, usize, usize)) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("fused_decode_mlp");
    let r = g.input("r", Shape::new(&[1, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let gate_w = g.param("gate_w", Shape::new(&[packed_lens.0], DType::U8));
    let up_w = g.param("up_w", Shape::new(&[packed_lens.1], DType::U8));
    let down_w = g.param("down_w", Shape::new(&[packed_lens.2], DType::U8));

    let normed = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![r, gamma, beta],
        Shape::new(&[1, H], f),
    );
    let gate = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![normed, gate_w],
        Shape::new(&[1, I], f),
    );
    let up = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![normed, up_w],
        Shape::new(&[1, I], f),
    );
    let gate_act = g.add_node(
        Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[1, I], f),
    );
    let prod = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_act, up],
        Shape::new(&[1, I], f),
    );
    let down = g.add_node(
        Op::DequantMatMul {
            scheme: down_scheme,
        },
        vec![prod, down_w],
        Shape::new(&[1, H], f),
    );
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![r, down],
        Shape::new(&[1, H], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Max abs error normalized by the reference's dynamic range. The down GEMV
/// sums I=512 terms so outputs are O(10); a raw 1e-4 bound is meaningless,
/// while reduction-order (simdgroup vs single-thread) noise scales with the
/// magnitude — relative error is the honest metric.
fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-3);
    max_abs(a, b) / scale
}

fn run_case(down_ggml: rlx_gguf::GgmlType, down_scheme: QuantScheme, label: &str) {
    let gate_p = weight(0.011, H, I, rlx_gguf::GgmlType::Q4K);
    let up_p = weight(0.017, H, I, rlx_gguf::GgmlType::Q4K);
    let down_p = weight(0.013, I, H, down_ggml);
    let g = build(down_scheme, (gate_p.len(), up_p.len(), down_p.len()));

    let r: Vec<f32> = (0..H).map(|i| ((i as f32) * 0.03).sin()).collect();
    let gamma: Vec<f32> = (0..H)
        .map(|i| 1.0 + ((i as f32) * 0.001).cos() * 0.1)
        .collect();
    let beta = vec![0.0f32; H];

    let run = |device: Device| -> Vec<f32> {
        let mut s = Session::new(device).compile(g.clone());
        s.set_param_typed("gate_w", &gate_p, DType::U8);
        s.set_param_typed("up_w", &up_p, DType::U8);
        s.set_param_typed("down_w", &down_p, DType::U8);
        s.run(&[("r", &r), ("gamma", &gamma), ("beta", &beta)])
            .remove(0)
    };

    let cpu = run(Device::Cpu);

    // ── Production config (simdgroup GEMV enabled): fuse ON vs OFF. ──────────
    rlx_ir::env::unset("RLX_METAL_Q4K_SG_DISABLE");
    let before = rlx_metal::thunk::fused_decode_mlp_blocks();
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE"); // default = ON
    let fused = run(Device::Metal);
    let fired = rlx_metal::thunk::fused_decode_mlp_blocks() - before;
    assert_eq!(fired, 1, "{label}: expected exactly one fused MLP block");

    rlx_ir::env::set("RLX_METAL_FUSE_DECODE", "0");
    let before_off = rlx_metal::thunk::fused_decode_mlp_blocks();
    let unfused = run(Device::Metal);
    assert_eq!(
        rlx_metal::thunk::fused_decode_mlp_blocks(),
        before_off,
        "{label}: off-switch must NOT fuse"
    );
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");

    let fu = max_rel(&fused, &unfused);
    let fc = max_rel(&fused, &cpu);
    let uc = max_rel(&unfused, &cpu);
    eprintln!(
        "[{label}] production (relative): fused-vs-unfused={fu:.3e} fused-vs-cpu={fc:.3e} \
         unfused-vs-cpu={uc:.3e}"
    );
    // Both fused and unfused are GEMV reductions; the fused single-thread order
    // differs from the unfused simdgroup order by reduction-order noise only.
    assert!(fu < 1e-3, "{label}: fused vs unfused (prod, rel) {fu}");
    assert!(fc < 5e-3, "{label}: fused vs cpu (rel) {fc}");

    // ── Bit-exactness config (single-thread GEMV both sides). ───────────────
    // With the simdgroup GEMV disabled the unfused gate/up/down run the same
    // single-thread q4k_mv_f32 accumulation the fused kernels inline, so the
    // all-Q4_K path is byte-identical (down Q6_K still uses the dequant+MPS
    // fallback when unfused, so it stays within tolerance, not exact).
    rlx_ir::env::set("RLX_METAL_Q4K_SG_DISABLE", "1");
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    let fused_st = run(Device::Metal);
    rlx_ir::env::set("RLX_METAL_FUSE_DECODE", "0");
    let unfused_st = run(Device::Metal);
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    rlx_ir::env::unset("RLX_METAL_Q4K_SG_DISABLE");
    let exact_abs = max_abs(&fused_st, &unfused_st);
    let exact_rel = max_rel(&fused_st, &unfused_st);
    eprintln!("[{label}] single-thread: fused-vs-unfused abs={exact_abs:.3e} rel={exact_rel:.3e}");
    if down_scheme == QuantScheme::GgufQ4K {
        // gate/up/down all run the single-thread q4k_mv_f32 accumulation in
        // BOTH paths and the fused kernels inline byte-identical source. The
        // residual difference is pure f32 FMA-contraction freedom between two
        // separately-compiled Metal kernels (the standalone GEMV vs the inlined
        // helper) — machine-epsilon, NOT an algorithmic divergence.
        assert!(
            exact_rel < 1e-5,
            "{label}: fused vs unfused (single-thread, rel) {exact_rel} — \
             expected machine-epsilon agreement"
        );
    } else {
        // Q6_K down: unfused m==1 uses dequant+MPS sgemm (different order), so
        // not bit-exact, but the fused q6k GEMV matches the validated CPU math.
        assert!(
            exact_rel < 5e-3,
            "{label}: fused vs unfused Q6_K (single-thread, rel) {exact_rel}"
        );
    }
}

#[test]
fn metal_fused_decode_mlp_matches_unfused_and_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // gate/up Q4_K, down Q4_K.
    run_case(rlx_gguf::GgmlType::Q4K, QuantScheme::GgufQ4K, "all-Q4_K");
    // gate/up Q4_K, down Q6_K (realistic GGUF Q4_K_M layout).
    run_case(rlx_gguf::GgmlType::Q6K, QuantScheme::GgufQ6K, "Q6_K-down");
    run_q5_combined_gelu_case("Q5_0-combined-gelu-direct-add");
    run_q5_combined_gelu_post_ffn_case("Q5_0-combined-gelu-post-ffn");
}

fn run_q5_combined_gelu_case(label: &str) {
    const H: usize = 640;
    const I: usize = 1024;
    let f = DType::F32;
    let gate_up_p = weight(0.011, H, 2 * I, rlx_gguf::GgmlType::Q5_0);
    let down_p = weight(0.013, I, H, rlx_gguf::GgmlType::Q5_0);

    let mut g = Graph::new("fused_decode_mlp_q5_gelu");
    let r = g.input("r", Shape::new(&[1, 1, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let gate_up_w = g.param("gate_up_w", Shape::new(&[gate_up_p.len()], DType::U8));
    let down_w = g.param("down_w", Shape::new(&[down_p.len()], DType::U8));

    let normed = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![r, gamma, beta],
        Shape::new(&[1, 1, H], f),
    );
    let combined = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ5_0,
        },
        vec![normed, gate_up_w],
        Shape::new(&[1, 1, 2 * I], f),
    );
    let gate = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: I,
        },
        vec![combined],
        Shape::new(&[1, 1, I], f),
    );
    let up = g.add_node(
        Op::Narrow {
            axis: 2,
            start: I,
            len: I,
        },
        vec![combined],
        Shape::new(&[1, 1, I], f),
    );
    let gate_act = g.add_node(
        Op::Activation(Activation::GeluApprox),
        vec![gate],
        Shape::new(&[1, 1, I], f),
    );
    let prod = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_act, up],
        Shape::new(&[1, 1, I], f),
    );
    let down = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ5_0,
        },
        vec![prod, down_w],
        Shape::new(&[1, 1, H], f),
    );
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![r, down],
        Shape::new(&[1, 1, H], f),
    );
    g.set_outputs(vec![out]);

    let r_vec: Vec<f32> = (0..H).map(|i| ((i as f32) * 0.03).sin()).collect();
    let gamma: Vec<f32> = (0..H)
        .map(|i| 1.0 + ((i as f32) * 0.001).cos() * 0.1)
        .collect();
    let beta = vec![0.0f32; H];

    let run = |device: Device| -> Vec<f32> {
        let mut s = Session::new(device).compile(g.clone());
        s.set_param_typed("gate_up_w", &gate_up_p, DType::U8);
        s.set_param_typed("down_w", &down_p, DType::U8);
        s.run(&[("r", &r_vec), ("gamma", &gamma), ("beta", &beta)])
            .remove(0)
    };

    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    let before = rlx_metal::thunk::fused_decode_mlp_blocks();
    let fused = run(Device::Metal);
    let fired = rlx_metal::thunk::fused_decode_mlp_blocks() - before;
    assert_eq!(
        fired, 1,
        "{label}: expected one combined Q5_0 GELU fused block"
    );

    rlx_ir::env::set("RLX_METAL_FUSE_DECODE", "0");
    let unfused = run(Device::Metal);
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");

    let cpu = run(Device::Cpu);
    let fu = max_rel(&fused, &unfused);
    let fc = max_rel(&fused, &cpu);
    eprintln!("[Q5_0-combined-gelu] fused-vs-unfused rel={fu:.3e} fused-vs-cpu rel={fc:.3e}");
    assert!(fu < 5e-3, "fused vs unfused rel {fu}");
    assert!(fc < 5e-2, "{label}: fused vs cpu rel {fc}");
}

fn run_q5_combined_gelu_post_ffn_case(label: &str) {
    const H: usize = 640;
    const I: usize = 1024;
    let f = DType::F32;
    let gate_up_p = weight(0.011, H, 2 * I, rlx_gguf::GgmlType::Q5_0);
    let down_p = weight(0.013, I, H, rlx_gguf::GgmlType::Q5_0);

    let mut g = Graph::new("fused_decode_mlp_q5_gelu_post_ffn");
    let r = g.input("r", Shape::new(&[1, 1, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let post_gamma = g.input("post_gamma", Shape::new(&[H], f));
    let gate_up_w = g.param("gate_up_w", Shape::new(&[gate_up_p.len()], DType::U8));
    let down_w = g.param("down_w", Shape::new(&[down_p.len()], DType::U8));

    let normed = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![r, gamma, beta],
        Shape::new(&[1, 1, H], f),
    );
    let combined = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ5_0,
        },
        vec![normed, gate_up_w],
        Shape::new(&[1, 1, 2 * I], f),
    );
    let gate = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: I,
        },
        vec![combined],
        Shape::new(&[1, 1, I], f),
    );
    let up = g.add_node(
        Op::Narrow {
            axis: 2,
            start: I,
            len: I,
        },
        vec![combined],
        Shape::new(&[1, 1, I], f),
    );
    let gate_act = g.add_node(
        Op::Activation(Activation::GeluApprox),
        vec![gate],
        Shape::new(&[1, 1, I], f),
    );
    let prod = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_act, up],
        Shape::new(&[1, 1, I], f),
    );
    let down = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ5_0,
        },
        vec![prod, down_w],
        Shape::new(&[1, 1, H], f),
    );
    let post_ffn = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![down, post_gamma, beta],
        Shape::new(&[1, 1, H], f),
    );
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![r, post_ffn],
        Shape::new(&[1, 1, H], f),
    );
    g.set_outputs(vec![out]);

    let r_vec: Vec<f32> = (0..H).map(|i| ((i as f32) * 0.03).sin()).collect();
    let gamma: Vec<f32> = (0..H)
        .map(|i| 1.0 + ((i as f32) * 0.001).cos() * 0.1)
        .collect();
    let beta = vec![0.0f32; H];

    let run = |device: Device| -> Vec<f32> {
        let mut s = Session::new(device).compile(g.clone());
        s.set_param_typed("gate_up_w", &gate_up_p, DType::U8);
        s.set_param_typed("down_w", &down_p, DType::U8);
        s.run(&[
            ("r", &r_vec),
            ("gamma", &gamma),
            ("beta", &beta),
            ("post_gamma", &gamma),
        ])
        .remove(0)
    };

    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    let before = rlx_metal::thunk::fused_decode_mlp_blocks();
    let fused = run(Device::Metal);
    let fired = rlx_metal::thunk::fused_decode_mlp_blocks() - before;
    assert_eq!(
        fired, 1,
        "{label}: expected gate_up-only fused block with post_ffn norm in the tail"
    );

    rlx_ir::env::set("RLX_METAL_FUSE_DECODE", "0");
    let unfused = run(Device::Metal);
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");

    let cpu = run(Device::Cpu);
    let fu = max_rel(&fused, &unfused);
    let fc = max_rel(&fused, &cpu);
    eprintln!("[{label}] fused-vs-unfused rel={fu:.3e} fused-vs-cpu rel={fc:.3e}");
    assert!(fu < 5e-3, "{label}: fused vs unfused rel {fu}");
    assert!(fc < 5e-2, "{label}: fused vs cpu rel {fc}");
}
