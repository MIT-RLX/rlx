// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Residual-add + RmsNorm fusion parity (Metal).
//!
//! Safe pattern (add dst dies at rms):
//!   out = rms(x + proj)
//! IR and/or Metal may fuse; result must match the unfused Metal path.
//!
//! Unsafe Qwen/Bonsai post-attn shape:
//!   h = h + attn; n = rms(h); out = h + ffn(n)
//! must NOT fuse (add dst still live for the second residual).

#![cfg(target_os = "macos")]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const H: usize = 2048; // ≥ 1024 so the Metal residual-rms matcher considers it

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_metal(g: Graph, feeds: &[(&str, &[f32])]) -> Vec<f32> {
    let mut s = Session::new(Device::Metal).compile(g);
    s.run(feeds).remove(0)
}

/// `out = rms(residual + proj)` — add dst only feeds rms.
fn build_safe() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("safe_residual_rms");
    let residual = g.input("residual", Shape::new(&[1, H], f));
    let x = g.input("x", Shape::new(&[1, H], f));
    let w = g.input("w", Shape::new(&[H, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let proj = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[1, H], f));
    let summed = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![residual, proj],
        Shape::new(&[1, H], f),
    );
    let out = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![summed, gamma, beta],
        Shape::new(&[1, H], f),
    );
    g.set_outputs(vec![out]);
    g
}

/// Post-attn style: keep summed residual live across rms for a second add.
fn build_unsafe_reuse() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("unsafe_residual_rms");
    let residual = g.input("residual", Shape::new(&[1, H], f));
    let x = g.input("x", Shape::new(&[1, H], f));
    let w_attn = g.input("w_attn", Shape::new(&[H, H], f));
    let ffn = g.input("ffn", Shape::new(&[1, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let attn = g.add_node(Op::MatMul, vec![x, w_attn], Shape::new(&[1, H], f));
    let h = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![residual, attn],
        Shape::new(&[1, H], f),
    );
    let _n = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![h, gamma, beta],
        Shape::new(&[1, H], f),
    );
    // Second residual reads `h` (add dst), not `_n` — fusion must refuse.
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![h, ffn],
        Shape::new(&[1, H], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn fill(seed: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * seed).sin() * 0.25 + 0.01)
        .collect()
}

#[test]
fn residual_rms_safe_matches_unfused() {
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    rlx_ir::env::unset("RLX_METAL_FUSE_RESIDUAL_RMS");
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let residual = fill(0.1, H);
    let x = fill(0.15, H);
    let w = fill(0.2, H * H);
    let gamma = fill(0.3, H);
    let beta = fill(0.4, H);
    let feeds: [(&str, &[f32]); 5] = [
        ("residual", &residual),
        ("x", &x),
        ("w", &w),
        ("gamma", &gamma),
        ("beta", &beta),
    ];

    // Default path may IR-fuse and/or Metal-thunk-fuse.
    let fused = run_metal(build_safe(), &feeds);

    // Force Metal thunk pass off; IR fusion may still apply — also kill IR via
    // comparing against an explicit expand: run with residual-rms metal off is
    // enough when IR already fused both the same way. Prefer CPU reference.
    let mut cpu = Session::new(Device::Cpu).compile(build_safe());
    let cpu_out = cpu.run(&feeds).remove(0);

    let err = max_abs(&fused, &cpu_out);
    assert!(err < 1e-4, "safe residual+rms Metal vs CPU max_abs={err}");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

#[test]
fn residual_rms_live_reuse_does_not_fuse_or_diverge() {
    rlx_ir::env::unset("RLX_METAL_FUSE_DECODE");
    rlx_ir::env::unset("RLX_METAL_FUSE_RESIDUAL_RMS");
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let residual = fill(0.11, H);
    let x = fill(0.16, H);
    let w_attn = fill(0.22, H * H);
    let ffn = fill(0.33, H);
    let gamma = fill(0.44, H);
    let beta = fill(0.55, H);
    let feeds: [(&str, &[f32]); 6] = [
        ("residual", &residual),
        ("x", &x),
        ("w_attn", &w_attn),
        ("ffn", &ffn),
        ("gamma", &gamma),
        ("beta", &beta),
    ];

    let before = rlx_metal::thunk::fused_residual_rms_blocks();
    let with_pass = run_metal(build_unsafe_reuse(), &feeds);
    let fused_n = rlx_metal::thunk::fused_residual_rms_blocks() - before;
    assert_eq!(
        fused_n, 0,
        "post-attn reuse must not Metal-fuse residual+rms (fused_n={fused_n})"
    );

    let mut cpu = Session::new(Device::Cpu).compile(build_unsafe_reuse());
    let cpu_out = cpu.run(&feeds).remove(0);
    let err = max_abs(&with_pass, &cpu_out);
    assert!(
        err < 1e-4,
        "unsafe-reuse residual path diverged from CPU max_abs={err}"
    );

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}
