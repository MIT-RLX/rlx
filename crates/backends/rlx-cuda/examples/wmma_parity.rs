// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Correctness check for the WMMA attention variants (d64 + d128, None +
//! Causal) against the CPU-validated scalar flash kernel — the scalar kernel is
//! the reference. Switches `RLX_CUDA_ATTENTION` between runs via
//! `reload_runtime_config`. Covers the paths the bshd unit test doesn't
//! (head_dim=128 and causal masking).
//!
//! cargo run --release -p rlx-cuda --example wmma_parity

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};

fn run_variant(
    exe: &mut CudaExecutable,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    variant: &str,
) -> Vec<f32> {
    // SAFETY: single-threaded example; set the policy then refresh the config.
    unsafe {
        std::env::set_var("RLX_CUDA_ATTENTION", variant);
    }
    rlx_cuda::reload_runtime_config();
    exe.run(&[("q", q), ("k", k), ("v", v)])
        .into_iter()
        .next()
        .unwrap()
}

fn check(b: usize, s: usize, h: usize, d: usize, mask: MaskKind, name: &str) -> bool {
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[b, s, h, d], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, h, d], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, h, d], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: mask,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, h, d], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    let n = b * s * h * d;
    let qv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
    let kv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 0.3).collect();
    let vv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) % 1.0 - 0.5).collect();

    let sca = run_variant(&mut exe, &qv, &kv, &vv, "scalar");
    let wm = run_variant(&mut exe, &qv, &kv, &vv, "wmma");
    let err = sca
        .iter()
        .zip(&wm)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ok = err < 1e-2;
    println!(
        "  d={d:<3} mask={name:<6} max|wmma-scalar|={err:.3e}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available — exiting.");
        return;
    }
    println!("rlx-cuda WMMA attention parity (vs scalar reference)");
    println!("----------------------------------------------------");
    let mut all = true;
    for &d in &[64usize, 128] {
        for &(mask, name) in &[(MaskKind::None, "none"), (MaskKind::Causal, "causal")] {
            all &= check(1, 96, 4, d, mask, name);
        }
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAILED" });
    std::process::exit(if all { 0 } else { 1 });
}
