// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The residual→norm tee must fire on transformer blocks, and must not change
//! the numbers.**
//!
//! `FuseResidualLN` / `FuseResidualRmsNorm` both require the residual sum to
//! have exactly one consumer. A transformer block gives it two — the norm and
//! the next residual (`h += attn; n = norm(h); h += ffn(n)`) — so the fusion
//! declines on every layer. `detect_residual_ln_tee_pattern` recovers that case
//! by emitting one step that writes the sum AND the normalised result.
//!
//! It was LayerNorm-only, which misses every Llama-class model. Widening it to
//! RmsNorm means both norms now share one kernel that selects its scaling on an
//! `is_rms` flag — and a mode flag on a shared kernel is exactly where a silent
//! numeric divergence hides. So this asserts both halves:
//!
//! * the tee actually fires (a dispatch-count fact, machine-independent), and
//! * wgpu still matches the CPU reference elementwise.
//!
//! Without the second check the first would happily pass on a kernel that
//! computes LayerNorm for an RmsNorm graph.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
const H: usize = 128;
const BLOCKS: usize = 6;

/// `BLOCKS` residual blocks whose sum feeds BOTH the norm and the next
/// residual — the shape the single-consumer guard rejects.
fn block_graph(rms: bool) -> (Graph, Vec<(String, usize)>) {
    let mut g = Graph::new("resid_norm");
    let x = g.input("x", Shape::new(&[1, H], F));
    let mut cur = x;
    let mut names = Vec::new();
    for l in 0..BLOCKS {
        let (gn, bn, wn) = (format!("g{l}"), format!("b{l}"), format!("w{l}"));
        let gam = g.param(&gn, Shape::new(&[H], F));
        let bet = g.param(&bn, Shape::new(&[H], F));
        let w = g.param(&wn, Shape::new(&[H, H], F));
        names.push((gn, H));
        names.push((bn, H));
        names.push((wn, H * H));
        let n = if rms {
            g.rms_norm(cur, gam, bet, 1e-5)
        } else {
            g.layer_norm(cur, gam, bet, -1, 1e-5, Shape::new(&[1, H], F))
        };
        let proj = g.matmul(n, w, Shape::new(&[1, H], F));
        cur = g.add(cur, proj);
    }
    g.set_outputs(vec![cur]);
    (g, names)
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.5
        })
        .collect()
}

fn run_cpu(rms: bool, x: &[f32]) -> Vec<f32> {
    let (g, names) = block_graph(rms);
    let mut c = Session::new(Device::Cpu).compile(g);
    for (n, len) in &names {
        c.set_param(n, &fill(*len, n.len() as u64 + 7));
    }
    c.finalize_params();
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn the_tee_fires_for_both_norms_and_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let x = fill(H, 4);

    for rms in [false, true] {
        let label = if rms { "RmsNorm" } else { "LayerNorm" };
        let (g, names) = block_graph(rms);
        let mut exe = WgpuExecutable::compile(g);
        for (n, len) in &names {
            exe.set_param(n, &fill(*len, n.len() as u64 + 7));
        }
        let got = exe.run(&[("x", &x)])[0].clone();

        // Numerics first — it applies on every adapter, however the ops routed.
        let want = run_cpu(rms, &x);
        let maxd = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            maxd < 1e-4,
            "{label}: wgpu differs from CPU by {maxd:.3e} — the tee kernel is \
             computing the wrong norm (check the `is_rms` selector)"
        );

        let counts = exe.step_kind_counts();
        let tees = counts.get("fused_residual_ln_tee").copied().unwrap_or(0);
        let binaries = counts.get("binary").copied().unwrap_or(0);
        let norms = counts.get("layer_norm").copied().unwrap_or(0);
        eprintln!("  {label}: {tees} tee, {binaries} binary, {norms} norm");

        // The tee is a GPU-path optimisation. Discrete Vulkan/DX12 route
        // elementwise ops and norms to the host entirely
        // (`wgpu_prefer_host_fallback`), so there is no GPU step to fuse and
        // ZERO of each is the correct outcome — not a regression. Asserting a
        // tee count unconditionally fails there for the wrong reason.
        if tees == 0 && binaries == 0 && norms == 0 {
            eprintln!("  {label}: ops are host-routed on this adapter — tee N/A");
            continue;
        }
        assert!(
            tees >= BLOCKS - 1,
            "{label}: only {tees} of {BLOCKS} blocks used the tee ({binaries} \
             binary + {norms} norm steps remain) — the multi-consumer residual \
             pattern is not being recognised, so every block pays a separate \
             Add dispatch"
        );
    }
}
