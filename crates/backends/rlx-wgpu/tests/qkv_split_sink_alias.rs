// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-QKV elision must refuse when the Q/K/V sinks alias **each other**.
//!
//! `compile_static_inner` folds `FusedMatMulBiasAct → Narrow×3` into one
//! `matmul_qkv` dispatch that routes each output column straight into the
//! narrows' slots, skipping the narrows. Its slot-safety guard (`live_across`)
//! checks every *other* tensor live across the matmul's step — and excludes q, k
//! and v from the scan by construction, so an aliasing *among the three sinks*
//! read as safe.
//!
//! They alias routinely. A `qkv_heads` helper emits
//! `narrow(q) → reshape → transpose → narrow(k) → reshape → transpose →
//! narrow(v) → …`, so each narrow dies at its own transpose one step before the
//! next narrow is born, and the planner — correctly for the unfused order —
//! hands all three the same offset. Fused, one dispatch writes Q, K and V into
//! those same bytes at once and all three heads come back holding a
//! race-dependent mix of one projection.
//!
//! Field repro: `rlx-axon` (`MannasAI/axon-eeg`, 118.6 M) at `block0_attn`
//! cos 0.605 / 0.614 / 0.651 / 0.726 / 0.804 — varying run to run, with the
//! three narrows all at arena offset 278969856 and Q and K returning an
//! identical max. Exact under `RLX_ARENA_NO_REUSE=1`, which is what made it look
//! like an arena bug for a while; the plan was valid throughout, and
//! `RLX_WGPU_HAZARD_REPORT=1` agreed (66 write-after-read hazards, 0 of them not
//! view-explained). `RLX_WGPU_ONE_OP_PER_PASS=1` does NOT fix it, so it is not
//! the intra-pass barrier race wgpu has shown elsewhere.
//!
//! The graph below is the triggering shape, which is just an attention block's
//! QKV split. `sinks_share_a_slot` asserts the aliasing the test needs actually
//! happens, so a future planner change cannot quietly make this pass by
//! separating the slots.
//!
//! **The other post-planning folds here were audited and are not this bug.**
//! `packed_bshd_attn` *skips* the narrows and reads Q/K/V from the packed parent
//! with a stride, so it never writes their slots at all — no write moves, and
//! `packed_bshd_narrow_elidable` still gates each elision. The
//! `FusedResidualLnTee` fold does move reads (to the norm's step) and carries its
//! own arena-safety veto plus `tests/residual_ln_tee_arena_safety.rs`; its field
//! repro was `rlx-eegdm` at cos 0.994. `qkv_split` was the one that wrote into
//! slots it had not checked against each other. The sibling bug in `rlx-metal`
//! is `fuse_narrow_clusters` — see the hazard note on that function and
//! `rlx-runtime/tests/arena_reuse_narrow_chain.rs`.

use rlx_ir::infer::GraphExt as _;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn vals(n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 31 + k * 17) % 97) as f32) - 48.0) * 0.021)
        .collect()
}

/// `x[n,L,d] → (matmul + bias) → [n,L,3d] → narrow×3 → heads → attention`.
fn build(n: usize, l: usize, d: usize, heads: usize) -> Graph {
    let hd = d / heads;
    let mut g = Graph::new("qkv_split");
    let x = g.input("x", Shape::new(&[n, l, d], F));
    let w = g.input("w", Shape::new(&[d, 3 * d], F));
    let b = g.input("b", Shape::new(&[3 * d], F));
    let proj = g.mm(x, w);
    let qkv = g.add(proj, b); // the matmul+bias fold makes this one FMB step
    let mut parts: Vec<NodeId> = Vec::with_capacity(3);
    for i in 0..3 {
        // Exactly `qkv_heads`: narrow, reshape, transpose — so this narrow is
        // dead at its transpose, one step before the next narrow is born.
        let part = g.narrow_(qkv, 2, i * d, d);
        let part = g.reshape_(part, vec![n as i64, l as i64, heads as i64, hd as i64]);
        parts.push(g.transpose_(part, vec![0, 2, 1, 3]));
    }
    let (q, k, v) = (parts[0], parts[1], parts[2]);
    let kt = g.transpose_(k, vec![0, 1, 3, 2]);
    let scores = g.mm(q, kt);
    let scale = g.constant(1.0 / (hd as f64).sqrt(), F);
    let scores = g.mul(scores, scale);
    let out = g.mm(scores, v); // no softmax: keep the graph about the split
    let out = g.transpose_(out, vec![0, 2, 1, 3]);
    let out = g.reshape_(out, vec![n as i64, l as i64, d as i64]);
    g.set_outputs(vec![out]);
    g
}

fn check(n: usize, l: usize, d: usize, heads: usize) {
    let g = build(n, l, d, heads);
    let (xv, wv, bv) = (vals(n * l * d, 1), vals(d * 3 * d, 2), vals(3 * d, 3));
    let feed = [
        ("x", xv.as_slice()),
        ("w", wv.as_slice()),
        ("b", bv.as_slice()),
    ];
    let want = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&feed)
        .remove(0);
    let scale = want.iter().fold(0f32, |m, &q| m.max(q.abs()));
    assert!(
        scale > 1e-3,
        "{n}x{l}x{d}: reference is all but zero (max {scale:e}) — the test would pass on anything"
    );
    // Run more than once: the fused write is a race, so one clean run proves
    // nothing. Before the fix this failed on run 1 anyway, with a different
    // cosine each time.
    let mut exe = Session::new(Device::Gpu).compile(g);
    for attempt in 0..3 {
        let got = exe.run(&feed).remove(0);
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 1e-4 * scale.max(1.0),
            "{n}x{l}x{d} heads={heads} attempt {attempt}: max |wgpu - cpu| = {worst:e} (scale {scale:e})"
        );
    }
}

#[test]
fn qkv_split_with_aliased_sinks_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    for (n, l, d, heads) in [(2, 16, 64, 4), (1, 32, 128, 8), (4, 8, 96, 6)] {
        check(n, l, d, heads);
    }
}

/// Non-vacuity: the three narrows really do land on one arena offset, which is
/// the whole premise. If a planner change separates them this test still passes
/// but stops covering anything, so say so out loud.
#[test]
fn sinks_share_a_slot() {
    let g = build(2, 16, 64, 4);
    // The exact planner the backend uses (`buffer::plan_f32_uniform`).
    let plan = rlx_compile::memory::plan_memory_f32_uniform(&g, 16);
    let narrows: Vec<NodeId> = g
        .nodes()
        .iter()
        .filter(|nd| matches!(nd.op, rlx_ir::Op::Narrow { axis: 2, .. }))
        .map(|nd| nd.id)
        .collect();
    assert_eq!(narrows.len(), 3, "expected the three qkv narrows");
    let offs: Vec<usize> = narrows
        .iter()
        .map(|id| plan.assignments[id].offset)
        .collect();
    assert!(
        offs[0] == offs[1] && offs[1] == offs[2],
        "the qkv narrows no longer share a slot ({offs:?}) — this file's premise \
         is gone and `qkv_split_with_aliased_sinks_matches_cpu` now covers nothing"
    );
}
