// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A re-bound param must reach a static weight pack.**
//!
//! The matmul-fusion passes fuse Q/K/V and gate/up into one GEMM by emitting a
//! `Concat` over the weight `Param`s. That concat is invariant across `run()`s,
//! and Vulkan re-records them into its command buffer every run — ~1.9 GB of
//! constants per token on a 28-layer Llama. Metal measures the same packs at
//! 47.7% of a decode step's DRAM traffic.
//!
//! Vulkan batches contiguous GPU steps into one command buffer, so the skip
//! filters indices out of that batch rather than skipping a per-step loop —
//! which is why it needs its own coverage rather than inheriting CUDA's.
//!
//! Those steps are now materialised once and skipped after. Skipping is only
//! correct while the params underneath hold still: without invalidation,
//! `run(); set_param(w, ..); run()` keeps the first run's pack and the consuming
//! GEMM silently uses the old weights. A parity test that binds once cannot see
//! that, which is why this binds twice.
//!
//! Requires a CUDA device; skips (loudly) without one. Deliberately tiny — it
//! must run on a rig whose GPU is mostly occupied by another job.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_vulkan::backend::VulkanExecutable;

/// `x @ concat([w_a, w_b], axis=1)` — the fused-projection shape, minimally.
fn packed_matmul_graph() -> Graph {
    let mut g = Graph::new("packed");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w_a = g.param("w_a", Shape::new(&[2, 2], DType::F32));
    let w_b = g.param("w_b", Shape::new(&[2, 2], DType::F32));
    let w = g.concat_(vec![w_a, w_b], 1);
    let y = g.matmul(x, w, Shape::new(&[1, 4], DType::F32));
    g.set_outputs(vec![y]);
    g
}

const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const X: [f32; 2] = [1.0, 2.0];

#[test]
fn rebinding_a_param_updates_a_static_weight_pack() {
    if device_missing() {
        return;
    }
    let mut exe = VulkanExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);

    let first = exe.run(&[("x", &X)])[0].clone();
    assert_eq!(first, vec![1.0, 2.0, 1.0, 2.0], "run 1 is the baseline");

    // Doubling `w_a` must double the first two outputs; `w_b` is untouched so
    // the last two must not move.
    exe.set_param("w_a", &[2.0, 0.0, 0.0, 2.0]);
    let second = exe.run(&[("x", &X)])[0].clone();
    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` did not reach the fused pack — the matmul is still \
         reading run 1's concat. Got {second:?}"
    );
}

/// `set_param_bytes` reaches device storage by a different route, so it needs
/// its own invalidation.
#[test]
fn rebinding_via_set_param_bytes_also_updates_the_pack() {
    if device_missing() {
        return;
    }
    let mut exe = VulkanExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);
    let _ = exe.run(&[("x", &X)]);

    let doubled: Vec<u8> = [2.0f32, 0.0, 0.0, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    exe.set_param_bytes("w_a", &doubled);
    let second = exe.run(&[("x", &X)])[0].clone();
    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` (bytes) did not reach the fused pack. Got {second:?}"
    );
}

/// Skipping the pack must not change the answer across repeated runs — the
/// direct check that the skip reads a live slot rather than one the planner
/// let an activation reuse.
#[test]
fn repeated_runs_agree_once_the_pack_is_skipped() {
    if device_missing() {
        return;
    }
    let mut exe = VulkanExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &[3.0, 0.0, 0.0, 3.0]);

    let first = exe.run(&[("x", &X)])[0].clone();
    for i in 1..5 {
        let again = exe.run(&[("x", &X)])[0].clone();
        assert_eq!(
            first, again,
            "run {i} disagrees with run 0 after the weight pack started being \
             skipped — the pack's arena slot is not exclusively owned"
        );
    }
    assert_eq!(first, vec![1.0, 2.0, 3.0, 6.0]);
}

/// **Is the skip even armed?**
///
/// The guards above pass either because invalidation works or because the pack
/// is recomputed every run anyway — they cannot tell those apart. `rlx-wgpu`
/// sat in the second state, with green re-bind tests and a dead optimisation,
/// which is why this check exists separately.
#[test]
fn the_static_weight_skip_arms_after_the_first_run() {
    if device_missing() {
        return;
    }
    let mut exe = VulkanExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);

    let (marked, armed) = exe.static_once_report();
    assert!(
        marked > 0,
        "lowering marked no static-weight steps for `concat(param, param)` — \
         the pack is not even a candidate for skipping"
    );
    assert!(!armed, "nothing has run yet");

    let _ = exe.run(&[("x", &X)]);
    let (_, armed_after) = exe.static_once_report();
    assert!(
        armed_after,
        "{marked} steps were marked static-once but the skip never armed — the \
         fused weight pack is re-launched on every run (5 launches/layer, 140 \
         per token on a 28-layer Llama)"
    );

    // And a param write must disarm it again.
    exe.set_param("w_a", &IDENT);
    let (_, after_rebind) = exe.static_once_report();
    assert!(
        !after_rebind,
        "a param write left the skip armed — the pack is now stale"
    );
}

/// Skip when there is no device — but LOUDLY, and refusably.
///
/// `if !is_available() { return; }` makes a test that never ran report `ok`,
/// which is indistinguishable from a real pass in the summary line. That is not
/// hypothetical: these very tests were reported as "hardware-validated on the
/// MI100" when every one of them had printed `skipping` and passed vacuously,
/// because `rlx_rocm::is_available()` is false over a plain `ssh` shell.
///
/// Set `RLX_REQUIRE_DEVICE=1` on a rig run and a missing device becomes a
/// failure instead of a silent pass.
fn device_missing() -> bool {
    // Was a hand-rolled copy of this assert — six such copies existed in
    // tree, and one had lost the RLX_REQUIRE_DEVICE check entirely.
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

/// **Does the skip still fire on a DEEP graph?**
///
/// A one-layer graph is not a test of this. On CUDA the packs were being
/// liveness-reused and the exclusivity check correctly refused to skip them —
/// 0 of 140 pack steps qualified on 28 layers — yet a single-layer graph marked
/// fine, so the guard tests passed while the optimisation was dead. The
/// difference is whether the planner pins packs to graph end; this asserts the
/// count actually scales with depth instead of collapsing.
#[test]
fn the_skip_scales_with_graph_depth() {
    if device_missing() {
        return;
    }
    const H: usize = 256;
    const FF: usize = 512;
    let build = |layers: usize| {
        let mut g = Graph::new("deep");
        let x = g.input("x", Shape::new(&[1, H], DType::F32));
        let mut cur = x;
        let mut names = Vec::new();
        for l in 0..layers {
            let (a, b) = (format!("a{l}"), format!("b{l}"));
            let pa = g.param(&a, Shape::new(&[H, FF], DType::F32));
            let pb = g.param(&b, Shape::new(&[H, FF], DType::F32));
            names.push((a, H * FF));
            names.push((b, H * FF));
            let pack = g.concat_(vec![pa, pb], 1);
            let m = g.matmul(cur, pack, Shape::new(&[1, 2 * FF], DType::F32));
            let n = g.narrow_(m, 1, 0, H);
            cur = g.add(cur, n);
        }
        g.set_outputs(vec![cur]);
        (g, names)
    };

    let mut counts = Vec::new();
    for layers in [1usize, 2, 8] {
        let (g, names) = build(layers);
        let mut exe = VulkanExecutable::compile(g);
        for (n, len) in &names {
            exe.set_param(n, &vec![0.001f32; *len]);
        }
        let xs = vec![1.0f32; H];
        let first = exe.run(&[("x", &xs)])[0].clone();
        let (marked, armed) = exe.static_once_report();
        let second = exe.run(&[("x", &xs)])[0].clone();
        assert_eq!(
            first, second,
            "{layers} layers: output changed once packs were skipped"
        );
        eprintln!("  {layers} layers: {marked} steps marked, armed={armed}");
        counts.push((layers, marked));
    }
    let (_, one) = counts[0];
    assert!(one > 0, "even one layer marked nothing");
    for &(layers, marked) in &counts[1..] {
        assert!(
            marked >= one * layers,
            "{layers} layers marked only {marked} steps but one layer marked {one} — \
             packs are being liveness-reused at depth, so the skip is dead on real \
             models even though shallow guards pass"
        );
    }
}
