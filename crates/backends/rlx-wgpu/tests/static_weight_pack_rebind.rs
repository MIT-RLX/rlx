// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A re-bound param must reach a static weight pack.**
//!
//! `lower.rs` marks every node whose value is fixed after param upload — a
//! `Concat`/`Cast`/`Expand` over `Param`s, i.e. the fused QKV and gate+up packs
//! the matmul-fusion passes build — as `static_once`, and `run.rs` skips those
//! steps from the second `run()` onward. That is a large win: on a Llama decode
//! the fused-weight packs are roughly half of all DRAM traffic, and recomputing
//! them per token is pure waste.
//!
//! It is only correct while the params underneath do not change. Nothing in
//! `set_param` invalidated the flag, so the sequence
//!
//! ```text
//! run()                    // pack materialised, static_once_done = true
//! set_param("w", new)      // param storage updated
//! run()                    // pack SKIPPED — matmul still reads the old pack
//! ```
//!
//! silently computed with the previous weights. That is the shape of every
//! weight-swap workload: a training step, a LoRA merge, quantisation re-binding,
//! or a sweep harness that reuses one executable across weight sets. It cannot
//! be caught by a parity test that binds once, which is why it survived.
//!
//! The graph here is the minimal form of the real pattern: a matmul against
//! `concat([w_a, w_b], axis=1)`, which is exactly what `shared_input_matmul`
//! and `swiglu_dual` emit.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// `x @ concat([w_a, w_b], axis=1)` — the fused-projection shape.
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

#[test]
fn rebinding_a_param_updates_a_static_weight_pack() {
    if device_missing() {
        return;
    }
    let mut exe = WgpuExecutable::compile(packed_matmul_graph());

    let ident = [1.0f32, 0.0, 0.0, 1.0];
    exe.set_param("w_a", &ident);
    exe.set_param("w_b", &ident);
    let x = [1.0f32, 2.0];

    // Run 1 materialises the pack. `[1,2] @ [[1,0,1,0],[0,1,0,1]]`.
    let first = exe.run(&[("x", &x)])[0].clone();
    assert_eq!(first, vec![1.0, 2.0, 1.0, 2.0], "run 1 is the baseline");

    // Re-bind the FIRST half of the pack. Doubling `w_a` must double the first
    // two outputs; `w_b` is untouched so the last two must not move.
    exe.set_param("w_a", &[2.0, 0.0, 0.0, 2.0]);
    let second = exe.run(&[("x", &x)])[0].clone();

    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` did not reach the fused pack — the matmul is still \
         reading run 1's concat. Got {second:?}"
    );
}

/// The same hazard through `set_param_bytes`, which takes a different path into
/// arena storage and so needs its own invalidation.
#[test]
fn rebinding_via_set_param_bytes_also_updates_the_pack() {
    if device_missing() {
        return;
    }
    let mut exe = WgpuExecutable::compile(packed_matmul_graph());
    let ident = [1.0f32, 0.0, 0.0, 1.0];
    exe.set_param("w_a", &ident);
    exe.set_param("w_b", &ident);
    let x = [1.0f32, 2.0];
    let _ = exe.run(&[("x", &x)]);

    let doubled: Vec<u8> = [2.0f32, 0.0, 0.0, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    exe.set_param_bytes("w_a", &doubled);
    let second = exe.run(&[("x", &x)])[0].clone();

    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` (bytes) did not reach the fused pack. Got {second:?}"
    );
}

/// **Is the skip even armed?**
///
/// The re-bind tests above pass either because invalidation works or because
/// the pack is recomputed every run anyway — they cannot tell the two apart,
/// and a guard that passes for the wrong reason is worse than none. This
/// separates them: lowering marks the pack's steps, so if the flag never arms
/// the optimisation is dead and every `run()` rebuilds a constant.
#[test]
fn the_static_weight_skip_arms_after_the_first_run() {
    if device_missing() {
        return;
    }
    let mut exe = WgpuExecutable::compile(packed_matmul_graph());
    let ident = [1.0f32, 0.0, 0.0, 1.0];
    exe.set_param("w_a", &ident);
    exe.set_param("w_b", &ident);
    let x = [1.0f32, 2.0];

    let (marked, armed) = exe.static_once_report();
    assert!(
        marked > 0,
        "lowering marked no static-weight steps for `concat(param, param)` — \
         the pack is not even a candidate"
    );
    assert!(!armed, "nothing has run yet");

    let _ = exe.run(&[("x", &x)]);
    let (_, armed_after_one) = exe.static_once_report();
    let _ = exe.run(&[("x", &x)]);
    let (_, armed_after_two) = exe.static_once_report();

    assert!(
        armed_after_one || armed_after_two,
        "{marked} steps were marked static-once but the skip never armed across \
         two runs — the fused weight pack is rebuilt on every run, which on a \
         decode graph is roughly half the DRAM traffic"
    );
}

/// The same thing at a real model's dimensions, to size what the skip is worth.
///
/// Carbon-500M geometry: hidden 1024, 16 heads / 8 KV heads (so q=1024,
/// k=v=512), d_ff 3072. The matmul-fusion passes turn that into two weight
/// packs per layer — `concat([q,k,v]) -> [1024,2048]` and
/// `concat([gate,up]) -> [1024,6144]` — which is 8.4 MB + 25.2 MB of f32 per
/// layer, read AND written, every run they are rebuilt.
///
/// Metal measures the whole 28-layer version at 1879 MB of the 3941 MB a decode
/// step moves (`RLX_METAL_DUMP_BYTES`). This asserts wgpu marks and skips the
/// same packs rather than rebuilding them per token.
#[test]
fn the_skip_covers_the_weight_packs_at_model_scale() {
    if device_missing() {
        return;
    }
    const H: usize = 1024;
    const KV: usize = 512;
    const FF: usize = 3072;

    let mut g = Graph::new("layer");
    let x = g.input("x", Shape::new(&[1, H], DType::F32));
    let q = g.param("q", Shape::new(&[H, H], DType::F32));
    let k = g.param("k", Shape::new(&[H, KV], DType::F32));
    let v = g.param("v", Shape::new(&[H, KV], DType::F32));
    let gate = g.param("gate", Shape::new(&[H, FF], DType::F32));
    let up = g.param("up", Shape::new(&[H, FF], DType::F32));

    let qkv = g.concat_(vec![q, k, v], 1);
    let a = g.matmul(x, qkv, Shape::new(&[1, H + 2 * KV], DType::F32));
    let gu = g.concat_(vec![gate, up], 1);
    let b = g.matmul(x, gu, Shape::new(&[1, 2 * FF], DType::F32));
    let head = g.narrow_(b, 1, 0, H + 2 * KV);
    let out = g.add(a, head);
    g.set_outputs(vec![out]);

    let mut exe = WgpuExecutable::compile(g);
    for (n, len) in [
        ("q", H * H),
        ("k", H * KV),
        ("v", H * KV),
        ("gate", H * FF),
        ("up", H * FF),
    ] {
        exe.set_param(n, &vec![0.001f32; len]);
    }
    let xs = vec![1.0f32; H];

    let (marked, _) = exe.static_once_report();
    assert!(marked > 0, "neither weight pack was marked static-once");

    let first = exe.run(&[("x", &xs)])[0].clone();
    let (_, armed) = exe.static_once_report();
    assert!(armed, "{marked} steps marked but the skip did not arm");

    // Skipping must not change the answer.
    let second = exe.run(&[("x", &xs)])[0].clone();
    assert_eq!(
        first, second,
        "output changed once the weight packs started being skipped"
    );
    eprintln!("model-scale packs: {marked} steps skipped from run 2 onward");
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
    rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available())
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
        let mut exe = WgpuExecutable::compile(g);
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
