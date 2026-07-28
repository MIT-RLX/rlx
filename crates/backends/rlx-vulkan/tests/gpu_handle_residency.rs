// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU-resident input handle (`bind_gpu_handle` + `set_gpu_handle_feed`) tests
//! for the native Vulkan backend. This is the KV-cache residency primitive: an
//! input slot is uploaded once, an output is folded back into it in-place after
//! each run (no host round-trip), and logits-only readback leaves it resident.
//!
//! Skips gracefully on hosts with no Vulkan device (CI / macOS without MoltenVK).

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_vulkan::backend::VulkanExecutable;

fn approx(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

fn s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// `acc += delta` driven entirely on-device: `acc` is a resident handle fed by
/// the graph output; only `delta` is uploaded per step and only output 0 is read
/// back. If residency were broken (acc re-seeded from a stale/empty host mirror)
/// the accumulator would not advance and the asserts would fail.
#[test]
fn resident_handle_accumulates_across_runs() {
    if !rlx_vulkan::is_available() {
        eprintln!("rlx-vulkan: no device — skipping residency test");
        return;
    }
    let rows = 4usize;
    let cols = 2usize;
    let n = rows * cols;

    let mut g = Graph::new("acc");
    let acc = g.input("acc", s(&[rows, cols]));
    let delta = g.input("delta", s(&[rows, cols]));
    let next = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![acc, delta],
        s(&[rows, cols]),
    );
    g.set_outputs(vec![next]);

    let mut exe = VulkanExecutable::compile(g);

    // Bind acc once (zeros) and wire output 0 → acc.
    assert!(exe.bind_gpu_handle("acc", &vec![0.0f32; n]), "bind acc");
    assert!(exe.has_gpu_handle("acc"));
    exe.set_gpu_handle_feed("acc", 0);

    let ones = vec![1.0f32; n];
    let steps = 5usize;
    for k in 0..steps {
        // Only `delta` is uploaded; only output 0 is read back (logits-only).
        let outs = exe.run_read_outputs(&[("delta", &ones)], Some(&[0]));
        assert_eq!(outs.len(), 1);
        let expected = (k + 1) as f32;
        assert!(
            outs[0].iter().all(|&v| (v - expected).abs() < 1e-6),
            "step {k}: expected all {expected}, got {:?}",
            outs[0]
        );
    }

    // The resident handle reflects the full accumulation, read back on demand.
    let acc_final = exe.read_gpu_handle("acc").expect("acc handle readable");
    assert!(
        acc_final.iter().all(|&v| (v - steps as f32).abs() < 1e-6),
        "final acc should be all {steps}, got {acc_final:?}"
    );
    eprintln!(
        "[rlx-vulkan] residency OK on {:?}: acc accumulated to {} over {steps} resident steps",
        rlx_vulkan::device_name(),
        steps
    );
}

/// Re-binding a resident handle re-seeds it from host (bucket-reinstall path):
/// after a fresh `bind_gpu_handle`, the next run must start from the new seed,
/// not the previously accumulated resident value.
#[test]
fn rebind_reseeds_resident_handle() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let n = 4usize;
    let mut g = Graph::new("acc2");
    let acc = g.input("acc", s(&[n]));
    let delta = g.input("delta", s(&[n]));
    let next = g.add_node(Op::Binary(BinaryOp::Add), vec![acc, delta], s(&[n]));
    g.set_outputs(vec![next]);
    let mut exe = VulkanExecutable::compile(g);

    exe.bind_gpu_handle("acc", &vec![0.0f32; n]);
    exe.set_gpu_handle_feed("acc", 0);
    let ones = vec![1.0f32; n];
    for _ in 0..3 {
        exe.run_read_outputs(&[("delta", &ones)], Some(&[0]));
    }
    let acc_after = exe.read_gpu_handle("acc").unwrap();
    assert!(acc_after.iter().all(|&v| (v - 3.0).abs() < 1e-6));

    // Re-seed to 10.0 and step once → 11.0 (not 4.0 from the old resident path).
    exe.bind_gpu_handle("acc", &vec![10.0f32; n]);
    let outs = exe.run_read_outputs(&[("delta", &ones)], Some(&[0]));
    assert!(
        outs[0].iter().all(|&v| (v - 11.0).abs() < 1e-6),
        "rebind should re-seed: expected 11.0, got {:?}",
        outs[0]
    );
}

/// Targeted row feed (`register_kv_row_feed` + `feed_kv_row`): the llama32-style
/// decode KV append where the graph emits the new token at the LAST row of a
/// bucket-padded output (`concat(past, tok)`), and we fold that row into the
/// resident `past` slot at the active position — leaving the prefix untouched.
#[test]
fn row_feed_appends_new_token_into_resident_slot() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let upper = 4usize; // bucket rows in the resident `past` slot
    let cols = 2usize;

    let mut g = Graph::new("row_feed");
    let past = g.input("past", s(&[upper, cols]));
    let tok = g.input("tok", s(&[1, cols]));
    // out = concat(past, tok) along rows → [upper+1, cols]; new token at row `upper`.
    let out = g.add_node(
        Op::Concat { axis: 0 },
        vec![past, tok],
        s(&[upper + 1, cols]),
    );
    g.set_outputs(vec![out]);
    let mut exe = VulkanExecutable::compile(g);

    exe.bind_gpu_handle("past", &vec![0.0f32; upper * cols]);
    exe.register_kv_row_feed("past", 0);

    // Append rows (1,1),(2,2),(3,3),(4,4) at positions 0..4.
    for k in 0..upper {
        let v = (k + 1) as f32;
        let outs = exe.run_read_outputs(&[("tok", &[v, v])], Some(&[0]));
        // Output row `upper` (last) must be the token we just fed in.
        let last_row = &outs[0][upper * cols..(upper + 1) * cols];
        assert!(
            approx(last_row, &[v, v]),
            "step {k}: out last row {last_row:?}"
        );
        // Fold new token (output row `upper`) into resident `past` row `k`.
        exe.feed_kv_row(upper, k, cols);
    }

    let past_final = exe.read_gpu_handle("past").expect("past readable");
    assert!(
        approx(&past_final, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        "resident past after row feeds: {past_final:?}"
    );
    eprintln!(
        "[rlx-vulkan] row-feed OK on {:?}: {past_final:?}",
        rlx_vulkan::device_name()
    );
}
