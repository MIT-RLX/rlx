// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transpose + summed-`Narrow` chain over a *computed* tensor, run with and
//! without arena slot reuse.
//!
//! **Status: SOLVED, and the arena was innocent.** This ladder was built to
//! reproduce what looked like a GPU slot-reuse miscompilation; it never did,
//! because there was no slot-reuse bug. The cause was
//! `fuse_narrow_clusters` in `rlx-metal/src/thunk/mod.rs` **hoisting writes**,
//! and the ladder is kept as the control that rules the shape itself out.
//!
//! ## What the bug actually was
//!
//! DIVER-1 (`braindecode/DIVER-1-0.1s-tiny`) on Metal: 10 of 11 stages cos
//! 1.000000, `stcpe_avg` at cos 0.929 — an overlap-average built from a rank-5
//! `Transpose` and 70 nested `Narrow`s, while the transpose's own input was
//! still bit-clean. Exact under `RLX_ARENA_NO_REUSE=1`, which is what sent this
//! after the arena.
//!
//! `fuse_narrow_clusters` merges `Narrow` thunks that share a source and tile
//! the axis into one `SplitLastAxis`. It rewrote the FIRST member in place and
//! `Nop`ed the rest, so every later member's write moved earlier in the stream.
//! The seven `narrow(axis=2, j, 1)` slices of the transpose tile its axis and
//! are interleaved with the `narrow`s and `add`s that consume them, so all
//! seven hoisted to the first one's position. The memory planner had packed
//! nine 768-byte scratch buffers *inside* slices 2..6, each legal because it
//! died one step before the slice it shares bytes with was written. Hoisted
//! above those scratch writes, the slices lost their first 1536 bytes — exactly
//! channel 0 of each — which is precisely the observed symptom.
//!
//! The fix requires a fused run to be already adjacent in the thunk stream
//! (`Nop`s allowed), so the merge can no longer reorder. It also fuses *more*
//! than before: two interleaved tilings of one source used to fail the
//! whole-group tiling test and now become two splits.
//!
//! ## The same class, in wgpu, found the same day
//!
//! AXON (`MannasAI/axon-eeg`) looked identical from the outside — exact under
//! `RLX_ARENA_NO_REUSE=1`, wrong without — and was also not the arena. wgpu's
//! `FusedMatMulBiasAct → Narrow×3` split-QKV elision writes Q/K/V straight into
//! the narrows' slots, and its guard checks every *other* tensor live across the
//! matmul's step while **excluding q, k and v from the scan**, so the three
//! sinks aliasing *each other* read as safe. `qkv_heads` makes each narrow die
//! at its own transpose one step before the next is born, so the planner gives
//! all three one offset; fused, one dispatch writes Q, K and V into the same
//! bytes. See `rlx-wgpu/tests/qkv_split_sink_alias.rs`.
//!
//! Both are rewrites chosen *after* planning, answering to a plan built before
//! they existed, and in both the planner was correct throughout. Any pass that
//! `Nop`s a sibling thunk or step is a candidate — there are about fifteen in
//! `rlx-metal/src/thunk/mod.rs` alone.
//!
//! ## How it was isolated, and the two instruments that mattered
//!
//! What it was NOT, each by measurement:
//!
//! * **Not the planner.** `RLX_MEM_VERIFY=1` reports 0 live+memory overlaps, 0
//!   reads-after-death and 0 view-past-root overflows over the 1031 real
//!   buffers; an independent pass over the dumped plan found no node reading a
//!   buffer whose bytes had already been rewritten. The plan is valid — *for
//!   the order the planner was given*.
//! * **Not a GPU race.** `RLX_METAL_CONCURRENT_SPLIT_ENC=1` gives every
//!   dispatch its own encoder, a strictly stronger ordering guarantee than a
//!   `memoryBarrier`, and the output was bit-identically wrong.
//! * **Not the narrow kernel.** `RLX_METAL_HOST_SLICE=1` runs the narrows as
//!   host `memcpy`s after `waitUntilCompleted`, and the output was again
//!   bit-identically wrong. Two different executions agreeing to the last bit
//!   is the signature of a wrong *schedule*, not a wrong kernel.
//! * **Not MPS, MPSGraph, or region fusion.** `RLX_DISABLE_MPS`,
//!   `RLX_DISABLE_MPSGRAPH`, `RLX_DECOMPOSE_FUSION_REGIONS`,
//!   `RLX_DISABLE_TRANSPOSE_ELISION`, `RLX_METAL_PRECISE`,
//!   `RLX_METAL_NARROW_BATCH=1` (encoder-level batching off),
//!   `RLX_METAL_CONCAT_HOST=1`: all bit-identical, all with the same
//!   `arena_size=92079492`.
//! * **Not this shape.** Every variant below matches CPU with and without reuse,
//!   on Metal and on wgpu, including a 12-deep matmul chain in front for arena
//!   pressure, rank-4 and rank-5 transposes, and a full pre-norm attention
//!   block plus gated FFN feeding the narrows.
//!
//! **`RLX_ARENA_NO_REUSE_OPS` sent this in the wrong direction and it is worth
//! knowing why.** Pinning `Transpose` (68 nodes) turned MISMATCH into PARITY,
//! which reads like "a transpose whose slot is reused". But pinning `Narrow` or
//! `Concat` fixed it too, and so did the *disjoint* halves `Transpose:0-33` and
//! `Transpose:34-67`. Every pin repacks the arena, so a pin that fixes a
//! layout-sensitive defect has destroyed the evidence it was meant to collect.
//!
//! The two instruments that did work were both **print-only**, so the failing
//! layout stayed intact: `RLX_ARENA_REPORT` (added for this; dumps offset / end
//! / size / op / live range / view root / inputs, sorted by offset, so a
//! buffer's arena neighbours are its neighbours in the output) and
//! `RLX_DUMP_SCHED`, whose `[sched 144] split_lastax` sitting one thunk after
//! `[sched 143] transpose` was the whole answer. The general lesson: when a
//! model is wrong under reuse and right without it, dump the plan and the
//! schedule before pinning anything.
//!
//! ## What these tests do
//!
//! Each variant builds a graph, runs it on CPU, then on the GPU backend twice —
//! once with slot reuse (the default) and once with `RLX_ARENA_NO_REUSE=1` — and
//! **reports** the two max-abs differences rather than asserting, so a future
//! gap does not break CI. The signature to look for is
//!
//! ```text
//!   reuse=<large>   no_reuse=<~0>
//! ```
//!
//! as opposed to both columns ~0 (this variant does not reproduce it) or both
//! large (an ordinary lowering bug, nothing to do with the arena).
//!
//! The regression test for the actual bug is
//! `rlx-metal`'s `thunk::fuse_narrow_cluster_tests`, which checks the pass
//! directly. Full-model check:
//!
//! ```text
//! RLX_FORCE_DEVICE=metal cargo test -p rlx-diver1 --test parity_real --features metal -- --nocapture
//! ```
//!
//! (needs `~/.cache/rlx-parity/diver1-real/`, generated by
//! `parity/models/diver1-real/export_parity.py` in the exg workspace from the
//! 50 MB public MIT checkpoint.)

#![cfg(all(any(feature = "metal", feature = "gpu"), feature = "cpu"))]

use rlx_ir::infer::GraphExt as _;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

fn seeded(n: usize, step: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % step) as f32 - (step as f32) / 2.0) * 0.25)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Build once, run on CPU and on `device` with and without slot reuse, report.
fn bisect(
    name: &str,
    device: Device,
    build: impl Fn(&mut Graph) -> NodeId,
    inputs: &[(&str, &[f32])],
    params: &[(&str, &[f32])],
) {
    let mut g = Graph::new(name);
    let out = build(&mut g);
    g.set_outputs(vec![out]);

    let run_on = |dev: Device, graph: Graph| -> Vec<f32> {
        let mut c = Session::new(dev).compile(graph);
        for (n, d) in params {
            c.set_param(n, d);
        }
        c.finalize_params();
        c.run(inputs).into_iter().next().unwrap()
    };

    let cpu = run_on(Device::Cpu, g.clone());

    let reuse = run_on(device, g.clone());

    rlx_ir::env::set("RLX_ARENA_NO_REUSE", "1");
    let no_reuse = run_on(device, g);
    rlx_ir::env::unset("RLX_ARENA_NO_REUSE");

    let scale = cpu.iter().map(|v| v.abs()).fold(1e-9, f32::max);
    eprintln!(
        "[arena-bisect:{name} on {device:?}] |cpu| = {scale:e}   reuse = {:e}   no_reuse = {:e}",
        max_abs_diff(&cpu, &reuse),
        max_abs_diff(&cpu, &no_reuse),
    );
}

/// A matmul chain, then a rank-4 Transpose, then a sum of nested Narrows of the
/// transposed view — the shape of DIVER-1's STCPE overlap-average, minus the
/// transformer in front of it.
fn build_narrow_sum(g: &mut Graph, depth: usize, rank5: bool) -> NodeId {
    // [rows, dim] -> a few matmuls, so earlier buffers die and slots get reused
    let (rows, dim) = (16 * 42, 32);
    let x = g.input("x", Shape::new(&[rows, dim], F));
    let mut h = x;
    for i in 0..depth {
        let w = g.param(format!("w{i}"), Shape::new(&[dim, dim], F));
        h = g.mm(h, w);
        h = g.silu(h);
    }
    // reshape to the (window, offset) grid DIVER-1 folds, then permute it
    let (nw, win, c) = (16usize, 7usize, 6usize);
    assert_eq!(nw * win * c, 672);
    let t = if rank5 {
        let v = g.reshape_(h, vec![1, nw as i64, c as i64, win as i64, dim as i64]);
        g.transpose_(v, vec![0, 2, 3, 1, 4]) // [1, C, win, nw, dim]
    } else {
        let v = g.reshape_(h, vec![nw as i64, c as i64, win as i64, dim as i64]);
        g.transpose_(v, vec![1, 2, 0, 3]) // [C, win, nw, dim]
    };
    // antidiagonal sums: for each kept position, sum one lane per offset
    let (ax_off, ax_win) = if rank5 {
        (2usize, 3usize)
    } else {
        (1usize, 2usize)
    };
    let kept = nw - win + 1;
    let mut parts = Vec::with_capacity(kept);
    for p in 0..kept {
        let mut acc: Option<NodeId> = None;
        for j in 0..win {
            let w = p + win - 1 - j;
            let a = g.narrow_(t, ax_off, j, 1);
            let b = g.narrow_(a, ax_win, w, 1);
            acc = Some(match acc {
                None => b,
                Some(prev) => g.add(prev, b),
            });
        }
        parts.push(acc.unwrap());
    }
    let cat = g.concat_(parts, ax_win);
    let inv = g.constant(1.0 / win as f64, F);
    g.mul(cat, inv)
}

fn params_for(depth: usize) -> Vec<(String, Vec<f32>)> {
    (0..depth)
        .map(|i| (format!("w{i}"), seeded(32 * 32, 23)))
        .collect()
}

/// The same narrow-sum, but fed by a small *attention* block rather than a
/// matmul chain — DIVER-1's STCPE shape: 16 windows x (6 channels x 7 offsets)
/// tokens of width 32, one head, pre-RMSNorm attention + gated FFN, then the
/// overlap-average reads the encoder output through a Transpose and 70 Narrows.
fn build_attn_then_narrow_sum(g: &mut Graph, rank5: bool) -> NodeId {
    let (nw, win, c, dim) = (16usize, 7usize, 6usize, 32usize);
    let (seqs, seq) = (nw, c * win); // 16 sequences of 42 tokens
    let x = g.input("x", Shape::new(&[seqs, seq, dim], F));

    let norm_w = g.param("norm_w", Shape::new(&[dim], F));
    let zero = g.zeros(&[dim], F);
    let nx = g.rms_norm(x, norm_w, zero, 1e-5);

    let wq = g.param("wq", Shape::new(&[dim, dim], F));
    let wk = g.param("wk", Shape::new(&[dim, dim], F));
    let wv = g.param("wv", Shape::new(&[dim, dim], F));
    let wo = g.param("wo", Shape::new(&[dim, dim], F));
    let q = g.mm(nx, wq);
    let k = g.mm(nx, wk);
    let v = g.mm(nx, wv);
    let kt = g.transpose_(k, vec![0, 2, 1]);
    let e = g.mm(q, kt);
    let sc = g.constant(1.0 / (dim as f64).sqrt(), F);
    let e = g.mul(e, sc);
    let p = g.sm(e, 2);
    let ctx = g.mm(p, v);
    let o = g.mm(ctx, wo);
    let h = g.add(x, o);

    let norm2_w = g.param("norm2_w", Shape::new(&[dim], F));
    let n2 = g.rms_norm(h, norm2_w, zero, 1e-5);
    let w1 = g.param("w1", Shape::new(&[dim, 4 * dim], F));
    let wg = g.param("wg", Shape::new(&[dim, 4 * dim], F));
    let w2 = g.param("w2", Shape::new(&[4 * dim, dim], F));
    let val = g.mm(n2, w1);
    let gate = g.mm(n2, wg);
    let gate = g.silu(gate);
    let hid = g.mul(val, gate);
    let ff = g.mm(hid, w2);
    let enc = g.add(h, ff);
    let out_w = g.param("out_w", Shape::new(&[dim], F));
    let enc = g.rms_norm(enc, out_w, zero, 1e-5);

    // ── the overlap-average: Transpose + nested Narrows of `enc` ──
    let t = if rank5 {
        let v5 = g.reshape_(enc, vec![1, nw as i64, c as i64, win as i64, dim as i64]);
        g.transpose_(v5, vec![0, 2, 3, 1, 4])
    } else {
        let v4 = g.reshape_(enc, vec![nw as i64, c as i64, win as i64, dim as i64]);
        g.transpose_(v4, vec![1, 2, 0, 3])
    };
    let (ax_off, ax_win) = if rank5 {
        (2usize, 3usize)
    } else {
        (1usize, 2usize)
    };
    let kept = nw - win + 1;
    let mut parts = Vec::with_capacity(kept);
    for pp in 0..kept {
        let mut acc: Option<NodeId> = None;
        for j in 0..win {
            let w = pp + win - 1 - j;
            let a = g.narrow_(t, ax_off, j, 1);
            let b = g.narrow_(a, ax_win, w, 1);
            acc = Some(match acc {
                None => b,
                Some(prev) => g.add(prev, b),
            });
        }
        parts.push(acc.unwrap());
    }
    let cat = g.concat_(parts, ax_win);
    let inv = g.constant(1.0 / win as f64, F);
    g.mul(cat, inv)
}

fn attn_params() -> Vec<(String, Vec<f32>)> {
    let dim = 32usize;
    let mut v: Vec<(String, Vec<f32>)> = vec![
        ("norm_w".into(), vec![1.0; dim]),
        ("norm2_w".into(), vec![1.0; dim]),
        ("out_w".into(), vec![1.0; dim]),
    ];
    for n in ["wq", "wk", "wv", "wo"] {
        v.push((n.into(), seeded(dim * dim, 23)));
    }
    v.push(("w1".into(), seeded(dim * 4 * dim, 29)));
    v.push(("wg".into(), seeded(dim * 4 * dim, 31)));
    v.push(("w2".into(), seeded(4 * dim * dim, 37)));
    v
}

fn run_ladder(device: Device) {
    let x = seeded(16 * 42 * 32, 17);
    for depth in [1usize, 4, 12] {
        for rank5 in [false, true] {
            let ps = params_for(depth);
            let pr: Vec<(&str, &[f32])> =
                ps.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
            bisect(
                &format!("narrow_sum depth={depth} rank{}", if rank5 { 5 } else { 4 }),
                device,
                |g| build_narrow_sum(g, depth, rank5),
                &[("x", &x)],
                &pr,
            );
        }
    }
    // The variant that matters: the narrows read an attention block's output.
    for rank5 in [false, true] {
        let ps = attn_params();
        let pr: Vec<(&str, &[f32])> = ps.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        bisect(
            &format!("attn+narrow_sum rank{}", if rank5 { 5 } else { 4 }),
            device,
            |g| build_attn_then_narrow_sum(g, rank5),
            &[("x", &x)],
            &pr,
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn narrow_sum_slot_reuse_metal() {
    let _gpu = common::serialize_gpu();
    run_ladder(Device::Metal);
}

#[cfg(feature = "gpu")]
#[test]
fn narrow_sum_slot_reuse_wgpu() {
    let _gpu = common::serialize_gpu();
    run_ladder(Device::Gpu);
}
