// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **An op whose operands land in different arena stripes must not return zeros.**
//!
//! When the logical arena is striped across several GPU buffers, a kernel binds
//! **one** stripe. Slot placement guarantees no single tensor straddles a
//! boundary — which is not the same as guaranteeing that an op's *operands*
//! share a stripe, because nothing in the planner knows which tensors are used
//! together. When they do not, the operand outside the bound window reads as
//! **zero**: no error, no warning, and a result that is entirely plausible.
//! `Arena::straddles_shards` was written to catch exactly this at compile time.
//!
//! Sharding normally needs a multi-GiB arena, so this pins a tiny shard cap
//! (`RLX_WGPU_SHARD_CAP_MIB`) and keeps ops on the GPU (`RLX_WGPU_SHARD_GPU=1`)
//! — hosting is lane- and stripe-safe, so it would hide the very thing under
//! test. Both knobs clamp toward smaller / more-GPU, so a device that cannot
//! shard this small simply reports and skips rather than passing vacuously.
//!
//! Note both knobs are process-global: this file must run single-threaded
//! (`--test-threads=1`) and is written as ONE test for that reason.
//!
//! Scope: only the `RLX_WGPU_SHARD_GPU` routing. The DEFAULT sharded routing
//! hosts these ops and was measured correct (0 of 786432 outputs wrong), so it
//! is not what can regress here.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_wgpu::backend::WgpuExecutable;

const F: DType = DType::F32;
/// Small enough that a few megabyte-scale tensors cannot share one stripe.
const SHARD_MIB: usize = 4;

#[test]
fn operands_in_different_stripes_do_not_silently_read_as_zero() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // SAFETY: single-threaded by construction (see module note); set before any
    // arena is built.
    unsafe {
        std::env::set_var("RLX_WGPU_SHARD_CAP_MIB", SHARD_MIB.to_string());
        std::env::set_var("RLX_WGPU_SHARD_GPU", "1");
        // Striping is a hard error by default — it produces wrong results
        // silently (SynthStrip at 192^3: every voxel wrong, 97.8% of scale,
        // run reported success). This test exists to exercise the straddle
        // guard *within* a striped arena, so it opts back in deliberately.
        std::env::set_var("RLX_WGPU_ALLOW_SHARD", "1");
    }

    // Three ~3 MiB tensors: with a 4 MiB stripe they cannot all co-reside, so
    // at least one binary op must have operands in different stripes.
    const N: usize = 3 * 1024 * 1024 / 4; // f32 elements ≈ 3 MiB

    let mut g = Graph::new("straddle");
    let a = g.input("a", Shape::new(&[N], F));
    let b = g.param("b", Shape::new(&[N], F));
    let c = g.param("c", Shape::new(&[N], F));
    // Chain so the intermediates stay live and get pushed into later stripes.
    let ab = g.add(a, b);
    let abc = g.add(ab, c);
    let out = g.mul(abc, ab);
    g.set_outputs(vec![out]);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &vec![2.0f32; N]);
    exe.set_param("c", &vec![3.0f32; N]);
    let av = vec![1.0f32; N];
    let got = exe.run(&[("a", &av)])[0].clone();

    // a=1, b=2, c=3 -> ab=3, abc=6, out=18. A stripe-straddling operand reading
    // as zero collapses this to 0 (or 9, if only `c` is lost).
    let want = 18.0f32;
    let bad = got.iter().filter(|v| (**v - want).abs() > 1e-3).count();
    let zeros = got.iter().filter(|v| **v == 0.0).count();

    eprintln!(
        "  N={N} elems, shard cap={SHARD_MIB} MiB: {bad}/{} wrong, {zeros} exactly zero",
        got.len()
    );
    assert_eq!(
        bad,
        0,
        "{bad} of {} outputs are wrong ({zeros} exactly zero) — operands landed \
         in different arena stripes and the unbound one read as zero. Expected \
         {want} everywhere, got e.g. {:?}",
        got.len(),
        &got[..got.len().min(4)]
    );
}
