// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An executable whose weight buffer passes 4 GiB runs on thunks unless
//! `RLX_METAL_MPSGRAPH_PAST_CLIFF=1`. This checks that flag end to end: a
//! 4 GiB weight ahead of a small one puts the small one past 4 GiB into the
//! weight buffer, and MPSGraph must compute what the thunk path does.
//! Ignored by default: it needs ~4.5 GB free.
//!
//! `cargo test -p rlx-runtime --features metal --test mps_graph_weights_past_cliff -- --ignored`

#![cfg(all(feature = "metal", target_vendor = "apple"))]

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

/// `[1, K]·big[K, BIG_N]` (4 GiB) and `[1, K]·small[K, N]`.
const K: usize = 1024;
const BIG_N: usize = (1 << 30) / K;
const N: usize = 256;

fn run(scope: &str) -> (Vec<Vec<f32>>, rlx_runtime::MemoryStats) {
    let f = DType::F32;
    let mut g = Graph::new("past_cliff");
    let x = g.input("x", Shape::new(&[1, K], f));
    let big = g.param("big", Shape::new(&[K, BIG_N], f));
    let small = g.param("small", Shape::new(&[K, N], f));
    let a = g.mm(x, big);
    let b = g.mm(x, small);
    let a = g.sum(a, vec![1], true);
    g.set_outputs(vec![a, b]);
    // A weight buffer this large is only split out under a sharing scope, which
    // is how the models that reach it compile.
    let scope = format!("{scope}:{}", std::process::id());
    let mut c =
        rlx_runtime::with_param_sharing_scope(&scope, || Session::new(Device::Metal).compile(g));
    // `big` stays zero (its pages untouched); `small` carries the answer.
    let w: Vec<f32> = (0..K * N)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
        .collect();
    c.set_param("small", &w);
    let x: Vec<f32> = (0..K).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let out = c.run(&[("x", &x)]);
    (out, c.memory_stats().expect("stats"))
}

#[test]
#[ignore]
fn mpsgraph_reads_weights_past_four_gib() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let (want, thunks) = run("past-cliff-thunks");
    assert!(
        thunks.weight_buffer_bytes >= 1 << 32 && !thunks.mpsgraph,
        "{thunks:?}"
    );
    rlx_ir::env::set("RLX_METAL_MPSGRAPH_PAST_CLIFF", "1");
    let (got, mps) = run("past-cliff-mpsgraph");
    rlx_ir::env::unset("RLX_METAL_MPSGRAPH_PAST_CLIFF");
    eprintln!("thunks {thunks:?}\nmpsgraph {mps:?}");
    assert!(mps.mpsgraph, "did not lower to MPSGraph: {mps:?}");
    // `small` follows `big` in the weight buffer, so it starts at 4 GiB; a read
    // from the wrong place would come back zero on both paths and still agree.
    assert!(
        want[1].iter().any(|v| v.abs() > 1e-3),
        "small·x read back zero"
    );
    for (o, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-3, "output {o} off by {d}");
    }
}
