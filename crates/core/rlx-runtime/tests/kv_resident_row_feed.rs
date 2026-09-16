// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The resident-KV row feed must actually accumulate on the device.**
//!
//! `Op::KvAppend` alone does not remove the O(context) decode cost: the concat
//! it replaces copies the cache on-GPU, but the runtime still re-uploads the
//! past KV as a graph input every step, so the cost merely moves. Residency is
//! the other half — `bind_gpu_handle` keeps the cache on the device and
//! `feed_kv_row` folds each new row into it without a host round trip.
//!
//! This is easy to wire up and have silently do nothing: registering a feed and
//! calling it returns `true` whether or not any bytes moved. So the check is
//! behavioural — run twice, feeding a different row each time, and require the
//! resident buffer to show BOTH rows. A no-op feed shows neither.
//!
//! Backends without residency return `false` from `register_kv_row_feed`; the
//! test reports and skips rather than asserting a capability they never claimed
//! (`ExecutableCapabilities::kv_resident`).

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;
const ROW: usize = 4;
const CAP: usize = 4;

/// `out = cache * 1.0` — an identity pass whose output shares the cache's
/// layout, so a row of the output can be fed back into the resident input.
fn feed_graph() -> Graph {
    let mut g = Graph::new("kv_feed");
    let cache = g.input("past_k", Shape::new(&[1, CAP, ROW], F));
    let one = g.param("one", Shape::new(&[1, CAP, ROW], F));
    let out = g.mul(cache, one);
    g.set_outputs(vec![out]);
    g
}

fn devices() -> Vec<(&'static str, Device)> {
    [
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("vulkan", Device::Vulkan),
    ]
    .into_iter()
    .filter(|(_, d)| rlx_runtime::is_available(*d))
    .collect()
}

#[test]
fn a_registered_row_feed_accumulates_into_the_resident_handle() {
    let _gpu = common::serialize_gpu();
    for (name, dev) in devices() {
        let mut c = Session::new(dev).compile(feed_graph());
        c.set_param("one", &[1.0f32; CAP * ROW]);
        c.finalize_params();

        // Cache starts zeroed and lives on the device.
        let zeros = vec![0.0f32; CAP * ROW];
        if !c.bind_gpu_handle("past_k", &zeros) {
            eprintln!("  {name}: no GPU-resident handles — skipping");
            continue;
        }
        if !c.register_kv_row_feed("past_k", 0) {
            eprintln!("  {name}: no resident KV row feed — skipping");
            continue;
        }

        // Step 1: put a marker in row 0 of the output, fold it into row 1.
        let mut seed = zeros.clone();
        seed[0..ROW].copy_from_slice(&[11.0, 12.0, 13.0, 14.0]);
        c.bind_gpu_handle("past_k", &seed);
        let _ = c.run(&[]);
        c.feed_kv_row(0, 1, ROW);

        // Step 2: a different marker in row 0, folded into row 2. If the feed
        // is a no-op this leaves row 1 empty; if it re-uploads from the host it
        // loses row 1 as well.
        let out = c.run(&[]);
        let after = &out[0];
        let row1 = &after[ROW..2 * ROW];

        assert_eq!(
            row1,
            &[11.0, 12.0, 13.0, 14.0],
            "{name}: row 1 of the resident cache does not hold what \
             `feed_kv_row(0, 1, ..)` folded into it — the feed moved no bytes \
             (got {row1:?})"
        );
        eprintln!("  {name}: row feed accumulated correctly");
    }
}
