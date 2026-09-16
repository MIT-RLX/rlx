// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`Op::KvAppend` must mean the same thing on every backend.**
//!
//! The op writes one row into a KV cache at `pos` and returns the `[..pos+1]`
//! prefix, aliasing the cache buffer — O(1) per decode step where the
//! `concat(past_kv, new_row)` it replaces is O(context). Backends that
//! implement it natively encode a single row write; the rest go through
//! `rlx_fusion::lower_kv_append`, which rebuilds it from `narrow` + `concat`.
//!
//! Two implementations of one op is exactly where they drift, and the failure
//! is quiet: a wrong row offset writes plausible-looking values one step early
//! or late, which a loss curve absorbs. So this pins the semantics against a
//! reference computed in the test itself, on every backend the build has.
//!
//! `pos == 0` is included deliberately: the lowering special-cases it (the
//! prefix IS the row, so no concat), and that is the branch a decode hits on
//! its first step.

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

/// `[batch, seq_cap, width]` cache, write `row` at `pos`.
///
/// The output is the `[..pos+1]` PREFIX of the cache (see `infer_shape`), not
/// the whole cache — it aliases the cache's buffer but declares fewer rows.
fn kv_graph(batch: usize, seq_cap: usize, width: usize, pos: usize) -> Graph {
    let mut g = Graph::new("kv_append");
    let cache = g.input("cache", Shape::new(&[batch, seq_cap, width], F));
    let row = g.input("row", Shape::new(&[batch, 1, width], F));
    let out = g.add_node(
        Op::KvAppend { axis: 1, pos },
        vec![cache, row],
        Shape::new(&[batch, pos + 1, width], F),
    );
    g.set_outputs(vec![out]);
    g
}

fn devices() -> Vec<(&'static str, Device)> {
    let mut v = vec![("cpu", Device::Cpu)];
    for (name, d) in [
        ("metal", Device::Metal),
        // MLX has no native `KvAppend` — its immutable array API has no
        // in-place row write — so it exercises the `lower_kv_append`
        // narrow+concat rebuild. That is exactly the path most likely to drift
        // from the native ones, which makes it worth a row here.
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("vulkan", Device::Vulkan),
    ] {
        if rlx_runtime::is_available(d) {
            v.push((name, d));
        }
    }
    v
}

#[test]
fn kv_append_writes_exactly_one_row_on_every_backend() {
    let _gpu = common::serialize_gpu();
    const CAP: usize = 8;
    const W: usize = 4;

    for pos in [0usize, 1, 5, CAP - 1] {
        let cache: Vec<f32> = (0..CAP * W).map(|i| i as f32).collect();
        let row: Vec<f32> = (0..W).map(|i| 1000.0 + i as f32).collect();

        // Output is the [..pos+1] prefix of the (updated) cache.
        let mut full = cache.clone();
        full[pos * W..pos * W + W].copy_from_slice(&row);
        let want = full[..(pos + 1) * W].to_vec();

        for (name, dev) in devices() {
            let mut c = Session::new(dev).compile(kv_graph(1, CAP, W, pos));
            let got = c.run(&[("cache", &cache), ("row", &row)]).remove(0);
            assert_eq!(
                got, want,
                "{name} pos={pos}: KvAppend did not write exactly one row.\n\
                 got  {got:?}\nwant {want:?}"
            );
        }
    }
}

/// **Batch > 1 pins the row stride.**
///
/// The output declares `[..pos+1]` rows while the buffer it aliases has
/// `seq_cap`. A backend that takes the stride between batch slices from the
/// OUTPUT shape gets `pos+1` instead of `seq_cap` and writes batch 1's row into
/// the middle of batch 0's cache. With `batch == 1` that stride is never used,
/// so the whole class of error is invisible — which is why it needs its own
/// case rather than another `pos` value.
#[test]
fn kv_append_uses_the_cache_stride_not_the_output_stride() {
    let _gpu = common::serialize_gpu();
    const B: usize = 2;
    const CAP: usize = 8;
    const W: usize = 4;
    let pos = 2usize;

    let cache: Vec<f32> = (0..B * CAP * W).map(|i| i as f32).collect();
    let row: Vec<f32> = (0..B * W).map(|i| 1000.0 + i as f32).collect();

    let mut full = cache.clone();
    for b in 0..B {
        let d = b * CAP * W + pos * W;
        full[d..d + W].copy_from_slice(&row[b * W..b * W + W]);
    }
    // Prefix per batch slice.
    let mut want = Vec::with_capacity(B * (pos + 1) * W);
    for b in 0..B {
        let base = b * CAP * W;
        want.extend_from_slice(&full[base..base + (pos + 1) * W]);
    }

    for (name, dev) in devices() {
        let mut c = Session::new(dev).compile(kv_graph(B, CAP, W, pos));
        let got = c.run(&[("cache", &cache), ("row", &row)]).remove(0);
        assert_eq!(
            got, want,
            "{name}: batched KvAppend used the wrong row stride.\n\
             got  {got:?}\nwant {want:?}"
        );
    }
}

/// **The row write is sized in BYTES, and half of the backends forgot.**
///
/// Every native implementation strides the cache by `row_elems * elem_bytes`,
/// and the element size has to come from the cache's real dtype. rlx-metal took
/// it from `HalfFlag`, which only distinguishes 4-byte from 2-byte and maps
/// BF16 to the 4-byte arm — so a BF16 cache strode twice as far as it should,
/// writing the new token past the row it was aimed at. rlx-wgpu copies whole
/// f32 words on all of its paths, which rounds a row that is not word-sized up
/// into its neighbour. rlx-cuda / rlx-rocm address f32 LANES, which is the
/// element count for most dtypes and not for byte-packed or complex ones.
///
/// The cache and row are declared low precision at the BOUNDARY (`Op::Input`),
/// which is the only way one survives: `promote_to_f32` rewrites interior
/// F16/BF16 nodes to F32 for execution, so a cache built with an interior
/// `Cast` is an f32 cache by the time a backend sees it and this test would
/// prove nothing. `KvAppend` itself is in that pass's layout-op set for the
/// same reason — it copies bits and computes nothing, so promoting it while its
/// aliased cache stayed F16 is what made the stride disagree with the buffer.
fn kv_graph_narrow(dt: DType, seq_cap: usize, width: usize, pos: usize) -> Graph {
    let mut g = Graph::new("kv_append_narrow");
    let cache = g.input("cache", Shape::new(&[1, seq_cap, width], dt));
    let row = g.input("row", Shape::new(&[1, 1, width], dt));
    let out = g.add_node(
        Op::KvAppend { axis: 1, pos },
        vec![cache, row],
        Shape::new(&[1, pos + 1, width], dt),
    );
    g.set_outputs(vec![out]);
    g
}

#[test]
fn kv_append_strides_a_narrow_cache_by_its_own_element_size() {
    let _gpu = common::serialize_gpu();
    const CAP: usize = 8;
    const W: usize = 4;

    // Small integers are exact in both f16 and bf16 (bf16 keeps 8 mantissa
    // bits), so a correct write round-trips bit for bit through the boundary
    // conversion and a mis-strided one does not.
    for dt in [DType::F16, DType::BF16] {
        for pos in [0usize, 1, 5, CAP - 1] {
            let cache: Vec<f32> = (0..CAP * W).map(|i| i as f32).collect();
            let row: Vec<f32> = (0..W).map(|i| 100.0 + i as f32).collect();

            let mut full = cache.clone();
            full[pos * W..pos * W + W].copy_from_slice(&row);
            let want = full[..(pos + 1) * W].to_vec();

            for (name, dev) in devices() {
                // NOT a KvAppend gap: rlx-metal cannot round-trip a BF16
                // BOUNDARY tensor at all. A graph that only narrows a BF16
                // `Op::Input` returns `[1, 3.004, 5.008, 7.008]` for an input of
                // `[0, 1, 2, 3]` — every other element, i.e. the buffer is read
                // at the wrong stride on the way in or out. BF16 is a weight
                // storage format there (`HalfFlag` is F32|F16, so no kernel
                // speaks it), and nothing in the KV path can fix that. Named
                // rather than skipped so the exclusion stays visible and gets
                // deleted when Metal grows BF16 I/O.
                if name == "metal" && dt == DType::BF16 {
                    continue;
                }
                let mut c = Session::new(dev).compile(kv_graph_narrow(dt, CAP, W, pos));
                let got = c.run(&[("cache", &cache), ("row", &row)]).remove(0);
                assert_eq!(
                    got, want,
                    "{name} {dt:?} pos={pos}: KvAppend wrote the wrong row.\n\
                     got  {got:?}\nwant {want:?}"
                );
            }
        }
    }
}
