// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two differently-shaped graphs over the same weights must share one copy.
//!
//! A transformer keeps several executables alive at once — a prefill graph per
//! prompt length, a decode graph per KV bucket — and they all want the same
//! weights. Inlining each graph's weights into its own activation arena made a
//! model's resident size grow with the number of shapes it had been asked for:
//! measured on an f32 0.6B, 2.95 GB after load became 12.89 GB after two prompt
//! shapes and two decode buckets.
//!
//! The split only used to fire when the arena crossed MPS's 4 GiB binding cliff,
//! which is a different problem — so a model just under it duplicated silently.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::sync::{Mutex, MutexGuard};

static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct Guard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Guard {
    fn new() -> Self {
        // The split threshold is read during compile, and these tests set it, so
        // they must not run concurrently with each other.
        Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        rlx_ir::env::unset("RLX_METAL_WEIGHT_SPLIT_MIN");
        rlx_ir::env::unset("RLX_METAL_NO_WEIGHT_SPLIT");
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

/// `x[rows, k] @ w[k, n]`, with `w` a param big enough to be worth splitting out.
/// Only `rows` differs between the two graphs — the same shape relationship a
/// prefill graph at two prompt lengths has.
fn matmul_graph(rows: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("w_share");
    let x = g.input("x", Shape::new(&[rows, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let y = g.matmul(x, w, Shape::new(&[rows, n], f));
    g.set_outputs(vec![y]);
    g
}

fn compile_with_weights(rows: usize, k: usize, n: usize, w: &[f32]) -> rlx_runtime::CompiledGraph {
    let mut c = Session::new(Device::Metal).compile(matmul_graph(rows, k, n));
    c.set_param("w", w);
    c
}

#[test]
fn two_shapes_share_one_weight_buffer_and_agree_with_cpu() {
    let _g = Guard::new();
    // Low enough that a modest param trips the split; the production default is
    // 512 MB, which a test should not have to allocate.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "262144");

    let (k, n) = (256usize, 256usize);
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();

    let mut first = compile_with_weights(4, k, n, &w);
    let mut second = compile_with_weights(9, k, n, &w);
    // Second graph retains the first's weight buffer rather than uploading again.
    let shared = second.share_params_from(&first);
    assert!(
        shared,
        "a second shape over identical params should share the weight buffer"
    );

    // Sharing is only worth anything if the shared weights are still read
    // correctly — a retained buffer that the kernels do not resolve would give
    // plausible-looking numbers from the wrong memory.
    for rows in [4usize, 9] {
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.02)
            .collect();
        let want = {
            let mut c = Session::new(Device::Cpu).compile(matmul_graph(rows, k, n));
            c.set_param("w", &w);
            c.run(&[("x", &x)]).remove(0)
        };
        let got = if rows == 4 {
            first.run(&[("x", &x)]).remove(0)
        } else {
            second.run(&[("x", &x)]).remove(0)
        };
        assert_eq!(got.len(), want.len(), "rows={rows}: length");
        let bad = got
            .iter()
            .zip(&want)
            .position(|(a, b)| (a - b).abs() > 1e-3);
        assert!(
            bad.is_none(),
            "rows={rows}: shared weights read back wrong at {:?} ({:?} vs {:?})",
            bad,
            bad.map(|i| got[i]),
            bad.map(|i| want[i]),
        );
    }
}

/// With the split off, sharing has nothing to share and must decline rather than
/// claim success — a false positive would skip the upload and leave a graph with
/// no weights at all.
#[test]
fn sharing_declines_when_weights_stay_inline() {
    let _g = Guard::new();
    rlx_ir::env::set("RLX_METAL_NO_WEIGHT_SPLIT", "1");

    let (k, n) = (256usize, 256usize);
    let w: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let first = compile_with_weights(4, k, n, &w);
    let mut second = compile_with_weights(9, k, n, &w);
    assert!(
        !second.share_params_from(&first),
        "no weight buffer means nothing to share; the caller must upload"
    );
}
