// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A backward graph reads every weight twice: `W` for the forward matmul and
//! `Wᵀ` for `dX = dY·Wᵀ`. Metal folds that transpose into the GEMM
//! (`transposeRight` on `W`), so the copy is never allocated and `W` can sit in
//! the shared weight buffer. These check the folded GEMM against the CPU, with
//! `W` both inline and in the weight buffer, for a matrix and a stack of rows on
//! the left, and against the unfolded Metal path.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};
use std::sync::{Mutex, MutexGuard};

static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct Guard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Guard {
    fn new() -> Self {
        // The env knobs below are read during compile.
        Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        for k in [
            "RLX_METAL_WEIGHT_SPLIT_MIN",
            "RLX_METAL_NO_WEIGHT_SPLIT",
            "RLX_METAL_NO_WEIGHT_TRANSPOSE_FOLD",
        ] {
            rlx_ir::env::unset(k);
        }
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

/// `y = x·W` and `z = g·Wᵀ` over one weight `W[k, n]`; `lead` is the shape of
/// the rows (`[rows]` or `[b, s]`).
fn graph(lead: &[usize], k: usize, n: usize, direct: bool) -> Graph {
    let f = DType::F32;
    let dims = |last: usize| -> Vec<usize> { lead.iter().copied().chain([last]).collect() };
    let mut g = Graph::new("w_t_fold");
    let w = g.param("w", Shape::new(&[k, n], f));
    let gy = g.input("g", Shape::new(&dims(n), f));
    let wt = g.transpose_(w, vec![1, 0]);
    let z = g.matmul(gy, wt, Shape::new(&dims(k), f));
    if direct {
        let x = g.input("x", Shape::new(&dims(k), f));
        let y = g.matmul(x, w, Shape::new(&dims(n), f));
        g.set_outputs(vec![y, z]);
    } else {
        g.set_outputs(vec![z]);
    }
    g
}

fn run(device: Device, lead: &[usize], k: usize, n: usize, direct: bool) -> Vec<Vec<f32>> {
    let rows: usize = lead.iter().product();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 19) as f32 - 9.0) * 0.01).collect();
    let x: Vec<f32> = (0..rows * k).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
    let gy: Vec<f32> = (0..rows * n).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();
    let mut c = Session::new(device).compile(graph(lead, k, n, direct));
    c.set_param("w", &w);
    if direct {
        c.run(&[("g", &gy), ("x", &x)])
    } else {
        c.run(&[("g", &gy)])
    }
}

fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: outputs");
    for (o, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: output {o} length");
        let bad = a.iter().zip(b).position(|(p, q)| (p - q).abs() > 1e-3);
        assert!(
            bad.is_none(),
            "{what}: output {o} differs at {bad:?} ({:?} vs {:?})",
            bad.map(|i| a[i]),
            bad.map(|i| b[i]),
        );
    }
}

/// 256×256 F32 is 256 KiB, the size at which a weight may leave the arena.
const K: usize = 256;
const N: usize = 256;

#[test]
fn folded_transpose_matches_cpu_with_weights_inline() {
    let _g = Guard::new();
    rlx_ir::env::set("RLX_METAL_NO_WEIGHT_SPLIT", "1");
    for lead in [&[5usize][..], &[2, 3]] {
        let want = run(Device::Cpu, lead, K, N, true);
        assert_close(&run(Device::Metal, lead, K, N, true), &want, &format!("inline {lead:?}"));
    }
}

#[test]
fn folded_transpose_matches_cpu_with_weights_in_the_weight_buffer() {
    let _g = Guard::new();
    // Low enough that this weight trips the share-driven split, which a folded
    // `Wᵀ` no longer blocks: both readers of `W` are GEMMs.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "262144");
    for lead in [&[5usize][..], &[2, 3]] {
        let want = run(Device::Cpu, lead, K, N, true);
        assert_close(&run(Device::Metal, lead, K, N, true), &want, &format!("split {lead:?}"));
    }
}

#[test]
fn fold_and_materialized_transpose_agree() {
    let _g = Guard::new();
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "262144");
    let folded = run(Device::Metal, &[2, 3], K, N, true);
    rlx_ir::env::set("RLX_METAL_NO_WEIGHT_TRANSPOSE_FOLD", "1");
    let copied = run(Device::Metal, &[2, 3], K, N, true);
    assert_close(&folded, &copied, "fold vs copy");
}

/// A weight read only through its transpose (an HF-layout linear) keeps the
/// existing path, and still computes the right thing.
#[test]
fn transpose_only_weight_is_left_alone() {
    let _g = Guard::new();
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "262144");
    let want = run(Device::Cpu, &[5], K, N, false);
    assert_close(&run(Device::Metal, &[5], K, N, false), &want, "transpose only");
}
