// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concurrent MPS matmul must not race, and must stay numerically right.
//!
//! `mps_blas` caches retained Objective-C objects keyed by shape. `CACHE_GUARD`
//! makes those pointers stay *alive* across an encode — it stops
//! `invalidate_caches` freeing one mid-use — but that is a lifetime guarantee,
//! not an exclusivity one. An `MPSMatrixMultiplication` is stateful and mutates
//! itself while encoding (`setIndexingArithmaticTypeMask:sourceArrays:…`), so
//! two threads that encoded the *same shape* shared one instance and raced
//! inside MPS: `EXC_BAD_ACCESS` in `MPSNDArrayMultiaryBase`, an Apple frame,
//! with nothing in rlx's own stack looking wrong.
//!
//! The kernel cache is now keyed by thread as well as shape. This pins that,
//! and pins the two things a "just stop it crashing" fix could still get wrong:
//!
//! * **Same shape on every thread** is the original reproducer — one cache key,
//!   maximal contention. It crashed 5/5 before the fix.
//! * **Numerics, not just survival.** A race that corrupts a kernel's stashed
//!   dimensions can produce a plausible wrong answer rather than a signal, so
//!   every result is checked against a CPU reference. A no-crash-only test would
//!   pass on a fix that merely narrowed the window.
//!
//! `RLX_METAL_SGEMM_MPS=1` forces the MPS path at shapes the cost model would
//! otherwise route to the hand-written sgemm — without it this test would
//! silently exercise the wrong kernel and prove nothing. It is set before any
//! thread starts, and the whole file is one `#[test]` because that switch is
//! process-global.

#![cfg(target_os = "macos")]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// Distinct enough that a swapped operand or a stale dimension shows up.
fn a_data(m: usize, k: usize, salt: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i + salt * 7) as f32 * 0.017).sin() + 0.25)
        .collect()
}

fn b_data(k: usize, n: usize, salt: usize) -> Vec<f32> {
    (0..k * n)
        .map(|i| ((i + salt * 13) as f32 * 0.023).cos() - 0.15)
        .collect()
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

fn matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("mps_conc");
    let a = g.input("a", Shape::new(&[m, k], f));
    let b = g.input("b", Shape::new(&[k, n], f));
    let c = g.matmul(a, b, Shape::new(&[m, n], f));
    g.set_outputs(vec![c]);
    g
}

/// One thread's work: compile once, run `iters` times, check every result.
fn hammer(m: usize, k: usize, n: usize, salt: usize, iters: usize) -> Result<(), String> {
    let a = a_data(m, k, salt);
    let b = b_data(k, n, salt);
    let want = reference(&a, &b, m, k, n);

    let mut compiled = Session::new(Device::Metal).compile(matmul_graph(m, k, n));
    for it in 0..iters {
        let out = compiled.run(&[("a", &a[..]), ("b", &b[..])]);
        let got = &out[0];
        if got.len() != want.len() {
            return Err(format!(
                "{m}x{k}x{n} salt={salt} iter={it}: len {} != {}",
                got.len(),
                want.len()
            ));
        }
        for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
            // f32 accumulation order differs from the reference; this is a
            // correctness bound, not a bit-exactness claim.
            let tol = 1e-3 * w.abs().max(1.0);
            if (g - w).abs() > tol {
                return Err(format!(
                    "{m}x{k}x{n} salt={salt} iter={it}: [{idx}] metal={g} cpu={w}"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn concurrent_mps_matmul_is_race_free_and_correct() {
    rlx_ir::env::set("RLX_METAL_SGEMM_MPS", "1");

    // Multiple of 8 in every extent: `pick_sgemm` only routes aligned shapes to
    // MPS, so an unaligned pick would quietly test the other kernel.
    const SHARED: (usize, usize, usize) = (64, 128, 64);
    const ITERS: usize = 12;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(4, 8))
        .unwrap_or(4);

    let errors: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    // Half the threads pound the SAME (m,k,n) — one cache key, maximal
    // contention, and the exact configuration that crashed.
    for t in 0..threads {
        let errs = Arc::clone(&errors);
        handles.push(std::thread::spawn(move || {
            let (m, k, n) = SHARED;
            if let Err(e) = hammer(m, k, n, t, ITERS) {
                errs.lock().expect("errors").push(e);
            }
        }));
    }
    // The other half vary the shape, so distinct keys are being inserted into
    // the same map while the shared key is being read.
    for t in 0..threads {
        let errs = Arc::clone(&errors);
        handles.push(std::thread::spawn(move || {
            let m = 8 * (t + 1);
            let (k, n) = (64, 32);
            if let Err(e) = hammer(m, k, n, 100 + t, ITERS) {
                errs.lock().expect("errors").push(e);
            }
        }));
    }

    let mut panicked = 0;
    for h in handles {
        if h.join().is_err() {
            panicked += 1;
        }
    }

    let errs = errors.lock().expect("errors");
    assert!(
        errs.is_empty() && panicked == 0,
        "{panicked} thread(s) panicked; {} mismatch(es):\n{}",
        errs.len(),
        errs.join("\n")
    );

    rlx_ir::env::unset("RLX_METAL_SGEMM_MPS");
}
