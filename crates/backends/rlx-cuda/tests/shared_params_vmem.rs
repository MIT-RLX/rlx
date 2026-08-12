// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Invariants of the shared parameter region (`RLX_CUDA_SHARED_PARAMS`,
//! [`rlx_cuda::vmem`]).
//!
//! The region is process-wide and name-addressed, so the risks are all about
//! *sharing between compiles* rather than any single graph:
//!
//!   * two models alive at once must not clobber each other's slots,
//!   * a param NAME reused at a different size must get a distinct slot
//!     (keying on name alone would alias two different tensors),
//!   * concurrent compiles must not corrupt the bump allocator, and
//!   * a param uploaded by one executable must be visible to the next, since
//!     `set_param` deliberately SKIPS re-uploading it (that skip is the dedup —
//!     if the sharing were broken, the second executable would silently read
//!     zeros).
//!
//! Every test no-ops on a CUDA-less host, like the rest of the crate's suite.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// The sharing scope is process-global (a model identity, not a per-thread
/// notion), so these tests must not run concurrently with each other — one
/// test's `set_scope` would otherwise land mid-way through another's compiles.
/// Serialise the file rather than requiring `--test-threads=1` at the call site.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `y = x * w` with `w` a named parameter — the smallest graph that exercises a
/// param slot end to end.
fn scale_graph(graph_name: &str, param: &str, n: usize) -> Graph {
    let mut g = Graph::new(graph_name);
    let shape = Shape::new(&[n], DType::F32);
    let x = g.input("x", shape.clone());
    let w = g.param(param, shape.clone());
    let y = g.binary(BinaryOp::Mul, x, w, shape);
    g.set_outputs(vec![y]);
    g
}

fn run_scale(param: &str, n: usize, w: &[f32], x: &[f32]) -> Vec<f32> {
    let g = scale_graph("scale", param, n);
    let mut exe = Session::new(Device::Cuda).compile(g);
    exe.set_param(param, w);
    exe.run(&[("x", x)])[0].clone()
}

fn expected(x: &[f32], w: &[f32]) -> Vec<f32> {
    x.iter().zip(w).map(|(a, b)| a * b).collect()
}

/// Two "models" (disjoint param names) compiled and run in the same process.
/// Both must see their own weights — a shared region that mixed slots up would
/// give one of them the other's numbers.
#[test]
fn two_models_in_one_process_keep_their_own_params() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);
    rlx_cuda::vmem::set_scope_from_label("two_models_test");

    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let wa = [10.0_f32, 20.0, 30.0, 40.0];
    let wb = [-1.0_f32, -2.0, -3.0, -4.0];

    let before = rlx_cuda::vmem::slots_allocated();
    let a1 = run_scale("model_a.w", 4, &wa, &x);
    assert!(
        rlx_cuda::vmem::slots_allocated() > before,
        "shared region handed out no slots — the test ran on the private-arena \
         fallback and would prove nothing"
    );
    let b1 = run_scale("model_b.w", 4, &wb, &x);
    // Re-run A after B compiled: A's slot must be untouched by B's allocation.
    let a2 = run_scale("model_a.w", 4, &wa, &x);

    assert_eq!(a1, expected(&x, &wa), "model A first run");
    assert_eq!(b1, expected(&x, &wb), "model B run");
    assert_eq!(a2, expected(&x, &wa), "model A after model B compiled");
}

/// The same param NAME at two different sizes. Keying the region on name alone
/// would hand both the same offset and silently truncate or overrun one.
#[test]
fn same_param_name_different_size_gets_distinct_slots() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);
    rlx_cuda::vmem::set_scope_from_label("collide_test");

    let x4 = [1.0_f32, 2.0, 3.0, 4.0];
    let w4 = [2.0_f32, 2.0, 2.0, 2.0];
    let x8: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let w8 = vec![3.0_f32; 8];

    let before = rlx_cuda::vmem::slots_allocated();
    let small = run_scale("collide.w", 4, &w4, &x4);
    assert!(
        rlx_cuda::vmem::slots_allocated() > before,
        "shared path not taken"
    );
    let large = run_scale("collide.w", 8, &w8, &x8);
    // And back to the small one — its slot must not have been overwritten by
    // the larger tensor sharing the name.
    let small_again = run_scale("collide.w", 4, &w4, &x4);

    assert_eq!(small, expected(&x4, &w4), "4-elem 'collide.w'");
    assert_eq!(large, expected(&x8, &w8), "8-elem 'collide.w'");
    assert_eq!(
        small_again,
        expected(&x4, &w4),
        "4-elem after 8-elem compiled"
    );
}

/// A second executable that declares an ALREADY-uploaded param must still read
/// the right numbers: `set_param` skips the re-upload, so this is the test that
/// actually proves the physical pages are shared rather than merely aliased in
/// bookkeeping.
#[test]
fn second_executable_sees_params_uploaded_by_the_first() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);
    rlx_cuda::vmem::set_scope_from_label("dedup_test");

    let x = [5.0_f32, 6.0, 7.0, 8.0];
    let w = [1.5_f32, 2.5, 3.5, 4.5];

    let before = rlx_cuda::vmem::slots_allocated();
    let first = run_scale("dedup.w", 4, &w, &x);
    assert!(
        rlx_cuda::vmem::slots_allocated() > before,
        "shared path not taken"
    );
    assert_eq!(first, expected(&x, &w), "first executable");

    // Same param, fresh executable. `set_param` will no-op because the region
    // already holds these bytes; the result must be unchanged.
    let g = scale_graph("scale2", "dedup.w", 4);
    let mut exe = Session::new(Device::Cuda).compile(g);
    exe.set_param("dedup.w", &w);
    let second = exe.run(&[("x", &x[..])])[0].clone();
    assert_eq!(
        second,
        expected(&x, &w),
        "second executable (upload skipped)"
    );

    // Even with a deliberately WRONG upload, the skip means the region's
    // original bytes win — documents the dedup's actual semantics.
    let g = scale_graph("scale3", "dedup.w", 4);
    let mut exe = Session::new(Device::Cuda).compile(g);
    exe.set_param("dedup.w", &[0.0_f32; 4]);
    let third = exe.run(&[("x", &x[..])])[0].clone();
    assert_eq!(
        third,
        expected(&x, &w),
        "skipped upload must leave the shared bytes intact"
    );
}

/// Concurrent compiles hammer the region's bump allocator and its growth path
/// from several threads at once.
#[test]
fn concurrent_compiles_are_consistent() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);
    rlx_cuda::vmem::set_scope_from_label("concurrent_test");

    let before = rlx_cuda::vmem::slots_allocated();
    const THREADS: usize = 8;
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            std::thread::spawn(move || {
                let n = 4 + t; // distinct sizes -> distinct slots
                let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
                let w: Vec<f32> = (0..n).map(|_| (t + 1) as f32).collect();
                let got = run_scale(&format!("thread_{t}.w"), n, &w, &x);
                assert_eq!(got, expected(&x, &w), "thread {t}");
            })
        })
        .collect();
    for (t, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("thread {t} panicked"));
    }
    // Every thread used a distinct (name, size), so the region must have grown
    // by at least THREADS slots — proves the compiles really went through the
    // shared allocator and that concurrent `slot()` calls did not lose any.
    // `>=` not `==`: the counter is process-global and the harness runs the
    // other tests in this file on parallel threads, so they contribute too.
    let grew = rlx_cuda::vmem::slots_allocated() - before;
    assert!(
        grew >= THREADS,
        "concurrent compiles lost slots: region grew by {grew}, expected >= {THREADS}"
    );
}

/// The soundness property that scoping exists for: two DIFFERENT models that
/// happen to declare the same param name at the same size must not alias.
///
/// This reproduces `mlx_dequant_matmul_parity`'s collision in miniature —
/// `run_affine` and `run_mxfp` both declare `"w"` of exactly 256 bytes with
/// different bytes. Before scoping, the second upload was skipped and the second
/// model silently read the first model's weights.
#[test]
fn different_scopes_do_not_alias_same_name_and_size() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);

    let x = [1.0_f32, 1.0, 1.0, 1.0];
    let w_a = [7.0_f32, 7.0, 7.0, 7.0];
    let w_b = [-3.0_f32, -3.0, -3.0, -3.0];

    // Same param name, same size, different weights, different models.
    rlx_cuda::vmem::set_scope_from_label("model_one");
    let a = run_scale("shared.name", 4, &w_a, &x);
    rlx_cuda::vmem::set_scope_from_label("model_two");
    let b = run_scale("shared.name", 4, &w_b, &x);
    // ...and back, to be sure model one's slot survived model two.
    rlx_cuda::vmem::set_scope_from_label("model_one");
    let a2 = run_scale("shared.name", 4, &w_a, &x);

    assert_eq!(a, expected(&x, &w_a), "model one");
    assert_eq!(
        b,
        expected(&x, &w_b),
        "model two must NOT see model one's weights"
    );
    assert_eq!(a2, expected(&x, &w_a), "model one after model two");
}

/// Scope 0 is the default and must disable sharing outright, so a process that
/// never declares a model identity can never alias two tensors.
#[test]
fn scope_zero_disables_sharing() {
    let _serial = serial();
    if !rlx_cuda::is_available() {
        return;
    }
    rlx_cuda::vmem::set_enabled_for_test(true);
    rlx_cuda::vmem::set_scope(0);
    assert!(
        !rlx_cuda::vmem::enabled(),
        "scope 0 must fall back to a private arena even with the flag on"
    );

    let x = [2.0_f32, 3.0, 4.0, 5.0];
    let w = [1.0_f32, 2.0, 3.0, 4.0];
    let before = rlx_cuda::vmem::slots_allocated();
    let got = run_scale("unscoped.w", 4, &w, &x);
    assert_eq!(got, expected(&x, &w));
    assert_eq!(
        rlx_cuda::vmem::slots_allocated(),
        before,
        "unscoped compile must not touch the shared region"
    );
}
