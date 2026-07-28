// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-cache behavior. Each `#[test]` runs on its own thread, so the
//! thread-local cache starts empty. Run:
//! `cargo test -p rlx-tensor --features transforms,eval`.
#![cfg(all(feature = "eval", feature = "transforms"))]

use rlx_tensor::{Func, Tensor, cache_stats, clear_cache, shape};

#[test]
fn func_run_is_jit() {
    clear_cache();
    let f = Func::new("sq", |s| {
        let x = s.input("x", shape![3]);
        (&x * &x).sum([0], false)
    });
    // Three runs with different inputs -> compile once, reuse twice.
    let a = f.run(&[("x", &[1.0, 2.0, 3.0])]);
    let b = f.run(&[("x", &[2.0, 2.0, 2.0])]);
    let c = f.run(&[("x", &[0.0, 0.0, 4.0])]);
    assert_eq!(a[0], vec![14.0]);
    assert_eq!(b[0], vec![12.0]);
    assert_eq!(c[0], vec![16.0]);

    let (hits, misses) = cache_stats();
    assert_eq!(misses, 1, "should compile exactly once");
    assert_eq!(hits, 2, "subsequent runs should hit the cache");
}

#[test]
fn repeated_tensor_eval_reuses() {
    clear_cache();
    // Rebuild a structurally + value identical graph each iteration.
    for _ in 0..5 {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = Tensor::ones([3]);
        assert_eq!((&a + &b).to_vec(), vec![2.0, 3.0, 4.0]);
    }
    let (hits, misses) = cache_stats();
    assert_eq!(misses, 1, "identical graphs compile once");
    assert_eq!(hits, 4);
}

#[test]
fn repeated_to_vec_same_tensor_is_clone_free() {
    clear_cache();
    // Big-ish constant: the old path deep-copied these bytes on every to_vec
    // (output_graph clone). Now the graph is borrowed for the fingerprint and
    // cloned only on the first (miss) compile.
    let a = Tensor::from_vec(vec![1.0; 10_000], [10_000]);
    let t = a.relu();
    for _ in 0..5 {
        let v = t.to_vec();
        assert_eq!(v.len(), 10_000);
    }
    let (hits, misses) = cache_stats();
    assert_eq!(misses, 1, "compile once");
    assert_eq!(hits, 4, "subsequent realizes hit the cache (no clone)");
}

#[test]
fn different_constants_do_not_collide() {
    clear_cache();
    // Same structure, different constant bytes -> must NOT share a compiled
    // graph (else the second result would be wrong).
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let y = Tensor::from_vec(vec![10.0, 20.0, 30.0], [3]);
    assert_eq!(x.relu().to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.relu().to_vec(), vec![10.0, 20.0, 30.0]);

    let (_, misses) = cache_stats();
    assert_eq!(misses, 2, "distinct constants -> distinct cache entries");
}
