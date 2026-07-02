// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compile-cache integration test for the TPU backend.
//!
//! Verifies that:
//!   1. `Device::Tpu` is registered and instantiable (registry wiring).
//!   2. `CompileCache::get_or_compile` for the same key compiles only
//!      once — the second call returns from cache without re-running
//!      `TpuExecutable::compile`.
//!   3. Compiled cache entries actually execute (catches a regression
//!      where the cache hands back a stale wrapper).
//!
//! Gated on the `tpu` feature **and** `LIBTPU_PATH` (the cache
//! reaches into `Session::new(Tpu).compile`, which dlopen's the
//! plugin). Skips cleanly without either.

#![cfg(feature = "tpu")]

use std::time::Instant;

use rlx_driver::Device;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::CompileCache;

fn skip_without_plugin() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[tpu_compile_cache] LIBTPU_PATH not set — skipping");
        return true;
    }
    false
}

fn build_add_graph() -> Graph {
    let mut g = Graph::new("ck_add");
    let s = Shape::new(&[6], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.binary(BinaryOp::Add, x, y, s);
    g.set_outputs(vec![z]);
    g
}

#[test]
fn tpu_compile_cache_hits() {
    if skip_without_plugin() {
        return;
    }

    let mut cache = CompileCache::new(Device::Tpu, 4);

    // First call — cold compile.
    let t0 = Instant::now();
    {
        let exec = cache.get_or_compile(0xa11ce, build_add_graph);
        let xs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ys: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let outs = exec.run(&[("x", &xs), ("y", &ys)]);
        assert_eq!(
            outs[0],
            vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
            "cold-compile output mismatch"
        );
    }
    let cold_us = t0.elapsed().as_micros();

    assert_eq!(cache.len(), 1, "first miss should add one entry");
    assert!(cache.contains(0xa11ce));

    // Second call with the same key — cache hit, should not recompile.
    let t1 = Instant::now();
    {
        let exec = cache.get_or_compile(0xa11ce, || {
            panic!("build closure should not run on cache hit");
        });
        let xs: Vec<f32> = vec![1.0, 0.0, 0.5, -1.0, 2.0, 100.0];
        let ys: Vec<f32> = vec![0.0, 0.0, 0.5, 1.0, 0.0, -100.0];
        let outs = exec.run(&[("x", &xs), ("y", &ys)]);
        assert_eq!(
            outs[0],
            vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            "warm-cache output mismatch — entry stale?"
        );
    }
    let warm_us = t1.elapsed().as_micros();

    // Sanity: warm exec should be substantially faster than cold
    // compile since it skips XLA compile entirely. The exact factor
    // varies with host load (Docker on Apple Silicon emulation can
    // see ~40 ms compile vs ~8 ms warm exec, i.e. a ~5× ratio); a
    // 2.5× bound is well above noise but still trips a regression
    // where the cache stops hitting.
    assert!(
        warm_us * 5 < cold_us * 2,
        "warm exec {warm_us} µs not << cold compile {cold_us} µs — \
             cache may not be hitting (need warm < cold/2.5)"
    );

    assert_eq!(cache.len(), 1, "no new entry should be added on hit");

    // A different key produces a second compile.
    {
        let _ = cache.get_or_compile(0xb0b, build_add_graph);
    }
    assert_eq!(cache.len(), 2);
    assert!(cache.contains(0xa11ce));
    assert!(cache.contains(0xb0b));
}
