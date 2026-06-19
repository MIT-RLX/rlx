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

//! Demonstrates the per-shape compile cost and how `CompileCache`
//! amortizes it for variable-shape callers.
//!
//! cargo run --release --example compile_cache_demo --features metal -p rlx-runtime

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_runtime::{CompileCache, Device, Session};
use std::collections::HashSet;
use std::time::Instant;

// Toy "model": small FFN-like graph, shape varies with seq.
fn build(seq: usize) -> Graph {
    let mut g = Graph::new("toy");
    let f = DType::F32;
    let h = 256usize;
    let int_dim = 1024usize;
    let x = g.input("x", Shape::new(&[seq, h], f));
    let w1 = g.param("w1", Shape::new(&[h, int_dim], f));
    let w2 = g.param("w2", Shape::new(&[int_dim, h], f));
    let mm1 = g.matmul(x, w1, Shape::new(&[seq, int_dim], f));
    let act = g.activation(Activation::Silu, mm1, Shape::new(&[seq, int_dim], f));
    let mm2 = g.matmul(act, w2, Shape::new(&[seq, h], f));
    let res = g.binary(BinaryOp::Add, mm2, x, Shape::new(&[seq, h], f));
    g.set_outputs(vec![res]);
    g
}

fn det(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 % 17.0) * 0.01 - 0.05).collect()
}

#[cfg(target_os = "macos")]
fn main() {
    // Realistic mix: requests cycle through a small set of common seq lengths.
    let calls: Vec<usize> = (0..200)
        .map(|i| match i % 4 {
            0 => 8,
            1 => 16,
            2 => 32,
            _ => 64,
        })
        .collect();

    let h = 256;
    let int_dim = 1024;
    let w1 = det(h * int_dim);
    let w2 = det(int_dim * h);

    // ── Without cache: recompile every call ──
    let t0 = Instant::now();
    for &seq in &calls {
        let session = Session::new(Device::Metal);
        let mut compiled = session.compile(build(seq));
        compiled.set_param("w1", &w1);
        compiled.set_param("w2", &w2);
        let x_data = det(seq * h);
        let _ = compiled.run(&[("x", &x_data)]);
    }
    let no_cache = t0.elapsed();

    // ── With cache: compile + load params once per unique shape ──
    let mut cache = CompileCache::new(Device::Metal, 8);
    let mut params_loaded: HashSet<u64> = HashSet::new();
    let t0 = Instant::now();
    for &seq in &calls {
        let key = seq as u64;
        let first_time = !params_loaded.contains(&key);
        let compiled = cache.get_or_compile(key, || build(seq));
        if first_time {
            compiled.set_param("w1", &w1);
            compiled.set_param("w2", &w2);
            params_loaded.insert(key);
        }
        let x_data = det(seq * h);
        let _ = compiled.run(&[("x", &x_data)]);
    }
    let with_cache = t0.elapsed();

    println!("Calls: {} (mixed seq lengths: 8, 16, 32, 64)", calls.len());
    println!(
        "  no cache    : {:>8} ms total ({:.2} ms/call)",
        no_cache.as_millis(),
        no_cache.as_secs_f64() * 1000.0 / calls.len() as f64
    );
    println!(
        "  with cache  : {:>8} ms total ({:.2} ms/call)",
        with_cache.as_millis(),
        with_cache.as_secs_f64() * 1000.0 / calls.len() as f64
    );
    let speedup = no_cache.as_secs_f64() / with_cache.as_secs_f64();
    println!(
        "  speedup     : {speedup:.2}x  (cached unique shapes: {})",
        cache.len()
    );
    println!();
    println!("(no-cache repeats compile + memory-plan + arena alloc + param");
    println!(" copy on every call. Cache hits skip all of that — the call");
    println!(" cost collapses to encode + commit + wait.)");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("compile_cache_demo requires macOS");
}
