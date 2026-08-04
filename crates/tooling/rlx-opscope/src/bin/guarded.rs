// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-guarded` — the closed loop: a real memoizing-matmul kernel behind a
//! runtime guard, driven by a recurring workload. Measures the speedup (≈ cache
//! hit rate) AND proves the output is **bit-exact** vs a dense reference — so
//! the guard (hash + verify) never returns a wrong answer.

use rlx_ir::Philox4x32;
use rlx_opscope::guard::{MemoMatmul, dense_matmul};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let (m, k, n) = (32usize, 512usize, 512usize);
    let steps = 200usize;
    let pool = 8usize; // 8 distinct inputs replayed → ~96% repeats

    let mut rng = Philox4x32::new(0x6041D);
    let mut w = vec![0f32; k * n];
    rng.fill_normal(&mut w);
    let inputs: Vec<Vec<f32>> = (0..pool)
        .map(|_| {
            let mut v = vec![0f32; m * k];
            rng.fill_normal(&mut v);
            v
        })
        .collect();
    let ref_out: Vec<Vec<f32>> = inputs
        .iter()
        .map(|x| dense_matmul(x, &w, m, k, n))
        .collect();
    let picks: Vec<usize> = (0..steps)
        .map(|_| (rng.next_u32() as usize) % pool)
        .collect();

    // Dense baseline.
    let t = Instant::now();
    for &p in &picks {
        black_box(dense_matmul(&inputs[p], &w, m, k, n));
    }
    let dense_ns = t.elapsed().as_nanos();

    // Guarded memoizing kernel.
    let mut memo = MemoMatmul::new(w.clone(), m, k, n, pool * 2);
    let t = Instant::now();
    let outs: Vec<Vec<f32>> = picks.iter().map(|&p| memo.run(&inputs[p])).collect();
    let memo_ns = t.elapsed().as_nanos();

    // Parity: every guarded output must equal the dense reference.
    let max_err = outs
        .iter()
        .zip(&picks)
        .map(|(o, &p)| {
            o.iter()
                .zip(&ref_out[p])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        })
        .fold(0f32, f32::max);

    println!("MemoMatmul guard — {m}×{k}×{n}, {steps} calls over {pool} distinct inputs");
    println!(
        "  cache hit rate : {:.0}%  ({} hits / {} misses)",
        memo.hit_rate() * 100.0,
        memo.hits,
        memo.misses
    );
    println!("  dense          : {:>7.2} ms", dense_ns as f64 / 1e6);
    println!(
        "  guarded (memo) : {:>7.2} ms   →  {:.1}× speedup",
        memo_ns as f64 / 1e6,
        dense_ns as f64 / memo_ns.max(1) as f64
    );
    println!(
        "  parity vs dense: max_abs {:.2e}   →  {}",
        max_err,
        if max_err == 0.0 {
            "BIT-EXACT ✓ (guard verify is safe)"
        } else {
            "MISMATCH ✗"
        }
    );
}
