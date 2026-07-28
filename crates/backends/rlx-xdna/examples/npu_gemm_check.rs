// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Verify + benchmark rlx running an INT8 GEMM on the XDNA NPU (persistent
// context: cold `open` once, warm `run` loop), checked vs a CPU reference.
//
//   RLX_XDNA_SHIM=<librlx_xdna_shim.so> XCLBIN=<final_*.xclbin> \
//   INSTS=<insts_*.bin> M=512 K=512 N=512 ITERS=50 \
//   cargo run -p rlx-xdna --features xrt --example npu_gemm_check

use rlx_xdna::npu_gemm::NpuGemm;
use std::time::Instant;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let xclbin = std::env::var("XCLBIN").expect("set XCLBIN");
    let insts_path = std::env::var("INSTS").expect("set INSTS");
    let (m, k, n): (usize, usize, usize) = (
        env("M", "512").parse().unwrap(),
        env("K", "512").parse().unwrap(),
        env("N", "512").parse().unwrap(),
    );
    let iters: usize = env("ITERS", "50").parse().unwrap();
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .expect("read insts")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Deterministic small INT8 operands (small so i32 accumulation is exact).
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i % 5) as i8) - 2).collect();

    // CPU reference: C = A · B in i32.
    let mut cref = vec![0i32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk] as i32;
            if av == 0 {
                continue;
            }
            for j in 0..n {
                cref[i * n + j] += av * b[kk * n + j] as i32;
            }
        }
    }

    // ── cold: one-time context setup (device/xclbin/hw_context/kernel/BOs) ──
    let t = Instant::now();
    let gemm = NpuGemm::open("", &xclbin, &insts, m, k, n).expect("NpuGemm::open");
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;

    // ── warm: reuse the context; this is the real per-call NPU latency ──
    let c = gemm.run(&a, &b).expect("run"); // warm-up + correctness sample
    let mut best_us = f64::MAX;
    let mut sum_us = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = gemm.run(&a, &b).expect("run");
        let us = t.elapsed().as_secs_f64() * 1e6;
        best_us = best_us.min(us);
        sum_us += us;
    }
    let avg_us = sum_us / iters as f64;

    let mism = c.iter().zip(&cref).filter(|(x, y)| x != y).count();
    if mism != 0 {
        let bad: Vec<_> = c
            .iter()
            .zip(&cref)
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .take(3)
            .collect();
        println!("NPU INT8 GEMM {m}x{k}x{n}: FAIL ✗ — {mism} mismatches, first: {bad:?}");
        std::process::exit(1);
    }
    let flops = 2.0 * (m * k * n) as f64;
    println!("NPU INT8 GEMM {m}x{k}x{n}: PASS ✓ (bit-exact vs CPU)");
    println!("  cold open (one-time setup): {cold_ms:.0} ms");
    println!(
        "  warm run (persistent ctx): avg {avg_us:.0} us / best {best_us:.0} us  →  \
         {:.0} / {:.0} GOP/s  (over {iters} iters)",
        flops / (avg_us * 1e3),
        flops / (best_us * 1e3),
    );
}
