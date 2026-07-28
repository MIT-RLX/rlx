// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Run an INT8 GEMM on the XDNA NPU through the DIRECT amdxdna ioctl path — no
// XRT, no C++ shim. rlx parses the xclbin PDI itself, drives CREATE_HWCTX /
// CONFIG_HWCTX / CREATE_BO / EXEC_CMD / SYNCOBJ_WAIT by hand, and checks the
// result bit-exact against a CPU reference.
//
//   XCLBIN=<final_*.xclbin> INSTS=<insts_*.bin> M=512 K=512 N=512 ITERS=50 \
//     cargo run -p rlx-xdna --features direct --example direct_gemm

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("direct_gemm is Linux-only (amdxdna)");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    use rlx_xdna::direct::Gemm;
    use std::time::Instant;

    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let xclbin_path = std::env::var("XCLBIN").expect("set XCLBIN");
    let insts_path = std::env::var("INSTS").expect("set INSTS");
    let (m, k, n): (usize, usize, usize) = (
        env("M", "512").parse().unwrap(),
        env("K", "512").parse().unwrap(),
        env("N", "512").parse().unwrap(),
    );
    let iters: usize = env("ITERS", "50").parse().unwrap();

    let xclbin = std::fs::read(&xclbin_path).expect("read xclbin");
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .expect("read insts")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Sanity-check the PDI extractor against the ground truth before we exec.
    match rlx_xdna::direct::axlf::parse(&xclbin) {
        Ok(p) => {
            let sum: u64 = p.pdi.iter().map(|&b| b as u64).sum();
            let head: Vec<String> = p.pdi.iter().take(8).map(|b| format!("{b:02x}")).collect();
            println!(
                "axlf: PDI {} bytes, column_width {} → num_tiles {}; PDI sum=0x{sum:x} head={}",
                p.pdi.len(),
                p.column_width,
                4 * p.column_width,
                head.join("")
            );
        }
        Err(e) => {
            eprintln!("axlf parse FAILED: {e}");
            std::process::exit(1);
        }
    }

    // Deterministic small INT8 operands (small so i32 accumulation is exact).
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i % 5) as i8) - 2).collect();

    // CPU reference.
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

    let t = Instant::now();
    let gemm = Gemm::open(&xclbin, &insts, m, k, n).expect("Gemm::open (direct)");
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;

    // Run once; on the exec hang, dump the firmware-side hwctx state (did the
    // command reach the firmware / complete / error, and where did the partition
    // land) before the context is torn down.
    let c = match gemm.run(&a, &b) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("run failed: {e}");
            if let Ok(rep) = gemm.hwctx_report() {
                eprintln!("=== firmware hwctx state ===\n{rep}");
            }
            std::process::exit(1);
        }
    };
    let mism = c.iter().zip(&cref).filter(|(x, y)| x != y).count();
    if mism != 0 {
        let bad: Vec<_> = c
            .iter()
            .zip(&cref)
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .take(3)
            .collect();
        println!("DIRECT NPU INT8 GEMM {m}x{k}x{n}: FAIL ✗ — {mism} mismatches, first: {bad:?}");
        std::process::exit(1);
    }

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
    let flops = 2.0 * (m * k * n) as f64;
    println!("DIRECT NPU INT8 GEMM {m}x{k}x{n}: PASS ✓ (bit-exact vs CPU, no XRT)");
    println!("  cold open (parse+hwctx+CU+BOs): {cold_ms:.0} ms");
    println!(
        "  warm run (EXEC_CMD+syncobj): avg {avg_us:.0} us / best {best_us:.0} us  →  {:.0} / {:.0} GOP/s",
        flops / (avg_us * 1e3),
        flops / (best_us * 1e3),
    );
}
