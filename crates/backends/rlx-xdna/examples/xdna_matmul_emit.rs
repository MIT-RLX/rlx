// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// rlx EMITS an i32 matmul AIE-MLIR kernel (single-tile scalar core), compiles it
// Python-free (native aiecc), runs it on the NPU via XRT, checks bit-exact vs a
// CPU reference, and benchmarks it. This is the matmul milestone on the
// rlx→AIE-MLIR compiler seam (a vectorized microkernel is the perf follow-on).
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. M=32 K=32 N=32 ITERS=50 \
//     cargo run -p rlx-xdna --features xrt --example xdna_matmul_emit

use rlx_xdna::aie::emit_matmul;
use rlx_xdna::npu_gemm::NpuGemm;
use std::time::Instant;

fn env(k: &str, d: &str) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| d.parse().unwrap())
}

fn main() {
    let (m, k, n) = (env("M", "32"), env("K", "32"), env("N", "32"));
    let iters = env("ITERS", "50");

    // 1) rlx EMITS the matmul kernel.
    let mlir = emit_matmul(m, k, n);
    println!("1. rlx emitted AIE-MLIR i8 matmul {m}x{k}x{n} ({} lines)", mlir.lines().count());

    // 2) COMPILE via native aiecc (no Python).
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");
    let tmp = "/tmp/rlx_mm_emit";
    std::fs::create_dir_all(tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/mm.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    let t = Instant::now();
    rlx_xdna::compile::compile_overlay(&rlx_xdna::compile::OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    })
    .expect("compile");
    println!("2. compiled via aiecc in {:.1}s", t.elapsed().as_secs_f64());
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 3) i8 operands (the AIE2 MAC is int8); small values so the i32 accum is exact.
    let a: Vec<i8> = (0..m * k).map(|i| (i % 5) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 3) as i8).collect();
    let mut cref = vec![0i32; m * n];
    for i in 0..m {
        for l in 0..k {
            let av = a[i * k + l] as i32;
            for j in 0..n {
                cref[i * n + j] += av * b[l * n + j] as i32;
            }
        }
    }

    // 4) run on NPU + validate.
    let mm = NpuGemm::open("", &xclbin, &insts, m, k, n).expect("NpuGemm::open");
    let c = mm.run(&a, &b).expect("run");
    let mism = (0..m * n).filter(|&i| c[i] != cref[i]).count();
    if mism != 0 {
        let bad: Vec<_> = (0..m * n).filter(|&i| c[i] != cref[i]).take(3).map(|i| (i, c[i], cref[i])).collect();
        println!("3. NPU matmul {m}x{k}x{n}: FAIL ✗ — {mism} mismatches (i,got,want): {bad:?}");
        std::process::exit(1);
    }
    println!("3. ran on NPU: PASS ✓ bit-exact vs CPU (i32 matmul {m}x{k}x{n})");

    // 5) warm bench.
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mm.run(&a, &b).expect("run");
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let flops = 2.0 * (m * k * n) as f64;
    // Scalar i8 MAC core (vectorized aievec.matmul blocked by Peano accumulator drain).
    println!("4. warm run best {best:.1} us  →  {:.2} GOP/s (scalar i8 MAC; vectorized = C++ aie::mmul)", flops / (best * 1e3));
}
