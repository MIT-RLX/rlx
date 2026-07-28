// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// First COMPUTE op through the rlx→AIE-MLIR compiler: rlx emits an AIE-MLIR
// elementwise kernel, compiles it Python-free (native aiecc), runs it on the NPU
// via XRT, and checks it bit-exact vs a CPU reference.
//
//   AIECC=<.../mlir_aie/bin/aiecc> PEANO=<.../llvm-aie> \
//   RLX_XDNA_SHIM=<librlx_xdna_shim.so> N=1024 OP=relu ITERS=50 \
//     cargo run -p rlx-xdna --features xrt --example xdna_eltwise

use rlx_xdna::aie::{Eltwise, emit_eltwise, emit_eltwise_multicol};
use std::time::Instant;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let n: usize = env("N", "1024").parse().unwrap();
    let chunk: usize = env("CHUNK", "1024").parse().unwrap();
    let cols: usize = env("COLS", "1").parse().unwrap();
    let iters: usize = env("ITERS", "50").parse().unwrap();
    let op = match env("OP", "relu").as_str() {
        "relu" => Eltwise::Relu,
        "add" => Eltwise::AddScalar(env("S", "7").parse().unwrap()),
        "mul" => Eltwise::MulScalar(env("S", "3").parse().unwrap()),
        other => panic!("unknown OP '{other}' (relu|add|mul)"),
    };

    // 1) rlx EMITS the AIE-MLIR compute kernel (n streamed via chunk-sized tiles,
    //    split across `cols` compute columns).
    let mlir = if cols > 1 {
        emit_eltwise_multicol(n, chunk, cols, &[op])
    } else {
        emit_eltwise(n, chunk, op)
    };
    println!(
        "1. rlx emitted AIE-MLIR: {} op, {n} i32, {cols} col(s), {}x{chunk}-chunks/col ({} lines)",
        op.name(),
        (n / cols) / chunk,
        mlir.lines().count()
    );

    // 2) COMPILE it with the native aiecc (no Python).
    let aiecc = std::env::var("AIECC").expect("set AIECC (.../mlir_aie/bin/aiecc)");
    let peano = std::env::var("PEANO").expect("set PEANO (.../llvm-aie)");
    let tmp = format!("/tmp/rlx_elt_{}", op.name());
    std::fs::create_dir_all(&tmp).ok();
    let mlir_path = format!("{tmp}/aie.mlir");
    std::fs::write(&mlir_path, &mlir).expect("write mlir");
    let xclbin = format!("{tmp}/elt.xclbin");
    let insts_path = format!("{tmp}/elt_insts.bin");
    let t = Instant::now();
    rlx_xdna::compile::compile_overlay(&rlx_xdna::compile::OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mlir_path,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    })
    .expect("compile_overlay");
    println!(
        "2. compiled via aiecc in {:.1}s → {xclbin}",
        t.elapsed().as_secs_f64()
    );

    let insts: Vec<u32> = std::fs::read(&insts_path)
        .expect("read insts")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 3) input (deterministic, includes negatives so ReLU is exercised).
    let input: Vec<i32> = (0..n).map(|i| (i as i32 % 17) - 8).collect();

    // 4) RUN on the NPU via a PERSISTENT context (warm) + validate.
    let t = Instant::now();
    let io = rlx_xdna::npu_gemm::NpuIo::open("", &xclbin, &insts, n).expect("NpuIo::open");
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    let out = io.run(&input).expect("run on NPU");
    let mism = (0..n).filter(|&i| out[i] != op.apply(input[i])).count();
    if mism != 0 {
        let bad: Vec<_> = (0..n)
            .filter(|&i| out[i] != op.apply(input[i]))
            .take(3)
            .map(|i| (i, input[i], out[i], op.apply(input[i])))
            .collect();
        println!(
            "3. NPU {} {n}: FAIL ✗ — {mism} mismatches (i,in,got,want): {bad:?}",
            op.name()
        );
        std::process::exit(1);
    }
    println!(
        "3. ran on NPU: PASS ✓ bit-exact vs CPU ({} over {n} i32)",
        op.name()
    );

    // 5) warm benchmark (persistent context — just sync-in / dispatch / sync-out).
    let mut best = f64::MAX;
    let mut sum = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = io.run(&input).expect("run");
        let us = t.elapsed().as_secs_f64() * 1e6;
        best = best.min(us);
        sum += us;
    }
    let avg = sum / iters as f64;
    let gbps = |us: f64| (n as f64 * 4.0 * 2.0) / (us * 1e3); // in+out bytes / ns
    println!(
        "4. cold open {cold_ms:.0} ms; warm run avg {avg:.1} us / best {best:.1} us  →  {:.1} / {:.1} GB/s (over {iters})",
        gbps(avg),
        gbps(best)
    );
}
