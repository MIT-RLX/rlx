// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// f32 activation through the rlx→AIE-MLIR compiler: rlx emits a vectorized f32
// ReLU kernel (AIE2 native f32 vector FPU, `arith.maximumf` on vector<16xf32>),
// compiles it Python-free (native aiecc), runs it on the NPU via XRT, and checks
// it bit-exact vs a CPU reference. This is the f32 twin of `xdna_eltwise` — real
// rlx activations are f32 — and the kernel Device::Xdna dispatches for Relu.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. N=1048576 CHUNK=2048 ITERS=100 \
//     cargo run -p rlx-xdna --features xrt --example xdna_relu_f32

use rlx_xdna::aie::{emit_relu_bf16, emit_relu_f32};
use rlx_xdna::npu_gemm::{bf16_to_f32, f32_to_bf16, NpuIoBf16, NpuIoF32};
use std::time::Instant;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let n: usize = env("N", "1048576").parse().unwrap();
    let chunk: usize = env("CHUNK", "2048").parse().unwrap();
    let iters: usize = env("ITERS", "100").parse().unwrap();
    // BF16=1 → the NATIVE vectorized float path (32-wide); else scalar f32.
    let bf16 = env("BF16", "0") == "1";

    // 1) rlx EMITS the ReLU kernel (bf16 = vectorized/native, f32 = scalar).
    let mlir = if bf16 { emit_relu_bf16(n, chunk) } else { emit_relu_f32(n, chunk) };
    println!(
        "1. rlx emitted AIE-MLIR: {} relu, {n} elems, {}x{chunk}-chunks ({} lines)",
        if bf16 { "bf16" } else { "f32" },
        n / chunk,
        mlir.lines().count()
    );

    // 2) COMPILE via native aiecc (no Python).
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");
    let tmp = "/tmp/rlx_relu_f32";
    std::fs::create_dir_all(tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/relu.xclbin");
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

    // 3) input with negatives (so ReLU is exercised) and non-integer f32 values.
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5 - (n as f32) * 0.25) * 1e-3).collect();

    // Reference. ReLU is exact (max(0,x), no rounding), so both dtypes are
    // bit-exact — the only rounding is the host f32→bf16 cast, which the bf16
    // reference applies too (relu(bf16(x)) round-tripped).
    let reference: Vec<f32> = if bf16 {
        input.iter().map(|&x| bf16_to_f32(f32_to_bf16(x)).max(0.0)).collect()
    } else {
        input.iter().map(|&x| x.max(0.0)).collect()
    };
    let ebytes = if bf16 { 2.0 } else { 4.0 };

    // 4) run on NPU (persistent context) + warm bench.
    let bench = |io_run: &dyn Fn(&[f32]) -> Vec<f32>| {
        let (mut best, mut sum) = (f64::MAX, 0.0);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = io_run(&input);
            let us = t.elapsed().as_secs_f64() * 1e6;
            best = best.min(us);
            sum += us;
        }
        (sum / iters as f64, best)
    };
    let (out, cold_ms, avg, best) = if bf16 {
        let t = Instant::now();
        let io = NpuIoBf16::open("", &xclbin, &insts, n).expect("NpuIoBf16::open");
        let cold = t.elapsed().as_secs_f64() * 1e3;
        let out = io.run(&input).expect("run on NPU");
        let (avg, best) = bench(&|x| io.run(x).expect("run"));
        (out, cold, avg, best)
    } else {
        let t = Instant::now();
        let io = NpuIoF32::open("", &xclbin, &insts, n).expect("NpuIoF32::open");
        let cold = t.elapsed().as_secs_f64() * 1e3;
        let out = io.run(&input).expect("run on NPU");
        let (avg, best) = bench(&|x| io.run(x).expect("run"));
        (out, cold, avg, best)
    };

    let ty = if bf16 { "bf16" } else { "f32" };
    let mism = (0..n).filter(|&i| out[i].to_bits() != reference[i].to_bits()).count();
    if mism != 0 {
        let bad: Vec<_> = (0..n)
            .filter(|&i| out[i].to_bits() != reference[i].to_bits())
            .take(3)
            .map(|i| (i, input[i], out[i], reference[i]))
            .collect();
        println!("3. NPU {ty} relu {n}: FAIL ✗ — {mism} mismatches (i,in,got,want): {bad:?}");
        std::process::exit(1);
    }
    println!("3. ran on NPU: PASS ✓ bit-exact vs CPU ({ty} relu over {n} elems)");

    let gbps = |us: f64| (n as f64 * ebytes * 2.0) / (us * 1e3);
    println!(
        "4. cold open {cold_ms:.0} ms; warm run avg {avg:.1} us / best {best:.1} us  →  {:.1} / {:.1} GB/s (over {iters})",
        gbps(avg), gbps(best)
    );
}
