// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// FUSION on the NPU: rlx emits `relu(w*x + b)` as ONE fused AIE-MLIR kernel
// (one dispatch, one DMA round-trip) and compares it to running the three ops
// (mul, add, relu) as SEPARATE kernels (three dispatches, three round-trips) —
// the rlx compiler advantage a prebuilt-overlay runtime can't get.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. N=262144 CHUNK=2048 ITERS=100 \
//     cargo run -p rlx-xdna --features xrt --example xdna_fused

use rlx_xdna::aie::{emit_eltwise, emit_eltwise_chain, Eltwise};
use rlx_xdna::npu_gemm::NpuIo;
use std::time::Instant;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn build(name: &str, mlir: &str) -> (String, Vec<u32>) {
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");
    let tmp = format!("/tmp/rlx_fuse_{name}");
    std::fs::create_dir_all(&tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    rlx_xdna::compile::compile_overlay(&rlx_xdna::compile::OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    })
    .expect("compile");
    let insts = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (xclbin, insts)
}

fn bench(io: &NpuIo, input: &[i32], iters: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = io.run(input).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    best
}

fn main() {
    let n: usize = env("N", "262144").parse().unwrap();
    let chunk: usize = env("CHUNK", "2048").parse().unwrap();
    let iters: usize = env("ITERS", "100").parse().unwrap();
    let (w, b) = (3i32, 5i32);
    let ops = [Eltwise::MulScalar(w), Eltwise::AddScalar(b), Eltwise::Relu];
    let input: Vec<i32> = (0..n).map(|i| (i as i32 % 21) - 10).collect();
    let want = |x: i32| (x.wrapping_mul(w).wrapping_add(b)).max(0); // relu(w*x+b)

    println!("relu({w}*x + {b}) over {n} i32 ({}x{chunk}-chunks)\n", n / chunk);

    // ── FUSED: one kernel ──────────────────────────────────────────────────
    let (xf, instf) = build("fused", &emit_eltwise_chain(n, chunk, &ops));
    let iof = NpuIo::open("", &xf, &instf, n).expect("open fused");
    let of = iof.run(&input).unwrap();
    let ok_f = (0..n).all(|i| of[i] == want(input[i]));
    let t_fused = bench(&iof, &input, iters);
    println!("FUSED    (1 kernel, 1 dispatch):  {t_fused:.1} us   [{}]", if ok_f { "bit-exact ✓" } else { "FAIL ✗" });

    // ── SEPARATE: three kernels, host-chained (3 dispatches, 3 round-trips) ──
    let (xm, instm) = build("mul", &emit_eltwise(n, chunk, Eltwise::MulScalar(w)));
    let (xa, insta) = build("add", &emit_eltwise(n, chunk, Eltwise::AddScalar(b)));
    let (xr, instr) = build("relu", &emit_eltwise(n, chunk, Eltwise::Relu));
    let iom = NpuIo::open("", &xm, &instm, n).unwrap();
    let ioa = NpuIo::open("", &xa, &insta, n).unwrap();
    let ior = NpuIo::open("", &xr, &instr, n).unwrap();
    let run_sep = |x: &[i32]| ior.run(&ioa.run(&iom.run(x).unwrap()).unwrap()).unwrap();
    let os = run_sep(&input);
    let ok_s = (0..n).all(|i| os[i] == want(input[i]));
    let mut t_sep = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = run_sep(&input);
        t_sep = t_sep.min(t.elapsed().as_secs_f64() * 1e6);
    }
    println!("SEPARATE (3 kernels, 3 dispatch):  {t_sep:.1} us   [{}]", if ok_s { "bit-exact ✓" } else { "FAIL ✗" });

    println!("\nfusion speedup: {:.2}x  ({:.1} us saved / call)", t_sep / t_fused, t_sep - t_fused);
}
