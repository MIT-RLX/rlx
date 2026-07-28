// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Row-softmax through the rlx→AIE-MLIR compiler: rlx emits a numerically-stable
// per-row softmax over [rows, cols] (max-reduce, Σexp-reduce, normalize — all in
// pure-arith), compiles it (native aiecc), runs it on the NPU, and checks it vs a
// CPU reference within tolerance. First row-reduction op on the NPU seam.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. ROWS=32 COLS=64 \
//     cargo run -p rlx-xdna --features xrt --example xdna_softmax

use rlx_xdna::aie::emit_softmax;
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuIoF32;

fn env(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

fn main() {
    let (rows, cols) = (env("ROWS", "32"), env("COLS", "64"));
    let n = rows * cols;
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");

    // 1) emit + 2) compile
    let mlir = emit_softmax(rows, cols);
    println!(
        "1. rlx emitted AIE-MLIR softmax {rows}x{cols} ({} lines)",
        mlir.lines().count()
    );
    let tmp = "/tmp/rlx_softmax";
    std::fs::create_dir_all(tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    })
    .expect("compile");
    println!("2. compiled via aiecc");
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 3) input with a spread of magnitudes (incl. negatives) per row.
    let input: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();

    // 4) run + validate vs CPU softmax (max-subtract stable reference).
    let io = NpuIoF32::open("", &xclbin, &insts, n).expect("open");
    let out = io.run(&input).expect("run");
    let mut cref = vec![0f32; n];
    for r in 0..rows {
        let row = &input[r * cols..r * cols + cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for c in 0..cols {
            let e = (row[c] - m).exp();
            cref[r * cols + c] = e;
            s += e;
        }
        for c in 0..cols {
            cref[r * cols + c] /= s;
        }
    }
    let mut maxrel = 0.0f32;
    for i in 0..n {
        maxrel = maxrel.max((out[i] - cref[i]).abs() / cref[i].abs().max(1e-4));
    }
    if maxrel.is_nan() || maxrel > 3e-3 {
        println!("3. NPU softmax {rows}x{cols}: FAIL ✗  max-rel-err {maxrel:.2e}");
        std::process::exit(1);
    }
    println!("3. ran on NPU: PASS ✓  max-rel-err {maxrel:.2e} (softmax {rows}x{cols})");
}
