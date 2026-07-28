// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Affine RMSNorm (out = x·rsqrt(mean(x²)+eps)·gamma + beta) through the
// rlx→AIE-MLIR compiler, run on the NPU via the generic 3-buffer NpuRun3
// (x, gamma‖beta packed, out) and checked vs CPU. Validates the multi-input path.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. ROWS=32 COLS=64 \
//     cargo run -p rlx-xdna --features xrt --example xdna_rmsnorm_affine

use rlx_xdna::aie::emit_rms_norm_affine;
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuRun3;

fn envn(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

fn main() {
    let (rows, cols) = (envn("ROWS", "32"), envn("COLS", "64"));
    let n = rows * cols;
    let eps = 1e-5f32;
    let (aiecc, peano) = (
        std::env::var("AIECC").unwrap(),
        std::env::var("PEANO").unwrap(),
    );

    let mlir = emit_rms_norm_affine(rows, cols, eps);
    println!(
        "1. rlx emitted AIE-MLIR affine rms_norm {rows}x{cols} ({} lines)",
        mlir.lines().count()
    );
    let tmp = "/tmp/rlx_rmsaff";
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
    println!("2. compiled");
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // inputs
    let x: Vec<f32> = (0..n)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.2 + 0.5)
        .collect();
    let gamma: Vec<f32> = (0..cols).map(|c| 0.5 + (c % 5) as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..cols).map(|c| ((c % 7) as f32 - 3.0) * 0.05).collect();
    let mut gb = gamma.clone();
    gb.extend_from_slice(&beta); // packed: gamma then beta

    // 3) run on NPU
    let io = NpuRun3::open("", &xclbin, &insts, n, 2 * cols, n).expect("open");
    let out = io.run(&x, &gb).expect("run");

    // reference
    let mut cref = vec![0f32; n];
    for r in 0..rows {
        let row = &x[r * cols..r * cols + cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            cref[r * cols + c] = row[c] * inv * gamma[c] + beta[c];
        }
    }
    let mut maxrel = 0.0f32;
    for i in 0..n {
        maxrel = maxrel.max((out[i] - cref[i]).abs() / cref[i].abs().max(1e-3));
    }
    if maxrel.is_nan() || maxrel > 3e-3 {
        println!("3. NPU affine rms_norm {rows}x{cols}: FAIL ✗  max-rel-err {maxrel:.2e}");
        std::process::exit(1);
    }
    println!("3. ran on NPU: PASS ✓  max-rel-err {maxrel:.2e} (affine rms_norm {rows}x{cols})");
}
