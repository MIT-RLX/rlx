// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Row RMSNorm / LayerNorm through the rlx→AIE-MLIR compiler (pure-arith reduction
// + rsqrt), run on the NPU and checked vs a CPU reference. Normalization core
// (no affine gamma/beta yet).
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. ROWS=32 COLS=64 \
//     cargo run -p rlx-xdna --features xrt --example xdna_norm

use rlx_xdna::aie::{emit_layer_norm, emit_rms_norm};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuIoF32;

fn envn(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

fn run(name: &str, mlir: String, rows: usize, cols: usize, cref: &[f32], input: &[f32]) -> bool {
    let n = rows * cols;
    let (aiecc, peano) = (
        std::env::var("AIECC").unwrap(),
        std::env::var("PEANO").unwrap(),
    );
    let tmp = format!("/tmp/rlx_norm_{name}");
    std::fs::create_dir_all(&tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    if let Err(e) = compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    }) {
        println!(
            "  {name:<10} COMPILE-FAIL ({})",
            format!("{e:?}").lines().next().unwrap_or("")
        );
        return false;
    }
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let io = NpuIoF32::open("", &xclbin, &insts, n).expect("open");
    let out = io.run(input).expect("run");
    let mut maxrel = 0.0f32;
    for i in 0..n {
        maxrel = maxrel.max((out[i] - cref[i]).abs() / cref[i].abs().max(1e-3));
    }
    if maxrel.is_nan() || maxrel > 3e-3 {
        println!("  {name:<10} FAIL ✗  max-rel-err {maxrel:.2e}");
        false
    } else {
        println!("  {name:<10} PASS ✓  max-rel-err {maxrel:.2e}");
        true
    }
}

fn main() {
    let (rows, cols) = (envn("ROWS", "32"), envn("COLS", "64"));
    let n = rows * cols;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.2 + 0.5)
        .collect();

    // rms reference
    let mut rms = vec![0f32; n];
    for r in 0..rows {
        let row = &input[r * cols..r * cols + cols];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            rms[r * cols + c] = row[c] * inv;
        }
    }
    // layernorm reference
    let mut ln = vec![0f32; n];
    for r in 0..rows {
        let row = &input[r * cols..r * cols + cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            ln[r * cols + c] = (row[c] - mean) * inv;
        }
    }

    println!("row-reduction norms {rows}x{cols}, eps={eps:e}\n");
    let a = run(
        "rms_norm",
        emit_rms_norm(rows, cols, eps),
        rows,
        cols,
        &rms,
        &input,
    );
    let b = run(
        "layer_norm",
        emit_layer_norm(rows, cols, eps),
        rows,
        cols,
        &ln,
        &input,
    );
    if !(a && b) {
        std::process::exit(1);
    }
}
