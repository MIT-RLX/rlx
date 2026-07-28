// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Row-reduction sweep (sum/mean/max/min/prod) through the rlx→AIE-MLIR compiler,
// run on the NPU and checked vs a CPU reference. The overlay broadcasts the
// per-row reduction across the row; we read column 0.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. ROWS=32 COLS=64 \
//     cargo run -p rlx-xdna --features xrt --example xdna_reduce

use rlx_xdna::aie::{ReduceOp, emit_reduce};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuIoF32;

fn envn(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

fn main() {
    let (rows, cols) = (envn("ROWS", "32"), envn("COLS", "64"));
    let n = rows * cols;
    let (aiecc, peano) = (
        std::env::var("AIECC").unwrap(),
        std::env::var("PEANO").unwrap(),
    );

    // Values near 1.0 so `prod` over `cols` stays well-conditioned.
    let input: Vec<f32> = (0..n).map(|i| 0.95 + (i % 7) as f32 * 0.015).collect();

    let ops = [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ];
    println!("row-reduction sweep {rows}x{cols}\n");
    let (mut pass, mut fail) = (0, 0);
    for op in ops {
        let tmp = format!("/tmp/rlx_red_{}", op.name());
        std::fs::create_dir_all(&tmp).ok();
        let mp = format!("{tmp}/aie.mlir");
        std::fs::write(&mp, emit_reduce(op, rows, cols)).unwrap();
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
                "  {:<5} COMPILE-FAIL ({})",
                op.name(),
                format!("{e:?}").lines().next().unwrap_or("")
            );
            fail += 1;
            continue;
        }
        let insts: Vec<u32> = std::fs::read(&insts_path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let io = NpuIoF32::open("", &xclbin, &insts, n).expect("open");
        let out = io.run(&input).expect("run");
        let mut maxrel = 0.0f32;
        for r in 0..rows {
            let want = op.apply(&input[r * cols..r * cols + cols]);
            let got = out[r * cols]; // column 0 holds the broadcast reduction
            maxrel = maxrel.max((got - want).abs() / want.abs().max(1e-3));
        }
        if maxrel.is_nan() || maxrel > 3e-3 {
            println!("  {:<5} FAIL ✗  max-rel-err {maxrel:.2e}", op.name());
            fail += 1;
        } else {
            println!("  {:<5} PASS ✓  max-rel-err {maxrel:.2e}", op.name());
            pass += 1;
        }
    }
    println!("\n{pass} passed, {fail} failed (of {})", ops.len());
    if fail > 0 {
        std::process::exit(1);
    }
}
