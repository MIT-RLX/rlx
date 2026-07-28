// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity sweep for the pure-Rust AIE-MLIR **unary activation** emitter: for each
// `UnaryOp`, rlx emits the f32 (scalar) or bf16 (vectorized) kernel, compiles it
// (native aiecc), runs it on the NPU, and checks it vs a CPU reference within a
// tolerance (transcendentals go through `math.*` → AIE approximations, not
// bit-exact). Reveals which `math.*` ops actually lower on AIE2/Peano.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. N=65536 CHUNK=2048 [BF16=1] \
//     cargo run -p rlx-xdna --features xrt --example xdna_unary

use rlx_xdna::aie::{Ty, UnaryOp, emit_unary};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::{NpuIoBf16, NpuIoF32};

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let n: usize = env("N", "65536").parse().unwrap();
    let chunk: usize = env("CHUNK", "2048").parse().unwrap();
    let bf16 = env("BF16", "0") == "1";
    let ty = if bf16 { Ty::Bf16 } else { Ty::F32 };
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");

    // Domain-restricted ops (log/sqrt/rsqrt/recip) get positive inputs; the rest
    // get a signed spread so the negative branches (sign/floor/elu/…) are tested.
    let pos: Vec<f32> = (0..n).map(|i| 0.1 + (i % 40) as f32 * 0.1).collect();
    let signed: Vec<f32> = (0..n).map(|i| ((i % 71) as f32 - 35.0) * 0.1).collect();
    let needs_pos = |op: UnaryOp| {
        matches!(
            op,
            UnaryOp::Log | UnaryOp::Sqrt | UnaryOp::Rsqrt | UnaryOp::Recip
        )
    };

    let ops = [
        UnaryOp::Relu,
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Tanh,
        UnaryOp::Recip,
        UnaryOp::Sigmoid,
        UnaryOp::Silu,
        UnaryOp::Gelu,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Round,
        UnaryOp::Sign,
        UnaryOp::Softplus,
        UnaryOp::Elu,
        UnaryOp::HardSwish,
        UnaryOp::HardSigmoid,
        UnaryOp::Mish,
        UnaryOp::Softsign,
        UnaryOp::LogSigmoid,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Erf,
    ];
    // bf16 has ~3 decimal digits, so allow a looser tolerance there.
    let tol = if bf16 { 3e-2 } else { 2e-3 };

    println!(
        "unary activation sweep — {} ({}), n={n}, tol={tol:.0e}\n",
        ty.mlir(),
        if bf16 { "vectorized 32-wide" } else { "scalar" }
    );
    let (mut pass, mut fail) = (0, 0);
    for op in ops {
        let input: &[f32] = if needs_pos(op) { &pos } else { &signed };
        let mlir = emit_unary(op, ty, n, chunk);
        let tmp = format!("/tmp/rlx_un_{}", op.name());
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
                "  {:<8} COMPILE-FAIL  ({})",
                op.name(),
                first_line(&format!("{e:?}"))
            );
            fail += 1;
            continue;
        }
        let insts: Vec<u32> = std::fs::read(&insts_path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let out = if bf16 {
            match NpuIoBf16::open("", &xclbin, &insts, n).and_then(|io| io.run(input)) {
                Ok(o) => o,
                Err(e) => {
                    println!("  {:<8} RUN-FAIL  ({})", op.name(), e.0);
                    fail += 1;
                    continue;
                }
            }
        } else {
            match NpuIoF32::open("", &xclbin, &insts, n).and_then(|io| io.run(input)) {
                Ok(o) => o,
                Err(e) => {
                    println!("  {:<8} RUN-FAIL  ({})", op.name(), e.0);
                    fail += 1;
                    continue;
                }
            }
        };
        // allclose-style: pass if |out-want| ≤ tol + tol·|want| (abs OR rel), so
        // transcendentals near a zero crossing (sin/cos) aren't flagged by a
        // blown-up relative error. Report the max absolute error.
        let mut maxabs = 0.0f32;
        let mut over = false;
        for i in 0..n {
            let want = op.apply_f32(input[i]);
            let ae = (out[i] - want).abs();
            maxabs = maxabs.max(ae);
            if ae.is_nan() || ae > tol + tol * want.abs() {
                over = true;
            }
        }
        if maxabs.is_nan() || over {
            println!("  {:<8} FAIL ✗  max-abs-err {maxabs:.2e}", op.name());
            fail += 1;
        } else {
            println!("  {:<8} PASS ✓  max-abs-err {maxabs:.2e}", op.name());
            pass += 1;
        }
    }
    println!("\n{pass} passed, {fail} failed (of {})", ops.len());
    if fail > 0 {
        std::process::exit(1);
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(80).collect()
}
