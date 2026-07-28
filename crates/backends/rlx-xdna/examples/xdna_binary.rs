// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity sweep for the pure-Rust AIE-MLIR **binary** elementwise emitter: for
// every `BinaryOp`, rlx emits the vectorized i32 kernel, compiles it (native
// aiecc), runs it two-input on the NPU (`NpuIo::run2`), and checks it bit-exact
// vs a CPU reference. Prints a per-op PASS/FAIL + GB/s table so we can see which
// arith ops legalize on AIE2 in one shot.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. N=1048576 CHUNK=2048 ITERS=50 \
//     cargo run -p rlx-xdna --features xrt --example xdna_binary

use rlx_xdna::aie::{BinaryOp, Ty, emit_binary};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuIo;
use std::time::Instant;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let n: usize = env("N", "1048576").parse().unwrap();
    let chunk: usize = env("CHUNK", "2048").parse().unwrap();
    let iters: usize = env("ITERS", "50").parse().unwrap();
    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");

    // Deterministic operands. `b` is 1..=7 so div/mod are well-defined and the
    // shift amounts stay in range.
    let a: Vec<i32> = (0..n).map(|i| (i as i32 % 251) - 125).collect();
    let b: Vec<i32> = (0..n).map(|i| (i as i32 % 7) + 1).collect();

    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::Mod,
        BinaryOp::BitAnd,
        BinaryOp::BitOr,
        BinaryOp::BitXor,
        BinaryOp::Shl,
        BinaryOp::Shr,
    ];

    println!("binary op sweep — i32, n={n}, chunk={chunk} (16-lane vectorized)\n");
    let (mut pass, mut fail) = (0, 0);
    for op in ops {
        let mlir = emit_binary(op, Ty::I32, n, chunk);
        let tmp = format!("/tmp/rlx_bin_{}", op.name());
        std::fs::create_dir_all(&tmp).ok();
        let mp = format!("{tmp}/aie.mlir");
        std::fs::write(&mp, &mlir).unwrap();
        let xclbin = format!("{tmp}/k.xclbin");
        let insts_path = format!("{tmp}/insts.bin");
        let cres = compile_overlay(&OverlaySpec {
            aiecc: &aiecc,
            peano: &peano,
            mlir: &mp,
            tmpdir: &format!("{tmp}/build"),
            out_xclbin: &xclbin,
            out_insts: &insts_path,
        });
        if let Err(e) = cres {
            println!(
                "  {:<7} COMPILE-FAIL  ({})",
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
        let io = match NpuIo::open("", &xclbin, &insts, n) {
            Ok(io) => io,
            Err(e) => {
                println!("  {:<7} OPEN-FAIL     ({})", op.name(), e.0);
                fail += 1;
                continue;
            }
        };
        let out = match io.run2(&a, &b) {
            Ok(o) => o,
            Err(e) => {
                println!("  {:<7} RUN-FAIL      ({})", op.name(), e.0);
                fail += 1;
                continue;
            }
        };
        let mism = (0..n)
            .filter(|&i| out[i] != op.apply_i32(a[i], b[i]))
            .count();
        if mism != 0 {
            let bad: Vec<_> = (0..n)
                .filter(|&i| out[i] != op.apply_i32(a[i], b[i]))
                .take(2)
                .map(|i| (a[i], b[i], out[i], op.apply_i32(a[i], b[i])))
                .collect();
            println!(
                "  {:<7} FAIL ✗ {mism} mism (a,b,got,want): {bad:?}",
                op.name()
            );
            fail += 1;
            continue;
        }
        let mut best = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let _ = io.run2(&a, &b).expect("run");
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        let gbps = (n as f64 * 4.0 * 3.0) / (best * 1e3); // a+b+out bytes / ns
        println!(
            "  {:<7} PASS ✓  best {best:7.1} us  {gbps:5.1} GB/s",
            op.name()
        );
        pass += 1;
    }
    println!("\n{pass} passed, {fail} failed (of {})", ops.len());
    if fail > 0 {
        std::process::exit(1);
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(80).collect()
}
