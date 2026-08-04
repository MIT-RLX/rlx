// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// MULTI-CORE vectorized int8 matmul on the AIE2 array: the output columns split
// across `COLS` AIE columns, each core an all-resident m×k×(n/cols) tile. Scales
// past the single-core 64³ wall AND uses all columns for ~cols× throughput, while
// keeping each core's nt≤4 to dodge the Peano accumulator-unroll miscompile.
// Checked bit-exact vs CPU + benchmarked vs the single-core vectorized path.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. M=128 K=128 N=128 COLS=4 ITERS=100 \
//     cargo run -p rlx-xdna --features xrt --example xdna_matmul_multicol

use rlx_xdna::aie::{
    emit_matmul_multicol, emit_matmul_tiled, matmul_signed_fixup, tile_a, tile_b, tile_b_multicol,
    untile_c, untile_c_multicol,
};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
use rlx_xdna::npu_gemm::NpuGemm;
use std::time::Instant;

fn env(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

fn compile(name: &str, mlir: &str) -> Vec<u32> {
    let (aiecc, peano) = (
        std::env::var("AIECC").unwrap(),
        std::env::var("PEANO").unwrap(),
    );
    let tmp = format!("/tmp/rlx_mmc_{name}");
    std::fs::create_dir_all(&tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, mlir).unwrap();
    compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/b"),
        out_xclbin: &format!("{tmp}/k.xclbin"),
        out_insts: &format!("{tmp}/i.bin"),
    })
    .expect("compile");
    std::fs::read(format!("{tmp}/i.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let (m, k, n) = (env("M", "128"), env("K", "128"), env("N", "128"));
    let cols = env("COLS", "4");
    let iters = env("ITERS", "100");
    let flops = 2.0 * (m * k * n) as f64;

    let a: Vec<i8> = (0..m * k).map(|i| (i % 7) as i8 - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 5) as i8 - 2).collect();
    let mut cref = vec![0i32; m * n];
    for i in 0..m {
        for l in 0..k {
            let av = a[i * k + l] as i32;
            for j in 0..n {
                cref[i * n + j] += av * b[l * n + j] as i32;
            }
        }
    }

    // ---- multi-core (COLS columns) ----
    let insts = compile("mc", &emit_matmul_multicol(m, k, n, cols));
    let mm = NpuGemm::open("", "/tmp/rlx_mmc_mc/k.xclbin", &insts, m, k, n).expect("open");
    let at = tile_a(&a, m, k);
    let bt = tile_b_multicol(&b, k, n, cols);
    let ct = mm.run(&at, &bt).expect("run");
    let mut c = untile_c_multicol(&ct, m, n, cols);
    matmul_signed_fixup(&mut c, &b, m, k, n);
    let mism = (0..m * n).filter(|&i| c[i] != cref[i]).count();
    if mism != 0 {
        let bad: Vec<_> = (0..m * n)
            .filter(|&i| c[i] != cref[i])
            .take(3)
            .map(|i| (i, c[i], cref[i]))
            .collect();
        println!("multicol {m}x{k}x{n} cols={cols}: FAIL ✗ {mism} mism (i,got,want): {bad:?}");
        std::process::exit(1);
    }
    let mut mbest = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mm.run(&at, &bt).expect("run");
        mbest = mbest.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let mgops = flops / (mbest * 1e3);
    println!("multicol (cols={cols})  PASS ✓  best {mbest:8.1} us  →  {mgops:7.2} GOP/s");

    // ---- single-core vectorized, for comparison (only if it fits / compiles) ----
    if std::env::var("NOSINGLE").is_err() {
        match std::panic::catch_unwind(|| compile("single", &emit_matmul_tiled(m, k, n))) {
            Ok(insts) => {
                let sm = NpuGemm::open("", "/tmp/rlx_mmc_single/k.xclbin", &insts, m, k, n)
                    .expect("open");
                let (sat, sbt) = (tile_a(&a, m, k), tile_b(&b, k, n));
                let sct = sm.run(&sat, &sbt).expect("run");
                let mut sc = untile_c(&sct, m, n);
                matmul_signed_fixup(&mut sc, &b, m, k, n);
                let sok = (0..m * n).all(|i| sc[i] == cref[i]);
                let mut sbest = f64::MAX;
                for _ in 0..iters {
                    let t = Instant::now();
                    let _ = sm.run(&sat, &sbt).expect("run");
                    sbest = sbest.min(t.elapsed().as_secs_f64() * 1e6);
                }
                let sgops = flops / (sbest * 1e3);
                println!(
                    "single-core            {}  best {sbest:8.1} us  →  {sgops:7.2} GOP/s   (multicol {:.1}×)",
                    if sok { "PASS ✓ " } else { "FAIL ✗ " },
                    mgops / sgops
                );
            }
            Err(_) => println!(
                "single-core            (aiecc-failed to fit {m}³ — multicol scales past it)"
            ),
        }
    }
}
