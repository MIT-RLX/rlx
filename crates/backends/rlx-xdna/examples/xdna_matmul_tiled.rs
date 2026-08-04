// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// VECTORIZED int8 matmul on the AIE2 hardware MAC — pure-Rust AIE-MLIR
// (vector.contract → aievec.matmul, unrolled-K chain, tile-contiguous layout).
// Host pre-tiles A/B, de-tiles C. Checked bit-exact vs CPU + benchmarked GOP/s,
// compared against the scalar emit_matmul.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. M=64 K=64 N=64 ITERS=100 \
//     cargo run -p rlx-xdna --features xrt --example xdna_matmul_tiled

use rlx_xdna::aie::{
    emit_matmul, emit_matmul_tiled, matmul_signed_fixup, tile_a, tile_b, untile_c,
};
use rlx_xdna::compile::{OverlaySpec, compile_overlay};
#[cfg(feature = "xrt")]
use rlx_xdna::npu_gemm::NpuGemm;
#[cfg(feature = "xrt")]
use std::time::Instant;

#[cfg(feature = "xrt")]
fn env(k: &str, d: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| d.parse().unwrap())
}

#[cfg(feature = "xrt")]
fn compile(name: &str, mlir: &str) -> Vec<u32> {
    let (aiecc, peano) = (
        std::env::var("AIECC").unwrap(),
        std::env::var("PEANO").unwrap(),
    );
    let tmp = format!("/tmp/rlx_mmt_{name}");
    std::fs::create_dir_all(&tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts = format!("{tmp}/i.bin");
    compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/b"),
        out_xclbin: &xclbin,
        out_insts: &insts,
    })
    .expect("compile");
    std::fs::read(&insts)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(feature = "xrt")]
fn main() {
    let (m, k, n) = (env("M", "64"), env("K", "64"), env("N", "64"));
    let iters = env("ITERS", "100");
    let flops = 2.0 * (m * k * n) as f64;

    // i8 operands; small so the i32 accumulation is exact. PROBE mode: A =
    // δ(i,k) so C = top rows of B, and B = 0..63 — reveals the output lane order.
    let probe: usize = std::env::var("PROBE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (a, b): (Vec<i8>, Vec<i8>) = match probe {
        // A=identity, B=0..63 → C = top rows of B (tests B layout)
        1 => (
            (0..m * k)
                .map(|x| if (x / k) == (x % k) { 1 } else { 0 })
                .collect(),
            (0..k * n).map(|i| (i % 64) as i8).collect(),
        ),
        // A=0..31, B=identity → C = A (tests A layout)
        2 => (
            (0..m * k).map(|i| (i % 32) as i8).collect(),
            (0..k * n)
                .map(|x| if (x / n) == (x % n) { 1 } else { 0 })
                .collect(),
        ),
        // A row0 = all ones (rest 0), B=0..63 → C[0][j] = Σ_k B[k][j] = 224+8j
        3 => (
            (0..m * k).map(|x| if x < k { 1 } else { 0 }).collect(),
            (0..k * n).map(|i| (i % 64) as i8).collect(),
        ),
        // NONNEG=1 → non-negative inputs (isolates a possible signedness bug)
        _ => {
            let nn = std::env::var("NONNEG").is_ok();
            (
                (0..m * k)
                    .map(|i| if nn { (i % 7) as i8 } else { (i % 7) as i8 - 3 })
                    .collect(),
                (0..k * n)
                    .map(|i| if nn { (i % 5) as i8 } else { (i % 5) as i8 - 2 })
                    .collect(),
            )
        }
    };
    let mut cref = vec![0i32; m * n];
    for i in 0..m {
        for l in 0..k {
            let av = a[i * k + l] as i32;
            for j in 0..n {
                cref[i * n + j] += av * b[l * n + j] as i32;
            }
        }
    }

    // ---- vectorized (aievec.matmul), tile-contiguous ----
    let insts = compile("vec", &emit_matmul_tiled(m, k, n));
    let mm = NpuGemm::open("", "/tmp/rlx_mmt_vec/k.xclbin", &insts, m, k, n).expect("open");
    let (at, bt) = (tile_a(&a, m, k), tile_b(&b, k, n));
    let ct = mm.run(&at, &bt).expect("run");
    let mut c = untile_c(&ct, m, n);
    matmul_signed_fixup(&mut c, &b, m, k, n); // undo the +128 A-bias (signed int8)
    if std::env::var("DUMP").is_ok() {
        println!("A       : {:?}", &a[..(m * k).min(32)]);
        println!("B       : {:?}", &b[..(k * n).min(64)]);
        println!("bt(tile): {:?}", &bt[..(k * n).min(16)]);
        println!("ct (raw): {:?}", &ct[..(m * n).min(32)]);
        println!("cref    : {:?}", &cref[..(m * n).min(32)]);
    }
    let mism = (0..m * n).filter(|&i| c[i] != cref[i]).count();
    if mism != 0 {
        let bad: Vec<_> = (0..m * n)
            .filter(|&i| c[i] != cref[i])
            .take(3)
            .map(|i| (i, c[i], cref[i]))
            .collect();
        println!("vectorized {m}x{k}x{n}: FAIL ✗ {mism} mism (i,got,want): {bad:?}");
        if std::env::var("DUMP").is_ok() {
            let b0 = (0..m * n).find(|&i| c[i] != cref[i]).unwrap();
            let lo = b0.saturating_sub(2);
            let hi = (b0 + 14).min(m * n);
            let cblk = (n / 8) * 32; // one C row-block (tiled)
            println!(
                "  first bad untiled idx {b0} (row {}, col {}); cblk={cblk}",
                b0 / n,
                b0 % n
            );
            println!("  c   [{lo}..{hi}]: {:?}", &c[lo..hi]);
            println!("  cref[{lo}..{hi}]: {:?}", &cref[lo..hi]);
            println!("  ct(raw)[{lo}..{hi}]: {:?}", &ct[lo..hi.min(ct.len())]);
        }
        std::process::exit(1);
    }
    let mut vbest = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mm.run(&at, &bt).expect("run");
        vbest = vbest.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let vgops = flops / (vbest * 1e3);
    println!("vectorized (aievec.matmul)  PASS ✓  best {vbest:8.1} us  →  {vgops:7.2} GOP/s");

    // ---- scalar (emit_matmul), for comparison ----
    let insts = compile("scalar", &emit_matmul(m, k, n));
    let sm = NpuGemm::open("", "/tmp/rlx_mmt_scalar/k.xclbin", &insts, m, k, n).expect("open");
    let sc = sm.run(&a, &b).expect("run"); // scalar uses row-major (no tiling)
    let smism = (0..m * n).filter(|&i| sc[i] != cref[i]).count();
    let mut sbest = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = sm.run(&a, &b).expect("run");
        sbest = sbest.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let sgops = flops / (sbest * 1e3);
    println!(
        "scalar     (per-elem MAC)   {}  best {sbest:8.1} us  →  {sgops:7.2} GOP/s",
        if smism == 0 { "PASS ✓ " } else { "FAIL ✗ " }
    );
    println!("\nspeedup: {:.1}× ({m}×{k}×{n})", vgops / sgops);
}

#[cfg(not(feature = "xrt"))]
fn main() {
    eprintln!("xdna_matmul_tiled requires --features xrt");
}
