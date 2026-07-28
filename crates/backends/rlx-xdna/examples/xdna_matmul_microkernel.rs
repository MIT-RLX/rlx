// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The reliable, fast NPU int8 matmul: the vendor `aie::mmul` microkernel driven by
// an rlx-emitted overlay (K-accumulation × `COLS` AIE columns). Computes
// `DIM × (KT·DIM) × (DIM·COLS)`; bit-exact vs CPU, benchmarked. This is the same
// path `Device::Xdna` uses for `Op::MatMul`. The kernel `.o` is compiled here
// automatically (Peano), so you only set the usual AIECC/PEANO/shim env:
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. LD_LIBRARY_PATH=<xrt lib> XILINX_XRT=.. \
//     DIM=64 KT=8 COLS=4 ITERS=50 \
//     cargo run --release -p rlx-xdna --features xrt --example xdna_matmul_microkernel

use rlx_xdna::aie::{emit_matmul_microkernel, tile_a_kacc, tile_b_kacc_multicol, untile_c_multicol};
use rlx_xdna::compile::{build_mm_kernel, compile_overlay_linked, OverlaySpec};
use rlx_xdna::npu_gemm::NpuGemm;
use std::time::Instant;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let (d, kt, cols) = (env("DIM", 64), env("KT", 8), env("COLS", 4));
    let iters = env("ITERS", 50);

    // Opt-in TURBO (needs root/CAP_SYS_ADMIN): clock the NPU to max DPM. Held for the
    // process so the mode persists across the benchmark loop. Built with `--features
    // xrt,direct`; a no-op without RLX_XDNA_TURBO.
    #[cfg(all(feature = "direct", target_os = "linux"))]
    if std::env::var("RLX_XDNA_TURBO").is_ok() {
        match rlx_xdna::direct::Npu::open("") {
            Ok(npu) => match npu.set_turbo() {
                Ok(()) => {
                    println!("[turbo] NPU power mode → TURBO (max DPM)");
                    std::mem::forget(npu);
                }
                Err(e) => println!("[turbo] SET_STATE(TURBO) failed: {e} — needs root/CAP_SYS_ADMIN"),
            },
            Err(e) => println!("[turbo] cannot open accel device: {e}"),
        }
    }
    let (m, k, n) = (d, kt * d, cols * d);

    let aiecc = std::env::var("AIECC").expect("set AIECC (native mlir-aie aiecc)");
    let peano = std::env::var("PEANO").expect("set PEANO (llvm-aie dir)");
    // The mlir_aie include tree (holds aie_kernels/aie2/mm.cc + aie_api/).
    // RLX_XDNA_AIE_INCLUDE overrides the AIECC-derived path (needed for pip mlir_aie
    // installs where bin/aiecc isn't at <mlir_aie>/bin) — mirrors the backend.
    let include = std::env::var("RLX_XDNA_AIE_INCLUDE").unwrap_or_else(|_| {
        std::path::Path::new(&aiecc)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| format!("{}/include", p.display()))
            .expect("derive mlir_aie include from AIECC")
    });

    let a: Vec<i8> = (0..m * k).map(|i| (i % 7) as i8 - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 5) as i8 - 2).collect();
    let mut cref = vec![0i32; m * n];
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                cref[i * n + j] += a[i * k + l] as i32 * b[l * n + j] as i32;
            }
        }
    }

    // Compile the kernel .o (Peano) + the overlay (aiecc, links the .o), both cached in tmp.
    let tmp = format!("/tmp/rlx_mmk_{d}_{kt}_{cols}");
    std::fs::create_dir_all(&tmp).ok();
    let kernel_o = format!("{tmp}/mm_{d}.o");
    build_mm_kernel(&format!("{peano}/bin/clang++"), &include, d, &kernel_o).expect("build kernel .o");
    let obj_base = format!("mm_{d}.o");
    std::fs::write(format!("{tmp}/aie.mlir"), emit_matmul_microkernel(d, kt, cols, &obj_base)).unwrap();
    compile_overlay_linked(
        &OverlaySpec {
            aiecc: &aiecc,
            peano: &peano,
            mlir: &format!("{tmp}/aie.mlir"),
            tmpdir: &format!("{tmp}/build"),
            out_xclbin: &format!("{tmp}/k.xclbin"),
            out_insts: &format!("{tmp}/i.bin"),
        },
        &[&kernel_o],
    )
    .expect("compile + link");

    let insts: Vec<u32> = std::fs::read(format!("{tmp}/i.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mm = NpuGemm::open("", &format!("{tmp}/k.xclbin"), &insts, m, k, n).expect("open");
    let (at, bt) = (tile_a_kacc(&a, d, kt), tile_b_kacc_multicol(&b, d, kt, cols));
    let ct = mm.run(&at, &bt).expect("run");
    let c = untile_c_multicol(&ct, m, n, cols);

    let mism = (0..m * n).filter(|&i| c[i] != cref[i]).count();
    if mism != 0 {
        let bad: Vec<_> = (0..m * n).filter(|&i| c[i] != cref[i]).take(4).map(|i| (i, c[i], cref[i])).collect();
        println!("microkernel {m}x{k}x{n} (DIM={d} KT={kt} COLS={cols}): FAIL ✗ {mism} mism (i,got,want): {bad:?}");
        std::process::exit(1);
    }
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let _ = mm.run(&at, &bt).expect("run");
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let gops = 2.0 * (m * k * n) as f64 / (best * 1e3);
    println!("microkernel {m}x{k}x{n} (DIM={d} KT={kt} COLS={cols})  PASS ✓  best {best:8.1} us  →  {gops:8.2} GOP/s");
}
