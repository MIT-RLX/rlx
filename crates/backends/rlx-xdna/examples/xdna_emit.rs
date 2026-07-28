// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// rlx emits AIE-MLIR → compiles it Python-free (native aiecc) → runs on the NPU.
// The whole codegen pipeline, verified: a DMA-passthrough design rlx generated
// itself, compiled, and executed on the AMD XDNA NPU (out == in).
//
//   AIECC=<.../mlir_aie/bin/aiecc> PEANO=<.../llvm-aie> \
//   RLX_XDNA_SHIM=<librlx_xdna_shim.so> LD_LIBRARY_PATH=<xrt lib> \
//   cargo run -p rlx-xdna --features xrt --example xdna_emit

fn main() {
    let (len, fifo) = (4096usize, 1024usize);

    // 1) rlx EMITS the AIE MLIR (no IRON/Python).
    let mlir = rlx_xdna::aie::emit_passthrough(len, fifo);
    let mlir_path = "/tmp/rlx_passthrough.mlir";
    std::fs::write(mlir_path, &mlir).expect("write mlir");
    println!(
        "1. rlx emitted {} lines of AIE-MLIR → {mlir_path}",
        mlir.lines().count()
    );

    // 2) COMPILE it with the native aiecc binary (no Python).
    let aiecc = std::env::var("AIECC").expect("set AIECC (.../mlir_aie/bin/aiecc)");
    let peano = std::env::var("PEANO").expect("set PEANO (.../llvm-aie)");
    let (xclbin, insts_path) =
        rlx_xdna::compile::compile_overlay(&rlx_xdna::compile::OverlaySpec {
            aiecc: &aiecc,
            peano: &peano,
            mlir: mlir_path,
            tmpdir: "/tmp/rlx_pt_build",
            out_xclbin: "/tmp/rlx_pt.xclbin",
            out_insts: "/tmp/rlx_pt_insts.bin",
        })
        .expect("compile_overlay");
    println!("2. compiled to {xclbin} (native aiecc, no Python)");

    // 3) RUN it on the NPU.
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .expect("read insts")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let input: Vec<i32> = (0..len as i32)
        .map(|i| i.wrapping_mul(7).wrapping_sub(3))
        .collect();
    let out = rlx_xdna::npu_gemm::run_passthrough("", &xclbin, &insts, &input).expect("run on NPU");

    // 4) VERIFY.
    if out == input {
        println!("3. ran on NPU: PASS ✓  out == in ({len} i32) — rlx→AIE-MLIR→NPU works");
    } else {
        let mism = out.iter().zip(&input).filter(|(a, b)| a != b).count();
        println!("3. ran on NPU: FAIL ✗ — {mism}/{len} mismatches");
        std::process::exit(1);
    }
}
