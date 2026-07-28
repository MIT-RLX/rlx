// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile an XDNA NPU overlay (`aie.mlir → .xclbin + insts.bin`) **without
//! Python**.
//!
//! The real MLIR-AIE compiler is a native ELF binary (`mlir_aie/bin/aiecc`,
//! ~212 MB, no `libpython`); `aiecc.py` is just a thin Python shim that execs
//! it. So driving overlay generation from rlx is a plain `std::process` call to
//! that binary — verified to run with `python3` blocked. Uses Peano (the
//! `llvm-aie` core compiler), not Vitis/Chess (`--no-xchesscc`), so it needs no
//! proprietary toolchain.

use std::path::Path;
use std::process::Command;

use crate::XdnaError;

/// Inputs for a Python-free overlay compile.
#[derive(Debug, Clone)]
pub struct OverlaySpec<'a> {
    /// The native `aiecc` binary (`.../mlir_aie/bin/aiecc`, NOT `aiecc.py`).
    pub aiecc: &'a str,
    /// The Peano install dir (`.../llvm-aie`) — the AIE core compiler.
    pub peano: &'a str,
    /// Input AIE MLIR (the IRON design's `aie.mlir`, or one rlx emits).
    pub mlir: &'a str,
    /// Scratch dir for intermediates.
    pub tmpdir: &'a str,
    /// Output `.xclbin` path.
    pub out_xclbin: &'a str,
    /// Output `insts_*.bin` path.
    pub out_insts: &'a str,
}

/// Compile `spec.mlir` to `spec.out_xclbin` + `spec.out_insts` by invoking the
/// **native** `aiecc` binary (no Python in the loop). Returns the two output
/// paths on success.
pub fn compile_overlay(spec: &OverlaySpec) -> Result<(String, String), XdnaError> {
    compile_overlay_linked(spec, &[])
}

/// Like [`compile_overlay`] but additionally makes each path in `link_objs` (a
/// pre-compiled AIE-core `.o`, e.g. the Peano-built `aie::mmul` microkernel)
/// available to aiecc's `link_with` resolution by copying it into the tmpdir under
/// its basename — so an emitted `func.func private @k(...) attributes {link_with =
/// "k.o"}` links against it. This is the seam for C++-microkernel cores (task #25).
pub fn compile_overlay_linked(
    spec: &OverlaySpec,
    link_objs: &[&str],
) -> Result<(String, String), XdnaError> {
    if !Path::new(spec.aiecc).exists() {
        return Err(XdnaError(format!(
            "native aiecc binary not found at {} (point at mlir_aie/bin/aiecc, not aiecc.py)",
            spec.aiecc
        )));
    }
    std::fs::create_dir_all(spec.tmpdir).ok();
    for obj in link_objs {
        let base = Path::new(obj)
            .file_name()
            .ok_or_else(|| XdnaError(format!("bad link obj path {obj}")))?;
        let dst = Path::new(spec.tmpdir).join(base);
        std::fs::copy(obj, &dst)
            .map_err(|e| XdnaError(format!("copy link obj {obj} → {}: {e}", dst.display())))?;
    }

    let status = Command::new(spec.aiecc)
        .args([
            "--no-xchesscc", // Peano, not Vitis/Chess
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            &format!("--tmpdir={}", spec.tmpdir),
            &format!("--peano={}", spec.peano),
            &format!("--xclbin-name={}", spec.out_xclbin),
            &format!("--npu-insts-name={}", spec.out_insts),
            spec.mlir,
        ])
        .status()
        .map_err(|e| XdnaError(format!("spawn aiecc: {e}")))?;

    if !status.success() {
        return Err(XdnaError(format!(
            "aiecc exited with {status} compiling {}",
            spec.mlir
        )));
    }
    for out in [spec.out_xclbin, spec.out_insts] {
        if !Path::new(out).exists() {
            return Err(XdnaError(format!("aiecc did not produce {out}")));
        }
    }
    Ok((spec.out_xclbin.to_string(), spec.out_insts.to_string()))
}

/// Compile the vendor `aie::mmul` int8 microkernel (`<include>/aie_kernels/aie2/
/// mm.cc`) to an AIE-core object `out_o` for a square `d×d×d` tile (DIM_M=DIM_K=
/// DIM_N=`d`, i8→i32, 4×8×8 subtiles), via Peano `clang++`. `clangxx` =
/// `<peano>/bin/clang++`, `include` = `<mlir_aie>/include`. The `.o` (symbols
/// `matmul_i8_i32` + `zero_i32`) is what [`compile_overlay_linked`] links against
/// an emitted microkernel overlay. Idempotent-safe (overwrites). Cheap (~1s).
pub fn build_mm_kernel(
    clangxx: &str,
    include: &str,
    d: usize,
    out_o: &str,
) -> Result<(), XdnaError> {
    let src = format!("{include}/aie_kernels/aie2/mm.cc");
    if !Path::new(&src).exists() {
        return Err(XdnaError(format!(
            "kernel source not found at {src} (bad mlir_aie include dir?)"
        )));
    }
    let wrap = format!("{out_o}.wrap.cc");
    std::fs::write(
        &wrap,
        format!(
            "#define DIM_M {d}\n#define DIM_K {d}\n#define DIM_N {d}\n#define combos(X) X(int8, i8, int32, i32, 4, 8, 8)\n#include \"{src}\"\n"
        ),
    )
    .map_err(|e| XdnaError(format!("write kernel wrapper {wrap}: {e}")))?;
    let status = Command::new(clangxx)
        .args([
            "-O2",
            "-std=c++20",
            "--target=aie2-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-Wno-missing-template-arg-list-after-template-kw",
            "-DNDEBUG",
            &format!("-I{include}"),
            "-D__AIE_API_AIE_ADF_HPP__",
            "-c",
            &wrap,
            "-o",
            out_o,
        ])
        .status()
        .map_err(|e| XdnaError(format!("spawn {clangxx}: {e}")))?;
    if !status.success() {
        return Err(XdnaError(format!(
            "Peano clang++ failed compiling {src} (DIM={d})"
        )));
    }
    if !Path::new(out_o).exists() {
        return Err(XdnaError(format!("kernel object {out_o} not produced")));
    }
    Ok(())
}
