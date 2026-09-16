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

/// Collapse repeated `%name = aie.tile(c, r)` declarations onto one SSA value.
///
/// AIE rows are fixed by hardware, so several logical roles legitimately land on
/// the same physical tile — an attention design has `shim_q`, `shim_kv` and
/// `shim_out`, and on a 1-column device all three ARE tile (0,0). Declaring it
/// three times parses and even builds an xclbin, but the objectfifo allocator
/// then treats them as three tiles and the design returns wrong data (attention
/// came back at max-rel-err 1.0 — the DMA channel assignment collides). One
/// declaration, three references, is both what the hardware is and what the
/// allocator needs.
fn dedupe_tile_decls(mlir: &str) -> String {
    use std::collections::HashMap;
    let mut canon: HashMap<(String, String), String> = HashMap::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut kept: Vec<String> = Vec::new();

    for line in mlir.lines() {
        let t = line.trim();
        let parsed = t
            .strip_prefix('%')
            .and_then(|r| r.split_once(" = aie.tile("))
            .and_then(|(name, rest)| {
                let inner = rest.strip_suffix(')')?;
                let (c, r) = inner.split_once(',')?;
                Some((name.to_string(), c.trim().to_string(), r.trim().to_string()))
            });
        match parsed {
            Some((name, c, r)) => match canon.get(&(c.clone(), r.clone())) {
                // Already declared this physical tile — drop the line, alias the name.
                Some(first) => {
                    alias.insert(name, first.clone());
                }
                None => {
                    canon.insert((c, r), name);
                    kept.push(line.to_string());
                }
            },
            None => kept.push(line.to_string()),
        }
    }
    let mut out = kept.join("\n");
    if mlir.ends_with('\n') {
        out.push('\n');
    }
    // Longest first: `%shim_q` must not be rewritten by a rule for `%shim`.
    let mut names: Vec<&String> = alias.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));
    for name in names {
        let to = &alias[name];
        out = out.replace(&format!("%{name}"), &format!("%{to}"));
    }
    out
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
    // Each attempt gets a CLEAN tmpdir. aiecc leaves a `.prj` tree and partial
    // objects behind on failure, and re-running into that directory makes the
    // retry fail the same way regardless of the flags — the -O0 rung below
    // compiles fine by hand and still failed here until this reset existed.
    let stage = |dir: &str| -> Result<(), XdnaError> {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).map_err(|e| XdnaError(format!("create tmpdir {dir}: {e}")))?;
        for obj in link_objs {
            let base = Path::new(obj)
                .file_name()
                .ok_or_else(|| XdnaError(format!("bad link obj path {obj}")))?;
            let dst = Path::new(dir).join(base);
            std::fs::copy(obj, &dst)
                .map_err(|e| XdnaError(format!("copy link obj {obj} → {}: {e}", dst.display())))?;
        }
        Ok(())
    };

    // Try the default optimisation level, then -O1, then -O0.
    //
    // At -O2 LLVM's own loop vectorizer re-widens a deliberately SCALAR core
    // loop into 16-lane vector arithmetic, and AIE2 has no datapath for some of
    // it — `LLVM ERROR: unable to legalize instruction: %98:_(<16 x s32>) =
    // G_MUL`. These are INT8/BF16 MAC tiles; a 32-bit vector multiply is not a
    // thing they can do. -O1 leaves the loop scalar and the same kernel builds
    // and runs. Retrying is better than pinning -O1 everywhere: the ops that do
    // vectorize keep their -O2 codegen.
    //
    // -O0 is the last rung, and it is about SIZE, not legality. An AIE2 core
    // has 64 KB of data memory but only **16 KB of program memory**, and an
    // optimised kernel can simply not fit: the attention core builds a 19,456 B
    // `.text` at -O1/-O2/-O3 and 11,184 B at -O0. aiecc reports that overflow as
    // `ValueError: Failed to generate cdo because:` with an EMPTY reason, which
    // is why it reads like a mystery — check `llvm-size -A` on
    // `<tmpdir>/main_core_0_2.elf` against 16 KB before assuming anything else.
    // Slower code that fits beats faster code that cannot be placed.
    let mut last_err = None;
    for opt in [None, Some("-O1"), Some("-O0")] {
        stage(spec.tmpdir)?;
        let src = std::fs::read_to_string(spec.mlir)
            .map_err(|e| XdnaError(format!("read {}: {e}", spec.mlir)))?;
        let deduped = Path::new(spec.tmpdir).join("deduped.mlir");
        std::fs::write(&deduped, dedupe_tile_decls(&src))
            .map_err(|e| XdnaError(format!("write {}: {e}", deduped.display())))?;
        let mut spec_d = spec.clone();
        let deduped_s = deduped.to_string_lossy().into_owned();
        spec_d.mlir = &deduped_s;
        match run_aiecc(&spec_d, opt) {
            Ok(v) => return Ok(v),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("at least one attempt"))
}

/// One `aiecc` invocation. `opt` optionally forces an AIE-core optimisation
/// level; the link objects are already staged in `spec.tmpdir` by the caller.
fn run_aiecc(spec: &OverlaySpec<'_>, opt: Option<&str>) -> Result<(String, String), XdnaError> {
    // Delete the outputs first. Success is judged by "did these files appear",
    // and they live OUTSIDE tmpdir, so a stale artifact from an earlier build
    // makes a FAILED compile report success and the caller then runs the
    // previous kernel. That produced a bogus PASS during debugging: aiecc died
    // on CDO generation, the old xclbin was still on disk, and the example
    // happily executed it.
    let _ = std::fs::remove_file(spec.out_xclbin);
    let _ = std::fs::remove_file(spec.out_insts);
    let mut extra: Vec<String> = Vec::new();
    if let Some(o) = opt {
        extra.push(o.to_string());
    }
    let status = Command::new(spec.aiecc)
        .args(extra)
        .args([
            // Peano, not Vitis/Chess. BOTH flags are required: `--no-xchesscc`
            // alone still routes core-ELF linking through `xchesscc_wrapper`,
            // which execs `xchesscc` from an AMD Vitis install that a
            // peano-only host does not have —
            //   `xchesscc_wrapper: line 53: xchesscc: command not found`
            // and aiecc exits 127 after having compiled everything else.
            "--no-xchesscc",
            "--no-xbridge",
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
