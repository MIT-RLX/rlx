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

/// Which `aiecc` command-line generation we are driving.
///
/// mlir-aie replaced `aiecc`'s CLI: the Python-era driver took "generate this"
/// toggles plus `--no-*` opt-outs, while the current native **declarative
/// driver** (SHA `95b3d1ccc0b`, Sep 2026) builds a compilation graph and you
/// REQUEST outputs by edge name. Every bare flag we used to pass disappeared in
/// that change — reported as issue #2, where `aiecc --help` from a current
/// install accepts none of `--no-xchesscc`, `--no-xbridge`,
/// `--aie-generate-xclbin`, `--aie-generate-npu-insts` or `--no-compile-host`.
///
/// Probed rather than pinned, because both generations are in the wild: a hard
/// switch would break anyone on an older mlir-aie, and the four `name=value`
/// flags we pass (`--tmpdir`, `--peano`, `--xclbin-name`, `--npu-insts-name`)
/// are valid in both, so only the bare flags need to differ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AieccCli {
    /// Pre-Sep-2026: `--aie-generate-*` toggles, `--no-xchesscc`/`--no-xbridge`
    /// opt-outs, `--no-compile-host`.
    Generate,
    /// Declarative driver: outputs requested via `--get`, and
    /// xchesscc/xbridge became opt-IN (so omitting them is Peano-only).
    Declarative,
}

/// What this `aiecc` accepts, probed once from its own `--help`.
#[derive(Clone, Debug)]
struct AieccCaps {
    cli: AieccCli,
    /// `--get-<name>` shorthands the binary actually registers. Discovered, not
    /// assumed: only `--get-xclbin` is named in the help prose, and guessing the
    /// edge name for the instruction stream would trade one wrong flag for
    /// another.
    get_shorthands: Vec<String>,
}

/// Ask `aiecc` what it supports.
///
/// Falls back to [`AieccCli::Generate`] when `--help` cannot be run or says
/// nothing recognisable, which keeps the historical behaviour for anyone whose
/// build worked before this probe existed.
fn probe_aiecc(aiecc: &str) -> AieccCaps {
    let help = Command::new(aiecc)
        .arg("--help")
        .output()
        .ok()
        .map(|o| {
            let mut t = String::from_utf8_lossy(&o.stdout).into_owned();
            t.push_str(&String::from_utf8_lossy(&o.stderr));
            t
        })
        .unwrap_or_default();

    // `--aie-generate-xclbin` is the clearest legacy marker; `--get` is the
    // declarative one. Check the legacy marker first so a driver that somehow
    // offers both keeps the behaviour that is known to work.
    classify_aiecc_help(&help)
}

/// Classify an `aiecc --help` dump. Split out of [`probe_aiecc`] so the flag
/// selection is testable on any host — choosing the wrong CLI generation is a
/// pure string decision, and it should not take an NPU to catch it.
fn classify_aiecc_help(help: &str) -> AieccCaps {
    // `--aie-generate-xclbin` is the clearest legacy marker; `--get` is the
    // declarative one. Check the legacy marker first so a driver that somehow
    // offers both keeps the behaviour that is known to work. An unreadable or
    // empty `--help` also lands on Generate, preserving historical behaviour.
    let cli = if help.contains("--aie-generate-xclbin") {
        AieccCli::Generate
    } else if help.contains("--get") {
        AieccCli::Declarative
    } else {
        AieccCli::Generate
    };

    let mut get_shorthands: Vec<String> = Vec::new();
    for line in help.lines() {
        let t = line.trim_start();
        if !t.starts_with("--get-") {
            continue;
        }
        let name: String = t
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if name.len() > "--get-".len() && !get_shorthands.contains(&name) {
            get_shorthands.push(name);
        }
    }
    AieccCaps {
        cli,
        get_shorthands,
    }
}

/// The output requests to pass a declarative `aiecc`.
///
/// `RLX_XDNA_AIECC_GET` overrides the whole list (comma-separated edge names),
/// which is the escape hatch for a newer mlir-aie that renames an edge: the
/// instruction-stream edge name is NOT discoverable from `--help` (only
/// `--get-xclbin` is named there), and `--emit-dot` is what prints the real
/// graph. Without the override we request what we can prove and let the
/// existing "did the file appear" check report the rest.
fn declarative_get_args(caps: &AieccCaps) -> Vec<String> {
    if let Some(list) = rlx_ir::env::var("RLX_XDNA_AIECC_GET") {
        let names: Vec<&str> = list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !names.is_empty() {
            return vec![format!("--get={}", names.join(","))];
        }
    }
    // Prefer shorthands the binary actually registers.
    let mut args: Vec<String> = Vec::new();
    for want in ["xclbin", "npu-insts", "npu_insts", "insts"] {
        let short = format!("--get-{want}");
        if caps.get_shorthands.contains(&short) {
            args.push(short);
        }
    }
    if args.is_empty() {
        // The help names `--get-xclbin` in prose even when it does not list the
        // shorthands as their own options, so request it by edge name.
        args.push("--get=xclbin".to_string());
    }
    args
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
    // Probe once, not per attempt: `--help` on a ~212 MB binary is cheap but the
    // retry loop runs it up to three times, and the answer cannot change mid-compile.
    let caps = probe_aiecc(spec.aiecc);
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
        match run_aiecc(&spec_d, opt, &caps) {
            Ok(v) => return Ok(v),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("at least one attempt"))
}

/// One `aiecc` invocation. `opt` optionally forces an AIE-core optimisation
/// level; the link objects are already staged in `spec.tmpdir` by the caller.
fn run_aiecc(
    spec: &OverlaySpec<'_>,
    opt: Option<&str>,
    caps: &AieccCaps,
) -> Result<(String, String), XdnaError> {
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
    // Bare flags differ by CLI generation; the `name=value` ones below are
    // accepted by both.
    match caps.cli {
        AieccCli::Generate => {
            // Peano, not Vitis/Chess. BOTH flags are required: `--no-xchesscc`
            // alone still routes core-ELF linking through `xchesscc_wrapper`,
            // which execs `xchesscc` from an AMD Vitis install that a
            // peano-only host does not have —
            //   `xchesscc_wrapper: line 53: xchesscc: command not found`
            // and aiecc exits 127 after having compiled everything else.
            extra.push("--no-xchesscc".to_string());
            extra.push("--no-xbridge".to_string());
            extra.push("--aie-generate-xclbin".to_string());
            extra.push("--aie-generate-npu-insts".to_string());
            extra.push("--no-compile-host".to_string());
        }
        AieccCli::Declarative => {
            // No `--no-*` opt-outs here: xchesscc/xbridge became opt-IN, so
            // omitting them is exactly the Peano-only path the comment above
            // describes. Nothing is built that was not requested either, which
            // is what `--no-compile-host` used to buy.
            extra.extend(declarative_get_args(caps));
        }
    }
    let status = Command::new(spec.aiecc)
        .args(&extra)
        .args([
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
            // On the declarative driver an artifact can go missing simply
            // because we never asked for it: outputs are requested by graph
            // edge name, and the instruction-stream edge is not discoverable
            // from `--help`. Say so, instead of leaving it as "aiecc did not
            // produce" with no next step.
            if caps.cli == AieccCli::Declarative {
                return Err(XdnaError(format!(
                    "aiecc did not produce {out} — this driver requests outputs by graph \
                     edge name and rlx asked for [{}]. Run `aiecc --emit-dot <mlir>` to list \
                     the real edge names, then set RLX_XDNA_AIECC_GET=<comma,separated> to \
                     override the request list.",
                    declarative_get_args(caps).join(" ")
                )));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `aiecc --help` lines from the current declarative driver, as
    /// attached to issue #2 (mlir-aie SHA `95b3d1ccc0b`, built Sep 11 2026).
    /// Verbatim on purpose: the point of the probe is to agree with the actual
    /// tool, and paraphrasing the fixture would let it agree with a paraphrase.
    const DECLARATIVE_HELP: &str = "\
OVERVIEW: aiecc declarative driver

USAGE: aiecc [options] <input mlir> [-- <host cc args>...]

OPTIONS:
  --get=<name>               - Request graph output(s) by edge name (repeatable / comma-separated); named artifacts also have --get-<name> shorthands (e.g. --get-xclbin)
  --npu-insts-name=<string>  - Output NPU insts filename template (use {0} for multi-device)
  --peano=<string>           - Peano install dir
  --tmpdir=<string>          - Intermediate workdir (default: <input>.prj in cwd)
  --xbridge                  - Link cores with the Chess toolchain (xbridge/BCF) instead of the default
  --xchesscc                 - Compile cores with the Chess toolchain (xchesscc) instead of the default
  --xclbin-name=<string>     - Output xclbin filename template (use {0} for multi-device)
";

    /// Shape of the pre-Sep-2026 driver we used to target exclusively.
    const LEGACY_HELP: &str = "\
OPTIONS:
  --aie-generate-xclbin      - Generate xclbin
  --aie-generate-npu-insts   - Generate NPU instructions
  --no-xchesscc              - Do not use xchesscc
  --no-xbridge               - Do not use xbridge
  --no-compile-host          - Do not compile the host program
  --xclbin-name=<string>     - Output xclbin
";

    #[test]
    fn current_driver_is_detected_as_declarative() {
        let caps = classify_aiecc_help(DECLARATIVE_HELP);
        assert_eq!(caps.cli, AieccCli::Declarative);
    }

    #[test]
    fn older_driver_is_detected_as_generate() {
        let caps = classify_aiecc_help(LEGACY_HELP);
        assert_eq!(caps.cli, AieccCli::Generate);
    }

    /// An `aiecc` we cannot interrogate must keep the behaviour that worked
    /// before the probe existed, not fall into the newer flag set.
    #[test]
    fn unreadable_help_falls_back_to_the_historical_flags() {
        assert_eq!(classify_aiecc_help("").cli, AieccCli::Generate);
        assert_eq!(classify_aiecc_help("garbage").cli, AieccCli::Generate);
    }

    /// The regression this guards: every bare flag rlx used to pass is absent
    /// from the current driver. If someone re-adds one unconditionally, the
    /// Declarative branch must not be the place it lands.
    #[test]
    fn none_of_the_legacy_bare_flags_exist_in_the_current_driver() {
        for f in [
            "--no-xchesscc",
            "--no-xbridge",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
        ] {
            assert!(
                !DECLARATIVE_HELP.contains(f),
                "{f} unexpectedly present in the declarative driver fixture"
            );
        }
        // …while the `name=value` flags we keep passing in BOTH modes are there.
        for f in [
            "--tmpdir=",
            "--peano=",
            "--xclbin-name=",
            "--npu-insts-name=",
        ] {
            assert!(DECLARATIVE_HELP.contains(f), "{f} missing from the fixture");
        }
    }

    /// Chess became opt-IN, so the Peano-only path is "say nothing" — the
    /// declarative request list must carry no `--no-*` opt-outs at all.
    #[test]
    fn declarative_requests_outputs_and_opts_out_of_nothing() {
        let caps = classify_aiecc_help(DECLARATIVE_HELP);
        let args = declarative_get_args(&caps);
        assert!(
            args.iter().all(|a| a.starts_with("--get")),
            "expected only output requests, got {args:?}"
        );
        assert!(
            args.iter().any(|a| a.contains("xclbin")),
            "xclbin was never requested: {args:?}"
        );
    }

    /// When the binary registers `--get-<name>` shorthands as real options we
    /// use them, rather than guessing edge names.
    #[test]
    fn get_shorthands_are_discovered_not_assumed() {
        let help = format!(
            "{DECLARATIVE_HELP}  --get-xclbin  - Request the xclbin\n  --get-npu-insts  - Request the NPU instruction stream\n"
        );
        let caps = classify_aiecc_help(&help);
        assert!(caps.get_shorthands.contains(&"--get-xclbin".to_string()));
        assert!(caps.get_shorthands.contains(&"--get-npu-insts".to_string()));
        let args = declarative_get_args(&caps);
        assert!(args.contains(&"--get-xclbin".to_string()), "{args:?}");
        assert!(args.contains(&"--get-npu-insts".to_string()), "{args:?}");
    }
}
