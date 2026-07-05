// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `rlx-torch-import` — map a `torch.export` graph (as `torch-ir.json` +
//! `weights.safetensors`) onto RLX: emit a runnable bundle and/or a generated
//! crate, and verify numeric parity against PyTorch.
//!
//! Usage:
//!   rlx-torch-import build <dir> [--emit bundle,crate] [--verify] [--crate-name N]

use anyhow::{Result, bail};
use rlx_torch_import::{ConvertOptions, convert};
use std::path::PathBuf;

fn rlx_root() -> PathBuf {
    // crates/io/rlx-torch-import → workspace root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: rlx-torch-import build <dir> [--emit bundle,crate] [--verify] [--crate-name N]"
        );
        std::process::exit(2);
    }
    match args[1].as_str() {
        "build" => cmd_build(&args[2..]),
        "onnx" => cmd_onnx(&args[2..]),
        other => bail!("unknown subcommand {other:?} (expected `build` or `onnx`)"),
    }
}

/// `onnx <model.onnx> -o <dir>` — import an ONNX model (via rlx-onnx-import's
/// full op registry) into a runnable bundle under `<dir>/bundle`.
fn cmd_onnx(args: &[String]) -> Result<()> {
    let mut onnx: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut verify = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--verify" => verify = true,
            s if !s.starts_with('-') => onnx = Some(PathBuf::from(s)),
            other => bail!("unknown flag {other:?}"),
        }
        i += 1;
    }
    let onnx = onnx.ok_or_else(|| anyhow::anyhow!("missing <model.onnx>"))?;
    let out = out.ok_or_else(|| anyhow::anyhow!("missing -o <dir>"))?;

    #[cfg(feature = "onnx")]
    {
        let bundle_dir = out.join("bundle");
        let meta = rlx_torch_import::onnx::emit_onnx_bundle(&onnx, &bundle_dir)?;
        println!("model  : {}", meta.name);
        println!(
            "inputs : {}",
            meta.inputs
                .iter()
                .map(|i| format!("{}{:?}", i.name, i.shape))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("outputs: {}", meta.output_count);
        println!("bundle : {}", bundle_dir.display());

        if verify {
            let reference = out.join("reference.safetensors");
            let report = rlx_torch_import::verify::verify_bundle(&bundle_dir, &reference)?;
            println!(
                "parity : {}",
                if report.passed {
                    "PASS ✓"
                } else {
                    "FAIL ✗"
                }
            );
            for o in &report.outputs {
                println!(
                    "  out[{}]  cosine={:.6}  max|err|={:.3e}  ({} elems)",
                    o.index, o.cosine, o.max_abs_err, o.numel
                );
            }
            if !report.passed {
                std::process::exit(1);
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (onnx, out, verify);
        bail!("ONNX import requires building with `--features onnx`")
    }
}

fn cmd_build(args: &[String]) -> Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut emit = "bundle,crate".to_string();
    let mut verify = false;
    let mut crate_name: Option<String> = None;
    let mut emit_style = "graph".to_string();
    let mut device = "cpu".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--emit" => {
                i += 1;
                emit = args.get(i).cloned().unwrap_or_default();
            }
            "--emit-style" => {
                i += 1;
                emit_style = args.get(i).cloned().unwrap_or_default();
            }
            "--device" => {
                i += 1;
                device = args.get(i).cloned().unwrap_or_default();
            }
            "--verify" => verify = true,
            "--no-verify" => verify = false,
            "--crate-name" => {
                i += 1;
                crate_name = args.get(i).cloned();
            }
            s if !s.starts_with("--") => dir = Some(PathBuf::from(s)),
            other => bail!("unknown flag {other:?}"),
        }
        i += 1;
    }
    let dir = dir.ok_or_else(|| anyhow::anyhow!("missing <dir>"))?;
    let emit_bundle = emit.split(',').any(|s| s.trim() == "bundle");
    let emit_crate = emit.split(',').any(|s| s.trim() == "crate");

    let opts = ConvertOptions {
        emit_bundle,
        emit_crate,
        verify,
        crate_name,
        rlx_root: rlx_root(),
        emit_style: rlx_torch_import::emit_styles::EmitStyle::parse(&emit_style)?,
        device,
    };
    let report = convert(&dir, &opts)?;

    // Machine-readable result for the Python wrapper.
    std::fs::write(
        dir.join("rlx-import-result.json"),
        serde_json::to_string_pretty(&report)?,
    )?;

    // Human summary.
    println!("model      : {}", report.model);
    println!(
        "graph      : {} inputs, {} params, {} ops",
        report.num_inputs, report.num_params, report.num_instrs
    );
    if let Some(b) = &report.bundle_dir {
        println!("bundle     : {b}");
    }
    if let Some(c) = &report.crate_dir {
        println!("crate      : {c}");
    }
    let mut ok = true;
    if let Some(p) = &report.parity {
        println!(
            "parity     : {} (cos≥{}, |err|≤{})",
            if p.passed { "PASS ✓" } else { "FAIL ✗" },
            p.cosine_threshold,
            p.max_abs_threshold
        );
        for o in &p.outputs {
            println!(
                "  out[{}]  cosine={:.6}  max|err|={:.3e}  rel={:.3e}  ({} elems)",
                o.index, o.cosine, o.max_abs_err, o.rel_err, o.numel
            );
        }
        ok = p.passed;
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
