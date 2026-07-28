// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo run -p rlx-fpga --bin rlx-fpga-emit -- [options]`
//!
//! ```text
//!   rlx-fpga-emit [--target latency|size|energy|precision|bandwidth]
//!                 [--hw generic|ecp5|ice40|xilinx7:PART]
//!                 [--out DIR]
//! ```
//!
//! Defaults: `--target precision`, `--hw generic` (target-agnostic Verilog),
//! out dir `rlx-fpga/hw/<model>__<target>__<hw>/`.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use rlx_fpga::codegen::emit_with_config;
use rlx_fpga::estimate::estimate;
use rlx_fpga::export_config::{FpgaExportConfig, HwTarget};
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::{optimize, summary};
use rlx_fpga::tune::{OptTarget, Tune};
use rlx_fpga::verilog::mem_hex_bytes;
use rlx_fpga::weights::TEST_IMAGE;

fn main() -> ExitCode {
    let model = tinyconv_mnist_from_cortexm();
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut opt_target = OptTarget::Precision;
    let mut hw = HwTarget::Generic;
    let mut out_dir: Option<PathBuf> = None;
    let mut in_iface = None;
    let mut out_iface = None;
    let mut bind_in: Option<String> = None;
    let mut bind_out: Option<String> = None;
    let mut sidebands: Vec<rlx_fpga::SidebandSpec> = Vec::new();
    let mut bind_sidebands: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--target" | "-t" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --target needs a value");
                    return ExitCode::FAILURE;
                };
                match parse_opt_target(v) {
                    Some(t) => opt_target = t,
                    None => {
                        eprintln!("error: unknown opt target {v:?}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--hw" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --hw needs a value");
                    return ExitCode::FAILURE;
                };
                match HwTarget::parse(v) {
                    Ok(t) => hw = t,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--in-iface" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --in-iface needs a value");
                    return ExitCode::FAILURE;
                };
                match rlx_fpga::InputIface::parse(v) {
                    Ok(t) => in_iface = Some(t),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--out-iface" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --out-iface needs a value");
                    return ExitCode::FAILURE;
                };
                match rlx_fpga::OutputIface::parse(v) {
                    Ok(t) => out_iface = Some(t),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--bind-in" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --bind-in needs a name");
                    return ExitCode::FAILURE;
                };
                bind_in = Some(v.clone());
            }
            "--bind-out" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --bind-out needs a comma-separated list");
                    return ExitCode::FAILURE;
                };
                bind_out = Some(v.clone());
            }
            "--sideband" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --sideband needs name[:bits[:signed]]");
                    return ExitCode::FAILURE;
                };
                match rlx_fpga::SidebandSpec::parse(v) {
                    Ok(s) => sidebands.push(s),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--bind-sideband" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --bind-sideband needs a comma-separated list");
                    return ExitCode::FAILURE;
                };
                for name in v.split(',') {
                    let name = name.trim();
                    if !name.is_empty() {
                        bind_sidebands.push(name.to_string());
                    }
                }
            }
            "--out" | "-o" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: --out needs a path");
                    return ExitCode::FAILURE;
                };
                out_dir = Some(PathBuf::from(v));
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: rlx-fpga-emit [--target T] [--hw HW] [--out DIR]\n\
                       [--in-iface memory|stream|both] [--out-iface scalar|memory|stream|scalar+memory|both]\n\
                       [--bind-in NAME] [--bind-out a,b]\n\
                       [--sideband name[:bits[:signed]]] [--bind-sideband a,b]\n\
                       T  = latency|size|energy|precision|bandwidth\n\
                       HW = generic|ecp5|ecp5:DEV:PKG|ice40|ice40:DEV:PKG|xilinx7:PART"
                );
                return ExitCode::SUCCESS;
            }
            // Positional backward-compat: TARGET [OUT]
            other if !other.starts_with('-') => {
                if let Some(t) = parse_opt_target(other) {
                    opt_target = t;
                    if let Some(p) = args.get(i + 1) {
                        if !p.starts_with('-') && parse_opt_target(p).is_none() {
                            out_dir = Some(PathBuf::from(p));
                            i += 1;
                        }
                    }
                } else if out_dir.is_none() {
                    out_dir = Some(PathBuf::from(other));
                } else {
                    eprintln!("error: unexpected argument {other:?}");
                    return ExitCode::FAILURE;
                }
            }
            other => {
                eprintln!("error: unknown flag {other:?}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let out = out_dir.unwrap_or_else(|| default_out(&model.name, opt_target, &hw));
    let tune = Tune::for_target(opt_target);
    let mut cfg = FpgaExportConfig::default()
        .with_tune(tune)
        .with_hw_target(hw.clone());
    if let Some(iface) = in_iface {
        cfg = cfg.with_input_iface(iface);
    }
    if let Some(iface) = out_iface {
        cfg = cfg.with_output_iface(iface);
    }
    if let Some(name) = bind_in {
        cfg = cfg.bind_input(name);
    }
    if let Some(list) = bind_out {
        let names: Vec<_> = list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        cfg = cfg.bind_outputs(names);
    }
    for sb in sidebands {
        cfg = cfg.sideband(sb);
    }
    if !bind_sidebands.is_empty() {
        cfg = cfg.bind_sideband_inputs(bind_sidebands);
    }

    eprintln!(
        "emitting {} (opt={:?}, hw={}) → {}",
        model.name,
        opt_target,
        hw,
        out.display()
    );
    eprintln!("  {}", cfg.tune);

    let opt = optimize(&model, &cfg.tune);
    eprintln!("  {}", summary(&opt));
    let est = estimate(&opt);
    eprintln!("  {}", est.summary());

    if let Err(e) = emit_with_config(&model, &cfg, &out) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    let mem_path = out.join("tb_image.mem");
    if let Err(e) = fs::write(&mem_path, mem_hex_bytes(TEST_IMAGE)) {
        eprintln!("error writing {}: {e}", mem_path.display());
        return ExitCode::FAILURE;
    }

    eprintln!(
        "done. RTL is target-agnostic; run `bash synth.sh` in {}",
        out.display()
    );
    ExitCode::SUCCESS
}

fn parse_opt_target(s: &str) -> Option<OptTarget> {
    Some(match s.to_ascii_lowercase().as_str() {
        "latency" => OptTarget::Latency,
        "size" => OptTarget::Size,
        "energy" => OptTarget::Energy,
        "precision" => OptTarget::Precision,
        "bandwidth" => OptTarget::Bandwidth,
        _ => return None,
    })
}

fn default_out(model_name: &str, target: OptTarget, hw: &HwTarget) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("hw");
    let tag = format!("{model_name}__{:?}__{}", target, hw.tag()).to_lowercase();
    p.push(tag);
    p
}
