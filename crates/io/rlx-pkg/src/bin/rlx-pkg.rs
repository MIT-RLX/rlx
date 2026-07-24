// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `rlx-pkg` — inspect / re-tier / verify / import GGUF into `.rlxp`.
//!
//! ```text
//! rlx-pkg inspect <path.rlxp|dir|zip>
//! rlx-pkg verify <path>
//! rlx-pkg tier <path> -o <out.rlxp> [--warm SUBSTR...] [--hot NAME...]
//! rlx-pkg import-gguf <model.gguf> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg convert <in> -o <out> [--container flat|zip|dir]
//! ```
//!
//! ONNX → RLXP (optional executable graph) lives on `rlx-bake --features onnx`.

use anyhow::{Context, Result, bail};
use rlx_pkg::{
    AutoTierOptions, ContainerKind, Package, PackedWeight, StorageTier, WriteOptions,
    apply_auto_tier, gguf_to_rlxp, infer_container, verify_package, write_package,
};
use rlx_pkg::{GgufImportOptions, MaterializeMode};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage(argv0: &str) -> String {
    format!(
        "usage:
  {argv0} inspect <path.rlxp|dir|.zip>
  {argv0} verify <path>
  {argv0} tier <path> -o <out> [--warm SUB]... [--hot NAME]... [--warm-min-mib N]
  {argv0} import-gguf <model.gguf> -o <out.rlxp> [--no-graph] [--no-auto-tier] [--container flat|zip|dir]
  {argv0} convert <in> -o <out> [--container flat|zip|dir]

ONNX → RLXP (optional graph): rlx-bake --features onnx -- import-onnx …
"
    )
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args: Vec<String> = env::args().collect();
    let argv0 = args.first().cloned().unwrap_or_else(|| "rlx-pkg".into());
    if args.len() < 2 {
        eprint!("{}", usage(&argv0));
        bail!("missing command");
    }
    let cmd = args.remove(1);
    match cmd.as_str() {
        "inspect" => cmd_inspect(&args)?,
        "verify" => cmd_verify(&args)?,
        "tier" => cmd_tier(&args)?,
        "import-gguf" => cmd_import_gguf(&args)?,
        "convert" => cmd_convert(&args)?,
        "-h" | "--help" | "help" => print!("{}", usage(&argv0)),
        other => bail!("unknown command {other}\n{}", usage(&argv0)),
    }
    Ok(())
}

fn cmd_inspect(args: &[String]) -> Result<()> {
    let path = args.first().context("inspect <path>")?;
    let pack = Package::open(path)?;
    let m = pack.manifest();
    println!("name: {}", m.name);
    println!("format: {} v{} (compat {})", m.format, m.format_version, m.compat_version);
    println!("producer: {}", m.producer.as_deref().unwrap_or("-"));
    println!("features: {}", m.features.join(", "));
    println!("graph: {} ({})", m.graph.path, m.graph.encoding);
    if let Some(idx) = pack.weights_index() {
        println!("tensors: {}", idx.tensors.len());
        let mut hot = 0usize;
        let mut warm = 0usize;
        let mut cold = 0usize;
        let mut stored = 0u64;
        let mut raw = 0u64;
        for t in &idx.tensors {
            match t.tier {
                StorageTier::Hot => hot += 1,
                StorageTier::Warm => warm += 1,
                StorageTier::Cold => cold += 1,
            }
            stored += t.length;
            raw += t.raw_length.unwrap_or(t.length);
        }
        println!("  hot={hot} warm={warm} cold={cold}");
        println!("  stored_bytes={stored} raw_bytes={raw}");
        for t in idx.tensors.iter().take(32) {
            println!(
                "  - {}  {:?} {:?}  stored={} raw={}",
                t.name,
                t.tier,
                t.codec,
                t.length,
                t.raw_length.unwrap_or(t.length)
            );
        }
        if idx.tensors.len() > 32 {
            println!("  … {} more", idx.tensors.len() - 32);
        }
    } else {
        println!("tensors: (none)");
    }
    println!("sidecars: {}", m.sidecars.len());
    for s in &m.sidecars {
        println!("  - {} ({:?})", s.id, s.media_type);
    }
    Ok(())
}

fn cmd_verify(args: &[String]) -> Result<()> {
    let path = args.first().context("verify <path>")?;
    let pack = Package::open(path)?;
    let report = verify_package(&pack)?;
    println!(
        "ok: tensors {}/{} (unchecked {}) sidecars {}",
        report.tensors_ok,
        report.tensors_checked,
        report.tensors_unchecked,
        report.sidecars_ok
    );
    Ok(())
}

fn take_flag_values(args: &mut Vec<String>, flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            out.push(args[i + 1].clone());
            args.remove(i);
            args.remove(i);
            continue;
        }
        i += 1;
    }
    out
}

fn take_opt(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            let v = args[i + 1].clone();
            args.remove(i);
            args.remove(i);
            return Some(v);
        }
        i += 1;
    }
    None
}

fn take_bool(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(pos) = args.iter().position(|a| a == flag) {
        args.remove(pos);
        true
    } else {
        false
    }
}

fn parse_container(s: &str) -> Result<ContainerKind> {
    Ok(match s {
        "flat" => ContainerKind::Flat,
        "zip" => ContainerKind::Zip,
        "dir" => ContainerKind::Dir,
        other => bail!("unknown container {other}"),
    })
}

fn cmd_tier(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("tier requires -o <out>")?;
    let warm_subs = take_flag_values(&mut args, "--warm");
    let hot_names = take_flag_values(&mut args, "--hot");
    let warm_min = take_opt(&mut args, "--warm-min-mib")
        .map(|s| s.parse::<usize>().unwrap_or(16) << 20)
        .unwrap_or(16 << 20);
    let path = args.first().context("tier <path>")?;
    let pack = Package::open(path)?;
    let graph = if pack.has_graph() {
        pack.graph_with(MaterializeMode::All)?
    } else {
        rlx_ir::Graph::new(pack.manifest().name.as_str())
    };
    let idx = pack.weights_index().context("no weights")?;
    let mut weights = Vec::new();
    for t in &idx.tensors {
        let data = pack.tensor_bytes(&t.name)?;
        weights.push(PackedWeight {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scheme: t.scheme.clone(),
            layout: t.layout.clone(),
            data,
            rank: t.rank,
            tier: StorageTier::Hot,
        });
    }
    let mut auto = AutoTierOptions {
        hot_names: hot_names.into_iter().collect::<HashSet<_>>(),
        warm_substrings: if warm_subs.is_empty() {
            AutoTierOptions::default().warm_substrings
        } else {
            warm_subs
        },
        warm_min_bytes: warm_min,
    };
    // keep CLI-forced hot names
    let _ = &mut auto;
    apply_auto_tier(&mut weights, &auto);
    let opts = WriteOptions {
        name: pack.manifest().name.clone(),
        producer: Some("rlx-pkg/tier".into()),
        container: infer_container(std::path::Path::new(&out), None)?,
        include_graph: pack.has_graph(),
        ..WriteOptions::default()
    };
    write_package(&out, &graph, &weights, &opts)?;
    println!("wrote {out}");
    Ok(())
}

fn cmd_import_gguf(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("import-gguf requires -o")?;
    let no_graph = take_bool(&mut args, "--no-graph");
    let no_auto = take_bool(&mut args, "--no-auto-tier");
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or(ContainerKind::Flat);
    let gguf = args.first().context("import-gguf <model.gguf>")?;
    let opts = GgufImportOptions {
        container,
        include_graph: !no_graph,
        compress_sidecars: true,
        auto_tier: !no_auto,
    };
    gguf_to_rlxp(gguf, &out, &opts)?;
    println!("wrote {out}");
    Ok(())
}

fn cmd_convert(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("convert requires -o")?;
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or_else(|| {
            infer_container(std::path::Path::new(&out), None).unwrap_or(ContainerKind::Flat)
        });
    let path = args.first().context("convert <in>")?;
    let pack = Package::open(path)?;
    let graph = if pack.has_graph() {
        pack.graph_with(MaterializeMode::All)?
    } else {
        rlx_ir::Graph::new(pack.manifest().name.as_str())
    };
    let mut weights = Vec::new();
    if let Some(idx) = pack.weights_index() {
        for t in &idx.tensors {
            weights.push(PackedWeight {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scheme: t.scheme.clone(),
                layout: t.layout.clone(),
                data: pack.tensor_bytes(&t.name)?,
                rank: t.rank,
                tier: t.tier,
            });
        }
    }
    let mut opts = WriteOptions {
        name: pack.manifest().name.clone(),
        producer: Some("rlx-pkg/convert".into()),
        container,
        include_graph: pack.has_graph(),
        ..WriteOptions::default()
    };
    for sc in &pack.manifest().sidecars {
        opts.sidecars
            .push((sc.id.clone(), sc.media_type.clone().unwrap_or_default(), pack.sidecar(&sc.id)?));
    }
    if let Some(pl) = pack.placement() {
        opts.placement = Some(pl.clone());
    }
    write_package(PathBuf::from(&out), &graph, &weights, &opts)?;
    println!("wrote {out}");
    Ok(())
}
