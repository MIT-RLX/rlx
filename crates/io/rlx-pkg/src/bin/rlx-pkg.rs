// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-pkg` — inspect / re-tier / verify / import GGUF into `.rlxp`.
//!
//! ```text
//! rlx-pkg inspect <path.rlxp|dir|zip>
//! rlx-pkg verify <path>
//! rlx-pkg tier <path> -o <out.rlxp> [--warm SUBSTR...] [--hot NAME...]
//! rlx-pkg import-gguf <model.gguf> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg import-mlx <dir|file> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg import-dduf <model.dduf> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg import-nemo <model.nemo> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg import-pt <model.pt|pth|bin> -o <out.rlxp> [--no-graph] [--no-auto-tier]
//! rlx-pkg convert <in> -o <out> [--container flat|zip|dir]
//! rlx-pkg build <in.rlxp> -o <out.rlxp> --target metal [--target cpu]… [--weights f16|bf16|f32] [--no-gqa]
//! ```
//!
//! ONNX → RLXP (optional executable graph) lives on `rlx-bake --features onnx`.

use anyhow::{Context, Result, bail};
use rlx_pkg::{
    AutoTierOptions, ContainerKind, Package, PackedWeight, StorageTier, WriteOptions,
    apply_auto_tier, dduf_to_rlxp, gguf_to_rlxp, infer_container, mlx_to_rlxp, nemo_to_rlxp,
    pt_to_rlxp, verify_package, write_package,
};
use rlx_pkg::{
    DdufImportOptions, GgufImportOptions, MaterializeMode, MlxImportOptions, NemoImportOptions,
    PtImportOptions,
};
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
  {argv0} import-mlx <dir|safetensors|npz|npy> -o <out.rlxp> [--no-graph] [--no-auto-tier] [--keep-packed] [--container flat|zip|dir]
  {argv0} import-dduf <model.dduf> -o <out.rlxp> [--no-graph] [--no-auto-tier] [--container flat|zip|dir]
  {argv0} import-nemo <model.nemo> -o <out.rlxp> [--no-graph] [--no-auto-tier] [--container flat|zip|dir]
  {argv0} import-pt <model.pt|pth|bin> -o <out.rlxp> [--no-graph] [--no-auto-tier] [--container flat|zip|dir]
  {argv0} convert <in> -o <out> [--container flat|zip|dir]
  {argv0} build <in.rlxp> -o <out.rlxp> --target metal|cpu [--target …] [--weights f16|bf16|f32] [--no-gqa]

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
    // Drop argv0 as well as the subcommand: every handler below indexes `args` from its
    // own first operand, so leaving the program path in would shift all of them by one.
    args.remove(0);
    let cmd = args.remove(0);
    match cmd.as_str() {
        "inspect" => cmd_inspect(&args)?,
        "verify" => cmd_verify(&args)?,
        "tier" => cmd_tier(&args)?,
        "import-gguf" => cmd_import_gguf(&args)?,
        "import-mlx" => cmd_import_mlx(&args)?,
        "import-dduf" => cmd_import_dduf(&args)?,
        "import-nemo" => cmd_import_nemo(&args)?,
        "import-pt" => cmd_import_pt(&args)?,
        "convert" => cmd_convert(&args)?,
        "build" => cmd_build(&args)?,
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
    println!(
        "format: {} v{} (compat {})",
        m.format, m.format_version, m.compat_version
    );
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
    if m.graph.encoding == "rlxmodel_v1" {
        print_module(&pack.module()?);
    }
    Ok(())
}

/// A `.rlxmodel` module as `inspect` shows it: who wrote it and who may
/// read it, its symbols, each function's inputs and outputs, the model
/// block's entries, and where the weights it does not hold come from.
fn print_module(module: &rlx_ir::rlxmodel::Module) {
    use rlx_ir::Op;
    use serde_json::Value;
    let versions = [
        module.rlx.as_ref().map(|v| format!("written by rlx {v}")),
        module
            .requires
            .as_ref()
            .map(|v| format!("needs rlx >= {v}")),
    ];
    let versions: Vec<String> = versions.into_iter().flatten().collect();
    if !versions.is_empty() {
        println!("rlxmodel: {}", versions.join(", "));
    }
    let model = module.model.clone().unwrap_or(Value::Null);
    let s = |k: &str| model.get(k).and_then(Value::as_str).unwrap_or("-");
    if !model.is_null() {
        println!(
            "model: {} {} (family {}, task {}, license {})",
            s("name"),
            s("version"),
            s("family"),
            s("task"),
            model.get("license").map_or("-".into(), |l| match l {
                Value::String(l) => l.clone(),
                other => other.to_string(),
            })
        );
    }
    if !module.dims.is_empty() {
        let dims: Vec<String> = module
            .dims
            .iter()
            .map(|(n, i)| format!("{n}={i}"))
            .collect();
        println!("dims: {}", dims.join(", "));
    }
    // A symbolic dim by the name the module gives it (`?0` → `cap`), the
    // longest ids first so `?1` does not eat `?10`.
    let mut names = module.dims.clone();
    names.sort_by_key(|(_, i)| std::cmp::Reverse(*i));
    let shape = |sh: &rlx_ir::Shape| {
        let mut t = sh.to_string();
        for (n, i) in &names {
            t = t.replace(&format!("?{i}"), n);
        }
        t
    };
    println!("functions: {}", module.graphs.len());
    for g in &module.graphs {
        let ins: Vec<String> = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(format!("{name} {}", shape(&n.shape))),
                _ => None,
            })
            .collect();
        let outs: Vec<String> = g
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                let n = g.node(o);
                let label = n.name.clone().unwrap_or_else(|| i.to_string());
                format!("{label} {}", shape(&n.shape))
            })
            .collect();
        let params = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Param { .. }))
            .count();
        let vars = if g.vars().is_empty() {
            String::new()
        } else {
            format!(", {} variables", g.vars().len())
        };
        println!(
            "  - {} ({} nodes, {params} params{vars})",
            g.name,
            g.nodes().len()
        );
        println!("      in:  {}", ins.join(", "));
        println!("      out: {}", outs.join(", "));
    }
    if let Some(entries) = model.get("entries").and_then(Value::as_object) {
        for (name, e) in entries {
            // Named apart from its kind: `image (pipeline)`.
            let kind = match e.get("kind").and_then(Value::as_str) {
                Some(k) if k != name => format!("{name} ({k})"),
                _ => name.clone(),
            };
            let mut parts = Vec::new();
            for key in ["functions", "function"] {
                if let Some(f) = e.get(key) {
                    parts.push(format!("{key} {f}"));
                }
            }
            if let Some(sizes) = e.get("sizes").and_then(Value::as_array) {
                parts.push(format!("{} sizes", sizes.len()));
            }
            for key in ["prefill", "check", "media", "batch", "decoder", "stages"] {
                if e.get(key).is_some_and(|v| !v.is_null()) {
                    parts.push(key.to_string());
                }
            }
            if let Some(c) = e.get("max_context") {
                parts.push(format!("context {c}"));
            }
            println!("entry {kind}: {}", parts.join(", "));
        }
    }
    if let Some(sources) = model.get("weights").and_then(Value::as_array) {
        println!("weight sources: {}", sources.len());
        for w in sources {
            let mapped = w
                .get("map")
                .and_then(Value::as_object)
                .map_or(0, |m| m.len());
            println!(
                "  - {} ({} params renamed or transposed, {} bytes)",
                w.get("uri").and_then(Value::as_str).unwrap_or("-"),
                mapped,
                w.get("size").and_then(Value::as_u64).unwrap_or(0)
            );
        }
    }
}

fn cmd_verify(args: &[String]) -> Result<()> {
    let path = args.first().context("verify <path>")?;
    let pack = Package::open(path)?;
    let report = verify_package(&pack)?;
    println!(
        "ok: tensors {}/{} (unchecked {}) sidecars {}",
        report.tensors_ok, report.tensors_checked, report.tensors_unchecked, report.sidecars_ok
    );
    Ok(())
}

/// Each portable function also built for every `--target`, the device's
/// recipe adjusted by `--weights` / `--no-gqa`.
fn cmd_build(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let targets = take_flag_values(&mut args, "--target");
    let out = take_flag_values(&mut args, "-o")
        .pop()
        .context("build <in.rlxp> -o <out.rlxp> --target <device>")?;
    let weights = take_flag_values(&mut args, "--weights").pop();
    let no_gqa = args.iter().any(|a| a == "--no-gqa");
    args.retain(|a| a != "--no-gqa");
    let input = args
        .first()
        .context("build <in.rlxp> -o <out.rlxp> --target <device>")?;
    if targets.is_empty() {
        bail!("build needs --target (metal, cpu)");
    }
    let mut recipes = Vec::new();
    for t in &targets {
        let mut r = rlx_pkg::build::TargetRecipe::for_target(t)
            .with_context(|| format!("no build recipe for target '{t}' (metal, cpu)"))?;
        match weights.as_deref() {
            None => {}
            Some("f16") => r.weight_dtype = Some(rlx_ir::DType::F16),
            Some("bf16") => r.weight_dtype = Some(rlx_ir::DType::BF16),
            Some("f32") => r.weight_dtype = None,
            Some(other) => bail!("--weights {other}: f16, bf16 or f32"),
        }
        if no_gqa {
            r.gqa_native = false;
        }
        recipes.push((t.as_str(), r));
    }
    let report = rlx_pkg::build::build_for_targets(
        std::path::Path::new(input),
        std::path::Path::new(&out),
        &recipes,
    )?;
    println!(
        "built {} ({} matmul weights narrowed, {} attentions on shared K/V heads) -> {out}",
        if report.built.is_empty() {
            "nothing".to_string()
        } else {
            report.built.join(", ")
        },
        report.weights_narrowed,
        report.attention_rewired
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

fn cmd_import_mlx(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("import-mlx requires -o")?;
    let no_graph = take_bool(&mut args, "--no-graph");
    let no_auto = take_bool(&mut args, "--no-auto-tier");
    let keep_packed = take_bool(&mut args, "--keep-packed");
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or(ContainerKind::Flat);
    let path = args.first().context("import-mlx <dir|file>")?;
    let opts = MlxImportOptions {
        container,
        include_graph: !no_graph,
        compress_sidecars: true,
        auto_tier: !no_auto,
        dequant_to_f32: !keep_packed,
        ..Default::default()
    };
    mlx_to_rlxp(path, &out, &opts)?;
    println!("wrote {out}");
    Ok(())
}

fn cmd_import_dduf(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("import-dduf requires -o")?;
    let no_graph = take_bool(&mut args, "--no-graph");
    let no_auto = take_bool(&mut args, "--no-auto-tier");
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or(ContainerKind::Flat);
    let path = args.first().context("import-dduf <model.dduf>")?;
    let opts = DdufImportOptions {
        container,
        include_graph: !no_graph,
        compress_sidecars: true,
        auto_tier: !no_auto,
    };
    dduf_to_rlxp(path, &out, &opts)?;
    println!("wrote {out}");
    Ok(())
}

fn cmd_import_nemo(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("import-nemo requires -o")?;
    let no_graph = take_bool(&mut args, "--no-graph");
    let no_auto = take_bool(&mut args, "--no-auto-tier");
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or(ContainerKind::Flat);
    let path = args.first().context("import-nemo <model.nemo>")?;
    let opts = NemoImportOptions {
        container,
        include_graph: !no_graph,
        compress_sidecars: true,
        auto_tier: !no_auto,
    };
    nemo_to_rlxp(path, &out, &opts)?;
    println!("wrote {out}");
    Ok(())
}

fn cmd_import_pt(args: &[String]) -> Result<()> {
    let mut args = args.to_vec();
    let out = take_opt(&mut args, "-o").context("import-pt requires -o")?;
    let no_graph = take_bool(&mut args, "--no-graph");
    let no_auto = take_bool(&mut args, "--no-auto-tier");
    let container = take_opt(&mut args, "--container")
        .map(|s| parse_container(&s))
        .transpose()?
        .unwrap_or(ContainerKind::Flat);
    let path = args.first().context("import-pt <model.pt|pth|bin>")?;
    let opts = PtImportOptions {
        container,
        include_graph: !no_graph,
        compress_sidecars: true,
        auto_tier: !no_auto,
    };
    pt_to_rlxp(path, &out, &opts)?;
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
        opts.sidecars.push((
            sc.id.clone(),
            sc.media_type.clone().unwrap_or_default(),
            pack.sidecar(&sc.id)?,
        ));
    }
    if let Some(pl) = pack.placement() {
        opts.placement = Some(pl.clone());
    }
    write_package(PathBuf::from(&out), &graph, &weights, &opts)?;
    println!("wrote {out}");
    Ok(())
}
