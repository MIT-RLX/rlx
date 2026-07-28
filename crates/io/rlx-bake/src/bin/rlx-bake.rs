// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-bake` — merge graph + weights into an optimized `*.rlx` / `.rlxp` file.
//!
//! ```text
//! rlx-bake <graph|hir|bundle> -o model.rlx [--weights …] [--opt PROFILE]
//! rlx-bake <graph|hir|bundle> -o model.rlxp --format rlxp [--weights …]
//! rlx-bake convert model.rlx -o model.rlxp
//! # with --features onnx:
//! rlx-bake import-onnx model.onnx -o model.rlxp [--no-graph]
//! # with --features encrypt:
//! rlx-bake … -o model.rlx --password …
//! rlx-bake decrypt <encrypted.rlx> -o plain.rlx --password …
//! ```

use anyhow::{Context, Result, bail};
use rlx_bake::{
    BakeOptions, BakeProfile, MemoryMode, WeightLoadPolicy, bake_bundle, convert_rlx_to_rlxp,
    load_graph, load_weights, write_rlx, write_rlxp,
};
use rlx_pkg::ContainerKind;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

fn usage(argv0: &str) -> String {
    let opt_help = "\
             \t\t[--opt merge|fold|exact|size] [--memory duplex|runtime|compact] \\\n\
             \t\t[--format rlx|rlxp] [--container flat|zip|dir] [--no-skip] [--no-ternary] [--quant|--no-quant] \\\n\
             \t\t[--no-unfold] [--no-fold] [--no-dce] [--no-simplify] \\\n\
             \t\t[--dedupe-constants|--no-dedupe-constants] \\\n\
             \t\t[--keep-folded-bindings|--no-folded-bindings]";
    #[cfg(feature = "encrypt")]
    {
        format!(
            "usage:\n  {argv0} <graph.json|model.hir.json|bundle-dir> -o <out.rlx|out.rlxp> \\\n\
             \t\t[--weights path] [--weights-policy f32|packed|auto] \\\n\
             {opt_help}\n\
             \t{argv0} convert <in.rlx> -o <out.rlxp>\n\
             \t{argv0} import-onnx <in.onnx> -o <out.rlxp> [--no-graph] [--container flat|zip|dir]\n\
             \t{argv0} decrypt <encrypted.rlx> -o <plain.rlx> \\\n\
             \t\t--password SECRET | --password-env VAR\n\n\
             profiles (--opt):\n\
             \tmerge   package only (same dense MatMul)\n\
             \tfold    fold weight-only math; keep dense GEMM\n\
             \texact   lossless skip + ternary + cleanup (default)\n\
             \tsize    exact + Q8_0 pack remaining matmul weights\n\
             weights-policy:\n\
             \tf32     f32-first GO — decode all weights to f32 (needed for exact/size)\n\
             \tpacked  f32-first NO-GO when packs/half exist — keep MLX/DDUF encoding\n\
             \tauto    NO-GO if MLX packs (or DDUF half) and ternary/quant off; else GO\n\
             memory (--memory):\n\
             \tduplex  weight bytes in graph and table (duplicate)\n\
             \truntime bytes in graph only; table is metadata\n\
             \tcompact bytes in table only; materialize before compile (default for exact/size)\n\
             format (--format):\n\
             \trlx     RLXBAKE1 single-file artifact (default for .rlx)\n\
             \trlxp    flat mmap package (default for .rlxp; use --container zip|dir to override)\n\
             fine flags override the profile after --opt is applied."
        )
    }
    #[cfg(not(feature = "encrypt"))]
    {
        format!(
            "usage: {argv0} <graph.json|model.hir.json|bundle-dir> -o <out.rlx|out.rlxp> \\\n\
             \t\t[--weights path] [--weights-policy f32|packed|auto] \\\n\
             {opt_help}\n\
             \t{argv0} convert <in.rlx> -o <out.rlxp>\n\
             \t{argv0} import-onnx <in.onnx> -o <out.rlxp> [--no-graph]  (needs `--features onnx`)\n\
             \t(encryption: rebuild with `--features encrypt`)\n\n\
             profiles (--opt):\n\
             \tmerge   package only (same dense MatMul)\n\
             \tfold    fold weight-only math; keep dense GEMM\n\
             \texact   lossless skip + ternary + cleanup (default)\n\
             \tsize    exact + Q8_0 pack remaining matmul weights\n\
             weights-policy:\n\
             \tf32     f32-first GO — decode all weights to f32 (needed for exact/size)\n\
             \tpacked  f32-first NO-GO when packs/half exist — keep MLX/DDUF encoding\n\
             \tauto    NO-GO if MLX packs (or DDUF half) and ternary/quant off; else GO\n\
             memory (--memory):\n\
             \tduplex  weight bytes in graph and table (duplicate)\n\
             \truntime bytes in graph only; table is metadata\n\
             \tcompact bytes in table only; materialize before compile (default for exact/size)\n\
             format (--format):\n\
             \trlx     RLXBAKE1 single-file artifact (default for .rlx)\n\
             \trlxp    flat mmap package (default for .rlxp; use --container zip|dir to override)\n\
             fine flags override the profile after --opt is applied."
        )
    }
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

#[cfg(feature = "encrypt")]
fn resolve_password(
    password: Option<String>,
    password_env: Option<String>,
) -> Result<Option<String>> {
    match (password, password_env) {
        (Some(_), Some(_)) => bail!("pass only one of --password / --password-env"),
        (Some(p), None) => Ok(Some(p)),
        (None, Some(var)) => {
            let v = env::var(&var).with_context(|| format!("reading password from env {var}"))?;
            if v.is_empty() {
                bail!("env {var} is empty");
            }
            Ok(Some(v))
        }
        (None, None) => Ok(None),
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let argv0 = args.first().map(|s| s.as_str()).unwrap_or("rlx-bake");
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage(argv0));
        return Ok(());
    }
    if args.len() < 2 {
        bail!("{}", usage(argv0));
    }

    if args[1] == "convert" {
        return run_convert(argv0, &args[2..]);
    }

    if args[1] == "import-onnx" {
        #[cfg(feature = "onnx")]
        {
            return run_import_onnx(argv0, &args[2..]);
        }
        #[cfg(not(feature = "onnx"))]
        {
            bail!("import-onnx requires rebuilding with `--features onnx`");
        }
    }

    #[cfg(feature = "encrypt")]
    if args[1] == "decrypt" {
        return run_decrypt(argv0, &args[2..]);
    }
    #[cfg(not(feature = "encrypt"))]
    if args[1] == "decrypt" {
        bail!("decrypt requires rebuilding with `--features encrypt`");
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut weights: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut container: Option<ContainerKind> = None;
    #[cfg(feature = "encrypt")]
    let mut password: Option<String> = None;
    #[cfg(feature = "encrypt")]
    let mut password_env: Option<String> = None;
    let mut opts = BakeOptions::default();
    let mut weights_policy = WeightLoadPolicy::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let p = args
                    .get(i)
                    .with_context(|| format!("{} requires a path", args[i - 1]))?;
                output = Some(PathBuf::from(p));
            }
            "--weights" | "-w" => {
                i += 1;
                let p = args
                    .get(i)
                    .with_context(|| format!("{} requires a path", args[i - 1]))?;
                weights = Some(PathBuf::from(p));
            }
            "--weights-policy" => {
                i += 1;
                let name = args
                    .get(i)
                    .with_context(|| "--weights-policy requires f32|packed|auto")?;
                weights_policy = WeightLoadPolicy::parse(name)?;
            }
            "--format" => {
                i += 1;
                format = Some(
                    args.get(i)
                        .with_context(|| "--format requires rlx|rlxp")?
                        .clone(),
                );
            }
            "--container" => {
                i += 1;
                let name = args
                    .get(i)
                    .with_context(|| "--container requires flat|zip|dir")?;
                container = Some(match name.as_str() {
                    "flat" => ContainerKind::Flat,
                    "zip" => ContainerKind::Zip,
                    "dir" => ContainerKind::Dir,
                    other => bail!("unknown --container {other} (expected flat|zip|dir)"),
                });
            }
            "--opt" | "--profile" => {
                i += 1;
                let name = args
                    .get(i)
                    .with_context(|| "--opt requires a profile name")?;
                let profile = BakeProfile::from_str(name).map_err(|e| anyhow::anyhow!(e))?;
                opts = BakeOptions::from_profile(profile);
            }
            "--memory" => {
                i += 1;
                let name = args
                    .get(i)
                    .with_context(|| "--memory requires duplex|runtime|compact")?;
                opts.memory = MemoryMode::from_str(name).map_err(|e| anyhow::anyhow!(e))?;
            }
            "--dedupe-constants" => opts.dedupe_constants = true,
            "--no-dedupe-constants" => opts.dedupe_constants = false,
            "--keep-folded-bindings" => opts.keep_folded_bindings = true,
            "--no-folded-bindings" => opts.keep_folded_bindings = false,
            #[cfg(feature = "encrypt")]
            "--password" => {
                i += 1;
                password = Some(
                    args.get(i)
                        .with_context(|| "--password requires a value")?
                        .clone(),
                );
            }
            #[cfg(feature = "encrypt")]
            "--password-env" => {
                i += 1;
                password_env = Some(
                    args.get(i)
                        .with_context(|| "--password-env requires a var name")?
                        .clone(),
                );
            }
            #[cfg(not(feature = "encrypt"))]
            "--password" | "--password-env" => {
                bail!("--password requires rebuilding with `--features encrypt`");
            }
            "--no-skip" => opts.skip_zero = false,
            "--skip" => opts.skip_zero = true,
            "--no-ternary" => opts.ternary = false,
            "--ternary" => opts.ternary = true,
            "--quant" => opts.quant = true,
            "--no-quant" => opts.quant = false,
            "--no-unfold" => opts.unfold = false,
            "--unfold" => opts.unfold = true,
            "--no-fold" => opts.constant_folding = false,
            "--fold" => opts.constant_folding = true,
            "--no-dce" => opts.dce = false,
            "--dce" => opts.dce = true,
            "--no-simplify" => opts.algebraic_simplify = false,
            "--simplify" => opts.algebraic_simplify = true,
            "--no-cleanup" => {
                opts.constant_folding = false;
                opts.dce = false;
                opts.algebraic_simplify = false;
            }
            other if other.starts_with('-') => {
                bail!("unknown flag {other}\n{}", usage(argv0));
            }
            other => {
                if input.is_some() {
                    bail!("unexpected argument {other}\n{}", usage(argv0));
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }

    let input = input.with_context(|| format!("missing input\n{}", usage(argv0)))?;
    let output = output.with_context(|| format!("missing -o <out>\n{}", usage(argv0)))?;

    let use_rlxp = match format.as_deref() {
        Some("rlxp") => true,
        Some("rlx") => false,
        Some(other) => bail!("unknown --format {other} (expected rlx|rlxp)"),
        None => output
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "rlxp"),
    };

    let loaded = load_graph(&input)?;
    let weights_path = weights.or(loaded.default_weights.clone());
    let mut bundle = rlx_bake::WeightBundle::default();
    let had_weights_file = weights_path.is_some();
    if let Some(wp) = &weights_path {
        bundle = load_weights(wp, weights_policy, &opts)?;
        if let Some(v) = &bundle.verdict {
            let tag = if v.is_go() {
                "f32-first GO"
            } else {
                "f32-first NO-GO"
            };
            eprintln!("{tag}: {}", v.reason());
        }
        eprintln!(
            "loaded {} f32 + {} packed/native tensors from {}",
            bundle.f32.len(),
            bundle.packed.len(),
            wp.display()
        );
    }

    eprintln!(
        "bake profile={} ({})  memory={} ({})  format={}  skip={} ternary={} quant={} \
         fold={} dce={} simplify={} unfold={} dedupe_const={} keep_folded={}",
        opts.profile,
        opts.profile.description(),
        opts.memory,
        opts.memory.description(),
        if use_rlxp { "rlxp" } else { "rlx" },
        opts.skip_zero,
        opts.ternary,
        opts.quant,
        opts.constant_folding,
        opts.dce,
        opts.algebraic_simplify,
        opts.unfold,
        opts.dedupe_constants,
        opts.keep_folded_bindings,
    );

    let (file, report) = bake_bundle(&loaded.graph, &bundle, &opts);

    if had_weights_file
        && report.params_baked == 0
        && !bundle.f32.is_empty()
        && bundle.packed.is_empty()
    {
        bail!(
            "weights file provided but zero params matched graph params (remaining: {:?})",
            report.params_remaining
        );
    }

    if use_rlxp {
        #[cfg(feature = "encrypt")]
        {
            let password = resolve_password(password, password_env)?;
            if password.is_some() {
                bail!(
                    "--password with --format rlxp is not supported yet; bake to .rlx then convert, or omit encryption"
                );
            }
        }
        write_rlxp(&output, &file, container)?;
    } else {
        #[cfg(feature = "encrypt")]
        {
            let password = resolve_password(password, password_env)?;
            if let Some(pw) = &password {
                rlx_bake::write_rlx_encrypted(&output, &file, pw)?;
                eprintln!("encrypted with ChaCha20-Poly1305 (Argon2id)");
            } else {
                write_rlx(&output, &file)?;
            }
        }
        #[cfg(not(feature = "encrypt"))]
        {
            write_rlx(&output, &file)?;
        }
    }

    eprintln!(
        "baked {} → {}  nodes {}→{}  params_baked={}  weights={} ({} bytes)  \
         skip={} ternary={} quant={}  mem[strip_graph={} strip_table={} dedupe={} dropped_folded={}]",
        loaded.source,
        output.display(),
        report.nodes_before,
        report.nodes_after,
        report.params_baked,
        report.weight_count,
        report.weight_bytes,
        report.optimize.skipped_zero_matmuls,
        report.optimize.ternary_packed,
        report.optimize.quant_packed,
        report.memory.graph_bytes_stripped,
        report.memory.table_bytes_stripped,
        report.memory.constants_deduped,
        report.memory.folded_bindings_dropped,
    );
    if !report.params_remaining.is_empty() {
        eprintln!("  unbound params: {:?}", report.params_remaining);
    }
    for w in &file.weights {
        eprintln!(
            "  weight {}: {} {:?} ({} bytes) — {}",
            w.name,
            w.encoding,
            w.shape,
            w.data.len(),
            w.note
        );
    }
    Ok(())
}

fn run_convert(argv0: &str, args: &[String]) -> Result<()> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let p = args
                    .get(i)
                    .with_context(|| format!("{} requires a path", args[i - 1]))?;
                output = Some(PathBuf::from(p));
            }
            other if other.starts_with('-') => {
                bail!("unknown flag {other}\n{}", usage(argv0));
            }
            other => {
                if input.is_some() {
                    bail!("unexpected argument {other}\n{}", usage(argv0));
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    let input = input.with_context(|| format!("missing .rlx input\n{}", usage(argv0)))?;
    let output = output.with_context(|| format!("missing -o <out.rlxp>\n{}", usage(argv0)))?;
    convert_rlx_to_rlxp(&input, &output, None)?;
    eprintln!("converted {} → {}", input.display(), output.display());
    Ok(())
}

#[cfg(feature = "onnx")]
fn run_import_onnx(argv0: &str, args: &[String]) -> Result<()> {
    use rlx_bake::{OnnxImportOptions, onnx_to_rlxp};
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut include_graph = true;
    let mut container = ContainerKind::Flat;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let p = args
                    .get(i)
                    .with_context(|| format!("{} requires a path", args[i - 1]))?;
                output = Some(PathBuf::from(p));
            }
            "--no-graph" => include_graph = false,
            "--container" => {
                i += 1;
                let name = args
                    .get(i)
                    .with_context(|| "--container requires flat|zip|dir")?;
                container = match name.as_str() {
                    "flat" => ContainerKind::Flat,
                    "zip" => ContainerKind::Zip,
                    "dir" => ContainerKind::Dir,
                    other => bail!("unknown --container {other}"),
                };
            }
            other if other.starts_with('-') => {
                bail!("unknown flag {other}\n{}", usage(argv0));
            }
            other => {
                if input.is_some() {
                    bail!("unexpected argument {other}\n{}", usage(argv0));
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    let input = input.with_context(|| format!("missing <in.onnx>\n{}", usage(argv0)))?;
    let output = output.with_context(|| format!("missing -o <out.rlxp>\n{}", usage(argv0)))?;
    let opts = OnnxImportOptions {
        container,
        include_graph,
        ..OnnxImportOptions::default()
    };
    onnx_to_rlxp(&input, &output, &opts)?;
    eprintln!(
        "imported {} → {} (graph={})",
        input.display(),
        output.display(),
        include_graph
    );
    Ok(())
}

#[cfg(feature = "encrypt")]
fn run_decrypt(argv0: &str, args: &[String]) -> Result<()> {
    use rlx_bake::{read_rlx_with_password, write_rlx};

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut password: Option<String> = None;
    let mut password_env: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                let p = args
                    .get(i)
                    .with_context(|| format!("{} requires a path", args[i - 1]))?;
                output = Some(PathBuf::from(p));
            }
            "--password" => {
                i += 1;
                password = Some(
                    args.get(i)
                        .with_context(|| "--password requires a value")?
                        .clone(),
                );
            }
            "--password-env" => {
                i += 1;
                password_env = Some(
                    args.get(i)
                        .with_context(|| "--password-env requires a var name")?
                        .clone(),
                );
            }
            other if other.starts_with('-') => {
                bail!("unknown flag {other}\n{}", usage(argv0));
            }
            other => {
                if input.is_some() {
                    bail!("unexpected argument {other}\n{}", usage(argv0));
                }
                input = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    let input = input.with_context(|| format!("missing encrypted input\n{}", usage(argv0)))?;
    let output = output.with_context(|| format!("missing -o <plain.rlx>\n{}", usage(argv0)))?;
    let password = resolve_password(password, password_env)?
        .with_context(|| "decrypt requires --password or --password-env")?;

    let file = read_rlx_with_password(&input, &password)?;
    write_rlx(&output, &file)?;
    eprintln!(
        "decrypted {} → {}  ({} weights, {} nodes)",
        input.display(),
        output.display(),
        file.weights.len(),
        file.graph.len()
    );
    Ok(())
}
