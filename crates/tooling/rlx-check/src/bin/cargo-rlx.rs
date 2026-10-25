// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx check` — device-free static analysis of an rlx graph.
//!
//! Cargo launches the external subcommand binary as `cargo-rlx rlx <args…>`,
//! passing the subcommand name (`rlx`) as the first argument; we skip it so the
//! tool works both as `cargo rlx check …` and invoked directly as
//! `cargo-rlx check …`.

use std::process::ExitCode;

use rlx_check::{
    CheckOptions, all_backends, check_graph, default_backends, demo, parse_backend,
    parse_graph_json, scaffold,
};

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Drop the `rlx` cargo passes through when invoked as `cargo rlx …`.
    if args.first().map(String::as_str) == Some("rlx") {
        args.remove(0);
    }

    match args.first().map(String::as_str) {
        Some("check") => run_check(&args[1..]),
        Some("debug") => run_debug(&args[1..]),
        Some("new-op") => run_scaffold(&args[1..], Kind::Op),
        Some("new-model") => run_scaffold(&args[1..], Kind::Model),
        None | Some("-h") | Some("--help") | Some("help") => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("cargo-rlx: unknown subcommand '{other}'\n");
            print_help();
            ExitCode::from(2)
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Op,
    Model,
}

/// `cargo rlx new-op <Name> [--stdout] [--force]` (or `new-model`) — write a
/// ready-to-fill extension template.
fn run_scaffold(args: &[String], kind: Kind) -> ExitCode {
    let mut name: Option<String> = None;
    let mut to_stdout = false;
    let mut force = false;
    for a in args {
        match a.as_str() {
            "--stdout" => to_stdout = true,
            "--force" | "-f" => force = true,
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                return usage_err(&format!("unknown flag '{other}'"));
            }
            other => name = Some(other.to_string()),
        }
    }
    let Some(name) = name else {
        return usage_err(match kind {
            Kind::Op => "new-op needs a name, e.g. `cargo rlx new-op GatedGate`",
            Kind::Model => "new-model needs a name, e.g. `cargo rlx new-model GatedMlp`",
        });
    };

    let (file, content) = match kind {
        Kind::Op => scaffold::op_template(&name),
        Kind::Model => scaffold::model_template(&name),
    };

    if to_stdout {
        print!("{content}");
        return ExitCode::SUCCESS;
    }

    if std::path::Path::new(&file).exists() && !force {
        eprintln!("cargo-rlx: '{file}' already exists (use --force to overwrite, or --stdout)");
        return ExitCode::from(2);
    }
    match std::fs::write(&file, content) {
        Ok(()) => {
            let seam = match kind {
                Kind::Op => {
                    "custom op — call `register()` at startup; build with `graph.custom_op(..)`"
                }
                Kind::Model => "LayerStage block — drop into a flow with `.layer_stage(..)`",
            };
            println!("wrote {file}  ({seam})");
            println!("next: see docs/extending.md for the full seam reference");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cargo-rlx: could not write '{file}': {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_check(args: &[String]) -> ExitCode {
    let mut file: Option<String> = None;
    let mut demo_name: Option<String> = None;
    let mut backends: Option<Vec<String>> = None;
    let mut use_all = false;
    let mut as_json = false;
    let mut quiet = false;
    let mut opts = CheckOptions::default();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--demo" => {
                i += 1;
                match args.get(i) {
                    Some(v) => demo_name = Some(v.clone()),
                    None => return usage_err("--demo needs a name (see --list-demos)"),
                }
            }
            "--backend" | "-b" => {
                i += 1;
                match args.get(i) {
                    Some(v) => {
                        backends = Some(v.split(',').map(|s| s.trim().to_string()).collect())
                    }
                    None => return usage_err("--backend needs a comma-separated list"),
                }
            }
            "--all-backends" => use_all = true,
            "--json" => as_json = true,
            "--quiet" | "-q" => quiet = true,
            "--no-fusion" => opts.fusion = false,
            "--no-numeric" => opts.numeric = false,
            "--no-dispatch" => opts.dispatch = false,
            "--list-demos" => {
                println!("demos (use --demo <name>):");
                for (name, about) in demo::DEMOS {
                    println!("  {name:<10} {about}");
                }
                return ExitCode::SUCCESS;
            }
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                return usage_err(&format!("unknown flag '{other}'"));
            }
            other => file = Some(other.to_string()),
        }
        i += 1;
    }

    // Resolve which backends to analyze against.
    if use_all {
        opts.backends = all_backends();
    } else if let Some(names) = backends {
        let mut resolved = Vec::new();
        for n in &names {
            match parse_backend(n) {
                Some(t) => resolved.push(t),
                None => return usage_err(&format!("unknown backend '{n}'")),
            }
        }
        if resolved.is_empty() {
            resolved = default_backends();
        }
        opts.backends = resolved;
    }

    // Resolve the graph to check.
    let graph = match load_graph(demo_name, file) {
        Ok(g) => g,
        Err(code) => return code,
    };

    let report = check_graph(&graph, &opts);

    if as_json {
        println!("{}", report.to_json());
    } else if quiet {
        println!(
            "{}: {} error(s), {} warning(s)",
            report.graph,
            report.errors(),
            report.warnings()
        );
    } else {
        print!("{}", report.render());
    }

    if report.has_errors() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// The graph named by `--demo` or a JSON file path.
fn load_graph(demo_name: Option<String>, file: Option<String>) -> Result<rlx_ir::Graph, ExitCode> {
    match (demo_name, file) {
        (Some(name), _) => demo::build(&name).ok_or_else(|| {
            usage_err(&format!(
                "unknown demo '{name}' — run `cargo rlx check --list-demos`"
            ))
        }),
        (None, Some(path)) => match std::fs::read_to_string(&path) {
            Ok(text) => parse_graph_json(&text).map_err(|e| {
                eprintln!("cargo-rlx: could not parse '{path}' as a graph JSON: {e}");
                ExitCode::FAILURE
            }),
            Err(e) => {
                eprintln!("cargo-rlx: could not read '{path}': {e}");
                Err(ExitCode::FAILURE)
            }
        },
        (None, None) => Err(usage_err(
            "provide a graph JSON file, or --demo <name> (see --list-demos)",
        )),
    }
}

/// `cargo rlx debug <GRAPH.json> --device <dev> [--probe] [--bisect]` — run the
/// graph on a device against CPU and diagnose a failure
/// (`rlx_runtime::debug`).
fn run_debug(args: &[String]) -> ExitCode {
    let mut file: Option<String> = None;
    let mut demo_name: Option<String> = None;
    let mut device: Option<rlx_runtime::Device> = None;
    let mut probe = false;
    let mut bisect = false;
    let mut atol: Option<f32> = None;
    let mut rtol: Option<f32> = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let value = |i: usize, what: &str| -> Result<String, ExitCode> {
            args.get(i)
                .cloned()
                .ok_or_else(|| usage_err(&format!("{a} needs {what}")))
        };
        match a {
            "--device" | "-d" => {
                i += 1;
                let v = match value(i, "a device name") {
                    Ok(v) => v,
                    Err(c) => return c,
                };
                match v.parse() {
                    Ok(d) => device = Some(d),
                    Err(_) => return usage_err(&format!("unknown device '{v}'")),
                }
            }
            "--demo" => {
                i += 1;
                match value(i, "a name (see `cargo rlx check --list-demos`)") {
                    Ok(v) => demo_name = Some(v),
                    Err(c) => return c,
                }
            }
            "--atol" | "--rtol" => {
                i += 1;
                let v = match value(i, "a number") {
                    Ok(v) => v,
                    Err(c) => return c,
                };
                let Ok(x) = v.parse::<f32>() else {
                    return usage_err(&format!("{a}: '{v}' is not a number"));
                };
                if a == "--atol" {
                    atol = Some(x);
                } else {
                    rtol = Some(x);
                }
            }
            "--probe" => probe = true,
            "--bisect" => bisect = true,
            "--all" => {
                probe = true;
                bisect = true;
            }
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                return usage_err(&format!("unknown flag '{other}'"));
            }
            other => file = Some(other.to_string()),
        }
        i += 1;
    }
    let Some(device) = device else {
        return usage_err("debug needs --device <name> (metal, mlx, gpu, cuda, rocm, …)");
    };
    let graph = match load_graph(demo_name, file) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let mut opts = rlx_runtime::debug::DebugOptions::new(device);
    opts.probe = probe;
    opts.bisect = bisect;
    if let Some(x) = atol {
        opts.atol = x;
    }
    if let Some(x) = rtol {
        opts.rtol = x;
    }
    let report = rlx_runtime::debug::debug_graph(&graph, &opts);
    print!("{report}");
    if report.outcome.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn usage_err(msg: &str) -> ExitCode {
    eprintln!("cargo-rlx: {msg}");
    ExitCode::from(2)
}

fn print_help() {
    println!(
        "cargo rlx — rlx developer tools\n\
         \n\
         SUBCOMMANDS:\n\
         \x20   check       device-free static analysis of an rlx graph\n\
         \x20   debug       run a graph on a device vs CPU; classify, probe options, bisect\n\
         \x20   new-op      scaffold a custom op (OpExtension + lower/kernel)\n\
         \x20   new-model   scaffold a LayerStage model block\n\
         \n\
         DEBUG:\n\
         \x20   cargo rlx debug <GRAPH.json> --device <dev> [--probe] [--bisect] [--all]\n\
         \x20                   [--atol X] [--rtol X]   (or --demo <name>)\n\
         \x20   --probe   rerun with one compile option changed at a time\n\
         \x20   --bisect  expose every op as an output; name the first that diverges\n\
         \x20   Backends need their feature: cargo install … --features metal,gpu\n\
         \n\
         SCAFFOLD:\n\
         \x20   cargo rlx new-op <Name>     [--stdout] [--force]\n\
         \x20   cargo rlx new-model <Name>  [--stdout] [--force]\n\
         \n\
         cargo rlx check — device-free static analysis of an rlx graph\n\
         \n\
         USAGE:\n\
         \x20   cargo rlx check <GRAPH.json> [options]\n\
         \x20   cargo rlx check --demo <name>  [options]\n\
         \n\
         INPUT:\n\
         \x20   <GRAPH.json>        an rlx Graph serialized to JSON\n\
         \x20   --demo <name>       use a built-in demo graph\n\
         \x20   --list-demos        list the built-in demos\n\
         \n\
         BACKENDS (op legality is resolved statically — no GPU/driver needed):\n\
         \x20   -b, --backend a,b   analyze only these (cpu,metal,mlx,wgpu,cuda,rocm,tpu)\n\
         \x20   --all-backends      analyze against every target\n\
         \n\
         CHECKS (all on by default):\n\
         \x20   --no-dispatch       skip native/common-ir/unsupported dispatch\n\
         \x20   --no-fusion         skip missed-fusion warnings\n\
         \x20   --no-numeric        skip provable NaN/Inf lint\n\
         \n\
         OUTPUT:\n\
         \x20   --json              structured JSON report\n\
         \x20   -q, --quiet         one-line summary\n\
         \n\
         Exit code is non-zero when any error-level finding is present.\n\
         \n\
         EXAMPLE:\n\
         \x20   cargo rlx check --demo swiglu\n\
         \x20   cargo rlx check model.json -b cuda,metal --json"
    );
}
