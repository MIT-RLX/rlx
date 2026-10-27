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
        Some("doctor") => run_doctor(&args[1..]),
        Some("dist") => rlx_check::launch::main(&args[1..]),
        Some("nodes") => rlx_check::nodes::main(&args[1..]),
        Some("setup") => run_setup(&args[1..]),
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

/// `cargo rlx doctor [--device a,b] [--no-run] [--json]`.
fn run_doctor(args: &[String]) -> ExitCode {
    use rlx_runtime::doctor::{DoctorOptions, EnvVarStatus, doctor};
    let mut opts = DoctorOptions::default();
    let mut as_json = false;
    let mut as_report = false;
    let mut known = false;
    let mut markdown = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => as_json = true,
            "--report" => as_report = true,
            "--known" => known = true,
            "--markdown" => markdown = true,
            "--no-run" => opts.no_run = true,
            "--device" | "-d" => {
                i += 1;
                let Some(list) = args.get(i) else {
                    return usage_err("--device needs a comma-separated list");
                };
                let mut devs = Vec::new();
                for name in list.split(',').map(str::trim) {
                    match name.parse::<rlx_runtime::Device>() {
                        Ok(d) => devs.push(d),
                        Err(_) => return usage_err(&format!("unknown device '{name}'")),
                    }
                }
                opts.devices = Some(devs);
            }
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other => return usage_err(&format!("unknown doctor flag '{other}'")),
        }
        i += 1;
    }
    if known {
        use rlx_runtime::doctor::known_issues::{ISSUES, markdown as known_md};
        if markdown {
            print!("{}", known_md());
        } else {
            for k in ISSUES {
                println!(
                    "{}  [{}{}]  {}\n    symptom: {}\n    fix: {}\n",
                    k.id,
                    k.severity.label(),
                    if k.detect.is_some() {
                        ", detected"
                    } else {
                        ", build time"
                    },
                    k.title,
                    k.symptom,
                    k.fix
                );
            }
        }
        return ExitCode::SUCCESS;
    }
    let report = doctor(&opts);
    if as_report {
        println!("{}", report.to_markdown());
        return if report.healthy() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    if as_json {
        let devices: Vec<serde_json::Value> = report
            .devices
            .iter()
            .map(|d| {
                serde_json::json!({
                    "device": d.device.to_string(),
                    "name": d.name,
                    "feature": d.feature,
                    "status": d.status.label(),
                    "fix": d.fix,
                    "detail": d.detail,
                    "tested": d.tested,
                    "toolchain_warning": d.toolchain_warning,
                    "check": d.check.as_ref().map(|c| serde_json::json!({
                        "passed": c.passed(),
                        "rel_err": c.rel_err,
                        "tolerance": c.tolerance,
                        "millis": c.millis,
                        "error": c.error,
                    })),
                })
            })
            .collect();
        let env: Vec<serde_json::Value> = report
            .env
            .iter()
            .map(|e| {
                let (status, hint) = match &e.status {
                    EnvVarStatus::Known { group } => ("known", Some(group.to_string())),
                    EnvVarStatus::Deprecated { use_instead } => {
                        ("deprecated", Some(use_instead.clone()))
                    }
                    EnvVarStatus::Unknown { did_you_mean } => {
                        ("unknown", did_you_mean.map(str::to_string))
                    }
                };
                serde_json::json!({ "name": e.name, "value": e.value, "status": status, "hint": hint })
            })
            .collect();
        let out = serde_json::json!({
            "version": report.version,
            "os": report.os,
            "arch": report.arch,
            "healthy": report.healthy(),
            "devices": devices,
            "env": env,
            "issues": report.issues.iter().map(|k| serde_json::json!({
                "id": k.id, "severity": k.severity.label(), "title": k.title, "fix": k.fix,
            })).collect::<Vec<_>>(),
            "versions": report.versions.iter().map(|(k, (a, b, c))| (k.to_string(), format!("{a}.{b}.{c}"))).collect::<std::collections::BTreeMap<_, _>>(),
            "implied": report.implied.iter().map(|(n, v, src)| serde_json::json!({
                "name": n, "value": v, "source": src,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!("{report}");
    }
    if report.healthy() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `cargo rlx setup <backend> …`.
fn run_setup(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("xdna") => setup_xdna(&args[1..]),
        Some(other) => usage_err(&format!(
            "no setup for '{other}' yet (available: xdna). For other backends, \
             `cargo rlx doctor` names what is missing and how to install it"
        )),
        None => usage_err("setup needs a backend, e.g. `cargo rlx setup xdna`"),
    }
}

#[cfg(not(feature = "xdna"))]
fn setup_xdna(_args: &[String]) -> ExitCode {
    usage_err(
        "this cargo-rlx was built without XDNA; reinstall with \
         `cargo install --path crates/tooling/rlx-check --features xdna`",
    )
}

#[cfg(feature = "xdna")]
fn setup_xdna(args: &[String]) -> ExitCode {
    use rlx_xdna::{setup, toolchain};
    let mut version: Option<(u32, u32, u32)> = None;
    let mut python: Option<String> = None;
    let mut dry_run = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dry-run" | "-n" => dry_run = true,
            "--version" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return usage_err("--version needs a release, e.g. 1.4.4");
                };
                let p: Vec<u32> = v.split('.').filter_map(|x| x.parse().ok()).collect();
                if p.len() != 3 {
                    return usage_err(&format!("'{v}' is not a release like 1.4.4"));
                }
                version = Some((p[0], p[1], p[2]));
            }
            "--python" => {
                i += 1;
                python = args.get(i).cloned();
            }
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            other => return usage_err(&format!("unknown setup flag '{other}'")),
        }
        i += 1;
    }
    if let Err(e) = setup::platform_supported()
        && !dry_run
    {
        return usage_err(&e);
    }
    let host = toolchain::Host::real();
    let plan = match setup::plan(&host, version, python.as_deref()) {
        Ok(p) => p,
        Err(e) => return usage_err(&e),
    };
    let (a, b, c) = plan.version;
    if let Ok(tc) = toolchain::resolve(&host)
        && tc.version == Some(plan.version)
        && tc.peano_mismatch().is_none()
        && !dry_run
    {
        println!(
            "mlir-aie {a}.{b}.{c} with its pinned Peano is already installed ({}).",
            tc.aiecc.display()
        );
    } else {
        println!(
            "Installing mlir-aie {a}.{b}.{c} + Peano {} into {}",
            plan.peano,
            plan.dir.display()
        );
        for cmd in &plan.commands {
            println!("  $ {}", cmd.join(" "));
            if dry_run {
                continue;
            }
            let ok = std::process::Command::new(&cmd[0])
                .args(&cmd[1..])
                .status()
                .is_ok_and(|s| s.success());
            if !ok {
                eprintln!(
                    "cargo-rlx: `{}` failed. Needs Python 3 with venv/pip and network access to \
                     github.com; on Debian/Ubuntu `sudo apt install python3-venv`. Use --python \
                     to pick another interpreter.",
                    cmd.join(" ")
                );
                return ExitCode::FAILURE;
            }
        }
        if dry_run {
            println!("(dry run — nothing was installed)");
            return ExitCode::SUCCESS;
        }
    }
    // What rlx now resolves (fresh, not the per-process cache).
    match toolchain::resolve(&host) {
        Ok(tc) => {
            println!(
                "toolchain: aiecc {} · Peano {}",
                tc.aiecc.display(),
                tc.peano_version.as_deref().unwrap_or("?")
            );
        }
        Err(e) => {
            eprintln!("cargo-rlx: installed, but rlx still cannot use it:\n{e}");
            return ExitCode::FAILURE;
        }
    }
    if !rlx_xdna::runtime_present() {
        println!("\nThe XRT runtime is still missing (needs root). For this system:");
        for line in setup::xrt_instructions(std::path::Path::new("/")) {
            println!("  {line}");
        }
    }
    println!("\nNext: `cargo rlx doctor --device xdna`");
    ExitCode::SUCCESS
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
         \x20   doctor      what runs on this host: every device, a check run, RLX_* env\n\
         \x20   dist        launch one distributed job on every rank of a cluster file\n\
         \x20   nodes       list nodes advertised on the LAN (Bonjour/Avahi), or advertise this one\n\
         \x20   setup       install a backend's toolchain (no root): `cargo rlx setup xdna`\n\
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
         DOCTOR:\n\
         \x20   cargo rlx doctor [--device a,b] [--no-run] [--json | --report]\n\
         \x20   cargo rlx doctor --known [--markdown]   every known issue and its fix\n\
         \x20   --report  paste-ready Markdown for a GitHub issue (home dir redacted)\n\
         \x20   Reports only the backends this binary was built with:\n\
         \x20   cargo install … --features metal,gpu (or cuda, rocm, vulkan, xdna…)\n\
         \x20   Exit code is non-zero when an available device fails its check run\n\
         \x20   or an RLX_* variable is unknown.\n\
         \n\
         SETUP:\n\
         \x20   cargo rlx setup xdna [--version 1.4.4] [--python python3] [--dry-run]\n\
         \x20   Installs a matrix-tested mlir-aie and the Peano build it pins into\n\
         \x20   ~/.local/share/rlx/toolchains (found automatically), then prints what\n\
         \x20   still needs root (the XRT runtime) for this distro.\n\
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
