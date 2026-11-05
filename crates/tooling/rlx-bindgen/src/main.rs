// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-bindgen` — write the generated bindings; `rlx-bindgen --check` — fail
//! if any checked-in one is stale; `rlx-bindgen foreign ...` — a foreign C
//! library's bindings from its header (see `foreign`).

use std::process::ExitCode;

/// `rlx-bindgen foreign --header H --lib L --name N [--env E] [--export M]
/// [--define D]... [--strip P]... --out DIR [--check]`: a foreign C
/// library's bindings, from its header
fn foreign(args: &[String]) -> ExitCode {
    let mut header = None;
    let mut out = None;
    let mut check = false;
    let mut cfg = rlx_bindgen::foreign::Config {
        lib: String::new(),
        name: String::new(),
        env: String::new(),
        header_name: String::new(),
        export: "extern".into(),
        defines: Vec::new(),
        strip: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_default();
        match a.as_str() {
            "--header" => header = Some(val()),
            "--lib" => cfg.lib = val(),
            "--name" => cfg.name = val(),
            "--env" => cfg.env = val(),
            "--export" => cfg.export = val(),
            "--define" => cfg.defines.push(val()),
            "--strip" => cfg.strip.push(val()),
            "--out" => out = Some(std::path::PathBuf::from(val())),
            "--check" => check = true,
            _ => {
                eprintln!("rlx-bindgen foreign: unknown argument {a}");
                return ExitCode::FAILURE;
            }
        }
    }
    let (Some(header), Some(out)) = (header, out) else {
        eprintln!(
            "usage: rlx-bindgen foreign --header H --lib L --name N [--env E] [--export M] [--define D]... [--strip P]... --out DIR [--check]"
        );
        return ExitCode::FAILURE;
    };
    if cfg.lib.is_empty() || cfg.name.is_empty() {
        eprintln!("rlx-bindgen foreign: --lib and --name are needed");
        return ExitCode::FAILURE;
    }
    if cfg.env.is_empty() {
        cfg.env = format!("{}_LIBRARY", cfg.lib.to_uppercase());
    }
    let text = match std::fs::read_to_string(&header) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("rlx-bindgen foreign: {header}: {e}");
            return ExitCode::FAILURE;
        }
    };
    cfg.header_name = std::path::Path::new(&header)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or(header.clone());
    let mut stale = false;
    for (path, text) in rlx_bindgen::foreign::generate(&text, &cfg) {
        let full = out.join(&path);
        if std::fs::read_to_string(&full).ok().as_deref() == Some(text.as_str()) {
            continue;
        }
        if check {
            eprintln!("  stale: {}", full.display());
            stale = true;
            continue;
        }
        if let Some(dir) = full.parent() {
            std::fs::create_dir_all(dir).expect("creating output directory");
        }
        std::fs::write(&full, text).unwrap_or_else(|e| panic!("writing {}: {e}", full.display()));
        println!("rlx-bindgen foreign: wrote {}", full.display());
    }
    if stale {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("foreign") {
        return foreign(&args[1..]);
    }
    let check = std::env::args().any(|a| a == "--check");
    let root = rlx_bindgen::workspace_root();
    if check {
        let stale = rlx_bindgen::stale(&root);
        if stale.is_empty() {
            println!("rlx-bindgen: all generated bindings are up to date");
            return ExitCode::SUCCESS;
        }
        eprintln!("rlx-bindgen: stale generated bindings — run `just gen-bindings`:");
        for p in stale {
            eprintln!("  {}", p.display());
        }
        return ExitCode::FAILURE;
    }
    for (path, text) in rlx_bindgen::outputs(&root) {
        let full = root.join(&path);
        if std::fs::read_to_string(&full).ok().as_deref() == Some(text.as_str()) {
            continue;
        }
        if let Some(dir) = full.parent() {
            std::fs::create_dir_all(dir).expect("creating output directory");
        }
        std::fs::write(&full, text).unwrap_or_else(|e| panic!("writing {}: {e}", full.display()));
        println!("rlx-bindgen: wrote {}", path.display());
    }
    ExitCode::SUCCESS
}
