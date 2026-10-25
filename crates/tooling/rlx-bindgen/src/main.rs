// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-bindgen` — write the generated bindings; `rlx-bindgen --check` — fail
//! if any checked-in one is stale.

use std::process::ExitCode;

fn main() -> ExitCode {
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
