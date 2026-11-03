// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-agent workspace-mcp --dir <folder> [--write <name>]…` — serve a
//! folder to an agent CLI as an MCP server ([`rlx_agent::workspace`]).

use rlx_agent::workspace::Workspace;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("workspace-mcp") => Workspace::from_args(&args[1..]).and_then(|w| w.serve()),
        _ => Err("usage: rlx-agent workspace-mcp --dir <folder> [--write <name>]…".to_string()),
    };
    if let Err(e) = result {
        eprintln!("rlx-agent: {e}");
        std::process::exit(2);
    }
}
