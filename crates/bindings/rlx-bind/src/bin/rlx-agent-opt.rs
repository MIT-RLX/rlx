// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-agent-opt`: the command line over agent runs
//! ([`rlx_bind::runtime::agent::cli`]).

use rlx_bind::runtime::agent::cli::{CliDefaults, main_with};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(main_with(&args, &CliDefaults::default()));
}
