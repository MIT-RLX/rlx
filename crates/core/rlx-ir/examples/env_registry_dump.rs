// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Dump the full `RLX_*` registry as markdown (used by docs tooling).
//!
//! ```sh
//! cargo run -p rlx-ir --example env_registry_dump
//! ```

fn main() {
    print!("{}", rlx_ir::format_registry_markdown());
}
