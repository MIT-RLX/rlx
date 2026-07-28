// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Print the curated `RLX_*` environment catalog.
//!
//! ```sh
//! just env-catalog
//! cargo run -p rlx-ir --example env_catalog
//! cargo run -p rlx-ir --example env_catalog -- metal
//! ```

fn main() {
    let group = std::env::args().nth(1);
    print!("{}", rlx_ir::format_env_catalog(group.as_deref()));
}
