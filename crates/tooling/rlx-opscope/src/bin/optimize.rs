// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-optimize` — rank optimization opportunities by fusing op **cost**,
//! **temporal data-recurrence** (does the input repeat over time?), and
//! **decomposability** (linear ops → memoize / delta-compute). Reads a
//! multi-step sketch CSV + its `site,flops` sidecar.
//!
//! Usage: `opscope-optimize <sketches.csv> [sketches.csv.sites]`

use rlx_opscope::optimize::{mine_opportunities, report};

fn main() -> std::io::Result<()> {
    let csv = std::env::args()
        .nth(1)
        .expect("usage: opscope-optimize <csv> [sidecar]");
    let sidecar = std::env::args()
        .nth(2)
        .unwrap_or_else(|| format!("{csv}.sites"));
    let opps = mine_opportunities(&csv, &sidecar)?;
    if opps.is_empty() {
        println!("(no multi-step sites — feed a CSV with >1 step, e.g. from opscope-replay)");
    } else {
        report(&opps);
    }
    Ok(())
}
