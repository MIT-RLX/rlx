// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MPSGraph / hybrid dispatch timing (`RLX_METAL_MPS_PROFILE=1`).
//!
//! Complements [`thunk_profile`] (per-thunk isolation). Records wall time
//! for full-graph MPSGraph runs, hybrid sub-graph steps, and thunk batches.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

static STATS: Mutex<Option<HashMap<String, SampleStats>>> = Mutex::new(None);

#[derive(Default, Clone)]
struct SampleStats {
    count: u64,
    total_ns: u128,
}

pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_MPS_PROFILE")
}

pub fn reset() {
    if enabled() {
        *STATS.lock().unwrap() = Some(HashMap::new());
    }
}

pub fn record(label: impl Into<String>, dt: Duration) {
    if !enabled() {
        return;
    }
    let label = label.into();
    let mut guard = STATS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let e = map.entry(label).or_default();
    e.count += 1;
    e.total_ns += dt.as_nanos();
}

pub fn print_summary() {
    if !enabled() {
        return;
    }
    let guard = STATS.lock().unwrap();
    let Some(map) = guard.as_ref() else {
        return;
    };
    if map.is_empty() {
        eprintln!("[rlx-metal] mps profile: no samples");
        return;
    }
    let total_ns: u128 = map.values().map(|s| s.total_ns).sum();
    let mut rows: Vec<_> = map
        .iter()
        .map(|(name, s)| {
            let ms = s.total_ns as f64 / 1e6;
            let pct = if total_ns > 0 {
                100.0 * s.total_ns as f64 / total_ns as f64
            } else {
                0.0
            };
            (name.clone(), s.count, ms, pct)
        })
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!(
        "[rlx-metal] mps profile (GPU-sync wall, {:.2} ms total):",
        total_ns as f64 / 1e6
    );
    eprintln!("{:<40} {:>6} {:>10} {:>7}", "label", "count", "ms", "pct");
    eprintln!("{}", "-".repeat(68));
    for (name, count, ms, pct) in &rows {
        eprintln!("{name:<40} {count:>6} {ms:>10.2} {pct:>6.1}%");
    }
}
