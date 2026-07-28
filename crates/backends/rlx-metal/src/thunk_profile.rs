// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-thunk GPU timing for Metal backward/forward tuning.
//!
//! Enable with `RLX_METAL_THUNK_PROFILE=1`. Runs each schedule thunk in
//! isolation (sequential commits with `wait_until_completed`) and prints
//! aggregate wall/GPU time by thunk name. Slow vs production dispatch but
//! accurate relative costs.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

static STATS: Mutex<Option<HashMap<&'static str, ThunkStats>>> = Mutex::new(None);

#[derive(Default, Clone)]
struct ThunkStats {
    count: u64,
    total_ns: u128,
}

pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_THUNK_PROFILE")
}

pub fn reset() {
    if enabled() {
        *STATS.lock().unwrap() = Some(HashMap::new());
    }
}

pub fn record(name: &'static str, dt: Duration) {
    if !enabled() {
        return;
    }
    let mut guard = STATS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let e = map.entry(name).or_default();
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
        eprintln!("[rlx-metal] thunk profile: no samples");
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
            (*name, s.count, ms, pct)
        })
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!(
        "[rlx-metal] thunk profile (GPU-sync wall time, {:.2} ms total):",
        total_ns as f64 / 1e6
    );
    eprintln!("{:<32} {:>6} {:>10} {:>7}", "thunk", "count", "ms", "pct");
    eprintln!("{}", "-".repeat(60));
    let mut buckets: HashMap<&'static str, f64> = HashMap::new();
    for (name, count, ms, pct) in &rows {
        eprintln!("{name:<32} {count:>6} {ms:>10.2} {pct:>6.1}%");
        let bucket = bucket_name(name);
        *buckets.entry(bucket).or_default() += *ms;
    }
    let mut bucket_rows: Vec<_> = buckets.into_iter().collect();
    bucket_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n[rlx-metal] buckets:");
    for (name, ms) in bucket_rows {
        let pct = 100.0 * ms / (total_ns as f64 / 1e6);
        eprintln!("  {name:<28} {ms:>10.2} ms ({pct:>5.1}%)");
    }
}

fn bucket_name(thunk: &str) -> &'static str {
    if thunk.contains("conv2d_backward") {
        "conv_backward"
    } else if thunk.contains("attention") {
        "attention"
    } else if thunk.contains("sgemm") || thunk.contains("fused_mm") || thunk.contains("matmul") {
        "matmul"
    } else if thunk.contains("rms_norm") || thunk.contains("layer_norm") {
        "norm"
    } else if thunk.contains("rope") {
        "rope"
    } else if thunk.contains("binary") || thunk.contains("activation") {
        "elemwise"
    } else {
        "other"
    }
}
