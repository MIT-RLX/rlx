// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! CPU-side per-thunk *encode* timing for the production dispatch path.
//!
//! Distinct from [`crate::thunk_profile`] (which commits+waits each thunk in
//! isolation, measuring GPU work) and [`crate::gpu_time`] (whole-buffer
//! GPU-busy). This times only the CPU cost of building each thunk's commands
//! inside `encode_commit`'s main loop — the `match` arm that calls
//! `set_pipeline`/`set_buffer`/`dispatch`. When a decode step is encode-bound
//! (GPU idle, wall ≫ GPU-busy), this is the breakdown that says *which* thunk
//! type's encoding dominates.
//!
//! Enable with `RLX_METAL_ENCODE_PROFILE=1`. The per-step summary is reset and
//! printed for each full-range `encode_commit` (i.e. each decode/prefill step).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static STATS: Mutex<Option<HashMap<&'static str, (u64, u128)>>> = Mutex::new(None);

pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_ENCODE_PROFILE")
}

pub fn reset() {
    if enabled() {
        *STATS.lock().unwrap() = Some(HashMap::new());
    }
}

fn record(name: &'static str, dt: Duration) {
    let mut guard = STATS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let e = map.entry(name).or_insert((0, 0));
    e.0 += 1;
    e.1 += dt.as_nanos();
}

/// RAII timer: starts on construction, records elapsed encode time on drop —
/// so it captures the arm cost even when the match arm `continue`s early.
pub struct EncodeTimer {
    name: &'static str,
    t0: Instant,
}

impl EncodeTimer {
    #[inline]
    pub fn new(name: &'static str) -> Option<Self> {
        if enabled() {
            Some(Self {
                name,
                t0: Instant::now(),
            })
        } else {
            None
        }
    }
}

impl Drop for EncodeTimer {
    fn drop(&mut self) {
        record(self.name, self.t0.elapsed());
    }
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
        return;
    }
    let total_ns: u128 = map.values().map(|(_, ns)| *ns).sum();
    let mut rows: Vec<_> = map
        .iter()
        .map(|(name, (count, ns))| (*name, *count, *ns as f64 / 1e6))
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "[rlx-metal] ENCODE profile (CPU encode time, {:.2} ms total):",
        total_ns as f64 / 1e6
    );
    eprintln!("{:<32} {:>6} {:>10} {:>7}", "thunk", "count", "ms", "pct");
    eprintln!("{}", "-".repeat(60));
    for (name, count, ms) in &rows {
        let pct = if total_ns > 0 {
            100.0 * (ms * 1e6) / total_ns as f64
        } else {
            0.0
        };
        eprintln!("{name:<32} {count:>6} {ms:>10.2} {pct:>6.1}%");
    }
}
