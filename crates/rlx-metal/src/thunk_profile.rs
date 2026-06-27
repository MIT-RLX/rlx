// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
    /// Wall-clock (encode + commit + wait) — includes per-commit sync latency.
    total_ns: u128,
    /// True GPU-busy (GPUEndTime - GPUStartTime) — device execution only.
    gpu_ns: u128,
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
    record_split(name, dt, Duration::ZERO);
}

/// Record one thunk sample with both wall time and true GPU-busy time.
pub fn record_split(name: &'static str, wall: Duration, gpu: Duration) {
    if !enabled() {
        return;
    }
    let mut guard = STATS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let e = map.entry(name).or_default();
    e.count += 1;
    e.total_ns += wall.as_nanos();
    e.gpu_ns += gpu.as_nanos();
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
    let total_wall: u128 = map.values().map(|s| s.total_ns).sum();
    let total_gpu: u128 = map.values().map(|s| s.gpu_ns).sum();
    // Attribution uses GPU-busy when available (it has no per-commit sync
    // distortion); fall back to wall if no GPU timestamps were captured.
    let by_gpu = total_gpu > 0;
    let denom = if by_gpu { total_gpu } else { total_wall } as f64;
    let mut rows: Vec<_> = map
        .iter()
        .map(|(name, s)| {
            let gpu_ms = s.gpu_ns as f64 / 1e6;
            let wall_ms = s.total_ns as f64 / 1e6;
            let key = if by_gpu { s.gpu_ns } else { s.total_ns } as f64;
            let pct = if denom > 0.0 {
                100.0 * key / denom
            } else {
                0.0
            };
            (*name, s.count, gpu_ms, wall_ms, pct)
        })
        .collect();
    rows.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!(
        "[rlx-metal] thunk profile — GPU-busy {:.2} ms | wall {:.2} ms (sync overhead {:.2} ms), sorted by {}:",
        total_gpu as f64 / 1e6,
        total_wall as f64 / 1e6,
        (total_wall.saturating_sub(total_gpu)) as f64 / 1e6,
        if by_gpu { "GPU-busy" } else { "wall" }
    );
    eprintln!(
        "{:<32} {:>6} {:>10} {:>10} {:>7}",
        "thunk", "count", "gpu_ms", "wall_ms", "pct"
    );
    eprintln!("{}", "-".repeat(70));
    let mut buckets: HashMap<&'static str, (f64, f64)> = HashMap::new();
    for (name, count, gpu_ms, wall_ms, pct) in &rows {
        eprintln!("{name:<32} {count:>6} {gpu_ms:>10.2} {wall_ms:>10.2} {pct:>6.1}%");
        let bucket = bucket_name(name);
        let e = buckets.entry(bucket).or_default();
        e.0 += *gpu_ms;
        e.1 += *wall_ms;
    }
    let mut bucket_rows: Vec<_> = buckets.into_iter().collect();
    let bkey = |b: &(f64, f64)| if by_gpu { b.0 } else { b.1 };
    bucket_rows.sort_by(|a, b| {
        bkey(&b.1)
            .partial_cmp(&bkey(&a.1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\n[rlx-metal] buckets (gpu_ms / wall_ms):");
    for (name, (gpu_ms, wall_ms)) in bucket_rows {
        let pct = if denom > 0.0 {
            100.0 * (if by_gpu { gpu_ms } else { wall_ms }) / (denom / 1e6)
        } else {
            0.0
        };
        eprintln!("  {name:<28} {gpu_ms:>10.2} {wall_ms:>10.2} ({pct:>5.1}%)");
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
