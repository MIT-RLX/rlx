// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! True per-op GPU time via Metal counter-sample buffers (timestamp counter
//! set), captured WITHIN one command buffer.
//!
//! Why this exists: [`crate::thunk_profile`] commits+waits each thunk in its
//! own command buffer — so every sample eats per-command-buffer GPU ramp-up
//! and CPU sync, inflating short ops. [`crate::gpu_time`] gives only the
//! whole-buffer span. This module wraps each thunk in its OWN compute encoder
//! with start/end timestamp sample attachments, all in ONE command buffer:
//! the GPU's own timestamps give each op's true on-device duration with no
//! per-buffer ramp and no CPU/commit distortion — the only way to tell, in the
//! real decode, whether a thunk is genuinely expensive or the step is just
//! gap/stall-bound.
//!
//! Apple Silicon (M-series) supports `AtStageBoundary` timestamps only (not
//! `AtDispatchBoundary` / mid-encoder `sampleCountersInBuffer`), hence the
//! one-encoder-per-thunk approach. Enable with `RLX_METAL_COUNTER_PROFILE=1`.
//!
//! Ticks→ms calibration is self-contained: the encoders run serially, so the
//! sum of per-encoder tick spans corresponds to the command buffer's measured
//! GPU-busy time (`GPUEndTime - GPUStartTime`).

use objc::{msg_send, sel, sel_impl};
use std::collections::HashMap;

pub fn enabled() -> bool {
    rlx_ir::env::flag("RLX_METAL_COUNTER_PROFILE")
}

/// Resolve `count` timestamp samples (u64 each) from the sample buffer.
/// metal-rs 0.30 lacks `resolve_counter_range`, so call `resolveCounterRange:`
/// + NSData `getBytes:length:` directly. The timestamp counter set resolves to
/// `MTLCounterResultTimestamp { uint64 timestamp; }` — one u64 per sample.
pub fn resolve(sbuf: &metal::CounterSampleBufferRef, count: u64) -> Vec<u64> {
    let mut data = vec![0u64; count as usize];
    if count == 0 {
        return data;
    }
    let range = metal::NSRange::new(0, count);
    let total_bytes = (count as usize * std::mem::size_of::<u64>()) as u64;
    unsafe {
        let ns_data: *mut objc::runtime::Object = msg_send![sbuf, resolveCounterRange: range];
        if ns_data.is_null() {
            return data;
        }
        let _: () = msg_send![ns_data, getBytes: data.as_mut_ptr() length: total_bytes];
    }
    data
}

/// Find the timestamp counter set, if the device exposes one and supports
/// stage-boundary sampling.
pub fn timestamp_counter_set(dev: &metal::DeviceRef) -> Option<metal::CounterSet> {
    if !dev.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    dev.counter_sets()
        .into_iter()
        .find(|s| s.name().to_lowercase().contains("timestamp"))
}

/// Aggregate resolved timestamps (2 per encoder: start, end) by thunk name and
/// print a per-op GPU-busy summary. `names[i]` is the thunk for encoder `i`;
/// `ticks` holds `2*names.len()` resolved counter values. `busy_ms` is the
/// command buffer's measured GPU-busy time, used to calibrate ticks→ms.
pub fn aggregate_and_print(names: &[&'static str], ticks: &[u64], busy_ms: f64) {
    if names.is_empty() || ticks.len() < names.len() * 2 {
        eprintln!("[rlx-metal] counter profile: no samples");
        return;
    }
    // Calibrate ticks→ms by the TOTAL timeline span (first op's start → last
    // op's end), which corresponds to the command buffer's GPU-busy window.
    // (Calibrating by the sum of op spans would force gap=0 and hide stalls.)
    let n = names.len();
    let first_start = ticks[0] as f64;
    let last_end = ticks[(n - 1) * 2 + 1] as f64;
    let total_span = (last_end - first_start).max(0.0);
    let ms_per_tick = if total_span > 0.0 {
        busy_ms / total_span
    } else {
        0.0
    };

    let mut by_name: HashMap<&'static str, (u64, f64)> = HashMap::new();
    let mut total_ms = 0.0;
    for (i, name) in names.iter().enumerate() {
        let s = ticks[i * 2] as f64;
        let e = ticks[i * 2 + 1] as f64;
        let span = if e > s { e - s } else { 0.0 };
        let ms = span * ms_per_tick;
        total_ms += ms;
        let entry = by_name.entry(name).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += ms;
    }

    let mut rows: Vec<_> = by_name.into_iter().collect();
    rows.sort_by(|a, b| {
        b.1.1
            .partial_cmp(&a.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!(
        "[rlx-metal] COUNTER profile (true per-op GPU-busy, in-buffer; {:.2} ms ops / {:.2} ms buffer GPU-busy):",
        total_ms, busy_ms
    );
    eprintln!(
        "{:<32} {:>6} {:>10} {:>7}",
        "thunk", "count", "gpu_ms", "pct"
    );
    eprintln!("{}", "-".repeat(60));
    for (name, (count, ms)) in &rows {
        let pct = if busy_ms > 0.0 {
            100.0 * ms / busy_ms
        } else {
            0.0
        };
        eprintln!("{name:<32} {count:>6} {ms:>10.2} {pct:>6.1}%");
    }
    let gap = busy_ms - total_ms;
    eprintln!(
        "  (gap/stall between ops: {:.2} ms = {:.1}% of GPU-busy)",
        gap,
        if busy_ms > 0.0 {
            100.0 * gap / busy_ms
        } else {
            0.0
        }
    );
}
