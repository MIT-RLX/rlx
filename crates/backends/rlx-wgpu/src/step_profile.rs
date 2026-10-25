// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Per-step GPU time for wgpu** — the twin of `RLX_METAL_THUNK_PROFILE`.
//!
//! Enable with `RLX_WGPU_STEP_PROFILE=1`. Every schedule step then runs in its
//! own compute pass (`RLX_WGPU_ONE_OP_PER_PASS` semantics) and the pass records
//! GPU timestamps at its start and end (`TIMESTAMP_QUERY`, requested from the
//! adapter only in this mode). Times are device ticks scaled by the queue's
//! timestamp period, so they exclude host encode, submit and readback — the
//! part a kernel change can move. Sums per step kind across runs and prints a
//! table every `RLX_WGPU_STEP_PROFILE_EVERY` runs (default 32).
//!
//! What it is not: production timing. One op per pass inserts a barrier per
//! op, so the total is an upper bound on the batched schedule; use it to rank
//! steps, not to quote throughput. Host-staged steps get no timestamps (they
//! are not GPU passes) — the table says how many passes it timed.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Query pairs per run; passes beyond this are run untimed.
/// wgpu caps a QuerySet at 4096 queries (two per pass).
const MAX_PASSES: u32 = 2048;

struct Gpu {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
}

#[derive(Default)]
struct RunState {
    /// Label per started pass (index = pass number this run).
    labels: Vec<&'static str>,
    /// Pass count already resolved into `resolve`/`readback`.
    resolved: u32,
}

#[derive(Default, Clone, Copy)]
struct Stat {
    count: u64,
    ns: f64,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
static RUN: Mutex<RunState> = Mutex::new(RunState {
    labels: Vec::new(),
    resolved: 0,
});
static STATS: Mutex<Option<HashMap<&'static str, Stat>>> = Mutex::new(None);
static RUNS: Mutex<(u64, u64)> = Mutex::new((0, 0)); // (runs, timed passes)

/// Requested by `RLX_WGPU_STEP_PROFILE=1`.
pub fn requested() -> bool {
    rlx_ir::env::flag("RLX_WGPU_STEP_PROFILE")
}

/// Extra device feature to request when profiling and the adapter offers it.
pub(crate) fn required_feature(adapter: &wgpu::Adapter) -> wgpu::Features {
    if requested() && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    }
}

fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(|| {
        if !requested() {
            return None;
        }
        let dev = crate::device::wgpu_device()?;
        if !dev
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            eprintln!(
                "rlx-wgpu: RLX_WGPU_STEP_PROFILE=1 but the adapter has no TIMESTAMP_QUERY — \
                 no per-step GPU times"
            );
            return None;
        }
        let n = MAX_PASSES * 2;
        let bytes = u64::from(n) * 8;
        Some(Gpu {
            queries: dev.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("rlx-wgpu step profile"),
                ty: wgpu::QueryType::Timestamp,
                count: n,
            }),
            resolve: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu step profile resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu step profile readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: dev.queue.get_timestamp_period(),
        })
    })
    .as_ref()
}

/// Profiling is live for this run (requested and supported).
pub(crate) fn active() -> bool {
    requested() && gpu().is_some()
}

pub(crate) fn begin_run() {
    if !active() {
        return;
    }
    let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
    r.labels.clear();
    r.resolved = 0;
}

/// Timestamp writes for the next compute pass, or `None` when inactive / full.
pub(crate) fn pass_writes() -> Option<wgpu::ComputePassTimestampWrites<'static>> {
    let g = gpu()?;
    if !requested() {
        return None;
    }
    let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let i = r.labels.len() as u32;
    if i >= MAX_PASSES {
        return None;
    }
    r.labels.push("(no dispatch)");
    Some(wgpu::ComputePassTimestampWrites {
        query_set: &g.queries,
        beginning_of_pass_write_index: Some(2 * i),
        end_of_pass_write_index: Some(2 * i + 1),
    })
}

/// Leak-intern a label. The set of distinct labels is bounded by the distinct
/// step shapes in the compiled schedules, so this does not grow per run.
pub(crate) fn intern(s: String) -> &'static str {
    static SEEN: Mutex<Option<std::collections::HashSet<&'static str>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let set = g.get_or_insert_with(Default::default);
    if let Some(&hit) = set.get(s.as_str()) {
        return hit;
    }
    let leaked: &'static str = Box::leak(s.into_boxed_str());
    set.insert(leaked);
    leaked
}

/// Name the most recently started pass after the step it dispatched.
pub(crate) fn label_last(name: &'static str) {
    if !active() {
        return;
    }
    let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = r.labels.last_mut() {
        *l = name;
    }
}

/// Resolve this encoder's new passes; call before `enc.finish()`.
pub(crate) fn resolve(enc: &mut wgpu::CommandEncoder) {
    let Some(g) = gpu() else { return };
    if !requested() {
        return;
    }
    let mut r = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let (lo, hi) = (r.resolved, r.labels.len() as u32);
    if hi <= lo {
        return;
    }
    // `resolve_query_set` needs a 256-aligned destination, so always resolve
    // at 0 and copy into this run's slot range (copies need only 4-byte
    // alignment). Queue order keeps a later section's resolve behind this copy.
    enc.resolve_query_set(&g.queries, 2 * lo..2 * hi, &g.resolve, 0);
    enc.copy_buffer_to_buffer(
        &g.resolve,
        0,
        &g.readback,
        u64::from(lo) * 16,
        u64::from(hi - lo) * 16,
    );
    r.resolved = hi;
}

/// After the run's GPU work has completed: read the timestamps and accumulate.
pub(crate) fn end_run() {
    let Some(g) = gpu() else { return };
    if !requested() {
        return;
    }
    let Some(dev) = crate::device::wgpu_device() else {
        return;
    };
    let labels = {
        let r = RUN.lock().unwrap_or_else(|e| e.into_inner());
        r.labels[..r.resolved as usize].to_vec()
    };
    if labels.is_empty() {
        return;
    }
    let bytes = labels.len() as u64 * 16;
    let slice = g.readback.slice(0..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    if rx.recv().ok().and_then(Result::ok).is_none() {
        return;
    }
    let ticks: Vec<u64> = {
        let view = slice.get_mapped_range().expect("mapped");
        bytemuck::cast_slice::<u8, u64>(&view).to_vec()
    };
    g.readback.unmap();
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let map = stats.get_or_insert_with(HashMap::new);
    let mut timed = 0u64;
    for (i, name) in labels.iter().enumerate() {
        let (b, e) = (ticks[2 * i], ticks[2 * i + 1]);
        if e <= b {
            continue;
        }
        let s = map.entry(name).or_default();
        s.count += 1;
        s.ns += (e - b) as f64 * f64::from(g.period_ns);
        timed += 1;
    }
    drop(stats);
    let runs = {
        let mut c = RUNS.lock().unwrap_or_else(|e| e.into_inner());
        c.0 += 1;
        c.1 += timed;
        c.0
    };
    let every = rlx_ir::env::parse_or("RLX_WGPU_STEP_PROFILE_EVERY", 32u64).max(1);
    if runs.is_multiple_of(every) {
        print_summary();
    }
}

/// Print the cumulative per-step-kind GPU time table.
pub fn print_summary() {
    let stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = stats.as_ref() else { return };
    let (runs, passes) = *RUNS.lock().unwrap_or_else(|e| e.into_inner());
    let total: f64 = map.values().map(|s| s.ns).sum();
    let mut rows: Vec<_> = map.iter().collect();
    rows.sort_by(|a, b| b.1.ns.partial_cmp(&a.1.ns).expect("finite"));
    eprintln!(
        "[rlx-wgpu] step profile: {runs} runs, {passes} timed passes, {:.2} ms GPU/run \
         (one op per pass — rank, don't quote)",
        total / 1e6 / runs.max(1) as f64
    );
    eprintln!(
        "  {:<28} {:>8} {:>10} {:>8} {:>6}",
        "step", "per run", "ms/run", "us/op", "pct"
    );
    for (name, s) in rows {
        eprintln!(
            "  {:<28} {:>8.1} {:>10.3} {:>8.1} {:>5.1}%",
            name,
            s.count as f64 / runs.max(1) as f64,
            s.ns / 1e6 / runs.max(1) as f64,
            s.ns / 1e3 / s.count.max(1) as f64,
            100.0 * s.ns / total.max(1.0)
        );
    }
}
