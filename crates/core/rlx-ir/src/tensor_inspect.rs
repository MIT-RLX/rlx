// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified **data inspection** — one recording format for any tensor stream, so
//! op outputs, KV-cache tensors, and retention selection-preferences can all be
//! captured, exported, and analyzed *together*.
//!
//! Three layers, from concrete to aggregate:
//! - [`Histogram`] — a fixed-bin value distribution with an ASCII sparkline.
//! - [`TensorStats`] — shape + count + min/max/mean/std/absmax + nan/inf + a
//!   [`Histogram`], computed over a flat `f32` buffer.
//! - [`InspectLog`] — a time series of named [`TensorStats`] streams plus
//!   dataflow edges, with CSV / histogram-CSV / Graphviz / text-report export.
//!
//! It lives in `rlx-ir` (the lowest common crate) so the CPU executor's op tap
//! ([`op_tap_record`]), `rlx-runtime`'s KV/selection recording, and downstream
//! probes share exactly one schema. The op tap is a process-global, env-gated
//! ([`RLX_INSPECT_OPS`](op_tap_enabled)) sink so any model's forward pass can be
//! inspected without threading a recorder through every backend.
//!
//! Distinct from the sibling [`inspect`](crate::inspect) module, which dumps
//! IR *structure* (HIR/MIR/LIR text); this one records tensor *data*.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// A fixed-bin value histogram over `[lo, hi]`. Values outside the range clamp
/// into the edge bins (so nothing is silently dropped).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    /// Lower edge of the first bin.
    pub lo: f32,
    /// Upper edge of the last bin.
    pub hi: f32,
    /// Per-bin counts (uniform width `(hi - lo) / bins.len()`).
    pub bins: Vec<u64>,
}

impl Histogram {
    /// Empty histogram with `nbins` uniform bins over `[lo, hi]`.
    pub fn new(lo: f32, hi: f32, nbins: usize) -> Self {
        Histogram {
            lo,
            hi,
            bins: vec![0; nbins.max(1)],
        }
    }
    /// Histogram of `data` over its own `[min, max]` (finite values only).
    pub fn of(data: &[f32], nbins: usize) -> Self {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in data {
            if x.is_finite() {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        if hi <= lo {
            hi = lo + 1.0; // degenerate (constant tensor): keep one non-empty bin
        }
        let mut h = Histogram::new(lo, hi, nbins);
        for &x in data {
            h.add(x);
        }
        h
    }
    /// Bin one value (finite values only; NaN/Inf are skipped — counted in
    /// [`TensorStats`] instead).
    pub fn add(&mut self, x: f32) {
        if !x.is_finite() {
            return;
        }
        let n = self.bins.len();
        let frac = ((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0);
        let mut idx = (frac * n as f32) as usize;
        if idx >= n {
            idx = n - 1;
        }
        self.bins[idx] += 1;
    }
    /// Total binned count.
    pub fn total(&self) -> u64 {
        self.bins.iter().sum()
    }
    /// A unicode sparkline of the bin counts (log-scaled so tails survive).
    pub fn sparkline(&self) -> String {
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let maxln = self
            .bins
            .iter()
            .map(|&c| ((c as f64) + 1.0).ln())
            .fold(0.0f64, f64::max)
            .max(1e-9);
        self.bins
            .iter()
            .map(|&c| {
                let frac = ((c as f64) + 1.0).ln() / maxln;
                let bi =
                    ((frac * (blocks.len() - 1) as f64).round() as usize).min(blocks.len() - 1);
                blocks[bi]
            })
            .collect()
    }
    /// Compact CSV cell: `lo|hi|b0;b1;…`.
    pub fn to_cell(&self) -> String {
        let bins = self
            .bins
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(";");
        format!("{}|{}|{}", self.lo, self.hi, bins)
    }
}

/// Shape + summary statistics + a value [`Histogram`] for one tensor snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorStats {
    /// Stream / op / tensor name.
    pub name: String,
    /// Logical shape (row-major dims).
    pub shape: Vec<usize>,
    /// Number of scalar elements inspected.
    pub count: usize,
    /// Min over finite values (`0` if none).
    pub min: f32,
    /// Max over finite values (`0` if none).
    pub max: f32,
    /// Mean over finite values.
    pub mean: f32,
    /// Population standard deviation over finite values.
    pub std: f32,
    /// Max absolute value (saturation / precision headroom signal).
    pub absmax: f32,
    /// Count of NaN elements.
    pub nan: usize,
    /// Count of ±Inf elements.
    pub inf: usize,
    /// Value distribution.
    pub hist: Histogram,
}

impl TensorStats {
    /// Compute stats + a `nbins`-bin histogram over `data`.
    pub fn compute(name: impl Into<String>, shape: &[usize], data: &[f32], nbins: usize) -> Self {
        let (mut min, mut max, mut absmax) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
        let (mut sum, mut n_finite, mut nan, mut inf) = (0.0f64, 0usize, 0usize, 0usize);
        for &x in data {
            if x.is_nan() {
                nan += 1;
            } else if x.is_infinite() {
                inf += 1;
            } else {
                min = min.min(x);
                max = max.max(x);
                absmax = absmax.max(x.abs());
                sum += x as f64;
                n_finite += 1;
            }
        }
        let mean = if n_finite > 0 {
            (sum / n_finite as f64) as f32
        } else {
            0.0
        };
        let mut var = 0.0f64;
        if n_finite > 0 {
            for &x in data {
                if x.is_finite() {
                    let d = x as f64 - mean as f64;
                    var += d * d;
                }
            }
            var /= n_finite as f64;
        }
        if n_finite == 0 {
            min = 0.0;
            max = 0.0;
        }
        TensorStats {
            name: name.into(),
            shape: shape.to_vec(),
            count: data.len(),
            min,
            max,
            mean,
            std: (var.sqrt()) as f32,
            absmax,
            nan,
            inf,
            hist: Histogram::of(data, nbins),
        }
    }
    /// Shape as `dxd x…` (or `scalar`).
    pub fn shape_str(&self) -> String {
        if self.shape.is_empty() {
            "scalar".to_string()
        } else {
            self.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x")
        }
    }
    /// One-line human summary.
    pub fn line(&self) -> String {
        let flags = if self.nan + self.inf > 0 {
            format!("  !!{}nan/{}inf", self.nan, self.inf)
        } else {
            String::new()
        };
        format!(
            "{:<28} [{:>12}] min {:+.4} max {:+.4} mean {:+.4} std {:.4} |max| {:.4} {}{}",
            self.name,
            self.shape_str(),
            self.min,
            self.max,
            self.mean,
            self.std,
            self.absmax,
            self.hist.sparkline(),
            flags,
        )
    }
}

/// One recorded snapshot: which `step`, which `stream`, and the [`TensorStats`].
#[derive(Clone, Debug)]
pub struct InspectRecord {
    /// Step / turn / sequence index this snapshot belongs to.
    pub step: usize,
    /// Stream name (e.g. `"kv.k"`, `"selection.importance"`, `"op.Matmul#42"`).
    pub stream: String,
    /// The snapshot.
    pub stats: TensorStats,
}

/// A time series of named [`TensorStats`] streams plus dataflow edges. The one
/// container ops / KV / selection all record into, so a session is analyzable as
/// a whole (shared `step` axis, one CSV, one dataflow graph).
#[derive(Clone, Debug, Default)]
pub struct InspectLog {
    records: Vec<InspectRecord>,
    /// Directed dataflow edges between stream names (producer → consumer).
    edges: Vec<(String, String)>,
}

impl InspectLog {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }
    /// Record a precomputed [`TensorStats`] under `stream` at `step`.
    pub fn record(&mut self, step: usize, stream: impl Into<String>, stats: TensorStats) {
        self.records.push(InspectRecord {
            step,
            stream: stream.into(),
            stats,
        });
    }
    /// Convenience: compute stats over `data` and record them.
    pub fn record_tensor(
        &mut self,
        step: usize,
        stream: &str,
        shape: &[usize],
        data: &[f32],
        nbins: usize,
    ) {
        let stats = TensorStats::compute(stream, shape, data, nbins);
        self.record(step, stream, stats);
    }
    /// Add a dataflow edge `from → to` (deduped).
    pub fn edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let e = (from.into(), to.into());
        if !self.edges.contains(&e) {
            self.edges.push(e);
        }
    }
    /// Recorded snapshots, in order.
    pub fn records(&self) -> &[InspectRecord] {
        &self.records
    }
    /// Dataflow edges.
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }
    /// Number of snapshots.
    pub fn len(&self) -> usize {
        self.records.len()
    }
    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Long-format CSV: one row per snapshot (scalar stats). Histograms are
    /// exported separately by [`to_hist_csv`](Self::to_hist_csv).
    pub fn to_csv(&self) -> String {
        let mut s = String::from("step,stream,shape,count,min,max,mean,std,absmax,nan,inf\n");
        for r in &self.records {
            let t = &r.stats;
            s.push_str(&format!(
                "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
                r.step,
                r.stream,
                t.shape_str(),
                t.count,
                t.min,
                t.max,
                t.mean,
                t.std,
                t.absmax,
                t.nan,
                t.inf,
            ));
        }
        s
    }
    /// Histogram CSV: one row per snapshot, `bins` packed into a `lo|hi|b0;b1;…`
    /// cell so a plotter can render the distribution's evolution over `step`.
    pub fn to_hist_csv(&self) -> String {
        let mut s = String::from("step,stream,hist\n");
        for r in &self.records {
            s.push_str(&format!(
                "{},{},{}\n",
                r.step,
                r.stream,
                r.stats.hist.to_cell()
            ));
        }
        s
    }
    /// Graphviz DOT of the dataflow edges.
    pub fn dataflow_dot(&self) -> String {
        let mut s = String::from("digraph inspect {\n  rankdir=LR;\n");
        for (from, to) in &self.edges {
            s.push_str(&format!("  {:?} -> {:?};\n", from, to));
        }
        s.push_str("}\n");
        s
    }
    /// Text report: the latest snapshot of each stream (one `line()` each).
    pub fn report(&self) -> String {
        use std::collections::BTreeMap;
        let mut latest: BTreeMap<&str, &TensorStats> = BTreeMap::new();
        for r in &self.records {
            latest.insert(r.stream.as_str(), &r.stats);
        }
        let mut s = format!(
            "inspect: {} snapshots across {} streams\n",
            self.records.len(),
            latest.len()
        );
        for (_, t) in latest {
            s.push_str("  ");
            s.push_str(&t.line());
            s.push('\n');
        }
        s
    }
}

// ── Process-global op tap (env-gated) ────────────────────────────────────────
//
// The CPU executor calls `op_tap_record` after each computed node when
// `RLX_INSPECT_OPS` is set, so any model's forward pass yields op-level
// shape/stats/histogram/dataflow with no per-model wiring. A probe takes the log
// back with `op_tap_take`.

static OP_TAP: OnceLock<Mutex<InspectLog>> = OnceLock::new();
static OP_TAP_BINS: OnceLock<usize> = OnceLock::new();
static OP_TAP_PASS: AtomicUsize = AtomicUsize::new(0);

fn op_tap() -> &'static Mutex<InspectLog> {
    OP_TAP.get_or_init(|| Mutex::new(InspectLog::new()))
}

/// Advance to a new forward-pass index (used as the `step` axis) and return it.
/// A backend executor calls this once per forward when the tap is enabled, so
/// ops from different passes are distinguishable in the shared time series.
pub fn op_tap_begin_pass() -> usize {
    OP_TAP_PASS.fetch_add(1, Ordering::Relaxed)
}

/// Whether the op tap is enabled — `RLX_INSPECT_OPS` set to a non-empty, non-`0`
/// value. Read fresh each call (an executor calls it once per forward), so
/// enabling it mid-process — e.g. right before a dedicated inspection pass —
/// takes effect even if an earlier forward ran with it off.
pub fn op_tap_enabled() -> bool {
    matches!(crate::env::var("RLX_INSPECT_OPS").as_deref(), Some(v) if !v.is_empty() && v != "0")
}

/// Histogram bin count for the op tap (`RLX_INSPECT_BINS`, default 24).
pub fn op_tap_bins() -> usize {
    *OP_TAP_BINS.get_or_init(|| {
        crate::env::var("RLX_INSPECT_BINS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(24)
    })
}

/// Record one op output into the global tap: `step` (forward/decode index),
/// `op_kind`, `node_id`, `shape`, output `data`, and the tapped input node ids
/// (for dataflow edges). Cheap no-op when the tap is disabled.
pub fn op_tap_record(
    step: usize,
    op_kind: &str,
    node_id: usize,
    shape: &[usize],
    data: &[f32],
    input_ids: &[usize],
) {
    // The backend executor only reaches this when the tap is enabled (it guards
    // per-node calls on `op_tap_enabled()` once per forward), so no re-check here.
    let name = format!("{op_kind}#{node_id}");
    let stats = TensorStats::compute(name.clone(), shape, data, op_tap_bins());
    if let Ok(mut log) = op_tap().lock() {
        for &inp in input_ids {
            log.edge(format!("#{inp}"), format!("#{node_id}"));
        }
        log.record(step, name, stats);
    }
}

/// Take the accumulated op-tap log, leaving the global sink empty.
pub fn op_tap_take() -> InspectLog {
    if let Ok(mut log) = op_tap().lock() {
        std::mem::take(&mut *log)
    } else {
        InspectLog::new()
    }
}

/// Number of op snapshots currently held by the global tap.
pub fn op_tap_len() -> usize {
    op_tap().lock().map(|l| l.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bins_and_clamps() {
        let h = Histogram::of(&[0.0, 0.5, 1.0, 1.0, 2.0], 4);
        assert_eq!(h.total(), 5);
        // range [0,2], 4 bins of width 0.5: {0}->b0, {0.5}->b1, {1,1}->b2, {2}->b3
        assert_eq!(h.bins, vec![1, 1, 2, 1]);
    }

    #[test]
    fn tensor_stats_flags_nan_inf_and_shape() {
        let data = [1.0, 2.0, 3.0, f32::NAN, f32::INFINITY];
        let t = TensorStats::compute("x", &[5], &data, 8);
        assert_eq!(t.count, 5);
        assert_eq!(t.nan, 1);
        assert_eq!(t.inf, 1);
        assert_eq!(t.min, 1.0);
        assert_eq!(t.max, 3.0);
        assert!((t.mean - 2.0).abs() < 1e-6);
        assert_eq!(t.shape_str(), "5");
    }

    #[test]
    fn inspect_log_csv_and_dataflow() {
        let mut log = InspectLog::new();
        log.record_tensor(0, "kv.k", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4);
        log.record_tensor(1, "kv.k", &[2, 3], &[0.0; 6], 4);
        log.edge("kv.k", "attn");
        assert_eq!(log.len(), 2);
        let csv = log.to_csv();
        assert!(csv.lines().next().unwrap().starts_with("step,stream,shape"));
        assert_eq!(csv.lines().count(), 3); // header + 2 rows
        assert!(log.to_hist_csv().contains("kv.k"));
        assert!(log.dataflow_dot().contains("\"kv.k\" -> \"attn\""));
    }

    #[test]
    fn op_tap_records_edges_and_drains() {
        let _ = op_tap_take(); // clear any prior global state
        let pass = op_tap_begin_pass();
        op_tap_record(pass, "Matmul", 7, &[2, 2], &[1.0, 2.0, 3.0, 4.0], &[3, 5]);
        let log = op_tap_take();
        assert!(log.records().iter().any(|r| r.stream == "Matmul#7"));
        assert!(log.edges().contains(&("#3".to_string(), "#7".to_string())));
        assert!(log.edges().contains(&("#5".to_string(), "#7".to_string())));
        // Draining leaves the global sink empty.
        assert_eq!(op_tap_len(), 0);
    }
}
