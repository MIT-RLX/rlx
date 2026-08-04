// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable **KV cache / context telemetry** — a model-agnostic time-series
//! recorder + analysis for how a decode loop's working set and reachable context
//! evolve over a session.
//!
//! It is deliberately decoupled from any model: it records plain per-step counts
//! (resident / evicted / retrieved / store), so it works for any generator that
//! decodes through a KV cache. [`KvRetentionManager`](crate::kv_retention::KvRetentionManager)
//! wires it in automatically (`enable_recording`), but a caller can also drive a
//! [`RetentionRecorder`] by hand to profile a plain (retention-free) cache.
//!
//! The headline it exposes: **context extension** — how far past the O(budget)
//! resident working set the system actually reached, thanks to retrieval — plus
//! the eviction/retrieval activity and (optionally) per-step decode latency, so
//! you can see whether throughput stays flat as context grows.

/// One decode step's cache/context snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepRecord {
    /// Decode step index (0-based within the recorded session).
    pub step: usize,
    /// Resident (attended) positions this step — the O(budget) working set.
    pub resident: usize,
    /// Positions evicted from the resident set this step.
    pub evicted: usize,
    /// Positions retrieved back from the store into the resident set this step.
    pub retrieved: usize,
    /// Cold blocks currently held in the store (retrievable, not resident).
    pub store_blocks: usize,
    /// Tokens the store currently holds.
    pub store_tokens: usize,
    /// Per-step decode latency in ms, if the caller measured it.
    pub decode_ms: Option<f32>,
}

impl StepRecord {
    /// Total context the system can reach this step: resident + stored. This is
    /// the "effective context" that grows unbounded under retrieval even while
    /// `resident` stays bounded.
    pub fn effective_context(&self) -> usize {
        self.resident + self.store_tokens
    }
}

/// Accumulates [`StepRecord`]s over a decode session and rolls them up into a
/// [`RetentionSummary`] / CSV. Cheap to clone; `Default` is an empty recorder.
#[derive(Clone, Debug, Default)]
pub struct RetentionRecorder {
    steps: Vec<StepRecord>,
}

impl RetentionRecorder {
    /// A fresh, empty recorder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append one step's snapshot.
    pub fn push(&mut self, r: StepRecord) {
        self.steps.push(r);
    }
    /// Attach a measured decode latency (ms) to the most-recently pushed record.
    /// Call right after the step's decode finishes — the recorder itself has no
    /// clock, keeping it usable in headless / deterministic contexts.
    pub fn set_last_decode_ms(&mut self, ms: f32) {
        if let Some(last) = self.steps.last_mut() {
            last.decode_ms = Some(ms);
        }
    }
    /// Recorded steps, in order.
    pub fn records(&self) -> &[StepRecord] {
        &self.steps
    }
    /// Number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }
    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
    /// Drop all recorded steps (reuse the allocation for a new session).
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    /// Roll the session up into summary statistics.
    pub fn summary(&self) -> RetentionSummary {
        let n = self.steps.len();
        if n == 0 {
            return RetentionSummary::default();
        }
        let mut resident: Vec<usize> = self.steps.iter().map(|s| s.resident).collect();
        resident.sort_unstable();
        let resident_max = *resident.last().unwrap();
        let resident_mean =
            self.steps.iter().map(|s| s.resident as f64).sum::<f64>() as f32 / n as f32;
        let store_tokens_max = self.steps.iter().map(|s| s.store_tokens).max().unwrap_or(0);
        let effective_context_max = self
            .steps
            .iter()
            .map(|s| s.effective_context())
            .max()
            .unwrap_or(0);
        let context_extension = effective_context_max as f32 / resident_max.max(1) as f32;
        let total_evicted = self.steps.iter().map(|s| s.evicted).sum();
        let total_retrieved = self.steps.iter().map(|s| s.retrieved).sum();
        let retrieval_steps = self.steps.iter().filter(|s| s.retrieved > 0).count();
        let retrieval_rate = retrieval_steps as f32 / n as f32;

        // Decode latency stats over whatever steps carry a measurement.
        let mut lat: Vec<f32> = self.steps.iter().filter_map(|s| s.decode_ms).collect();
        let (decode_ms_mean, decode_ms_p95) = if lat.is_empty() {
            (None, None)
        } else {
            let mean = lat.iter().copied().sum::<f32>() / lat.len() as f32;
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (Some(mean), Some(percentile_f32(&lat, 95.0)))
        };

        RetentionSummary {
            steps: n,
            resident_mean,
            resident_p50: percentile_usize(&resident, 50.0),
            resident_p95: percentile_usize(&resident, 95.0),
            resident_max,
            store_tokens_max,
            effective_context_max,
            context_extension,
            total_evicted,
            total_retrieved,
            retrieval_rate,
            decode_ms_mean,
            decode_ms_p95,
        }
    }

    /// The full time series as CSV (one row per step, header included). Suitable
    /// for piping into a plotter — the columns tell the story of resident vs.
    /// effective context over time.
    pub fn to_csv(&self) -> String {
        let mut s = String::from(
            "step,resident,evicted,retrieved,store_blocks,store_tokens,effective_context,decode_ms\n",
        );
        for r in &self.steps {
            let ms = match r.decode_ms {
                Some(v) => format!("{v:.3}"),
                None => String::new(),
            };
            s.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                r.step,
                r.resident,
                r.evicted,
                r.retrieved,
                r.store_blocks,
                r.store_tokens,
                r.effective_context(),
                ms,
            ));
        }
        s
    }
}

/// Rolled-up analysis over a recorded session.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetentionSummary {
    /// Number of decode steps recorded.
    pub steps: usize,
    /// Mean resident (working-set) size.
    pub resident_mean: f32,
    /// Median resident size.
    pub resident_p50: usize,
    /// 95th-percentile resident size.
    pub resident_p95: usize,
    /// Peak resident size (should track the policy's budget).
    pub resident_max: usize,
    /// Peak tokens held in the store.
    pub store_tokens_max: usize,
    /// Peak reachable context (resident + store).
    pub effective_context_max: usize,
    /// `effective_context_max / resident_max` — how far past the working set the
    /// system reached. `1.0` = no extension; `> 1` = retrieval extending context.
    pub context_extension: f32,
    /// Total positions evicted over the session.
    pub total_evicted: usize,
    /// Total positions retrieved over the session.
    pub total_retrieved: usize,
    /// Fraction of steps that retrieved ≥ 1 block.
    pub retrieval_rate: f32,
    /// Mean per-step decode latency (ms), if any step carried a measurement.
    pub decode_ms_mean: Option<f32>,
    /// 95th-percentile per-step decode latency (ms), if measured.
    pub decode_ms_p95: Option<f32>,
}

impl RetentionSummary {
    /// A compact, human-readable multi-line report.
    pub fn report(&self) -> String {
        let mut s = format!("KV cache/context over {} decode steps\n", self.steps);
        s.push_str(&format!(
            "  resident (working set): mean {:.1}  p50 {}  p95 {}  max {}\n",
            self.resident_mean, self.resident_p50, self.resident_p95, self.resident_max,
        ));
        s.push_str(&format!(
            "  store: up to {} tokens held offline\n",
            self.store_tokens_max,
        ));
        s.push_str(&format!(
            "  effective context: {} tokens peak  ->  {:.2}x extension over resident\n",
            self.effective_context_max, self.context_extension,
        ));
        s.push_str(&format!(
            "  activity: {} evicted total, {} retrieved total, retrieval on {:.0}% of steps\n",
            self.total_evicted,
            self.total_retrieved,
            self.retrieval_rate * 100.0,
        ));
        if let (Some(mean), Some(p95)) = (self.decode_ms_mean, self.decode_ms_p95) {
            let tps = if mean > 0.0 { 1000.0 / mean } else { 0.0 };
            s.push_str(&format!(
                "  decode: mean {mean:.2} ms/step  p95 {p95:.2} ms  ({tps:.0} tok/s)\n",
            ));
        }
        s
    }
}

/// Nearest-rank percentile of a **sorted** `usize` slice (`p` in `[0, 100]`).
fn percentile_usize(sorted: &[usize], p: f32) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Nearest-rank percentile of a **sorted** `f32` slice (`p` in `[0, 100]`).
fn percentile_f32(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_recorder_yields_zero_summary() {
        let rec = RetentionRecorder::new();
        assert!(rec.is_empty());
        let s = rec.summary();
        assert_eq!(s.steps, 0);
        assert_eq!(s.context_extension, 0.0);
    }

    #[test]
    fn summary_tracks_extension_and_activity() {
        let mut rec = RetentionRecorder::new();
        // Resident bounded at 10; store grows to 40 as context is offloaded.
        for step in 0..5 {
            rec.push(StepRecord {
                step,
                resident: 10,
                evicted: 2,
                retrieved: if step >= 2 { 3 } else { 0 },
                store_blocks: step * 2,
                store_tokens: step * 8,
                decode_ms: Some(12.0),
            });
        }
        let s = rec.summary();
        assert_eq!(s.steps, 5);
        assert_eq!(s.resident_max, 10);
        assert_eq!(s.resident_p50, 10);
        // Peak effective context = 10 resident + 32 stored (step 4) = 42.
        assert_eq!(s.effective_context_max, 42);
        assert!((s.context_extension - 4.2).abs() < 1e-4);
        assert_eq!(s.total_evicted, 10);
        assert_eq!(s.total_retrieved, 9); // steps 2,3,4 * 3
        assert!((s.retrieval_rate - 0.6).abs() < 1e-6); // 3 of 5 steps
        assert_eq!(s.decode_ms_mean, Some(12.0));
    }

    #[test]
    fn csv_has_header_and_one_row_per_step() {
        let mut rec = RetentionRecorder::new();
        rec.push(StepRecord {
            step: 0,
            resident: 8,
            evicted: 0,
            retrieved: 0,
            store_blocks: 0,
            store_tokens: 0,
            decode_ms: None,
        });
        rec.set_last_decode_ms(9.5);
        let csv = rec.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 row
        assert!(lines[0].starts_with("step,resident,"));
        assert!(lines[1].starts_with("0,8,0,0,0,0,8,9.500"));
    }
}
