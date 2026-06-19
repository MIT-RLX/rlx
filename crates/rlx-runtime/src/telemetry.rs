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

//! Telemetry primitives (plan #65).
//!
//! Borrowed from MAX's `serve/telemetry/` module shape (counters,
//! histograms, stopwatches) but stripped to a pure data layer.
//! Today this is a single in-process `MetricsRegistry`; a future
//! serving crate can route the registry to a sidecar process /
//! Prometheus / OTel exporter without changing the call sites.
//!
//! Why now (without a serving crate to consume it)?
//!   - Lets the autotuner / fusion passes record decisions while
//!     they're being made, viewable later via [`MetricsRegistry::
//!     snapshot`].
//!   - Establishes the "metrics live in their own type, not
//!     scattered through hot paths" pattern before we accumulate
//!     a hundred ad-hoc counters.

use std::collections::BTreeMap;
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

/// Monotonic 64-bit counter. Cheap (atomic add); safe to call from
/// any thread.
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
    // #[inline(always)] (plan #76) — these are the per-event hot
    // paths; we want callers to see one atomic instruction, not a
    // function-call indirection through the Counter type.
    #[inline(always)]
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn add(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Fixed-bucket exponential histogram — 16 buckets covering up to
/// `2^16 ≈ 65k` of the chosen unit. Right for ns-to-ms latency
/// distributions or 1-100k sample-count distributions.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; 16],
    sum: AtomicU64,
    count: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub const fn new() -> Self {
        // `[Z; N]` array-init produces N independent AtomicU64s, not
        // shared interior-mutable const aliases — same idiom std uses
        // in `[AtomicUsize::new(0); N]`.
        #[allow(clippy::declare_interior_mutable_const)]
        const Z: AtomicU64 = AtomicU64::new(0);
        Self {
            buckets: [Z; 16],
            sum: Z,
            count: Z,
        }
    }

    pub fn record(&self, value: u64) {
        let bucket = (value.checked_ilog2().unwrap_or(0) as usize).min(15);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
    pub fn sum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }
    pub fn mean(&self) -> Option<f64> {
        let c = self.count();
        if c == 0 {
            None
        } else {
            Some(self.sum() as f64 / c as f64)
        }
    }
    pub fn bucket_counts(&self) -> [u64; 16] {
        let mut out = [0u64; 16];
        for (i, b) in self.buckets.iter().enumerate() {
            out[i] = b.load(Ordering::Relaxed);
        }
        out
    }
}

/// Global registry of named counters and histograms. Indexed by
/// static string keys; lookups are O(log N) on a BTreeMap. Lock
/// is only held during register/lookup, not during increment.
pub struct MetricsRegistry {
    counters: Mutex<BTreeMap<&'static str, &'static Counter>>,
    histograms: Mutex<BTreeMap<&'static str, &'static Histogram>>,
}

impl MetricsRegistry {
    /// Process-wide registry. First access lazily initializes.
    pub fn global() -> &'static Self {
        static R: OnceLock<MetricsRegistry> = OnceLock::new();
        R.get_or_init(|| MetricsRegistry {
            counters: Mutex::new(BTreeMap::new()),
            histograms: Mutex::new(BTreeMap::new()),
        })
    }

    /// Register a `'static` counter so it shows up in
    /// `snapshot()`. Idempotent — re-registering with the same
    /// pointer is a no-op.
    pub fn register_counter(&self, name: &'static str, c: &'static Counter) {
        self.counters
            .lock()
            .expect("registry poisoned")
            .insert(name, c);
    }

    pub fn register_histogram(&self, name: &'static str, h: &'static Histogram) {
        self.histograms
            .lock()
            .expect("registry poisoned")
            .insert(name, h);
    }

    /// Snapshot all metrics into a serializable map. Useful for
    /// dumping to a log / exporting to Prometheus.
    pub fn snapshot(&self) -> Snapshot {
        let counters = self
            .counters
            .lock()
            .unwrap()
            .iter()
            .map(|(&n, c)| (n.to_string(), c.get()))
            .collect();
        let histograms = self
            .histograms
            .lock()
            .unwrap()
            .iter()
            .map(|(&n, h)| {
                (
                    n.to_string(),
                    HistogramSnapshot {
                        count: h.count(),
                        sum: h.sum(),
                        buckets: h.bucket_counts(),
                    },
                )
            })
            .collect();
        Snapshot {
            counters,
            histograms,
        }
    }
}

#[derive(Debug)]
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum: u64,
    pub buckets: [u64; 16],
}

#[derive(Debug, Default)]
pub struct Snapshot {
    pub counters: BTreeMap<String, u64>,
    pub histograms: BTreeMap<String, HistogramSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_basic() {
        let c = Counter::new();
        c.inc();
        c.inc();
        c.add(10);
        assert_eq!(c.get(), 12);
    }

    #[test]
    fn histogram_records_in_bucket() {
        let h = Histogram::new();
        h.record(0); // bucket 0 (ilog2(0).unwrap_or(0) = 0)
        h.record(1); // bucket 0
        h.record(7); // bucket 2 (ilog2(7) = 2)
        h.record(1024); // bucket 10
        let b = h.bucket_counts();
        assert_eq!(h.count(), 4);
        assert_eq!(b[0] + b[2] + b[10], 4);
    }

    #[test]
    fn registry_round_trip() {
        static C: Counter = Counter::new();
        static H: Histogram = Histogram::new();
        let r = MetricsRegistry::global();
        r.register_counter("test_count", &C);
        r.register_histogram("test_hist", &H);
        C.inc();
        H.record(42);
        let snap = r.snapshot();
        assert!(snap.counters.contains_key("test_count"));
        assert!(snap.histograms.contains_key("test_hist"));
    }
}
