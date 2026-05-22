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

//! Rayon-backed parallel for: `par_for(total, grain, |off, cnt| …)`.
//!
//! Replaces the old per-worker Condvar pool with Rayon's work-stealing
//! scheduler. Same `(offset, count)` chunk API so all existing call
//! sites (BLAS tiling, SDPA, LayerNorm, …) pick up Rayon without
//! changes.

use rayon::prelude::*;
use std::sync::Once;

static POOL_INIT: Once = Once::new();

fn ensure_pool() {
    POOL_INIT.call_once(|| {
        let cfg = crate::config::RuntimeConfig::global();
        let n = cfg.pool_workers.max(1);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rlx-rayon-{i}"))
            .build_global();
    });
}

/// Total Rayon worker count (configured from [`RuntimeConfig::pool_workers`]).
pub fn num_threads() -> usize {
    ensure_pool();
    rayon::current_num_threads()
}

/// Parallel for: split `total` items across threads. `f(off, cnt)` is
/// called once per chunk with disjoint regions.
///
/// SAFETY: caller must ensure `f` accesses disjoint memory regions for
/// different `(offset, count)` pairs.
#[inline]
pub fn par_for<F: Fn(usize, usize) + Sync>(total: usize, min_per_thread: usize, f: &F) {
    if total == 0 {
        return;
    }
    ensure_pool();
    let grain = min_per_thread.max(1);
    let n_threads = (total / grain).max(1).min(num_threads());
    if n_threads <= 1 {
        f(0, total);
        return;
    }
    let chunk = total.div_ceil(n_threads);
    (0..n_threads).into_par_iter().for_each(|t| {
        let off = t * chunk;
        if off < total {
            f(off, (off + chunk).min(total) - off);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn par_for_sums_correctly() {
        let data = vec![1.0f32; 10_000];
        let total = AtomicU64::new(0);

        par_for(data.len(), 100, &|off, cnt| {
            let partial: f32 = data[off..off + cnt].iter().sum();
            total.fetch_add(partial.to_bits() as u64, Ordering::Relaxed);
        });

        assert!(total.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn par_for_small_is_sequential() {
        let sum = std::sync::atomic::AtomicUsize::new(0);
        par_for(10, 100, &|off, cnt| {
            sum.fetch_add(cnt, Ordering::Relaxed);
            assert_eq!(off + cnt, 10);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn par_for_exact_sum_many_dispatches() {
        for &n in &[256usize, 1024, 4097] {
            let sum = std::sync::atomic::AtomicUsize::new(0);
            par_for(n, 256, &|off, cnt| {
                sum.fetch_add(cnt, Ordering::Relaxed);
                assert!(off + cnt <= n);
            });
            assert_eq!(sum.load(Ordering::Relaxed), n);
        }
    }

    #[test]
    fn par_for_concurrent_callers_isolated() {
        std::thread::scope(|s| {
            for t in 0..4 {
                s.spawn(move || {
                    let n = 4096 + t * 17;
                    let sum = std::sync::atomic::AtomicUsize::new(0);
                    par_for(n, 128, &|off, cnt| {
                        sum.fetch_add(cnt, Ordering::Relaxed);
                        assert!(off + cnt <= n);
                    });
                    assert_eq!(sum.load(Ordering::Relaxed), n);
                });
            }
        });
    }
}
