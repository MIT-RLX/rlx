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

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Once;

#[cfg(not(target_arch = "wasm32"))]
static POOL_INIT: Once = Once::new();

#[cfg(not(target_arch = "wasm32"))]
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
///
/// On wasm there is no thread pool — the browser is single-threaded — so
/// this is always 1.
#[cfg(not(target_arch = "wasm32"))]
pub fn num_threads() -> usize {
    ensure_pool();
    rayon::current_num_threads()
}

#[cfg(target_arch = "wasm32")]
pub fn num_threads() -> usize {
    1
}

/// Parallel for: split `total` items across threads. `f(off, cnt)` is
/// called once per chunk with disjoint regions.
///
/// SAFETY: caller must ensure `f` accesses disjoint memory regions for
/// different `(offset, count)` pairs.
/// Parallel `for i in 0..n` over an index range, one task per index.
/// Serial on wasm (no thread pool). The closure must be `Sync + Send`
/// because tasks may run on Rayon workers natively.
#[inline]
pub fn par_range<F: Fn(usize) + Sync + Send>(n: usize, f: F) {
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).for_each(f);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..n).into_par_iter().for_each(f);
    }
}

/// Minimum elements that make threading an element-wise kernel worthwhile.
/// Below this the rayon hand-off costs more than the work it saves. Kept in
/// one place so conv/transpose/pool/relu/region kernels share a single,
/// tunable cutover instead of sprinkling magic constants at each call site.
pub const ELEMENTWISE_PAR_FLOOR: usize = 4096;

/// Should an op over `work` elements run in parallel? True only with >1 worker
/// and enough work to keep them busy past the amortization floor. Use this at
/// every data-parallel kernel call site instead of a hand-picked threshold.
#[inline]
pub fn should_parallelize(work: usize) -> bool {
    num_threads() > 1 && work >= ELEMENTWISE_PAR_FLOOR * 2
}

/// `par_for` chunk floor for a flat range of `total` elements: aim for one
/// chunk per worker, never below the amortization floor.
#[inline]
pub fn chunk_floor(total: usize) -> usize {
    (total / num_threads().max(1)).max(ELEMENTWISE_PAR_FLOOR)
}

/// `par_for` chunk floor for an *outer* loop of `units` independent items
/// (e.g. (n,c) planes): one chunk per worker, at least one unit.
#[inline]
pub fn outer_chunk(units: usize) -> usize {
    (units / num_threads().max(1)).max(1)
}

#[inline]
pub fn par_for<F: Fn(usize, usize) + Sync>(total: usize, min_per_thread: usize, f: &F) {
    if total == 0 {
        return;
    }
    // wasm is single-threaded: run the whole range inline. (Rayon's
    // work-stealing scheduler needs OS threads, which the browser lacks.)
    #[cfg(target_arch = "wasm32")]
    {
        let _ = min_per_thread;
        f(0, total);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
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
