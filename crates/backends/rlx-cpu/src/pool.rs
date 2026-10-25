// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rlx-rayon-{i}"))
            .build_global();
        // Someone used rayon first, so its default pool stands. Harmless
        // unless the user asked for a size: then say so instead of ignoring it.
        if built.is_err()
            && rlx_ir::env::var("RLX_WORKERS").is_some()
            && rayon::current_num_threads() != n
        {
            eprintln!(
                "rlx-cpu: RLX_WORKERS={n} not applied — the global rayon pool already exists with {} threads \
                 (build it before any rayon use to size it)",
                rayon::current_num_threads()
            );
        }
        // Pin OpenBLAS/MKL/Accelerate to 1 thread — Rayon owns outer
        // parallelism (par_sgemm splits). Nested N² threads are catastrophic.
        crate::blas::limit_inner_threads();
    });
}

/// Total Rayon worker count (configured from `RuntimeConfig::pool_workers`).
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
///
/// Measured (14-core Apple M-series, training on the synthetic suite), with
/// identical scores at every value: 4096 → 65536 took TensorCSPNet 20.4 → 10.0 s
/// (system time 11.1 → 0.24 s), EEGNet 8.6 → 4.5 s, ShallowFBCSPNet 5.5 → 4.1 s.
/// At 4096 an 8k-element add split 14 ways spent its time waking workers.
pub const ELEMENTWISE_PAR_FLOOR: usize = 65536;

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

/// Minimum work per parallel task, in multiply-adds (or comparable element
/// operations). Below it a rayon hand-off — waking a parked worker, the join —
/// costs more than the task: a batch of 22×22 GEMMs (~10k MACs each) split one
/// item per task spent as much time in the kernel (sleep/wake) as computing.
pub const TASK_WORK_FLOOR: usize = 1 << 17;

/// `par_for` grain for `units` independent items of `work_per_unit` each: enough
/// items per task to reach [`TASK_WORK_FLOOR`]. `par_for` then runs inline when
/// the whole range is below one task's worth.
#[inline]
pub fn grain_for(work_per_unit: usize) -> usize {
    TASK_WORK_FLOOR.div_ceil(work_per_unit.max(1)).max(1)
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
        // Pin BLAS to 1 before Rayon fan-out so a preceding huge MT GEMM
        // cannot oversubscribe with worker-local sgemm / elementwise work.
        crate::blas::ensure_blas_threads(1);
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
    fn grain_reaches_the_work_floor() {
        assert_eq!(grain_for(TASK_WORK_FLOOR), 1);
        assert_eq!(grain_for(0), TASK_WORK_FLOOR);
        let g = grain_for(22 * 22 * 22);
        assert!(g * 22 * 22 * 22 >= TASK_WORK_FLOOR && (g - 1) * 22 * 22 * 22 < TASK_WORK_FLOOR);
    }

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
