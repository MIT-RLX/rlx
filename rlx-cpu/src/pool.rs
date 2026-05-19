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

//! Persistent thread pool: per-worker `Mutex<Option<Task>> + Condvar`
//! slots, pool-wide `POOL_LOCK` to serialize concurrent batches,
//! pool-wide `IN_FLIGHT` counter + `DONE_CV` for batch completion.
//!
//! Two design points worth flagging:
//!
//! 1. **Workers park on idle.** Each worker thread blocks on its
//!    own slot's `Condvar` until a task lands. Idle workers consume
//!    zero CPU — the previous busy-spin design contended for cores
//!    with the actual compute and blew up dispatch latency by
//!    100×+ in containerized environments (CI, Docker on Apple
//!    Silicon, anywhere with shared cores).
//!
//! 2. **Concurrent `par_for` callers serialize on `POOL_LOCK`.**
//!    The pool slots and the `IN_FLIGHT` counter are shared static
//!    state; without serialization, two threads dispatching at
//!    once can overwrite each other's tasks (slot has only one
//!    `Option<Task>`) or race on the counter (causing missed
//!    notifies). The lock is held for the duration of one batch
//!    (dispatch + wait_all). Workers run in parallel **within** a
//!    batch — that's where the parallelism is meaningful.
//!
//!    The right next step if concurrent batches are ever a real
//!    throughput need is per-worker queues with batch ids. We
//!    tried `mpsc::channel` and `Mutex<VecDeque>`; both regressed
//!    single-batch throughput by 2-100× because of the per-dispatch
//!    overhead they add (channel send mutex + allocation, queue
//!    push under contention with worker pop). The static-slot
//!    design is faster for the common single-caller case.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

const MAX_WORKERS: usize = 15;

type Task = Box<dyn FnOnce() + Send + 'static>;

struct WorkerSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

struct SlotState {
    task: Option<Task>,
    shutdown: bool,
}

impl WorkerSlot {
    const fn new() -> Self {
        Self {
            state: Mutex::new(SlotState {
                task: None,
                shutdown: false,
            }),
            cv: Condvar::new(),
        }
    }
}

static WORKERS: [WorkerSlot; MAX_WORKERS] = {
    // Each `[SLOT; MAX_WORKERS]` element is a freshly-constructed
    // WorkerSlot (with its own Mutex / Condvar) — that's the
    // intended array-init semantics, not a shared interior-mutable
    // const. Same idiom std uses in `[AtomicUsize::new(0); N]`.
    #[allow(clippy::declare_interior_mutable_const)]
    const SLOT: WorkerSlot = WorkerSlot::new();
    [SLOT; MAX_WORKERS]
};

static NUM_WORKERS: AtomicUsize = AtomicUsize::new(0);
static POOL_INIT: std::sync::Once = std::sync::Once::new();

/// Serializes concurrent `par_for` callers — see module doc.
static POOL_LOCK: Mutex<()> = Mutex::new(());

/// Outstanding-task counter for the active batch. Main `fetch_add`s
/// before dispatching; workers `fetch_sub` after running. When the
/// count drops to zero, the worker that completed last notifies
/// `DONE_CV`. With `POOL_LOCK` held by main for the duration of one
/// batch, only one batch's tasks are in flight at a time, so the
/// counter is effectively per-batch.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static DONE_LOCK: Mutex<()> = Mutex::new(());
static DONE_CV: Condvar = Condvar::new();

fn ensure_pool() {
    POOL_INIT.call_once(|| {
        let cfg = crate::config::RuntimeConfig::global();
        let n = cfg.pool_workers.min(MAX_WORKERS);
        NUM_WORKERS.store(n, Ordering::Relaxed);

        for i in 0..n {
            std::thread::Builder::new()
                .name(format!("rlx-w{i}"))
                .spawn(move || worker_loop(i))
                .expect("spawn rlx worker");
        }
    });
}

fn worker_loop(slot: usize) {
    let w = &WORKERS[slot];
    loop {
        let task: Task = {
            let mut s = w.state.lock().expect("pool slot poisoned");
            while s.task.is_none() && !s.shutdown {
                s = w.cv.wait(s).expect("pool slot poisoned");
            }
            if s.shutdown {
                return;
            }
            s.task.take().unwrap()
        };
        task();
        if IN_FLIGHT.fetch_sub(1, Ordering::AcqRel) == 1 {
            // notify_all (not notify_one) covers the case where the
            // pool's locking layer changes in the future to allow
            // multiple concurrent waiters; today only one main is
            // ever waiting (POOL_LOCK serializes), so the difference
            // is academic.
            let _g = DONE_LOCK.lock().expect("pool done lock poisoned");
            DONE_CV.notify_all();
        }
    }
}

/// Hand a closure to worker `slot`. Caller must already have
/// `IN_FLIGHT.fetch_add(1)`'d.
fn dispatch_to(slot: usize, f: impl FnOnce() + Send + 'static) {
    let boxed: Task = Box::new(f);
    let w = &WORKERS[slot];
    let mut s = w.state.lock().expect("pool slot poisoned");
    debug_assert!(s.task.is_none(), "worker slot {slot} already has a task");
    s.task = Some(boxed);
    drop(s);
    w.cv.notify_one();
}

fn wait_all() {
    if IN_FLIGHT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut g = DONE_LOCK.lock().expect("pool done lock poisoned");
    while IN_FLIGHT.load(Ordering::Acquire) != 0 {
        g = DONE_CV.wait(g).expect("pool done lock poisoned");
    }
}

/// Total thread count (workers + main).
pub fn num_threads() -> usize {
    ensure_pool();
    NUM_WORKERS.load(Ordering::Relaxed) + 1
}

/// Parallel for: split `total` items across all threads. `f(off, cnt)`
/// is called once per thread with disjoint regions.
///
/// SAFETY: caller must ensure `f` accesses disjoint memory regions
/// for different (offset, count) pairs.
#[inline]
pub fn par_for<F: Fn(usize, usize) + Sync>(total: usize, min_per_thread: usize, f: &F) {
    ensure_pool();
    let nw = NUM_WORKERS.load(Ordering::Relaxed);
    let max_threads = (total / min_per_thread.max(1)).max(1).min(nw + 1);
    if max_threads <= 1 {
        f(0, total);
        return;
    }
    // Serialize concurrent `par_for` callers — see module doc.
    let _guard = POOL_LOCK.lock().expect("pool lock poisoned");
    let workers = max_threads - 1;
    let chunk = total / max_threads;

    IN_FLIGHT.fetch_add(workers, Ordering::AcqRel);
    for i in 0..workers {
        let off = i * chunk;
        // SAFETY: `f` outlives `wait_all` below — main only returns
        // from `par_for` after every dispatched task has completed.
        let f_ptr = f as *const F as usize;
        dispatch_to(i, move || {
            let f_ref = unsafe { &*(f_ptr as *const F) };
            f_ref(off, chunk);
        });
    }
    f(workers * chunk, total - workers * chunk);
    wait_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn par_for_sums_correctly() {
        let data = vec![1.0f32; 10000];
        let total = AtomicU64::new(0);

        par_for(data.len(), 100, &|off, cnt| {
            let partial: f32 = data[off..off + cnt].iter().sum();
            total.fetch_add(partial.to_bits() as u64, Ordering::Relaxed);
        });

        assert!(total.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn par_for_small_is_sequential() {
        let result = std::sync::atomic::AtomicU32::new(0);
        par_for(10, 100, &|off, cnt| {
            for i in off..off + cnt {
                result.fetch_add(i as u32, std::sync::atomic::Ordering::Relaxed);
            }
        });
        assert_eq!(result.load(std::sync::atomic::Ordering::Relaxed), 45);
    }

    #[test]
    fn par_for_exact_sum_many_dispatches() {
        let n = 10_000usize;
        for _ in 0..32 {
            let acc = AtomicUsize::new(0);
            par_for(n, 256, &|off, cnt| {
                let mut s: usize = 0;
                for i in off..off + cnt {
                    s += i;
                }
                acc.fetch_add(s, Ordering::Relaxed);
            });
            let expect = (n - 1) * n / 2;
            assert_eq!(acc.load(Ordering::Relaxed), expect);
        }
    }

    /// Concurrent callers: each must see only its own answer
    /// (no cross-batch counter mixing). With `POOL_LOCK` they
    /// serialize internally — the test verifies correctness, not
    /// concurrent throughput.
    #[test]
    fn par_for_concurrent_callers_isolated() {
        let n = 5_000usize;
        let n_callers = 8;
        let handles: Vec<_> = (0..n_callers)
            .map(|caller_id| {
                std::thread::spawn(move || {
                    let acc = AtomicUsize::new(0);
                    par_for(n, 128, &|off, cnt| {
                        let mut s = 0;
                        for i in off..off + cnt {
                            s += i + caller_id;
                        }
                        acc.fetch_add(s, Ordering::Relaxed);
                    });
                    let expect = (n - 1) * n / 2 + caller_id * n;
                    assert_eq!(
                        acc.load(Ordering::Relaxed),
                        expect,
                        "caller {caller_id} got wrong sum"
                    );
                })
            })
            .collect();
        for h in handles {
            h.join().expect("caller thread panicked");
        }
    }
}
