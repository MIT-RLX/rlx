// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A shared pool that bounds how many jobs (fit / predict / compile work) run
//! at once, by count and by memory.
//!
//! First come, first served: jobs start in the order they asked, so the work
//! handed out first (heavy jobs included) is not overtaken at random and left
//! to run alone at the end. With `gate_memory`, the next job also waits while
//! the process RSS plus the memory each running job has added so far would
//! exceed the soft budget ([`crate::soft_memory_budget_bytes`]: 80 % of RAM,
//! `RLX_SOFT_MEMORY_FRACTION` / `RLX_SOFT_MEMORY_BUDGET_BYTES`). The per-job
//! figure is measured, so ten copies of a large model run as many at once as
//! fit, and small ones are not held back. A job always starts when none is
//! running, so this never deadlocks.
//!
//! Moved from exgb, where it bounds concurrent training folds.

use std::sync::{Condvar, Mutex};

/// The pool. Make one `static` per kind of job.
pub struct JobSlots {
    /// `(running, next ticket, ticket now allowed to start, RSS when last idle)`.
    state: Mutex<(usize, u64, u64, usize)>,
    freed: Condvar,
}

/// A held slot; released on drop.
pub struct JobSlot(&'static JobSlots);

impl JobSlots {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new((0, 0, 0, 0)),
            freed: Condvar::new(),
        }
    }

    /// Wait for a slot among `cap`; `None` when serial (`cap <= 1`): there is
    /// nothing to bound.
    pub fn take(&'static self, cap: usize, gate_memory: bool) -> Option<JobSlot> {
        if cap <= 1 {
            return None;
        }
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let ticket = st.1;
        st.1 += 1;
        loop {
            if st.0 < cap && st.2 == ticket {
                if st.0 == 0 {
                    st.3 = crate::process_rss_bytes().unwrap_or(0);
                }
                if !gate_memory
                    || memory_admits(
                        st.0,
                        crate::process_rss_bytes(),
                        st.3,
                        crate::soft_memory_budget_bytes(),
                    )
                {
                    break;
                }
            }
            // a finishing job notifies; memory is re-measured then
            st = self.freed.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.0 += 1;
        st.2 += 1;
        drop(st);
        self.freed.notify_all();
        Some(JobSlot(self))
    }
}

impl Default for JobSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for JobSlot {
    fn drop(&mut self) {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner()).0 -= 1;
        self.0.freed.notify_all();
    }
}

/// Whether one more job fits: `rss + (rss − idle_rss) / running ≤ budget`.
/// Unknown RSS or budget admits (the count cap still applies).
pub fn memory_admits(
    running: usize,
    rss: Option<usize>,
    idle_rss: usize,
    budget: Option<usize>,
) -> bool {
    let (Some(rss), Some(budget)) = (rss, budget) else {
        return true;
    };
    if running == 0 {
        return true;
    }
    let per_job = rss.saturating_sub(idle_rss) / running;
    rss.saturating_add(per_job) <= budget
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The memory gate: the first job always starts; later ones start while
    /// RSS plus one more job's measured growth fits the budget.
    #[test]
    fn the_memory_gate_admits_what_fits_and_never_blocks_an_idle_pool() {
        const G: usize = 1 << 30;
        // idle pool: admitted even far over budget
        assert!(memory_admits(0, Some(100 * G), 0, Some(G)));
        // 2 jobs grew RSS from 1 to 5 GiB: 2 GiB each; 5 + 2 ≤ 8 fits, ≤ 6 does not
        assert!(memory_admits(2, Some(5 * G), G, Some(8 * G)));
        assert!(!memory_admits(2, Some(5 * G), G, Some(6 * G)));
        // unknown RSS or budget: the count cap alone decides
        assert!(memory_admits(3, None, 0, Some(G)));
        assert!(memory_admits(3, Some(100 * G), 0, None));
    }

    /// At most `cap` slots are held at once, and a serial pool hands out none.
    #[test]
    fn the_count_cap_holds() {
        static POOL: JobSlots = JobSlots::new();
        assert!(POOL.take(1, false).is_none());
        let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        std::thread::scope(|s| {
            for _ in 0..12 {
                s.spawn(|| {
                    let _slot = POOL.take(3, false);
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        let p = peak.load(Ordering::SeqCst);
        assert!((1..=3).contains(&p), "peak {p} with cap 3");
    }
}
