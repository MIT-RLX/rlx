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

//! Worker pool with isolation primitives (plan #36).
//!
//! Borrowed from MAX's `serve/worker_interface/`. The serving
//! pattern: engines run in workers (eventually subprocesses); a
//! main router forwards requests via IPC. One worker crashing
//! doesn't take the server down.
//!
//! This module ships the in-process layer (testable, deterministic)
//! plus the trait surface that a future `SubprocessWorker` will
//! implement. The IPC plumbing (stdin/stdout JSON-lines, recovery
//! on crash) is intentionally out of scope until a serving binary
//! consumes it; we'd rather build it once against a real consumer
//! than build it twice.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Stable worker identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct WorkerHealth {
    /// Outstanding requests this worker is processing.
    pub in_flight: u32,
    /// Lifetime requests handled (successful + errored).
    pub completed: u64,
    /// Lifetime requests that errored.
    pub errored: u64,
}

/// Trait every worker implements. `Req` and `Resp` are
/// caller-defined; the future subprocess flavour will use a
/// serde-friendly wire type as both parameters.
pub trait Worker<Req, Resp>: Send + Sync {
    fn id(&self) -> WorkerId;
    fn health(&self) -> WorkerHealth;
    /// Block until this request finishes. Errors propagate the
    /// engine's failure mode without crashing the worker.
    fn dispatch(&self, req: Req) -> Result<Resp, WorkerError>;
}

#[derive(Debug, Clone)]
pub enum WorkerError {
    /// The handler returned a domain error (request was bad,
    /// model rejected it, etc.). Worker stays healthy.
    Domain { reason: String },
    /// The worker itself failed (panic, OOM, lost subprocess).
    /// Pool will route around it on the next request.
    WorkerCrash { reason: String },
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain { reason } => write!(f, "domain error: {reason}"),
            Self::WorkerCrash { reason } => write!(f, "worker crash: {reason}"),
        }
    }
}

impl std::error::Error for WorkerError {}

/// In-process worker — runs the handler closure inline. Useful
/// for tests and for single-process serving. Tracks in-flight /
/// completed / errored counts atomically.
pub struct InProcessWorker<Req, Resp, F>
where
    F: Fn(Req) -> Result<Resp, WorkerError> + Send + Sync,
{
    id: WorkerId,
    handler: F,
    in_flight: AtomicU32,
    completed: AtomicU64,
    errored: AtomicU64,
    // PhantomData<fn() -> _> is always Send + Sync regardless
    // of T's bounds — we don't actually own a Req or Resp.
    _p: std::marker::PhantomData<fn() -> (Req, Resp)>,
}

impl<Req, Resp, F> InProcessWorker<Req, Resp, F>
where
    F: Fn(Req) -> Result<Resp, WorkerError> + Send + Sync,
{
    pub fn new(id: WorkerId, handler: F) -> Self {
        Self {
            id,
            handler,
            in_flight: AtomicU32::new(0),
            completed: AtomicU64::new(0),
            errored: AtomicU64::new(0),
            _p: std::marker::PhantomData,
        }
    }
}

impl<Req, Resp, F> Worker<Req, Resp> for InProcessWorker<Req, Resp, F>
where
    Req: Send,
    Resp: Send,
    F: Fn(Req) -> Result<Resp, WorkerError> + Send + Sync,
{
    fn id(&self) -> WorkerId {
        self.id
    }

    fn health(&self) -> WorkerHealth {
        WorkerHealth {
            in_flight: self.in_flight.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            errored: self.errored.load(Ordering::Relaxed),
        }
    }

    fn dispatch(&self, req: Req) -> Result<Resp, WorkerError> {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let result = (self.handler)(req);
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.completed.fetch_add(1, Ordering::Relaxed);
        if result.is_err() {
            self.errored.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

/// Pool dispatch policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPolicy {
    /// Round-robin: deterministic, ignores load.
    RoundRobin,
    /// Least-loaded: pick the worker with the fewest in-flight
    /// requests; ties broken by `id`.
    LeastLoaded,
}

/// Pool of workers. Generic over `(Req, Resp)`; works with the
/// `Worker` trait directly.
pub struct WorkerPool<Req, Resp> {
    workers: Vec<Arc<dyn Worker<Req, Resp>>>,
    next_rr: Mutex<usize>,
    pub policy: DispatchPolicy,
}

impl<Req, Resp> WorkerPool<Req, Resp> {
    pub fn new(policy: DispatchPolicy) -> Self {
        Self {
            workers: Vec::new(),
            next_rr: Mutex::new(0),
            policy,
        }
    }

    pub fn add(&mut self, worker: Arc<dyn Worker<Req, Resp>>) {
        self.workers.push(worker);
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Pick a worker per `policy`.
    pub fn select(&self) -> Option<&Arc<dyn Worker<Req, Resp>>> {
        if self.workers.is_empty() {
            return None;
        }
        match self.policy {
            DispatchPolicy::RoundRobin => {
                let mut rr = self.next_rr.lock().unwrap();
                let pick = *rr % self.workers.len();
                *rr = (*rr + 1) % self.workers.len();
                Some(&self.workers[pick])
            }
            DispatchPolicy::LeastLoaded => self
                .workers
                .iter()
                .min_by_key(|w| (w.health().in_flight, w.id().0)),
        }
    }

    /// Dispatch a request through the chosen worker.
    pub fn dispatch(&self, req: Req) -> Result<Resp, WorkerError> {
        match self.select() {
            Some(w) => w.dispatch(req),
            None => Err(WorkerError::WorkerCrash {
                reason: "no workers available".into(),
            }),
        }
    }

    /// Snapshot of every worker's health.
    pub fn health(&self) -> Vec<(WorkerId, WorkerHealth)> {
        let mut h: Vec<_> = self.workers.iter().map(|w| (w.id(), w.health())).collect();
        h.sort_by_key(|(id, _)| id.0);
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(
        id: u32,
    ) -> Arc<InProcessWorker<u32, u32, impl Fn(u32) -> Result<u32, WorkerError> + Send + Sync>>
    {
        Arc::new(InProcessWorker::new(WorkerId(id), |x: u32| Ok(x * 2)))
    }

    #[test]
    fn in_process_worker_handles_dispatch() {
        let w = make_worker(7);
        assert_eq!(w.dispatch(5).unwrap(), 10);
        let h = w.health();
        assert_eq!(h.completed, 1);
        assert_eq!(h.errored, 0);
        assert_eq!(h.in_flight, 0);
    }

    #[test]
    fn errors_increment_errored_count() {
        let w: Arc<InProcessWorker<u32, u32, _>> =
            Arc::new(InProcessWorker::new(WorkerId(1), |_x: u32| {
                Err(WorkerError::Domain {
                    reason: "bad".into(),
                })
            }));
        let _ = w.dispatch(1);
        let h = w.health();
        assert_eq!(h.errored, 1);
        assert_eq!(h.completed, 1);
    }

    #[test]
    fn round_robin_visits_each_worker() {
        let mut pool: WorkerPool<u32, u32> = WorkerPool::new(DispatchPolicy::RoundRobin);
        for i in 0..3 {
            pool.add(make_worker(i));
        }

        let mut ids = Vec::new();
        for _ in 0..6 {
            let w = pool.select().unwrap();
            ids.push(w.id().0);
        }
        // 6 picks across 3 workers RR → each worker hit twice in
        // a deterministic 0,1,2,0,1,2 sequence.
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn least_loaded_picks_quietest() {
        // Build three workers; bump in_flight on two of them so
        // the third is the obvious least-loaded pick.
        let w0 = make_worker(0);
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // Manually bump w0 + w1 in-flight via fetch_add.
        w0.in_flight.fetch_add(5, Ordering::Relaxed);
        w1.in_flight.fetch_add(3, Ordering::Relaxed);

        let mut pool: WorkerPool<u32, u32> = WorkerPool::new(DispatchPolicy::LeastLoaded);
        pool.add(w0);
        pool.add(w1);
        pool.add(w2);

        let pick = pool.select().unwrap();
        assert_eq!(
            pick.id().0,
            2,
            "least-loaded should pick the worker with 0 in-flight"
        );
    }

    #[test]
    fn empty_pool_dispatch_errors() {
        let pool: WorkerPool<u32, u32> = WorkerPool::new(DispatchPolicy::RoundRobin);
        let err = pool.dispatch(1).unwrap_err();
        assert!(matches!(err, WorkerError::WorkerCrash { .. }));
    }

    #[test]
    fn health_snapshot_includes_every_worker() {
        let mut pool: WorkerPool<u32, u32> = WorkerPool::new(DispatchPolicy::RoundRobin);
        for i in 0..3 {
            pool.add(make_worker(i));
        }
        let _ = pool.dispatch(1);
        let _ = pool.dispatch(2);
        let h = pool.health();
        assert_eq!(h.len(), 3);
        let total_completed: u64 = h.iter().map(|(_, hh)| hh.completed).sum();
        assert_eq!(total_completed, 2);
    }
}
