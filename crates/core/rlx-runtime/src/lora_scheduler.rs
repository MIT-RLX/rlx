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

//! LoRA-aware request scheduling (plan #33).
//!
//! Borrowed from MAX's `serve/scheduler/lora_scheduler_utils.py`.
//! When multiple LoRA adapters are loaded, each request specifies
//! which adapter it wants. Naïvely interleaving requests forces an
//! adapter swap per request — wasted bandwidth + latency. The
//! scheduler here groups consecutive same-adapter requests into
//! one "batch", so a swap happens once per batch boundary instead
//! of once per request.
//!
//! Pure data-layer scheduling — no executor, no compiled graphs.
//! Plug into a future serving loop by:
//!   1. Push incoming requests into [`LoraScheduler::push`].
//!   2. Drain runnable batches with [`LoraScheduler::drain_batch`].
//!   3. For each batch, swap to that adapter once and run all
//!      requests in the batch back-to-back.
//!
//! Pairs with #24 (named weights registry) which owns the adapter
//! bytes, and #9 (LoRA kernel) which is the actual compute.

use crate::weight_registry::WeightRegistry;
use std::collections::VecDeque;

/// One serving request with its target LoRA adapter (or `None`
/// for "use the base model"). The opaque `id: u64` is whatever
/// the caller wants — typically a request UUID hash.
#[derive(Debug, Clone)]
pub struct LoraRequest {
    pub id: u64,
    pub adapter: Option<String>,
    /// Caller-provided payload. The scheduler doesn't interpret
    /// it; downstream code reads it after `drain_batch`.
    pub payload: LoraPayload,
}

/// Generic request payload. The scheduler is generic over
/// payload type via `Box<dyn Any>`; we keep this concrete here
/// for simplicity and the common case (text generation).
#[derive(Debug, Clone)]
pub struct LoraPayload {
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
}

/// One batch handed to the executor. All requests in a batch
/// target the same adapter (or all None).
#[derive(Debug)]
pub struct LoraBatch {
    pub adapter: Option<String>,
    pub requests: Vec<LoraRequest>,
}

impl LoraBatch {
    pub fn len(&self) -> usize {
        self.requests.len()
    }
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// FIFO-ish scheduler with same-adapter coalescing.
pub struct LoraScheduler {
    /// Pending requests; insertion order preserved.
    pending: VecDeque<LoraRequest>,
    /// Maximum requests per drained batch.
    pub max_batch: usize,
    /// Optional reference to a registry — `validate` checks that
    /// adapter names are registered before push. Storing as a raw
    /// pointer avoids lifetime entanglement; callers ensure the
    /// registry outlives the scheduler.
    registry: Option<*const WeightRegistry>,
}

// `*const WeightRegistry` is not Send/Sync by default; we mark
// the scheduler Send because the pointer is only used for read
// queries on a registry that's itself Send + Sync once we wrap
// it in Arc<RwLock>. The pointer should never be mutated.
unsafe impl Send for LoraScheduler {}

impl LoraScheduler {
    pub fn new(max_batch: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            max_batch,
            registry: None,
        }
    }

    /// Bind a registry for adapter-name validation. Caller must
    /// ensure `registry` outlives the scheduler.
    pub fn bind_registry(&mut self, registry: &WeightRegistry) {
        self.registry = Some(registry as *const _);
    }

    /// Push a request. Returns `Err` only if a registry is bound
    /// and the adapter name isn't registered.
    pub fn push(&mut self, req: LoraRequest) -> Result<(), UnknownAdapter> {
        if let (Some(reg_ptr), Some(adapter)) = (self.registry, &req.adapter) {
            // Safety: caller guaranteed the registry outlives us.
            let reg = unsafe { &*reg_ptr };
            if reg.lora_adapter_handles(adapter).is_empty() {
                return Err(UnknownAdapter {
                    name: adapter.clone(),
                });
            }
        }
        self.pending.push_back(req);
        Ok(())
    }

    /// Look at the next batch's adapter without draining.
    pub fn peek_adapter(&self) -> Option<Option<String>> {
        self.pending.front().map(|r| r.adapter.clone())
    }

    /// Drain the next runnable batch — up to `max_batch` requests
    /// that all share the same `adapter`. Returns `None` if empty.
    pub fn drain_batch(&mut self) -> Option<LoraBatch> {
        let head = self.pending.pop_front()?;
        let target = head.adapter.clone();
        let mut requests = vec![head];
        while requests.len() < self.max_batch {
            match self.pending.front() {
                Some(next) if next.adapter == target => {
                    requests.push(self.pending.pop_front().unwrap());
                }
                _ => break,
            }
        }
        Some(LoraBatch {
            adapter: target,
            requests,
        })
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct UnknownAdapter {
    pub name: String,
}

impl std::fmt::Display for UnknownAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "adapter `{}` is not registered", self.name)
    }
}
impl std::error::Error for UnknownAdapter {}

/// Count adapter swaps that would occur if a sequence of
/// requests were processed in declared order without coalescing.
/// Useful for observability / unit tests.
pub fn naive_swap_count(reqs: &[LoraRequest]) -> usize {
    let mut swaps: usize = 0;
    let mut last: Option<&Option<String>> = None;
    for r in reqs {
        if last.map(|l| l != &r.adapter).unwrap_or(true) {
            swaps += 1;
        }
        last = Some(&r.adapter);
    }
    swaps.saturating_sub(1) // first "swap" is the initial setup, not a swap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_registry::{WeightKind, WeightRegistry};
    use rlx_ir::{DType, Shape};
    use std::sync::Arc;

    fn req(id: u64, adapter: Option<&str>) -> LoraRequest {
        LoraRequest {
            id,
            adapter: adapter.map(|s| s.to_string()),
            payload: LoraPayload {
                prompt_tokens: vec![],
                max_new_tokens: 4,
            },
        }
    }

    #[test]
    fn coalesces_same_adapter_runs() {
        let mut s = LoraScheduler::new(8);
        // Mixed-adapter input order:
        //   code, code, math, math, code, base, base
        for r in [
            req(1, Some("code")),
            req(2, Some("code")),
            req(3, Some("math")),
            req(4, Some("math")),
            req(5, Some("code")),
            req(6, None),
            req(7, None),
        ] {
            s.push(r).unwrap();
        }

        let b1 = s.drain_batch().unwrap();
        assert_eq!(b1.adapter.as_deref(), Some("code"));
        assert_eq!(b1.len(), 2); // 1, 2 — stops at the math at front

        let b2 = s.drain_batch().unwrap();
        assert_eq!(b2.adapter.as_deref(), Some("math"));
        assert_eq!(b2.len(), 2);

        let b3 = s.drain_batch().unwrap();
        assert_eq!(b3.adapter.as_deref(), Some("code"));
        assert_eq!(b3.len(), 1);

        let b4 = s.drain_batch().unwrap();
        assert!(b4.adapter.is_none());
        assert_eq!(b4.len(), 2);

        assert!(s.drain_batch().is_none());
    }

    #[test]
    fn respects_max_batch_cap() {
        let mut s = LoraScheduler::new(3);
        for i in 0..10 {
            s.push(req(i, Some("code"))).unwrap();
        }
        let b = s.drain_batch().unwrap();
        assert_eq!(b.len(), 3, "max_batch=3 should split a long run");
        assert_eq!(s.pending(), 7);
    }

    #[test]
    fn registry_validation_rejects_unknown_adapter() {
        let mut reg = WeightRegistry::new();
        reg.register(
            "ffn",
            Shape::new(&[8, 8], DType::F32),
            Arc::from(vec![0u8; 256]),
            WeightKind::Base,
        );
        reg.register(
            "ffn.lora.a",
            Shape::new(&[8, 4], DType::F32),
            Arc::from(vec![0u8; 128]),
            WeightKind::LoraAdapter {
                adapter: "code".into(),
            },
        );

        let mut s = LoraScheduler::new(4);
        s.bind_registry(&reg);

        // Known adapter passes.
        assert!(s.push(req(1, Some("code"))).is_ok());
        // None (base model) always passes.
        assert!(s.push(req(2, None)).is_ok());
        // Unknown adapter rejected.
        let err = s.push(req(3, Some("nonexistent"))).unwrap_err();
        assert_eq!(err.name, "nonexistent");
    }

    #[test]
    fn swap_count_metric() {
        let reqs = [
            req(1, Some("a")),
            req(2, Some("a")),
            req(3, Some("b")),
            req(4, Some("a")),
        ];
        // Sequence transitions: a, a→a (no swap), a→b (swap), b→a
        // (swap) = 2 swaps after initial.
        assert_eq!(naive_swap_count(&reqs), 2);
    }
}
