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

//! Paged KV cache + continuous batching (plan #31).
//!
//! Borrowed from MAX's `serve/scheduler/{prefill_scheduler,
//! decode_scheduler, text_generation_scheduler, batch_constructor/}`.
//! The standard LLM-serving arch:
//!
//!   - KV cache lives in fixed-size **pages** (block of N tokens
//!     per page per layer), not contiguous per-sequence buffers.
//!     A sequence is a list of page IDs; reaching the end of a
//!     page allocates the next from a pool.
//!   - **Continuous batching**: prefill chunks of new sequences
//!     pack into the same forward pass as decode steps of
//!     in-flight sequences. The batch constructor decides which
//!     work goes into the next forward.
//!
//! Throughput vs. naive max-padding: 5-10× higher at the same
//! latency budget, mostly because GPU utilization stays high
//! when sequences finish at different times.
//!
//! This module is the **data layer** — pool allocation, the
//! sequence-to-page mapping, and the batch packing logic. Kernel
//! integration (gather KV bytes from pages into the attention
//! input) is per-attention-kernel work that lands when an
//! autoregressive LLM enters `rlx-models`.

use std::collections::{BTreeSet, VecDeque};

/// Opaque physical-page identifier from a [`KvPagePool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPageId(pub u32);

/// Fixed-size KV-cache page descriptor. Owns a contiguous range
/// of byte offsets into a backing arena (managed externally).
#[derive(Debug, Clone, Copy)]
pub struct KvPageDesc {
    /// Byte offset into the KV arena for this page.
    pub offset: usize,
    /// Bytes per page = `tokens_per_page * num_layers * 2 (k+v) * num_heads * head_dim * dtype_bytes`.
    pub bytes: usize,
    /// How many tokens this page holds in its leading slots.
    /// Used during prefill where a page may be partial.
    pub filled: u16,
}

/// Pool of fixed-size physical pages. Allocates from a free list;
/// `free` returns a page so the next allocation can reuse it.
///
/// Pool capacity is `num_pages` * `bytes_per_page`. Caller owns the
/// underlying byte arena (typically a single large MTLBuffer or
/// host Vec); the pool tracks which IDs are free.
pub struct KvPagePool {
    /// Sorted set of free page IDs.
    free: BTreeSet<u32>,
    /// Per-page metadata. `descs[i].offset = i * bytes_per_page`.
    descs: Vec<KvPageDesc>,
    /// Constants exposed for ergonomics.
    pub bytes_per_page: usize,
    pub tokens_per_page: u16,
}

impl KvPagePool {
    pub fn new(num_pages: u32, bytes_per_page: usize, tokens_per_page: u16) -> Self {
        let descs: Vec<KvPageDesc> = (0..num_pages)
            .map(|i| KvPageDesc {
                offset: (i as usize) * bytes_per_page,
                bytes: bytes_per_page,
                filled: 0,
            })
            .collect();
        let free: BTreeSet<u32> = (0..num_pages).collect();
        Self {
            free,
            descs,
            bytes_per_page,
            tokens_per_page,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.descs.len() as u32
    }
    pub fn free_count(&self) -> u32 {
        self.free.len() as u32
    }
    pub fn used_count(&self) -> u32 {
        self.capacity() - self.free_count()
    }

    /// Allocate one page. Returns `None` when the pool is empty.
    pub fn alloc(&mut self) -> Option<KvPageId> {
        let id = *self.free.iter().next()?;
        self.free.remove(&id);
        // Reset filled count on alloc — caller starts fresh.
        self.descs[id as usize].filled = 0;
        Some(KvPageId(id))
    }

    pub fn free(&mut self, id: KvPageId) {
        self.free.insert(id.0);
    }

    pub fn descriptor(&self, id: KvPageId) -> &KvPageDesc {
        &self.descs[id.0 as usize]
    }

    pub fn descriptor_mut(&mut self, id: KvPageId) -> &mut KvPageDesc {
        &mut self.descs[id.0 as usize]
    }
}

/// Per-sequence map of logical-token-position → physical page.
/// Token `t` lives at `pages[t / tokens_per_page]` slot
/// `t % tokens_per_page`.
#[derive(Debug, Clone, Default)]
pub struct KvBlockTable {
    pages: Vec<KvPageId>,
    /// Number of tokens this sequence currently has cached.
    pub seq_len: u32,
}

impl KvBlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new page from the pool. Used when seq_len mod
    /// tokens_per_page == 0 (boundary).
    pub fn push_page(&mut self, page: KvPageId) {
        self.pages.push(page);
    }

    /// Look up which page holds token `t`. Returns `None` if `t`
    /// is past the cached region.
    pub fn page_for_token(&self, t: u32, tokens_per_page: u16) -> Option<KvPageId> {
        let idx = (t / tokens_per_page as u32) as usize;
        self.pages.get(idx).copied()
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Free every page back to the pool. Called when a sequence
    /// finishes / is evicted.
    pub fn release(&mut self, pool: &mut KvPagePool) {
        for p in self.pages.drain(..) {
            pool.free(p);
        }
        self.seq_len = 0;
    }

    /// Slice of page IDs. Useful for kernel-side gather.
    pub fn pages(&self) -> &[KvPageId] {
        &self.pages
    }
}

/// One slot in a continuous batch — either a decode step
/// (single new token from a sequence with prior cache) or a
/// prefill chunk (multiple new tokens for a fresh sequence).
#[derive(Debug, Clone)]
pub struct BatchEntry {
    pub seq_id: u64,
    pub kind: BatchKind,
    /// Tokens to feed in this forward pass.
    pub input_tokens: Vec<u32>,
    /// Pre-existing KV-cache length (number of cached tokens before this batch).
    pub cached_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    /// Prefill (or prefill chunk): N new tokens for a sequence
    /// with `cached_len` already prefilled.
    Prefill,
    /// Decode: one token sampled from the previous step.
    Decode,
}

/// Constructs a continuous batch: pulls pending decode requests
/// first (cheap, one-token forwards), then fills remaining
/// **token budget** with prefill chunks. The token budget — not
/// the sequence count — is the gating constraint because
/// prefills can be arbitrarily long.
pub struct BatchConstructor {
    /// Maximum tokens per forward across all entries.
    pub max_tokens_per_batch: usize,
    /// Maximum entries per forward (also bounds memory/scheduler
    /// overhead).
    pub max_entries: usize,
}

impl BatchConstructor {
    pub fn new(max_tokens_per_batch: usize, max_entries: usize) -> Self {
        Self {
            max_tokens_per_batch,
            max_entries,
        }
    }

    /// Build the next batch. Walks `decode_queue` first (one
    /// token each, cheap), then fills remaining token budget by
    /// chunking from `prefill_queue`. Sequences that didn't fit
    /// stay in their queues for the next call.
    pub fn build(
        &self,
        decode_queue: &mut VecDeque<BatchEntry>,
        prefill_queue: &mut VecDeque<BatchEntry>,
    ) -> Vec<BatchEntry> {
        let mut batch: Vec<BatchEntry> = Vec::new();
        let mut tokens_used = 0usize;

        while batch.len() < self.max_entries {
            if let Some(d) = decode_queue.front() {
                let need = d.input_tokens.len();
                if tokens_used + need > self.max_tokens_per_batch {
                    break;
                }
                batch.push(decode_queue.pop_front().unwrap());
                tokens_used += need;
            } else {
                break;
            }
        }

        while batch.len() < self.max_entries {
            let want = match prefill_queue.front() {
                Some(p) => p.input_tokens.len(),
                None => break,
            };
            let remaining = self.max_tokens_per_batch.saturating_sub(tokens_used);
            if remaining == 0 {
                break;
            }

            if want <= remaining {
                batch.push(prefill_queue.pop_front().unwrap());
                tokens_used += want;
            } else {
                // Chunk: take `remaining` tokens off the front,
                // leave the rest for the next batch.
                let mut p = prefill_queue.pop_front().unwrap();
                let chunk: Vec<u32> = p.input_tokens.drain(..remaining).collect();
                let chunk_entry = BatchEntry {
                    seq_id: p.seq_id,
                    kind: BatchKind::Prefill,
                    input_tokens: chunk,
                    cached_len: p.cached_len,
                };
                p.cached_len += remaining as u32;
                batch.push(chunk_entry);
                prefill_queue.push_front(p);
                break;
            }
        }

        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_alloc_free_round_trip() {
        let mut pool = KvPagePool::new(4, 1024, 16);
        assert_eq!(pool.free_count(), 4);
        let p1 = pool.alloc().unwrap();
        let p2 = pool.alloc().unwrap();
        assert_eq!(pool.free_count(), 2);
        pool.free(p1);
        pool.free(p2);
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn pool_returns_none_when_exhausted() {
        let mut pool = KvPagePool::new(2, 64, 4);
        let _a = pool.alloc().unwrap();
        let _b = pool.alloc().unwrap();
        assert!(pool.alloc().is_none());
    }

    #[test]
    fn pool_descriptor_offsets_are_unique_and_aligned() {
        let pool = KvPagePool::new(4, 256, 16);
        for i in 0..4u32 {
            let d = pool.descriptor(KvPageId(i));
            assert_eq!(d.offset, i as usize * 256);
            assert_eq!(d.bytes, 256);
        }
    }

    #[test]
    fn block_table_page_for_token() {
        let mut pool = KvPagePool::new(8, 64, 4);
        let mut bt = KvBlockTable::new();
        for _ in 0..3 {
            bt.push_page(pool.alloc().unwrap());
        }
        // tokens_per_page = 4 → tokens 0..4 in page 0, 4..8 in page 1, ...
        assert_eq!(bt.page_for_token(0, 4), Some(bt.pages()[0]));
        assert_eq!(bt.page_for_token(7, 4), Some(bt.pages()[1]));
        assert_eq!(bt.page_for_token(11, 4), Some(bt.pages()[2]));
        assert_eq!(bt.page_for_token(12, 4), None);
    }

    #[test]
    fn block_table_release_returns_pages() {
        let mut pool = KvPagePool::new(8, 64, 4);
        let mut bt = KvBlockTable::new();
        for _ in 0..3 {
            bt.push_page(pool.alloc().unwrap());
        }
        assert_eq!(pool.free_count(), 5);
        bt.release(&mut pool);
        assert_eq!(pool.free_count(), 8);
        assert_eq!(bt.page_count(), 0);
    }

    #[test]
    fn batch_constructor_decodes_first_then_prefill() {
        let bc = BatchConstructor::new(8, 16);
        let mut decodes: VecDeque<BatchEntry> = (0..3)
            .map(|i| BatchEntry {
                seq_id: i,
                kind: BatchKind::Decode,
                input_tokens: vec![100 + i as u32],
                cached_len: 50,
            })
            .collect();
        let mut prefills: VecDeque<BatchEntry> = (0..2)
            .map(|i| BatchEntry {
                seq_id: 100 + i,
                kind: BatchKind::Prefill,
                input_tokens: vec![1; 3],
                cached_len: 0,
            })
            .collect();

        let batch = bc.build(&mut decodes, &mut prefills);
        // Three decodes (3 tokens) + first prefill (3 tokens) =
        // 6 tokens, fits in budget 8. Second prefill chunks (2
        // tokens of 3) into the remaining 2 slots; the rest
        // stays for the next call.
        assert_eq!(batch.len(), 5);
        // First three are decodes.
        for entry in batch.iter().take(3) {
            assert_eq!(entry.kind, BatchKind::Decode);
        }
        for entry in batch.iter().skip(3).take(2) {
            assert_eq!(entry.kind, BatchKind::Prefill);
        }
        let total_tokens: usize = batch.iter().map(|e| e.input_tokens.len()).sum();
        assert_eq!(total_tokens, 8);

        // The leftover prefill should still be queued (1 token left).
        assert_eq!(prefills.len(), 1);
        assert_eq!(prefills[0].input_tokens.len(), 1);
        assert_eq!(prefills[0].cached_len, 2);
    }

    #[test]
    fn batch_constructor_respects_max_entries() {
        let bc = BatchConstructor::new(1024, 2); // very generous tokens, only 2 entries
        let mut decodes: VecDeque<BatchEntry> = (0..5)
            .map(|i| BatchEntry {
                seq_id: i,
                kind: BatchKind::Decode,
                input_tokens: vec![1],
                cached_len: 0,
            })
            .collect();
        let mut prefills: VecDeque<BatchEntry> = VecDeque::new();
        let batch = bc.build(&mut decodes, &mut prefills);
        assert_eq!(batch.len(), 2);
        assert_eq!(decodes.len(), 3);
    }
}
