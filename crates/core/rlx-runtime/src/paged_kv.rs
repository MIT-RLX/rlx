// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
//!   - **Prefix caching** ([`KvPrefixCache`], MAX `genai-paged-attention`
//!     / SGLang radix trie): full pages are keyed by the token chunk they
//!     hold under their parent, so a new prompt walks the trie and reuses
//!     every page it shares with an earlier one. Pages are refcounted;
//!     unreferenced cached pages are evicted LRU when the pool runs dry.
//!
//! This module is the **data layer** — pool allocation, the
//! sequence-to-page mapping, prefix sharing, and the batch packing logic. Kernel
//! integration (gather KV bytes from pages into the attention
//! input) is per-attention-kernel work that lands when an
//! autoregressive LLM model runners.

use std::collections::{BTreeSet, HashMap, VecDeque};

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
    /// Owners per page (block tables + prefix cache). 0 ⇔ on the free list.
    refs: Vec<u32>,
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
            refs: vec![0; num_pages as usize],
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

    /// Allocate one page with a single owner. Returns `None` when the
    /// pool is empty.
    pub fn alloc(&mut self) -> Option<KvPageId> {
        let id = *self.free.iter().next()?;
        self.free.remove(&id);
        // Reset filled count on alloc — caller starts fresh.
        self.descs[id as usize].filled = 0;
        self.refs[id as usize] = 1;
        Some(KvPageId(id))
    }

    /// Add an owner to an allocated page (prefix sharing).
    pub fn retain(&mut self, id: KvPageId) {
        let r = &mut self.refs[id.0 as usize];
        assert!(*r > 0, "retain of free KV page {}", id.0);
        *r += 1;
    }

    /// Drop one owner; the page returns to the free list when none remain.
    pub fn free(&mut self, id: KvPageId) {
        let r = &mut self.refs[id.0 as usize];
        assert!(*r > 0, "double free of KV page {}", id.0);
        *r -= 1;
        if *r == 0 {
            self.free.insert(id.0);
        }
    }

    /// Current owner count (0 = free).
    pub fn ref_count(&self, id: KvPageId) -> u32 {
        self.refs[id.0 as usize]
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

/// Result of [`KvPrefixCache::match_prefix`]: the reusable head of a prompt.
#[derive(Debug, Clone, Default)]
pub struct PrefixMatch {
    /// Shared full pages, in order; each was retained once for the caller.
    /// Push them onto the new sequence's [`KvBlockTable`] as-is.
    pub pages: Vec<KvPageId>,
    /// Tokens covered by `pages` plus `copy` (skip these in prefill).
    pub matched_tokens: u32,
    /// Copy-on-write of a partially shared page, when one exists.
    pub copy: Option<PageCopy>,
}

/// Copy the leading `tokens` slots of `src` into the freshly allocated `dst`
/// (owned by the caller) before writing new rows into `dst`'s tail. Cheaper
/// than recomputing those projections; `src` stays immutable for its owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCopy {
    pub src: KvPageId,
    pub dst: KvPageId,
    pub tokens: u16,
}

#[derive(Debug)]
struct TrieNode {
    page: KvPageId,
    parent: usize,
    /// The `tokens_per_page` tokens this node's page holds.
    chunk: Box<[u32]>,
    children: HashMap<Box<[u32]>, usize>,
    last_use: u64,
}

/// Radix trie over full KV pages (SGLang/MAX prefix cache).
///
/// Each edge is exactly one page of tokens, so lookups are O(prompt pages)
/// hash probes. A page's K/V depends on every token before it, which is why
/// the key is the *path* (chunk under its parent), never the chunk alone.
/// The cache holds one reference on every page it indexes; a page whose only
/// owner is the cache is evictable, leaves first, least recently used first.
#[derive(Debug)]
pub struct KvPrefixCache {
    tokens_per_page: u16,
    /// Slot 0 is the root (no page). Freed slots are reused via `vacant`.
    nodes: Vec<Option<TrieNode>>,
    vacant: Vec<usize>,
    clock: u64,
}

const ROOT: usize = 0;

impl KvPrefixCache {
    pub fn new(tokens_per_page: u16) -> Self {
        assert!(tokens_per_page > 0);
        let root = TrieNode {
            page: KvPageId(u32::MAX),
            parent: ROOT,
            chunk: Box::new([]),
            children: HashMap::new(),
            last_use: 0,
        };
        Self {
            tokens_per_page,
            nodes: vec![Some(root)],
            vacant: Vec::new(),
            clock: 0,
        }
    }

    fn node(&self, i: usize) -> &TrieNode {
        self.nodes[i].as_ref().expect("live trie node")
    }

    fn node_mut(&mut self, i: usize) -> &mut TrieNode {
        self.nodes[i].as_mut().expect("live trie node")
    }

    /// Pages indexed by the cache.
    pub fn cached_pages(&self) -> usize {
        self.nodes.len() - 1 - self.vacant.len()
    }

    /// Longest page-aligned cached prefix of `tokens`, retained for the
    /// caller. At least one token is always left unmatched so the forward
    /// has a position to produce logits from. With `pool`, a partially
    /// matching next page is planned as a [`PageCopy`] into a fresh page
    /// (skipped when the pool has no free page).
    pub fn match_prefix(&mut self, tokens: &[u32], pool: &mut KvPagePool) -> PrefixMatch {
        let tpp = self.tokens_per_page as usize;
        let limit = tokens.len().saturating_sub(1);
        self.clock += 1;
        let now = self.clock;
        let mut at = ROOT;
        let mut out = PrefixMatch::default();
        let mut pos = 0usize;
        while pos + tpp <= limit {
            let Some(&child) = self.node(at).children.get(&tokens[pos..pos + tpp]) else {
                break;
            };
            let n = self.node_mut(child);
            n.last_use = now;
            pool.retain(n.page);
            out.pages.push(n.page);
            at = child;
            pos += tpp;
        }
        // Partial page: the child sharing the longest leading run.
        let rest = &tokens[pos..limit];
        let best = self
            .node(at)
            .children
            .values()
            .map(|&c| {
                let n = self.node(c);
                let r = n.chunk.iter().zip(rest).take_while(|(a, b)| a == b).count();
                (r, c)
            })
            .max();
        if let Some((r, c)) = best
            && r > 0
            && let Some(dst) = pool.alloc()
        {
            let n = self.node_mut(c);
            n.last_use = now;
            out.copy = Some(PageCopy {
                src: n.page,
                dst,
                tokens: r as u16,
            });
            pos += r;
        }
        out.matched_tokens = pos as u32;
        out
    }

    /// Index every full page of a sequence whose K/V for `tokens` is now in
    /// `table`. Pages already cached under the same path are left as-is (the
    /// sequence keeps its own copy); new ones gain a cache reference.
    pub fn insert(&mut self, tokens: &[u32], table: &KvBlockTable, pool: &mut KvPagePool) {
        let tpp = self.tokens_per_page as usize;
        let full = (tokens.len() / tpp).min(table.page_count());
        self.clock += 1;
        let now = self.clock;
        let mut at = ROOT;
        for (i, &page) in table.pages().iter().enumerate().take(full) {
            let chunk = &tokens[i * tpp..(i + 1) * tpp];
            if let Some(&child) = self.node(at).children.get(chunk) {
                self.node_mut(child).last_use = now;
                at = child;
                continue;
            }
            pool.retain(page);
            let node = TrieNode {
                page,
                parent: at,
                chunk: chunk.into(),
                children: HashMap::new(),
                last_use: now,
            };
            let id = match self.vacant.pop() {
                Some(v) => {
                    self.nodes[v] = Some(node);
                    v
                }
                None => {
                    self.nodes.push(Some(node));
                    self.nodes.len() - 1
                }
            };
            self.node_mut(at).children.insert(chunk.into(), id);
            at = id;
        }
    }

    /// Evict up to `want` cache-only leaf pages (LRU first), returning them to
    /// `pool`. Returns how many pages were freed.
    pub fn evict(&mut self, want: u32, pool: &mut KvPagePool) -> u32 {
        let mut freed = 0;
        while freed < want {
            let victim = self
                .nodes
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(i, n)| n.as_ref().map(|n| (i, n)))
                .filter(|(_, n)| n.children.is_empty() && pool.ref_count(n.page) == 1)
                .min_by_key(|(_, n)| n.last_use)
                .map(|(i, _)| i);
            let Some(v) = victim else { break };
            let n = self.nodes[v].take().expect("victim is live");
            self.node_mut(n.parent).children.remove(&n.chunk);
            self.vacant.push(v);
            pool.free(n.page);
            freed += 1;
        }
        freed
    }

    /// [`KvPagePool::alloc`], evicting one cached page when the pool is dry.
    pub fn alloc(&mut self, pool: &mut KvPagePool) -> Option<KvPageId> {
        pool.alloc().or_else(|| {
            self.evict(1, pool);
            pool.alloc()
        })
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
    fn pool_refcount_frees_on_last_owner() {
        let mut pool = KvPagePool::new(2, 64, 4);
        let p = pool.alloc().unwrap();
        pool.retain(p);
        pool.free(p);
        assert_eq!(pool.free_count(), 1);
        pool.free(p);
        assert_eq!(pool.free_count(), 2);
    }

    /// Prefill `tokens` into a fresh block table, skipping the matched head.
    fn prefill(
        cache: &mut KvPrefixCache,
        pool: &mut KvPagePool,
        tokens: &[u32],
    ) -> (KvBlockTable, PrefixMatch) {
        let m = cache.match_prefix(tokens, pool);
        let mut bt = KvBlockTable::new();
        for &p in &m.pages {
            bt.push_page(p);
        }
        if let Some(c) = m.copy {
            bt.push_page(c.dst);
        }
        let tpp = pool.tokens_per_page as usize;
        while bt.page_count() * tpp < tokens.len() {
            bt.push_page(cache.alloc(pool).unwrap());
        }
        bt.seq_len = tokens.len() as u32;
        cache.insert(tokens, &bt, pool);
        (bt, m)
    }

    #[test]
    fn prefix_cache_shares_full_pages() {
        let mut pool = KvPagePool::new(16, 64, 4);
        let mut cache = KvPrefixCache::new(4);
        // "I like to eat | cheese cake and ice"
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (bt_a, m_a) = prefill(&mut cache, &mut pool, &a);
        assert_eq!(m_a.matched_tokens, 0);
        assert_eq!(cache.cached_pages(), 2);
        // Same first page, diverging second page: page 0 shared, 3-token COW.
        let b = [1, 2, 3, 4, 5, 6, 7, 99, 10, 11];
        let (bt_b, m_b) = prefill(&mut cache, &mut pool, &b);
        assert_eq!(m_b.pages, vec![bt_a.pages()[0]]);
        let copy = m_b.copy.unwrap();
        assert_eq!((copy.src, copy.tokens), (bt_a.pages()[1], 3));
        assert_eq!(m_b.matched_tokens, 7);
        assert_eq!(bt_b.pages()[0], bt_a.pages()[0]);
        assert_ne!(bt_b.pages()[1], bt_a.pages()[1]);
        // seq A + seq B + cache all own the shared page.
        assert_eq!(pool.ref_count(bt_a.pages()[0]), 3);
    }

    #[test]
    fn prefix_cache_keys_on_path_not_chunk() {
        let mut pool = KvPagePool::new(16, 64, 2);
        let mut cache = KvPrefixCache::new(2);
        prefill(&mut cache, &mut pool, &[1, 2, 3, 4, 0]);
        // Page [3,4] appears, but after a different first page: no reuse.
        let m = cache.match_prefix(&[9, 9, 3, 4, 0], &mut pool);
        assert!(m.pages.is_empty() && m.copy.is_none());
        assert_eq!(m.matched_tokens, 0);
    }

    #[test]
    fn prefix_cache_leaves_one_token_for_logits() {
        let mut pool = KvPagePool::new(16, 64, 4);
        let mut cache = KvPrefixCache::new(4);
        let t = [1, 2, 3, 4, 5, 6, 7, 8];
        prefill(&mut cache, &mut pool, &t);
        let m = cache.match_prefix(&t, &mut pool);
        // Identical prompt: page 0 shared, page 1 copied up to the last token.
        assert_eq!(m.pages.len(), 1);
        assert_eq!(m.copy.unwrap().tokens, 3);
        assert_eq!(m.matched_tokens, 7);
    }

    #[test]
    fn prefix_cache_evicts_lru_unreferenced_leaves() {
        let mut pool = KvPagePool::new(4, 64, 2);
        let mut cache = KvPrefixCache::new(2);
        let (mut bt_a, _) = prefill(&mut cache, &mut pool, &[1, 2, 3, 4]);
        let (mut bt_b, _) = prefill(&mut cache, &mut pool, &[5, 6, 7, 8]);
        assert_eq!(pool.free_count(), 0);
        // Live sequences pin their pages: nothing evictable.
        assert_eq!(cache.evict(1, &mut pool), 0);
        bt_a.release(&mut pool);
        bt_b.release(&mut pool);
        // Touch B so A is LRU; leaf [3,4] goes before its parent [1,2].
        let _ = cache.match_prefix(&[5, 6, 7, 8, 0], &mut pool);
        assert_eq!(cache.evict(2, &mut pool), 2);
        assert!(cache.match_prefix(&[1, 2, 0], &mut pool).pages.is_empty());
        assert_eq!(cache.cached_pages(), 2);
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
