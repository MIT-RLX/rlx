// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic KV-cache **retention & retrieval** — the core seam for
//! extending effective context far beyond a fixed resident budget.
//!
//! Decode-time attention over a growing KV cache costs O(context) per token.
//! Naive sliding-window fixes the cost but is "amnesia": it drops old tokens by
//! recency alone. This seam instead keeps a bounded **resident set** of the most
//! *useful* positions — attention sinks, recent tokens, and **heavy-hitters**
//! (positions the model actually attends to) — and offloads the rest to a
//! **block store** that is **retrieved on demand** by query relevance. Effective
//! context is unbounded while per-step attention stays O(budget).
//!
//! This module owns the *decisions* and the *evicted data*; the model/backend
//! owns the live resident K/V tensors and applies each [`RetentionPlan`]:
//! keep/evict resident rows and splice in retrieved blocks. It is deliberately
//! backend- and model-agnostic (K/V are flat `f32` rows of width `kv_dim`), so
//! every model that decodes through a KV cache inherits it.
//!
//! **Staging.** Stage 1 (this module) is the policy engine + block store +
//! `Auto` selection, standalone and unit-tested. Stage 2 wires it into a
//! generator's decode loop; Stage 3 exports real per-position attention weights
//! from the SDPA kernels to drive [`observe_attention`](KvRetentionManager::observe_attention)
//! (until then a caller may feed a proxy signal, e.g. query·key scores).
//!
//! Full design + configuration + RoPE-correctness rationale: `docs/kv-retention.md`.

use std::collections::VecDeque;

/// How importance is scored and which cached positions stay resident.
///
/// All budgets are in **tokens** unless noted. `sinks` are the first-N absolute
/// positions (StreamingLLM's "attention sinks" — models dump excess attention
/// there; keeping them preserves calibration). `recent` is the last-N positions.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum KvRetentionPolicy {
    /// Keep everything. O(context) attention — the exact baseline.
    #[default]
    Full,
    /// StreamingLLM: keep the first `sinks` + last `window` positions, drop the
    /// middle. Cheap and bounded, but recency-only ("dementia" for old facts).
    Sinks { sinks: usize, window: usize },
    /// Keep `sinks` + `recent` + the top-`budget` middle positions by attention
    /// mass (H2O / Scissorhands "heavy-hitters"). Selective, but evicted
    /// positions are **dropped** (not retrievable).
    HeavyHitter {
        sinks: usize,
        recent: usize,
        budget: usize,
    },
    /// Block retrieval: group positions into `block`-sized blocks; evict cold
    /// blocks to the store, and each step **retrieve** the `resident_blocks`
    /// most query-relevant blocks back (plus `sinks` + `recent`). Effective
    /// context is unbounded; per-step attention stays O(resident).
    Retrieval {
        block: usize,
        resident_blocks: usize,
        sinks: usize,
        recent: usize,
    },
    /// Pick automatically from context length and attention concentration:
    /// short context → `Full`; long + concentrated attention → `HeavyHitter`;
    /// long + diffuse attention (broad recall) → `Retrieval`. `max_resident`
    /// bounds the resident budget in every branch.
    Auto { max_resident: usize },
}

/// Per-resident-position metadata used to score retention.
#[derive(Clone, Debug)]
struct Slot {
    /// Absolute position (for RoPE / ordering / sink detection).
    abs_pos: usize,
    /// Accumulated attention mass this position has received (heavy-hitter
    /// signal). Decayed slightly each step so stale hitters fade.
    attn_mass: f32,
    /// Step index at which it was last meaningfully attended (recency tiebreak).
    last_used: u64,
    /// Store-block id this position belongs to (for retrieval grouping).
    #[allow(dead_code)] // populated for retrieval grouping; not yet read on every path
    block: usize,
}

/// A cold block offloaded to the store, keyed by a summary for retrieval.
#[derive(Clone, Debug)]
pub struct StoredBlock {
    /// Block id (stable, monotonic).
    pub id: usize,
    /// Absolute position of the block's first row.
    pub start_pos: usize,
    /// Number of rows (`≤ block`).
    pub rows: usize,
    /// Retrieval key: the mean K over the block's rows (from a representative
    /// layer), width `kv_dim`. Query relevance is `query · key`.
    pub key: Vec<f32>,
    /// Per-layer row-major K: `k[layer]` is `rows × kv_dim`.
    pub k: Vec<Vec<f32>>,
    /// Per-layer row-major V: `v[layer]` is `rows × kv_dim`.
    pub v: Vec<Vec<f32>>,
}

/// The decision for one decode step: how the caller should reshape its resident
/// K/V before the next attention. Indices are into the caller's **current**
/// resident set (same order the manager last saw via [`KvRetentionManager::sync_resident_len`]).
#[derive(Clone, Debug, Default)]
pub struct RetentionPlan {
    /// Resident indices to **retain**, in the new resident order (before any
    /// retrieved blocks are appended).
    pub keep: Vec<usize>,
    /// Resident indices being **evicted** this step. Under a retrieval policy the
    /// caller hands their K/V to [`KvRetentionManager::push_evicted_block`];
    /// otherwise they are dropped.
    pub evict: Vec<usize>,
    /// Store block ids to **retrieve** and splice into the resident set (call
    /// [`KvRetentionManager::take_block`] for their K/V). Ordered by relevance.
    pub retrieve: Vec<usize>,
    /// Whether evicted rows should be stored (`Retrieval`) vs dropped
    /// (`Sinks`/`HeavyHitter`).
    pub store_evicted: bool,
}

impl RetentionPlan {
    /// No-op plan: keep all `n` resident rows, evict/retrieve nothing.
    fn keep_all(n: usize) -> Self {
        RetentionPlan {
            keep: (0..n).collect(),
            evict: Vec::new(),
            retrieve: Vec::new(),
            store_evicted: false,
        }
    }
    /// True when the caller need not reshape anything.
    pub fn is_noop(&self) -> bool {
        self.evict.is_empty() && self.retrieve.is_empty()
    }
}

/// Manages retention metadata + the evicted block store and produces a
/// [`RetentionPlan`] per decode step. One manager per (model, sequence); layers
/// share the retention decision (positions are evicted/retrieved across all
/// layers together), so `k`/`v` in the store are per-layer-concatenated by the
/// caller, or a manager is held per layer — the manager itself is layer-agnostic.
#[derive(Clone, Debug)]
pub struct KvRetentionManager {
    policy: KvRetentionPolicy,
    kv_dim: usize,
    /// Resident position metadata, in resident order.
    resident: Vec<Slot>,
    /// Cold blocks, keyed by id.
    store: Vec<StoredBlock>,
    next_block_id: usize,
    step: u64,
    /// Rolling attention-concentration estimate (top-mass / total) for `Auto`.
    concentration: f32,
    /// Absolute position of the next token to be appended.
    next_pos: usize,
    /// Optional per-step cache/context telemetry, recorded at `commit`.
    recorder: Option<crate::kv_metrics::RetentionRecorder>,
}

/// Decay applied to `attn_mass` each step so a position that stops being
/// attended slowly loses heavy-hitter status (Scissorhands-style forgetting).
const MASS_DECAY: f32 = 0.98;

impl KvRetentionManager {
    /// New manager. `kv_dim` is the per-position K/V width (num_kv_heads·head_dim).
    pub fn new(policy: KvRetentionPolicy, kv_dim: usize) -> Self {
        KvRetentionManager {
            policy,
            kv_dim,
            resident: Vec::new(),
            store: Vec::new(),
            next_block_id: 0,
            step: 0,
            concentration: 0.0,
            next_pos: 0,
            recorder: None,
        }
    }

    pub fn policy(&self) -> &KvRetentionPolicy {
        &self.policy
    }
    /// Start recording per-step cache/context telemetry (resident/evict/retrieve/
    /// store) at each `commit`. Off by default (zero cost). Read back via
    /// [`recorder`](Self::recorder) / [`take_recorder`](Self::take_recorder).
    pub fn enable_recording(&mut self) {
        if self.recorder.is_none() {
            self.recorder = Some(crate::kv_metrics::RetentionRecorder::new());
        }
    }
    /// Borrow the recorded cache/context telemetry, if recording is enabled.
    pub fn recorder(&self) -> Option<&crate::kv_metrics::RetentionRecorder> {
        self.recorder.as_ref()
    }
    /// Mutably borrow the recorder (e.g. to attach a per-step decode latency).
    pub fn recorder_mut(&mut self) -> Option<&mut crate::kv_metrics::RetentionRecorder> {
        self.recorder.as_mut()
    }
    /// Take the recorder out (leaving recording disabled).
    pub fn take_recorder(&mut self) -> Option<crate::kv_metrics::RetentionRecorder> {
        self.recorder.take()
    }
    /// Snapshot of the resident set's **selection preferences** — per-position
    /// `(abs_pos, attn_mass, last_used)` in resident order — for inspection of
    /// *why* positions are kept (which are heavy-hitters, which are stale). The
    /// rolling attention concentration (drives `Auto`) is returned alongside.
    pub fn selection_snapshot(&self) -> (Vec<(usize, f32, u64)>, f32) {
        let sel = self
            .resident
            .iter()
            .map(|s| (s.abs_pos, s.attn_mass, s.last_used))
            .collect();
        (sel, self.concentration)
    }
    /// Number of resident positions the manager is tracking.
    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }
    /// Whether this policy consumes an attention/importance signal — the caller
    /// can skip computing one for purely position/query-based policies.
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.policy,
            KvRetentionPolicy::HeavyHitter { .. } | KvRetentionPolicy::Auto { .. }
        )
    }
    /// Number of cold blocks in the store (retrievable context).
    pub fn stored_blocks(&self) -> usize {
        self.store.len()
    }
    /// Total tokens the store holds (retrievable-but-not-resident context).
    pub fn stored_tokens(&self) -> usize {
        self.store.iter().map(|b| b.rows).sum()
    }

    /// Register `n` prefill positions as resident (absolute positions `0..n`).
    /// A fresh prefill starts a new sequence, so the cold store + block keys are
    /// cleared (multi-turn continuation does NOT call this, so cross-turn context
    /// is preserved).
    pub fn on_prefill(&mut self, n: usize) {
        self.resident.clear();
        self.store.clear();
        self.resident.reserve(n);
        let block_sz = self.block_size().max(1);
        for pos in 0..n {
            self.resident.push(Slot {
                abs_pos: pos,
                attn_mass: 0.0,
                last_used: 0,
                block: pos / block_sz,
            });
        }
        self.next_pos = n;
        self.next_block_id = self.next_block_id.max(n.div_ceil(block_sz));
    }

    /// Append one just-decoded token as the newest resident position.
    pub fn append(&mut self) {
        let block_sz = self.block_size().max(1);
        let pos = self.next_pos;
        self.resident.push(Slot {
            abs_pos: pos,
            attn_mass: 0.0,
            last_used: self.step,
            block: pos / block_sz,
        });
        self.next_pos += 1;
    }

    /// Record the attention distribution the last step placed over the resident
    /// set (`weights.len() == resident_len()`), aggregated across heads/layers by
    /// the caller. Updates the heavy-hitter mass + recency + concentration. A
    /// caller without real weights may pass a proxy (e.g. softmax of query·key).
    pub fn observe_attention(&mut self, weights: &[f32]) {
        let n = self.resident.len().min(weights.len());
        // Decay then accumulate.
        for s in self.resident.iter_mut() {
            s.attn_mass *= MASS_DECAY;
        }
        let mut total = 0.0f32;
        let mut peak = 0.0f32;
        for i in 0..n {
            let w = weights[i].max(0.0);
            self.resident[i].attn_mass += w;
            if w > 1e-4 {
                self.resident[i].last_used = self.step;
            }
            total += w;
            if w > peak {
                peak = w;
            }
        }
        // Concentration ≈ peak / total (1 ⇒ one dominant key; ~1/n ⇒ diffuse).
        if total > 0.0 {
            let c = peak / total;
            self.concentration = 0.9 * self.concentration + 0.1 * c;
        }
        self.step += 1;
    }

    /// The block size implied by the policy (for grouping/store), or 0 if the
    /// policy is not block-based.
    fn block_size(&self) -> usize {
        match self.policy {
            KvRetentionPolicy::Retrieval { block, .. } => block,
            KvRetentionPolicy::Auto { max_resident } => (max_resident / 8).max(16),
            _ => 0,
        }
    }

    /// Resolve `Auto` to a concrete policy given the current state.
    fn effective_policy(&self) -> KvRetentionPolicy {
        match self.policy {
            KvRetentionPolicy::Auto { max_resident } => {
                let ctx = self.resident.len() + self.stored_tokens();
                if ctx <= max_resident {
                    // Fits — no eviction needed.
                    KvRetentionPolicy::Full
                } else if self.concentration >= 0.15 {
                    // Attention is concentrated on a few keys → heavy-hitters
                    // capture it well; keep a resident budget, drop the rest.
                    let sinks = 4.min(max_resident / 8);
                    let recent = max_resident / 2;
                    let budget = max_resident.saturating_sub(sinks + recent);
                    KvRetentionPolicy::HeavyHitter {
                        sinks,
                        recent,
                        budget,
                    }
                } else {
                    // Diffuse attention (broad recall) → block retrieval so old
                    // context can be pulled back when a later query needs it.
                    let block = (max_resident / 8).max(16);
                    let sinks = 4.min(max_resident / 8);
                    let recent = max_resident / 2;
                    let resident_blocks =
                        max_resident.saturating_sub(sinks + recent) / block.max(1);
                    KvRetentionPolicy::Retrieval {
                        block,
                        resident_blocks: resident_blocks.max(1),
                        sinks,
                        recent,
                    }
                }
            }
            ref p => p.clone(),
        }
    }

    /// Decide how to reshape the resident set for the next step. `query_key` is
    /// the current query summary (width `kv_dim`) used to score block relevance
    /// under `Retrieval`; pass `None` for non-retrieval policies.
    pub fn plan(&mut self, query_key: Option<&[f32]>) -> RetentionPlan {
        let n = self.resident.len();
        match self.effective_policy() {
            KvRetentionPolicy::Full => RetentionPlan::keep_all(n),
            KvRetentionPolicy::Sinks { sinks, window } => {
                self.plan_keep_set(sinks, window, 0, |_| 0.0, false, None)
            }
            KvRetentionPolicy::HeavyHitter {
                sinks,
                recent,
                budget,
            } => {
                let masses: Vec<f32> = self.resident.iter().map(|s| s.attn_mass).collect();
                self.plan_keep_set(sinks, recent, budget, |i| masses[i], false, None)
            }
            KvRetentionPolicy::Retrieval {
                block,
                resident_blocks,
                sinks,
                recent,
            } => self.plan_retrieval(block, resident_blocks, sinks, recent, query_key),
            KvRetentionPolicy::Auto { .. } => unreachable!("resolved by effective_policy"),
        }
    }

    /// Shared keep-set builder: always keep the first `sinks` and last `recent`
    /// resident rows; from the middle, keep the top-`budget` by `score` (0 for
    /// pure sinks+window). Evicts the rest.
    fn plan_keep_set(
        &self,
        sinks: usize,
        recent: usize,
        budget: usize,
        score: impl Fn(usize) -> f32,
        store_evicted: bool,
        retrieve: Option<Vec<usize>>,
    ) -> RetentionPlan {
        let n = self.resident.len();
        let sinks = sinks.min(n);
        let recent = recent.min(n.saturating_sub(sinks));
        let mid_lo = sinks;
        let mid_hi = n - recent;
        // Rank the middle by score, keep the top `budget`.
        let mut mid: Vec<usize> = (mid_lo..mid_hi).collect();
        if budget < mid.len() {
            mid.sort_by(|&a, &b| {
                score(b)
                    .partial_cmp(&score(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            mid.truncate(budget);
        }
        let mut keep_flag = vec![false; n];
        for i in 0..sinks {
            keep_flag[i] = true;
        }
        for i in mid_hi..n {
            keep_flag[i] = true;
        }
        for &i in &mid {
            keep_flag[i] = true;
        }
        let keep: Vec<usize> = (0..n).filter(|&i| keep_flag[i]).collect();
        let evict: Vec<usize> = (0..n).filter(|&i| !keep_flag[i]).collect();
        RetentionPlan {
            keep,
            evict,
            retrieve: retrieve.unwrap_or_default(),
            store_evicted,
        }
    }

    /// Retrieval plan: keep sinks + recent; evict the cold middle to the store;
    /// retrieve the most query-relevant stored blocks up to `resident_blocks`.
    /// Evict-all/retrieve-all every step — the per-step re-selection + re-chunking
    /// is maximally query-responsive, which the probe showed is what preserves
    /// recall (3/3). An incremental variant that keeps stable blocks in place cut
    /// churn ~91% but regressed recall to 1/3 because fixed-alignment blocks are
    /// coarser than this path's adaptive re-chunking — reverted; see
    /// `docs/kv-retention.md`.
    fn plan_retrieval(
        &mut self,
        _block: usize, // store-block size is applied by the caller when chunking evictions
        resident_blocks: usize,
        sinks: usize,
        recent: usize,
        query_key: Option<&[f32]>,
    ) -> RetentionPlan {
        let retrieve: Vec<usize> = match query_key {
            Some(q) if !self.store.is_empty() => {
                let mut scored: Vec<(usize, f32)> =
                    self.store.iter().map(|b| (b.id, dot(q, &b.key))).collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored
                    .into_iter()
                    .take(resident_blocks)
                    .map(|(id, _)| id)
                    .collect()
            }
            _ => Vec::new(),
        };
        // `budget = 0` ⇒ evict the whole cold middle; keep sinks + recent.
        self.plan_keep_set(sinks, recent, 0, |_| 0.0, true, Some(retrieve))
    }

    /// Store an evicted contiguous run of rows as a cold block. `k`/`v` are
    /// per-layer, each `rows × kv_dim` row-major. The retrieval key is the mean K
    /// of the representative layer (index 0). Returns the new block id.
    pub fn push_evicted_block(
        &mut self,
        start_pos: usize,
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
    ) -> usize {
        let rows = if self.kv_dim > 0 {
            k.first().map(|l| l.len() / self.kv_dim).unwrap_or(0)
        } else {
            0
        };
        let key = mean_rows(
            k.first().map(|l| l.as_slice()).unwrap_or(&[]),
            self.kv_dim,
            rows,
        );
        let id = self.next_block_id;
        self.next_block_id += 1;
        self.store.push(StoredBlock {
            id,
            start_pos,
            rows,
            key,
            k,
            v,
        });
        id
    }

    /// Block size implied by the policy (for the caller's eviction chunking), or
    /// 0 for non-block policies.
    pub fn store_block_size(&self) -> usize {
        self.block_size()
    }

    /// Remove and return a stored block's data by id (called on retrieval).
    pub fn take_block(&mut self, id: usize) -> Option<StoredBlock> {
        if let Some(idx) = self.store.iter().position(|b| b.id == id) {
            Some(self.store.remove(idx))
        } else {
            None
        }
    }

    /// Apply a plan's `keep` to the manager's own resident metadata (the caller
    /// applies it to the real K/V in the same order). Call after acting on the
    /// plan so the manager's view stays in sync. `appended_positions` are the
    /// absolute positions of any retrieved blocks' rows spliced in, in order.
    pub fn commit(&mut self, plan: &RetentionPlan, retrieved_positions: &[usize]) {
        let mut next: Vec<Slot> = Vec::with_capacity(plan.keep.len() + retrieved_positions.len());
        for &i in &plan.keep {
            if let Some(s) = self.resident.get(i) {
                next.push(s.clone());
            }
        }
        let block_sz = self.block_size().max(1);
        for &pos in retrieved_positions {
            next.push(Slot {
                abs_pos: pos,
                attn_mass: 0.0,
                last_used: self.step,
                block: pos / block_sz,
            });
        }
        // Keep resident in absolute-position order (attention/RoPE expect it),
        // deduped by position so re-chunked overlapping blocks can't double a row
        // — the caller's K/V rebuild dedups the same way.
        next.sort_by_key(|s| s.abs_pos);
        next.dedup_by_key(|s| s.abs_pos);
        self.resident = next;

        if let Some(rec) = &mut self.recorder {
            let step = rec.len();
            rec.push(crate::kv_metrics::StepRecord {
                step,
                resident: self.resident.len(),
                evicted: plan.evict.len(),
                retrieved: retrieved_positions.len(),
                store_blocks: self.store.len(),
                store_tokens: self.store.iter().map(|b| b.rows).sum(),
                decode_ms: None,
            });
        }
    }

    /// Absolute positions of the current resident set, in resident order —
    /// needed by the caller to rebuild RoPE / masks after a reshape.
    pub fn resident_positions(&self) -> Vec<usize> {
        self.resident.iter().map(|s| s.abs_pos).collect()
    }
}

/// Dot product — the block-relevance metric. (An earlier attempt to
/// cosine-normalize this was reverted: the re-bench showed cosine *lowered* the
/// importance contrast and regressed retrieval recall 3/3→0/3, while raw dot
/// matches the pre-opt selection. The relevance ceiling is the signal itself —
/// K·K similarity — not the normalization; sharpening it needs exact attention
/// weights, Stage-3+.)
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

/// Mean over `rows` row-major rows of width `inner` (the block retrieval key).
fn mean_rows(data: &[f32], inner: usize, rows: usize) -> Vec<f32> {
    if inner == 0 || rows == 0 {
        return vec![0.0; inner];
    }
    let mut out = vec![0.0f32; inner];
    for r in 0..rows {
        let base = r * inner;
        for j in 0..inner {
            out[j] += data[base + j];
        }
    }
    let inv = 1.0 / rows as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

// A small ring for callers that want a fixed recent window without policy help.
#[allow(dead_code)]
type RecentRing = VecDeque<usize>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_keeps_everything() {
        let mut m = KvRetentionManager::new(KvRetentionPolicy::Full, 4);
        m.on_prefill(10);
        let plan = m.plan(None);
        assert_eq!(plan.keep.len(), 10);
        assert!(plan.evict.is_empty());
        assert!(plan.is_noop());
    }

    #[test]
    fn sinks_keeps_first_and_last() {
        let mut m = KvRetentionManager::new(
            KvRetentionPolicy::Sinks {
                sinks: 2,
                window: 3,
            },
            4,
        );
        m.on_prefill(10);
        let plan = m.plan(None);
        // keep positions 0,1 (sinks) and 7,8,9 (window)
        assert_eq!(plan.keep, vec![0, 1, 7, 8, 9]);
        assert_eq!(plan.evict, vec![2, 3, 4, 5, 6]);
        assert!(!plan.store_evicted);
    }

    #[test]
    fn heavy_hitter_keeps_high_attention_middle() {
        let mut m = KvRetentionManager::new(
            KvRetentionPolicy::HeavyHitter {
                sinks: 1,
                recent: 1,
                budget: 2,
            },
            4,
        );
        m.on_prefill(8);
        // Give position 3 and 5 lots of attention mass; 2,4,6 little.
        let mut w = vec![0.0f32; 8];
        w[3] = 5.0;
        w[5] = 4.0;
        w[2] = 0.1;
        w[4] = 0.1;
        w[6] = 0.1;
        m.observe_attention(&w);
        let plan = m.plan(None);
        // sinks=0, recent=7, plus top-2 middle by mass = {3,5}.
        assert!(plan.keep.contains(&0)); // sink
        assert!(plan.keep.contains(&7)); // recent
        assert!(plan.keep.contains(&3)); // heavy hitter
        assert!(plan.keep.contains(&5)); // heavy hitter
        assert!(plan.evict.contains(&2)); // low mass evicted
        assert!(plan.evict.contains(&4));
    }

    #[test]
    fn retrieval_stores_and_retrieves_by_query() {
        let kv = 2;
        let mut m = KvRetentionManager::new(
            KvRetentionPolicy::Retrieval {
                block: 2,
                resident_blocks: 1,
                sinks: 1,
                recent: 1,
            },
            kv,
        );
        // Sinks + recent resident (pos 0,1); two cold blocks already offloaded to
        // the store with distinct keys.
        m.on_prefill(2);
        let a = m.push_evicted_block(
            2,
            vec![vec![1.0, 0.0, 1.0, 0.0]],
            vec![vec![9.0, 9.0, 9.0, 9.0]],
        ); // key ≈ [1,0]
        let b = m.push_evicted_block(
            4,
            vec![vec![0.0, 1.0, 0.0, 1.0]],
            vec![vec![8.0, 8.0, 8.0, 8.0]],
        ); // key ≈ [0,1]
        assert_eq!(m.stored_blocks(), 2);
        assert_eq!(m.stored_tokens(), 4);
        // A query aligned with block `b`'s key [0,1] retrieves `b`, not `a`.
        let q = vec![0.0, 1.0];
        let plan2 = m.plan(Some(&q));
        assert_eq!(plan2.retrieve, vec![b]);
        assert_ne!(plan2.retrieve, vec![a]);
        let blk = m.take_block(b).unwrap();
        assert_eq!(blk.start_pos, 4);
        assert_eq!(m.stored_blocks(), 1);
    }

    #[test]
    fn auto_full_when_fits_then_selective_when_long() {
        let mut m = KvRetentionManager::new(KvRetentionPolicy::Auto { max_resident: 64 }, 4);
        m.on_prefill(32);
        // Fits within budget → Full.
        assert!(m.plan(None).is_noop());
        // Grow past budget with concentrated attention → heavy-hitter (evicts).
        m.on_prefill(100);
        let mut w = vec![0.01f32; 100];
        w[50] = 10.0; // concentrated
        for _ in 0..5 {
            m.observe_attention(&w);
        }
        let plan = m.plan(None);
        assert!(
            !plan.evict.is_empty(),
            "long+concentrated context should evict"
        );
    }

    #[test]
    fn commit_reorders_by_absolute_position() {
        let mut m = KvRetentionManager::new(
            KvRetentionPolicy::Sinks {
                sinks: 1,
                window: 1,
            },
            2,
        );
        m.on_prefill(4);
        let plan = m.plan(None); // keep {0,3}
        m.commit(&plan, &[1]); // splice retrieved position 1 back
        assert_eq!(m.resident_positions(), vec![0, 1, 3]);
    }
}
