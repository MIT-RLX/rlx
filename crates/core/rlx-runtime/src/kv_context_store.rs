// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Million-token KV context store** — the IO-tiered backing that lets KV
//! retention reach ~1e6 tokens of context on a bounded RAM/GPU working set.
//!
//! Two "smart arrangements" combine here:
//!
//! 1. **Selection is sub-linear.** Every block contributes a tiny *key* (mean K,
//!    `kv_dim` floats) to an in-RAM [`Hnsw`](crate::hnsw::Hnsw) index. A query
//!    finds the top-k relevant blocks in `O(log N)` — so navigating 15k–125k
//!    blocks (a million tokens) costs microseconds, not a linear key scan.
//! 2. **Data lives on disk, paged on demand.** All block K/V rows are appended,
//!    **quantized**, to a per-layer memory-mapped file
//!    ([`MmapKvLayer`](crate::quantized_kv::mmap::MmapKvLayer)). The store is
//!    **append-only** (blocks are written once as context grows; retrieval
//!    *copies* the selected rows out, never moving them), and only the retrieved
//!    top-k blocks' pages fault in. At Q4_0 that is ~32 GB on disk for 1e6 tokens
//!    of a 28-layer / kv_dim-1024 model, with a working set of just
//!    `budget + k·block` rows.
//!
//! RAM cost is `keys + HNSW graph` — for 1e6 tokens in 64-row blocks that is
//! ~15.6k keys ≈ tens of MB, independent of the tens-of-GB of K/V on disk.
//!
//! Requires the `mmap-kv` feature. Compose it into a generator's retention loop:
//! append a block when it ages out of the recent window; each decode step,
//! `retrieve(query, k)` the relevant blocks and splice them into the bounded
//! resident set.

use anyhow::Result;
use std::path::Path;

use crate::hnsw::{Hnsw, HnswConfig};
use crate::quantized_kv::KvQuant;
use crate::quantized_kv::mmap::MmapKvLayer;

/// Provenance of a context block — where its tokens came from. Tracking this
/// lets a caller weight/filter retrieved context by source (e.g. prefer the
/// user's files over the model's own generations, or tag retrieved spans).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// A user query / prompt turn.
    Query,
    /// Ingested file / document content.
    File,
    /// The model's own generated output.
    Generated,
    /// System / instruction context.
    System,
    /// Previously retrieved-and-reinserted context.
    Retrieved,
    /// Anything else, tagged with a caller-defined id.
    Other(u16),
}

/// Where a block's rows live + its provenance.
#[derive(Clone, Debug)]
struct BlockRef {
    /// First row index in every layer's mmap.
    start_row: usize,
    /// Number of rows.
    rows: usize,
    /// Absolute position of the block's first token (for RoPE / ordering).
    start_pos: usize,
    /// Provenance of this block's tokens.
    origin: Origin,
    /// Optional caller source id (file index, turn number, …).
    source_id: u32,
}

/// A retrieved block, materialized (dequantized) to f32.
#[derive(Clone, Debug)]
pub struct RetrievedBlock {
    /// Absolute position of the first row.
    pub start_pos: usize,
    /// Rows in this block.
    pub rows: usize,
    /// Relevance score (inner product of query · block key).
    pub score: f32,
    /// Provenance of this block (query / file / generated / …).
    pub origin: Origin,
    /// Caller source id (file index, turn number, …).
    pub source_id: u32,
    /// `true` if pulled in as a *similar neighbor* of a primary hit rather than a
    /// direct top-k match (neighbor-expanded context).
    pub via_neighbor: bool,
    /// Per-layer row-major K, each `rows × kv_dim`.
    pub k: Vec<Vec<f32>>,
    /// Per-layer row-major V, each `rows × kv_dim`.
    pub v: Vec<Vec<f32>>,
}

/// How a block's HNSW navigation keys are chosen from its rows. Both keep the
/// index size at `centroids_per_block` keys/block; they differ in whether those
/// keys are *averaged* (recall-limited) or *actual rows* (late-interaction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KeyMode {
    /// k-means centroids (or a single mean when `centroids_per_block == 1`).
    /// Averages cluster members, so a lone exact-fact token is diluted in its
    /// key and may not be navigable.
    #[default]
    Centroid,
    /// The block's most-salient actual K rows (highest L2 norm) as keys — no
    /// averaging, so the query navigates HNSW *toward the strong-matching token*
    /// itself. This is the index-side of MaxSim: the fact row becomes directly
    /// reachable. Pairs with [`retrieve_maxsim`](KvContextStore::retrieve_maxsim).
    RowKeys,
}

/// Append-only, disk-tiered, HNSW-indexed block store for unbounded context.
pub struct KvContextStore {
    kv_dim: usize,
    n_layers: usize,
    #[allow(dead_code)]
    scheme: KvQuant,
    /// One append-only quantized mmap per layer.
    layers: Vec<MmapKvLayer>,
    /// Block metadata.
    blocks: Vec<BlockRef>,
    /// Relevance index over block **centroids** (k-means, not a single mean — a
    /// mean averages away the block's discriminative content, hurting recall).
    hnsw: Hnsw,
    /// Maps an HNSW node id → its block id (a block contributes
    /// `centroids_per_block` centroids, so ids are not 1:1 with blocks).
    centroid_block: Vec<u32>,
    /// Centroids per block (1 = plain mean; >1 = k-means sub-keys). Also caps the
    /// number of row-keys per block in [`KeyMode::RowKeys`].
    centroids_per_block: usize,
    /// How per-block HNSW keys are derived (averaged centroids vs salient rows).
    key_mode: KeyMode,
    /// Per-block last-access step, for memory decay.
    last_used: Vec<usize>,
    /// Global access clock (advances per decayed retrieval).
    clock: usize,
    /// Per-step recency multiplier ∈ (0,1] applied by age in decayed retrieval;
    /// `1.0` = no decay.
    decay: f32,
    /// Search width (≥ k). Higher = better recall, more work.
    ef_search: usize,
    total_rows: usize,
    // ── Semantic embedding index (dual-encoder retrieval) ──
    /// Secondary HNSW over per-block **content embeddings** (one/block, 1:1 with
    /// `embed_block`). A retrieval-trained embedding is a well-conditioned,
    /// content-similarity space — unlike raw post-RoPE K, HNSW navigates it with
    /// near-exact recall AND it's far more selective, so small-topk retrieval lands
    /// the right block at 1M scale. `None` until `enable_embeddings`.
    embed_hnsw: Option<Hnsw>,
    /// Maps an embed-HNSW node id → block id (1:1: one embedding per block).
    embed_block: Vec<u32>,
    /// Raw per-block embedding vectors (parallel to `embed_block`), kept so
    /// [`retrieve_embed_exact`](KvContextStore::retrieve_embed_exact) can brute-force
    /// — cheap at block granularity (~31k vecs for 1M tokens) and immune to the
    /// HNSW navigation-recall loss that ANN suffers on clustered indexes.
    embed_vecs: Vec<Vec<f32>>,
    /// Embedding dimensionality (0 until enabled).
    embed_dim: usize,
    // ── Lexical (hybrid) retrieval ──
    /// Per-block token ids (for lexical / BM25-lite scoring). Empty = no lexical.
    block_tokens: Vec<Vec<u32>>,
    /// Inverted index: token id → block ids containing it (for sub-linear lexical
    /// candidate gathering, so exact-token matches — numbers, names — are found
    /// even when dense K·K similarity misses them).
    inverted: std::collections::HashMap<u32, Vec<u32>>,
    /// Document frequency per token id (for IDF weighting).
    doc_freq: std::collections::HashMap<u32, u32>,
    /// Retrieval read telemetry (interior-mutable so it accrues on `&self` reads):
    /// number of `read_blocks_batched` calls, blocks materialized, and nanoseconds
    /// spent in the disk/dequant read. A harness contrasts a cold first query with
    /// warm repeats to see whether page faults (mmap) dominate. See `read_stats`.
    read_calls: std::sync::atomic::AtomicU64,
    read_blocks_ct: std::sync::atomic::AtomicU64,
    read_nanos: std::sync::atomic::AtomicU64,
}

impl KvContextStore {
    /// Create a store for `n_layers × kv_dim`, quantized with `scheme`, sized for
    /// `capacity_rows` tokens. `dir = Some` persists one file per layer
    /// (`ctx_kv_{i}.bin`); `None` uses anonymous (swap-backed) maps.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_layers: usize,
        kv_dim: usize,
        scheme: KvQuant,
        capacity_rows: usize,
        dir: Option<&Path>,
        hnsw_cfg: HnswConfig,
        ef_search: usize,
        centroids_per_block: usize,
        decay: f32,
    ) -> Result<Self> {
        Self::new_with_reuse(
            n_layers,
            kv_dim,
            scheme,
            capacity_rows,
            dir,
            hnsw_cfg,
            ef_search,
            centroids_per_block,
            decay,
            false,
        )
    }

    /// Like [`new`](Self::new), with optional reuse of existing mmap layer files.
    ///
    /// When `reuse_existing_files` is true and `dir` contains prior `ctx_kv_*.bin`
    /// files of the expected size, mappings are opened without truncation so
    /// persisted K/V rows can be read again while retrieval metadata is rebuilt.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_reuse(
        n_layers: usize,
        kv_dim: usize,
        scheme: KvQuant,
        capacity_rows: usize,
        dir: Option<&Path>,
        hnsw_cfg: HnswConfig,
        ef_search: usize,
        centroids_per_block: usize,
        decay: f32,
        reuse_existing_files: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer = match dir {
                Some(d) => {
                    std::fs::create_dir_all(d).ok();
                    let path = d.join(format!("ctx_kv_{i}.bin"));
                    if reuse_existing_files && path.exists() {
                        MmapKvLayer::open_existing(path, kv_dim, scheme, capacity_rows)?
                    } else {
                        MmapKvLayer::open(path, kv_dim, scheme, capacity_rows)?
                    }
                }
                None => MmapKvLayer::anonymous(kv_dim, scheme, capacity_rows)?,
            };
            // The store is a random-access block index: retrieval touches a handful
            // of scattered blocks per query. Disable sequential read-ahead so a cold
            // block read faults in only its own pages, not unused neighbors.
            layer.advise_random();
            layers.push(layer);
        }
        Ok(Self {
            kv_dim,
            n_layers,
            scheme,
            layers,
            blocks: Vec::new(),
            hnsw: Hnsw::new(kv_dim, hnsw_cfg),
            centroid_block: Vec::new(),
            centroids_per_block: centroids_per_block.max(1),
            key_mode: KeyMode::default(),
            last_used: Vec::new(),
            clock: 0,
            decay: decay.clamp(0.0, 1.0),
            ef_search: ef_search.max(1),
            total_rows: 0,
            embed_hnsw: None,
            embed_block: Vec::new(),
            embed_vecs: Vec::new(),
            embed_dim: 0,
            block_tokens: Vec::new(),
            inverted: std::collections::HashMap::new(),
            doc_freq: std::collections::HashMap::new(),
            read_calls: std::sync::atomic::AtomicU64::new(0),
            read_blocks_ct: std::sync::atomic::AtomicU64::new(0),
            read_nanos: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Import block metadata for an already-persisted K/V span without writing
    /// any new rows. This is intended for store-reuse flows that reconstruct
    /// retrieval state from a sidecar manifest.
    ///
    /// `rows` rows are assumed to already exist at the next contiguous store
    /// offset (`total_rows`) in every layer file.
    pub fn import_block(
        &mut self,
        start_pos: usize,
        rows: usize,
        origin: Origin,
        source_id: u32,
    ) -> Result<usize> {
        if rows == 0 {
            return Ok(usize::MAX);
        }
        if self.total_rows + rows > self.layers[0].capacity_rows {
            anyhow::bail!(
                "import_block: would exceed capacity ({} + {} > {})",
                self.total_rows,
                rows,
                self.layers[0].capacity_rows
            );
        }
        let start_row = self.total_rows;
        let block_id = self.blocks.len() as u32;
        self.blocks.push(BlockRef {
            start_row,
            rows,
            start_pos,
            origin,
            source_id,
        });
        self.last_used.push(self.clock);
        self.total_rows += rows;
        for layer in &mut self.layers {
            layer.past_len = self.total_rows;
        }
        Ok(block_id as usize)
    }

    /// Choose how per-block HNSW keys are derived. Call before appending blocks
    /// (existing blocks keep the keys they were indexed with). [`KeyMode::RowKeys`]
    /// makes navigation late-interaction-aware (salient rows, no averaging) — the
    /// index-side complement to [`retrieve_maxsim`](Self::retrieve_maxsim).
    pub fn set_key_mode(&mut self, mode: KeyMode) {
        self.key_mode = mode;
    }

    /// Attach the token ids of block `block_id` for lexical retrieval (call right
    /// after [`append_block`](Self::append_block)). Dedups tokens per block and
    /// updates the inverted index + document frequencies.
    pub fn attach_tokens(&mut self, block_id: usize, tokens: &[u32]) {
        if block_id >= self.blocks.len() {
            return;
        }
        while self.block_tokens.len() <= block_id {
            self.block_tokens.push(Vec::new());
        }
        let mut uniq: Vec<u32> = tokens.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        for &t in &uniq {
            self.inverted.entry(t).or_default().push(block_id as u32);
            *self.doc_freq.entry(t).or_insert(0) += 1;
        }
        self.block_tokens[block_id] = uniq;
    }

    /// **Hybrid** retrieval: blend the dense (HNSW K/centroid) score with a
    /// **lexical** BM25-lite score (IDF-weighted query-token overlap via the
    /// inverted index), then take top-k. `lexical_weight ∈ [0,1]` mixes them
    /// (0 = pure dense, 1 = pure lexical). Lexical rescues exact-token facts
    /// (numbers, names, shared keywords) that K·K similarity misses.
    pub fn retrieve_hybrid(
        &self,
        dense_query: &[f32],
        query_tokens: &[u32],
        k: usize,
        lexical_weight: f32,
    ) -> Vec<RetrievedBlock> {
        use std::collections::HashMap;
        // Dense candidates (over-fetch so lexical can re-rank a wide pool).
        let pool = (k * 8).max(64);
        let dense = self.block_hits(dense_query, pool);
        // Normalize dense scores to [0,1] within the pool.
        let (dmin, dmax) = dense
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &(_, s)| {
                (a.min(s), b.max(s))
            });
        let drange = (dmax - dmin).max(1e-6);
        let mut score: HashMap<u32, f32> = HashMap::new();
        for &(b, s) in &dense {
            score.insert(b, (1.0 - lexical_weight) * ((s - dmin) / drange));
        }
        // Lexical candidates from the inverted index (IDF-weighted overlap).
        if lexical_weight > 0.0 && !query_tokens.is_empty() {
            let n = self.blocks.len().max(1) as f32;
            let mut q: Vec<u32> = query_tokens.to_vec();
            q.sort_unstable();
            q.dedup();
            let mut lex: HashMap<u32, f32> = HashMap::new();
            for &t in &q {
                if let Some(blocks) = self.inverted.get(&t) {
                    let df = *self.doc_freq.get(&t).unwrap_or(&1) as f32;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                    for &b in blocks {
                        *lex.entry(b).or_insert(0.0) += idf;
                    }
                }
            }
            let lmax = lex.values().cloned().fold(1e-6f32, f32::max);
            for (b, l) in lex {
                *score.entry(b).or_insert(0.0) += lexical_weight * (l / lmax);
            }
        }
        let mut ranked: Vec<(u32, f32)> = score.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        ranked
            .into_iter()
            .filter_map(|(b, s)| self.read_block(b, s, false))
            .collect()
    }

    /// Enable the semantic-embedding index: a secondary HNSW over one
    /// content-embedding per block (dim `dim`). Call once before appending; each
    /// `append_block` should be followed by [`append_embed`](Self::append_embed)
    /// with that block's embedding. Embeddings are the SELECTIVE, scale-friendly
    /// retrieval signal (K·K is not); this is the dual-encoder path.
    pub fn enable_embeddings(&mut self, dim: usize, cfg: HnswConfig) {
        self.embed_dim = dim;
        self.embed_hnsw = Some(Hnsw::new(dim, cfg));
        self.embed_block.clear();
        self.embed_vecs.clear();
    }

    /// True once [`enable_embeddings`](Self::enable_embeddings) has been called.
    pub fn embeddings_enabled(&self) -> bool {
        self.embed_hnsw.is_some()
    }

    /// Attach `block_id`'s content embedding to the semantic index (1 per block).
    /// No-op if embeddings aren't enabled or the dim mismatches.
    pub fn append_embed(&mut self, block_id: usize, embed_key: &[f32]) {
        if embed_key.len() != self.embed_dim {
            return;
        }
        if let Some(h) = self.embed_hnsw.as_mut() {
            h.insert(embed_key);
            self.embed_block.push(block_id as u32);
            self.embed_vecs.push(embed_key.to_vec());
        }
    }

    /// **Exact** semantic retrieval: brute-force cosine over EVERY block embedding
    /// (no HNSW). At block granularity (~31k vecs for 1M tokens × 384-d ≈ 12M
    /// mults/query) this is sub-millisecond AND immune to HNSW's navigation-recall
    /// loss on clustered indexes — the correct choice up to ~1M-token scale; HNSW
    /// is for far larger. Returns top-k blocks by cosine (embeddings are unit-norm).
    pub fn retrieve_embed_exact(&self, query: &[f32], k: usize) -> Vec<RetrievedBlock> {
        if query.len() != self.embed_dim {
            return Vec::new();
        }
        let mut scored: Vec<(u32, f32)> = self
            .embed_vecs
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let d: f32 = e.iter().zip(query).map(|(a, b)| a * b).sum();
                (self.embed_block[i], d)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        self.read_blocks_batched(scored)
    }

    /// Embedding-index block hits (best-first). 1 embedding/block so node id maps
    /// straight through `embed_block` (no centroid dedup needed).
    fn embed_hits(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let Some(h) = self.embed_hnsw.as_ref() else {
            return Vec::new();
        };
        let want = k.max(1);
        h.search(query, want, self.ef_search.max(want))
            .into_iter()
            .filter_map(|(node, s)| self.embed_block.get(node as usize).map(|&b| (b, s)))
            .collect()
    }

    /// **Semantic retrieval**: top-k blocks by content-embedding similarity
    /// (secondary HNSW). The selective, 1M-scalable path — a retrieval-trained
    /// embedding puts a question next to its answer where K·K cannot.
    pub fn retrieve_embed(&self, query: &[f32], k: usize) -> Vec<RetrievedBlock> {
        self.embed_hits(query, k)
            .into_iter()
            .take(k)
            .inspect(|&(b, _)| self.prefetch(b))
            .filter_map(|(b, score)| self.read_block(b, score, false))
            .collect()
    }

    /// IDF (BM25-lite) lexical scores over `query_tokens`, block → summed IDF of
    /// shared tokens (via the inverted index). Empty if no lexical tokens attached.
    fn lexical_scores(&self, query_tokens: &[u32]) -> std::collections::HashMap<u32, f32> {
        use std::collections::HashMap;
        let mut lex: HashMap<u32, f32> = HashMap::new();
        if query_tokens.is_empty() {
            return lex;
        }
        let n = self.blocks.len().max(1) as f32;
        let mut q: Vec<u32> = query_tokens.to_vec();
        q.sort_unstable();
        q.dedup();
        for &t in &q {
            if let Some(blocks) = self.inverted.get(&t) {
                let df = *self.doc_freq.get(&t).unwrap_or(&1) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                for &b in blocks {
                    *lex.entry(b).or_insert(0.0) += idf;
                }
            }
        }
        lex
    }

    /// Exact-embedding block scores (cosine), block → similarity. All blocks.
    fn embed_scores_exact(&self, query: &[f32]) -> Vec<(u32, f32)> {
        if query.len() != self.embed_dim {
            return Vec::new();
        }
        self.embed_vecs
            .iter()
            .enumerate()
            .map(|(i, e)| {
                (
                    self.embed_block[i],
                    e.iter().zip(query).map(|(a, b)| a * b).sum::<f32>(),
                )
            })
            .collect()
    }

    /// **Reciprocal Rank Fusion** of semantic-embedding and lexical (BM25-lite)
    /// rankings (+ optional K·K dense). RRF combines *ranks* — `Σ 1/(rrf_k + rank)`
    /// — so it's immune to the score-SCALE mismatch that made weighted-sum lexical
    /// blending noisy. Embedding catches paraphrase; lexical nails exact tokens
    /// (numbers / names / IPs) where the encoder is weakest → recovers the
    /// borderline needles a pure-embedding top-k crowds out. `rrf_k≈60` standard.
    #[allow(clippy::too_many_arguments)]
    pub fn retrieve_rrf(
        &self,
        embed_query: &[f32],
        dense_query: &[f32],
        query_tokens: &[u32],
        k: usize,
        rrf_k: f32,
        use_lex: bool,
        use_dense: bool,
    ) -> Vec<RetrievedBlock> {
        use std::collections::HashMap;
        let pool = (k * 16).max(128);
        let mut fused: HashMap<u32, f32> = HashMap::new();
        let mut add_ranked = |mut ranked: Vec<(u32, f32)>| {
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            for (rank, (b, _)) in ranked.into_iter().take(pool).enumerate() {
                *fused.entry(b).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
            }
        };
        add_ranked(self.embed_scores_exact(embed_query));
        if use_lex {
            add_ranked(self.lexical_scores(query_tokens).into_iter().collect());
        }
        if use_dense && !dense_query.is_empty() {
            add_ranked(self.block_hits(dense_query, pool));
        }
        let mut ranked: Vec<(u32, f32)> = fused.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        self.read_blocks_batched(ranked)
    }

    /// **3-way hybrid**: blend semantic embedding + IDF lexical (BM25-lite) [+ K·K
    /// via `dense_query` when `w_dense > 0`]. Scores are min-max normalized within
    /// each candidate pool before weighting. Embedding catches paraphrase, lexical
    /// nails exact tokens (numbers/names), K·K adds the model's attention geometry
    /// — together they push selective recall toward 10/10.
    #[allow(clippy::too_many_arguments)]
    pub fn retrieve_hybrid3(
        &self,
        embed_query: &[f32],
        dense_query: &[f32],
        query_tokens: &[u32],
        k: usize,
        w_embed: f32,
        w_lex: f32,
        w_dense: f32,
        gate: f32,
    ) -> Vec<RetrievedBlock> {
        use std::collections::HashMap;
        let pool = (k * 8).max(64);
        let mut score: HashMap<u32, f32> = HashMap::new();
        let mut blend = |cands: Vec<(u32, f32)>, w: f32| {
            if w == 0.0 || cands.is_empty() {
                return;
            }
            let (lo, hi) = cands
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &(_, s)| {
                    (a.min(s), b.max(s))
                });
            let spread = hi - lo;
            for (b, s) in cands {
                // All-equal / single candidate → full weight (they're all the best
                // in their pool); min-max only when there's a real spread.
                let norm = if spread > 1e-6 {
                    (s - lo) / spread
                } else {
                    1.0
                };
                *score.entry(b).or_insert(0.0) += w * norm;
            }
        };
        if w_embed > 0.0 {
            blend(self.embed_hits(embed_query, pool), w_embed);
        }
        if w_dense > 0.0 && !dense_query.is_empty() {
            blend(self.block_hits(dense_query, pool), w_dense);
        }
        if w_lex > 0.0 && !query_tokens.is_empty() {
            let n = self.blocks.len().max(1) as f32;
            let mut q: Vec<u32> = query_tokens.to_vec();
            q.sort_unstable();
            q.dedup();
            let mut lex: HashMap<u32, f32> = HashMap::new();
            for &t in &q {
                if let Some(blocks) = self.inverted.get(&t) {
                    let df = *self.doc_freq.get(&t).unwrap_or(&1) as f32;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
                    for &b in blocks {
                        *lex.entry(b).or_insert(0.0) += idf;
                    }
                }
            }
            blend(lex.into_iter().collect(), w_lex);
        }
        let mut ranked: Vec<(u32, f32)> = score.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);
        // RELEVANCE GATE (noise minimizer): keep only blocks whose blended score is
        // within `gate` of the top hit. When one block clearly dominates, splice
        // just it; when nothing is competitive, splice fewer (or none) — so the
        // model isn't flooded with irrelevant retrieved context (the failure that
        // blanks generation at large topk). `gate = 0` keeps all top-k (off).
        if gate > 0.0 {
            if let Some(&(_, top)) = ranked.first() {
                let floor = gate * top;
                ranked.retain(|&(_, s)| s >= floor);
            }
        }
        self.read_blocks_batched(ranked)
    }

    /// Append one block (written once): its per-layer K/V rows are quantized onto
    /// disk, and **`centroids_per_block` k-means centroids** of its rows are
    /// inserted into the HNSW index (each mapped back to this block). Using
    /// centroids rather than a single mean preserves the block's discriminative
    /// content — a buried fact stays findable via its own centroid instead of
    /// being averaged into the block mean. `key` is used only when
    /// `centroids_per_block == 1` (plain-mean mode). Returns the block id.
    pub fn append_block(
        &mut self,
        start_pos: usize,
        origin: Origin,
        source_id: u32,
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        key: &[f32],
    ) -> Result<usize> {
        let rows = if self.kv_dim > 0 {
            k.first().map(|l| l.len() / self.kv_dim).unwrap_or(0)
        } else {
            0
        };
        if rows == 0 {
            return Ok(usize::MAX);
        }
        let start_row = self.total_rows;
        for l in 0..self.n_layers {
            self.layers[l].append_rows(&k[l], &v[l])?;
        }
        self.total_rows += rows;
        let block_id = self.blocks.len() as u32;
        // Per-block HNSW navigation keys. Centroid mode averages (recall-limited);
        // RowKeys mode indexes the most-salient actual rows so the query navigates
        // toward the strong-matching token itself (late-interaction in the index).
        let keys: Vec<Vec<f32>> = match self.key_mode {
            KeyMode::Centroid if self.centroids_per_block <= 1 => vec![key.to_vec()],
            KeyMode::Centroid => {
                kmeans_centroids(&k[0], rows, self.kv_dim, self.centroids_per_block)
            }
            KeyMode::RowKeys => salient_rows(&k[0], rows, self.kv_dim, self.centroids_per_block),
        };
        for c in &keys {
            self.hnsw.insert(c);
            self.centroid_block.push(block_id);
        }
        self.blocks.push(BlockRef {
            start_row,
            rows,
            start_pos,
            origin,
            source_id,
        });
        self.last_used.push(self.clock);
        Ok(block_id as usize)
    }

    /// Read one **block** (by block id; dequant all layers) into a `RetrievedBlock`.
    fn read_block(&self, block_id: u32, score: f32, via_neighbor: bool) -> Option<RetrievedBlock> {
        let br = self.blocks.get(block_id as usize)?.clone();
        let mut k_layers = Vec::with_capacity(self.n_layers);
        let mut v_layers = Vec::with_capacity(self.n_layers);
        for layer in &self.layers {
            let (kk, vv) = layer.read_rows(br.start_row, br.rows).ok()?;
            k_layers.push(kk);
            v_layers.push(vv);
        }
        Some(RetrievedBlock {
            start_pos: br.start_pos,
            rows: br.rows,
            score,
            origin: br.origin,
            source_id: br.source_id,
            via_neighbor,
            k: k_layers,
            v: v_layers,
        })
    }

    /// Collapse raw centroid hits (HNSW ids) to unique **block** hits, keeping the
    /// best centroid score per block, best-first.
    fn dedup_to_blocks(&self, hits: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
        use std::collections::HashMap;
        let mut best: HashMap<u32, f32> = HashMap::new();
        for (cid, s) in hits {
            if let Some(&b) = self.centroid_block.get(cid as usize) {
                best.entry(b).and_modify(|e| *e = e.max(s)).or_insert(s);
            }
        }
        let mut v: Vec<(u32, f32)> = best.into_iter().collect();
        // Score desc, then block id asc as a deterministic tiebreak (the HashMap
        // order is not stable, and equal-score ties must resolve reproducibly).
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        v
    }

    /// HNSW search → unique block hits (centroids deduped), best-first. Over-fetches
    /// centroids so `k` distinct blocks survive the dedup.
    fn block_hits(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let want = (k * self.centroids_per_block).max(k) * 2;
        let raw = self.hnsw.search(query, want, self.ef_search.max(want));
        let mut b = self.dedup_to_blocks(raw);
        b.truncate(k);
        b
    }

    fn prefetch(&self, block_id: u32) {
        if let Some(br) = self.blocks.get(block_id as usize) {
            for layer in &self.layers {
                layer.prefetch_rows(br.start_row, br.rows);
            }
        }
    }

    /// Read the selected `(block, score)` hits into materialized blocks, issuing a
    /// `WILLNEED` prefetch for **every** hit up front so their cold pages fault in
    /// concurrently, then reading. A lazy `.inspect(prefetch).filter_map(read)`
    /// chain instead interleaves prefetch/read per block, so each read blocks on
    /// its own fault with no overlap — this batches the hint phase. Order and
    /// selection are unchanged (behavior-preserving).
    fn read_blocks_batched(&self, hits: Vec<(u32, f32)>) -> Vec<RetrievedBlock> {
        use std::sync::atomic::Ordering::Relaxed;
        for &(b, _) in &hits {
            self.prefetch(b);
        }
        let t0 = std::time::Instant::now();
        let out: Vec<RetrievedBlock> = hits
            .into_iter()
            .filter_map(|(b, score)| self.read_block(b, score, false))
            .collect();
        let dt = t0.elapsed().as_nanos() as u64;
        self.read_calls.fetch_add(1, Relaxed);
        self.read_blocks_ct.fetch_add(out.len() as u64, Relaxed);
        self.read_nanos.fetch_add(dt, Relaxed);
        if std::env::var_os("RLX_KVSTORE_READ_STATS").is_some() {
            let us = dt as f64 / 1000.0;
            let n = out.len().max(1);
            eprintln!(
                "[kvstore-read] {} block(s) in {us:.1} µs ({:.1} µs/block, {} layers each)",
                out.len(),
                us / n as f64,
                self.n_layers,
            );
        }
        out
    }

    /// Retrieval read telemetry: `(calls, blocks_read, total_read_nanos)`. The
    /// mmap page-fault cost hides here — compare a cold first query against warm
    /// repeats (µs/block should drop once the pages are resident) to judge whether
    /// the disk tier or the compute is the retrieval bottleneck. See the
    /// `RLX_KVSTORE_READ_STATS` env var for a per-call print.
    pub fn read_stats(&self) -> (u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.read_calls.load(Relaxed),
            self.read_blocks_ct.load(Relaxed),
            self.read_nanos.load(Relaxed),
        )
    }

    /// Retrieve the `k` most query-relevant blocks (HNSW top-k over centroids),
    /// reading + dequantizing only those blocks' rows from the mmap tier.
    pub fn retrieve(&self, query: &[f32], k: usize) -> Vec<RetrievedBlock> {
        self.retrieve_filtered(query, k, |_| true)
    }

    /// Like [`retrieve`](Self::retrieve) but drops blocks whose [`Origin`] fails
    /// `keep` — e.g. `|o| o != Origin::Generated` to prefer sources over output.
    pub fn retrieve_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(Origin) -> bool,
    ) -> Vec<RetrievedBlock> {
        let hits: Vec<(u32, f32)> = self
            .block_hits(query, k * 2)
            .into_iter()
            .filter(|&(b, _)| self.blocks.get(b as usize).is_some_and(|x| keep(x.origin)))
            .take(k)
            .collect();
        self.read_blocks_batched(hits)
    }

    /// Exact **MaxSim** (late-interaction) score of a block against `query`:
    /// `max` over the block's layer-0 K rows of `query · K_row`. Unlike the
    /// mean/centroid HNSW key, this keeps a single strongly-matching token (e.g.
    /// the row carrying an exact fact) from being averaged away — the block scores
    /// as high as its BEST-matching position, which is what attention actually does.
    /// Always dot-based (the attention Q·K interpretation), independent of the
    /// HNSW navigation metric.
    fn block_maxsim(&self, block_id: u32, query: &[f32]) -> f32 {
        let br = match self.blocks.get(block_id as usize) {
            Some(b) => b,
            None => return f32::NEG_INFINITY,
        };
        let (k, _v) = match self
            .layers
            .first()
            .and_then(|l| l.read_rows(br.start_row, br.rows).ok())
        {
            Some(kv) => kv,
            None => return f32::NEG_INFINITY,
        };
        let d = self.kv_dim;
        let mut best = f32::NEG_INFINITY;
        for r in 0..br.rows {
            let row = &k[r * d..(r + 1) * d];
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += query[i] * row[i];
            }
            if dot > best {
                best = dot;
            }
        }
        best
    }

    /// **MaxSim retrieve-then-rerank**: over-fetch `k * overfetch` candidate blocks
    /// via the HNSW centroid index (cheap, sub-linear), then re-rank them by exact
    /// late-interaction [`block_maxsim`](Self::block_maxsim) and keep the top-k. This
    /// attacks the mean-pooling relevance ceiling: a block containing one exact-fact
    /// token that the mean key dilutes still ranks first if any of its rows matches
    /// the query strongly. `overfetch ≥ 1` (clamped); larger widens the re-rank pool.
    pub fn retrieve_maxsim(
        &self,
        query: &[f32],
        k: usize,
        overfetch: usize,
    ) -> Vec<RetrievedBlock> {
        let of = overfetch.max(1);
        let cand = self.block_hits(query, (k * of).max(k));
        let mut scored: Vec<(u32, f32)> = cand
            .into_iter()
            .map(|(b, _)| (b, self.block_maxsim(b, query)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        self.read_blocks_batched(scored)
    }

    /// **Exact** retrieval: brute-force MaxSim over EVERY block (no HNSW
    /// approximation), top-k. HNSW greedy navigation has poor recall on the
    /// degenerate distribution of raw post-RoPE K keys (they cluster), so for
    /// moderate block counts it silently misses the true nearest block — the
    /// in-RAM manager avoids this by scoring all blocks exactly (and recalls
    /// where the HNSW store did not). O(blocks · rows · dim): use below ~50k
    /// blocks; HNSW remains the path at million-block scale. Also the exact
    /// upper bound to measure HNSW recall against.
    pub fn retrieve_exact(&self, query: &[f32], k: usize) -> Vec<RetrievedBlock> {
        let mut scored: Vec<(u32, f32)> = (0..self.blocks.len() as u32)
            .map(|b| (b, self.block_maxsim(b, query)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        self.read_blocks_batched(scored)
    }

    /// All-layer MaxSim of a block: for each row, sum the per-layer dot
    /// `query_layers[l] · K_l[row]`, and take the max over rows. Layer-0-only
    /// scoring misses facts that the model attends to in MIDDLE layers; summing
    /// every layer's contribution is a much stronger K-space relevance signal
    /// (the same all-layer sum the Stage-3 importance proxy uses). `query_layers`
    /// must have `n_layers` entries of `kv_dim` each.
    fn block_maxsim_multilayer(&self, block_id: u32, query_layers: &[Vec<f32>]) -> f32 {
        let br = match self.blocks.get(block_id as usize) {
            Some(b) => b,
            None => return f32::NEG_INFINITY,
        };
        let d = self.kv_dim;
        // Read each layer's K rows once.
        let layer_k: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| {
                l.read_rows(br.start_row, br.rows)
                    .map(|(k, _)| k)
                    .unwrap_or_default()
            })
            .collect();
        let mut best = f32::NEG_INFINITY;
        for r in 0..br.rows {
            let mut s = 0.0f32;
            for (l, q) in query_layers.iter().enumerate() {
                let Some(k) = layer_k.get(l) else { continue };
                let base = r * d;
                if base + d > k.len() || q.len() < d {
                    continue;
                }
                for i in 0..d {
                    s += q[i] * k[base + i];
                }
            }
            if s > best {
                best = s;
            }
        }
        best
    }

    /// Exact all-layer MaxSim retrieval (brute-force over every block, all layers).
    /// The strongest K-space selection signal here: exact (no HNSW recall loss) +
    /// late-interaction (max row, no mean dilution) + all layers (catches
    /// middle-layer fact attention that layer-0 scoring misses). O(blocks · rows ·
    /// n_layers · dim) — moderate scale. `query_layers` = newest token's K per layer.
    pub fn retrieve_exact_multilayer(
        &self,
        query_layers: &[Vec<f32>],
        k: usize,
    ) -> Vec<RetrievedBlock> {
        let mut scored: Vec<(u32, f32)> = (0..self.blocks.len() as u32)
            .map(|b| (b, self.block_maxsim_multilayer(b, query_layers)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        self.read_blocks_batched(scored)
    }

    /// **Fuzzy** retrieval: top-k blocks whose relevance clears `min_score`.
    pub fn retrieve_fuzzy(&self, query: &[f32], k: usize, min_score: f32) -> Vec<RetrievedBlock> {
        let want = (k * self.centroids_per_block).max(k) * 2;
        let raw = self
            .hnsw
            .search_fuzzy(query, want, self.ef_search.max(want), min_score);
        let hits: Vec<(u32, f32)> = self.dedup_to_blocks(raw).into_iter().take(k).collect();
        self.read_blocks_batched(hits)
    }

    /// **Radius / range** retrieval: every block within similarity `threshold`.
    pub fn retrieve_radius(
        &self,
        query: &[f32],
        threshold: f32,
        max: usize,
    ) -> Vec<RetrievedBlock> {
        let want = (max * self.centroids_per_block).max(max) * 2;
        let raw = self
            .hnsw
            .search_radius(query, threshold, want, self.ef_search.max(want));
        let hits: Vec<(u32, f32)> = self.dedup_to_blocks(raw).into_iter().take(max).collect();
        self.read_blocks_batched(hits)
    }

    /// **Memory-decayed** retrieval: re-rank the candidate blocks by relevance ×
    /// recency (`decay^age`, age = accesses since the block was last retrieved),
    /// and mark the returned blocks as freshly used. Stale context fades so recent
    /// / frequently-used memory wins ties — Scissorhands-style forgetting for the
    /// unbounded store. `&mut` because it advances the access clock.
    pub fn retrieve_decayed(&mut self, query: &[f32], k: usize) -> Vec<RetrievedBlock> {
        let hits = self.block_hits(query, (k * 3).max(k));
        if hits.is_empty() {
            return Vec::new();
        }
        let min_s = hits.iter().map(|&(_, s)| s).fold(f32::INFINITY, f32::min);
        let mut scored: Vec<(u32, f32)> = hits
            .into_iter()
            .map(|(b, s)| {
                let age = self
                    .clock
                    .saturating_sub(*self.last_used.get(b as usize).unwrap_or(&0));
                let recency = self.decay.powi(age.min(i32::MAX as usize) as i32);
                // Shift to positive within the set so the multiplicative decay is
                // metric-/sign-agnostic (Dot, Cosine, negative-L2 all work).
                ((b), ((s - min_s) + 1e-3) * recency)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        self.clock += 1;
        for &(b, _) in &scored {
            if let Some(lu) = self.last_used.get_mut(b as usize) {
                *lu = self.clock;
            }
        }
        scored
            .into_iter()
            .filter_map(|(b, eff)| self.read_block(b, eff, false))
            .collect()
    }

    /// Retrieve the top-`k` relevant blocks **plus** up to `neighbors_per_hit`
    /// similar-neighbor blocks of each hit (from the HNSW centroid graph) — added
    /// semantic context beyond the exact matches, flagged `via_neighbor = true`.
    pub fn retrieve_expanded(
        &self,
        query: &[f32],
        k: usize,
        neighbors_per_hit: usize,
    ) -> Vec<RetrievedBlock> {
        let want = (k * self.centroids_per_block).max(k) * 2;
        let raw = self.hnsw.search(query, want, self.ef_search.max(want));
        let primary = self.dedup_to_blocks(raw.clone());
        let mut seen: std::collections::HashSet<u32> =
            primary.iter().take(k).map(|&(b, _)| b).collect();
        let mut out: Vec<RetrievedBlock> = primary
            .iter()
            .take(k)
            .filter_map(|&(b, s)| self.read_block(b, s, false))
            .collect();
        // Neighbor expansion over the HNSW graph (centroid ids → blocks).
        for &(cid, _) in raw.iter().take(k) {
            for &nb in self.hnsw.neighbors(cid).iter().take(neighbors_per_hit) {
                if let Some(&b) = self.centroid_block.get(nb as usize) {
                    if seen.insert(b) {
                        if let Some(rb) = self.read_block(b, f32::NAN, true) {
                            out.push(rb);
                        }
                    }
                }
            }
        }
        out
    }

    /// Blocks stored.
    pub fn len_blocks(&self) -> usize {
        self.blocks.len()
    }
    /// Total tokens held (across all blocks).
    pub fn total_tokens(&self) -> usize {
        self.total_rows
    }
    /// Approximate **RAM** footprint (keys + HNSW graph), i.e. what stays hot
    /// regardless of the on-disk data volume. Bytes.
    pub fn resident_index_bytes(&self) -> usize {
        // One key per centroid (kv_dim f32) plus HNSW links (~m0 u32 per node).
        self.centroid_block.len() * (self.kv_dim * 4 + 40 * 4)
    }
    /// Approximate **disk/mmap** footprint of the quantized K/V. Bytes.
    pub fn data_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.bytes()).sum()
    }
    /// Flush dirty pages to backing files (no-op for anonymous maps).
    pub fn flush(&self) -> Result<()> {
        for l in &self.layers {
            l.flush()?;
        }
        Ok(())
    }
}

/// Pick up to `k` **salient rows** as HNSW keys — the highest-L2-norm rows (the
/// most information-bearing tokens; near-zero rows carry little relevance signal).
/// No averaging, so an exact-fact row is inserted verbatim and stays navigable.
/// Deterministic (norm desc, row index asc tiebreak). Returns actual row copies.
fn salient_rows(rows: &[f32], n_rows: usize, dim: usize, k: usize) -> Vec<Vec<f32>> {
    if n_rows == 0 {
        return vec![vec![0.0; dim]];
    }
    let k = k.clamp(1, n_rows);
    let mut idx: Vec<(usize, f32)> = (0..n_rows)
        .map(|r| {
            let row = &rows[r * dim..(r + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum();
            (r, norm)
        })
        .collect();
    idx.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    idx.truncate(k);
    // Keep the selected rows in ascending row order for deterministic layout.
    idx.sort_by_key(|&(r, _)| r);
    idx.into_iter()
        .map(|(r, _)| rows[r * dim..(r + 1) * dim].to_vec())
        .collect()
}

/// Deterministic small k-means over a block's rows → `k` centroid keys. No RNG
/// (evenly-spaced row init) so an index rebuilt from the same blocks is
/// identical. `k` clamps to `[1, n_rows]`; empty clusters keep their centroid.
/// This is the "less averaging" that keeps each block's discriminative content
/// findable instead of collapsing it to a single mean.
fn kmeans_centroids(rows: &[f32], n_rows: usize, dim: usize, k: usize) -> Vec<Vec<f32>> {
    let k = k.clamp(1, n_rows.max(1));
    if n_rows == 0 {
        return vec![vec![0.0; dim]];
    }
    let mean = |ids: &[usize]| -> Vec<f32> {
        let mut c = vec![0.0f32; dim];
        for &r in ids {
            for j in 0..dim {
                c[j] += rows[r * dim + j];
            }
        }
        let inv = 1.0 / ids.len().max(1) as f32;
        for x in c.iter_mut() {
            *x *= inv;
        }
        c
    };
    if k == 1 {
        return vec![mean(&(0..n_rows).collect::<Vec<_>>())];
    }
    // Evenly-spaced row seeds.
    let mut cent: Vec<Vec<f32>> = (0..k)
        .map(|i| {
            let r = i * n_rows / k;
            rows[r * dim..(r + 1) * dim].to_vec()
        })
        .collect();
    let mut assign = vec![0usize; n_rows];
    for _ in 0..4 {
        for (r, a) in assign.iter_mut().enumerate() {
            let row = &rows[r * dim..(r + 1) * dim];
            let (mut best, mut bd) = (0usize, f32::INFINITY);
            for (ci, c) in cent.iter().enumerate() {
                let mut d = 0.0f32;
                for j in 0..dim {
                    let x = row[j] - c[j];
                    d += x * x;
                }
                if d < bd {
                    bd = d;
                    best = ci;
                }
            }
            *a = best;
        }
        for ci in 0..k {
            let ids: Vec<usize> = (0..n_rows).filter(|&r| assign[r] == ci).collect();
            if !ids.is_empty() {
                cent[ci] = mean(&ids);
            }
        }
    }
    cent
}

/// Streaming ingest in front of a [`KvContextStore`]. Push K/V **rows** as they
/// arrive — from a file being read, a query being typed, or the model generating
/// — and the streamer buffers them into `block`-sized, origin-homogeneous blocks
/// (keyed by mean-K) and appends them to the store on the fly. A change of origin
/// flushes the current partial block, so provenance stays clean per block.
///
/// This makes the store *live*: context can be injected mid-conversation and the
/// model's own streamed generation can be folded straight back in as
/// [`Origin::Generated`] context, immediately retrievable on the next step.
pub struct ContextStreamer {
    store: KvContextStore,
    block: usize,
    kv_dim: usize,
    n_layers: usize,
    buf_k: Vec<Vec<f32>>,
    buf_v: Vec<Vec<f32>>,
    buf_rows: usize,
    buf_origin: Option<(Origin, u32)>,
    next_pos: usize,
}

impl ContextStreamer {
    /// Wrap a store; `block` is the target block size (rows).
    pub fn new(store: KvContextStore, block: usize) -> Self {
        let (kv_dim, n_layers) = (store.kv_dim, store.n_layers);
        ContextStreamer {
            store,
            block: block.max(1),
            kv_dim,
            n_layers,
            buf_k: vec![Vec::new(); n_layers],
            buf_v: vec![Vec::new(); n_layers],
            buf_rows: 0,
            buf_origin: None,
            next_pos: 0,
        }
    }

    /// Push `n` rows (per-layer `n × kv_dim` K and V) tagged with `origin`.
    /// Blocks are emitted to the store as they fill.
    pub fn push(
        &mut self,
        origin: Origin,
        source_id: u32,
        k_rows: &[Vec<f32>],
        v_rows: &[Vec<f32>],
    ) -> Result<()> {
        // Origin change → flush the partial block so blocks stay homogeneous.
        if let Some(cur) = self.buf_origin {
            if cur != (origin, source_id) {
                self.flush()?;
            }
        }
        self.buf_origin = Some((origin, source_id));
        for l in 0..self.n_layers {
            self.buf_k[l].extend_from_slice(&k_rows[l]);
            self.buf_v[l].extend_from_slice(&v_rows[l]);
        }
        self.buf_rows += k_rows.first().map(|r| r.len() / self.kv_dim).unwrap_or(0);
        while self.buf_rows >= self.block {
            self.emit(self.block)?;
        }
        Ok(())
    }

    /// Emit `rows` from the front of the buffer as one block.
    fn emit(&mut self, rows: usize) -> Result<()> {
        let take = rows.min(self.buf_rows);
        if take == 0 {
            return Ok(());
        }
        let cut = take * self.kv_dim;
        let mut k = Vec::with_capacity(self.n_layers);
        let mut v = Vec::with_capacity(self.n_layers);
        for l in 0..self.n_layers {
            k.push(self.buf_k[l][..cut].to_vec());
            v.push(self.buf_v[l][..cut].to_vec());
            self.buf_k[l].drain(..cut);
            self.buf_v[l].drain(..cut);
        }
        self.buf_rows -= take;
        let key = {
            let mut key = vec![0.0f32; self.kv_dim];
            for r in 0..take {
                for j in 0..self.kv_dim {
                    key[j] += k[0][r * self.kv_dim + j];
                }
            }
            for x in key.iter_mut() {
                *x /= take as f32;
            }
            key
        };
        let (origin, source_id) = self.buf_origin.unwrap_or((Origin::Other(0), 0));
        let pos = self.next_pos;
        self.next_pos += take;
        self.store
            .append_block(pos, origin, source_id, &k, &v, &key)?;
        Ok(())
    }

    /// Flush any buffered partial block to the store.
    pub fn flush(&mut self) -> Result<()> {
        while self.buf_rows > 0 {
            self.emit(self.block)?;
        }
        Ok(())
    }

    /// Borrow the underlying store (e.g. to `retrieve`).
    pub fn store(&self) -> &KvContextStore {
        &self.store
    }
    /// Mutably borrow / take the store back.
    pub fn store_mut(&mut self) -> &mut KvContextStore {
        &mut self.store
    }
    pub fn into_store(mut self) -> Result<KvContextStore> {
        self.flush()?;
        Ok(self.store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_key(k0: &[f32], kv_dim: usize, rows: usize) -> Vec<f32> {
        let mut key = vec![0.0f32; kv_dim];
        for r in 0..rows {
            for j in 0..kv_dim {
                key[j] += k0[r * kv_dim + j];
            }
        }
        for x in key.iter_mut() {
            *x /= rows.max(1) as f32;
        }
        key
    }

    #[test]
    fn append_and_retrieve_roundtrips_through_disk() {
        let n_layers = 2;
        let kv_dim = 32; // multiple of Q8_0 block (32)
        let rows = 4;
        let mut store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::Q8_0,
            256,
            None, // anonymous mmap
            HnswConfig::default(),
            64,
            1,   // centroids/block (mean mode → uses passed key)
            1.0, // no decay
        )
        .unwrap();

        // 50 blocks; block b's key points mostly along axis (b % kv_dim).
        for b in 0..50usize {
            let axis = b % kv_dim;
            let k: Vec<Vec<f32>> = (0..n_layers)
                .map(|_| {
                    let mut buf = vec![0.05f32; rows * kv_dim];
                    for r in 0..rows {
                        buf[r * kv_dim + axis] = 1.0;
                    }
                    buf
                })
                .collect();
            let v: Vec<Vec<f32>> = (0..n_layers)
                .map(|_| vec![b as f32 * 0.01; rows * kv_dim])
                .collect();
            let key = mean_key(&k[0], kv_dim, rows);
            // Alternate origins to exercise provenance tracking.
            let origin = if b % 2 == 0 {
                Origin::File
            } else {
                Origin::Generated
            };
            store
                .append_block(b * rows, origin, b as u32, &k, &v, &key)
                .unwrap();
        }
        assert_eq!(store.len_blocks(), 50);
        assert_eq!(store.total_tokens(), 50 * rows);

        // Query aligned with axis 6 → block 6 (even → File origin, source_id 6).
        let mut q = vec![0.0f32; kv_dim];
        q[6] = 1.0;
        let got = store.retrieve(&q, 3);
        assert!(!got.is_empty());
        assert_eq!(got[0].start_pos, 6 * rows);
        assert_eq!(got[0].origin, Origin::File);
        assert_eq!(got[0].source_id, 6);
        assert!(!got[0].via_neighbor);
        // Data round-trips through Q8_0: layer count + shape preserved.
        assert_eq!(got[0].k.len(), n_layers);
        assert_eq!(got[0].k[0].len(), rows * kv_dim);
        let vmean = got[0].v[0].iter().sum::<f32>() / got[0].v[0].len() as f32;
        assert!((vmean - 0.06).abs() < 0.02, "V round-trip off: {vmean}");

        // Origin filter: exclude Generated → only File blocks come back.
        let only_files = store.retrieve_filtered(&q, 5, |o| o != Origin::Generated);
        assert!(only_files.iter().all(|r| r.origin == Origin::File));

        // Neighbor expansion adds similar blocks beyond the exact top-k.
        let base = store.retrieve(&q, 2).len();
        let expanded = store.retrieve_expanded(&q, 2, 4);
        assert!(
            expanded.len() >= base,
            "expansion should not shrink the set"
        );
        assert!(expanded.iter().any(|r| r.via_neighbor) || expanded.len() == store.len_blocks());
    }

    #[test]
    fn maxsim_beats_mean_pool_for_single_row_fact() {
        // The relevance ceiling: a block whose MEAN key is diluted still contains
        // one row that matches the query strongly. Mean-pool retrieval ranks a
        // block with a higher average above it; MaxSim (max over rows) surfaces the
        // block with the best single-row match — what attention actually rewards.
        let (n_layers, kv_dim, rows) = (1, 32, 4);
        let mut store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::F16, // near-lossless so the row dot is faithful
            256,
            None,
            HnswConfig::default(),
            64,
            1, // one mean key/block (so HNSW nav uses the diluted mean)
            1.0,
        )
        .unwrap();

        let axis = 6;
        // Filler blocks along other axes so retrieval has a realistic pool.
        for b in 0..20usize {
            let a = (b % kv_dim).max(1); // avoid the query axis
            let a = if a == axis { a + 1 } else { a };
            let k = vec![{
                let mut buf = vec![0.02f32; rows * kv_dim];
                for r in 0..rows {
                    buf[r * kv_dim + a] = 0.4;
                }
                buf
            }];
            let key = mean_key(&k[0], kv_dim, rows);
            store
                .append_block(b * rows, Origin::File, b as u32, &k, &k, &key)
                .unwrap();
        }
        // DISTRACTOR: every row 0.5 along the query axis → mean 0.5, MaxSim 0.5.
        let dist = vec![{
            let mut buf = vec![0.02f32; rows * kv_dim];
            for r in 0..rows {
                buf[r * kv_dim + axis] = 0.5;
            }
            buf
        }];
        let dist_key = mean_key(&dist[0], kv_dim, rows);
        let dist_pos = 100 * rows;
        store
            .append_block(dist_pos, Origin::File, 100, &dist, &dist, &dist_key)
            .unwrap();
        // FACT: one row 1.0 along the query axis, rest ~0 → mean ≈0.25, MaxSim 1.0.
        let fact = vec![{
            let mut buf = vec![0.02f32; rows * kv_dim];
            buf[axis] = 1.0; // only row 0 carries the fact
            buf
        }];
        let fact_key = mean_key(&fact[0], kv_dim, rows);
        let fact_pos = 200 * rows;
        store
            .append_block(fact_pos, Origin::File, 200, &fact, &fact, &fact_key)
            .unwrap();

        let mut q = vec![0.0f32; kv_dim];
        q[axis] = 1.0;

        // Mean-pool retrieval ranks the distractor first (higher average).
        let mean_top = store.retrieve(&q, 1);
        assert_eq!(
            mean_top[0].start_pos, dist_pos,
            "mean-pool should favor the higher-average block"
        );
        // MaxSim re-ranking surfaces the fact block (best single-row match).
        let ms = store.retrieve_maxsim(&q, 2, 8);
        assert_eq!(
            ms[0].start_pos, fact_pos,
            "MaxSim should surface the single-row fact block first"
        );
    }

    #[test]
    fn embedding_index_retrieves_and_hybrid_blends() {
        let (n_layers, kv_dim, rows, edim) = (1, 32, 4, 8);
        let mut store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::F16,
            512,
            None,
            HnswConfig::default(),
            64,
            1,
            1.0,
        )
        .unwrap();
        store.enable_embeddings(edim, HnswConfig::default());
        // 12 blocks; block b gets a distinct one-hot-ish embedding along axis b%edim
        // and lexical token = 1000+b. K rows are uninformative (all ~equal) so ONLY
        // the embedding (or lexical) can discriminate — exactly the K·K-fails case.
        for b in 0..12usize {
            let k = vec![vec![0.1f32; rows * kv_dim]];
            let key = mean_key(&k[0], kv_dim, rows);
            let id = store
                .append_block(b * rows, Origin::File, b as u32, &k, &k, &key)
                .unwrap();
            let mut emb = vec![0.0f32; edim];
            emb[b % edim] = 1.0;
            store.append_embed(id, &emb);
            store.attach_tokens(id, &[1000 + b as u32]);
        }
        // Query embedding along axis 3 → block 3 (and block 11, also axis 3).
        let mut q = vec![0.0f32; edim];
        q[3] = 1.0;
        let got = store.retrieve_embed(&q, 1);
        assert_eq!(
            got[0].start_pos % (edim * rows),
            3 * rows,
            "embed retrieval hits the axis-3 block"
        );
        // Hybrid: embedding axis-7 + lexical token 1005 → should surface both 7 and 5.
        let mut qe = vec![0.0f32; edim];
        qe[7] = 1.0;
        let h = store.retrieve_hybrid3(&qe, &[], &[1005], 4, 1.0, 1.0, 0.0, 0.0);
        let starts: Vec<usize> = h.iter().map(|r| r.start_pos / rows).collect();
        assert!(
            starts.contains(&7),
            "embedding term surfaces block 7: {starts:?}"
        );
        assert!(
            starts.contains(&5),
            "lexical term surfaces block 5: {starts:?}"
        );
    }

    #[test]
    fn row_keys_index_navigates_to_fact_without_rerank() {
        // Index-side of MaxSim: with KeyMode::RowKeys the block's salient rows are
        // HNSW keys, so plain retrieve (no re-rank) already navigates to the fact
        // block via its strong row — the diluted mean never had to be trusted.
        let (n_layers, kv_dim, rows) = (1, 32, 4);
        let mut store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::F16,
            256,
            None,
            HnswConfig::default(),
            64,
            4, // key budget per block (row-keys use up to this many salient rows)
            1.0,
        )
        .unwrap();
        store.set_key_mode(KeyMode::RowKeys);

        let axis = 6;
        for b in 0..20usize {
            let a = {
                let a = (b % kv_dim).max(1);
                if a == axis { a + 1 } else { a }
            };
            let k = vec![{
                let mut buf = vec![0.02f32; rows * kv_dim];
                for r in 0..rows {
                    buf[r * kv_dim + a] = 0.4;
                }
                buf
            }];
            let key = mean_key(&k[0], kv_dim, rows);
            store
                .append_block(b * rows, Origin::File, b as u32, &k, &k, &key)
                .unwrap();
        }
        // Distractor: uniform 0.5 along axis (mean 0.5, best row 0.5).
        let dist = vec![{
            let mut buf = vec![0.02f32; rows * kv_dim];
            for r in 0..rows {
                buf[r * kv_dim + axis] = 0.5;
            }
            buf
        }];
        let dist_key = mean_key(&dist[0], kv_dim, rows);
        store
            .append_block(100 * rows, Origin::File, 100, &dist, &dist, &dist_key)
            .unwrap();
        // Fact: one row 1.0 along axis (mean ≈0.25, best row 1.0).
        let fact = vec![{
            let mut buf = vec![0.02f32; rows * kv_dim];
            buf[axis] = 1.0;
            buf
        }];
        let fact_key = mean_key(&fact[0], kv_dim, rows);
        let fact_pos = 200 * rows;
        store
            .append_block(fact_pos, Origin::File, 200, &fact, &fact, &fact_key)
            .unwrap();

        let mut q = vec![0.0f32; kv_dim];
        q[axis] = 1.0;
        // Plain retrieve (mean-diluted key would favor the distractor) now returns
        // the fact first because its 1.0-row is an HNSW key.
        let got = store.retrieve(&q, 1);
        assert_eq!(
            got[0].start_pos, fact_pos,
            "RowKeys index should navigate to the fact block"
        );
    }

    #[test]
    fn ram_index_bounded_vs_disk_data() {
        // The whole point: RAM (keys+graph) grows with block COUNT, while the
        // K/V data (tens of GB at scale) lives on the mmap tier. With realistic
        // multi-row blocks the shared per-block key is amortized, so data ≫ index.
        let (n_layers, kv_dim, block_rows) = (28, 64, 64);
        let mut store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::Q4_0,
            300 * block_rows,
            None,
            HnswConfig::default(),
            64,
            2,   // k-means centroids/block
            1.0, // no decay
        )
        .unwrap();
        for b in 0..200usize {
            let k: Vec<Vec<f32>> = (0..n_layers)
                .map(|_| vec![(b % 7) as f32 * 0.1; block_rows * kv_dim])
                .collect();
            let v = k.clone();
            let key = vec![(b % 7) as f32 * 0.1; kv_dim];
            store
                .append_block(b * block_rows, Origin::File, 0, &k, &v, &key)
                .unwrap();
        }
        let idx = store.resident_index_bytes();
        let data = store.data_bytes();
        // Data dwarfs the RAM index (the disk tier holds the bulk).
        assert!(data > idx * 5, "data {data} should ≫ index {idx}");
        assert_eq!(store.total_tokens(), 200 * block_rows);
    }

    #[test]
    fn multi_centroid_retrievable_by_either_cluster() {
        // A block whose rows form TWO clusters (axis 3 and axis 25). With k-means
        // centroids (not one mean), the block is retrievable by EITHER cluster's
        // direction — the discriminative content isn't averaged away.
        let (kv_dim, rows) = (32usize, 16usize);
        let cfg = HnswConfig {
            metric: crate::hnsw::Metric::Dot,
            ..Default::default()
        };
        let mut store =
            KvContextStore::new(1, kv_dim, KvQuant::F16, 1024, None, cfg, 64, 2, 1.0).unwrap();
        // Distractor blocks on other axes.
        for b in 0..8usize {
            let a = 5 + b;
            let mut row = vec![0.0f32; rows * kv_dim];
            for r in 0..rows {
                row[r * kv_dim + a] = 1.0;
            }
            store
                .append_block(
                    b * rows,
                    Origin::File,
                    0,
                    &[row.clone()],
                    &[row],
                    &vec![0.0; kv_dim],
                )
                .unwrap();
        }
        // Target: half the rows on axis 3, half on axis 25.
        let mut row = vec![0.0f32; rows * kv_dim];
        for r in 0..rows / 2 {
            row[r * kv_dim + 3] = 1.0;
        }
        for r in rows / 2..rows {
            row[r * kv_dim + 25] = 1.0;
        }
        let tgt_pos = 8 * rows;
        store
            .append_block(
                tgt_pos,
                Origin::File,
                0,
                &[row.clone()],
                &[row],
                &vec![0.0; kv_dim],
            )
            .unwrap();
        // Queried on either cluster axis, the target block comes back.
        let mut q3 = vec![0.0f32; kv_dim];
        q3[3] = 1.0;
        assert_eq!(
            store.retrieve(&q3, 1)[0].start_pos,
            tgt_pos,
            "axis-3 cluster"
        );
        let mut q25 = vec![0.0f32; kv_dim];
        q25[25] = 1.0;
        assert_eq!(
            store.retrieve(&q25, 1)[0].start_pos,
            tgt_pos,
            "axis-25 cluster"
        );
    }

    #[test]
    fn decayed_retrieval_runs_and_advances_clock() {
        let (kv_dim, rows) = (32usize, 8usize);
        let cfg = HnswConfig {
            metric: crate::hnsw::Metric::Dot,
            ..Default::default()
        };
        let mut store =
            KvContextStore::new(1, kv_dim, KvQuant::F16, 1024, None, cfg, 64, 2, 0.9).unwrap();
        for b in 0..12usize {
            let a = b % kv_dim;
            let mut row = vec![0.0f32; rows * kv_dim];
            for r in 0..rows {
                row[r * kv_dim + a] = 1.0;
            }
            store
                .append_block(
                    b * rows,
                    Origin::File,
                    0,
                    &[row.clone()],
                    &[row],
                    &vec![0.0; kv_dim],
                )
                .unwrap();
        }
        let mut q = vec![0.0f32; kv_dim];
        q[4] = 1.0;
        let got = store.retrieve_decayed(&q, 3);
        assert!(!got.is_empty());
        // Block 4 (axis 4) is the strongest match.
        assert_eq!(got[0].start_pos, 4 * rows);
        // Determinism: same query, same top block.
        let got2 = store.retrieve_decayed(&q, 3);
        assert_eq!(got2[0].start_pos, 4 * rows);
    }

    #[test]
    fn hybrid_lexical_rescues_exact_token() {
        // A fact block carries a distinctive token (7731) that dense K·K misses.
        // Lexical (inverted index + IDF) surfaces it via that token.
        let (kv_dim, rows) = (32usize, 8usize);
        let mut store = KvContextStore::new(
            1,
            kv_dim,
            KvQuant::F16,
            1024,
            None,
            HnswConfig::default(),
            64,
            1,
            1.0,
        )
        .unwrap();
        for b in 0..20usize {
            let mut row = vec![0.0f32; rows * kv_dim];
            for r in 0..rows {
                row[r * kv_dim + (b % kv_dim)] = 0.3;
            }
            let key = mean_key(&row, kv_dim, rows);
            let id = store
                .append_block(
                    b * rows,
                    Origin::File,
                    b as u32,
                    &[row.clone()],
                    &[row],
                    &key,
                )
                .unwrap();
            // Common tokens everywhere; the unique fact token only in block 13.
            if b == 13 {
                store.attach_tokens(id, &[100, 200, 7731]);
            } else {
                store.attach_tokens(id, &[100, 200]);
            }
        }
        // Dense query points at block 5 (axis 5), NOT block 13.
        let mut q = vec![0.0f32; kv_dim];
        q[5] = 1.0;
        // Hybrid with the fact token in the lexical query → block 13 surfaces.
        let hybrid = store.retrieve_hybrid(&q, &[7731], 3, 0.8);
        assert!(
            hybrid.iter().any(|r| r.start_pos == 13 * rows),
            "lexical should surface the exact-token block 13"
        );
        // Pure dense (lexical_weight 0) with the same token query does NOT.
        let dense = store.retrieve_hybrid(&q, &[7731], 3, 0.0);
        assert!(
            dense.iter().all(|r| r.start_pos != 13 * rows),
            "dense alone shouldn't find 13"
        );
    }

    #[test]
    fn streaming_ingest_forms_blocks_and_tags_origin() {
        let (n_layers, kv_dim, block) = (2, 32, 8);
        let store = KvContextStore::new(
            n_layers,
            kv_dim,
            KvQuant::Q8_0,
            256,
            None,
            HnswConfig::default(),
            64,
            1,
            1.0,
        )
        .unwrap();
        let mut streamer = ContextStreamer::new(store, block);
        // Stream 20 File rows in ragged pushes (3,5,7,5), then 6 Generated rows.
        let mk = |axis: usize, n: usize| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let mut row = vec![0.0f32; n * kv_dim];
            for r in 0..n {
                row[r * kv_dim + axis] = 1.0;
            }
            let k: Vec<Vec<f32>> = (0..2).map(|_| row.clone()).collect();
            (k.clone(), k)
        };
        for &n in &[3usize, 5, 7, 5] {
            let (k, v) = mk(3, n);
            streamer.push(Origin::File, 1, &k, &v).unwrap();
        }
        let (k, v) = mk(9, 6);
        streamer.push(Origin::Generated, 2, &k, &v).unwrap();
        streamer.flush().unwrap();
        // 20 File rows → blocks (8+8) + partial 4 flushed at origin change; 6 Gen.
        let store = streamer.into_store().unwrap();
        assert_eq!(store.total_tokens(), 26);
        // Query axis 3 → a File block; axis 9 → the Generated block.
        let mut q = vec![0.0f32; kv_dim];
        q[3] = 1.0;
        let got = store.retrieve(&q, 1);
        assert_eq!(got[0].origin, Origin::File);
        let mut q2 = vec![0.0f32; kv_dim];
        q2[9] = 1.0;
        let got2 = store.retrieve(&q2, 1);
        assert_eq!(got2[0].origin, Origin::Generated);
        assert_eq!(got2[0].source_id, 2);
    }
}
