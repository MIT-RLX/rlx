# Selective KV Retention & Retrieval

Extending effective context beyond a fixed resident budget, without the per-step
cost growing with total context — and without the "amnesia" of a naive sliding
window.

## Why

Autoregressive decode attends, at every step, over the **entire** KV cache. That
read is `O(context)`:

```
KV bytes / step = context × kv_dim × n_layers × 2(K+V) × dtype_bytes
```

For qwen3-0.6B (`kv_dim = 1024`, 28 layers): ~0.23 MB/token, so at 8k tokens the
KV read (~1.9 GB f32) *overtakes* the constant ~1.1 GB weight read — decode
latency grows with context (measured: ~12 ms/token at short context → ~50 ms at
2k). See [`metal-qwen3-decode-perf.md`](metal-qwen3-decode-perf.md).

A **sliding window** bounds the cost but is *dementia*: it drops old tokens by
recency alone, so a fact stated early — or the task instruction itself — is gone.

**Selective retention** keeps a bounded set of the *useful* positions and
**retrieves** the rest on demand, so effective context is unbounded while
per-step attention stays `O(budget)`.

## Architecture — a model-agnostic core seam

`rlx_runtime::kv_retention` owns the **decisions and the evicted data**; the
model/backend owns the live resident K/V tensors and applies each plan. K/V are
flat `f32` rows of width `kv_dim`, so any model that decodes through a KV cache
inherits it — nothing here is qwen-specific.

```
                        KvRetentionManager  (rlx-runtime, core)
                        ┌───────────────────────────────────────┐
   per-step importance→ │ resident metadata: abs_pos, attn_mass, │
   signal ────────────→ │   recency, block id                    │
                        │ block store: evicted K/V + retrieval   │
                        │   key (mean K), by block id            │
   query summary ─────→ │ plan(query) → RetentionPlan            │
                        └───────────────────────────────────────┘
                                        │  keep / evict / retrieve
                                        ▼
                   model backend (e.g. Qwen3Generator::step_cached)
                   reshapes its resident K/V + splices retrieved blocks
```

The manager never touches device memory or a specific tensor layout — it emits a
`RetentionPlan { keep, evict, retrieve, store_evicted }` of **indices**, and the
caller materializes it against its own K/V.

## Policies

Configured per sequence. `sinks` are the first-N absolute positions (StreamingLLM
"attention sinks" — models dump surplus attention there; keeping them preserves
calibration). `recent` is the last-N positions.

| Policy | Keeps | Evicts | Retrieves | Needs |
|---|---|---|---|---|
| **`Full`** | everything | — | — | — (exact `O(context)` baseline) |
| **`Sinks{sinks, window}`** | first `sinks` + last `window` | the middle | — | — (recency-only; the "amnesia" baseline) |
| **`HeavyHitter{sinks, recent, budget}`** | sinks + recent + top-`budget` middle by **attention mass** | low-attention middle | — | an importance signal |
| **`Retrieval{block, resident_blocks, sinks, recent}`** | sinks + recent + top-`resident_blocks` **query-relevant** blocks | cold blocks → store | most relevant blocks back | a query summary |
| **`Auto{max_resident}`** | picks below | — | — | fits within `max_resident` |

**`Auto`** resolves each step from context length + observed attention
*concentration* (peak / total attention mass, smoothed):

- context ≤ `max_resident` → `Full` (no eviction needed).
- long **and** concentrated (a few dominant keys) → `HeavyHitter` — heavy-hitters
  capture the attention; drop the rest.
- long **and** diffuse (broad recall) → `Retrieval` — old context can be pulled
  back when a later query needs it.

## Importance signals

`observe_attention(weights)` (one entry per resident position, aggregated across
heads/layers by the caller) updates, per position:

- **attention mass** — cumulative attention received, **decayed** each step
  (`MASS_DECAY = 0.98`) so a position that stops being attended slowly loses
  heavy-hitter status (Scissorhands-style forgetting).
- **recency** — last step it was meaningfully attended.
- **concentration** — a rolling `peak/total` estimate that drives `Auto`.

The caller supplies this signal each step. `Qwen3Generator` computes, host-side
from the KV mirror it already holds, `importance[i] = newest-token-K · resident-K[i]`
summed over layers — a **K-similarity proxy** for the model's Q·K attention (the
same signal the retrieval path scores blocks with, and which produces coherent
retrieval in practice). It is gated on `manager.needs_attention()`, so
position/query-only policies (`Sinks`, `Retrieval`) pay nothing for it.

The *exact* per-position softmax weights live inside the online-softmax
`sdpa_decode_m1*` kernels; exporting them means a per-step weight readback plus
either a K re-read or heavy threadgroup scratch — expensive for decode, and the
K-similarity proxy already ranks positions well enough to drive `HeavyHitter`/
`Auto`. The exact-weight export remains a possible refinement, not a prerequisite.

## Retrieval (block store)

Cold positions are grouped into `block`-sized blocks and offloaded to the store,
each keyed by a **block summary** (mean K over its rows). Each step, stored blocks
are scored by **`query · key`** and the top-`resident_blocks` are pulled back into
the resident set. `take_block(id)` returns the block's K/V; `push_evicted_block`
stores one. Evicted↔resident is a cycle (no data loss): a block re-evicted next
step is re-stored, a still-relevant block stays resident.

This is what makes context effectively unbounded: the store holds *all* history,
resident attention stays `O(budget)`, and relevance — not recency — decides what
the model sees.

## Decode-loop integration

In `Qwen3Generator` (`rlx-qwen3`):

- **Seed:** `on_prefill(seq)` registers the prompt's positions so the resident
  metadata tracks the KV mirror row-for-row.
- **Per step** (`step_cached`, after the KV update): `apply_retention()` calls
  `append()` → `plan(query)` → offloads evicted rows to the store as per-layer
  blocks → retrieves the top-k query-relevant blocks → rebuilds `cache.layers_k/v`
  as `kept + retrieved` rows in absolute-position order, sets `past_len`,
  `commit()`s, and resets the GPU-resident binding so the next step rebinds from
  the (now `O(budget)`) trimmed mirror.

### Why RoPE stays correct

The kept/retrieved positions are generally **non-contiguous** (the middle is
gone). That's fine: **K is stored *post*-rotation**, so the query at absolute
position `p_q` and a kept key at its original `p_k` dot-product to the *relative*
rotation `p_q − p_k` — exactly as if the evicted positions were still there. The
decode mask marks every resident position valid (all are past), so ordering is
irrelevant to correctness; the manager keeps resident sorted by `abs_pos` only so
"recent" means genuinely recent for the next eviction.

## Configuration

Opt-in (default is `Full` — keep everything):

- **Env** (qwen3): `RLX_QWEN3_RETENTION=` one of
  `sinks:S:W` · `heavy:S:R:B` · `retrieval:BLK:RB:S:R` · `auto:MAX`.
  Example: `RLX_QWEN3_RETENTION=sinks:4:32` (4 sinks + 32 recent).
- **Builder:** `Qwen3Generator::with_retention(KvRetentionPolicy::…)`.

## Measurement & inspection tools

Reusable, model-agnostic recorders (they live in rlx core, so any model that
decodes through a KV cache inherits them):

- **Cache/context telemetry** — `rlx_runtime::kv_metrics::RetentionRecorder`.
  `KvRetentionManager::enable_recording()` snapshots every `commit`
  (resident / evicted / retrieved / store) and rolls up a `RetentionSummary`:
  resident percentiles, **effective context** (`resident + store`), the
  **context-extension factor** (`effective_context_max / resident_max`),
  eviction/retrieval activity, optional decode latency. `to_csv()` for plotting.
- **Data inspection** — `rlx_ir::tensor_inspect`. `TensorStats` (shape + min/max/
  mean/std/absmax + nan/inf + a value `Histogram` with a sparkline) and
  `InspectLog` (named streams over a shared `step` axis + dataflow edges;
  `to_csv` / `to_hist_csv` / `dataflow_dot` / `report`). The Qwen3 generator
  records `kv.k` / `kv.v` / `selection.importance` / `selection.attn_mass` /
  `selection.concentration` each step (`enable_inspect()`), so *what the cache
  holds* and *why positions are kept* are analyzable together.
- **Op inspection** — two complementary views:
  - *Structure* (compiled graph): `rlx-opscope`'s static analysis on the built
    graph — per-op shape (`m·k·n`) / FLOPs / bytes / arithmetic intensity /
    roofline class (`op_costs`) and recurring dataflow cones
    (`repeated_flow_patterns`). No execution needed.
  - *Values* (eager run): the CPU **reference** executor (`execute`) feeds the
    same `InspectLog` schema through a process-global tap when `RLX_INSPECT_OPS=1`
    (`RLX_INSPECT_BINS` sets bins): each f32 op output's shape/stats/histogram +
    input→output dataflow, drained with `rlx_ir::tensor_inspect::op_tap_take()`.
    (The compiled thunk executor uses opaque per-op closures; a value tap there
    is a follow-up — use opscope's stat-injection for compiled-path values.)

**The harness:** `rlx-qwen3/examples/memory_probe.rs` — a multi-shot
memory/context-retrieval test. It plants facts, buries them under filler turns
past the resident budget, then asks about them, scoring recall per policy while
recording all three streams above + per-turn throughput, and writes CSV/DOT under
`--out`. A deterministic **greedy top-k** sampler (`--temp 0` ⇒ argmax) keeps the
policy comparison apples-to-apples. Always include a `full` baseline: it is the
upper bound (`3/3` on qwen3-0.6B), so a gap to `full` localizes a miss to
retention/relevance rather than the model or sampler.

## Status

- **Stage 1 — core seam** ✅ `KvRetentionManager` + all policies + block store +
  `Auto` selection + importance scoring, model-agnostic, unit-tested
  (`kv_retention::tests`).
- **Stage 2 — decode integration** ✅ wired into `step_cached`; eviction reshapes
  the resident K/V; RoPE verified correct (coherent multi-turn output under
  `Sinks`; bounded KV). Weight-free policies run today.
- **Stage 2b — retrieval splice** ✅ evicted rows offload to per-layer blocks in
  the store; the top-k query-relevant blocks are pulled back and spliced into the
  resident K/V each step (rebuilt in abs-pos order). Verified: the evict↔retrieve
  cycle runs with bounded resident + coherent output (`RLX_QWEN3_RETENTION_DEBUG=1`
  shows per-step evicted/retrieved/store counts).
- **Stage 3 — importance signal wired** ✅ `Qwen3Generator` feeds
  `observe_attention` a per-step, per-position importance (newest-token-K ·
  resident-K, summed over layers; gated on `manager.needs_attention()`), so
  `HeavyHitter`/`Auto` rank by relevance, not recency. Verified: `HeavyHitter`
  bounds resident to `sinks + recent + budget` and evicts by attention mass;
  `Auto` resolves `Full`/`HeavyHitter`/`Retrieval` from context + concentration
  and stays within `max_resident` — both coherent multi-turn.
  *Refinement (not required):* exact per-position softmax weights from the
  `sdpa_decode_m1*` kernels would replace the K-similarity proxy at the cost of a
  per-step weight readback + K re-read.

## Optimization results (qwen3-0.6B, M4 Pro, greedy; from `memory_probe`)

Data-driven, measured before/after with the probe's recorders:

- **Incremental retrieval** ❌ *Attempted, then fully reverted.* Rank candidates
  (resident ∪ store) by relevance and move only the marginal blocks (keep top-k in
  place; stable position-aligned block ids; merge partial pushes). It cut host
  eviction **churn −91%** (`retrieval:8:24:4:32` 198→18 rows/step) — but the probe
  showed it **regressed retrieval recall 3/3 → 0/3** (a partial-push data-loss bug
  fixed it to 1/3, still short). Root cause, confirmed by isolation runs: the
  evict-all/retrieve-all path **re-chunks blocks every step**, and that churn is
  *load-bearing* — it makes block boundaries adaptive and query-responsive.
  Fixed position-aligned blocks (required to keep blocks stable across steps) are
  coarser and dilute a buried fact's mean-K key, so even the non-incremental
  variant with position-blocks fell to 0/3. Recall is the system's whole point, so
  the default is restored to the re-chunking path (3/3). Two lessons: the
  "wasteful" churn was doing real work, and a bounded budget with +1 token/step
  must evict ≥1 row/step so no step is ever a true no-op — cutting the per-step
  GPU rebind needs incremental *rebind* (upload only changed rows) or watermarked
  batch eviction, not cheaper *selection*. Deferred.
- **f16-resident weights** (decode default, `--no-f16` opts out) ✅ 71% of decode
  bytes are weights; f16 halves them → **decode ~13–16 → ~30–47 tok/s (≈2.5–3×)**,
  **recall unchanged: `full` 3/3 and `retrieval` 3/3**. (An earlier "f16 hurts
  retrieval" reading was a misdiagnosis — the culprit was the incremental-retrieval
  bug below; with that reverted, f16 is clean on both paths.)
- **Mixed-precision KV** (`--kv-quant`: K→f16, V→int8 per-tensor) ✅ measured V
  range ±2.5 (int8-safe), K outliers 24×std (kept f16); **KV −62%** (96→36 MB @
  420 tok) with `full` recall preserved. Realized on the host mirror; the native
  int8-V attention kernel (GPU traffic savings) is the follow-up.
- **Cosine-normalized importance** ❌ *Reverted.* The re-bench disproved the
  hypothesis: cosine **lowered** the importance contrast (CV 0.127→0.116) and the
  block ranking is invariant to the DC/scale transforms available, so it can't
  sharpen *selection* — only a better signal (exact attention weights, Stage-3+)
  can. Raw dot restored; kept as a recorded negative result.

## Scaling to a million tokens — the tiered store

A million tokens of KV is far too big for RAM/VRAM (qwen3-0.6B: ~224 GB at f32).
Reaching it is an **IO-arrangement** problem, solved by two composable pieces:

- **Sub-linear selection** — `rlx_runtime::hnsw::Hnsw`. Each block contributes a
  tiny *key* (mean K, `kv_dim` floats) to an in-RAM HNSW graph; a query finds the
  top-k relevant blocks in `O(log N)` (≈14 hops for 1e6/64 = 15.6k blocks) instead
  of a linear key scan. Append-only (blocks written once, never deleted —
  retrieval *copies* them), which is HNSW's sweet spot; deterministic levels
  (splitmix64, no RNG) for reproducible decode. **Metrics:** `Dot` (MIPS, matches
  the exact relevance), `Cosine` (magnitude-invariant), and `L2` (Euclidean — a
  *true metric*, so greedy small-world navigation is theoretically sound; often
  better-conditioned for retrieval). **Fuzzy search:** `search_fuzzy(q, k, ef,
  min_score)` drops matches below a relevance floor (no weak hits forced in), and
  `search_radius(q, threshold, max, ef)` is a range query returning *all*
  sufficiently-relevant blocks — tolerant recall not capped at a fixed k. The
  store surfaces these as `retrieve_fuzzy` / `retrieve_radius`; `enable_kv_store`
  takes the metric + a fuzzy floor.
- **On-demand data tiering** — `rlx_runtime::kv_context_store::KvContextStore`
  (feature `mmap-kv`). All block K/V rows are appended, **quantized** (Q4_0 / Q8_0
  / F16), to a per-layer memory-mapped file (`quantized_kv::MmapKvLayer`, with
  `read_rows`/`prefetch_rows` for random-access block fetch). Only the retrieved
  top-k blocks' pages fault in; the OS page cache keeps hot blocks resident.

**1e6-token budget (qwen3-0.6B, `kv_dim=1024`, 28 layers, 64-row blocks):**

| resource | cost | note |
|---|---|---|
| disk (Q4_0) | **32 GB** | 61 GB Q8_0, 115 GB F16 |
| RAM (HNSW keys+graph) | **66 MB** | grows with block *count*, not data volume |
| retrieval IO / step | 33–66 MB (16–32 blocks) | mostly served from page cache |
| HNSW search / step | <1M ops | vs 16M for a linear key scan |

So the working set stays `budget + k·block` rows on GPU while effective context
reaches a million tokens on a laptop NVMe. Wiring: append a block to the store
when it ages out of the recent window; each step `retrieve(query, k)` and splice
the returned blocks into the bounded resident set (same splice as `Retrieval`).
Tested: HNSW recall vs brute force (top-1 ≥90%, recall@10 ≥85%), append→disk→
retrieve round-trip through Q8_0, and RAM-index ≪ disk-data at scale.

**Measured — 100k-token multi-shot on Metal** (`rlx-qwen3 --example context_scale_bench`,
qwen3-0.6B, Q4_0, file-backed): as context grows 10k→100k the per-shot retrieval
latency and GPU decode stay **flat**, RAM index stays tiny, data on disk:

| ctx | disk | RAM index | retrieve | recall | GPU decode |
|---|---|---|---|---|---|
| 10k | 0.32 GB | 0.7 MB | 38 ms | 100% | (warmup) |
| 33k | 1.08 GB | 2.2 MB | 57 ms | 100% | ~11 tps |
| 66k | 2.15 GB | 4.4 MB | 33 ms | 100% | ~11 tps |
| 100k | 3.22 GB | **6.6 MB** | **30 ms** | **100%** | ~11 tps |

Retrieval does *not* grow with context (HNSW `O(log N)` + `O(k)` fetch); GPU decode
is context-independent (bounded resident). 100k tokens sit in 3.2 GB on disk with
a 6.6 MB RAM index. Ingest ~20–30k tok/s. Whole run ~38 s.

### Richer context: neighbors, provenance, streaming

- **Similar-neighbor expansion** — `retrieve_expanded(q, k, n)` returns the top-k
  *plus* each hit's HNSW graph neighbors (semantically adjacent blocks, free from
  the small-world graph), for context that spans blocks near the best match.
- **Provenance** — every block carries an [`Origin`] (Query / File / Generated /
  System / Retrieved / Other) + a `source_id`; retrieval returns it, and
  `retrieve_filtered(q, k, |o| …)` can prefer sources over the model's own output.
- **Streaming ingest** — `ContextStreamer` buffers K/V rows as they arrive (a file
  being read, a query typed, the model generating) into origin-homogeneous blocks
  and appends them live, so injected context — and the model's own streamed
  generation folded back as `Generated` — is immediately retrievable next step.
- **k-means block keys (less averaging)** — instead of one mean-K key per block
  (which averages away the block's discriminative content — the main reason a
  buried fact fails to rank), each block contributes `centroids_per_block`
  **k-means centroids** to the index (deterministic, no RNG). Retrieval finds a
  block via *any* of its centroids, then dedups centroid→block. A fact stays
  findable through its own centroid rather than being washed into the mean.
- **Memory decay** — `retrieve_decayed` re-ranks candidates by relevance ×
  `decay^age` (age = accesses since a block was last retrieved) and marks
  returned blocks fresh, so stale context fades and recent/frequently-used memory
  wins ties (Scissorhands-style forgetting for the unbounded store).
- **Hybrid lexical retrieval** — `retrieve_hybrid` blends the dense (HNSW K/
  centroid) score with a **BM25-lite lexical** score (IDF-weighted query-token
  overlap via an inverted index over per-block token ids). Rescues exact-token
  facts — numbers, names, shared keywords — that K·K similarity misses. The
  generator tags each offloaded block with its token ids (`self.tokens` by abs
  position) and uses the recent-window tokens (≈ the current question) as the
  lexical query. `lexical_weight ∈ [0,1]` mixes dense↔lexical.
- **Builder** — `KvStoreConfig::new().dir(…).topk(…).centroids_per_block(4)
  .metric(Metric::L2).decay(0.999).lexical_weight(0.7)…` →
  `Qwen3Generator::enable_kv_store(cfg)` (feature `mmap-kv`).

### End-to-end wiring (generator ↔ store)

`Qwen3Generator::enable_kv_store(dir, capacity, block, sinks, recent, topk,
neighbors)` makes the context store the retention backend (opt-in, feature
`mmap-kv`; `memory_probe --policies kvstore:BLOCK:SINKS:RECENT:TOPK:NEIGH`). In
`apply_retention` each decode step: aged-out blocks are offloaded **append-once**
(dedup by start-pos), the top-k (+neighbors) relevant blocks are HNSW-retrieved
from the store and **spliced** into the resident cache. **Tested in multi-shot
generation on Metal**: runs end-to-end (12 turns, coherent output, GPU decode
30–46 tps, `retrieved=k` every step confirmed via `RLX_QWEN3_RETENTION_DEBUG`) —
the offload→retrieve→splice loop fires live during decode.

*Recall caveat:* end-to-end fact recall is gated by the **relevance signal**, not
the wiring. The store faithfully returns its top-k, but a buried fact's block may
not rank top-k by `query·mean-K` (the K-similarity proxy) — the same limitation
the earlier probe surfaced. The store path is append-only fixed blocks (it must be,
to scale), so it can't re-chunk adaptively like the in-RAM `Retrieval` path; larger
`topk`/`neighbors` or a stronger relevance signal (exact attention weights,
Stage-3+) is the lever. Full-context stays the recall upper bound.

## Trade-offs

- Retention resets the GPU-resident binding when the resident set changes, so
  those steps re-upload `O(budget)` K/V (vs the in-place row-feed). Net win once
  `budget ≪ context`.
- `Sinks` with too small a `window` is amnesia by design — use `Retrieval` or
  `HeavyHitter` to keep *important* (not merely recent) context.
- Block retrieval currently re-chunks the evicted middle each step (simple,
  correct; a stable-block index is a later optimization).
