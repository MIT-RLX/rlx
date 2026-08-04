# rlx-opscope

A **data-pattern recording harness**. Run existing op graphs with varied data,
record cheap *sketches* of every tensor flowing through the ops we care about
(matmuls first), then mine those sketches offline for exploitable structure —
sparsity, per-channel outliers, quantization headroom, low rank, sequence
structure — that justifies building a **specialized kernel** to cut compute.

This is the MVP slice of a larger loop:

```
 drive data ──▶ inject stat taps ──▶ record sketches ──▶ mine patterns ──▶ (specialize + guard)
```

## The load-bearing idea: rewrite the graph, don't tap the executor

rlx has two CPU execution paths and opaque GPU backends, so a per-op host
callback can only see CPU intermediates on the slow path. Instead we **rewrite
the graph**: for each op-site, [`inject_matmul_stats`] appends reduction /
histogram nodes on the op's `lhs`, `rhs`, and `out` tensors and marks them as
extra graph outputs. The stats are then computed by *the backend's own kernels*
— so this works identically on CPU / Metal / CUDA / …, reuses peak kernels, and
costs nothing when the pass isn't applied. The primary output keeps index 0, so
the injected graph is a drop-in (a correctness gate in `opscope-sweep` asserts
`injected_output == matmul_output`).

This is exactly why `Op::Histogram` was added to the core: value histograms are
part of the sketch set, native O(n) on CPU and decomposing to
`Compare + Reduce::Sum` everywhere else.

## Sketches recorded per tensor

| sketch | how | reveals |
|---|---|---|
| `min`/`max`/`mean` | `Reduce` | range, bias |
| `l1` / `sumsq` | `abs`/`x²` → `Reduce::Sum` | magnitude, L2 |
| `nnz` | `(x!=0)` → `Reduce::Sum` | **density → sparse GEMM** |
| `hist` | `Op::Histogram` (32 bins) | **distribution shape → quantize/LUT** |
| `chan_maxabs` | `max|x|` over all-but-last axis | **per-channel outliers → int quant** |
| `pos_sumsq` | `x²` summed over last axis | **per-position energy → sequence structure** |
| `adj_sumsq` | `Σ(rowₜ − rowₜ₋₁)²` (Narrow+Sub+Reduce) | **adjacent-row coherence → delta-compute** |

The *sequence/temporal* tracking has three axes: per-channel (feature) outliers,
per-position (token) energy, and adjacent-row coherence **within** a call; plus a
`step`-indexed time series **across** calls (the miner's cross-step section
separates a stationary weight — `precompute/prepack` — from drifting activations).

## Usage

```sh
# Synthetic sweep (shapes × 7 distributions) → tidy CSV, then mine it
cargo run -p rlx-opscope --bin opscope-sweep --release -- sweep.csv
cargo run -p rlx-opscope --bin opscope-mine  --release -- sweep.csv

# Decode-like stepped sequence (fixed weight, drifting activations) → temporal
cargo run -p rlx-opscope --bin opscope-seq   --release -- seq.csv 16
cargo run -p rlx-opscope --bin opscope-mine  --release -- seq.csv

# Real MNIST MLP on real MNIST pixels (reads ~/.cache/torchvision-mnist)
cargo run -p rlx-opscope --bin opscope-mnist --release -- mnist.csv 12
cargo run -p rlx-opscope --bin opscope-mine  --release -- mnist.csv

# Recurring op-subsequence motifs (linear chains) on a synthetic MLP stack
cargo run -p rlx-opscope --bin opscope-motifs --release -- 6

# Repeated DATAFLOW sub-DAGs (branching cones) → decomposition/fusion candidates
cargo run -p rlx-opscope --bin opscope-flow --release -- 6 transformer   # or: mlp | moe
cargo run -p rlx-opscope --bin opscope-flow --release -- 6 moe

# Same analysis on a graph dumped from ANOTHER workspace (real rlx-models model):
#   the model repo walks its graph → edge-list (no opscope dep); opscope loads it.
cargo run -p rlx-opscope --bin opscope-graph --release -- dumped_graph.txt
```

The CSV is **tidy/long**
(`run_id,step,backend,dist,M,K,N,numel,site,role,stat,idx,value`) so scalars,
per-channel vectors, and histograms share one flat schema that pandas/polars can
`groupby`. `numel` is each tapped tensor's own element count, so multi-site
graphs (different shapes per matmul) work without decoding M/K/N. No Arrow/Parquet
dependency yet — a drop-in swap for the [`Recorder`] sink later.

### Real MNIST result (what the harness actually finds)

Running `opscope-mnist` (untrained MLP, real pixels) the miner reports, per site:

```
mnist  fc1  lhs  density 0.188  → sparse-GEMM (skip 81% zeros)   # MNIST border sparsity
mnist  fc2  lhs  density 0.500  → sparse-GEMM (skip 50% zeros)   # post-ReLU activation sparsity
...  fc1/rhs, fc2/rhs → STATIONARY across steps → precompute/prepack   (weights fixed)
...  all activations  → drifting across steps → temporal coherence
```

Both sparsities are real and training-independent; the stationarity falls out of
the fixed weights streamed across decode steps. (The histogram is **scale-
normalized** by default — `x / (max|x| + eps)` over `[-1,1]` — so tightly-scaled
weights no longer false-flag as "spiky/quantizable".)

### Dataflow decomposition (repeated patterns → fusion candidates)

`opscope-flow`/`opscope-graph` mine repeated **dataflow sub-DAGs** (each node's
branching input cone, merkle-hashed) — the units that can be *decomposed once and
shared* or *fused*. On a transformer stack the repeated attention cone (with
`Softmax`) recurs per layer → `FusedAttentionBlock`; the FFN cone (`Silu`) →
`FusedMatMulBiasAct`. On a MoE stack the `GroupedMatMul`+`TopK`-gating expert
block recurs → *fuse grouped-matmul + gating*. `opscope-graph` runs this on a
graph **dumped from another workspace** (a real rlx-models model), so no shared
build or runtime dependency is needed — the model repo just walks its graph.

## Inference-optimization tiers

Beyond the value sketches, opscope mines four tiers of signal, each pointing at a
specific kernel:

- **Tier 1 — inference dynamics** (`opscope-infer`): taps `Softmax` (per-query
  peak / per-key received attention mass → sparse/windowed attention, KV
  eviction) and `TopK` (per-expert load → drop cold / prefetch hot experts).
  These only exist at run time. `inject_infer_stats`.
- **Tier 2 — host probes** (`opscope-probe`, `probe.rs`): on a deep-dumped
  tensor, the signals the CSV sketches can't see — effective rank (→ factored
  matmul), 2:4 feasibility (→ sparse tensor cores), quant error @ bits (→ int
  kernel), value cardinality (→ LUT). `save_tensor`/`load_tensor` capture.
- **Tier 3 — workload/roofline** (`opscope-shapes`, `shapes.rs`): per-op
  FLOPs/bytes/intensity → compute- vs memory-bound split (fusion strategy) + a
  hot-GEMM-shape histogram (autotune/dispatch-table targets).
- **Tier 4 — actuation** (`opscope-plan`, `plan.rs`): mined CSV → per-site
  exploit + a synthesized **runtime guard** predicate + guard **stability**
  (cross-step σ) + **estimated** and **measured** (A/B micro-bench) speedup, plus
  stationary-weight prepack items. This is the half that makes mining *decide*.
- **Cross-cutting** (`online.rs`): reservoir / t-digest-lite quantiles / HLL
  cardinality — bounded one-pass sketches for a *sampled* production-inference
  recorder (near-zero overhead), deterministic, unit-tested.

## Scope / what's next

- **Now:** matmul taps + CPU exec on synthetic + real (MNIST) data; tidy-CSV;
  rule-based miner; within-call adjacency + cross-call time-series; **adaptive
  (scale-normalized) histogram**; linear op-subsequence motifs; **repeated
  dataflow sub-DAG mining** (transformer/MoE fusion candidates) + a dump-loader
  bridge to profile real rlx-models graphs structurally.
- **Live on a real rlx-models model:** `rlx-vision-bench/examples/opscope_mnist.rs`
  runs opscope on the real `build_eval_graph` MLP + real MNIST via link-local
  (opscope added as a dev-dep, like `rlx-collectives`). Two cross-workspace
  gotchas: Cargo finds the link-local `[patch]` in `.cargo/config.toml` from the
  **current dir**, so `cd rlx-models` first (don't use `--manifest-path` from
  elsewhere); and a `Cargo.toml` change stales the lock, so the first build may
  fail — retry or `cargo update`.
- **Next:** the same driver on real LLM/MoE runners (not just the MLP); Parquet;
  an effective-rank probe (SVD — not sketch-observable today); and the
  specialization + runtime-guard half of the loop.

Low rank is intentionally flagged as **not sketch-observable** by the miner
(these reductions can't see singular values) — it's a deep-dump candidate, which
is the honest signal the miner should give.
