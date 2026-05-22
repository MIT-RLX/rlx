# Changelog

All notable changes to RLX. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
tracks SemVer with the understanding that any `0.x → 0.(x+1)`
bump may carry breaking changes per `0.x`-semver convention.

## [0.2.0] — 2026-05

The first release with end-to-end **Qwen3 LM inference** on Apple
Silicon (safetensors + GGUF, F32, parity-checked against the
HuggingFace reference), a high-level **`rlx::run`** runner API, a
**`rlx-run`** CLI, and **GGUF K-quant dequantization** baked into
`Op::DequantMatMul`.

### Added

- **`rlx::run` runner API** (`model builders` crate, `run` module):
  builder-style entry points for the supported model families,
  re-exported in the prelude under the `models` cargo feature.
  - `Qwen3Runner::builder()` — `.weights(p)`, `.device(d)`,
    `.max_seq(n)`, `.precision(F32 | F16LmHead)`,
    `.max_memory_gb(g)`, `.stream(bool)`, `.use_mtp(bool)`,
    `.sample(opts)`, `.config(ConfigSource::…)`,
    `.format(WeightFormat::…)`, `.build()`.
  - `SamRunner::builder(SamArch::Sam1 | Sam2 | Sam3)` — uniform
    builder shape, `.predict_image(...)` method dispatches to the
    per-arch `Sam{,2,3}::from_safetensors_on` + forward call.
  - Helpers `open_loader(path)`, `list_mtp_keys(path)`,
    `debug_resolve_name(hf_name)`.
- **`rlx-run` CLI** (`model builders` crate, `rlx-run` binary): subcommands
  `qwen3`, `sam1`, `sam2`, `sam3`, `inspect`, `help`. Hand-rolled
  arg parser — no clap dep. Mirrors the builder API 1:1.
- **`Op::DequantMatMul` GGUF schemes** (`rlx-ir/src/quant.rs`):
  `QuantScheme::GgufQ4K`, `GgufQ5K`, `GgufQ6K`, `GgufQ8K`. CPU
  implementation in `rlx-cpu` dequants the packed bytes to f32
  scratch then sgemm — keeps the arena footprint small (Q4_K ≈
  4.5 bpe vs F32's 32 bpe) at the cost of per-call dequant. Metal
  lowering is on the roadmap (per-op thunk path still dequants at
  load time today).
- **GGUF K-quant decoders** (`rlx-gguf`): Q4_K, Q5_K, Q6_K, Q8_K
  block decoders, mirroring llama.cpp's `ggml-quants.c` reference.
  Made `pub` so `rlx-cpu`'s `DequantMatMul` GGUF arm can call them.
- **`GgufLoader`** (`model builders::weight_loader`): pluggable
  `WeightLoader` for `.gguf` files with transparent
  HF↔GGUF name resolution (`hf_to_gguf_name` /
  `gguf_to_hf_name`), MTP-head isolation (`is_mtp_weight`,
  `mtp_keys`), and shape normalization (innermost-first GGUF dims
  reversed to safetensors order without byte movement).
- **Qwen3 graph builder** (`model builders` crate, `qwen3`): GQA via
  graph-level KV head repetition, QK-norm, RoPE, SwiGLU,
  tied-embedding LM head with build-time weight pre-transpose
  (eliminates 600 MB per-call Transpose op), prefill + cached
  decode generators with bucketed compile cache.
- **MPSGraph Metal fast path** (`rlx-metal`):
  - `rms_norm` via `normalizationWithTensor:mean=0:variance=mean(x²):`
    (uses Apple's fused norm kernel).
  - `attention_causal` via `scaledDotProductAttention` builtin with
    in-graph constant causal mask — bypasses the slice-of-computed
    MPSGraph optimizer bug that hits the BERT QKV-split pattern.
  - `ElementwiseRegion` chain replay for fused SwiGLU.
  - Pre-compiled `MPSGraphExecutable` with feed/result permutation
    recovered from `executable.feedTensors`/`targetTensors`; per-call
    dispatch is one ObjC call with the input/output `NSArray`s built
    once at compile (`bind_arena` + `run_cached`).
  - Default-on whenever lowering succeeds; opt out via
    `RLX_DISABLE_MPSGRAPH` / `RLX_DISABLE_MPSGRAPH_EXECUTABLE`.
  - Opt-in `RLX_MPSGRAPH_PARAM_CONST=1` bakes weights as graph
    constants (production single-shape callers).
- **F16 LM-head path** (opt-in via `RLX_QWEN3_F16_LM_HEAD=1`): casts
  hidden + lm-head weight to F16 before the final matmul. Wins
  1.3-1.45× on B≥2, L≥64 `last` cells.
- **Examples per model family** (`model builders` repo `examples/`):
  `run_qwen3_safetensors.rs`, `run_qwen3_gguf.rs`, `run_sam1.rs`,
  `run_sam2.rs`, `run_sam3.rs`, plus `qwen3_gguf_inference.rs` and
  `gguf_qwen3_probe.rs` for deeper walk-throughs.
- **Publish script** (`scripts/publish.sh`): tier-ordered workspace
  publisher with active sparse-index polling, HTTP 429 backoff, and
  live countdown timers. See `--help`.

### Changed

- **`Op::DequantMatMul::num_inputs()` is now scheme-dependent**
  (was always 4). Returns 2 for GGUF schemes (`[x, packed_w]`),
  4 for legacy Int8 schemes (`[x, w_q, scale, zp]`). **Breaking**
  for any downstream code that hard-coded the input count — match
  on `scheme.is_gguf()` before reading inputs.
- **`GgufLoader::take_transposed` now actually transposes**
  (was a buggy no-op that returned GGUF native bytes with the GGUF
  shape label, silently producing wrong logits when the builder
  expected `[in, out]` row-major). The fix routes through
  `GgufLoader::take` which now normalizes GGUF's innermost-first
  shape convention to safetensors' outermost-first ordering (no
  byte movement — only the shape label flips). **Breaking** for
  any downstream code that compensated for the old buggy
  behavior; drop the workaround.
- **`Qwen3Generator::from_loader`** canonicalizes cache keys to the
  HF naming convention (via `gguf_to_hf_name`) so the same generator
  works against safetensors OR GGUF loaders without builder changes.
- **`set_param_typed`** on the f32-arena backends (CPU, Metal, wgpu)
  now accepts `DType::U8` and `DType::I8` via the existing
  `set_param_bytes` path. Needed by the GGUF `Op::DequantMatMul`
  path to hand raw packed bytes to the arena. Behavior for
  F32/F16/BF16 is unchanged.
- **Pre-transposed tied LM-head embedding** in the qwen3 builder:
  computed once at graph-build time as a distinct param of shape
  `[hidden, vocab]`. The earlier scheme emitted a runtime
  `Transpose(embed_w, [1,0])` op that materialized ~600 MB per
  forward. CPU `last`-mode prefill drops from ~970 ms → ~70 ms on
  this fix alone.
- **MPSGraph lowering is opt-out** (was opt-in). Env-var name
  changed from `RLX_USE_MPSGRAPH=1` to **`RLX_DISABLE_MPSGRAPH=1`**.
  The matrix harness `model builders` example `qwen3_matrix.rs` no longer needs to
  set anything to engage the fast path.
- `WeightFormat::from_path` / `ConfigSource` / `Precision` /
  `SamArch` enums + the runner builders are re-exported as
  `rlx::run::*` (under the `models` feature).
- `rlx::QuantScheme` flat re-export added to the prelude.
- Workspace version bumped from `0.1.0` → `0.2.0` (all 23
  crates).

### Fixed

- **`Q8K` block byte count off by 16** in `QuantScheme::gguf_block_bytes()`
  (was 276, should be 292 = 4 + 256 + 32). Caught by the new
  `dequant_matmul_q8k_matches_dequant_then_matmul` integration test.
- **MPSGraph attention `MaskKind::Causal`** is now lowered correctly
  (was returning `None` from `try_lower` and falling back to the
  per-op encoder path; now uses Apple's fused SDPA with an in-graph
  constant causal mask).
- **`Op::DequantMatMul` `scheme` field** is now used by the
  CPU lowerer to dispatch to the right kernel; previously the
  GGUF schemes panicked with "scheme not implemented".
- Three pre-existing `model builders` warnings (unused
  `multihead_attention` import in `sam3/detector_decoder.rs`,
  unused `data` arg in `sam3/detector_encoder_ir.rs:add_param`,
  dead `sigmoid` fn in `sam3/tensor.rs`) cleaned up so the
  publish script's `clippy -- -D warnings` gate passes.

### Performance

- **Qwen3-0.6B prefill on Apple Silicon (Metal):** RLX beats
  Python+PyTorch+MPS in 11/23 (B, L, mode) cells, ties in 6,
  with the win margin growing from ~5% at L=32 to 1.45× at L=128.
  Beats Candle CPU on every cell tested (2.6×–9×).
- **Qwen3-0.6B Q4_K_M GGUF on Metal end-to-end:** cosine 0.976 vs
  F32 safetensors — textbook Q4_K_M loss, no NaN, top-1 plausible.

### Docs

- New `CHANGELOG.md` (this file).
- `model builders` repo README: added a Qwen3 section, runner DX section,
  per-example table, env-var matrix for the MPSGraph fast path.
- `rlx-ir/README.md`: added a `QuantScheme` table covering legacy
  Int8 + new GGUF schemes, and a Gotchas note about
  `Op::DequantMatMul`'s variable input count.
- `rlx-gguf/README.md`: replaced overclaiming feature list with an
  honest per-format table; documents the shape-convention quirk
  callers need to know about.
- Root `README.md`: new runner section, Status-by-area entries for
  Qwen3 LM + Op::DequantMatMul GGUF schemes + rlx::run.

### Performance / memory

- **Packed-weights qwen3 builder** (`build_qwen3_graph_sized_packed`):
  K-quant matmul weights stay packed in the arena and the graph
  emits `Op::DequantMatMul { scheme }` per projection. On
  Qwen3-0.6B Q4_K_M: arena drops from 2.22 GB → 1.42 GB end-to-end
  with **bit-exact parity** against the F32-load path (cosine
  1.00000, max\|Δ\| 0.000, top-1 match). End-to-end example at
  `model builders` example `qwen3_packed_inference.rs`; set
  `RLX_QWEN3_PARITY=1` to also build the F32 reference for the
  same file and report cosine.
- **`Qwen3RunnerBuilder::packed_weights(true)`** + CLI `--packed`
  flag — high-level entry to the packed-weights path. Builds the
  packed prefill graph, uploads K-quant params as U8 byte tensors
  via `set_param_typed`, exposes `Qwen3Runner::predict_logits` for
  a single forward AND `Qwen3Runner::generate_packed` for
  streaming via repeated prefills. `generate(...)` auto-routes to
  the packed path in packed mode, so the same caller-side code
  works in both modes. Trade-off: each generated token costs one
  full prefill (no decode-graph KV cache in packed mode yet —
  bucketed decode-graph machinery is still F32-only); throughput
  is ~`max_seq` × slower than the F32 streaming path but memory
  stays packed — the only path that fits 14 B+ Q4_K_M GGUFs on
  commodity Macs today.
- **Layout bug fixed** in CPU `Op::DequantMatMulGguf`: the dequant
  output is `[n, k]` row-major (GGUF byte order), not `[k, n]` —
  the original arm called `sgemm` which silently produced wrong
  outputs for `n > 1` cells. Now uses `sgemm_bt` (B transposed).
  Pinned by a new `dequant_matmul_q8k_correct_layout_for_n_gt_1`
  regression test that's specifically picked to fail under the
  old layout.

### Known limitations

- **Qwen3.5 / Qwen3.6 (`qwen35`) hybrid gated-DeltaNet + attention**:
  the unsloth/froggeric `Qwen3.5-0.8B-MTP-GGUF` and
  `Qwen3.6-27B-MTP-GGUF` files both tag `general.architecture =
  "qwen35"` (Qwen3-Next style: gated DeltaNet "linear attention"
  trunk layers interspersed with standard attention every
  `full_attention_interval`, plus an MTP head). End-to-end forward
  pipeline shipped this release:
  - `Op::GatedDeltaNet { state_size }` — new IR op + CPU
    autoregressive scan kernel mirroring
    `delta-net-base.cpp::build_delta_net_autoregressive`. Parity-
    tested against a scalar reference + per-batch state-reset test
    (`rlx-runtime/tests/cpu_gated_delta_net_parity.rs`, 2/2 green).
  - `Qwen35Config::from_gguf` + `Qwen35Weights::from_loader{,_packed}`
    — full per-layer tensor bundle. Auto-detects linear-attn vs
    full-attn layers from `full_attention_interval`; loads the MTP
    layer's NextN `eh_proj` / `enorm` / `hnorm` / optional
    `embed_tokens` / `shared_head_*`. `MatWeight::{F32, Packed}`
    enum routes K-quant matmul weights through `Op::DequantMatMul`
    when `from_loader_packed` is used.
  - `build_qwen35_graph_sized` — full prefill IR: gated-DeltaNet
    trunk (norm → joint qkv+gate split → α/β/dt + softplus gate
    → unrolled k=4 depthwise causal conv → SiLU → q/k/v split →
    L2-norm → GQA-repeat → `Op::GatedDeltaNet` → silu(z)-gated
    norm → `ssm_out`) + every-`full_attention_interval` standard
    attention block (joint Q+gate, sigmoid-gated attn output) +
    optional MTP head. 2/2 basic tests green (graph builds,
    executes, produces finite logits on both trunk + MTP outputs).
  - `Qwen35Runner` / `Qwen35RunnerBuilder` — mirrors the
    `Qwen3Runner` API; `.packed_weights(true)` opts into the K-
    quant in-arena path. `.generate(prompt_ids, n_new, on_token)`
    runs autoregressive greedy generation via repeated prefills.
  - `rlx-run qwen35` CLI subcommand + `examples/run_qwen35.rs`
    end-to-end. Flags: `--packed`, `--mtp`, `--max-tokens N`,
    `--prompt-ids 1,2,3`, `--max-seq N`.
  - Deviations from the llama.cpp reference (flagged for the
    next-slice parity oracle): standard per-axis RoPE substituted
    for the rope-sections MRoPE; depthwise k=4 conv unrolled into
    narrow+mul+add (no `Op::Conv`); per-batch state reset (no
    decode-time state cache).
  Memory: F32 dequant path needs ~1.5 GB for 0.8B (fits) /
  ~65 GB for 27B (doesn't fit). Packed path drops 27B to ~16 GB
  (fits) by keeping K-quant bytes in the arena. Numerical parity
  vs llama.cpp on a real GGUF is the next milestone.

  **Packed-loader perf**: zero-copy upload path. `take_packed`
  used to `.to_vec()` each K-quant tensor's bytes (~16 GB of
  memcpy on 27 B Q4_K_M). New flow:
  `take_packed_metadata` records `(scheme, shape)` only,
  `MatWeight::Packed` holds the loader key, and the runner uploads
  via `loader.tensor_bytes_borrowed(key) → compiled.set_param_typed`
  — bytes flow straight from mmap into the arena, no intermediate
  Vec. Also: reuse the loader's already-parsed `GgufFile` for
  `Qwen35Config::from_gguf` (was re-parsing 800+ tensor headers,
  ~10 s saved on 27 B). Builder/runner now log per-phase timing
  via `eprintln!` so future regressions surface.
- **`Op::DequantMatMul` on Metal** still falls through to the
  per-op thunk path; the GGUF schemes only have CPU lowerings
  today. On Apple GPUs the F32-load path is the working option
  until the native Metal Dequant kernel lands.
- **Streaming decode tok/s** in `Qwen3Runner::generate` recompiles
  per token in `stream(true)` mode — the bucketed compile cache
  doesn't get hit until the second pass. Fix in 0.2.0: callback
  threaded through `Qwen3Generator::generate_cached` so a single
  compile covers the whole `n_new` decode loop.
- **Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q1_0**
  GGUF formats are not decoded. Files containing them raise a
  clean "dequant for {type} not implemented yet" error.
- **27 B-class GGUF on Mac**: requires the Metal `Op::DequantMatMul`
  kernel above (108 GB F32-dequant footprint doesn't fit anywhere
  affordable). Models up to ~8 B Q4_K_M load and run today on a
  32 GB unified-memory Mac.
- **MTP heads** are now loadable end-to-end on
  `unsloth/Qwen3.6-27B-MTP-GGUF`-style files: pass
  `--use-mtp` (CLI) or `.use_mtp(true)` (runner builder) to flip
  the `GgufLoader::include_mtp` visibility; MTP tensors are drained
  into the generator's weights cache and a diagnostic logs how many
  heads were captured. Direct access via `GgufLoader::take_mtp(name)`
  is also exposed. The base generation path still runs single-token
  decode (the speculative + verify loop that would *use* the heads
  is the follow-up); inference succeeds either way.

### Internal

- `Op::DequantMatMulGguf` thunk variant added in `rlx-cpu` to
  carry the GGUF scheme through scheduling + VJP recompute paths
  cleanly.
- Workspace member layout unchanged.

---

## [0.1.0] — 2026-04

Initial release. Tracked at [git history root].
