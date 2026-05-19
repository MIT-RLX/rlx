# rlx-models

Concrete model graph builders + weight loaders for RLX. The "what
actually runs" layer.

## What's here

- **`qwen3`** — Qwen3 decoder LM (GQA, QK-norm, RoPE, SwiGLU, tied
  embeddings). Prefill + cached decode generators; safetensors and
  GGUF loaders. Production for Qwen3-0.6B; loads any
  Qwen3-arch GGUF up to whatever fits in unified memory at F32
  dequant. See [Qwen3 below](#qwen3).
- **`qwen35`** — Qwen3.5 / Qwen3.6 hybrid trunk (gated DeltaNet
  "linear attention" layers + every-Nth standard attention layer +
  optional MTP head). Powers `unsloth/Qwen3.5-0.8B-MTP-GGUF` and
  `froggeric/Qwen3.6-27B-MTP-GGUF`. End-to-end forward via
  `Qwen35Runner::builder().packed_weights(true).build()` (CPU,
  with `Op::GatedDeltaNet`). Numerical parity vs llama.cpp is the
  next milestone — kernel parity is currently self-consistency
  only. See `examples/run_qwen35.rs`.
- **`bert`** — BERT graph builder. Powers MiniLM, BGE, all-MiniLM-L6-v2.
- **`nomic`** — NomicBERT (uses RoPE + SwiGLU).
- **`vision`** — vision encoder graphs (NomicVision-style).
- **`dinov2`** — DINOv2 ViT encoder (B/14, L/14, g/14).
- **`sam`**, **`sam2`**, **`sam3`** — Segment Anything image encoders
  + mask decoders.
- **`embed::RlxEmbed`** — high-level wrapper that pairs a graph with a
  tokenizer + pooling. One-liner `from_pretrained("...")` if the
  `hf-download` feature is enabled.
- **`config`** — HuggingFace `config.json` parser.
- **`weight_loader`** — pluggable `WeightLoader` trait. Today's
  adapters: `WeightMap` (safetensors), `GgufLoader` (GGUF, including
  Q4_K / Q5_K / Q6_K / Q8_K K-quants with HF↔GGUF name mapping and
  MTP-head isolation).
- **`run`** — high-level `Qwen3Runner` / `SamRunner` builders for
  one-call inference (weights + device + memory ceiling + streaming).
  See [the runner section below](#high-level-runner-dx). Powers the
  `rlx-run` binary.

## Install

```toml
[dependencies]
rlx-models = "0.1"
```

For HF-hub download support:

```toml
rlx-models = { version = "0.1", features = ["hf-download"] }
```

## Quickstart — embeddings

```rust
use rlx_models::embed::{Pooling, RlxEmbed};

let mut model = RlxEmbed::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
let hidden = model.forward(&[("input_ids", &ids), ("attention_mask", &mask)], 1, 16)?;
```

## High-level runner DX

For people who just want to load a model and pull tokens, `rlx_models::run`
exposes builder-style entry points (also re-exported as `rlx::run`):

```rust
use rlx_models::run::{Qwen3Runner, Precision};
use rlx_runtime::Device;

let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-0.6B-Q4_K_M.gguf")       // safetensors OR gguf
    .device(Device::Metal)                    // cpu | metal | mlx | gpu
    .max_seq(128)                             // prefill bucket
    .precision(Precision::F32)                // F32 | F16LmHead
    .max_memory_gb(16.0)                      // soft cap, errors if exceeded
    .stream(true)                             // call on_token per id
    .use_mtp(false)                           // ignore MTP heads (default)
    .packed_weights(false)                    // keep K-quants packed (low memory; CPU-only)
    .build()?;

runner.generate(&prompt_ids, 32, |tok| {
    print!("{tok} ");
})?;
```

For models that don't fit in unified memory after F32 dequant
(Qwen3-14B+, Qwen3.6-27B, …), opt into **packed weights**:

```rust,ignore
let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-14B-Q4_K_M.gguf")
    .packed_weights(true)               // cuts arena ~6× on Q4_K_M; CPU-only
    .max_seq(128)
    .build()?;

// Same `generate(...)` surface as the F32 path — in packed mode it
// auto-routes to `generate_packed`, which runs the prefill graph
// autoregressively (one full prefill per token; slow but
// memory-frugal).
runner.generate(&prompt_ids, 16, |tok| {
    print!(" {tok}");
})?;

// Or a single forward if you just want the next-token logits:
let logits = runner.predict_logits(&prompt_ids)?;
```

Or from the CLI:

```sh
cargo run --release -p rlx-models --bin rlx-run --features metal -- \
    qwen3 --weights Qwen3-14B-Q4_K_M.gguf --packed --max-seq 128 \
          --max-tokens 16 --prompt-ids 1,17,42
```

The format (`safetensors` vs `gguf`) is auto-detected from the file
extension. For GGUF inputs the config is read from embedded metadata;
for safetensors, from a sibling `config.json` (override via
`.config(ConfigSource::JsonFile(path))`).

SAM 1/2/3 use the parallel `SamRunner::builder(SamArch::Sam2)` shape;
the underlying graph builders are unchanged.

### `rlx-run` CLI

The same builder is wired into a small command-line tool:

```sh
cargo run --release -p rlx-models --bin rlx-run --features metal -- \
    qwen3 --weights Qwen3-0.6B-Q4_K_M.gguf \
          --device metal \
          --max-tokens 32 \
          --max-seq 128 \
          --prompt-ids 1,17,42,314

cargo run --release -p rlx-models --bin rlx-run -- \
    inspect Qwen3-0.6B-Q4_K_M.gguf
```

`rlx-run inspect <path>` dumps format, tensor count, dtype histogram,
GGUF metadata architecture, and any MTP heads present. Useful before
deciding whether to enable `--use-mtp` or how much memory to budget.

### One-file examples per model

`examples/` ships a focused runner per supported family — read them
as templates, copy + tweak for your workload:

| File | What it does |
|---|---|
| `run_qwen3_safetensors.rs` | Qwen3 from HF safetensors, builder API, streaming greedy decode |
| `run_qwen3_gguf.rs` | Same but from a `.gguf` (Q4_K_M / Q5_K_M / Q6_K), with MTP head detection |
| `run_sam1.rs` | SAM 1 (vit_b / vit_l / vit_h) — encode image, run prompt encoder + mask decoder |
| `run_sam2.rs` | SAM 2 (hiera tiny / small / base_plus / large) — image segmentation with FPN + memory attention |
| `run_sam3.rs` | SAM 3 — text-conditioned detection + mask prediction |
| `qwen3_gguf_inference.rs` | Detailed Qwen3 GGUF walk-through with safetensors parity check |
| `gguf_qwen3_probe.rs` | Validate that `hf_to_gguf_name` resolves cleanly against a real GGUF |
| `qwen3_matrix.rs` | Full (B, L, mode) × (CPU, Metal, MLX, wgpu) parity + perf sweep vs candle |

Run any of them with:

```sh
cargo run --release -p rlx-models --features metal --example <name> -- [args]
```

The SAM examples synthesize a 1024×1024 RGB gradient so they run
without an external image dep — swap in `image::open(path).to_rgb8().as_raw()`
for a real picture.

## Qwen3

Qwen3 prefill + autoregressive decode runs on CPU, Metal, MLX, and
wgpu with the same graph builder. Parity is 100% top-1 against the
HuggingFace `transformers` reference across every (B, L) cell tested
(`tests/qwen3_parity.rs`).

### From safetensors

```rust
use rlx_models::qwen3::{Qwen3Config, build_qwen3_graph_sized_last_logits};
use rlx_models::weight_map::WeightMap;
use rlx_runtime::{Device, Session};

let cfg = Qwen3Config::from_file("weights/Qwen3-0.6B/config.json".as_ref())?;
let mut wm = WeightMap::from_file("weights/Qwen3-0.6B/model.safetensors")?;
let (graph, params) = build_qwen3_graph_sized_last_logits(
    &cfg, &mut wm, /*batch*/ 1, /*seq*/ 128, /*with_kv_outputs*/ false,
)?;
let mut compiled = Session::new(Device::Metal).compile(graph);
for (name, data) in &params { compiled.set_param(name, data); }
let logits = compiled.run(&[("input_ids", &input_ids_f32)]).remove(0);
```

### From GGUF (Q4_K_M, Q5_K_M, Q6_K supported out of the box)

`GgufLoader` translates HF tensor names → GGUF's `blk.N.attn_*` /
`ffn_*` convention transparently, decodes K-quants, and hides
Multi-Token-Prediction heads from non-MTP builders. The qwen3
builder needs no changes:

```rust
use rlx_models::weight_loader::GgufLoader;

let mut wm = GgufLoader::from_file("Qwen3-0.6B-Q4_K_M.gguf")?;
let (graph, params) = build_qwen3_graph_sized_last_logits(
    &cfg, &mut wm, 1, 128, false,
)?;
// …same compile + run as above
```

End-to-end demo: `cargo run --release --example qwen3_gguf_inference
-- path/to/model.gguf [RLX_QWEN3_WEIGHTS=…/safetensors for parity]`.
Verified against `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf`
(cosine 0.976 vs F32 safetensors — within textbook Q4_K_M loss).

### Apple Silicon performance

The Metal path lowers the entire forward to one MPSGraph executable
(per shape), using Apple's `scaledDotProductAttention` builtin +
fused `normalizationWithTensor` for RMSNorm + Q6_K-aware matmul
paths. On Qwen3-0.6B prefill, RLX matches or beats Python+PyTorch+MPS
in 11 of 23 (B, L, mode) cells and ties in 6 more (see
`examples/qwen3_matrix.rs` for the harness; numbers in
`qwen3_metal_perf` memory note). Opt-in toggles:

| env var | what it does |
|---|---|
| `RLX_DISABLE_MPSGRAPH=1` | drop back to per-op Metal thunks |
| `RLX_DISABLE_MPSGRAPH_EXECUTABLE=1` | use JIT MPSGraph instead of precompiled `MPSGraphExecutable` |
| `RLX_MPSGRAPH_PARAM_CONST=1` | bake weights into the executable as graph constants (production single-shape callers; OOMs the multi-shape bench harness) |
| `RLX_MPSGRAPH_PARAM_CONST_CAP=N` | per-param byte ceiling for the above (default 4 MB) |
| `RLX_QWEN3_F16_LM_HEAD=1` | cast input + lm_head weight to F16 for the final matmul (1.3-1.45× on B≥2, L≥64 `last` cells; loses on small cells) |
| `RLX_MPSGRAPH_TRACE=1` | print which op blocked lowering, if any |

## Build / test

```sh
cargo build -p rlx-models
cargo test  -p rlx-models                              # unit + structure tests
cargo test  -p rlx-models --features parity-candle     # cross-framework parity
```

burnembed (`/Users/Shared/burnembed`) is the integration testbed for
embedding workloads — loads weights via `hf-hub`, benchmarks RLX vs.
ORT vs. ndarray-fused.

## Status

| family | safetensors | GGUF | parity |
|---|---|---|---|
| `bert`, `nomic`, `vision`, `dinov2` | yes | n/a | production |
| `sam`, `sam2`, `sam3` | yes | n/a | production |
| `qwen3` (LM) | yes | yes (Q4_K_M / Q5_K_M / Q6_K) | top-1 vs HF reference |

For multi-tenant serving (paged KV cache, continuous batching) the
autoregressive runtime lives in `rlx_runtime::paged_kv`; the
generators in `qwen3::generator` are single-stream.

## Gotchas

- Weight names in safetensors don't always match the IR `Param` name —
  `weight_map.rs` does the rename. For GGUF, `weight_loader::GgufLoader`
  handles HF↔GGUF translation; new architectures only need their
  unique mappings added there.
- **GGUF shape convention**: GGUF reports tensor dims innermost-first
  (`ne[0]` is the fastest-varying), safetensors reports outermost-first.
  Byte layout is *identical* row-major — `GgufLoader::take` just
  reverses the shape label. **Do not physically transpose in `take`**
  (the old bug produced cosine ≈ 0 logits).
- **Unsupported GGUF quants**: Q1_0, Q2_K, Q3_K, IQ* families aren't
  decoded yet. Caller gets a clean "dequant for X not implemented"
  error.
- **27B-class GGUFs on Mac**: load-time dequant to F32 = 108 GB.
  Doesn't fit. Needs `Op::DequantMatMul` on Metal (CPU has it) to
  keep weights packed in 13.5 GB through inference; tracked as
  follow-up work.
- Pooling (`mean`, `cls`) is per-model and lives in burnembed's bench
  examples, not here.
- New model arch onboarding: add a module here, optionally a
  `WeightLoader` mapping, optionally a parity test under `tests/`.

## License

GPL-3.0-only.
