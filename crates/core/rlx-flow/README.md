# rlx-flow

Block assembly-line API for RLX model builders — **no HIR/Graph in model code** unless you opt into tier 2.

## Three tiers

| Tier | You write | Example |
|------|-----------|---------|
| 0 | `ModelFlow` + small blocks | `.token_embed().layer("L0", \|s\| s.rms_norm(...))` |
| 1 | `CompileProfile` / `*.rlx.toml` | fusion, precision, passes |
| 2 | `.custom()` + `rlx_flow::escape::Emit` | novel subgraphs — promote to blocks when stable |

## Getting started for model authors

```rust
use rlx_flow::prelude::*;
use rlx_ir::{DType, Shape};

// Generic causal LM skeleton
let flow = ModelFlow::new("my_lm")
    .profile_prefill()
    .input("tokens", Shape::new(&[1, 128], DType::F32))
    .rope_tables(tables)
    .zero_beta(hidden)
    .token_embed()
    .repeat_layers(num_layers, |i| {
        llama_prefill_layer_fused(i, layer_spec(i))
    })
    .final_norm(eps)
    .lm_head(vocab, hidden, tied);

let built = flow.build(&mut weights)?;
```

## Weights

`build` takes any `WeightSource`. Checkpoint files (feature `formats`):

```rust
use rlx_flow::formats::{self, GgufSource, SafetensorsSource};
let mut w = SafetensorsSource::open("model_dir")?;            // file or shards
let mut w = GgufSource::open("m.gguf")?.packed(true);          // DequantMatMul
let mut w = formats::open(path)?;                              // by extension
```

Adapters (`WeightSourceExt`, in the prelude) — all pass packed weights through:

```rust
let mut w = CloningSource(&store);              // non-consuming, any `TensorLookup`
let mut w = w.with_prefix("language_model.");   // names differ by a prefix
let mut w = w.renamed(|k| k.replace("layers.", "blk."));
let mut w = w.alias("lm_head.weight", "model.embed_tokens.weight"); // tied head
let mut w = w.map_tensors(|k| k.ends_with("q_proj.weight"), |_, d, s| Ok((d, s)));
let mut w = w.or(extra);                        // fill gaps from a second source
let mut w = w.tracked();
flow.build(&mut w)?;
eprintln!("{}", w.report());                    // served / missing / unused keys
```

A missing key's error names the closest keys the source has
(`missing weight: model.x (closest: language_model.model.x)`).

### Weight storage

Loaders hand builders f32, so a bf16 checkpoint used to land in the arena at
twice its file size. `ModelFlow::with_weight_storage(WeightStorage::Auto)` (or
`RLX_WEIGHT_STORAGE=auto|f32|f16|bf16` for every flow in the process) stores
rank-2 matmul weights at their own width instead — every backend's matmul reads
f16 / bf16 weights and computes in f32, so `Auto` gives the same products at
half the memory. Packed quantized weights stay packed; norms, biases,
embeddings and conv kernels stay f32. Without either, flows keep f32.

## Small composable blocks

| Block | DSL | Purpose |
|-------|-----|---------|
| `token_embed` | `.token_embed()` | HF embedding table |
| `rms_norm` | `.rms_norm(key, eps)` | pre-norm |
| `layer_norm` | `.layer_norm(gamma, beta, eps)` | BERT-style LayerNorm |
| `gelu_ffn` | `.gelu_ffn(layer_prefix)` | BERT GELU FFN |
| `bert_encoder_layer` | `.bert_encoder_layer(spec)` / `.repeat_bert_layers(...)` | full BERT encoder layer |
| `NomicEncoderLayerStage` | `.repeat_nomic_layers(...)` | NomicBERT RoPE + SwiGLU layer |
| `gather_add` | `.gather_add(input, weight)` | add position/type embeddings |
| `linear` | `.linear(key, transpose)` | matmul |
| `residual_save` / `residual_add` | `.residual_save()` … `.residual_add()` | skip connections |
| `self_attn_prefill` | `.self_attn_prefill(spec)` | QKV + RoPE + GQA + causal |
| `swiglu` | `.swiglu_hf_mlp(prefix)` | SwiGLU FFN |
| `LayerStack` | `.layer("L0", \|s\| s....)` | named sub-layer composer |

Fused fast path: `llama_prefill_layer_fused(i, spec)` → one HIR composite.  
Composable path: `llama_prefill_layer_composed(i, spec)` → small blocks above.

## LLaMA 3.2 (recommended)

```rust
// Llama32Flow lives in the model-builders repo (see root README).
Llama32Flow::for_prefill(&cfg, 1, 128)
    .last_token_logits()
    .profile_near(&weights_path)
    .build(&mut weights)?;
```

Customize one layer without IR:

```rust
Llama32Flow::for_prefill(&cfg, 1, 128)
    .layer(|ctx| {
        if ctx.index() == 0 {
            llama_prefill_layer_composed(0, spec.clone())
        } else {
            ctx.default_stage()
        }
    })
    .build(&mut weights)?;
```

See [`DESIGN.md`](DESIGN.md).

## License

MIT OR Apache-2.0.
