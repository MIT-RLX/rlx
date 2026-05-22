# rlx-flow

Block assembly-line API for RLX model builders — **no HIR/Graph in model code** unless you opt into tier 2.

## Three tiers

| Tier | You write | Example |
|------|-----------|---------|
| 0 | `ModelFlow` + small blocks | `.token_embed().layer("L0", \|s\| s.rms_norm(...))` |
| 1 | `CompileProfile` / `*.rlx.toml` | fusion, precision, passes |
| 2 | `.custom()` + `rlx_flow::escape::Emit` | novel subgraphs — promote to blocks when stable |

## Model author quick start

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

## Small composable blocks

| Block | DSL | Purpose |
|-------|-----|---------|
| `token_embed` | `.token_embed()` | HF embedding table |
| `rms_norm` | `.rms_norm(key, eps)` | pre-norm |
| `layer_norm` | `.layer_norm(gamma, beta, eps)` | BERT-style LayerNorm |
| `gelu_ffn` | `.gelu_ffn(layer_prefix)` | BERT GELU FFN |
| `bert_encoder_layer` | `.bert_encoder_layer(spec)` / `.repeat_bert_layers(...)` | full BERT encoder layer |
| `nomic_encoder_layer` | `.repeat_nomic_layers(...)` | NomicBERT RoPE + SwiGLU layer |
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
