# Adding a new model

Borrowed from MAX's `pipelines/architectures/<name>/` 4-file layout
(plan #79). Once you adopt the structure, adding a new arch becomes
filling in slots, not editing every consumer.

Today the in-tree models (BERT, NomicBERT, NomicVision) live as flat
`bert.rs` / `nomic.rs` / `vision.rs` files. As they grow or as new
archs land, prefer this structure:

```
rlx-models/src/<arch_name>/
├── arch.rs              # Architecture spec + registration
├── model.rs             # Graph builder
├── model_config.rs      # HF-config struct + parsing
└── weight_adapters.rs   # HF→RLX weight name remap
```

## File responsibilities

### `arch.rs`

Tiny — registers the arch into [`arch_registry`](src/arch_registry.rs):

```rust
use crate::arch_registry::{register, ArchSpec, ArchFamily};

pub fn register_self() {
    register(ArchSpec {
        name: "my-llama",
        family: ArchFamily::Other, // pick the family that fits
        description: "My fork of LLaMA-3 with custom RoPE base.",
    });
}
```

### `model_config.rs`

The HF `config.json` shape as a serde struct, plus any
defaults / validation:

```rust
#[derive(Deserialize)]
pub struct MyLlamaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub rope_theta: f32,
    // ...
}

impl MyLlamaConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> { /* ... */ }
}
```

### `weight_adapters.rs`

Maps HF weight names to whatever names the graph builder expects.
For BERT-style models that's mostly identity; for forks (e.g. a
re-named QKV scheme) this is where the rename rules go:

```rust
pub fn adapt(loader: &mut dyn WeightLoader, hf_key: &str) -> Result<...> {
    let renamed = match hf_key {
        "model.embed_tokens.weight" => "embeddings.word_embeddings",
        // ...
        other => other,
    };
    loader.take(renamed)
}
```

### `model.rs`

The graph builder — `pub fn build_my_llama_graph(...)`. Reads the
config, calls into the weight adapter, emits IR ops. Most of the
actual code lives here.

## Why this layout

- **Separation of concerns**: the four jobs (registration, config
  parsing, weight loading, graph building) move at different speeds.
  Renaming an HF weight key shouldn't touch the graph builder; bumping
  the rope_theta default shouldn't touch the registration.

- **Mechanical onboarding**: a new arch is "make the directory, fill
  in the four files, register." No surgery on every consumer.

- **Cross-team contributions**: someone porting weights from a new
  format only edits `weight_adapters.rs`; someone adjusting the
  graph topology only edits `model.rs`.

## Migration

Don't refactor existing single-file models for the sake of it — only
when they grow large enough that the four concerns are tangled. The
convention is a target for *new* archs.
