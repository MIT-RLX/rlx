# rlx-peft

Parameter-efficient adaptation on RLX graphs — LoRA, IA3, AdaLoRA, DoRA and OFT
as operations on an `rlx_ir::Graph` rather than as wrappers around another
framework. **Downstream package**: builds on `rlx-ir` / `rlx-runtime` without touching
framework core.

## What's here

Each adapter reproduces HuggingFace PEFT's definition, which is what published
results are measured under:

| adapter | update | trainable |
|---|---|---|
| `lora_delta` | `W + (α/r)·B A` | `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}` |
| `ia3_apply` | `(x Wᵀ) ⊙ l` | `l ∈ ℝ^{out}` |
| `adalora_delta` | `W + (α/r)·P diag(Λ) Q` | `P`, `Λ`, `Q`, with rank annealing |
| `dora_weight` | `m ⊙ (W + BA) / ‖W + BA‖_col` | `m ∈ ℝ^{out}` plus LoRA's `A`, `B` |
| `oft_weight` | `R W`, `R` block-diagonal orthogonal | skew-symmetric `Q` per block |

- **`adapters`** — the host-arithmetic definitions above, plus `LoraConfig`
  (`r`, `alpha`) and `dora_init_magnitude`.
- **`oft`** — `skew` and `cayley_orthogonal` (`R = (I − Q)(I + Q)⁻¹`), the
  Cayley transform that makes the block rotation orthogonal by construction.
- **`graph`** — the same updates as RLX graphs: `lora_delta_node`,
  `lora_delta_graph`, `ia3_graph`, `adalora_delta_graph`, `dora_graph`. These
  run on any backend the runtime has compiled in.
- **`ParamBudget` / `lora_param_count` / `lora_is_economical`** — trainable-vs-total
  accounting. "LoRA matches full fine-tuning" means nothing without the
  denominator, and `r(in + out)` stops saving anything as `r` approaches
  `min(in, out)`.

## Initialisation conventions

Two are reproduced rather than chosen, because getting them wrong is a common
source of "PEFT hurt my model" reports:

- **`B` starts at zero**, so `ΔW = 0` and the adapted model is identical to the
  base model at step 0. A nonzero `B` perturbs a pretrained network before any
  training happens.
- **IA3's `l` starts at one**, for the same reason.

## Features

- `cpu` *(default)* — CPU execution for the graph API.
- `metal` / `mlx` / `coreml` / `cuda` / `gpu` — forward the corresponding
  `rlx-runtime` backend feature.

## Install

```toml
[dependencies]
rlx-peft = "0.2"
```

## Quickstart

```rust
use rlx_peft::{lora_delta, LoraConfig};

// ΔW for one [out, in] layer, given the trained factors.
let delta = lora_delta(&a, &b, in_features, out_features, LoraConfig { r: 8, alpha: 16.0 })?;
```

## License

MIT OR Apache-2.0.
