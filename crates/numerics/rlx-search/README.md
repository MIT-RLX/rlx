# rlx-search

Host-side sequence search shared by every RLX decoder: the code between a
model's score rows and its tokens.

| Module | What |
|---|---|
| `select` | `argmax` (lane-parallel), `top_k`, `log_softmax`, `log_sum_exp`; ties go to the lowest index |
| `beam` | Beam search as a state machine (`BeamSearch::advance`) or a loop (`beam_search`); Hugging Face and `openai/whisper` rules as presets; `Constraint` masks |
| `ctc` | Greedy, streaming prefix beam (`PrefixBeam`), CTC trellis `forward::<Log \| Tropical>` |
| `align` | Monotonic Viterbi forced alignment |
| `semiring` | `Log` (sum over paths) and `Tropical` (best path) |

```rust
use rlx_search::{BeamConfig, beam_search};

let cfg = BeamConfig::new(4, 64).stop([eos]);
let best = beam_search::<anyhow::Error>(cfg, &prompt, &first_logits, &mut (), |step| {
    // Reorder per-hypothesis state by `step.parents`, feed `step.tokens`,
    // return one logits row per hypothesis.
    model.decode(step.parents, step.tokens)
})?;
```

No dependencies. `rlx-runtime` re-exports it as `rlx_runtime::search`.

License: MIT OR Apache-2.0.
