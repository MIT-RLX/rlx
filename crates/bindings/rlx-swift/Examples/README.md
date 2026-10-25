# Swift examples

Build the C library, then run any example by its product name:

```bash
cargo build -p rlx-capi                       # → target/debug/librlx_capi.*
swift run --package-path crates/bindings/rlx-swift rlx-example-inference
swift run --package-path crates/bindings/rlx-swift rlx-example-training metal
```

The same six examples exist for Python and JS, printing the same numbers
([`crates/bindings/README.md`](../../README.md)); `cargo test -p rlx-capi` runs
these and checks their output.

| Product | Source | What it shows |
|---------|--------|---------------|
| `rlx-example-inference` | [`Inference/main.swift`](Inference/main.swift) | build a classifier, compile for a device, set weights, run; a typed (byte) run |
| `rlx-example-training` | [`Training/main.swift`](Training/main.swift) | `Trainer.run` with an LR schedule and early stop; GGUF checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| `rlx-example-loading` | [`Loading/main.swift`](Loading/main.swift) | `GgufWriter` / `Gguf` (f32 + Q8_0), safetensors into a graph, quantize; `RLX_TOKENIZER=…` adds the tokenizer |
| `rlx-example-distributed` | [`Distributed/main.swift`](Distributed/main.swift) | three ranks on threads: collectives, pipelined inference, data-parallel training with identical replicas |
| `rlx-example-backends` | [`Backends/main.swift`](Backends/main.swift) | devices, cost-model pick, `FlexibleSession`, `Runner` benchmark / fallback, `Router` |
| `rlx-example-models` | [`Models/main.swift`](Models/main.swift) | save a graph as an `.rlxp` model, run it by name with `Model`; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

In an app, build `RLX.xcframework` (`ios/build-xcframework.sh --capi --apple`)
and depend on the package with `RLX_XCFRAMEWORK=1`; see the package
[README](../README.md).

License: MIT OR Apache-2.0.
