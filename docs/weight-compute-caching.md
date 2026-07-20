# Computing weight-derived tensors once, not every forward

Many graphs contain work that depends **only on weights**, never on the
per-forward activation inputs — a transposed weight, a dequantized weight, a
BatchNorm scale folded into a conv filter, a RoPE cos/sin table, a merged
LoRA weight. Left alone, that work re-runs on **every** forward pass even though
its result never changes between forwards.

RLX removes that cost two complementary ways, depending on **when the weight
values become known**:

| | Flag | When weights are known | What happens | Per-forward cost |
|---|---|---|---|---|
| **Compile-time fold** | `CompileOptions::param_bindings` | at **compile** | weights baked to `Op::Constant`, the weight-only compute folded away into a single baked constant | **zero** |
| **Runtime hoist** | `CompileOptions::cache_param_invariant` / `RLX_CACHE_PARAM_INVARIANT=1` | only at **run time** | the weight-only closure is split into a *prepare* graph run **once**; its outputs feed the main graph across forwards | zero (CPU) / one weight-sized copy (CUDA) |

They are complementary, not competing: if `param_bindings` already folded the
weight-only compute away at compile time, the runtime hoist finds nothing left
to hoist and is a no-op. Turn on whichever fits how your weights arrive — or
both.

## Offline bake: `rlx-bake` → `*.rlx`

When you want a **merged** deploy artifact (graph + weights in one file), use
[`rlx-bake`](../crates/io/rlx-bake/). It specializes params, then applies
weight-aware opts (skip zero matmuls, pack exact ternary as TQ2_0, optional
Q8_0 quant), unfolds weights into an explicit table, and writes binary
`*.rlx` (magic `RLXBAKE1`, schema v2 = graph + weights). Optional cargo
feature `encrypt` (`--password` / `write_rlx_encrypted`) seals the **entire**
file (ChaCha20-Poly1305 + Argon2id, magic `RLXENC01`). Load with
`rlx_bake::read_rlx` or `read_rlx_with_password` and compile `file.graph` —
baked params need no `set_param`.

**Full walkthrough (MNIST train → bake → encrypt → run):**
[rlx-bake.md](rlx-bake.md).

```bash
cargo run -p rlx-bake -- path/to/bundle -o model.rlx
cargo run -p rlx-bake --features encrypt -- path/to/bundle -o model.rlx --password-env RLX_BAKE_PASSWORD
# or: graph.json --weights weights.safetensors -o model.rlx [--quant]
```

## Compile-time: `param_bindings`

When you know the weight *values* at compile time, hand them to the compiler.
`specialize_params` rewrites each named `Op::Param` into an `Op::Constant`
**before** fusion, and `ConstantFolding` then evaluates every pure weight-only
subgraph — including **broadcasting** ops like a per-channel `Mul(w[C,I,H,W],
scale[C,1,1,1])` — down to one baked constant. Nothing weight-derived survives
into the run loop.

```rust
use std::collections::HashMap;
use rlx_runtime::{CompileOptions, Session, Device};

let mut bindings = HashMap::new();
bindings.insert("conv.weight".to_string(), weight_data);
bindings.insert("bn.scale".to_string(),   scale_data);

let mut opts = CompileOptions::new();
opts.param_bindings = Some(bindings);

let compiled = Session::new(Device::Cuda).compile_with(graph, &opts);
// `Mul(weight, scale)` etc. are gone — folded into a constant weight.
```

Cost: the weights must be resident at compile, and any weight change means
re-compiling (re-specializing). Best for deploy-time / AOT paths where the
weights are fixed.

## Run time: `cache_param_invariant`

When weights are only known at run time (you `set_param` after `compile`),
enable hoisting instead. The graph is split into two:

- a **prepare** graph — the *param-invariant closure*: every node reachable only
  from `Op::Param`/`Op::Constant`, never from an `Op::Input` (RNG/`Sample` ops
  count as dynamic, so they are never hoisted). Its outputs are the *boundary*
  tensors that the rest of the graph consumes.
- a **main** graph — everything else, with each boundary tensor turned into a
  named input.

The runtime runs `prepare` **once** (lazily, on the first forward) and injects
its outputs into `main`. The public `CompiledGraph` API is unchanged — routing
`set_param` to the right half and running `prepare` once are transparent.

```rust
use rlx_runtime::{CompileOptions, Session, Device};

let mut opts = CompileOptions::new();
opts.cache_param_invariant = true;          // or RLX_CACHE_PARAM_INVARIANT=1

let mut compiled = Session::new(Device::Cuda).compile_with(graph, &opts);
compiled.set_param("w", &w);                // routed to prepare/main by owner
compiled.set_param("scale", &scale);
let y0 = compiled.run(&[("x", &x0)]);       // prepare runs here, once
let y1 = compiled.run(&[("x", &x1)]);       // reuses the prepared tensors
```

How the prepared tensors reach `main` depends on the backend:

- **CPU** binds them as *persistent handles* (`bind_handle`) — zero copy per
  forward.
- **CUDA** (and any backend without persistent handles) uses a **feed fallback**
  — the prepared tensors are fed as ordinary inputs each forward. The expensive
  *compute* still happens once; only a weight-sized host→device copy remains.

**Weight updates re-prepare.** Calling `set_param` on a weight that feeds the
prepare graph invalidates the cache, so the next forward recomputes it — correct
for weight reloads and training loops.

### Scope

Hoisting runs **pre-fusion**, so it catches weight-only compute present in the
source graph — weight transposes, dequant of constant weights, RoPE tables, LoRA
weight merges, host-pre-folded BatchNorm affines expressed as weight ops, etc.
It does **not** catch weight-only tensors that a *fusion pass* creates later
(e.g. the `Mul(w, scale)` that [`FuseConvAffineAct`](../crates/core/rlx-fusion/src/fusion/conv_bias_act.rs)
emits when folding a BatchNorm scale into a conv filter) — those are handled by
`param_bindings` above, or stay as an ordinary per-forward op.

## Where it lives

- `rlx_compile::split_param_invariant` (`crates/core/rlx-compile/src/param_hoist.rs`)
  — the pre-fusion closure analysis + graph split.
- `rlx_compile::specialize_params` + `ConstantFolding`
  (`crates/core/rlx-compile/src/{param_specialize,const_fold}.rs`) — the
  compile-time fold; `const_fold`'s evaluator does NumPy broadcasting so
  per-channel weight math folds.
- [`rlx-bake`](../crates/io/rlx-bake/) — offline merge of graph + weights into
  `*.rlx`, with skip / ternary / quant weight-aware passes.
- `CompiledGraph` staging (`crates/core/rlx-runtime/src/compiled.rs`) — the
  transparent prepare-once + bind/feed injection.

Tests: `crates/core/rlx-runtime/tests/param_hoist.rs` (CPU) and
`cuda_param_hoist.rs` (CUDA); `crates/io/rlx-bake/tests/bake_roundtrip.rs`.
