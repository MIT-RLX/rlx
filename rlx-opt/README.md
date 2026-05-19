# rlx-opt

Graph rewrites + autodiff for `rlx-ir`. Stateless passes (every pass
takes `&mut Graph` and mutates in place), JAX-shaped transforms, opt-in
legalization.

## Default pass pipeline

Every backend runs roughly:

1. **`ConstantFolding`** — fold compile-time-known subgraphs.
2. **`fusion::*`** — `MatMul + bias + Activation`, residual + LayerNorm,
   QKV concat, SwiGLU, attention block, BERT layer.
3. **`MarkElementwiseRegions`** — collapse element-wise chains into a
   single region op (one kernel per chain).
4. **`legalize_for_backend`** — reject ops the target backend can't lower,
   so missing op coverage fails at compile time instead of runtime.
5. **`memory::*`** — liveness analysis → arena buffer assignment.
6. **`dce`** — dead-code elimination. Always last.

## Opt-in passes

Run by specific backends or user code:

- **`LegalizeBroadcast`** — materialize non-trailing broadcasts via
  `Op::Expand`. Required for TPU (HLO) and rlx-cortexm; CPU/Metal handle
  modulo broadcasts inline.
- **`insert_q_dq`** — post-training quantization Q/DQ insertion. Caller
  supplies a `CalibrationRecord`.
- **`LowerControlFlow`** / **`LowerDotGeneral`** — lower XLA-shaped
  primitives to the standard op set.

## Transforms (JAX-shaped)

- **`autodiff::grad_with_loss`** — reverse-mode AD. Phases 1–9 cover
  every non-fused op + `If` / `While` / `Scan` / `SelectiveScan` / fused
  attention / fused transformer layer.
- **`jvp` / `hvp`** — forward-mode AD; Hessian-vector products via
  forward-over-reverse.
- **`vmap`** — batched function transform (leading-axis batching).
- **`Op::CustomFn`** — `custom_vjp` / `custom_jvp`-style overrides for
  user-defined sub-graphs.

## What's here

- `pass.rs` — `Pass` trait + the canonical pipeline order.
- `dce.rs` — dead-code elimination.
- `const_fold.rs` — fold constant subgraphs at compile time.
- `fusion.rs` — pattern-match fusion. Patterns live next to the op they
  produce.
- `precision.rs` — auto-mixed precision policy (f32 → f16/bf16 around
  matmul; cast tax handled in Phase G).
- `memory.rs` — liveness analysis + arena assignment. Output is an
  offset-per-node map.
- `autodiff.rs` — reverse-mode AD.
- `lower_dot_general.rs` — XLA-style DotGeneral → MatMul + reshapes.

## Install

```toml
[dependencies]
rlx-opt = "0.1"
```

Most users want the [`rlx`](https://crates.io/crates/rlx) prelude crate;
it re-exports `rlx_opt` as `rlx::opt`.

## Build / test

```sh
cargo build -p rlx-opt
cargo test  -p rlx-opt       # ~17 tests, fusion-heavy
```

## Gotchas

- Pass order matters. `const_fold` must run before `fusion` (fusion
  patterns assume constant inputs are already inlined). `memory.rs`
  must run last; it depends on the final node count.
- Don't introduce a new fused op without also: adding it to `Op`,
  shape inference, both backends, cost model, **and** the verifier.
- `precision.rs` inserts `Cast` nodes; some are eliminated by the cast-
  elision peephole in fusion. Don't double-handle.
- Fusion is conservative — it only fires when `num_consumers == 1` for
  intermediate nodes. Multi-consumer ops stay unfused on purpose.

## License

GPL-3.0-only.
