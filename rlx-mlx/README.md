# rlx-mlx

Apple MLX backend for RLX — vendored MLX via a hand-rolled C++ shim,
eager + lazy + compiled execution.

## Modes

- **Lazy** *(default)* — build the entire MLX graph in `run()`, then
  call `mlx::core::eval` once on all outputs. Lets MLX's optimizer
  schedule the whole DAG, equivalent in spirit to the `mps_graph`
  path in rlx-metal.
- **Eager** — eval after every op. Slower; useful for debugging
  because failures surface at the offending op rather than at the
  final eval.
- **Compiled** — `mlx::compile`-built persistent function for repeated
  shapes; trace-cache amortizes re-runs.

Mode is set per-compile via `MlxExecutable::compile_with_mode`, or
globally via `RLX_MLX_MODE=eager|lazy|compiled` (default lazy).

## What's here

- `cpp/rlx_mlx_shim.{h,cpp}` — C ABI over `mlx::core::*`. Catches every
  C++ exception and surfaces it through thread-local `last_error`.
- `build.rs` — drives MLX's CMake (vendored at `../vendor/mlx`) into a
  static `libmlx.a`, then `cc::Build`-compiles the shim and links them.
- `src/ffi.rs` — raw `extern "C"` declarations matching the header.
- `src/array.rs` — RAII `Array` wrapper, `MlxError`, top-level `eval`.
- `src/ops.rs` — typed wrappers: matmul / add / mul / sub / div /
  softmax / gelu / silu / cast / layer_norm.
- `src/lower.rs` — walks `rlx_ir::Graph` in topo order, building MLX
  arrays for each node. Rebuilds the graph fresh each `run()` (see the
  comment in lower.rs for why).
- `src/backend.rs` — `MlxExecutable` (set_param / run / handles).
- Tier-1 / Tier-2 / Tier-3 backward op parity with `rlx-cpu` for
  reverse-mode autodiff (relu, activation, softmax cross-entropy, layer
  norm, conv2d, max-pool, fake-quantize).

## Install

> **Not on crates.io for 0.1.0.** `build.rs` reads `../vendor/mlx`,
> which sits outside the rlx-mlx crate boundary and isn't included in
> a `cargo publish` tarball. Until the submodule is relocated under
> `rlx-mlx/vendor/mlx/` or the build script learns to fetch MLX at
> build time, depend on it via the workspace git tree:

```toml
[dependencies]
rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["mlx"] }
# or directly:
rlx-mlx = { git = "https://github.com/MIT-RLX/rlx" }
```

`vendor/mlx` is a git submodule — initialize after clone:

```sh
git submodule update --init
```

The first build compiles MLX from source — minutes, not seconds.

## Build / test

```sh
cargo build -p rlx-mlx --release
cargo test  -p rlx-mlx --release
```

Through `rlx-runtime`:

```sh
cargo build -p rlx-runtime --features mlx --release
```

## Status

Mature on Apple Silicon (M1 / M2 / M3 / M4). On Intel Macs MLX falls
back to its CPU path; supported but rarely the right choice.

## Gotchas

- **Op coverage.** First cut handled MatMul, Binary (Add/Mul/Sub/Div),
  Activation (Gelu/Silu), Cast, Softmax, LayerNorm. Now covers matmul,
  all binary / activation / cast / reduce / softmax / layer-norm /
  RMS-norm, fused attention (SDPA via
  `fast::scaled_dot_product_attention`), pool composition, dot-general,
  selective-scan unroll, calibrated cost model, async commit + sync.
  Anything else returns `MlxError("unsupported op …")` from
  `lower::lower_and_run`. Adding an op means: an entry in `cpp/shim.h`,
  the matching impl in `shim.cpp`, an `extern "C"` decl in `ffi.rs`,
  a wrapper in `ops.rs`, and a match arm in `lower.rs`.
- **Fresh-graph-per-run.** Every `run()` rebuilds the MLX graph from
  scratch. MLX's own trace cache amortizes this, but if you need lower
  per-run latency, the next step is `mlx::compile`-style placeholder
  bindings (track the input/param NodeIds → MLX placeholder handles,
  reuse the compiled graph across runs).
- **F32 I/O default.** Inputs/params come in as `&[f32]` and outputs come
  out as `Vec<f32>`. The shim casts to/from MLX's per-array dtype
  internally (so AutoMixedPrecision still does the right thing inside
  the graph). The runtime trait now exposes
  `set_param_typed(name, &[u8], dtype)` and
  `run_typed(inputs: &[(&str, &[u8], DType)]) -> Vec<(Vec<u8>, DType)>`;
  default impls handle F32 only; the MLX backend overrides with the
  zero-widen path through `Array::from_bytes` / `Array::to_bytes`. CPU
  and Metal inherit the F32 default — they panic for non-F32 typed
  inputs (override is a future PR for those backends).
- **Constants must be F32.** Non-F32 `Op::Constant` payloads error in
  lower.rs — the constant byte format is little-endian f32. Add F16/I32
  constant decoding when a model needs it.
- **Async pipeline:** `commit_no_wait` schedules the lowered graph via
  `mlx::core::async_eval` and stashes the output handles; `sync_pending`
  calls `mlx::core::synchronize` and drops them. `run()` always calls
  `sync_pending()` first, so an explicit run() after a commit is safe.
  No per-stream isolation yet — synchronize() drains every MLX stream.
- **KV-cache pattern:** if an output slot's name is `out{i}` and a
  handle of the same name is bound, `run()` syncs the f32 result back
  into the handle so the next iteration picks it up as input.
- **`run_slots` arena:** the slot path keeps a synthetic `Vec<u8>`
  arena owned by the executable. Outputs are copied into it after each
  `run_slots` call so callers can read results via
  `arena_ptr().add(offset)` without per-output `Vec<f32>` allocations.
  Cheaper than `run()` when output sizes are tiny but the per-call
  bookkeeping cost matters.
- **Attention `SlidingWindow` mask:** synthesized host-side as an
  additive `[seq_q, seq_k]` mask (0 where allowed, -inf elsewhere),
  then passed through `fast::scaled_dot_product_attention` with
  `mode="array"`. MLX has no native sliding-window mode.
- **Sample:** temperature scaling + `top_k` filter + `top_p`
  (nucleus) filter + `mlx::random::categorical`. top_k uses `mc::topk`
  for the threshold; top_p sorts descending (via `sort` + negate),
  takes an exclusive cumsum of the sorted probs, masks entries whose
  cumsum < top_p, picks the smallest probability still in that
  nucleus as the threshold, and applies it back to the original
  logits via `where(p >= threshold, logits, -∞)`.
- **Persistent compiled graph (`MlxMode::Compiled`):** the executable
  builds a `CompiledFn` lazily on first `run()`. Internally a Rust
  callback walks the IR via `lower::lower_with_env`; the shim wraps it
  as `std::function`, hands it to `mc::compile`, and stores the
  returned function. Subsequent calls replay the optimized trace.
- **Calibration + cost model:** `calibrate::Calibration::load_or_measure()`
  measures sgemm GF/s at one large + one small shape plus a tiny-graph
  round-trip overhead, **plus** memory bandwidth (large contiguous
  copy), attention throughput (1×4×128×64 SDPA), and reduce throughput
  (1024×1024 sum-along-last-axis). Caches at
  `~/.cache/rlx/mlx-calib-<sanitized-device-name>.json` and feeds
  `rlx_runtime::cost::MlxCostModel` so `pick_best_device` can rank MLX
  honestly.
- **Pool composition:** `Op::Pool` is lowered by composing
  `slice_strided` over the kernel grid plus a reduction.
  Supports 1D / 2D / 3D inputs (channels-first layout) and all five
  reduction kinds (max/min/sum/mean/prod). Constant-pad with -∞ for
  max-pool, +∞ for min-pool, 1.0 for prod, 0 elsewhere.
- **DotGeneral lowering:** the canonical 2D pattern (no batch dims,
  contract `lhs[1]` × `rhs[0]`) reduces to a plain `MatMul`, matching
  what the optimizer's `LowerDotGeneral` pass would have produced.
  Non-canonical patterns (batched, alternative contracting axes) error
  with a clear diagnostic — same coverage as the optimizer pass.
- **FusedTransformerLayer composition:** the full BERT-style post-norm
  block (attention → residual+LN → FFN → residual+LN) composed from
  primitives. Honors all four mask kinds via the underlying SDPA path.
- **`Op::If` / `Op::While`** are now lowered. We adopt a positional
  binding convention between the sub-graph's `Op::Input` nodes (in
  topo order) and the parent's captures (`inputs[1..]` for `If`,
  `inputs[..]` for `While`); sub-graph `Op::Param` nodes look up by
  name in the parent's param maps; sub-graph `Op::Constant` nodes are
  inline. `Op::If` evaluates both branches and combines via
  `mc::where`. `Op::While` requires `max_iterations` and unrolls; an
  active-mask gate via `where(active && cond, body_out, carried)`
  freezes loop-carried values once the condition becomes false. Single-
  output `While` only — multi-output convention isn't defined in the
  IR. Compile mode (`MlxMode::Compiled`) doesn't yet recurse through
  sub-graph leaves; `If`/`While` inside a compiled trace will fail
  with a missing-param diagnostic. Use `Lazy`/`Eager` for control flow.
- **SelectiveScan composition:** `Op::SelectiveScan` (Mamba SSM step)
  is lowered by unrolling the time loop into seq many op chains.
  At each t we slice δ/x/B/C, broadcast against A, update the
  running state via `exp(δA) * state + δ*B*x`, and accumulate
  `sum_n(C * state)` as the output. Per-call cost amortizes through
  `mlx::compile`'s trace cache. Acceptable for static-shape graphs
  (which all our graphs are); for very long sequences a custom Metal
  kernel via `fast::metal_kernel` would beat this on raw throughput.
- **Native ElementwiseRegion lowering (PLAN L2):** `Op::ElementwiseRegion`
  is lowered in `lower.rs` by composing `ops::*` per `ChainStep`
  (Activation/Cast/Binary/Compare) directly into MLX's lazy trace.
  Each step is resolved positionally — `ChainOperand::Input(i)` reads
  `node.inputs[i]` and `ChainOperand::Step(i)` reads the array
  produced by chain step `i`. Because the whole chain becomes a sub-DAG
  inside MLX's trace, `mlx::compile` and the lazy evaluator get to
  fuse it into a single kernel — no decomposer round-trip and no
  extra Op nodes for the executor to walk. The runtime backend now
  runs `MarkElementwiseRegions` (instead of `UnfuseElementwiseRegions`)
  ahead of MLX compilation so chains are collapsed before lowering.

## License

GPL-3.0-only.
