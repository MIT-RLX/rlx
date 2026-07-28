# Debugging NaN / Inf — compiler & IR level

When a downstream model or developer produces a `NaN` or `Inf`, a raw bad
float tells you *what* went wrong but not *where* or *why*. RLX localizes it at
the compiler/IR level, where there's enough structure to name the exact op,
say whether that op **created** the bad value or merely **inherited** one, and
suggest a fix.

There are two entry points — a **static** one that runs at compile time and a
**dynamic** one that runs at execution time. Both are off by default (zero
production cost) and enabled by an environment variable.

| | Env var | When it runs | Catches | Cost when on |
|---|---|---|---|---|
| **Static lint** | `RLX_LINT_NUMERICS=1` | compile time | *provable* non-finite constants (e.g. a folded `1/0`) — no input data needed | negligible (one const-eval walk) |
| **Runtime localizer** | `RLX_DEBUG_NANS=1` | every op, execution time | data-dependent NaN/Inf (softmax overflow, variance underflow, div-by-zero on a real tensor) | O(n) scan per op buffer |

`RLX_DEBUG_NANS=abort` additionally **fails fast** — it panics on the first bad
value (JAX `jax_debug_nans` style) so a stack trace points at the call site.

## Quick start

```bash
# Compile-time: report constants that fold to NaN/Inf, with provenance.
RLX_LINT_NUMERICS=1 rlx-run qwen3 ...

# Runtime: on the first bad op, print who produced it and how to fix it.
RLX_DEBUG_NANS=1 rlx-run qwen3 ...

# Runtime, fail-fast: panic on the first NaN so you get a backtrace.
RLX_DEBUG_NANS=abort rlx-run qwen3 ...
```

## What you get

### Runtime localizer (`RLX_DEBUG_NANS`)

```
rlx nan-check: NaN at index 0 of %137 Rsqrt "layer3.attn.rms_norm"
  → inputs finite, this op produced it
  fix: rsqrt/sqrt of a negative or zero — norm variance underflow; raise eps or clamp input ≥ 0
```

The two lines under the location are the point:

* **culprit vs propagator** — `check_node` compares the op's *inputs* against
  its *output*. If the inputs were finite and the output isn't, **this op
  created the NaN** (a real bug site). If an input was already non-finite, the
  message instead reads `→ propagated: input %NN was already non-finite (look
  upstream)` and omits the fix — the real bug is wherever that input was first
  flagged. Because the executor scans in **topological order**, the *first*
  node that trips is the origin, not the hundreds of downstream ops that
  inherit the NaN.
* **fix hint** — a one-line remedy keyed off the op kind (unstable softmax →
  subtract row-max; div-by-zero → guard the denominator; log of ≤0 → clamp;
  etc.).

### Static lint (`RLX_LINT_NUMERICS`)

```
rlx numeric-lint: %2 +inf "%2": constant subgraph folds to a non-finite value
  fix: division by zero — guard the denominator with eps, or mask with where(denom != 0, …)
```

This catches the class of NaN sources that **don't depend on input data** —
the compiler can prove them wrong before anything runs. It reuses the
constant-folding evaluator to walk constant-input subgraphs and reports any op
whose value is non-finite (a `1/0`, `log(0)`, `sqrt(-1)`, a literal NaN/Inf
`Constant`, …). These are **zero false positives**: an unguarded `Div` on a
*runtime* tensor is only *maybe* a bug and is deliberately **not** flagged, so
the report stays signal-dense.

## Per-backend coverage

All backends honor `RLX_DEBUG_NANS` (off by default → zero production cost). How
much they can localize depends on their execution model — a host-iterated or
host-visible backend can pin the *internal* culprit op; an opaque whole-graph
dispatch can only scan the output boundary. When only the output can be scanned,
**replay the same graph on the CPU backend** for internal localization —
provenance is backend-neutral, so the op names match.

| Backend | Coverage | How |
|---|---|---|
| **CPU** | **Full per-op internal** — culprit vs propagator + fix hint | executor epilogue, topo order |
| **wgpu** | **Full per-node internal** — reads every node back from the arena post-run | `RLX_DEBUG_NANS` (or legacy `RLX_WGPU_NAN_TRACE`) |
| **CoreML** | **Per-op for host segments** + output boundary | host-segment scan + output scan (MIL segments are opaque) |
| **Vulkan** | Output boundary (arena is HOST_COHERENT) | `finish_run` |
| **Metal** | Output boundary (MPSGraph is opaque) | `run_read_outputs` |
| **MLX** | Output boundary (C++ runtime is opaque) | `run_read_outputs_inner` |
| **CUDA** | Output boundary (per-op D2H would perturb timing) | `run_read_outputs` (full-output reads) |
| **ROCm** | Output boundary | `run_read_outputs` (full-output reads) |
| **TPU** | Not wired — opaque PJRT/HLO with no retained IR graph; use CPU replay | — |

`abort` mode (`RLX_DEBUG_NANS=abort`) applies everywhere the scan runs.

## Provenance — naming *your* op

Both tools label the node via `rlx_ir::provenance::node_label`, which resolves
to (in order): a leaf's own declared name (`Input`/`Param` → e.g.
`qkv.weight`), the node's cross-stage origin label, its HIR block, its name, or
finally its IR id (`%137`).

**Imported models are named end-to-end.** When lowering one source op, each
importer records the HIR node range it produces and stamps the source name
onto *all* of them via `HirModule::label_nodes_since` — so a multi-op lowering
(a Softmax that becomes sub/exp/reduce/div, a fused attention, a layer-norm)
has every intermediate carry the source identity, not a generic `"mir"`. The
HIR→MIR lowering (`tag_hir_subgraph`) then propagates that label to every
derived MIR node, so a NaN in any of them localizes back to the user's op:

* **ONNX** (`rlx-onnx-import`, `lower/ops/mod.rs`) — prefers the ONNX node name,
  falling back to the first output tensor name (`/…/Softmax_output_0`), which
  is present and descriptive even when the node name is empty.
* **torch** (`rlx-torch-import`, `hir_build.rs`) — uses the instruction's result
  value name (the FX/aten output).

A hand-built graph with no metadata still shows `%137`; that's expected — the
provenance is only as rich as what the front-end supplies.

## Where it lives

| Piece | Location | Notes |
|---|---|---|
| Localizer + scanner | `rlx-ir/src/numeric_check.rs` | `check_node`, `first_bad`, `fix_hint`, `NanReport`, and the `DebugScanner` (env policy + print/abort). In **rlx-ir** so *every* backend can reach it (backends depend on rlx-ir, not rlx-runtime). |
| CPU wiring (reference) | `rlx-cpu/src/executor.rs` | `scan_node_for_nans` per-op epilogue. |
| Backend wiring | `rlx-<backend>/…` | one `DebugScanner` at each run loop / output readback — see the coverage table. |
| Static lint | `rlx-compile/src/numeric_lint.rs` | `lint_numerics(&Graph) -> Vec<NumericLint>`; wired behind `RLX_LINT_NUMERICS` in `compiler.rs`. |

The localizer is backend-agnostic: it takes already-materialized `f32` host
slices plus the `Graph`. The shared `DebugScanner` centralizes the
`RLX_DEBUG_NANS` policy so each backend adds only a few lines.

## Extending to another backend

Build one scanner before the run loop, then hand each computed node (or the
outputs) its buffers:

```rust
let scanner = rlx_ir::numeric_check::DebugScanner::from_env("mybackend");
// per-op (host-iterated backends):
if scanner.enabled() {
    scanner.check(graph, node_id, out, &inputs); // prints + aborts per policy
}
// …or, for opaque whole-graph backends, at the end:
scanner.check_outputs(graph, &outputs);
```

`check` classifies culprit vs propagator from the operand slices you pass;
`check_outputs` scans the graph outputs positionally. Both are no-ops when
`RLX_DEBUG_NANS` is unset. See `rlx-cpu/src/executor.rs::scan_node_for_nans`
for the reference per-op gather (F32-only, skips buffers the backend doesn't
own) and `rlx-wgpu` for a full post-run per-node scan.

## Adding a fix hint for a new op

Extend the match in `rlx_ir::numeric_check::fix_hint`. Keep hints to a single
actionable sentence; ops with no single obvious remedy return `None` (the
location + culprit flag still localize them).
## License

MIT OR Apache-2.0.
