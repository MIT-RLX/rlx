# rlx-check — `cargo rlx check`

Device-free static analysis of an rlx graph. It surfaces, at check time, the
diagnostics rlx already computes internally but only exposes through env-gated
`eprintln!` (`RLX_DISPATCH_REPORT`, `RLX_FUSION_REPORT`, `RLX_LINT_NUMERICS`) or
compile-time panics — folded into one structured report you can read or emit as
JSON.

## What it checks

| Axis | Source | Severity |
|------|--------|----------|
| **shape / dtype** | `rlx_ir::verify::verify_all` re-derives every node's shape and diffs it against the declared one | error |
| **backend dispatch** | per backend: ops that run native vs. portable common-IR vs. can't be lowered at all | note / **error** |
| **fusion** | patterns that *should* have collapsed but didn't, with the fix hint | warning |
| **numerics** | constant subgraphs that provably fold to NaN/Inf | warning |

Everything except execution legality is fully **device-free** — no GPU, no
driver. Op legality for the **CPU** reference backend is always available
(CPU is compiled in by default). Fusion coverage is reported for **every**
target (`cpu, metal, mlx, wgpu, cuda, rocm, tpu`) with no hardware, because it
runs entirely in the compiler.

To check another backend's *execution* legality (native / common-ir /
unsupported against its real op claim), compile that backend in — the claim is
read driver-free from the registry, so you still don't need the hardware, only
that backend's build deps:

```
cargo install --path crates/tooling/rlx-check --features metal,cuda
```

## Usage

```
cargo rlx check <GRAPH.json> [options]
cargo rlx check --demo <name>  [options]
```

Install the subcommand once with `just install-check` (or
`cargo install --path crates/tooling/rlx-check`); then `cargo rlx check` works
from any rlx crate. Without installing, run it through cargo:

```
just check-graph "--demo swiglu"
cargo run -p rlx-check --bin cargo-rlx -- rlx check --demo swiglu
```

### Options

- `<GRAPH.json>` — an rlx `Graph` serialized to JSON (serde).
- `--demo <name>` / `--list-demos` — built-in graphs, one per diagnostic class.
- `-b, --backend a,b` — analyze only these targets (`cpu,metal,mlx,wgpu,cuda,rocm,tpu`).
- `--all-backends` — analyze against every target.
- `--no-dispatch` / `--no-fusion` / `--no-numeric` — turn a check off.
- `--json` — structured report for editors / CI.
- `-q, --quiet` — one-line summary.

Exit code is non-zero when any **error**-level finding is present, so it drops
into a pre-commit hook or CI gate.

### Example

```
$ cargo rlx check --demo swiglu
rlx check — graph "swiglu" (7 nodes)

warning[missed-fusion]: matmul_bias_act fusion not applied (MultiConsumer)
  --> %4
  = help: single-consumer chain required — clone input or use HirOp::LinearFused

backends:
  cpu    ready    native=5 common-ir=0 rewritten=0 unsupported=0  fused=1 missed=1
  metal  legality n/a (build --features metal)           fused=1 missed=1
  ...
```

## As a library

The CLI is a thin shell over one pure function — `rlx_runtime::check::check_graph`,
re-exported here — so you can gate on the same findings from a model crate's
test or build:

```rust
let g = /* build or load your Graph */;
let report = rlx_check::check_graph(&g, &rlx_check::CheckOptions::default());
assert!(!report.has_errors(), "{}", report.render());
```

## `#[rlx_model(check)]` — automatic self-check

The checker lives in `rlx_runtime::check` precisely so the `#[rlx_model]` macro
can call it with no extra dependency. Opt a model in and every build of it runs
the checker on the traced graph:

```rust
#[rlx_model(check)]
fn my_encoder(t: &Tracer) -> Vec<TracedTensor> { /* … */ }
```

Findings print to stderr when the model is first built. Control at runtime with
`RLX_CHECK`: `off`/`0` (silent), `all` (every backend, always print), `strict`
(panic on any error-level finding). This is the "diagnostics in your normal dev
loop" integration; an editor LSP would call the same `check_graph`.
