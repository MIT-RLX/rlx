# Contributing to RLX

Thanks for your interest in RLX. This file is the short version; the deeper
guide is [`docs/development.md`](docs/development.md), the agent-facing
conventions live in [`AGENTS.md`](AGENTS.md), and every crate has its own
`README.md` documenting its public surface, build commands, and gotchas.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting the code

RLX vendors MLX as a git submodule, so clone with it:

```sh
git clone --recurse-submodules https://github.com/MIT-RLX/rlx
# already cloned without it? pull the submodule in place:
git submodule update --init --recursive
```

You need a Rust toolchain that satisfies the workspace MSRV — **1.87**, edition
2024 (see `rust-version` in [`Cargo.toml`](Cargo.toml)) — plus
[`just`](https://github.com/casey/just) for the task recipes. Install the
pre-commit hook so formatting and lint run before every commit:

```sh
just install-git-hooks   # cargo fmt --all (re-stages fixed files) + clippy
```

Backends are feature-gated and auto-detected — you build only what your hardware
supports. On Apple Silicon the primary path is `cpu` + `metal` (optional `mlx`);
CUDA/ROCm/TPU/wgpu/Vulkan and the Cortex-M / FPGA products live behind their own
features. See [`AGENTS.md`](AGENTS.md#backend-notes) and each backend crate's
`README.md` for the specifics.

## Ground rules

- **Per-backend peak performance is the north star.** Each *(backend × hardware)*
  combination should be the fastest, most precise path on that hardware.
  Correctness, tests, macros, and abstractions exist to protect that — not as
  ends in themselves. A change that regresses a backend's hot path needs a
  strong reason.
- **Core stays application-agnostic.** JAX-shaped primitives belong in core;
  concrete models and domain code live downstream in
  [`rlx-models`](https://github.com/MIT-RLX/rlx-models) and other sibling repos.
- **Cross-platform, one command.** Backend tooling must work on macOS / Linux /
  Windows with auto-detection; prefer Python or Rust over shell, and a single
  canonical command per task.
- **Minimal diffs.** Match the surrounding style; do not refactor unrelated code
  in the same change.

## Dev loop

- **Local gate:** `just ci` is the full build, workspace tests, `fmt-check`,
  lint, and pyrlx pytest suite. It spans the GPU / ROCm / wasm matrix, so run
  the subset your hardware supports (see [What CI expects](#what-ci-expects)).
- **Format & lint:** `just fmt` / `just lint` (clippy with `-D warnings`). The
  pre-commit hook and `just ci` run these for you.
- **Static graph check:** `cargo rlx check` folds shape/dtype verification,
  backend dispatch, missed-fusion hints, and the provable-NaN/Inf lint into one
  report (no GPU required for the CPU-legality, fusion, shape, and numeric
  passes).
- **Bench every change.** Note the before/after bench delta in the PR —
  even "within noise" is data worth recording.
- **Gate benches on thermal throttle** (`just throttle`, or `RLX_ALLOW_THROTTLE=1`
  for one-offs); silent 10× slowdowns under thermal pressure are a real failure
  mode on laptops.

## Adding an `Op`

New ops touch the whole stack. At minimum: `rlx-ir` (`op.rs`, `infer.rs`,
`graph.rs`, `verify.rs`), every backend's thunks + cost models (sister-crate
ports are usually mechanical), the optimizer's fusion patterns, and an autodiff
VJP/JVP with a finite-difference guard. Run with `RLX_DISPATCH_REPORT=1` after
compile to confirm the kind lands **native** rather than **common-ir**, and add
a cross-backend parity test. `just new-op MyOp` prints the full checklist;
`just gen-op-coverage` refreshes [`docs/op-coverage.md`](docs/op-coverage.md).

To extend RLX **without editing core** — a new model, numerics crate, or
research op — use the stable seams in
[`rlx-extend`](crates/core/rlx-extend/README.md) (`LayerStage` block seam +
`OpExtension` op seam) rather than reaching into internal crates.

## Testing & parity

- New backend work needs a **cross-backend parity test** against the CPU
  reference — a numeric match, not a shape-or-compile check. Prefer meaningful
  parity/integration coverage over trivial asserts.
- `just test` runs the workspace suite; `just test-gpu` adds the host GPU
  backends and runtime feature tests. Backend-specific recipes (`just test-mlx`,
  `just test-rocm`, `just test-fpga`, …) live in the [`Justfile`](Justfile).

## Before you start

For anything beyond a small, self-contained change, **open an issue first** to
agree on the approach — it spares you building something that gets rejected, and
spares us reviewing it. Small changes (a typo, a doc fix, a single-backend bug
fix with a test) can go straight to a PR.

## PRs

Open PRs against `main`. The
[pull request template](.github/pull_request_template.md) **is** the canonical
checklist — fill it out; this guide does not restate it. Keep commits scoped,
describe the motivation (which issue / plan item, what it speeds up or fixes),
and add a `CHANGELOG.md` entry under `[Unreleased]` for any user-visible change.

Not every checklist item applies to every change: the parity, native-dispatch,
and bench rows are for op / backend work; a docs, tooling, or CI PR should strike
through what doesn't apply rather than leaving it unchecked.

### What CI expects

`just ci` is the full gate, but it includes GPU / ROCm / wasm steps most
contributors can't run locally. Run what your hardware supports — at minimum
`just fmt`, `just lint`, and `just test` — and let the project's CI cover the
rest of the matrix. A red check on a path you couldn't run locally is expected;
note it in the PR and a maintainer will help validate on the right hardware.

### Review expectations

Maintainers aim to triage new PRs within a few business days. Expect at least one
review pass before merge; larger or backend-spanning changes may take multiple
rounds and validation on hardware you don't have. If a PR goes quiet, a polite
nudge after about a week is welcome.

**Vocabulary.** A short list of words is avoided in RLX-authored text — code,
comments, docs, and identifiers alike. See the *Forbidden vocabulary* section of
[`AGENTS.md`](AGENTS.md#forbidden-vocabulary) for the list and preferred
replacements; if a contribution trips it, a maintainer will suggest a reword —
it won't block a merge on its own.

## Reporting bugs & security

- **Bugs / feature requests:** open a GitHub issue with the version or commit,
  the platform and backend (CPU / Metal / CUDA / …), and a minimal reproducing
  graph or input.
- **Vulnerabilities:** do **not** open a public issue — follow
  [`SECURITY.md`](SECURITY.md) and file a private advisory instead.

## Licensing

**AI use and ownership.** Using AI tools to help write code, docs, or tests is
welcome. Regardless of the tooling, you are responsible for what you submit and
must have the right to contribute it under the licenses below — the same standard
that applies to hand-written work.

By contributing you agree that your contributions are dual-licensed under the
project's [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE) licenses. Unless
you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
