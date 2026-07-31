# Contributing to RLX

Thanks for your interest in RLX. This file is the short version; the deeper
guide is [`docs/development.md`](docs/development.md), and every crate has its
own `README.md` documenting its public surface, build commands, and gotchas.

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

## Dev loop

- **Fast local gate:** `just ci` — build, workspace tests, lint, and the pyrlx
  pytest suite. Run it before opening a PR.
- **Static graph check:** `cargo rlx check` folds shape/dtype verification,
  backend dispatch, missed-fusion hints, and the provable-NaN/Inf lint into one
  report (no GPU required for the CPU-legality, fusion, shape, and numeric
  passes).
- **Bench every change.** Note the before/after bench delta in the PR —
  even "within noise" is data worth recording.
- **Gate benches on thermal throttle** (`scripts/check-throttle.sh`); silent
  10× slowdowns under thermal pressure are a real failure mode on laptops.

## Adding an `Op`

New ops touch the whole stack. At minimum: `rlx-ir` (`op.rs`, `infer.rs`,
`graph.rs`, `verify.rs`), every backend's thunks + cost models (sister-crate
ports are usually mechanical), the optimizer's fusion patterns, and an autodiff
VJP/JVP with a finite-difference guard. Run with `RLX_DISPATCH_REPORT=1` after
compile to confirm the kind lands **native** rather than **common-ir**, and add
a cross-backend parity test. `just new-op` prints the full checklist.

To extend RLX **without editing core** — a new model, numerics crate, or
research op — use the stable seams in
[`rlx-extend`](crates/core/rlx-extend/README.md) (`LayerStage` block seam +
`OpExtension` op seam) rather than reaching into internal crates.

## PRs

- Keep the change focused and describe the motivation (which plan item, what it
  speeds up or fixes).
- Match the surrounding code's style, comment density, and idioms.
- Make sure `just ci` is green.

By contributing you agree that your contributions are dual-licensed under the
project's [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE) licenses. Unless
you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
