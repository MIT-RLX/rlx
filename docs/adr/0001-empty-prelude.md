# ADR 0001 — The `rlx` prelude has no default features

## Status

Accepted.

## Context

`cargo add rlx` is the front door. Whatever the prelude enables by default,
every new user compiles: backends, BLAS, companion crates. That cost is paid
before the user has asked for anything, and it hides which parts of rlx a
program actually depends on.

## Decision

The `rlx` crate has `default = []`. Every `pub use rlx_*` in `rlx/src/lib.rs`
is behind a feature (`ir`, `driver`, `opt`, `macros`, `runtime`, and one per
backend or companion crate; backend features imply `runtime`). `rlx-opt` is
also `default = []`: consumers opt into `compile` and/or `training`.

The rule is scoped to the prelude. Other workspace crates keep their defaults
where those protect performance or the obvious use (`rlx-cpu`'s `blas`,
`rlx-runtime`'s `cpu`); the prelude consumes them with
`default-features = false`.

## Alternatives

- **A "batteries" default set.** Faster first run, but the dependency surface
  stops being explicit, and removing a default later is a breaking change.
- **Emptying every crate's defaults.** Would remove BLAS from `rlx-cpu` users
  who never asked to lose it — a silent performance regression.

## Consequences

- A new re-export in the prelude must carry `#[cfg(feature = "...")]`.
- Docs and examples must name the features they need.
- `rlx-runtime` still has source with ungated `rlx_cpu::` references, which is
  why it keeps `default = ["cpu"]`.
## License

MIT OR Apache-2.0.
