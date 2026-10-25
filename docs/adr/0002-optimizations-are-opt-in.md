# ADR 0002 — Optimizations ship behind opt-in features, default off

## Status

Accepted — 2026-08-01.

## Context

rlx's goal is that each backend is the fastest path on its hardware. Much of
the work toward that is experimental: decomposition kernels (Monarch, Tucker,
TT, low-rank, sparse, quantized matmul), per-layer mining and allocation,
quantization-actuation recipes. Most of it is heavy, some of it does not pay
off on most models, and all of it adds build time and code on the hot path.

## Decision

New optimization code lives behind an opt-in cargo feature, default off.
`pub mod` items are `#[cfg(feature = "...")]`; binaries and examples that need
it declare `required-features`. Defaults on workspace crates are not flipped to
turn one on.

An optimization becomes a default only on measured evidence, recorded where
the default is set.

## Alternatives

- **Runtime env flags only.** The code still compiles into every build and
  still sits on the hot path behind a branch.
- **Default-on with an opt-out.** Every user pays for every experiment; a
  regression ships to everyone at once.

## Consequences

- Feature combinations multiply; CI covers the defaults, and `just lint` does
  not see feature-gated code (a known gate gap).
## License

MIT OR Apache-2.0.
