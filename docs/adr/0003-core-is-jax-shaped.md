# ADR 0003 — Core ships JAX-shaped primitives; domain composites live downstream

## Status

Accepted — 2026-05-09. Carve-out for SPD-manifold layers 2026-07-13.

## Context

rlx is a framework modeled on JAX: tensor IR, autodiff, backend execution,
extension points. Two failure modes pull against that: coupling the framework
build to optional system libraries most users never need (SuiteSparse, MKL
Pardiso, …), and blurring the audience — application code has different
reviewers and a different release cadence than the framework.

## Decision

Core crates (`rlx-ir`, `rlx-opt`, the backends, `rlx-runtime`, `rlx-driver`,
`rlx-macros`) own:

- **JAX-shaped primitives** — anything `jax.lax` would expose: `Sin`, `Fft`,
  `DenseSolve`, `Scan`, … Their docs describe the math, not who uses it.
- **Extension surface** — `OpExtension`, `register_op`, `Op::Custom`,
  `CpuKernel`, the custom-op scaffold.

Composite or domain-specific ops — sparse LU, pruned FFT, eigensolvers, Krylov
methods, photonics, third-party solver bindings — live in downstream crates
(`rlx-sparse`, `rlx-linalg`, …) and register through the extension surface.

Test: would JAX put it in `jax.lax`, or in a separate package? The first is
core; the second is downstream.

**Carve-out:** the Riemannian / SPD-manifold layers (`BiMap`, `ReEig`,
`LogEig`, `SpdBatchNorm`, `SpdKarcherMean` and their backwards) are core `Op`
variants, chosen because first-class ops get native per-backend kernels more
readily than the CPU-first extension path.

## Alternatives

- **Everything in core.** Simplest to find, but the framework build grows
  system-library dependencies and domain churn.
- **Everything as extensions.** Primitives every backend must lower natively
  would lose fused GPU kernels.

## Consequences

- Reviews of new `Op` variants ask the `jax.lax` question.
- The SPD carve-out sits above an `eigh` that itself stays downstream; that
  tension is accepted and recorded here.
## License

MIT OR Apache-2.0.
