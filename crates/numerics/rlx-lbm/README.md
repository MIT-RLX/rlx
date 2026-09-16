# rlx-lbm

Moment-encoded lattice Boltzmann (HOME-LBM) as RLX graphs. Stores the first
three velocity moments per node and rebuilds the populations inside the
kernel, instead of storing 9 (D2Q9) or 27 (D3Q27) distribution values.

After Li, Wang, Pan, Gao, Wu and Desbrun, *High-Order Moment-Encoded Kinetic
Simulation of Turbulent Flows*, ACM TOG 42(6), 2023.

## Why this lives in RLX

Two reasons, neither of them fluid dynamics.

1. **It is the same trick as the quantized-weight path.** `DequantMatMul`,
   `ScaledMatMul` and `SynthMatMul` all keep a compact representation and
   expand it in the inner loop, trading DRAM for ALU. HOME-LBM does exactly
   that for a stencil: 27 values become 10, and streaming becomes a gather of
   moments. Same shape, a different access pattern.
2. **It exercises access patterns nothing else in the workspace does.** A
   27-point periodic stencil and a large-grid-to-small-array reduction look
   nothing like a transformer, so they stress the fusion and region machinery
   where transformer graphs never reach.

## What's here

- **`lattice`** — D2Q9 / D3Q27 velocity sets, weights, Hermite tensors.
- **`moment`** — the stored state, third-order reconstruction, closed-form
  collision.
- **`invariant`** — the moment round-trip oracle (`round_trip_d2q9`).
- **`sim`** — a periodic host reference simulator and Taylor–Green setup.
- **`graph`** *(feature `ir`)* — one LBM step as an `rlx_ir::Graph`, streaming
  via `Op::Roll`, so the solver runs on every backend through `Session`
  instead of only on the host.

## Correctness posture

The reconstruction has no reference implementation to check against — it *is*
the scheme. So the tests assert properties instead: the moment round-trip,
exact mass conservation, and Taylor–Green decay against the analytic
`exp(−2νk²t)`. That last one measures effective viscosity, which is precisely
what a mis-scaled reconstruction corrupts while still producing
plausible-looking flow.

## Features

- `ir` — build one LBM step as an rlx `Graph` and run it through `Session`.
  Off by default; the host reference simulator needs no backend.

## Install

```toml
[dependencies]
rlx-lbm = { version = "0.2", features = ["ir"] }
```

## Quickstart

```rust
use rlx_lbm::invariant::{round_trip_d2q9, sample_states_2d};

for m in sample_states_2d() {
    assert!(round_trip_d2q9(&m).within(1e-12));
}
```

## License

MIT OR Apache-2.0.
