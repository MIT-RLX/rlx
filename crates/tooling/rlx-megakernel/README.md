# rlx-megakernel

**A standalone study, not a feature.** How far can RLX compose a MoE block
toward a single device program — and what does fusing actually change?

## The question

CAKE §5.1 rewrites Alpha-MoE into one fused MoE megakernel: routed gather, two
projections, activation, requantization and route-weighted output accumulation
in a single device program, where the reference launches five GPU activities.
Its API-level speedups come substantially from removed scheduling gaps, not
from better arithmetic.

This crate asks the narrower question RLX can actually answer today: **how many
separate operations does an MoE block still take after RLX's fusion pipeline
has run, and does fusing change the answer it computes?**

## Why dispatch count and not wall time

Wall time on a contended machine is not a measurement. This tree produced a 9×
spread across repeated runs of an identical configuration, and a confident
table built on it that had to be retracted.

Scheduled-op count is exact, device-free, reproducible anywhere, and is the
quantity CAKE itself reports for this result. It is a *proxy* for launch
overhead. This crate does not claim it is a speedup.

## What it does not do

It does not implement a megakernel. RLX has no schedule IR in which "one
device program" is expressible — roles, barriers and pipeline stages are not
declared, so there is nothing to fuse them into. This measures the gap rather
than closing it, which is the honest thing a standalone study can do.

## Running it

```bash
cargo test -p rlx-megakernel
```

The tests report dispatch counts before and after fusion, and assert that
fusion is numerically neutral against the CPU backend.

## License

MIT OR Apache-2.0.
