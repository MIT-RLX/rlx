# rlx-corpus

A **named corpus** of graphs that every compiler change must clear, with
per-family reporting.

RLX has thousands of tests, but they are organized by the crate that owns the
code, not by what a compiler change puts at risk. Changing the memory planner
or a fusion pass means running everything and reading the wreckage; nothing
answers *"which families did I break"*. This crate does.

## What it is

Each `Case` is a graph plus the family it belongs to. The corpus runs every
case through the device-free gate stack — structural verify, shape verify,
representation compatibility, and the memory-plan program-safety /
schedule-semantics gates — under **every planner configuration the backends
actually use**, because a plan that is safe when slots are pinned is not
necessarily safe when they are reused.

## What it is not

It is not a numerical test suite. Those live with their backends and need
devices. This answers a narrower question that nothing else does — does a
compiler change keep every family structurally sound — and it answers it on
any machine, with no GPU.

## Coverage is reported, never implied

`coverage()` returns how many distinct `OpKind`s the corpus touches out of how
many exist. That number is currently a *minority* of the op surface, and
printing it is the point: a corpus that reports "all green" without saying
what it covers reads like completeness it has not earned.

## Running it

```bash
# device-free gate stack — the default, works anywhere
cargo test -p rlx-corpus

# add the device arm on a host that has the backend
just check-corpus-device
```

The `[features]` list (`apple`, `gpu`, `metal`, `vulkan`, `cuda`, `rocm`) is
empty by default on purpose. A bare `cargo test -p rlx-corpus` stays
device-free and fast — and, more importantly, cargo unifies features across a
workspace build, so defaulting an Apple-only backend on here would turn it on
for every crate on Linux too.

## Evolution ledger

```bash
cargo run -p rlx-corpus --example evolution_ledger
```

Prints the per-family record so a compiler change can be judged against what
it was supposed to affect.

## License

MIT OR Apache-2.0.
