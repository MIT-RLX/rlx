# rlx-ffi

C ABI for the RLX **distributed node** — link a mesh worker into an iOS app, an
embedded host, or anything else that speaks C.

A node is not necessarily a workstation. The RLX transport layer has no platform
gating and no default C dependencies, so anything with a CPU, a TCP stack and
`std` can be a rank. This crate is the seam that lets a non-Rust shell be one.

```c
#include "rlx_node.h"

/* Join a 2-rank mesh as worker 1, serving inference. */
rlx_node_start(1, 2, "192.168.1.10:29500", "auto", "infer");

char buf[256];
rlx_node_status(buf, sizeof buf);   /* "running", then "ok: …" */
rlx_node_stop();
```

Every entry point returns immediately — a serving loop parks in `recv` between
activations, and a UI thread must not block on it. Poll `rlx_node_status`.

## Modes

* `"infer"` — serve a stage the coordinator ships. Stoppable between
  activations.
* `"train"` — join a data-parallel training run. **Not** stoppable partway: the
  gradient reduce is a barrier, so a rank that leaves stalls every other rank.

## Building

`staticlib` only. A shared library must resolve every symbol at link time and
`zstd-sys` (pulled in by the GGUF reader) does not on iOS; a static archive
links happily on both iOS and embedded Linux.

```sh
cargo build -p rlx-ffi --release --target aarch64-apple-ios
```

A static archive carries no link directives, so the consuming target must name
what the Rust code needs itself. On Apple platforms that is `-framework
Accelerate` (rlx-cpu's BLAS / LAPACK / vForce), plus `-lresolv` and `-lc++` for
Rust `std`. See `ios/Demo/project.yml` in the repository for a worked set.

Features: `apple` (Metal + ANE), `gpu` (portable wgpu). Both off by default so
the archive cross-compiles to targets with no such frameworks.

## iOS

`ios/build-xcframework.sh` wraps this crate into `RlxNode.xcframework`, with a
Swift wrapper and a demo app. **iOS needs `NSLocalNetworkUsageDescription` in
Info.plist for any LAN peer traffic** — without it the node reaches nothing and
says nothing. See `ios/README.md`.

## Trying it

The desktop side has a self-test that needs no device:

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --self-test
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --self-test --train
```

## License

MIT OR Apache-2.0.
