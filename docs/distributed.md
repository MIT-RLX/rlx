# Distributed computing in RLX

How RLX runs training and inference across a network of nodes — the layers, the
collectives, node discovery, and the two runnable examples. For the deeper
integration seams (NCCL/RCCL, RDMA/UCX, NVSHMEM), see
[`crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md`](../crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md).
For **NAT-traversing runs across the internet** — reach peers by `EndpointId`
with no `ip:port` and no port forwarding — see
[iroh-transport.md](iroh-transport.md).

## TL;DR — run a job in one file

`crates/core/rlx-collectives/examples/dist_job.rs` + `rlx-dist.toml` are the minimal
path. Put the same TOML on every machine and launch each node with its rank:

```bash
RANK=0 dist_job rlx-dist.toml     # node A (rank 0 = coordinator)
RANK=1 dist_job rlx-dist.toml     # node B
```

```toml
[job]
task  = "train"     # "train" (data-parallel SGD) | "infer" (tensor-parallel matmul)
world = 2
steps = 200

[net]
device    = "auto"     # auto | cpu | metal | cuda   (per node)
topology  = "mesh"     # mesh | star
discovery = "static"   # static | mdns | rendezvous
peers     = ["127.0.0.1:29500", "127.0.0.1:29501"]   # for discovery = static
# coordinator = "AAA.BBB.CCC.DDD"                      # for discovery = rendezvous
```

Both tasks compile the model on **each node's own device** (Metal / CUDA / CPU)
via `rlx-runtime`, so a heterogeneous fleet just works.

## The layers

Distributed RLX is built bottom-up so each layer is swappable:

```
  your job (dist_job / dist_node / a model runner)
        │
  ProcessGroup            collectives: all_reduce / all_gather / broadcast / barrier
        │                 (rlx_driver::ProcessGroup, ReduceKind)
  Node builder            one-call bring-up + discovery  (rlx_driver::Node)
        │
  Transport (trait)       TcpTransport (mesh) · NetTransport (star) · IrohTransport (NAT) · SymmetricTransport
        │                 — swap in NCCL/UCX/RDMA here without touching layers above
  the wire                TCP · QUIC-over-iroh (relay + hole-punch); RDMA/Thunderbolt/NVLink are drop-in Transports
```

- **`Transport`** (`rlx-driver/src/transport.rs`) is the one trait everything
  rides on: `send_bytes` / `recv_bytes` / `barrier`. Every faster fabric
  (NCCL, UCX, RDMA, NVSHMEM) is "just another `Transport`" — as is
  **`IrohTransport`** (QUIC + relay, for NAT traversal by `EndpointId`; feature
  `iroh`, see [iroh-transport.md](iroh-transport.md)).
- **`ProcessGroup`** wraps a transport and exposes the collectives a data- or
  tensor-parallel layer needs: `all_reduce` (Sum/Mean/Max/Min), `all_gather`,
  `broadcast`, `barrier`, plus typed reductions (`all_reduce_typed` for
  f16/bf16/i8 — SIMD, for edge/MCU links) and bounded-staleness
  `federated_average`.
- **`Node`** (`rlx-driver/src/node.rs`) is the ergonomic front door:
  `Node::from_env()?.connect()?` returns a ready `ProcessGroup`, hiding
  mesh-vs-star, discovery, and `Arc` wrapping.

## Core concepts

- **Rank / world.** Each node has a `rank` (`0..world`); `world` is the node
  count. Rank 0 is the coordinator by convention.
- **Topology.**
  - `mesh` — every rank connects to every other (peer-to-peer). Needed for
    ring collectives (`all_reduce` across all ranks).
  - `star` — the coordinator listens; workers **dial out** and need no inbound
    port. NAT/mobile/Pi/Docker-friendly, and the shape for the
    coordinator/worker ship-graph model and federated averaging.
- **Data parallel** — every node holds the full model and a slice of the data;
  gradients are averaged each step (`all_reduce(Mean)`). This is `task = train`.
  For the heterogeneous-cluster details (CPU + GPU per node, unbiased weighting)
  and the reproducible/precise reduce, see [Ship-graph training](#ship-graph-training--distrun_train)
  and [Reproducible & precise gradient reduction](#reproducible--precise-gradient-reduction).
- **Tensor parallel** — the model is sharded across nodes; partial results are
  summed (`all_reduce(Sum)`). This is `task = infer`.
- **Pipeline parallel** — the model is split into sequential stages, one per
  node; activations flow rank→rank (`dist_node`'s `pipeline` mode).

## Node discovery

How nodes find each other. The design is two-tier because **no discovery
protocol crosses arbitrary networks/NAT without one known meeting point.**

| `discovery`  | Mechanism                                                       | Scope |
|--------------|----------------------------------------------------------------|-------|
| `static`     | hand-listed `peers = ["ip:port", …]`, one per rank             | any (you address them) |
| `mdns`       | mDNS-SD (Bonjour/Avahi): coordinator advertises `_rlx-coord._udp.local`, workers browse. Multicast `224.0.0.251` — switches/APs forward it far more reliably than raw broadcast. | one LAN, zero-config (`--features mdns`) |
| `rendezvous` | workers unicast a query to a known `coordinator` host and learn its data port; then dial it. The host can be a LAN IP, `host.docker.internal`, or a **Tailscale MagicDNS name**. | across NAT / different networks |
| `iroh` | QUIC over iroh with n0 public relays + pkarr/DNS discovery — peers are reached by **`EndpointId` alone**, hole-punching through NAT and falling back to a relay, with no known meeting point. A distinct `Transport` (`IrohTransport`), not a `Node` discovery mode — see [iroh-transport.md](iroh-transport.md). | across NAT / the internet, no coordinator |

Notes:
- mDNS is opt-in (`rlx-driver`'s `mdns` feature) — the unicast rendezvous path
  needs no extra dependencies. On Apple platforms mDNS is the native Bonjour
  path; on Linux it interoperates with Avahi.
- The unicast responder is deliberately **pure** (no self-broadcast): a
  coordinator that both beacons and receives on one socket starves incoming
  queries (its own zero-latency loopback always wins the recv race).
- For genuinely cross-network jobs, put the nodes on a Tailscale/WireGuard
  overlay and set `coordinator` to a MagicDNS name — every node then has a
  stable address reachable from any network.

## Example 1 — `dist_job` (the simple starter)

Config-driven training or inference over predefined or self-discovered nodes.
Reads `rlx-dist.toml`, builds the `Node`, joins the network, runs the task.

```bash
cargo build -p rlx-collectives --example dist_job            # + --features mdns for mDNS
RANK=0 ./target/debug/examples/dist_job rlx-dist.toml &
RANK=1 ./target/debug/examples/dist_job rlx-dist.toml
```

- `task = "train"` → data-parallel SGD on a tiny linear model; the gradient is
  averaged across nodes with `all_reduce(Mean)` each step.
- `task = "infer"` → the contraction dim is sharded across nodes; each computes
  a partial matmul, `all_reduce(Sum)` combines them, rank 0 checks the result.

This is the file to copy when wiring your own distributed run.

## Example 2 — `dist_node` (the full harness)

`crates/core/rlx-collectives/examples/dist_node.rs` exercises every distributed
capability. Select with `MODE=`:

| `MODE` | What it does |
|--------|--------------|
| `both` / `infer` / `train` | tensor-parallel inference + data-parallel training smoke tests |
| `bench` | link profiler (α/β latency-bandwidth model, ring vs. naive, crossover FLOP/byte) |
| `pipeline` | pipeline-parallel stages, activations flow rank→rank |
| `topology` | topology-aware planner recommendation |
| `placement` | per-op/region device placement within one graph |
| `multidev` | per-region multi-backend placement, host transfer at boundaries |
| `federated` | bounded-staleness federated averaging (drops stragglers) |
| `coordinator` / `worker` | **ship-graph** model: workers run a serialized graph the coordinator sends, resolving weights from URIs — build the worker once, run any model |
| `trainserve` | **ship-graph training**: a generic worker runs a shipped `TrainSpec`'s data-parallel loop on its local hardware, with no model code (see below) |
| `parity` | **cross-backend divergence diagnostic**: run one graph on the CPU oracle + every local backend, flag round-off vs. a candidate kernel bug (see below) |
| `fft` | distributed `Op::Fft`: coordinator scatters signals, each node FFTs on its device, gathers + verifies vs a DFT reference |

Env knobs: `RANK`/`WORLD`, `PEERS` or `DISCOVER=1` (+ `DISCOVER_HOST`),
`TOPOLOGY=mesh|star` (`DIAL_OUT=1` = star), `DEVICE`, `WORKER_DEVICE`,
`WEIGHTS_DIR`, `WEIGHTS_FMT` (`seed`/`file`/`safetensors`/`q8`), `STEPS`, and
`RLX_DETERMINISTIC_REDUCE=1` (reproducible gradient reduce — see below).

Quantized-on-device example (Q8_0-packed weights run through `DequantMatMul`,
native quant memory — the shape a Raspberry Pi wants):

```bash
WEIGHTS_DIR=/tmp WEIGHTS_FMT=q8 RANK=0 WORLD=2 MODE=coordinator dist_node
RANK=1 WORLD=2 MODE=worker dist_node
```

## The reusable ship-graph API — `rlx_runtime::dist`

The "build the worker once, run any model" primitive that lets a model runner
(e.g. a crate in `../rlx-models`) go distributed with no per-node/per-model
recompile:

- **Coordinator** ships each worker a `StageSpec` — a serialized subgraph + I/O
  names + weight-source URIs + a device directive.
- **Worker** (`dist::serve_stage` / `recv_stage`) compiles and runs it, resolving
  weights **locally** so they never cross the wire. Built-in resolvers cover
  `gguf://` (dequant or `packed` → `DequantMatMul` at native quant memory),
  `safetensors://`, and `file://`, behind a parse-once `WeightCache`; a caller
  closure handles other schemes.

Only KB-sized specs and activations go over the network — never the weights.

### Ship-graph training — `dist::run_train`

The training counterpart of `StageSpec`. The coordinator ships a **`TrainSpec`**
(serialized backward graph + trainable-param URIs + a data-shard plan + an
optimizer directive); a generic worker (`dist::run_train`, driven by `dist_node
MODE=trainserve`) resolves its params and data node-locally and runs the
data-parallel loop on its own hardware — **no model code baked in**. Gradients
are averaged across workers by a caller-supplied `reduce` closure, so the crate
stays collective-free (the worker plugs in `rlx-collectives`).

Two properties make it correct on a **heterogeneous** cluster:

- **CPU + GPU on every node.** `device: "all"` fans the backward out over every
  live local backend at once (intra-node data parallelism), so the CPU trains
  alongside the GPU instead of sitting idle.
- **Sample-count-weighted reduce.** A node casts one weighted vote *per local
  lane*: its bucket is `[Σ_lane grad …, lane_count]`, and a single `Mean`-reduce
  then yields `Σ_all grad / Σ_all lanes` — the true global mean gradient. So a
  3-lane Mac and a 1-lane CUDA box combine **without bias**, and because every
  rank sends the same bucket the same number of times, the collective never
  deadlocks. (A uniform-shard guard fails fast, identically on every rank, if the
  master ships unequal shard sizes — the one thing that *would* desync the sync
  cadence.)

## Reproducible & precise gradient reduction

Floating-point addition isn't associative, so a cluster's reduced gradient
depends on *how* the partial sums are combined — and by default that differs from
a single-machine run and shifts with the node count. RLX exposes a **reduction
mode** to make it reproducible and precise, on the host `ProcessGroup::all_reduce`
*and* the in-graph `collective.all_reduce` (so both the async/DDP-bucket and the
sync in-graph training paths honor it):

| `ReduceMode` | Algorithm | Guarantees | Cost |
|---|---|---|---|
| `Ring` (default) | bandwidth-optimal f32 ring | fastest; f32 precision; last ulp depends on world size | 1× |
| `Deterministic` | same ring, **reduce-scatter accumulates in f64** (all-gather stays f32) | bitwise reproducible run-to-run; correctly-rounded exact cross-rank sum (no precision/quality loss); effectively world-size-independent | ~1.5× reduce bytes, **ring bandwidth** — not a gather-to-root's `O(n·len)` |

Two ways to select it (every rank must agree):

```bash
# 1. Env — whole cluster, no code change (host + in-graph paths both read it):
RLX_DETERMINISTIC_REDUCE=1  RANK=0 WORLD=2 … dist_node
```

```rust
// 2. Baked into a graph — deterministic by construction, independent of env.
//    (The mode is carried through autodiff, so the backward reduce inherits it.)
use rlx_collectives::{all_reduce_op_mode, ReduceMode};
let g = all_reduce_op_mode(&mut bwd, grad, group_id, ReduceKind::Mean, ReduceMode::Deterministic);
// programmatic host call: group.all_reduce_mode(&mut buf, ReduceKind::Mean, ReduceMode::Deterministic)?;
```

`rlx-vision-bench` turns this on by default (`Config.deterministic`, CLI
`--no-deterministic` to opt out). Measured on a 2-rank loopback, the deterministic
mode runs at ~1.3–1.55× the f32 ring's time at bandwidth-bound sizes (16 MiB:
8.2 ms → 10.8 ms) and **scales like a ring**, so it stays cheap as the fleet
grows — the price of an exact, reproducible gradient.

> Note: this fixes the **cross-rank** combination. Each node's *local* reduction
> still runs in its backend's kernel order, so a mixed-backend cluster is not
> bitwise identical to one machine — that residue is per-backend kernel drift,
> which the parity diagnostic below characterizes.

## One machine vs. the cluster — the parity diagnostic

When "the numbers differ between one machine and the cluster", the dominant cause
on a heterogeneous fleet is **different backends running the same graph** (Metal
vs. CUDA vs. CPU), most of which is bounded f32 round-off — but a divergence *too
large* to be rounding is a kernel bug. `dist::backend_divergence` (and
`dist_node MODE=parity`) tells them apart: it runs a graph on the CPU oracle and
every other local backend and reports each one's worst gap, flagging anything past
a relative-error tolerance as **SUSPECT**.

```
[mac]  metal vs cpu: max_abs=1.118e-8 max_rel=1.789e-7 (output #0) — round-off (explainable)
[mac]    gpu vs cpu: max_abs=1.490e-8 max_rel=2.386e-7 (output #0) — round-off (explainable)
[mac] PARITY: all local backends within round-off of CPU ✓
```

Run it on each node; a green report means the fleet's cross-backend drift is
bounded round-off, a red one names the backend + output to chase (drill in with
`RLX_PARITY_DEVICE=cuda cargo test -p rlx-runtime --test cuda_backprop_parity`).

## Running across real machines

The workspace is validated across a Mac (arm64/Metal), a Linux CUDA rig, and an emulated aarch64 node under QEMU:

- **Local**: multiple ranks on `127.0.0.1` (loopback).
- **Cross-machine**: point `peers`/`coordinator` at LAN IPs; run one node per
  box (`ssh`).
- **Across NAT / the internet (iroh)**: no `ip:port` — run one process per box
  with `TOPOLOGY=iroh` and a shared `RLX_IROH_SEED` (or explicit
  `RLX_IROH_PEERS`); relays + pkarr discovery traverse firewalls. Validated
  Mac ↔ a Linux CUDA rig **through its inbound-blocking firewall**
  (hybrid Metal + CUDA training). See [iroh-transport.md](iroh-transport.md).
- **Raspberry-Pi-class aarch64**: cross-compile + run under QEMU with
  `./rig.sh test-pi` (includes a quantized `DequantMatMul` stage across two
  emulated nodes). See [rig notes in `rig.sh`](../rig.sh).

A worked three-way run (Mac coordinator on Metal + a Linux worker on CPU + a
QEMU/Docker worker on aarch64, all joined by dial-out star with rendezvous
discovery) is exactly what `dist_node MODE=fft` and `dist_job` demonstrate.

## Where things live

| Path | Role |
|------|------|
| `crates/core/rlx-driver/src/transport.rs` | `Transport` trait, `ProcessGroup`, collectives, `ReduceMode` (Ring / Deterministic) |
| `crates/core/rlx-driver/src/net.rs` | `TcpTransport` (mesh), `NetTransport` (star), symmetric heap |
| `crates/core/rlx-driver/src/node.rs` | `Node` builder + discovery (static / mDNS / rendezvous) |
| `crates/core/rlx-driver/src/iroh_transport.rs` | `IrohTransport` (QUIC + relay, NAT traversal by `EndpointId`) + `process_group_from_env` — feature `iroh`; see [iroh-transport.md](iroh-transport.md) |
| `crates/core/rlx-collectives/` | in-graph `collective.all_reduce` op (+ `all_reduce_op_mode`) + the examples |
| `crates/core/rlx-runtime/src/dist/` | ship-graph `{inference, training, diagnostics}` submodules + shared weight resolvers (`mod.rs`) |
| `crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md` | the deeper tiers (NCCL/RCCL, UCX/RDMA, NVSHMEM) |
