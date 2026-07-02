# Distributed computing in RLX

How RLX runs training and inference across a network of nodes — the layers, the
collectives, node discovery, and the two runnable examples. For the deeper
integration seams (NCCL/RCCL, RDMA/UCX, NVSHMEM), see
[`crates/rlx-collectives/DISTRIBUTED_ROADMAP.md`](../crates/rlx-collectives/DISTRIBUTED_ROADMAP.md).

## TL;DR — run a job in one file

`crates/rlx-collectives/examples/dist_job.rs` + `rlx-dist.toml` are the minimal
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
  Transport (trait)       TcpTransport (mesh) · NetTransport (star) · SymmetricTransport
        │                 — swap in NCCL/UCX/RDMA here without touching layers above
  the wire                TCP today; RDMA/Thunderbolt/NVLink are drop-in Transports
```

- **`Transport`** (`rlx-driver/src/transport.rs`) is the one trait everything
  rides on: `send_bytes` / `recv_bytes` / `barrier`. Every faster fabric
  (NCCL, UCX, RDMA, NVSHMEM) is "just another `Transport`".
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

`crates/rlx-collectives/examples/dist_node.rs` exercises every distributed
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
| `fft` | distributed `Op::Fft`: coordinator scatters signals, each node FFTs on its device, gathers + verifies vs a DFT reference |

Env knobs: `RANK`/`WORLD`, `PEERS` or `DISCOVER=1` (+ `DISCOVER_HOST`),
`TOPOLOGY=mesh|star` (`DIAL_OUT=1` = star), `DEVICE`, `WORKER_DEVICE`,
`WEIGHTS_DIR`, `WEIGHTS_FMT` (`seed`/`file`/`safetensors`/`q8`), `STEPS`.

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

## Running across real machines

The workspace is validated across a Mac (arm64/Metal), a Linux CUDA rig
(`msi`), and an emulated aarch64 node under QEMU:

- **Local**: multiple ranks on `127.0.0.1` (loopback).
- **Cross-machine**: point `peers`/`coordinator` at LAN IPs; run one node per
  box (`ssh`).
- **Raspberry-Pi-class aarch64**: cross-compile + run under QEMU with
  `./rig.sh test-pi` (includes a quantized `DequantMatMul` stage across two
  emulated nodes). See [`msi`/rig notes in `rig.sh`](../rig.sh).

A worked three-way run (Mac coordinator on Metal + `msi` worker on CPU + a
QEMU/Docker worker on aarch64, all joined by dial-out star with rendezvous
discovery) is exactly what `dist_node MODE=fft` and `dist_job` demonstrate.

## Where things live

| Path | Role |
|------|------|
| `crates/rlx-driver/src/transport.rs` | `Transport` trait, `ProcessGroup`, collectives |
| `crates/rlx-driver/src/net.rs` | `TcpTransport` (mesh), `NetTransport` (star), symmetric heap |
| `crates/rlx-driver/src/node.rs` | `Node` builder + discovery (static / mDNS / rendezvous) |
| `crates/rlx-collectives/` | in-graph `collective.all_reduce` op + the two examples |
| `crates/rlx-runtime/src/dist.rs` | ship-graph worker/coordinator + weight resolvers |
| `crates/rlx-collectives/DISTRIBUTED_ROADMAP.md` | the deeper tiers (NCCL/RCCL, UCX/RDMA, NVSHMEM) |
