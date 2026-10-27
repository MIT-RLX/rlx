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
- **WideEP (Phase 0/1 + EPLB)** — expert-parallel MoE: each rank owns a shard of
  experts; tokens move with variable-size `all_to_all_v` via
  `collective.moe_dispatch` / `collective.moe_combine`, then a local
  `GroupedMatMul`. Builder: `rlx_collectives::moe_ep_ffn`. Online load
  balancing: `rebalance_placement` / `rebalance_with_replicas` (`num_slots > E`
  for hot-expert copies) + `register_ep_placement` /
  `register_ep_replicas`; weight remap via `migrate_to_replica_map` /
  `migrate_to_placement` (slab exchange over `all_to_all_v`). Toy parity:
  `cargo run -p rlx-collectives --example wide_ep_toy`. Device-resident EP:
  enable `rlx-cuda` `--features nccl`, bootstrap with
  `rlx_cuda::distributed::{new_nccl_id,id_to_bytes,id_from_bytes,init_and_register}`
  — `collective.all_reduce` / `collective.all_to_all` then use NCCL when a
  comm is registered; `moe_dispatch`/`moe_combine` still use host
  `all_to_all_v` (see
  [`DISTRIBUTED_ROADMAP.md`](../crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md)).

## Node discovery

How nodes find each other. The design is two-tier because **no discovery
protocol crosses arbitrary networks/NAT without one known meeting point.**

| `discovery`  | Mechanism                                                       | Scope |
|--------------|----------------------------------------------------------------|-------|
| `static`     | hand-listed `peers = ["ip:port", …]`, one per rank             | any (you address them) |
| `mdns`       | mDNS-SD (Bonjour/Avahi): coordinator advertises `_rlx-coord._udp.local`, workers browse. Multicast `224.0.0.251` — switches/APs forward it far more reliably than raw broadcast. | one LAN, zero-config (`--features mdns`) |
| `rendezvous` | workers unicast a query to a known `coordinator` host and learn its data port; then dial it. The host can be a LAN IP, `host.docker.internal`, or a **Tailscale MagicDNS name**. | across NAT / different networks |
| TCP/UDP rendezvous (`RENDEZVOUS=tcp://host:port` / `udp://…`, `Node::rendezvous`) | rank 0 serves at one known address; every rank registers the address its connection came from (so a multi-homed host advertises the interface that reaches the cluster) on an **ephemeral** data port, and gets the full peer list back. Workers may use `RANK=auto` (assigned in arrival order) and omit `WORLD`. UDP retransmits until acknowledged. | any network where workers reach rank 0 |
| zero-config rendezvous (`RENDEZVOUS=auto` on workers, `udp://auto` or `tcp://auto` on rank 0) | rank 0 announces its rendezvous URL every 200 ms by IPv4 broadcast, IPv4 multicast `239.72.66.1` and IPv6 multicast `ff02::7266` (UDP 29399); workers need **no** address, rank or world. | one LAN |
| LAN mesh (`DISCOVER=1`, `Node::discover`) | every rank announces `rank → ip:port` by IPv4 broadcast **and** IPv6 multicast `ff02::7266` until all `WORLD` addresses are known, then the full mesh connects. | one LAN, zero-config, no extra deps |
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
- The LAN mesh announces on a 100 ms timer, and each announcement carries a
  `done` flag. A rank that has found everyone keeps announcing until every
  peer reports `done` (at most 3 s), so a peer that missed the last packets
  (Wi-Fi drops broadcast/multicast under load) still completes. Wire format:
  `RLXDISC <rank> <ip:port> <done>`; older nodes send no flag and still
  interoperate.

### IPv6

Every place that takes an address accepts IPv6 — `PEERS`, `peers = […]` in
`dist_job` TOML, the `rlx-distributed` hostfile and RPC client — through
[`rlx_driver::resolve_addr`]:

| Form | Example |
|------|---------|
| IPv4 / hostname | `10.0.0.2:29500`, `node-a.local:29500` |
| global IPv6 | `[2001:db8::7]:29500` |
| link-local, scope by interface **name** | `[fe80::1%en0]:29500` |
| link-local, scope by interface index | `[fe80::1%2]:29500` |

A link-local scope names the **local** interface the peer is reached through,
so the same peer is written differently on each node (`%en0` on a Mac,
`%enp2s0` / `%wlo1` on Linux). That makes link-local useful when a host has no
routable address (no DHCP lease), since every host on the link still has one.

Discovery is dual-stack:

- **LAN mesh** (`DISCOVER=1`): one `[::]` socket hears both IPv4 broadcasts
  (as `::ffff:a.b.c.d`) and the `ff02::7266` multicast, so IPv4 and v6-only
  nodes find each other. macOS rejects a link-local multicast send without an
  interface, and std cannot list interfaces, so the first announcement tries
  interface indices 0–64 and later ones use only those that sent.
- **Star** (`TOPOLOGY=star DISCOVER=1`): the coordinator's data listener and
  UDP responder bind `[::]`, so workers can set `DISCOVER_HOST` to an IPv4
  address, an IPv6 literal (`DISCOVER_HOST=2001:db8::7`, no brackets needed) or
  a hostname.
- `local_ip()` (the address a rank announces) falls back to an IPv6 probe
  on a host with no IPv4 route.

Limits: on **Windows** the wildcard binds stay IPv4 (its sockets default to
v6-only), so discovery there is IPv4; static IPv6 `PEERS` still work. The
`rlx-distributed` `subnet = "…/24"` scan is IPv4-only (a /64 cannot be scanned).

[`rlx_driver::resolve_addr`]: ../crates/core/rlx-driver/src/node.rs

### Transport: TCP or QUIC over UDP

Whatever found the peers, `TRANSPORT=tcp` (default) carries the collectives over
TCP and `TRANSPORT=quic` over QUIC/UDP between the same addresses, straight on
the LAN — no relays (`IrohTransport::connect_lan`; build with
`rlx-collectives`'/`rlx-driver`'s `iroh` feature). Keys come from
`RLX_IROH_SEED` (hex) or, without it, from the peer list — enough to keep two
jobs apart, not to keep an attacker out. Measured mm4 ↔ msi ↔ amd (all-reduce
ring): TCP α 1.27 ms / β 92 MB/s, QUIC α 1.14 ms / β 85 MB/s. Every
coordination × transport combination (static / TCP rendezvous / UDP rendezvous
× TCP / QUIC) ran the 3-node Metal → CUDA → ROCm pipeline to 5e-8.

### Failures stop the job, with the rank

A rank whose connection ends (process exited, host gone) fails every receive
waiting on it with `rank N disconnected …` — on TCP and QUIC — instead of
hanging the others; a receive from a rank there is no link to (a star's
worker↔worker) fails at once. `RLX_COLLECTIVE_TIMEOUT=<s>` bounds any single
wait. Pair with a training `checkpoint_dir` and the launcher's `--retries` to
restart and resume.

A *device* that fails inside a rank does not stop the job. Each training lane
reports whether it compiled. A lane that cannot (an allocation failure on a
GPU busy with another job, an integrated GPU too small for the replica, a
backend limit) is dropped before planning, and a rank left with no lane trains
on its CPU. A lane that panics mid-run has its unanswered micro-batches re-run
on a surviving lane, or on a new CPU lane given the current params. The
cluster's plan and sync cadence never change. Qwen3-0.6B LoRA on mm4 + msi +
amd: msi's CUDA (6.2 GiB allocation beside another user's job) and amd's ROCm
(an arena past what was then a 4 GiB kernel-offset limit; amd's 780M now
trains that replica itself) both fell back to their CPUs,
with no `RLX_TRAIN_LANES` needed. Metal trained alongside them (26 syncs, none
skipped), and the first batch's loss went from 6.92 to 0.0018 in 212 s. `RLX_TRAIN_LANE_FAULT=<lane>@<run>` injects
such a failure for tests. An inference stage (`MODE=model`) does the same: if
its GPU panics while compiling or running, the stage is rebuilt on that
node's CPU and the request is retried.

On amd (a 780M APU on a two-node host), ROCm runs would sometimes end in
`HipError(719)` and a GPU reset. The cause was the kernel's automatic NUMA
balancing: its page sampling made amdgpu evict the process's GPU queues, and
the gfx1103 firmware hangs doing that. rlx-rocm now opts its process out
(`MPOL_LOCAL`, set before `hipInit`), which took that failure from 5 of 6 runs
to 0 of 4. The opt-out is per-thread, so threads started before ROCm comes up
keep the default. Run under `numactl --localalloc`, or set
`kernel.numa_balancing=0`, to cover a whole process. `cargo rlx doctor`
reports this host setup as RLX-ROCM-003.

### Multicast / broadcast on the LAN — what it is good for

Measured on this LAN (mm4 wired → msi on Wi-Fi + amd wired, 1400 B datagrams,
~2 s per rate):

| path | 2 MB/s | 10 MB/s | 40 MB/s |
|---|---|---|---|
| IPv4 multicast → Wi-Fi host | 100% | 100% | 100% |
| IPv6 multicast (`ff02::`) → Wi-Fi host | 27% | 5% | 1% |
| unicast → Wi-Fi host | 100% | 100% | 94% |
| any of them → wired host | 100% | 100% | 100% |

- **Configuration / discovery: yes.** Messages are tiny and repeated, so even
  a path that drops 99% (IPv6 multicast through this access point) delivers;
  sending IPv4 broadcast + IPv4 multicast + IPv6 multicast together covers
  every host. That is what `DISCOVER=1` and `RENDEZVOUS=auto` do.
- **Bulk data (weights, init params, data shards): not yet.** IPv4 multicast
  could cut the root's upload from (n−1)× to 1×, but it needs its own
  NACK/retransmit layer, depends on each AP/switch (the IPv6 row), and stops
  at the first router. A chained TCP broadcast gets most of the same win
  reliably; do that first if broadcasts become the bottleneck.

### Finding nodes — Bonjour / Avahi (mDNS)

A machine that can be a rank advertises itself as `_rlx-node._tcp` over
mDNS (Bonjour on macOS and iOS, Avahi on Linux), with TXT records for
`platform`, `devices`, `mem_gb` and `version`. Nothing in the record is trusted
or acted on by itself; it only lets you find machines without knowing their
addresses.

```sh
MODE=advertise NAME=msi ./target/release/examples/dist_node   # with this binary's backends
cargo rlx nodes advertise --name mm4 --devices metal,cpu      # or from the CLI
cargo rlx nodes                                               # who is out there
```

```text
name                   address                platform devices                       mem  version
Yellow iPad            10.0.0.13:29503        ios      cpu,metal,ane                6 GB  0.2.19
amd                    10.0.0.204:29502       linux    cpu,rocm,xdna,vulkan        44 GB  0.2.19
mm4                    10.0.0.185:29500       macos    cpu,metal,ane,gpu           64 GB  0.2.19
msi                    10.0.0.172:29501       linux    cpu,cuda,vulkan             61 GB  0.2.19
```

- **Advertise from the rank binary.** `cargo-rlx` is built with the CPU
  backend only, so it cannot see a GPU; `dist_node MODE=advertise` lists the
  backends that will actually run, and `cargo rlx nodes advertise` warns
  unless given `--devices`.
- **Which address.** A node answers once per interface: Wi-Fi, a USB or
  Thunderbolt link (`169.254/16`), a VPN (Tailscale's `100.64/10`), IPv6. The
  browser keeps them all and picks one on this machine's own subnet, then a
  private IPv4, then other routable addresses, and link-local last. Taking
  the first answer gave a USB link-local for the iPad and a VPN address for
  mm4.
- **In a cluster file**, `addr = "mdns"` resolves the host's `name` at launch
  (`[launch] msi → 10.0.0.172 (mDNS)`).
- **Rendezvous** servers advertise `_rlx-rdv._tcp` as well, and
  `RENDEZVOUS=auto` browses mDNS alongside the UDP announce, using whichever
  answers first.
- Both are `rlx-driver`'s `mdns` feature (`rlx_driver::mdns`): pure Rust, no
  daemon needed. They interoperate with the system ones: macOS's
  `dns-sd -B _rlx-node._tcp` lists and resolves the Linux nodes.

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

Addresses can be IPv4 or IPv6, including scoped link-local — see
[IPv6](#ipv6).

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
  `rlxp://` (package tensors; f32 or packed),
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

### Cluster training options — `dist::run_train_group`

A `TrainSpec` with `options: Some(TrainOptions { … })` trains through
`run_train_group`, which drives the `ProcessGroup` itself (the `dist_node
MODE=trainserve` worker picks it automatically; `rlx-vision-bench
--coordinate --cluster …` is the master). Every rank: checks the job matches,
times each device lane, all-gathers the timings, computes the same plan, and
trains in lock-step syncs.

| option | what it does | MNIST, mm4 + msi + amd, 3 epochs |
|---|---|---|
| (legacy `run_train`) | f32 all-reduce every batch | 24.6 s, 95.35% |
| defaults | same, through the new loop | 22.6 s, 95.32% |
| `accum: 4` | 4 micro-batches per sync | 9.9 s, 91.8% (raise the LR: 9.6 s, 94.6%) |
| `wire: Bf16` | half the gradient bytes (f64 accumulate) | with accum 4: 9.1 s |
| `balance: true` | each **lane** gets micro-batches ∝ its measured speed, each rank a data slice to match | with accum 4 + bf16 + LR: 8.6 s, 93.5% |
| `local_steps: 4` | local SGD: average parameters every 4 syncs | 11.4 s, **95.44%** |
| `overlap: true` | reduce sync k while computing k+1 (1-step-stale) | 21.0 s, 89.8% — hides little when compute ≪ comm; staleness hurts with momentum |
| `zero: true` | ZeRO-1: momentum sharded (reduce-scatter → slice update → all-gather) | 26.5 s, 95.32% — memory ÷ world, not speed |
| `checkpoint_dir` | rank 0 saves params + momentum per epoch; restart resumes | resume ≡ uninterrupted (tested bit-exact) |

`device: "all"` (or `RLX_TRAIN_LANES` on one node) runs one lane per physical
compute unit: the CPU plus each native GPU backend; wgpu/Vulkan/MLX beside a
native backend and ANE/XDNA are skipped. A list (`cuda,cpu,vulkan`) pins lanes —
e.g. msi's Intel Iris Xe through Vulkan (`VK_ICD_FILENAMES=…/intel_icd.json`).
All seven units of mm4 (CPU, Metal), msi (CUDA, CPU, Iris Xe) and amd (CPU,
ROCm) trained one model together, each lane balanced to ~8 ms per sync: 96.07%
in 14.5 s (12 epochs) vs the GPUs alone 96.80% in 16.8 s (6 epochs).

Parameters are uploaded once per sync (not per micro-batch), shared across
lanes, and only loss + gradients are read back (`upload_s` in the metrics).
Keeping parameters device-resident would not cut traffic while the reduce is
host-staged — the averaged gradient comes back over the same bus — so that
waits for device collectives (NCCL/RCCL).

### Choosing precision in a cluster

Three separate choices, each measured on mm4 (Metal) + msi (CUDA,
3080 Ti laptop) + amd (ROCm, 780M) — `dist_node MODE=dtypes` reprints the
compute table on any node:

**1. Compute format** — `TrainOptions::number_format` / `StageSpec::number_format`
(any [`NumberFormat`]: `f32`, `f64`, `f64x2`, `tf32`, `bf16`, `f16`, `f8e4m3`,
`f8e5m2`, `f6`, `f4`, `fNeXmY`, `weights/activations` like `f4/f8`). Narrow
formats are *emulated*: weights rounded on upload, every intermediate (forward
and backward) rounded after the op that makes it, f32 accumulation inside
matmuls — so the numbers are the format's on every backend, but there is no
speed gain. Master weights, momentum and the gradient sum stay f32; f16 and
narrower start loss-scaled (1024, halved on overflow).

| format | MNIST 3-node (acc, wall) | 24-layer pipeline error vs f32 | use it for |
|---|---|---|---|
| f32 (default) | 95.32%, 24–30 s | 0.9e-3 (CUDA TF32 dominates) | everything |
| tf32 | 95.33%, 24 s | 1.9e-3 | matching an NVIDIA run on every backend |
| bf16 | 95.26%, 27 s | 1.7e-2 | studying a bf16 deployment; QAT |
| f16 | 95.29%, 24 s | 2.4e-3 | same, with f16 range |
| f8e4m3 | 95.41%, 26 s | 2.9e-1 | fp8 QAT / deployment studies |
| f64 | 95.34%, 29 s | 2.8e-6 | references, verification |

f64 is native on the CPU and double-single (~48 bits) on Metal, CUDA, ROCm and
wgpu — 3–5× slower than f32 there (Metal 1024² GEMM: 1150 → 215 GFLOP/s).

**2. Gradient wire** — `TrainOptions::wire`, default **`Auto`**: rank 0
measures α/β and picks **bf16 with error feedback** when the gradient bucket is
bandwidth-bound (`bytes/β > α`), else f32. MNIST's 795 KiB bucket on this LAN:
bf16, 17.6–20.2 s vs 23.9–30.4 s at the same accuracy (95.26–95.35%). bf16 over
f16 because a gradient bucket spans many binades and f16 underflows below
6e-5. Error feedback carries each sync's rounding into the next, so it averages
out. Parameters (local SGD) always average in f32.

**3. Activation wire** (pipelines) — `send_activation_as(…, WireDtype::Auto)`:
**f16 per message** when every value fits ±65504, else bf16. 24-layer pipeline,
8 micro-batches: f32 wire 0.9–1.4e-3 error, f16/auto 0.9–1.4e-3 (CUDA's TF32
dominates), bf16 4.4–5.2e-3; 16-bit is never slower and halves the bytes.

Notes:
- **CUDA runs f32 matmuls in TF32 by default**: 200× the error of the other
  backends (2.7e-4 vs 1.3e-6) for 1.1× speed on 1–2k GEMMs (1.35× on a small
  training GEMM). Set `RLX_CUDA_NO_TF32=1` when ranks must agree with each
  other or with a CPU reference.
- **Apple's CPU (Accelerate), Metal and wgpu GEMMs are bit-identical**: all
  three sum each output strictly in k order with FMAs (checked by an
  order-sensitive probe), so a Mac's CPU and GPU lanes produce the same
  numbers. GEMV (decode) differs: split reductions, Metal's more accurate.
- **A `Cast` to f16/bf16 in a user graph keeps the f32 value by default**:
  CPU and GPU backends execute half graphs in f32 (half storage only). Set
  `RLX_HALF_CASTS=round` (`CompileOptions::half_casts = HalfCasts::Round`,
  `TrainOptions::half_casts`, `StageSpec::round_half_casts()`,
  `rlx-vision-bench --round-half-casts`) to round them to the format — verified
  on CPU, Metal, wgpu, CUDA, ROCm and Vulkan. Use `number_format` for
  low-precision arithmetic over the whole graph.
- Reductions accumulate in f64 on 16-bit wires; `RLX_DETERMINISTIC_REDUCE=1`
  makes the f32 ring bitwise reproducible (1.5× its bytes).

### Streaming pipeline inference

- `planner::split_layers` splits layers across devices by measured FLOP/s and
  memory (minimize the slowest stage, then total time; a device too slow to help
  gets no layers).
- `planner::recommend_inference` picks single / pipeline / tensor-parallel for
  a latency or throughput objective, the micro-batch count, and 16-bit
  activations when a hop is bandwidth-bound.
- `dist::pipeline_run` (rank 0) / `dist::serve_stage_stream` (workers) stream
  micro-batches with a sender thread per stage, over `send_activation_as`
  (f32 / bf16 / f16).

`dist_node MODE=pipestream` runs it planned from the live mesh. 24 × [1024²]
MLP, 512 tokens: one batch with f32 activations vs 8 bf16 micro-batches —
3 GPUs 83.6 → 42.4 ms (1.97×), all 7 devices 92.0 → 34.8 ms (2.65×), within
bf16 tolerance of a CPU reference. The planner still says **single device** —
Metal alone runs it in 7.8 ms, because a 2 MB activation hop over this LAN
costs 17–30 ms; distributing inference pays when the model does not fit.

## Any model, distributed — `MODE=model`

Every exported model is a `.rlxp` package of functions; `rlx_runtime::model::distributed`
trains or runs any of them across the cluster with no model code anywhere.
Rank 0 holds the package (and dataset) and sends the rest what they need;
workers (`model::distributed::serve`) need neither the package nor the weights
on disk.

```sh
# what's in it
cargo rlx dist cluster.toml --set MODE=model --set PKG=qwen3.rlxp --set TASK=info
# greedy generation across the mesh, checked against one node
… --set TASK=generate --set DIMS=seq=32 --set PROMPT="The capital of France is" --set NEW=8 --set VERIFY=1
# (SPEEDS=43,27,30 pins the split instead of the measured speeds)
# any function on given inputs
… --set TASK=infer --set FUNCTION=forward --set INPUTS=x.safetensors --set VERIFY=1
# fine-tune: LoRA on q/v, AdamW, data-parallel with every cluster option
# (DATA=corpus.txt is tokenized by the package; or a safetensors dataset)
… --set TASK=train --set DATA=corpus.txt --set 'TRAIN=lora:*q_proj*,*v_proj*:8' \
  --set 'LOSS={"cross_entropy":{"logits":"#0","labels":"labels"}}' --set LR=1e-3 \
  --set EPOCHS=2 --set ACCUM=4 --set OUT=qwen3-lora.rlxp
```

- **Inference** (`InferJob`): the function, sized by `DIMS`, is split into one
  contiguous stage per rank with estimated FLOPs proportional to each rank's
  measured GEMM speed. Each stage's weights go out once, read by the package's
  own loader, so references and stored layouts apply. Requests then stream
  through, and each stage forwards only the values later stages read
  (`WIRE=auto`: f16 per message). `generate` decodes greedily for an LM
  function whose input is tokens (plus an optional `last` position).
  Qwen3-0.6B Metal → CUDA → ROCm: same tokens as one CPU, 3.6 tok/s.
- **Speculative decoding** (`SPECULATE=1`): a package whose `generate`
  entry has a `verify` step (`rlx-export --variant mtp` on a Qwen3.5 MTP GGUF)
  runs it instead of the one-token step: each pass the last token and the MTP
  head's guess at the next, both rows' logits, and the next guess
  (`InferJob::decode_verify`). A right guess keeps both rows, two new tokens;
  a wrong one drops the pass. Every stage holds a pass's state until the
  next pass's `KEEP` value says which (`model::distributed::KEEP`). The run
  then retries the last token with the right one, so the tokens are greedy
  one-token decoding's (`VERIFY=1` checks them against one CPU, one token a
  step; `decode_local_verify` is the unsplit reference). Qwen3.5-0.8B Q8_0
  over mm4 (Metal) + CUDA + ROCm, split pinned, same tokens: 60 tokens of
  prose 12.0 → 8.2–10.1 s, 120 of code 17.9 → 11.6–15.2 s (the range is
  load on the Metal node). It keeps 60% and 92% of the drafts.
- **Layout.** A cut may move up to 25% of its stage's share (by FLOPs) to
  where the fewest bytes cross (`split_function_cut`). One cut between a
  KV cache and the attention reading it sent the whole expanded cache,
  2 MB a step. `ARGMAX=1` reduces the step's logits (and draft logits) to
  their argmax before splitting (`StepLayout::argmax`), so the last stage
  sends token ids, not 1 MB rows of a 248K vocabulary. Every rank measures
  its links to every other at the speed exchange (`exchange_speeds_links`:
  latency, bandwidth each way). `TOPOLOGY=1` puts the workers in the order
  the estimate of each stage's FLOPs at its speed plus each hop's latency
  and bytes over bandwidth makes fastest. Each stage then carries its own
  `prev` / `next` rank. Qwen3.5 verify step over 5 nodes: the hop bytes went
  from 2.7 MB / 59 KB / 43 KB / 6.3 MB / 3.0 MB to 13 KB … 9 KB. For one
  stream, though, a step's time is the *sum* of every stage and every hop.
  A model that fits on one node decodes faster there; the cluster pays in
  capacity, or in throughput with several requests in flight.
- **Calibration.** Each stage adds its compute seconds to the values it
  passes on (`TIME`), so rank 0 knows every stage's time each step
  (`InferJob::stage_seconds`, `measured_speeds`). `InferJob::calibrate` runs
  a few throwaway requests under `KEEP` (nothing kept), turns the times into
  each rank's measured GFLOP/s, and lays the step out again (order and
  cuts). If that is estimated faster by more than the given gain, the
  workers take their new stages mid-job (a `RESTAGE` marker down the
  pipeline, then a plan and weights each). `dist_node TOPOLOGY=1` calibrates
  automatically before decoding (`CALIBRATE=0`: not). GEMM GFLOP/s, the
  starting point, misjudge decode stages: amd measured above msi and ran
  its share ~3x slower.
- **Training** (`train_job`): the package trainer's preparation (function,
  dims, loss, `train`: all | globs | LoRA | SAE) is differentiated into a
  `TrainSpec`. Trained params, frozen params (`TrainSpec::frozen`) and dataset
  slices go out from rank 0, and `run_train_group` trains it with AdamW (or SGD)
  plus every cluster option. `OUT` saves it as the package trainer does: the
  package, a LoRA adapter, or an SAE. The dataset is a safetensors file with
  one tensor per function input, samples on the first axis (any int or float
  dtype), or plain text (`DATA=*.txt`, `train_job_text`).
- **Text** uses the package's own tokenizer (`package_codec`: the first
  `entries.*.tokenizer`). A text `PROMPT` is encoded, and `generate` prints the
  decoded text too. A text dataset becomes next-token windows the length of
  the token input, every `STRIDE` tokens (`text_samples`). A one-value input
  (a position like `last`) gets the window's last index, a window-long input
  (a mask) gets ones, and the labels get the next token or the shifted window.
  No Python helper is needed.
- Output names: a loss can name an output by its node name, the graph's
  output name, or `#i`.
- `HELDOUT=<text.txt>` (with `HELDOUT_BATCHES`, default 32) scores windows the
  job never trains on, before and after (`TrainJob::eval_loss`), as loss and
  perplexity. SmolLM2-135M LoRA (q/v, rank 8) on 120 KB of this repo's docs,
  4 nodes (Metal, CUDA, ROCm, iPad Metal), 64-token windows, 2 epochs:
  held-out perplexity went 70.7 → 56.3 at lr 3e-4 in 262 s. At lr 2e-3 it
  diverged (70.7 → 1538).
- **Activation wire.** Under `WIRE=auto`, a message of up to 1 MiB
  (`EXACT_ACTIVATION_BYTES`) now goes f32, and only larger ones go f16/bf16.
  A decode step sends ~256 KB, latency-bound, so f16 saved ~4 ms per hop while
  rounding the hidden state. That flipped a near-tied greedy token in a
  2-stage Qwen3 split (f16: 0/2 matched one node; f32: 2/2). Kernels differ
  per backend too, so a near-tie can still flip: a 4-stage split differed at
  the same token, `:` vs `,`. Exact token match is a weak check at near-ties.
  Compare logits.
- A training run whose every sync is non-finite now fails on all ranks with
  the lanes that produced it, instead of silently skipping every update.

## Launcher — `cargo rlx dist`

```toml
[job]
bin = "target/release/examples/dist_node"
port = 29500
coordinate = "static"     # static (PEERS) | tcp | udp (rendezvous on `port`)
transport = "tcp"         # tcp | quic
build = "cargo build --release -p rlx-collectives --example dist_node"
env = { MODE = "pipeline" }

[[host]]
name = "local"            # or an ssh alias
addr = "10.0.0.185"
repo = "/Users/Shared/rlx"
ranks = [{ device = "metal" }, { device = "cpu" }]

[[host]]
name = "msi"
addr = "10.0.0.172"
repo = "~/rlx"
env = { LD_LIBRARY_PATH = "/usr/local/cuda/lib64" }
ranks = [{ device = "cuda" }]
```

`cargo rlx dist cluster.toml [--sync] [--build] [--set K=V]… [--retries N]
[--timeout S] [--dry-run]` starts every rank (several per host = one per
device) with RANK/WORLD/PEERS (or RENDEZVOUS)/DEVICE/LABEL, prefixes each rank's
output, and stops the whole job when one rank fails or the timeout passes —
remote ranks run in their own process group and die with their ssh session, so
nothing is left behind. A rank's `cmd` replaces the binary (e.g. a master).

A host can also be an **iPhone or iPad attached to the launching machine**.
Its ranks run the RLX node app (`ios/Demo`) through `xcrun devicectl`, the
rank's settings passed as launch arguments, and its console is the rank's
output. `--sync` and `--build` skip it:

```toml
[[host]]
name = "Yellow iPad"      # the name it advertises, for addr = "mdns"
addr = "mdns"             # or its Wi-Fi address
repo = "-"
ios_device = "Yellow iPad"        # devicectl name or identifier
# ios_bundle = "com.mit.rlx.nodedemo"
ranks = [{ device = "metal" }]
```

The app needs `job.coordinate = "static"`, and with `addr = "mdns"` it must be
open (idle is enough) before the first launch, since that is when it
advertises. It stays open between jobs. The launcher starts it without
`devicectl --console`, which would forward the end of the session to the app
as a signal and kill it. With `-logfile`, the app writes its output to
`Documents/rlx-node.log`, headed by the launch's `-jobid`. The rank's command
polls that file with `devicectl device copy from`, prints new lines, and ends
on the app's final status. `-nodename` keeps the app advertised under the
cluster file's host name. Three consecutive MNIST jobs ran without the app
being reopened.

**Over USB-C.** A host's `via` table says how *its* ranks reach particular
other hosts, in place of their `addr`. `"usb"` joins `local` and a wired
iPad over the cable: the iPad's link-local IPv4 on its USB network interface
(its node advertises one per interface) from the Mac, and the Mac's address
on that interface from the iPad. Other links keep `addr`:

```toml
[[host]]
name = "local"
via = { "Yellow iPad" = "usb", "iPad MM1" = "usb" }
# …
[[host]]
name = "Yellow iPad"
via = { local = "usb" }
# …
```

An iPad accepts no connection over its cable but can open one. So where it
is the higher rank of a USB-C pair, which in a mesh the lower rank dials,
the launcher passes `RLX_MESH_REVERSE` (`0-1,0-4`), and the iPad dials
instead. The CoreDevice tunnel `devicectl` reports reaches only the system's
own services, not an app's sockets. Ping over the cable is ~1 ms, against
~90 ms over this Wi-Fi. Put the iPads next to `local` in rank order: the
pipeline runs rank 0 → 1 → … → 0, so only those hops can use the cable.

**Over Bluetooth LE.** `via = { "Yellow iPad" = "ble" }` on `local`
(cargo-rlx built `--features ble`, the app built with `BLE=1`) carries that
link over an L2CAP channel, through
[webbluetooth](https://crates.io/crates/webbluetooth) (`rlx_driver::ble`).
The node, given `RLX_BLE_SERVE=<its name>`, publishes a channel and a GATT
service holding its PSM, advertises under its name, and joins each channel
to its own mesh port on localhost. The launcher bridges a local port to it.
The mesh still speaks TCP, end to end. mm4 ↔ iPad measured **~125 ms
one-way and ~24 KB/s each way**, against ~1 ms / tens of MB/s over USB-C,
so it suits control and fallback traffic, not weights: a stage's weights
cross the same link. The first use asks for Bluetooth permission on the
iPad.

**Persistent workers.** `cargo rlx dist --persist` (or `persist = true` in
`[job]`), for `MODE = "model"`, keeps every rank but rank 0 running between
runs. A worker (`dist_node PERSIST=1`, or the iOS app with `PERSIST=1`)
connects, does its part in one job (the same barriers a one-job process
runs), leaves the mesh and connects again. It keeps its stage, compiled and
with its weights. `deploy` sends each worker its plan and waits for an
answer, and sends the weights only where the plan differs from the one the
worker kept (`InferJob::kept` counts them). A run launches rank 0 and only
the workers that aren't already running with the same settings. Those
settings are the program, the rank's environment, and of the `--set`
overrides only `MODE`, `DEVICE`, `TRANSPORT` and `RLX_*`: `PKG`, `PROMPT`
and the rest belong to rank 0's job. For a running worker, the launcher
follows its log until the worker logs its part done. `--restart`, `--build`
and `--sync` start the workers again.

- A desktop worker runs detached in its own session, with its log, pid and
  fingerprint under `$TMPDIR/rlx-node-<port>-<rank>/`.
- An iOS worker is the app. The launcher starts it again only when its
  process (found by the bundle's install path, through `devicectl`) is gone
  or was started with other settings.
- Plans only repeat if the split does. The speed exchange plans by the last
  job's speeds while every rank is on the device it was on and each
  worker's new measurement is within 3× of them. The GEMM probe is noisy:
  an iPad's swung 2× between runs and a busy mm4's 4× (desktop workers
  held within 7%). Rank 0's number isn't compared, since it's the rank
  started afresh each run. A rank that really got slower keeps the old
  split until `--restart`, `SPEEDS`, or calibration (`TOPOLOGY=1`).
- Calibrated layouts carry over. Each stage plan records the speeds and
  pipeline order the step was laid out by (`StagePlan::layout`), and a
  worker remembers the one it was last deployed with, after calibration
  re-laid it out. While the ranks are steady, the exchange returns those
  speeds, with the order in `Links::order`, so the next job's first layout
  is the calibrated one. Its calibration then searches orders again, and
  `calibrate_for(…, Some(steps))` re-lays out only where the time saved
  over the job's steps beats moving the changed stages' weights from rank 0
  (`dist_node` passes prompt + new tokens). Without that, calibration
  noise re-laid a 60-token job out for an estimated ~1.5 s saving that
  cost ~15 s of weights over Wi-Fi, and the new layout decoded slower.
- A worker waits for the next rank 0 as long as it takes (`RLX_CONNECT_WAIT=0`).

**Integrated GPUs between requests.** A pipeline stage spends most of a step
waiting for the others, and an integrated GPU powers down in that time. On
amd's 780M the same Q8_0 GEMV took 128 µs back to back and 389 µs after
300 ms idle. That held even with `power_dpm_force_performance_level = high`
(every clock at its maximum) and with the CPU kept busy: it's the GPU
itself waking up. `RLX_STAGE_KEEP_WARM=1` keeps the device busy with a small
matmul while the stage waits, and stops it when the next request arrives.
amd's stage went from 31.6 ms a step (default power management) to 21.1 ms
(`high`) and 11.2 ms (`high` + keep-warm). It made no difference on the
iPads, and msi (a discrete GPU) was slightly slower with it, so set it per
host: `env = { RLX_STAGE_KEEP_WARM = "1" }` on the APU's `[[host]]`. It
costs power for the job's length.

`MODE=trainserve` jobs (e.g. `rlx-vision-bench --coordinate --cluster` with
`RLX_PUSH_DATA=1`) run on the app through `dist::trainserve_worker`. That
function is the trainserve worker's collective sequence, and
`trainserve_worker_matches_the_masters_sequence` pins it. An MNIST MLP trained
data-parallel on mm4 Metal + msi CUDA + amd ROCm + iPad Metal: 4 epochs,
95.85%, ~32 s, 86–93% of it communication over this LAN (936 syncs).

A rank's `RLX_*` variables reach the app through `devicectl
--environment-variables`, so `RLX_TRAIN_LANES = "metal,cpu"` gives the iPad
a CPU lane (Accelerate) beside its GPU, as on any other node. MNIST, local SGD
(`--local-steps 4 --balance`) on all four machines:

| lanes per node | epochs, lr | time | samples/s | accuracy |
|---|---|---|---|---|
| GPU only | 4, 0.02 | 8.7 s | 16,320 | 93.78% |
| CPU + GPU | 4, 0.02 | 4.4 s | 107,671 | 89.60% |
| CPU + GPU | 8, 0.1 | 6.5 s | — | **95.87%** |

The balancer gave the CPU lanes the larger share on mm4 (17 vs 8 micro-batches)
and the iPad (4 vs 1). A sync then covers ~5× the samples, which means fewer
optimizer steps per epoch. At the default learning rate that costs accuracy,
and scaling the rate with the batch recovers it.

**Core ML / the Neural Engine as a training lane (`ane`) is accepted, and the
planner sits it out.** Core ML bakes weights into its compiled model, so every
parameter upload, one per sync, recompiles it. On the iPad (`metal,cpu,ane`)
the ANE lane got a full share from a probe that never uploaded parameters, and
the job slowed from 6.5 s to 20.8 s, with every other rank waiting 16 s of 17.
Two changes since:

- The probe uploads the trained params before every timed step, as a sync
  does.
- A balanced plan gives no micro-batches to a lane slower than `SLOW_LANE`
  (8×) its rank's fastest lane, and a rank always keeps one lane.

The same job now plans `[metal 3, cpu 14, ane 0]` and takes 5.4 s, against
5.1 s for plain CPU+GPU. The ANE is for inference, where weights are fixed.
AMX needs no lane of its own: the CPU lane's Accelerate BLAS runs on it (SME
on M4-class chips).

**Learning-rate scaling** (`TrainOptions::lr_scaling`, `--lr-scaling
sqrt|linear`, `LR_SCALING=`). The spec's rate is taken as tuned for one
`batch`. An update covers the cluster's samples per sync (per rank, under
local SGD), and the rate grows by √ or × that ratio, with a linear warmup over
5% of the updates (`warmup`). CIFAR-10 CNN, 6 epochs, CPU+GPU(+ANE) lanes on
all four machines, 208–272 samples per update:

| learning rate | accuracy |
|---|---|
| fixed 0.02 (the default) | ~53.9% |
| 0.02, linear (×4.25) | 60.74% |
| 0.04, linear (×3.25) | **69.75%** (GPU-only, fixed 0.1: 71.77%) |

**Work stealing inside a rank.** A rank's lanes pull micro-batches from one
queue, two in flight each, instead of a fixed per-lane count. A lane slowed
mid-run (mm4's CPU, with other jobs compiling at load average 50+) takes fewer
and the others more. The rank's total, and so the cluster plan, is unchanged.
Gradients are summed in batch order when they fit 256 MB, so the result does
not depend on which lane ran what. Lanes the plan sat out (an `ane` lane) pull
nothing.

**Core ML for inference.** The bench evaluates on the ANE where there is one
(CIFAR-10 test set: 39,859–45,150 img/s on mm4, against 84,000 on Metal and
19,600 on the CPU). A Qwen3 pipeline stage runs on it too: half the model on
mm4's ANE plus msi CUDA, 4.0 tok/s (6.8 on Metal), logits within 4.4e-4 of
one CPU. On the iPad an ANE stage that holds the LM head was killed without a
crash log, most likely for memory; keep the iPad on Metal for that stage.

**Stateful steps across ranks** (Qwen3.5's hybrid decode: gated delta-net
layers with recurrent state, attention layers with a KV cache). A package
whose `generate` entry lists `state` decodes through
`InferJob::start_with_state` and `InferJob::decode`. Each stage keeps its own
layers' state (`StagePlan::state`: replaced, or a row appended per step), so
only the hidden state crosses the network, ~4–33 KB per step. `decode_local`
is the unsplit reference. Qwen3.5-0.8B (`unsloth/Qwen3.5-0.8B-MTP-GGUF`, BF16)
on mm4 + msi + amd + iPad matched it token for token, with logits within
3.3e-6.

Where the time goes, per token (40 tokens, the workers report it):

| | mm4 Metal | msi CUDA | amd | iPad Metal | steps/s |
|---|---|---|---|---|---|
| amd on ROCm | 33 ms | 15 ms | **99 ms** (5% of the model) | 37 + 7 ms | 4.8 |
| amd on CPU (split gives it nothing) | 36 ms | 8 ms | — | — | 9.0 |

The network costs ~25 ms over the four hops. The split weighs ranks by a
1024³ GEMM, which says nothing about a stage of small kernels plus 1 MB of
recurrent state in and out per layer. amd's 780M is fast at the first and
slow at the second. A 0.8B model is no faster split than on one Mac. The
split buys capacity; for speed, use the fewest ranks that hold the model.

**What made it slow, and what fixed it.** `RLX_STAGE_PROFILE=1` breaks a
stage's step into state in / run / state out, and `RLX_ROCM_PROFILE_STEPS=1`
breaks a ROCm run down by op kind:

- **ROCm ran the gated delta-net on the CPU.** It claimed the op but routed it
  through `gdn_host`, downloading the operands and uploading the result:
  512 ms a launch, 95% of amd's stage. It now launches the shared GPU kernel
  (as CUDA does) when the state is carried, as it is in a decode step. amd's
  half of the model went from 2,171 ms to 65 ms a step, with logits still
  within 3.1e-6 of one CPU.
- **Moving state through the host is cheap.** Building a stage's state
  inputs and taking its outputs is ~0.5 ms of a 40 ms step, and the copies
  inside the run are about a millisecond on unified memory. That is not
  where the time goes.
- **Narrower weights would not speed up decode yet.** mm4 reads the model at
  ~89 GB/s and msi at ~45 GB/s, well under their bandwidth, so the steps are
  bound by many small kernels, not by memory. Shipping weights is a different
  matter. They now go bf16 when every value is exactly a bf16 (a BF16
  checkpoint), which is lossless and half the bytes. Stages ship in 16.6 s.

| 4 ranks, 40 tokens | mm4 | msi | amd | iPad | steps/s |
|---|---|---|---|---|---|
| before | 33 ms | 15 ms | 99 ms | 44 ms | 4.8 |
| GPU delta-net on ROCm, bf16 weight shipping | 45 ms* | 16 ms | 11 ms | 47 ms | 7.2 |

\* mm4 was loaded (load average 13–22). For comparison, mm4 alone, quiet,
runs the whole model at 27.5 steps/s. A 0.8B split pays ~27 ms of network and
the sum of four stages per token.

MTP self-speculation does not pay across a pipeline as built. rlx-qwen35's
`spec.rs` commits accepted tokens one decode step at a time, and in a
pipeline each step is a full round trip. Paying would take a k-token step
that verifies and commits in one pass, which means emitting the gated
delta-net state after each position, so a stage can keep the one the accept
rule picks.

Startup: `PackageWeights::values` reparsed the GGUF header, including its
~150k-entry vocabulary, for every one of 321 params (197–427 s). It now keeps
the opened file (20 s).

**Checking a split decode** (`VERIFY=1`). Besides the free-running token
comparison, it now runs a teacher-forced logit comparison: at every step both
sides see the split's own prefix. It reports the worst relative logit error,
and for any step where the top token differs, the reference's margin there.
That separates a near-tie (a margin no larger than the error) from a real
bug. Qwen3 on 4 ranks (Metal, CUDA, ROCm, iPad Metal) gave 3.2e-4 with no
differing step; on 2 ranks, 3.8e-4 on Metal and 4.4e-4 on the ANE.

The iOS node builds with unwinding panics when `--model` is given (the
`release` profile, not `apple`). Lane fallback drops a failing lane by
catching its panic, and `panic = "abort"` turned a Core ML "missing param"
into the whole app dying. A lane listed in `RLX_TRAIN_LANES` that the binary
lacks is now named and skipped, rather than silently skipped.

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
  emulated nodes). See rig notes in `rig.sh`.

A worked three-way run (Mac coordinator on Metal + a Linux worker on CPU + a
QEMU/Docker worker on aarch64, all joined by dial-out star with rendezvous
discovery) is exactly what `dist_node MODE=fft` and `dist_job` demonstrate.

## Non-desktop nodes — phones, embedded hosts, FPGAs

A node is not necessarily a workstation. The transport layer has **no platform
gating and no default C dependencies**, so anything with a CPU, a TCP stack and
`std` can be a rank. What used to block non-desktop nodes was packaging, not
capability: the node driver lived only in `rlx-collectives/examples/dist_node.rs`,
and an *example* cannot be linked into an Android `.so` or an iOS framework.

`rlx_runtime::dist::node` is that driver as a library — fully programmatic, so a
platform with no shell environment can still configure a rank:

```rust
use rlx_runtime::dist::node::{NodeConfig, NodeControl, serve_worker};

let cfg = NodeConfig::new(1, 2)
    .peers(["192.168.1.10:29500", "192.168.1.11:29501"])?
    .device("auto");                 // or "cpu", "metal", "gpu", …
let group = cfg.connect()?;
let report = serve_worker(&group, |_uri| Vec::new())?;
```

`NodeControl` bounds the serving loop with a step budget and a stop flag another
thread can raise — a mobile node must be able to leave the mesh when the app is
backgrounded. The flag is observed **between** activations: a node parked in
`recv` exits when its peer sends or the link drops.

### A physical iPad in the cluster

The node app on a real iPad (iPad A16, 6 GB) joined mm4 + msi + amd as a
fourth rank, launched by `cargo rlx dist` like any other host:

- **Inference:** Qwen3-0.6B split over four GPUs by measured speed (Metal
  3025, CUDA 797, ROCm 869, iPad Metal 537 GFLOP/s). The iPad held the last
  stage, the LM head, and the tokens matched one CPU's ("Paris. The capital of
  Italy is Rome").
- **Training:** SmolLM2-135M LoRA, data-parallel over four GPU lanes, with
  Metal, CUDA, ROCm and the iPad's Metal each training its 40 samples.
  Qwen3-0.6B does not fit: iOS killed the app (signal 9) while it received
  the frozen weights, and the job stopped with that rank named rather than
  hanging.

What a non-`dist_node` rank has to match:

- **The whole collective sequence.** `dist_node` runs a barrier once
  connected, the speed exchange, the job, then a post-job and a final barrier.
  A rank that skips any of them deadlocks the mesh. The first iPad build went
  straight to the speed exchange and hung all four ranks. It now calls
  `model::distributed::worker`, which is that sequence, and
  `a_worker_rank_matches_dist_nodes_sequence` pins it against rank 0's side.
- **Local network permission.** The first Bonjour publish, not the first TCP
  connection, raised iOS's "find devices on your local network" prompt.
  Until it is allowed, nothing is published.

`ios/build-xcframework.sh --apple --model` builds the node with `mode =
"model"` (rlx-ffi feature `model`).

### Training on a non-desktop node

A node can join a **data-parallel training run**, not just serve inference:

```rust
use rlx_runtime::dist::node::{NodeConfig, serve_trainer_here};

let group = NodeConfig::new(1, 2).peers(["192.168.1.10:29500"])?.connect()?;
let report = serve_trainer_here(&group, rlx_runtime::dist::uri_resolver, false)?;
println!("{} samples, loss {} -> {}",
    report.metrics.samples, report.metrics.first_loss, report.metrics.last_loss);
```

The worker is model-agnostic — it receives a `TrainSpec` and trains a model it
has no code for. The cross-rank gradient reduce rides on the group's own
`all_reduce`, so the node driver needs no dependency on `rlx-collectives`.

Three things that are easy to get wrong, and each fails quietly:

- **Every rank must use the same reduce.** `mean_reduce` is public for that
  reason: the coordinator trains alongside the workers, and a group where one
  side sums while another averages scales the step differently per replica —
  slow divergence, not an error.
- **`push_data` is mandatory for a real node.** A handset has no shared
  filesystem, so the spec's `file://` URIs — paths on the coordinator's disk —
  resolve to nothing. The reduce counts then desync and the barrier deadlocks.
  Only loopback can use `push_data: false`.
- **A training rank cannot drop out partway.** The reduce is a barrier, so a
  node that leaves stalls every other rank. `NodeControl`'s stop flag is
  honoured between inference activations but **not** mid-training-run.

Ask before shipping: `NodeCaps::can_train()` refuses a fixed-function rank,
because a bitstream has no backward pass. Without that check the `TrainSpec`
lands on a tag the board never reads and the coordinator waits forever.

```sh
just demo-coordinator "--world 2 --self-test --train"   # loopback, verifies agreement
just demo-coordinator "--world 2 --train --peers 0.0.0.0:29500"  # wait for a phone
```

Verified on this machine: an **Android emulator trained on GPU (wgpu)** while
the Mac ran CPU — heterogeneous data-parallel, loss 0.2875 → 0.0000; the iOS
simulator likewise. The loopback self-test additionally checks that both ranks
end bit-identical, which only holds if the all-reduce actually ran.

### Shells

| Target | Shell | Notes |
|---|---|---|
| iOS / macOS | `crates/bindings/rlx-ffi` (C ABI) + `ios/` | `ios/build-xcframework.sh` → `RlxNode.xcframework`; Swift wrapper in `ios/Sources/RlxNode`. **Needs `NSLocalNetworkUsageDescription`**, and the multicast entitlement for UDP discovery — see [ios/README.md](../ios/README.md) |
| Android | `android/rlx-jni` + `RlxNode.kt` | `nodeStart` / `nodeStatus` / `nodeStop`; `RlxNode.start(discovery = true)` takes a `MulticastLock`, without which UDP discovery silently finds no peers |
| Embedded Linux host | `crates/bindings/rlx-ffi` (staticlib) | Same C ABI; the host for an FPGA rank |

`rlx-ffi` builds a `staticlib` only. A `cdylib` must resolve every symbol at
link time and `zstd-sys` (pulled in by the GGUF reader) does not on iOS.

### Demo apps

Runnable demos for both mobile platforms, plus the desktop coordinator they
pair with.

```sh
just demo-coordinator          # desktop side, loopback self-test (no phone)
just demo-ios                  # xcframework → XcodeGen project → simulator build
just demo-android              # cross-build the .so → APK
```

The coordinator ships a parameter-free stage (`gelu(x * 3 + 1)` over 64
elements), streams activations around the ring, and checks each result against
an f64 reference — so "it connected" and "it computed the right thing" are
separate, visible outcomes. With two phones the activation is transformed
twice, which makes it a pipeline rather than a round trip.

To involve a handset, give it the coordinator's address:

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --peers 0.0.0.0:29500
```

A worker only needs **one** address — the coordinator's. It dials out and
nothing dials it back, because most Wi-Fi will not route inbound connections to
a handset. That is why the mobile shells default to a star: a lone peer address
(or discovery) selects `Topology::Star`, while a full per-rank list selects a
mesh.

Both apps are scriptable, so a run needs no UI driving:

```sh
# iOS — UserDefaults picks up `-key value` launch arguments
xcrun simctl launch <sim> com.mit.rlx.nodedemo \
    -rank 1 -world 2 -peers 127.0.0.1:29500 -autojoin YES

# Android — MainActivity forwards the extras (NodeActivity stays unexported)
adb shell am start -n com.mit.rlx.demo/com.mit.rlx.MainActivity \
    --ei rank 1 --ei world 2 --es peers 10.0.2.2:29500 --ez autojoin true
```

The iOS simulator shares the host network stack, so `127.0.0.1` reaches a
coordinator on the Mac. **An Android emulator does not** — its loopback is its
own, and the host is `10.0.2.2`.

Verified end-to-end on this machine: an iOS simulator joined as
`rank 1 — ios [worker] devices: cpu` and an Android emulator as
`rank 1 — android [worker] devices: cpu, gpu`, both matching the reference to
6e-8.

### FPGA — pre-synthesized fixed-function ranks

An FPGA cannot host a runtime. `rlx-fpga` is a Verilog **code generator**; there
is no `Device::Fpga` and it deliberately does not implement the `Backend` trait,
because synthesis takes minutes and a shipped graph would need it in
milliseconds.

So an FPGA joins as a rank that advertises the *one* datapath its bitstream
implements, and the coordinator only ever assigns it that stage:

```rust
use rlx_runtime::dist::node::{FixedFunction, serve_fixed_function_n, NodeControl};

let ff = FixedFunction::new("relu4-int8-v1", 4, 4, DType::F32);
serve_fixed_function_n(&group, &ff, &mut board, &NodeControl::unbounded())?;
```

Two guards, because a wrong feed clocked into fabric produces numbers rather
than an error:

- **At placement** — `NodeCaps::accepts(&spec)` refuses a `StageSpec` whose
  `stage_id` does not match the bitstream, *and* refuses an untagged spec. The
  coordinator calls `collect_caps` before shipping anything.
- **At execution** — the serving loop checks every activation's length against
  the descriptor before touching the board.

The board itself is behind the `FixedFunctionDevice` trait (one method:
`execute`), so the link is board-specific — PCIe, AXI on a Zynq PS, USB, UART —
while the mesh side stays testable. `LoopbackFixedFunction` is the software
stand-in used by `crates/core/rlx-runtime/tests/dist_node_fixed_function.rs`,
which runs a real 3-rank mesh (coordinator → general worker → fixed-function
rank) with no hardware attached.

**Writing the `FixedFunctionDevice` for a specific board is still yours to do** —
that part cannot be validated without the hardware.

### Verifying

`just check-nodes` cross-compiles the whole node stack for iOS and Android and
runs the node tests. Android is skipped, loudly, if no NDK is installed.

## Where things live

| Path | Role |
|------|------|
| `crates/core/rlx-driver/src/transport.rs` | `Transport` trait, `ProcessGroup`, collectives, `ReduceMode` (Ring / Deterministic) |
| `crates/core/rlx-driver/src/net.rs` | `TcpTransport` (mesh), `NetTransport` (star), symmetric heap |
| `crates/core/rlx-driver/src/node.rs` | `Node` builder + discovery (static / mDNS / rendezvous) |
| `crates/core/rlx-driver/src/iroh_transport.rs` | `IrohTransport` (QUIC + relay, NAT traversal by `EndpointId`) + `process_group_from_env` — feature `iroh`; see [iroh-transport.md](iroh-transport.md) |
| `crates/core/rlx-collectives/` | in-graph collectives (`all_reduce`, `moe_dispatch` / `moe_combine`, …) + examples |
| `crates/core/rlx-collectives/examples/wide_ep_toy.rs` | WideEP Phase 0: 2-expert / 2-rank padded EP MoE parity |
| `crates/core/rlx-runtime/src/dist/` | ship-graph `{inference, training, diagnostics}` submodules + shared weight resolvers (`mod.rs`) |
| `crates/core/rlx-runtime/src/dist/node.rs` | platform-neutral node driver: `NodeConfig`, `NodeControl`, `serve_worker`, fixed-function (FPGA) ranks |
| `crates/bindings/rlx-ffi/` | C ABI for the node — iOS `staticlib` + embedded C hosts |
| `ios/` | `xcframework` build + Swift wrapper + entitlement notes |
| `android/rlx-jni/` | JNI node entry points; `RlxNode.kt` owns the multicast lock + lifecycle |
| `crates/bindings/rlx-ffi/examples/node_coordinator.rs` | desktop coordinator for the mobile demos (`--self-test` runs the workers on loopback) |
| `ios/Demo/` | SwiftUI node demo (XcodeGen `project.yml`) |
| `android/app/.../NodeActivity.kt` | Android node demo screen |
| `crates/core/rlx-collectives/DISTRIBUTED_ROADMAP.md` | the deeper tiers (NCCL/RCCL, UCX/RDMA, NVSHMEM) |
## License

MIT OR Apache-2.0.
