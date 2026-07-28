# iroh transport — NAT-traversing distributed RLX

The [`Transport`](../crates/core/rlx-driver/src/transport.rs) layer in
[distributed.md](distributed.md) rides on TCP by default: it needs every rank on
one routable IP network with reachable listen ports. That's exactly the
constraint that forces a `TOPOLOGY=star` (dial-out) workaround when a firewall
or NAT sits between two machines.

**`IrohTransport`** removes that constraint. It carries the same two-sided,
tagged, matched `send`/`recv` surface as every other `Transport` — so every
collective (`all_reduce`, `all_gather`, `broadcast`, `barrier`) and therefore
data-, tensor-, and pipeline-parallel training/inference works over it
unchanged — but over [iroh](https://iroh.computer) QUIC connections that
hole-punch through NAT and fall back to a relay. Ranks are addressed by their
stable ed25519 **`EndpointId`** instead of an `ip:port`, so a job can span
machines on different networks / behind CGNAT with **no port forwarding**.

It is feature-gated (`rlx-driver`'s `iroh` feature) and off by default — the
zero-dep native `TcpTransport` stays the base path.

## Where it sits

`IrohTransport` is a drop-in `Transport`, so nothing above it changes:

```
  your job (rlx-vision-bench / dist_job / a model runner)
        │
  ProcessGroup            all_reduce / all_gather / broadcast / barrier  (unchanged)
        │
  Transport (trait)       TcpTransport (mesh) · NetTransport (star) · IrohTransport ← new
        │                 send_bytes / recv_bytes / barrier
  the wire                TCP · QUIC-over-iroh (relay + hole-punch) ← new
```

Compared to the [node discovery](distributed.md#node-discovery) tiers, iroh adds
a fourth row — reach a peer **by `EndpointId` alone**, across NAT, with no known
meeting point beyond the public relays:

| discovery | mechanism | scope |
|-----------|-----------|-------|
| `static` | hand-listed `ip:port` per rank | any (you address them) |
| `mdns` | mDNS-SD (Bonjour/Avahi) | one LAN, zero-config |
| `rendezvous` | unicast a known coordinator (LAN IP / Tailscale MagicDNS) | across NAT, one meeting point |
| **iroh** | **QUIC + n0 relays + pkarr/DNS discovery; peers found by `EndpointId`** | **across NAT / internet, no coordinator** |

## The transport design

### Async iroh under a sync trait

The `Transport` trait is synchronous and blocking; iroh is async. `IrohTransport`
owns a multi-thread Tokio runtime. A background accept loop drains inbound
frames into a `Mutex + Condvar` mailbox keyed by `(from_rank, tag)`; the blocking
`recv_bytes` parks on that mailbox exactly like a socket reader thread.
`send_bytes` `block_on`s a short task that writes the frame. A `to == self.rank`
send short-circuits straight into the local mailbox (loopback needs no network).

### One ordered stream per directed edge (FIFO)

Each directed edge (sender → receiver) uses **one long-lived, ordered QUIC
uni-stream**, dialed by the sender on first use and cached. Every message is a
length-prefixed frame on it:

```
┌──────────── 12-byte little-endian header ────────────┐
│ from_rank: u32 │ tag: u32 │ len: u32 │  … len bytes …  │
└──────────────────────────────────────────────────────┘
```

Reusing a single ordered stream (rather than one stream per message) preserves
**FIFO order between same-tag frames** — the property the ring collectives
(`all_reduce`) rely on once there are ≥ 3 ranks, where a rank sends several
same-tag frames to a peer before the peer consumes them. QUIC gives no ordering
*across* independent streams, so a stream-per-message design silently corrupts
≥ 3-rank reductions; the per-edge stream fixes it (this is validated by the
3-rank cross-machine test below). `MAX_FRAME_BYTES` caps a frame at 512 MiB.

## Relay & discovery modes

Four constructors, from LAN-direct to fully NAT-traversing. Every rank must pass
the same `alpn` (default `RLX_PIPELINE_ALPN` = `rlx-pipeline/1`).

| constructor | relay mode | discovery | reach a peer by… | use |
|-------------|-----------|-----------|------------------|-----|
| `connect` | `Disabled` | none | id + explicit addr/relay | direct / LAN |
| `connect_relayed` | `Default` (n0 relays) | none | id + relay URL | NAT, relay-addressed peers |
| `connect_with(…, RelayMode)` | explicit (`Disabled`/`Default`/`Staging`/`Custom(RelayMap)`) | none | as configured | custom relay map |
| **`connect_discovered`** | `Default` (n0 relays) | **pkarr + DNS** | **`EndpointId` alone** | **cross-internet, zero addressing** |

`connect_discovered` uses iroh's `N0` preset (public relays + pkarr/DNS
publish+resolve), so each rank publishes its current address and the dialer
resolves a peer from just its id — the "just works across the internet" path,
at the cost of a few seconds of first-dial resolution latency (covered by the
60 s dial-retry window). It needs a TLS crypto provider — iroh's default
`tls-ring` supplies it. `RelayMode`/`RelayMap` are re-exported from `rlx-driver`.

### Peer addressing

A peer is an **`IrohPeer`**: an `EndpointId` plus optional relay URL / direct
socket addresses.

```rust
use rlx_driver::{IrohPeer, IrohTransport, RLX_PIPELINE_ALPN};
use iroh::SecretKey;

// Reach rank 1 by its EndpointId alone (discovery resolves the rest):
let peers = vec![
    IrohPeer::from_id_str("bed7d2ab…")?,           // rank 0
    IrohPeer::from_id_str("353c8a7f…")?,           // rank 1
];
let t = IrohTransport::connect_discovered(rank, world, my_secret_key, peers, RLX_PIPELINE_ALPN)?;
// t.endpoint_id_string() prints this rank's id, to hand to the others.
```

`IrohPeer::new(id)` / `.with_relay(url)?` / `.with_direct(socket_addr)` build a
peer manually for the non-discovery modes.

## The env-driven `ProcessGroup` — `process_group_from_env`

The NAT-traversing analog of [`Node::from_env().connect()`](distributed.md#the-layers).
Returns a ready `Arc<ProcessGroup>` whose collectives ride iroh, from
environment variables — so a launcher needs no iroh code:

| env var | meaning |
|---------|---------|
| `RANK`, `WORLD` | this rank / total ranks (required) |
| `RLX_IROH_SEED=<hex>` | a **shared** seed → per-rank keys derived deterministically, so every rank knows every `EndpointId` from the seed alone. Zero-config; **not secret** (anyone with the seed can impersonate a rank) — for a *trusted* cluster / demo. |
| `RLX_IROH_PEERS=<id>,<id>,…` | **or:** each rank's `EndpointId` (hex), indexed by rank… |
| `RLX_IROH_SECRET=<64-hex>` | …together with this rank's secret key (whose public id must equal `PEERS[RANK]`). The secure path. |
| `RLX_IROH_ALPN=<str>` | optional ALPN (default `rlx-pipeline/1`) |

It uses `connect_discovered`, so peers are reached by `EndpointId` alone.

```rust
let group = rlx_driver::process_group_from_env()?;   // Arc<ProcessGroup> over iroh
```

## The training topology — `TOPOLOGY=iroh`

`rlx-vision-bench` (the canonical data-parallel trainer) selects the transport at
launch. In its multi-node path (`run_node_from_env`, one process = one rank):

- **`TOPOLOGY=iroh`** (or `RLX_TRANSPORT=iroh`) → build the group with
  `process_group_from_env()` (iroh, relays + discovery).
- otherwise → the default TCP/Thunderbolt path via `Node::from_env()` (which
  itself honours `TOPOLOGY=mesh|star`).

Enable the path with the crate's `iroh` feature (`iroh = ["rlx-driver/iroh"]`);
without it, `TOPOLOGY=iroh` errors with a rebuild hint. The gradient all-reduce
is transport-agnostic, so **training is identical either way — only the wire
under the collectives changes**.

Two orthogonal knobs complete the picture:

- **`RLX_DEVICE`** (`metal` / `cuda` / `mlx` / `cpu`, else the fastest backend
  compiled in and live) — each rank picks its own compute device, so a
  **heterogeneous** cluster (Mac on Metal, a CUDA box on its GPU) just works.
  The collective host-delegates to the CPU on every GPU backend, so the pattern
  is **GPUs for compute (speed) + CPU for the reduce (precision)** automatically.
- **`RLX_DETERMINISTIC_REDUCE=1`** — the f64 deterministic ring reduce
  (correctly-rounded, world-size-independent cross-rank sum) — see
  [Reproducible & precise gradient reduction](distributed.md#reproducible--precise-gradient-reduction).

## Run recipes

Every node runs one process; ranks rendezvous at the first collective.

```bash
# Same LAN or cross-internet — reach peers by EndpointId, no ip:port, no port
# forwarding. Share one hex seed; each node gets a distinct RANK.
# Mac:
RANK=0 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 \
  cargo run -p rlx-vision-bench --features iroh -- --epochs 5
# CUDA box (behind an inbound-blocking firewall — traversed via relay):
RANK=1 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 \
  cargo run -p rlx-vision-bench --features iroh -- --epochs 5
```

**Hybrid GPUs + CPUs, maximum speed + precision** — each rank on its fastest GPU,
gradients reduced in f64 on the host:

```bash
# Mac rank 0 on Metal:
RANK=0 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 \
  RLX_DEVICE=metal RLX_DETERMINISTIC_REDUCE=1 \
  cargo run -p rlx-vision-bench --features iroh -- --model cnn --epochs 5

# CUDA box rank 1 (cudarc dlopens the driver; add cuDNN for fast conv):
RANK=1 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 \
  RLX_DEVICE=cuda RLX_DETERMINISTIC_REDUCE=1 \
  LD_LIBRARY_PATH=/path/to/cudnn/lib:/usr/local/cuda/lib64 \
  cargo run -p rlx-vision-bench --features iroh -- --model cnn --epochs 5
```

## Validated

Cross-machine over the public internet, **Mac (arm64) ↔ a Linux CUDA box (x86_64)**, with its firewall **blocking inbound** — the exact case the
TCP path needed `TOPOLOGY=star` for:

- **Reachability**: rank 0 dialed rank 1 by `EndpointId` alone (`relay_url: None,
  direct_addrs: []`) — pkarr discovery + relay traversed the firewall,
  bidirectionally, sub-second once resolved.
- **Collectives**: 2-rank and 3-rank data-parallel SGD (rank 0 on Mac, ranks
  1–2 on the Linux box) converged in lockstep to the exact cross-rank mean — the 3-rank
  run is what exercises (and validates) the per-edge FIFO stream.
- **MNIST DP training**: all-CPU **89.76 %**; hybrid **Metal (Mac) + CUDA**
  MLP **89.83 %** (the 0.07 % delta is the heterogeneous-GPU f32 signature — the
  f64 reduce keeps the sync exact); CNN with **Metal MPS conv + CUDA cuDNN conv**
  **94.96 %**. All with `RLX_DETERMINISTIC_REDUCE=1`.

Note: for small MNIST models the wall-clock is **WAN-comm-bound** — a synchronous
per-batch all-reduce across the internet dominates, so GPU compute speed shows in
`compute_s`/throughput, not wall time. Use `--async` (overlap the reduce with the
next batch), larger batches, and gradient bucketing to make it compute-bound.

## Feature flags & runtime deps

- `rlx-driver` **`iroh`** feature → `dep:iroh` + `dep:tokio` (both optional). Off
  by default. `IrohTransport`/`IrohPeer`/`process_group_from_env`/`RelayMode` are
  exported only under it.
- `rlx-vision-bench` **`iroh`** feature → forwards to `rlx-driver/iroh` and enables
  the `TOPOLOGY=iroh` branch.
- iroh is pinned to `=1.0.0-rc.1` (the inspected release); bump to the 1.0 stable
  line as needed.

## Caveats

- **Discovery latency**: `connect_discovered` waits on pkarr publish→propagate→
  resolve for the first dial (seconds); the 60 s dial-retry window covers it.
  Start peers before the joining rank, or accept the first-dial delay.
- **`RLX_IROH_SEED` is not secret** — deterministic keys from a shared seed are
  for trusted clusters/demos. Use `RLX_IROH_PEERS` + per-rank `RLX_IROH_SECRET`
  (persisted stable keys) on untrusted networks.
- **Relay dependency**: the discovery/relayed modes rely on n0's public relays.
  For a private deployment pass `RelayMode::Custom(RelayMap)` via `connect_with`.
- Teardown closes the endpoint with a bounded (2 s) `close()` so `Drop` can't
  hang if the peer left first.

## Where things live

| Path | Role |
|------|------|
| `crates/core/rlx-driver/src/iroh_transport.rs` | `IrohTransport` (async↔sync bridge, per-edge FIFO streams, wire framing), `IrohPeer`, `connect{,_relayed,_with,_discovered,_ephemeral}`, `process_group_from_env`, `RLX_PIPELINE_ALPN` |
| `crates/core/rlx-driver/src/transport.rs` | the `Transport` trait + `ProcessGroup` it plugs into |
| `crates/core/rlx-driver` `iroh` feature | gates all of the above (`dep:iroh`, `dep:tokio`) |
| `../rlx-models/crates/rlx-vision-bench` (`iroh` feature) | `build_multinode_group()` / `TOPOLOGY=iroh` + `training_device()` (`RLX_DEVICE`) — the launcher wiring |
## License

MIT OR Apache-2.0.
