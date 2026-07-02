# Distributed RLX — integration seams (Tiers 2–5)

**Implemented and validated** (Mac↔msi rig + emulated aarch64 under QEMU):

- **Tier 1** — bandwidth-optimal ring all-reduce, coalesced wire, grad-accum +
  non-blocking overlap (ring + coalesce ≈ 2× effective link throughput,
  ~1.35× lower latency; see `examples/dist_node.rs` `bench` mode).
- **Tier 4/5 decision layer** — topology `planner`, pipeline-parallel demo,
  UDP auto-discovery, comm profiler.
- **Edge tier** — typed collectives (`all_reduce_typed`: f16/bf16 SIMD, >2× fp16
  in release; **int8/u8** for MCU/FPGA/ANE), **bounded-staleness federated
  averaging** (drops stragglers), **NAT-friendly dial-out star** transport
  (workers dial the coordinator, no inbound port), and the **coordinator/worker
  ship-graph** model (serialized IR + weight-source URIs).
- **Raspberry Pi (aarch64)** — the collective layer *and* the full compute path
  (`rlx-cpu` SIMD gemm fallback, no OpenBLAS) cross-compile to a static
  aarch64-musl binary and run under QEMU. One command: `./rig.sh test-pi`.
- **DX — one-call node bring-up.** `rlx_driver::Node::from_env()?.connect()?`
  returns a ready `ProcessGroup`, hiding mesh-vs-star, peer discovery, and Arc
  wrapping. Env: `RANK`/`WORLD` + `PEERS` or `DISCOVER=1`, `TOPOLOGY=mesh|star`.
- **Build the worker once, run any model.** `rlx_runtime::dist` is a reusable,
  **model-agnostic** ship-graph worker/coordinator: the coordinator ships a
  `StageSpec` (serialized subgraph + I/O names + weight-source URIs + device);
  the worker (`dist::serve_stage` / `recv_stage`) compiles and runs it. Weights
  load **locally** via built-in `gguf://` / `safetensors://` / `file://`
  resolvers behind a parse-once `WeightCache` (one GGUF parse serves all a
  stage's tensors), with a caller closure for other schemes. `packed` weights
  load as **quantized bytes** into a `U8` param for a `DequantMatMul` graph —
  native quant memory (e.g. Q4_K), no dequant round-trip. No model is baked
  into the worker binary → **no per-node / per-model recompile**. This is how
  `../rlx-models` runs across nodes: build one generic worker per arch, drive
  any model from the coordinator.

The tiers below are **designed, not yet validated**, because the hardware isn't
on this rig (no second NVIDIA node, no RDMA NIC / InfiniBand, WiFi-only). The
seams already exist — none of this is a rewrite.

---

## Tier 2 — device-resident collectives (NCCL / RCCL)

**Why:** today `collective.all_reduce` only has a CPU kernel, so a GPU
all-reduce round-trips `GPU→host→TCP→host→GPU`. NCCL/RCCL keep the reduction
on-device (NVLink / PCIe P2P / GPUDirect-RDMA) with ring/tree algorithms.

**Template that already exists:** `rlx-mlx/src/distributed.rs`.
`MlxTransport` (a) implements `rlx_driver::Transport`, (b) exposes native
`all_sum`/`all_gather` delegating to MLX's distributed backend, and (c)
registers a device-resident `collective.all_reduce` kernel. Mirror it.

**Seam (rlx-cuda):**
1. Add `cudarc`'s `nccl` feature; create `rlx-cuda/src/distributed.rs`.
2. `struct NcclComm { comm: cudarc::nccl::Comm, rank, world }`, built from a
   bootstrap `ncclUniqueId` broadcast over an existing `rlx_driver::Transport`
   (reuse `TcpTransport` or the UDP discovery in `dist_node`).
3. Keep a `OnceLock<RwLock<HashMap<u64, NcclComm>>>` group registry keyed by
   group id — identical shape to `groups()` in `lib.rs`.
4. Register a CUDA kernel for `ALL_REDUCE` (`"collective.all_reduce"`) that
   reads the group id from `attrs[..8]`, looks up the `NcclComm`, and calls
   `ncclAllReduce` on the device buffer in-place. No host copy.
5. `rlx-rocm` is the same source against RCCL (the crates already share
   `rlx-gpu-kernels`).

**Cannot validate here:** NCCL is NVIDIA-only, so a Mac(Metal)↔msi(CUDA)
all-reduce is fundamentally impossible; and msi has a single GPU, so even
intra-node NCCL has nothing to talk to. Needs ≥2 NVIDIA GPUs or ≥2 NVIDIA
nodes on a real fabric.

---

## Tier 3 — RDMA / UCX transport, NVSHMEM one-sided

**Why:** the measured crossover (~3×10⁵ FLOP/byte over WiFi) is set by the
link. RDMA over IB/RoCE or Thunderbolt collapses it ~1000×, which is what
makes tensor-parallel inference viable (and is exactly exo's TB5-RDMA bet).

**Seam — it is just another `Transport`:** `ProcessGroup` is generic over
`Arc<dyn Transport>`; `NetTransport` is one impl. Add:

- `UcxTransport` (recommended): one `ucx`/`ucp` API auto-selects shared-mem /
  TCP / RDMA / CUDA-IPC. Implement `Transport::{send_bytes,recv_bytes}` over
  UCP tag-send/recv, and `SymmetricTransport::{put,get}` over UCP RMA. Drop-in
  next to `TcpTransport`; `dist_node` only changes which constructor it calls.
- `libfabric` (OFI) or raw `libibverbs` are lower-level alternatives behind
  the same trait.

**NVSHMEM maps 1:1 onto the existing `SymmetricTransport`.** That trait is
already a symmetric heap with one-sided `put`/`get`/`barrier`
(`rlx-driver/src/symmetric.rs`, `net.rs::NetTransport`). An `NvshmemTransport`
implements the same three methods with `nvshmem_putmem`/`getmem`/`barrier` on
the device symmetric heap — no new abstraction, the seam was designed for it.

**Thunderbolt:** `ThunderboltTransport` exists but is TCP-over-TB today. The
RDMA-over-TB path slots in behind the unchanged `Transport`/`SymmetricTransport`
traits (the type already documents this intent).

**Cannot validate here:** no RDMA-capable NIC, no InfiniBand/RoCE, the two
boxes share only a WiFi/GbE LAN. The TCP-over-Thunderbolt path *could* be
benchmarked if the machines are bridged (offered separately).

---

## Tier 4/5 — implemented vs. remaining

| Item | Status |
|------|--------|
| Topology-aware planner (`src/planner.rs`) | ✅ implemented + 5 unit tests |
| Pipeline-parallel demo (`MODE=pipeline`) | ✅ validated Mac/Metal→msi/CUDA |
| UDP auto-discovery (`DISCOVER=1`) | ✅ validated cross-machine |
| Comm profiler (`MODE=bench` → α/β/crossover) | ✅ implemented + feeds planner |
| Ring/tree as `ProcessGroup` default | ✅ ring shipped (T1) |
| Typed collectives — f16/bf16 (SIMD) + int8/u8 | ✅ `all_reduce_typed`, validated incl. under QEMU aarch64 |
| Bounded-staleness federated averaging | ✅ `federated_average` (drops late workers), `MODE=federated` |
| Dial-out star transport (NAT-friendly workers) | ✅ `coordinator_listen`/`worker_dial`, validated Mac↔msi + `DIAL_OUT=1` |
| Coordinator/worker ship-graph (serialized IR + weight URIs) | ✅ `MODE=coordinator`/`worker`; `seed://`/`file://`/`safetensors://`/`gguf://` sources, parse-once cache |
| Quantized-on-device stage (`WEIGHTS_FMT=q8`) | ✅ Q8_0-packed GGUF → `U8` param → `DequantMatMul` across the mesh (Metal + CPU), matches dequant reference |
| aarch64 / Raspberry Pi (cross-compile + QEMU) | ✅ BLAS-optional `rlx-cpu`; `./rig.sh cross-pi` / `test-pi` |
| Per-region multi-backend placement | ✅ `MODE=multidev` (region→device, host transfer at boundary) |
| 1F1B / interleaved pipeline scheduler + bubble model | ⏳ planner treats PP as ideal; needs micro-batch schedule |
| ZeRO/FSDP optimizer-state sharding | ⏳ design only |
| MPI-shaped launcher / `rsmpi` interop | ⏳ `dist_node` + UDP discovery is the minimal launcher |
| NIXL (KV/activation transfer for disaggregated inference) | ⏳ design only — NVIDIA lib, not testable here |

---

## One-line summary

The collective **algorithm** (Tier 1), the **decision layer** (planner /
profiler / discovery / pipeline), the **edge tier** (typed + int8 collectives,
federated averaging, dial-out), and the **aarch64 / Raspberry Pi** path (build
+ QEMU) are done and measured. The remaining tiers are **library adoptions
behind interfaces that already exist** — `Transport`/`SymmetricTransport` for
UCX/RDMA/NVSHMEM, backend kernel registration (à la `rlx_mlx::distributed`) for
NCCL/RCCL — gated only by hardware this rig doesn't have.
