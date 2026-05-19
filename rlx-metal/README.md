# rlx-metal

Apple GPU backend for RLX — MSL kernels + MPSGraph + ICB-batched
dispatch. Two coexisting strategies:

1. **Thunk path** (`thunk.rs`) — per-op MSL kernel + dispatch. Fine
   control, mature; default for now.
2. **MPSGraph path** (`mps_graph.rs` + `mps_graph_lower.rs`) — lower
   subgraphs to MPSGraph and let Metal optimize the schedule. Opt-in
   per-op (e.g., `RLX_MPSGRAPH_ATTENTION=1`). Phase J extended this to
   Concat / FusedSwiGLU / RoPE cos-sin slice.

ICB (Indirect Command Buffer) batching (`icb.rs`) is the cross-cutting
throughput unlock — Phase H made matmul-interleaved schedules use it.

## What's here

- **MSL kernel library** (`kernels.rs`, 2.2k LOC) — softmax, layer norm,
  RMS norm, attention, fused SwiGLU, RoPE, BERT-layer fusion. f16/f32
  via a `HalfFlag` dispatch (Phase F). Phase I added f16 variants for
  rms_norm / softmax / reduce.
- **MPSGraph bridge** — opt-in lowering to Apple's high-level graph
  compiler for the attention / concat / SwiGLU / RoPE-cos-sin paths.
- **MPS BLAS** (`mps_blas.rs`) — descriptor-cached MPS matrix multiply.
- **ICB (Indirect Command Buffer) batching** — segmented matmul
  schedules issue as one indirect dispatch instead of N command buffers.
- **`thunk.rs`** — Thunk enum + Op→Thunk lowering.
- **`backend.rs`** — top-level Backend impl + execution.
- **`calibrate.rs`** — measured GFLOP/s per kernel variant; cached in
  `~/.cache/rlx/metal-calib-<hwid>.json`. Uses `rlx_ir::Tick`.
- **`cost.rs`** — cost model that consumes calibration values.
- **`device.rs` / `arena.rs`** — Metal device + buffer arena.
- **`op_registry`** — `MetalKernel` trait + `register_metal_kernel` for
  downstream custom ops.

## Cargo features

The crate is built unconditionally on macOS via [`rlx`](https://crates.io/crates/rlx)'s
`metal` feature; on other platforms it stubs out at link time.

## Install

```toml
[dependencies]
rlx-metal = "0.1"
```

Or, more typically:

```toml
[dependencies]
rlx = { version = "0.1", features = ["metal"] }
```

## Build / test

```sh
cargo build -p rlx-metal --release
cargo test  -p rlx-metal --release
```

Gating env vars worth knowing:

- `RLX_MPSGRAPH_ATTENTION=1` — opt into MPSGraph attention lowering
  (otherwise thunks).
- `RLX_VERBOSE=1` — calibration log.

## Status

Mature for the BERT / Nomic inference path used in burnembed. ICB
matmul + MPSGraph attention are production. Tier-2 fused ops
(FusedAttnBlock, FusedBertLayer) work; FusedNomicLayer is disabled
pending a SwiGLU stride fix (see `thunk.rs:3315`).

## Gotchas

- Per-run cost is dominated by `wait_until_completed` (~150 µs);
  encoding cost is comparatively small. Fusing op chains into one
  command buffer is far more valuable than reducing kernel count.
- `Thunk::Attention` only supports `MaskKind::Custom` (plan #20). The
  lowering asserts; non-Custom kinds are a future kernel addition.
  MPSGraph attention bails to thunks for non-Custom.
- Don't trust microbenchmarks under thermal throttle. Run
  `scripts/check-throttle.sh` before measuring.
- Phase G eliminated the f32↔f16 cast tax inside AutoMixedPrecision;
  follow-on work that adds new ops should respect the registry of
  natively-half kernels.

## License

GPL-3.0-only.
