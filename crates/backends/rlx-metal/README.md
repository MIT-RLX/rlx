# rlx-metal

Apple GPU backend for RLX — MSL kernels + MPSGraph + ICB-batched
dispatch. Two coexisting strategies:

1. **Thunk path** (`thunk/`) — per-op MSL kernel + dispatch. Fine
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
- **`thunk/`** — Thunk enum + Op→Thunk lowering (`ThunkSchedule` split into
  submodules; see `mod.rs`).
- **`backend/`** — top-level Backend impl + execution; `MetalExecutable`
  decomposed into `compile`/`run`/`encode`/`bind`/`set`/`read`/`output`
  submodules (each a sibling `impl MetalExecutable`; see `mod.rs`).
- **`calibrate.rs`** — measured GFLOP/s per kernel variant; cached in
  `~/.cache/rlx/metal-calib-<hwid>.json`. Uses `rlx_ir::Tick`.
- **`cost.rs`** — cost model that consumes calibration values.
- **`device.rs` / `arena.rs`** — Metal device + buffer arena.
- **`op_registry`** — `MetalKernel` trait + `register_metal_kernel` for
  downstream custom ops.
- **FFT** — `fft_gpu.msl` pow-2 path: large N goes multi-kernel (bit-reverse
  gathers src→dst on-GPU — so `src` may be a preceding GPU op like rfft's
  `Concat` — then tiled inner + radix-4/2 outer stages); under `native-gpu-fft`,
  N ≤ 4096 uses an on-chip single-kernel radix-2/4/8 transform plus real-input
  (`Concat([sig, zeros])`) fusion. `Op::Fft` thunk / host fallback for
  f64 / C64 / non-pow2. MPSGraph skips graphs containing `Op::Fft`; `fft_real`
  subgraphs route through thunks automatically.
- **GGUF dequant** (`dequant_gguf.msl` + `backend::encode_dequant_gguf`) —
  on-device dequant for every GGUF scheme (ids 0–23), including **Q4_1**
  (21) and **Q5_0 / Q5_1** (22 / 23).
  (id 21). IQ-family schemes consult a 33 KB grid LUT staged at session
  init (`Kernels::iq_grid_buffer`). Query runtime coverage with
  [`backend::has_metal_dequant_kernel`].
- **Fused decode GEMV** — single-pass matvec when `m == 1` (skips f32 scratch):
  Q4_K, Q4_0, Q4_1, Q8_0, IQ4NL, IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M (`dequant_gguf.msl`).
  See [docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md) for shape
  constraints and disable env vars.
- **FP8 / NVFP4 block matmul** — `dequant_matmul_fp8` / `dequant_matmul_nvfp4`
  MSL for non-GGUF `QuantScheme::Fp8*` / `Nvfp4Block` (CPU fallback when
  deferred host ops are pending).

Full backend matrix: [docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md).
Parity vs `rlx_gguf` on real Qwen3-0.6B weights:
[`tests/iq_full_real_weights.rs`](tests/iq_full_real_weights.rs),
[`tests/iq_mv_parity.rs`](tests/iq_mv_parity.rs),
[`tests/iq4_dequant_parity.rs`](tests/iq4_dequant_parity.rs),
[`tests/q8_q4_dequant_parity.rs`](tests/q8_q4_dequant_parity.rs).

## Cargo features

| Feature | Description |
|---------|-------------|
| `native-splat` (default) | RLX-owned MSL tile raster (`splat.msl`) + CPU project/bin/sort via `slang-splat-ref` (Rust reference, no Slang compiler). |

The crate is built unconditionally on macOS via [`rlx`](https://crates.io/crates/rlx)'s
`metal` feature (which enables `native-splat`); on other platforms it stubs out at link time.

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
- `RLX_METAL_DEQUANT_GPU_DISABLE=1` — CPU GGUF dequant instead of MSL.
- `RLX_METAL_Q4K_FUSED_DISABLE=1` / `RLX_METAL_Q4K_SG_DISABLE=1` —
  disable fused Q4_K GEMV paths.
- `RLX_METAL_Q40_FUSED_DISABLE=1` / `RLX_METAL_Q41_FUSED_DISABLE=1` /
  `RLX_METAL_Q80_FUSED_DISABLE=1` — disable fused Q4_0 / Q4_1 / Q8_0 GEMV.
- `RLX_METAL_IQ4NL_FUSED_DISABLE=1` and `RLX_METAL_IQ{2,3,1}*_FUSED_DISABLE=1` —
  disable fused IQ-family GEMV paths.

See [docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md) for the
full GGUF env table.

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
