# GGUF backend execution paths

How `Op::DequantMatMul { scheme }` and grouped GGUF matmul reach each RLX
backend (current tree). For the full op matrix see
[op-coverage.md](op-coverage.md); for scheme definitions see
[`crates/rlx-ir/src/quant.rs`](../crates/rlx-ir/src/quant.rs).

## IR contract

| Input layout | Schemes |
|--------------|---------|
| 4 inputs: `x`, `w_q`, `scale`, `zp` | `Int8Block`, `Int8BlockAsym`, `Int4Block`, `Fp8E4m3`, `Fp8E5m2`, `Nvfp4Block` |
| 2 inputs: `x`, `packed_w` | All `Gguf*` variants (`scheme.is_gguf()`) |

GGUF schemes embed scales, mins, and sub-block metadata inside the packed
weight bytes — no separate scale/zp tensors.

## Shared GPU scheme ids

Metal, CUDA, ROCm, and WGPU use the same integer ids in `dequant_gguf`
kernels (`gguf_scheme_id` / `scheme_from_id` in each backend's `gguf_host`):

| id | `QuantScheme` | Block (elements) | Block (bytes) |
|----|---------------|------------------|---------------|
| 0 | `GgufQ4K` | 256 | 144 |
| 1 | `GgufQ5K` | 256 | 176 |
| 2 | `GgufQ6K` | 256 | 210 |
| 3 | `GgufQ8K` | 256 | 292 |
| 4 | `GgufQ2K` | 256 | 84 |
| 5 | `GgufQ3K` | 256 | 110 |
| 6 | `GgufIQ4NL` | 32 | 18 |
| 7 | `GgufIQ4XS` | 256 | 136 |
| 8 | `GgufTQ1_0` | 256 | 54 |
| 9 | `GgufTQ2_0` | 256 | 66 |
| 10 | `GgufMXFP4` | 32 | 17 |
| 11 | `GgufNVFP4` | 16 | 9 |
| 12–18 | IQ2 / IQ3 / IQ1 family | 256 | varies |
| 19 | `GgufQ4_0` | 32 | 18 |
| 20 | `GgufQ8_0` | 32 | 34 |
| 21 | `GgufQ4_1` | 32 | 20 |
| 22 | `GgufQ5_0` | 32 | 22 |
| 23 | `GgufQ5_1` | 32 | 24 |

Map GGUF file dtypes to IR: [`rlx_cpu::quant_scheme_for_ggml`](../crates/rlx-cpu/src/gguf_scheme.rs)
(`GgmlType` → `QuantScheme` for loader / graph builders).

IQ-family kernels take a ~33 KB grid LUT (see `rlx_gguf::iq_grids`); staged
once per device context on Metal/CUDA/ROCm/WGPU.

## Per-backend summary

| Backend | GPU dequant | Fused GEMV (`m = 1`) | On-device constexpr (ANE) | Host / compile fallback |
|---------|-------------|----------------------|----------------------------|-------------------------|
| **CPU** | — | block-wise fused matmul in `rlx-cpu::gguf_matmul` | — | always available |
| **Metal** | `dequant_gguf.msl` → MPS sgemm | Q4_K, Q4_0, Q4_1, Q8_0, IQ4NL, IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M | MIL `mul`+`sub` for K/IQ/TQ/MX (see ANE) | `RLX_METAL_DEQUANT_GPU_DISABLE=1` |
| **CUDA / ROCm** | `dequant_gguf.cu` → BLAS | — | — | same disable pattern |
| **WGPU** | `dequant_gguf.wgsl` → `matmul_bt`; grouped MoE GPU | — | — | `gguf_host` when scratch exceeds limits |
| **ANE (CoreML)** | hybrid host segments | — | MIL `mul` + optional `sub` | `RLX_COREML_HOST_DEQUANT=1` |
| **TPU** | — | — | — | host dequant at HLO emit (`Constant`, `quant_param_bindings`, or runtime `set_param_typed` on deferred Param) |
| **MLX** | host dequant + cache | — | — | primary Apple path when MLX enabled |

Non-GGUF FP8 / NVFP4 block matmul on Metal uses `dequant_matmul_fp8` /
`dequant_matmul_nvfp4` MSL when no deferred host ops are pending.

## P0–P5 deliverables

| Id | What | Primary files |
|----|------|-----------------|
| **P0** | ANE on-device Q2/Q3/Q6_K (+ per-element `[nb,32]` scales) | `rlx-coreml/src/mil/helpers.rs`, `mod.rs` (`bake_ondevice_weight`) |
| **P1** | WGPU GPU dequant path | `rlx-wgpu/src/kernels/dequant_gguf.wgsl`, `gguf_gpu.rs`, `backend.rs` |
| **P2** | Metal FP8 / NVFP4 GPU dequant matmul | `rlx-metal/src/kernels.rs` (`dequant_matmul_fp8`, `dequant_matmul_nvfp4`) |
| **P3** | `GgufQ4_1` IR scheme + kernel id 21 | `rlx-ir/src/quant.rs`, all `gguf_host.rs`, MSL/CUDA/WGSL |
| **P4** | TPU compile-time GGUF → f32 HLO dot | `rlx-tpu/src/lower.rs` (`lower_dequant_matmul_gguf`) |
| **P5** | Metal fused decode GEMV (Q4_K, Q4_0, Q4_1, Q8_0, IQ4NL, IQ2/3/1 family) | `rlx-metal/src/dequant_gguf.msl`, `backend.rs`, `tests/iq_mv_parity.rs` |

## Backlog follow-up — landed

| Item | What | Primary files |
|------|------|-----------------|
| **B1** | MLX + `dequant_cache` Q4_1 / Q5_0 / Q5_1 | `rlx-mlx/src/lower.rs`, `rlx-cpu/src/dequant_cache.rs` |
| **B2** | WGPU + Metal dequant parity; Metal fused GEMV tests | `rlx-wgpu/tests/gguf_dequant_parity.rs`, `rlx-metal/tests/iq4_dequant_parity.rs`, `iq_mv_parity.rs`, `q40_q80_mv_parity.rs` |
| **B3** | CoreML MIL on-device IQ/TQ/MX + K-quants | `rlx-coreml/tests/mil_p15.rs`, `mil/helpers.rs` |
| **B4** | WGPU grouped MoE GPU path | `rlx-wgpu/src/gguf_gpu.rs` |
| **B5** | `GgufQ5_0` / `GgufQ5_1` IR + kernel ids 22 / 23; `quant_scheme_for_ggml` | `rlx-ir`, MSL/CUDA/WGSL, `rlx-cpu/src/gguf_scheme.rs` |

**Kernel routing note:** legacy 32-element schemes (ids 19–23) must be excluded
from the 256-element K-quant branch guard at the top of `dequant_gguf` (MSL/CUDA/WGSL).
Otherwise Q4_1+ fall through with wrong block stride.

## Metal

**Default path:** `encode_dequant_gguf` writes f32 `[n,k]` scratch, then
`encode_mps_sgemm_bt`.

**Fused GEMV** (skips scratch; decode-only, `m == 1`):

| Scheme | Constraint | Disable env |
|--------|------------|-------------|
| Q4_K | `k % 256 == 0` | `RLX_METAL_Q4K_FUSED_DISABLE=1` |
| Q4_K simdgroup | above + `n % 8 == 0` | `RLX_METAL_Q4K_SG_DISABLE=1` |
| Q4_0 | `k % 32 == 0` | `RLX_METAL_Q40_FUSED_DISABLE=1` |
| Q4_1 | `k % 32 == 0` | `RLX_METAL_Q41_FUSED_DISABLE=1` |
| Q8_0 | `k % 32 == 0` | `RLX_METAL_Q80_FUSED_DISABLE=1` |
| IQ4_NL | `k % 32 == 0` | `RLX_METAL_IQ4NL_FUSED_DISABLE=1` |
| IQ2_XXS | `k % 256 == 0` | `RLX_METAL_IQ2XXS_FUSED_DISABLE=1` |
| IQ2_XS | `k % 256 == 0` | `RLX_METAL_IQ2XS_FUSED_DISABLE=1` |
| IQ2_S | `k % 256 == 0` | `RLX_METAL_IQ2S_FUSED_DISABLE=1` |
| IQ3_XXS | `k % 256 == 0` | `RLX_METAL_IQ3XXS_FUSED_DISABLE=1` |
| IQ3_S | `k % 256 == 0` | `RLX_METAL_IQ3S_FUSED_DISABLE=1` |
| IQ1_S | `k % 256 == 0` | `RLX_METAL_IQ1S_FUSED_DISABLE=1` |
| IQ1_M | `k % 256 == 0` | `RLX_METAL_IQ1M_FUSED_DISABLE=1` |

**Query coverage:** [`has_metal_dequant_kernel`](../crates/rlx-metal/src/backend.rs).

## WGPU

1. Arena planner reserves `dequant_gguf_scratch_bytes(graph)` tail space.
2. If scratch fits `max_buffer_size`, dispatch `dequant_gguf` then `matmul_bt`
   (or grouped MoE GPU path below).
3. Otherwise fall back to `gguf_host` (CPU dequant + wgpu matmul).

**Grouped MoE (`DequantGroupedMatmulGguf`):** when scratch fits, CUDA-style GPU
path — host sort/unpermute, per-expert `dequant_gguf` + `matmul_bt` on sorted
token batches (`run_dequant_grouped_matmul_gguf_gpu` in `gguf_gpu.rs`).

**Limits:**

- Byte offsets in kernels are `u32` (arenas ≥ 4 GB need the host path).
- IQ branches mirror Metal/CUDA; WGPU parity tests cover Q4_0 / Q8_0 / Q4_1
  (`gguf_dequant_parity.rs`); Q5_0 / Q5_1 use the same kernel ids as Metal.
- Grouped MoE IQ/TQ integration: `rlx-runtime/tests/dequant_grouped_matmul_gguf.rs`
  (IQ2/IQ3 XXS+S, TQ2_0, IQ1_S × CPU / Metal / WGPU / CUDA). Run with
  `--test-threads=1` when multiple GPU backends are enabled in one binary.

## ANE (CoreML)

On-device constexpr (`LowerOptions::ondevice_dequant`) uses MIL `mul` and
optional `sub`:

| Schemes | Scale / offset tensor shape |
|---------|----------------------------|
| Q4_0, Q8_0, Q4_1, Q5_0, Q5_1, IQ4NL, IQ4XS, TQ1_0, TQ2_0, MXFP4, Q4_K, Q5_K, Q8_K, IQ2_XXS, IQ3_XXS, IQ3_S, IQ1_S | `[nb, 1]` broadcast over 32 quants |
| NVFP4, IQ2_XS, IQ2_S, IQ1_M | `[nb, 32]` per-element (two NVFP4 halves or sub-group scales within a chunk) |
| Q2_K, Q3_K, Q6_K | `[nb, 32]` per-element (sub-block scales vary within a chunk) |

IQ1_S / IQ1_M use MIL `sub` for the δ nudge (`offset = scale × δ`).

Schemes without a MIL lowering host-dequant at bake time via hybrid segments
or `RLX_COREML_HOST_DEQUANT=1`.

## TPU

GGUF `DequantMatMul` is **not** lowered to on-device dequant HLO. Three paths:

**Compile-time bake** (`Constant` or `CompileOptions::quant_param_bindings` / `LowerParamBytes`):

1. Read packed bytes from `Op::Constant { data }` or the param-bytes map.
2. Host-dequant via `rlx_gguf` (`dequant_gguf_bytes` in `lower.rs`).
3. Embed f32 weights as an HLO constant → `dot_general`.

**Runtime Param** (no compile-time bytes):

1. Lower weight as an f32 HLO parameter `[k, n]` (`HloModule::gguf_deferred`).
2. `set_param_typed(name, packed_u8, U8)` host-dequants before PJRT upload.
3. `dot_general` uses the uploaded f32 slab.

Non-GGUF schemes still use the in-HLO `convert + scale/zp tile + dot` path.

## Environment variables

| Variable | Backend | Effect |
|----------|---------|--------|
| `RLX_METAL_DEQUANT_GPU_DISABLE=1` | Metal | Force CPU GGUF dequant |
| `RLX_METAL_Q4K_FUSED_DISABLE=1` | Metal | Disable Q4_K fused GEMV |
| `RLX_METAL_Q4K_SG_DISABLE=1` | Metal | Disable simdgroup Q4_K GEMV |
| `RLX_METAL_Q40_FUSED_DISABLE=1` | Metal | Disable Q4_0 fused GEMV |
| `RLX_METAL_Q41_FUSED_DISABLE=1` | Metal | Disable Q4_1 fused GEMV |
| `RLX_METAL_Q80_FUSED_DISABLE=1` | Metal | Disable Q8_0 fused GEMV |
| `RLX_METAL_IQ4NL_FUSED_DISABLE=1` | Metal | Disable IQ4_NL fused GEMV |
| `RLX_METAL_IQ2XXS_FUSED_DISABLE=1` | Metal | Disable IQ2_XXS fused GEMV |
| `RLX_METAL_IQ2XS_FUSED_DISABLE=1` | Metal | Disable IQ2_XS fused GEMV |
| `RLX_METAL_IQ3XXS_FUSED_DISABLE=1` | Metal | Disable IQ3_XXS fused GEMV |
| `RLX_METAL_IQ2S_FUSED_DISABLE=1` | Metal | Disable IQ2_S fused GEMV |
| `RLX_METAL_IQ3S_FUSED_DISABLE=1` | Metal | Disable IQ3_S fused GEMV |
| `RLX_METAL_IQ1S_FUSED_DISABLE=1` | Metal | Disable IQ1_S fused GEMV |
| `RLX_METAL_IQ1M_FUSED_DISABLE=1` | Metal | Disable IQ1_M fused GEMV |
| `RLX_COREML_HOST_DEQUANT=1` | ANE | Bake full f32 weights at compile |
| (arena planning) | WGPU | Auto host fallback when scratch does not fit |

## Code map

| Concern | Location |
|---------|----------|
| IR schemes | `crates/rlx-ir/src/quant.rs` |
| CPU reference + MoE | `crates/rlx-cpu/src/gguf_matmul.rs` |
| Parse / dequant | `crates/rlx-gguf/src/lib.rs` |
| GgmlType → IR | `crates/rlx-cpu/src/gguf_scheme.rs` |
| Dequant cache | `crates/rlx-cpu/src/dequant_cache.rs` |
| Metal MSL + encode | `crates/rlx-metal/src/dequant_gguf.msl`, `backend.rs` |
| CUDA / ROCm | `crates/rlx-gpu-kernels/kernels/dequant_gguf.cu`, `rlx-cuda/src/gguf_gpu.rs` |
| WGPU | `crates/rlx-wgpu/src/kernels/dequant_gguf.wgsl`, `gguf_gpu.rs` |
| ANE MIL | `crates/rlx-coreml/src/mil/helpers.rs`, `mod.rs` |
| TPU HLO | `crates/rlx-tpu/src/lower.rs` |
| Python bindings | `crates/pyrlx/src/gguf.rs`, `gguf_convert.rs` |
| Convert CLI / lib | `crates/rlx-gguf-convert/` |

## Maintenance

When adding a GGUF scheme:

1. `QuantScheme` variant + `gguf_block_size` / `gguf_block_bytes` in `rlx-ir`.
2. `dequant_*` in `rlx-gguf`.
3. Branch in MSL, CUDA, and WGPU `dequant_gguf` (next scheme id).
4. `gguf_scheme_id` / `scheme_from_id` in every backend `gguf_host`.
5. CPU `gguf_matmul::dequant_block`, `dequant_cache`, MLX `dequant_gguf_weight`.
6. CoreML: host and/or on-device split in `mil/helpers.rs` if ANE-targeted.
7. Python: `pyrlx.convert_to_gguf` (safetensors → GGUF), `pyrlx.load_gguf` /
   `pyrlx.write_gguf` for direct file I/O.

## Python (`pyrlx`)

No backend required for pack/unpack or file I/O:

```python
import pyrlx as rlx

packed = rlx.quantize(weights_f32, dtype="IQ2_XXS")
back = rlx.dequant(packed, dtype="IQ2_XXS", num_elements=len(weights_f32))

rlx.convert_to_gguf("model.safetensors", "model.q4_k.gguf", "Q4_K", architecture="llama")
f = rlx.load_gguf("model.q4_k.gguf")
w = f.dequant_tensor("token_embd.weight")
```

Build with `maturin develop --features cpu,gguf-convert` (default). Optional
`gguf-onnx` / `gguf-pt` cargo features for ONNX / PyTorch checkpoints.
Tests: `crates/pyrlx/tests/test_gguf_*.py`; `just test-pyrlx`.
7. Parity test (Metal / WGPU / runtime integration as appropriate).
8. Refresh this doc and [op-coverage.md](op-coverage.md).
