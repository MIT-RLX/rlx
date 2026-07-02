# rlx-gguf

GGUF v1 / v2 / v3 parser, dequantization, **quantization encoders**,
and a **file writer**. Standalone — no `rlx-*` deps; usable from any
Rust ML project.

## Supported quantizations

| Format | Block size | Bits / elem | Decode | Encode | Notes |
|---|---|---|---|---|---|
| `F32`, `F16`, `BF16` | n/a | 32 / 16 / 16 | ✅ | ✅ | direct read/write |
| `Q8_0` | 32 | 8.5 | ✅ | ✅ | f16 scale + 32 i8 |
| `Q4_0` / `Q4_1` | 32 | 4.5 / 5 | ✅ | ✅ | per-block scale (+ min for Q4_1) |
| `Q5_0` / `Q5_1` | 32 | 5.5 / 6 | ✅ | ✅ | adds 5th bit via packed `qh` |
| `Q2_K` | 256 | 2.625 | ✅ | ✅ | super-block + packed 4-bit scales/mins |
| `Q3_K` | 256 | 3.4375 | ✅ | ✅ | super-block + signed 6-bit scales |
| `Q4_K` | 256 | 4.5 | ✅ | ✅ | super-block + 8 packed sub-scales/mins |
| `Q5_K` | 256 | 5.5 | ✅ | ✅ | super-block + high-bit plane |
| `Q6_K` | 256 | 6.5 | ✅ | ✅ | super-block + per-sub-block signed scale |
| `Q8_K` | 256 | 8.6 | ✅ | ✅ | super-block + i16 partial sums (sums ignored on dequant) |
| `IQ4_NL` | 32 | 4.5 | ✅ | ✅ | 16-entry non-linear LUT |
| `IQ4_XS` | 256 | 4.25 | ✅ | ✅ | super-block IQ4 |
| `IQ2_XXS` / `IQ2_XS` / `IQ2_S` | 256 | 2.06 / 2.31 / 2.5 | ✅ | ✅ | kmap + sign extract (llama.cpp-style) |
| `IQ3_XXS` / `IQ3_S` | 256 | 3.06 / 3.44 | ✅ | ✅ | 3-bit grid + signs |
| `IQ1_S` / `IQ1_M` | 256 | 1.56 / 1.75 | ✅ | ✅ | 1-bit grid + δ nudge |
| `TQ1_0` / `TQ2_0` | 256 | 1.69 / 2.06 | ✅ | ✅ | BitNet-style ternary {−1,0,+1} |
| `MXFP4` | 32 | 4.25 | ✅ | ✅ | E8M0 scale + E2M1 nibbles (OCP MX) |
| `NVFP4` | 16 | 4.5 | ✅ | ✅ | E4M3 scale + E2M1 nibbles |

Not yet decoded: `Q1_0`. Files that contain it raise a clean
`"dequant for {type} not implemented yet"` error instead of
returning garbage.

The **decoder** path mirrors llama.cpp's `ggml-quants.c` reference
implementation byte-for-byte. Verified element-wise against
`llama-quantize` output on Qwen3-0.6B for every IQ/TQ/MX format
([`tests/iq_tq_real_weights.rs`](tests/iq_tq_real_weights.rs)).
IQ-family grid LUTs are auto-extracted from `ggml-common.h` and
checked into [`src/iq_grids.rs`](src/iq_grids.rs).

The **encoder** path covers legacy Q/K-quants plus IQ/TQ/MX (`iq_quantize.rs`,
`tq_quantize.rs`, `mx_quantize.rs`, `iq2_encode.rs`, `iq3_encode.rs`,
`iq1_encode.rs`). IQ2 uses llama.cpp kmap + sign-extraction (uniform
weights); IQ3/IQ1 use precomputed grids with reduced search. Blocks quantize
in parallel via `rayon`. For peak IQ quality with imatrix weighting, keep
using `llama-quantize`.

## Backend integration

RLX backends consume packed bytes through `Op::DequantMatMul { scheme }`.
GPU kernels (Metal/CUDA/ROCm/WGPU) share integer **scheme ids** 0–23;
decode reference implementations live in this crate.

| id | Scheme |
|----|--------|
| 19 | Q4_0 |
| 20 | Q8_0 |
| 21 | Q4_1 |
| 22 | Q5_0 |
| 23 | Q5_1 |

Per-backend dispatch (GPU vs host, fused GEMV, ANE MIL constexpr, TPU
compile-time bake): [docs/gguf-backend-paths.md](../docs/gguf-backend-paths.md).

When adding a format: implement `dequant_*` here, assign the next scheme id,
then update MSL/CUDA/WGSL kernels and each backend's `gguf_scheme_id`.

## Install

```toml
[dependencies]
rlx-gguf = "0.2"
```

## Quickstart: dequant

```rust,ignore
use rlx_gguf::GgufFile;

let f = GgufFile::from_path("model.gguf")?;
let (data, shape) = f.dequant_f32("token_embd.weight")?;
// `shape` is in GGUF order — innermost dim first. Reverse for
// safetensors / PyTorch convention; the byte layout is identical
// row-major in both.
```

## Quickstart: quantize + write

```rust,ignore
use rlx_gguf::{GgmlType, GgufWriter, MetaValue, quantize};

let weights: Vec<f32> = /* ... */;
let q4k_bytes = quantize(&weights, GgmlType::Q4K)?;

let mut w = GgufWriter::new();
w.set_arch("llama");
w.set_meta("general.name", MetaValue::String("my-model".into()));
w.add_tensor_bytes("token_embd.weight", vec![4096, 32000], GgmlType::Q4K, q4k_bytes)?;
w.write_to_path("out.gguf")?;
```

For end-to-end conversion from safetensors / ONNX / PyTorch, see the companion
[`rlx-gguf-convert`](../rlx-gguf-convert/) crate or **pyrlx** (`convert_to_gguf`,
`load_gguf`, `write_gguf` — no backend required).

For HF-name lookup + MTP-head isolation, use the `GgufLoader` adapter
in the separate model-builders repo (applies the safetensors
convention swap automatically for HF-named keys).

## Build / test

```sh
cargo test -p rlx-gguf
```

Unit tests cover each block format with hand-encoded fixtures, plus
round-trip cosine checks (`quantize → dequant`) for every supported
encoder.

## License

GPL-3.0-only.
