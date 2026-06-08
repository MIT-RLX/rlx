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

Not yet decoded: `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`,
`Q1_0`. Files that contain these raise a clean `"dequant for {type}
not implemented yet"` error instead of returning garbage.

The **decoder** path mirrors llama.cpp's `ggml-quants.c` reference
implementation byte-for-byte (verified against the upstream block
layout and a known-good Qwen3-0.6B Q4_K_M GGUF).

The **encoder** path uses a per-sub-block min/max quantizer — simpler
than upstream's iterative `make_qx_quants` search but byte-compatible
with the decode side. Round-trip cosine ≥ 0.99 on transformer
weights; for peak quality keep using `llama-quantize`, for
shrink-on-first-load pipelines this avoids the C++ dependency.

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

For end-to-end conversion from safetensors / ONNX, see the companion
[`rlx-gguf-convert`](../rlx-gguf-convert/) crate.

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
