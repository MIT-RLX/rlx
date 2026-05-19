# rlx-gguf

GGUF v1 / v2 / v3 parser + dequantization to f32. Standalone — no
`rlx-*` deps; usable from any Rust ML project.

## Supported quantizations

| Format | Block size | Bits / elem | Notes |
|---|---|---|---|
| `F32`, `F16`, `BF16` | n/a | 32 / 16 / 16 | direct read |
| `Q8_0` | 32 | 8.5 | f16 scale + 32 i8 |
| `Q4_0` / `Q4_1` | 32 | 4.5 / 5 | per-block scale (+ min for Q4_1) |
| `Q5_0` / `Q5_1` | 32 | 5.5 / 6 | adds 5th bit via packed `qh` |
| `Q4_K` | 256 | 4.5 | super-block + 8 packed sub-scales/mins |
| `Q5_K` | 256 | 5.5 | super-block + high-bit plane |
| `Q6_K` | 256 | 6.5 | super-block + per-sub-block signed scale |
| `Q8_K` | 256 | 8.6 | super-block + i16 partial sums (sums ignored on dequant) |

Not yet decoded: `Q2_K`, `Q3_K`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`,
`IQ4_NL`, `IQ4_XS`, `Q1_0`. Files that contain these raise a clean
`"dequant for {type} not implemented yet"` error instead of returning
garbage.

The K-quant decoders mirror llama.cpp's `ggml-quants.c` reference
implementation byte-for-byte (verified against the upstream block
layout and a known-good Qwen3-0.6B Q4_K_M GGUF — see
`rlx-models/examples/qwen3_gguf_inference.rs` for the end-to-end
parity check against safetensors F32).

## Install

```toml
[dependencies]
rlx-gguf = "0.1"
```

## Quickstart

```rust
use rlx_gguf::GgufFile;

let f = GgufFile::from_path("model.gguf")?;
let (data, shape) = f.dequant_f32("token_embd.weight")?;
// `shape` is in GGUF order — innermost dim first. Reverse for
// safetensors / PyTorch convention; the byte layout is identical
// row-major in both. (`rlx-models::weight_loader::GgufLoader`
// applies this convention swap automatically for HF-named lookups.)
```

For HF-name lookup + MTP-head isolation, use the higher-level
`GgufLoader` in `rlx-models`:

```rust
use rlx_models::weight_loader::GgufLoader;

let mut loader = GgufLoader::from_file("Qwen3-0.6B-Q4_K_M.gguf")?;
let (data, shape) = loader.take("model.embed_tokens.weight")?;
// shape returned as [vocab, hidden] (safetensors convention)
```

## Build / test

```sh
cargo test -p rlx-gguf
```

Unit tests cover each block format with hand-encoded fixtures.

## License

GPL-3.0-only.