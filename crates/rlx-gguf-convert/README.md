# rlx-gguf-convert

Convert `.safetensors` (and ONNX initializer tensors) into GGUF with
per-tensor quantization, so you can shrink a model's memory + on-disk
footprint at first load and reuse the GGUF dump on subsequent runs.

Pairs with [`rlx-gguf`](../rlx-gguf/) — the resulting files load
through the existing `GgufFile::dequant_f32` path with no special
handling.

## What it does

* Reads source tensors as f32 (lifting from F16/BF16/F64/I32/...).
* Quantizes each tensor to a chosen GGML scheme: F32 / F16 / BF16,
  the legacy formats Q8_0 / Q4_0 / Q4_1 / Q5_0 / Q5_1, the K-quant
  family Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K, the IQ family
  IQ4_NL / IQ4_XS / IQ2_XXS … IQ1_M, ternary TQ1_0 / TQ2_0, and
  microscaling MXFP4 / NVFP4.
* Writes a v3 GGUF file with metadata + tensors, ready to be served
  by any GGUF-aware runtime (including RLX's own).

K-quants in this crate use a per-sub-block min/max quantizer —
simpler than upstream `llama-quantize`'s iterative search, but
producing byte-compatible GGUF that round-trips through every public
RLX dequant kernel. For peak quality keep using `llama-quantize`; for
"shrink-on-first-load" pipelines this trades a notch of quality for
not depending on the C++ tool.

## Real-weight benchmarks

Validated end-to-end on two production checkpoints. The convert
command writes a fresh GGUF and the companion `fidelity_check`
example dequantizes back and compares against the safetensors values
tensor-by-tensor.

| Model | Source size | Scheme | Output | Shrink | Mean cosine | Wall time |
|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 416 MB **F32** | `Q8_0` | 113 MB | 3.75× | 0.999984 | 0.27 s |
| Bio_ClinicalBERT | 416 MB **F32** | `Q6_K` | 86 MB | 4.85× | 0.999815 | 0.22 s |
| Bio_ClinicalBERT | 416 MB **F32** | `Q4_K` | 59 MB | 7.05× | 0.996785 | 0.44 s |
| Bio_ClinicalBERT | 416 MB **F32** | `Q4_0` | 59 MB | 7.05× | 0.996169 | 0.44 s |
| Qwen3-TTS 0.6B | 1.7 GB **BF16** | `Q4_K` | 491 MB | 3.55× | 0.996712 | 3.7 s |

All numbers are *macros over every weight tensor in the model*
(LayerNorm / bias rows are kept at native precision by the default
`skip_quant_for` rule and contribute mean cosine 1.000000 to the
average — they're excluded from the table). M2 mini, release build.

## Quick start

```rust,ignore
use rlx_gguf_convert::{Converter, Scheme};

let report = Converter::from_safetensors("model.safetensors")?
    .default_scheme(Scheme::Q4_K)
    .skip_quant_for(|name, shape| {
        shape.len() < 2 || name.contains("norm") || name.contains("bias")
    })
    .scheme_for_name("model.embed_tokens.weight", Scheme::Q6_K)
    .architecture("llama")
    .write_gguf("model.q4_k.gguf")?;

eprintln!("{:.2}× smaller", report.compression_ratio());
```

### Per-tensor schemes

Three priority levels, applied in order:

1. **Exact-name override** — `.scheme_for_name("foo.weight", Scheme::Q6_K)`
2. **Predicate override** — `.scheme_for(|name, shape| { ... })` returns `Some(Scheme)` to override, `None` to fall through.
3. **Default** — `.default_scheme(Scheme::Q4_K)`.

Tensors whose element count doesn't divide the chosen scheme's block
size silently fall back to F16 (preferred over failing the entire
convert — embeddings often have head-aligned outer dims but odd inner
shapes).

### Skipping quantization

`.skip_quant_for(|name, shape| bool)` preserves the source dtype for
matching tensors. Common pattern:

```rust,ignore
.skip_quant_for(|name, shape| {
    // Keep 1-D tensors and norm/bias parameters at full precision —
    // they're tiny, and quantizing them costs disproportionate accuracy.
    shape.len() < 2 || name.contains("norm") || name.contains("bias")
})
```

## CLI examples

```sh
# One-shot convert
cargo run --release --example convert -p rlx-gguf-convert -- \
    model.safetensors model.q4_k.gguf Q4_K llama

# Fidelity report (per-tensor cosine + max abs error + RMS)
cargo run --release --example fidelity_check -p rlx-gguf-convert -- \
    model.safetensors model.q4_k.gguf
```

## Features

| Feature        | Default | Purpose                                          |
|----------------|---------|--------------------------------------------------|
| `safetensors`  | yes     | `Converter::from_safetensors(path)`              |
| `onnx`         | no      | `Converter::from_onnx(path)` via rlx-onnx-import |

The `Converter` accepts any custom `TensorReader`, so plugging in
another source format is one trait impl.

## When to use

* **First-time model load.** Read the original file once, quantize,
  cache the GGUF — subsequent loads are smaller in both disk + RAM.
* **Pipeline integration.** Enable `rlx`'s `gguf-convert` feature to
  access the same API as `rlx::gguf_convert::Converter`.
* **Python (pyrlx).** Same converter from notebooks / CI:

```python
import pyrlx as rlx

report = rlx.convert_to_gguf(
    "model.safetensors",
    "model.q4_k.gguf",
    "Q4_K",
    architecture="llama",
)
print(report.compression_ratio)
```

Build pyrlx with `maturin develop --features cpu,gguf-convert` (add `gguf-onnx`
for ONNX sources). See `crates/pyrlx/tests/test_gguf_convert.py`.
* **Hardware-targeted variants.** Q5_K_M for memory-constrained edge,
  Q8_0 for fastest decode, Q6_K for highest fidelity at < 7 bits.

## License

GPL-3.0-only — same as the rest of RLX.
