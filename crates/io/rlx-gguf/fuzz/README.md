# rlx-gguf fuzz targets

`cargo-fuzz` (libFuzzer) harnesses for the GGUF parser and dequantizers.
GGUF files are untrusted input — downloaded from model hubs — so the
byte-level parser and the per-scheme block decoders are the crate's
attack surface.

This directory is its **own** cargo workspace (note the empty `[workspace]`
table in `Cargo.toml`), so it is invisible to the root rlx workspace and
never touches the top-level `Cargo.toml` / `Cargo.lock`.

## Targets

| target          | entry point                                             | what it exercises |
|-----------------|---------------------------------------------------------|-------------------|
| `gguf_parse`    | `GgufFile::header_from_bytes` + `GgufFile::from_reader` | magic/version gate, KV-metadata loop, tensor table, alignment padding, data slurp |
| `gguf_dequant`  | full parse → `GgufFile::dequant_f32` per tensor         | every block decoder (Q4/Q5/Q8, K-quants, I-quants, TQ/MX/FV5, I8/I16/I32) on fuzzer-controlled bytes |

Both discard the parse/decode `Result` — a finding is a panic, abort, or
memory error, not a rejected file.

## Prerequisites

```sh
cargo install cargo-fuzz     # one-time
rustup toolchain install nightly
```

cargo-fuzz requires a **nightly** toolchain (it relies on
`-Z sanitizer=address` / SanitizerCoverage).

## Run

```sh
# from crates/io/rlx-gguf/
cargo +nightly fuzz run gguf_parse
cargo +nightly fuzz run gguf_dequant

# time-boxed smoke run
cargo +nightly fuzz run gguf_parse -- -max_total_time=30

# just build the instrumented binaries (no run)
cargo +nightly fuzz build
```

Seeding the corpus with a few real `.gguf` files (or ones emitted by
`rlx_gguf::GgufWriter`) lets libFuzzer get past the `GGUF` magic + version
check immediately and reach the interesting metadata/tensor paths much
faster:

```sh
mkdir -p corpus/gguf_parse && cp some-model.gguf corpus/gguf_parse/
```

## Follow-up: safetensors

There is no high-value RLX-owned safetensors *binary* header parser to
target here:

* Binary `.safetensors` parsing is delegated to the upstream `safetensors`
  crate (`SafeTensors::deserialize`), which is fuzzed upstream.
* RLX's own binary validator, `rlx_hub::download::verify_safetensors_structure`,
  is **private** and takes a `&Path` (not bytes) — deliberately not exposed
  just for fuzzing.
* The only public RLX safetensors-family byte parser,
  `rlx_hub::SafetensorsIndex::parse(&[u8])`, is a thin `serde_json::from_slice`
  wrapper over `model.safetensors.index.json` — low marginal value.

If a dedicated safetensors index/header fuzz target is wanted, it belongs
in an `rlx-hub/fuzz/` workspace targeting `SafetensorsIndex::parse`, not here.
