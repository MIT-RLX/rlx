# DDUF weight loading

HuggingFace **DDUF** (`.dduf`) is a ZIP of nested `*.safetensors` plus JSON
(`model_index.json`, per-component `config.json`). RLX reads it via
[`rlx-dduf`](../crates/io/rlx-dduf/).

## Name convention

Tensors are qualified as `{component}/{tensor_name}`, e.g.
`transformer/weight`, `vae/decoder.conv_in.weight`. Root-level members use
`./{tensor_name}`.

## CLI / APIs

```sh
rlx-pkg import-dduf model.dduf -o model.rlxp --no-graph
rlx-bake graph.json -o model.rlxp --weights model.dduf --weights-policy auto
```

DDUF has **no MLX-style affine/mxfp packs**. “Packed” for DDUF means keep
native `f16`/`bf16` bytes (`rlx_dduf::load_native` / bake
`--weights-policy packed|auto`) instead of widening everything to f32.

```rust
use rlx_dduf::{DdufFile, load_native, load_shaped_f32, visit_f32_tensors};
let f = DdufFile::open("model.dduf")?;
let w = f.tensor_f32("transformer/weight")?;
let shaped = load_shaped_f32("model.dduf")?; // keeps on-disk dims
let native = load_native("model.dduf")?; // encoding = f16|bf16|f32|…
// Large packs: stream one safetensors ZIP member at a time
visit_f32_tensors("model.dduf", |t| { /* use t.shape + t.data */ Ok(()) })?;
```

`import-dduf` uses [`visit_f32_tensors`](../crates/io/rlx-dduf/) so peak memory is
roughly one member + the growing weight list (not the full ZIP decoded at once).

Dist URI: `dduf://<path>#<component/tensor>`.

v1 is **read/import only** (no DDUF writer).
## License

MIT OR Apache-2.0.
