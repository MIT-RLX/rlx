# rlx-dduf

Load HuggingFace **DDUF** (`.dduf`) packages — ZIP archives of nested
`*.safetensors` plus `model_index.json` / component `config.json`.

Tensor names are qualified as `{component}/{tensor}` (e.g.
`transformer/weight`).

```rust
use rlx_dduf::DdufFile;
let f = DdufFile::open("model.dduf")?;
let w = f.tensor_f32("transformer/weight")?;
```

MIT OR Apache-2.0.
