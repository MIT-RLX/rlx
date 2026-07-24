# rlx-pkg

RLX ship format (`.rlxp`): **flat mmap by default** with **hybrid hot / warm /
cold** storage, optional ZIP / directory layouts, and an **optional executable
MIR graph** (ONNX-style graph + weights, or weight-only).

Spec: [`docs/rlxp.md`](../../../docs/rlxp.md).

| Container | Use |
|-----------|-----|
| **Flat** (`.rlxp`) | Deploy / serve — one `RLXPFLAT` mmap, tiny TOC |
| **ZIP** (`.zip`) | Inspect with `unzip -l` |
| **Dir** | Edit as a tree during development |

| Tier | Codec | API |
|------|-------|-----|
| Hot | raw | `tensor_mmap` |
| Warm | `zstd_blocks` | `tensor_bytes` / `tensor_warm_block` |
| Cold | zstd | `sidecar` |

Warm compression is for rarely touched host blobs — **not** a substitute for
GGUF-class quants (`Q4_K`, …). Keep active LLM weights hot + packed.

## Usage

```rust,ignore
use rlx_pkg::{
    Package, PackedWeight, StorageTier, WriteOptions, write_package,
};

let mut opts = WriteOptions {
    name: "model".into(),
    // include_graph: true by default — set false for weight-only packs
    ..WriteOptions::default()
};
opts.sidecars.push(("tokenizer".into(), "application/json".into(), tok));

write_package(
    "model.rlxp",
    &graph,
    &[
        PackedWeight::hot("w", shape, "f32", "row_major", bytes),
        PackedWeight {
            name: "expert".into(),
            tier: StorageTier::Warm,
            ..PackedWeight::hot("expert", eshape, "f32", "row_major", ebytes)
        },
    ],
    &opts,
)?;

let pack = Package::open("model.rlxp")?;
pack.advise_hot_willneed();                 // optional page prefetch
let _ = pack.tensor_mmap("w")?;             // zero-copy hot
let _ = pack.tensor_bytes("expert")?;       // inflate warm (parallel for large)
let g = pack.graph()?;                      // materializes hot Constants by default
```

Runtime compile:

```rust,ignore
use rlx_runtime::{Device, Session};
use rlx_runtime::pkg::compile_rlxp;

let session = Session::new(Device::Cpu);
let compiled = compile_rlxp(&session, "model.rlxp")?;
```

## CLI

```bash
cargo run -p rlx-pkg -- inspect model.rlxp
cargo run -p rlx-pkg -- verify model.rlxp
cargo run -p rlx-pkg -- import-gguf model.gguf -o model.rlxp --no-graph
cargo run -p rlx-pkg -- convert in.zip -o out.rlxp --container flat
cargo run -p rlx-pkg -- tier model.rlxp -o out.rlxp --warm expert --hot tok_embd
```

ONNX → RLXP (optional executable graph) is on **`rlx-bake`**:

```bash
cargo run -p rlx-bake --features onnx -- import-onnx model.onnx -o model.rlxp
cargo run -p rlx-bake --features onnx -- import-onnx model.onnx -o w.rlxp --no-graph
```

## Features

| Feature | Enables |
|---------|---------|
| *(default)* | Flat / ZIP / dir, hybrid tiers, GGUF import, verify, CLI |
| `encrypt` | `seal_bytes` / `unseal_bytes` (`RLXSEAL1`) for cold blobs |
| `remote` | HTTP(S) Range `RemoteFlat` (TOC + on-demand tensors) |

## Benches / examples

```bash
just throttle
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-pkg --example bench_hybrid --release
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-pkg --example bench_compare --release
RLX_ALLOW_THROTTLE=1 cargo bench -p rlx-pkg --bench hybrid_load --bench q4k_compare
# MNIST end-to-end (bake + RLXP load/compile/run):
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bake --example mnist_bench_rlxp --features runtime --release
```

## Python (`pyrlx`)

```python
import pyrlx
pyrlx.load_rlxp("model.rlxp")
pyrlx.convert_gguf_to_rlxp("m.gguf", "m.rlxp", include_graph=False)
```

## License

GPL-3.0-only.
