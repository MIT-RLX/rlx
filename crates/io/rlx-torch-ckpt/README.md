# rlx-torch-ckpt

Native reader for PyTorch **`torch.save`** checkpoints — `.pt`, `.pth`,
`pytorch_model.bin` — with no libtorch, no Python, and no protobuf.

A `torch.save` file (PyTorch ≥ 1.6, the default since 2020) is a ZIP holding a
pickled object graph in `data.pkl` plus one raw storage blob per tensor under
`data/<key>`. This crate indexes that container by seeking, unpickles the state
dict into a flat `name → TensorMeta` table, and materializes tensors on demand
as contiguous row-major `f32` — whatever their on-disk dtype (fp32 / fp16 /
bf16 / int). A multi-gigabyte checkpoint is never slurped into RAM.

```rust,no_run
use rlx_torch_ckpt::PtModel;

let m = PtModel::open(std::path::Path::new("4xNomos2_realplksr_dysample.pth"))?;
for name in m.names() {
    let t = m.tensor(&name)?;          // -> PtTensor { shape, dtype, data: Vec<f32> }
    println!("{name}: {:?} ({:?})", t.shape, t.dtype);
}
# anyhow::Ok(())
```

Shapes and dtypes are available without reading data (`shape_of`, `dtype_of`),
so a loader can validate an architecture against a checkpoint before paying for
the weights.

## Scope

Both `torch.save` containers are read, chosen by the file header:

| Container | Layout | Read strategy |
|---|---|---|
| ZIP (PyTorch ≥ 1.6) | `data.pkl` + one blob per storage | indexed; blobs read on demand |
| legacy (pre-1.6) | 5 pickles + a flat storage stream | read whole — the stream has no index |

The legacy path covers most of the ESRGAN-era model zoo. Its storage section
records an element *count* and no dtype, so the object pickle is parsed first
to learn each storage's element width; a storage that no tensor references
cannot be sized, and is a hard error rather than a guess that would
desynchronize every storage after it.

The pickle VM implements only what `torch.save` emits — this is a weights
reader, not a general unpickler, and it will not execute arbitrary pickle
opcodes.

## Relationship to the other torch crates

| Crate | Reads | Produces |
|---|---|---|
| `rlx-torch-ckpt` | weights — a saved `state_dict` | `f32` tensors |
| [`rlx-torch-import`](../rlx-torch-import) | programs — a `torch.export`ed graph | RLX HIR |
| [`rlx-nemo`](../rlx-nemo) | `.nemo` — this checkpoint + tar + YAML | `f32` tensors + config |

`archive` is public because container formats that wrap a checkpoint (`.nemo`'s
tar, for one) need the same seek/list/read primitives.

## License

MIT OR Apache-2.0
