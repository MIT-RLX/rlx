# rlx-nemo

Native loader for NVIDIA **NeMo** `.nemo` model files — tar archive +
`torch.save` checkpoint + YAML config, with no Python or libtorch
dependency.

## What's here

A `.nemo` is an (uncompressed) TAR archive containing:

- `model_config.yaml` — hyperparameters (`NemoConfig`),
- `model_weights.ckpt` — a `torch.save` ZIP of the state dict, and
- optional tokenizer artifacts (SentencePiece `*.model`, `vocab.txt`).

- **`NemoModel::open`** — indexes the archive and the embedded checkpoint
  *without* decompressing or copying the multi-gigabyte weight blob.
- **`NemoModel::tensor`** — pulls individual tensors on demand as contiguous
  `f32`, regardless of on-disk dtype (fp32 / fp16 / bf16 / int).
- **`NemoModel::config` / `names` / `shape_of` / `tokenizers`** — metadata
  access over the archive.
- **`PtModel`** — a standalone loader for plain PyTorch `.pt` / `.pth` /
  `pytorch_model.bin` checkpoints, driven by the same `torch.save` /
  pickle-parsing machinery (the checkpoint ZIP without the `.nemo`
  tar + YAML wrapper).

The pickle / storage / dtype plumbing (`pickle`, `torch`, `storage`, `dtype`
modules) reads the `torch.save` container format directly — a minimal pickle VM
plus the tensor-storage layout — so weights map straight to `f32` buffers.

## Architecture → graph

- **`build_nemo_encoder_graph`** reconstructs the **Conformer / FastConformer
  encoder** (the architecture behind essentially every modern NeMo ASR `.nemo`)
  as primitive rlx ops, binding the checkpoint's weights by name: Macaron
  feed-forwards, relative-position multi-head attention (Transformer-XL
  `pos_bias_u/v` + `rel_shift`), the convolution module (pointwise → GLU →
  depthwise → BatchNorm/LayerNorm → Swish → pointwise), every LayerNorm +
  residual, and the `√d_model` input scaling. Set `EncoderOpts::mel_frames` to
  also prepend the `dw_striding` conv subsampling so the graph runs straight
  from mel features. The graph is shape-specialized (rlx graphs are static);
  geometry is read from the YAML config with fall-backs derived from the weight
  shapes.
- **`build_nemo_probe_graph`** returns the full encoder for a Conformer
  checkpoint, else falls back to a single-Linear probe — so callers such as
  `rlx-pkg`'s `nemo_to_rlxp(include_graph = true)` always get a valid graph.

The mapping is structurally faithful to the NeMo reference; numerical parity
against a reference forward pass is not asserted here (no bundled checkpoint) —
the tests verify structure and end-to-end shape inference.

## Install

```toml
[dependencies]
rlx-nemo = "0.2"
```

## Quickstart

```rust
use rlx_nemo::NemoModel;

let m = NemoModel::open(std::path::Path::new("model.nemo"))?;
let d_model = m.config().get_usize("encoder.d_model");
let w = m.tensor("encoder.layers.0.norm_out.weight")?; // -> NemoTensor (f32)
# anyhow::Ok(())
```

## License

MIT OR Apache-2.0.
