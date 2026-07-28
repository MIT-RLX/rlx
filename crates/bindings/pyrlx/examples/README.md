# pyrlx examples — export PyTorch models to RLX

`pyrlx.from_torch(model, example_inputs, out_dir, ...)` converts a live
`torch.nn.Module` into an RLX model: it runs `torch.export`, maps each ATen op
directly onto RLX ops, and emits a runnable bundle and/or a standalone RLX crate
— then verifies numeric parity against PyTorch.

## One file per model

Each script defines a model and exports it in **every form** via
`export_all_forms` (see [`_common.py`](_common.py)):

| Script | Model | What it exercises |
|--------|-------|-------------------|
| [`mlp.py`](mlp.py) | LayerNorm + Linear + GELU | the simplest end-to-end path |
| [`encoder_decoder.py`](encoder_decoder.py) | enc/dec blocks | self- **and cross-attention** |
| [`cnn.py`](cnn.py) | Conv + BatchNorm + pool | vision stack, layout-correct BN |
| [`mnist.py`](mnist.py) | LeNet-style classifier | a conventional deployable model |
| [`llama.py`](llama.py) | real HF `LlamaForCausalLM` | rotary + GQA + causal mask + SwiGLU |
| [`dino.py`](dino.py) | DINO-style ViT | patch-embed + transformer encoder |

```bash
cd crates/bindings/pyrlx/examples
python mlp.py          # or cnn.py / llama.py / dino.py / …
```

Each run prints the four output forms and leaves them under `out_<name>/`:

```
 MLP
========================================================================
  1. bundle  parity PASS ✓: cosine=1.000000  max|err|=1.19e-07  (128 elems)
             HIR graph: 19 nodes  [Mir×12, Param×6, Input×1]
             out_mlp/bundle/model.hir.json
  2. crate/graph   out_mlp/rlx-mlp                 src/{graph.rs, lib.rs, weights.rs}
  3. crate/tensor  out_mlp/style_tensor/rlx-mlp    src/{graph.rs, lib.rs, weights.rs}
  4. crate/flow    out_mlp/style_flow/rlx-mlp      src/{graph.rs, lib.rs, weights.rs}
```

- **bundle** — a runnable RLX file: the serialized **HIR graph**
  (`bundle/model.hir.json`) + weights, checked against PyTorch on CPU.
- **crate/{graph,tensor,flow}** — a standalone RLX crate in three authoring
  styles. `graph` (raw HIR builder) and `flow` (`ModelFlow`) cover every op;
  `tensor` (PyTorch-like `Tensor` DSL) is the most readable but can't express a
  few ops (pooling, computed-mask attention) — those models report it as `n/a`.

## Options reference

[`torch_to_rlx.py`](torch_to_rlx.py) is an all-in-one script that documents
**every `from_torch` option** with a worked example each
(`python torch_to_rlx.py options`) and runs the whole model zoo
(`python torch_to_rlx.py all`).

## Running the generated crate

```bash
cd out_mlp/rlx-mlp && cargo run --example verify   # runs the crate vs the golden output
```

All six models verify at **cosine 1.000000** vs PyTorch on both **CPU and CUDA**
(NVIDIA GPU). Integer token ids / params are fed as f32 + a cast, so the same
artifacts run on the GPU.
## License

MIT OR Apache-2.0.
