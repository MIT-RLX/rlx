# rlx-torch-import

Convert a **PyTorch `nn.Module` directly into an RLX model** — no ONNX in the
loop. Point it at a module + example inputs and get back a runnable bundle
and/or a standalone generated RLX crate, verified numerically against PyTorch.

```
PyTorch nn.Module + example inputs
   │  torch.export → run_decompositions()   (pyrlx.torch_import, Python)
   ▼
torch-ir.json  +  weights.safetensors        ← Core ATen graph, concrete shapes
   │  aten → rlx op registry                 (this crate, Rust)
   ▼
Lowered  (shared Call vocabulary)
   ├─► bundle/   model.hir.json + weights.safetensors + meta.json   (run now)
   ├─► rlx-<name>/   Cargo.toml + src/{lib,graph,weights}.rs        (ship a crate)
   └─► parity check vs PyTorch (cosine + max abs err) on CPU
```

## Why direct `torch.export` (not ONNX)

`torch.export` + `run_decompositions()` lowers to **Core ATen IR** — a small,
stable op set. Every FX node carries `node.meta['val']` (a FakeTensor with
concrete **shape + dtype**), which supplies the explicit output shapes RLX's
graph builders need. High-level ops are *preserved* (via a filtered decomposition
table) so they land on **fused** RLX ops rather than a decomposed soup:

| PyTorch / ATen                         | RLX op                    |
|----------------------------------------|---------------------------|
| `scaled_dot_product_attention` (+ mask)| `Op::Attention` (fused)   |
| `_scaled_dot_product_{flash,efficient,math}_attention` | `Op::Attention` (variant args) |
| `layer_norm` / `native_layer_norm`     | `Op::LayerNorm`           |
| `rms_norm`                             | `Op::RmsNorm`             |
| `group_norm` / `native_group_norm`     | `Op::GroupNorm` (NCHW)    |
| `batch_norm` (inference)               | decomposed (layout-agnostic) |
| `gelu` / `silu` / `relu` / `leaky_relu` / `hardswish` / `hardsigmoid` / `hardtanh` | `Op::Activation` / clamp+mul |
| `convolution` / transposed             | `Op::Conv` / `Op::ConvTranspose2d` |
| `upsample_nearest2d` (2ⁿ×, NCHW)       | `Op::ResizeNearest2x` (chained) |
| `upsample_{bilinear,bicubic}2d` (+ `_aa`) (NCHW, any size) | `resize_{bilinear,bicubic}2d[_aa]` (separable interp matmuls) |
| `pixel_shuffle` / `pixel_unshuffle`    | reshape + permute         |
| `grid_sampler[_2d]` (all modes/paddings) | `grid_sample2d` (gather + arithmetic) |
| `masked_fill`                          | `x + mask·(v−x)` (arith)  |
| `max_pool2d` / `avg_pool2d` / adaptive | `Op::Pool` / `Op::Reduce` |
| `embedding` / `index` (1-axis)         | `Op::Gather`              |
| `addmm` / `mm` / `bmm` / `baddbmm` / `linear` | `Op::MatMul` (+ scale/`Add`) |
| scalar math (`add`/`mul`/`pow`/`clamp`)| `Op::Binary` + const-fill |
| `full` / `zeros` / `ones` / `empty`    | `Op::Constant` (fill)     |
| `split` / `chunk` / `split_with_sizes` / `select` | narrow tuple |
| `constant_pad_nd` (constant mode)      | `concat` + const fill     |
| `view` / `permute` / `slice` / `cat`   | reshape/transpose/narrow/concat |

The aten→rlx mapping lives in exactly one place ([`src/lower.rs`](src/lower.rs));
two walkers consume its output — [`src/hir_build.rs`](src/hir_build.rs) builds a
live `HirModule` to run/verify, and [`src/emit.rs`](src/emit.rs) prints the
generated crate — so run-parity and generated-code parity can never drift.

## Usage

### One call (Python)

```python
import pyrlx, torch

model = MyModel().eval()
example = (torch.randn(1, 3, 224, 224),)
pyrlx.from_torch(model, example, out_dir="out/", verify=True)
# → out/torch-ir.json, out/weights.safetensors,
#   out/bundle/, out/rlx-mymodel/,  parity report
```

### One command (CLI)

```bash
# model.py exposes module-level `model` + `example_inputs`
# (or a get_model()/build() returning (model, example_inputs))
rlx-torch-import model.py -o out/ --emit bundle,crate
```

The Python CLI runs the front-end, then invokes this crate's Rust worker
(located via `$RLX_TORCH_IMPORT_BIN` or a `cargo run` fallback).

### Rust worker directly

```bash
# after the front-end has written torch-ir.json + weights.safetensors into <dir>
cargo run -p rlx-torch-import -- build <dir> --emit bundle,crate --verify
```

## Decomposition levels & provenance

`--decomposition` (CLI) / `decomposition=` (Python) trades staying close to the
source module against landing on primitives the registry covers:

- **`aten`** — no `run_decompositions`; the raw exported graph. `linear`,
  `layer_norm`, `sdpa` stay as single ops → *closest to the original, best for
  reconstruction* (fewest nodes).
- **`high`** *(default)* — Core ATen but high-level ops preserved → fused RLX ops.
- **`core`** — full Core ATen (most primitive; may exceed current op coverage).

### Auto-decompose fallback (`pyrlx.from_torch`, on by default)

If the importer reports ops the registry doesn't cover, `from_torch` re-exports
with **those ops decomposed** (via torch's full decomposition registry) and
retries — up to `max_decompose_rounds`. So an unsupported op that torch knows how
to break down is handled automatically, without hand-adding a lowering; the ops
it decomposed land in `summary["auto_decomposed_ops"]`. Decompositions that emit
`prims.*` (which break torch's functionalization) are detected and **skipped**
one at a time, so a single un-decomposable op never crashes the import — it's
reported as a genuine coverage gap instead. Disable with `auto_decompose=False`.

### Dynamic shapes

Export with `dynamic_shapes` and the symbolic dims flow through:

```python
from torch.export import Dim
pyrlx.from_torch(model, (torch.randn(3, 4),), out_dir="out/",
                 dynamic_shapes={"x": {0: Dim("batch", min=1, max=64)}})
```

The front-end marks each dynamic input axis with a stable symbol id (`torch-ir.json`
`inputs[*].dynamic = [sym | -1, …]`); the importer builds `Dim::Dynamic(sym)` inputs,
the compile pass re-infers the symbolic graph, and a `DimBinding` specializes it per
run. So a model is imported **once** and run at any batch/seq — no re-export. The
Rust side runs a dynamic HIR with `run::run_dynamic(hir, params, inputs, device,
&DimBinding::from_pairs(&[(sym, size)]))`; `--verify` binds the reference input's
extents automatically. (Ops that are inherently static — e.g. a constant sized by a
dynamic dim — still specialize per binding rather than staying symbolic.)

Regardless of level, `torch-ir.json` records every op with args/shapes/dtypes,
and the **generated crate is annotated** for traceability — a header with the
original aten op histogram, and a per-op comment giving the source op + input
names/shapes + output shape/dtype:

```rust
// aten.scaled_dot_product_attention.default  (transpose[1,4,6,8], transpose_1[1,4,6,8], transpose_2[1,4,6,8]) -> [1,4,6,8] f32
let t_sdpa = b.attention_kind(t_transpose, t_transpose_1, t_transpose_2, 4, 8, MaskKind::None, /*…*/);
```

## Extending the registry (macro-driven)

"Direct" ops — those that lower to a single `add_node(Op::X, inputs, shape)` —
are declared **once** in the [`nodeop`](src/nodeop.rs) table via
`define_node_ops!`, which generates both the live-HIR build and the
generated-crate source from one entry:

```rust
Compare [a, b] { op: CmpOp }
    build = rlx_ir::Op::Compare(*op),
    src   = format!("rlx_ir::Op::Compare(rlx_ir::op::CmpOp::{})", cmp_variant(*op));
```

`hir_build`/`emit` then need only a single generic `Call::Node` arm, so adding a
direct op is a one-line table entry instead of edits in three files. (Ops with
bespoke shape logic still get a hand-written `lower` handler — that part is
inherently per-op.)

## Generated-crate authoring layer (`--emit-style`)

The generated crate can target three rlx authoring layers (`--emit-style` /
`emit_style=`):

- **`graph`** *(default)* — the raw `HirMut` / `HirGraphExt` builder. Covers
  every op; verified across all models.
- **`tensor`** — the operator-overloaded `rlx_tensor::Tensor` DSL, the most
  readable/editable (PyTorch-like) output:
  ```rust
  let t_h  = t_x.layer_norm(&w, &b, 1e-5);
  let t_a  = t_q.attention(&t_k, &t_v, 4, 8, MaskKind::None);
  let t_o  = (&t_x + &t_proj).silu();
  ```
  Compile-verified on MLP + transformer. It reports clearly (naming the op) when
  a model uses something the Tensor API doesn't expose (Pool / TopK /
  ConvTranspose / mask-tensor attention) — use `graph` for those.
- **`flow`** — an `rlx_flow::ModelFlow` with a single custom stage
  (`FlowStage::Custom`) that builds the graph through the HIR builder and
  integrates with the `WeightSource` (`MapWeights`) / `BuiltModel` runner
  ecosystem. Covers all ops (its per-op emission is the *same* shared builder as
  `graph`, via `emit::emit_hir_ops`). Compile-verified **and run-verified** at
  cosine 1.0.

All three styles are compile- and run-verified (cosine 1.0) on the transformer;
`graph`/`flow` cover every op, `tensor` covers the ops the Tensor DSL exposes.

## ONNX front-end (`--features onnx`)

Rather than re-port the ONNX operator catalog, the tool reuses
[`rlx-onnx-import`](../rlx-onnx-import)'s opset-versioned ONNX→HIR importer
(~88 ops) as a **second front-end** that lands on the same runnable bundle:

```bash
rlx-torch-import onnx model.onnx -o out/ --verify   # needs --features onnx
```

The wiring (import → HIR → bundle → verify) is validated on `rlx-onnx-import`'s
own fixtures. Note that `rlx-onnx-import`'s *lowering* currently has gaps on some
TorchScript-exported patterns (scalar broadcast, `Gemm`/transB, conv) — those
are pre-existing issues in that crate, not the front-end wiring.

## Weight names

Weights are keyed by their `state_dict` FQN (e.g. `model.layers.0.self_attn.q_proj.weight`),
which for HuggingFace models is the HF-canonical name — so RLX's existing
loaders consume them with no remapping.

## Coverage & limitations (v1)

Verified to **cosine ≈ 1.0** (max abs err < 3e-7) end-to-end on:

| Model class                                                   | Status |
|---------------------------------------------------------------|--------|
| MLP (linear + layernorm + gelu)                               | ✅     |
| Transformer block (`sdpa` + SwiGLU + layernorm + MHA reshape) | ✅     |
| **Encoder-decoder** with self- **and cross-attention**        | ✅     |
| **Classification head** (encoder + mean-pool + linear)        | ✅     |
| **CNN** (conv + batchnorm + relu + maxpool + adaptive-pool + head) | ✅ |
| **LSTM / GRU recurrence** (torch unrolls it; we map the primitives) | ✅ |
| **Masked attention** (additive float mask)                    | ✅     |
| **Transposed conv** (upsampling decoder)                      | ✅     |
| **Scalar math** (`*`/`+`/`pow`/`clamp`/`div`)                 | ✅     |
| **MoE** with top-k routing (`topk` + expert gather)           | ✅     |
| **Decoder LM with token-id inputs** (embedding + causal attn) | ✅     |
| **Real HF Llama** (rotary + GQA + causal mask + SwiGLU)       | ✅     |
| **Real HF BERT** / **DINO ViT** encoders                      | ✅     |
| **MNIST CNN** (conv + pool + head)                            | ✅     |
| **FLUX** diffusion transformer (MMDiT: joint attn + adaLN + axial RoPE) | ✅ |
| Training step (`autograd.grad` in `forward`)                  | ❌ (`torch.export` traces inference graphs only) |

Worked examples for every model class + every option live in
[`../../bindings/pyrlx/examples/torch_to_rlx.py`](../../bindings/pyrlx/examples/torch_to_rlx.py)
(`python torch_to_rlx.py all` / `options`).

**Device:** `--device {cpu,cuda,metal,…}` runs the parity check on that backend
(`--features cuda` for GPU). **All six showcase models (MLP, encoder-decoder,
CNN, MNIST, Llama, DINO) verify at cosine 1.000000 on an RTX 3080 Ti** (CUDA
13.1). Integer token inputs / params are fed as **f32 + a cast** so they cross
the f32-arena GPU host surface. Bringing Llama up on CUDA surfaced and fixed four
rlx-cuda bugs: the `unary.cu` activation kernel was missing sin/cos/tan/atan/round
(op ids 13–16 → identity, breaking rotary); `Op::Constant` upload reinterpreted
integer bytes as f32 instead of widening; and `Bool` compare outputs were sized
1 byte in the f32 arena while the compare kernel writes f32 (mask corruption).

All ✅ verified at **cosine = 1.000000** (max abs err < 3e-7) vs PyTorch on CPU.

Notes / deferred:
- **Static shapes only** — export with concrete example inputs
  (`dynamic_shapes=None`).
- **Integer inputs** (token ids) run through `run_typed`; parity is verified for
  decoder LMs. rlx `Op::TopK` is argtopk (indices only), so top-k *values* are
  reconstructed by gathering `x` at flat `row*E + idx` offsets.
- **Training** is out of scope for a `torch.export` front-end (it captures the
  eval graph). For on-device training, import the *forward* and use RLX's own
  autodiff / `rlx-optim`.
- Still deferred: multi-axis advanced indexing, non-global adaptive pooling,
  and reconstructing RoPE from its decomposed rotate-half pattern (a `Rope`
  Call exists but no matcher emits it).
- Unsupported ops are reported **all at once** with the exact aten names, so
  extending [`src/lower.rs`](src/lower.rs) is a matter of adding a handler + a
  `SUPPORTED` entry (and, for the generated crate, an `emit` arm).
