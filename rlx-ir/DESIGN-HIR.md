# rlx-ir DESIGN — HIR / MIR / LIR

## Three-level pipeline (fusion-first)

```text
  Model builders                Optimizer (rlx-opt)              Backends
  ─────────────                 ───────────────────              ────────
  HirModule  ──lower──▶  MirModule  ──passes──▶  MirModule  ──plan──▶  LirModule  ──▶  thunks/kernels
     HIR                    MIR (raw)              MIR (opt)                  LIR
   FusionPolicy            FusionReport           buffer plan
```

Fusion is a **first-class citizen** at every stage:

- **HIR** — block ops express fusion *intent*. [`FusionPolicy::Direct`]
  (default for new code) lowers straight to fused MIR ops
  (`FusedMatMulBiasAct`, `FusedSwiGLU`, `FusedResidualRmsNorm`). 
  [`FusionPolicy::Fusable`] lowers to primitive chains the optimizer
  recognizes (legacy / debugging).
- **MIR** — fusion passes close gaps for `HirOp::Mir` escape hatches
  and legacy `Graph` builders. [`FusionReport`] records what fused
  and what missed.
- **LIR** — optimized MIR + [`LirBufferPlan`]. [`CompileResult`] in
  `rlx_opt` bundles LIR + fusion diagnostics for backends.

[`CompilePipeline`](../../rlx-compile/src/compiler.rs) wires the stages.
[`rlx_runtime::stages`](../../rlx-runtime/src/stages.rs) connects
devices to fusion targets. Backends implement [`Backend::compile_hir`]
and [`Backend::compile_lir`].

## HIR block ops

| HIR op | Direct lowering | Pass fallback (`Fusable`) |
|--------|-----------------|---------------------------|
| `Linear` + bias | `FusedMatMulBiasAct` | `MatMul → Add → Act` |
| `SwiGLU` | `concat → MatMul → FusedSwiGLU → MatMul` | `shared_matmul_pair → silu → mul → mm` |
| `ResidualRmsNorm` | `FusedResidualRmsNorm` | `add → rms_norm` |
| `SharedLinearPair` | shared matmul pair (cached) | same → `FuseSharedInputMatMul` |
| `DepthwiseConv1dCausal` | pad → concat → NCHW grouped `Op::Conv` | same (Direct == Fusable) |

## Primary DX: `GraphModule` and `rlx-flow`

[`GraphModule`](src/module.rs) is the unified IR entry above all three stages.
For **model authors**, prefer the tier-0 block assembly line in
[`rlx-flow`](../rlx-flow/README.md):

```rust
use rlx_flow::{CompileProfile, FlowStage, ModelFlow};
use rlx_flow::blocks::{EmbedStage, LlamaDecoderStage, /* … */};

let flow = ModelFlow::new("my_model")
    .with_profile(CompileProfile::from_toml_path("model.rlx.toml")?)
    .input("tokens", shape)
    .stage(FlowStage::Embed(EmbedStage::token("model.embed_tokens.weight")))
    // …
    .output("logits");

let built = flow.build(&mut weights)?;
```

Tier 1: `*.rlx.toml` compile profiles (fusion, precision, passes).
Tier 2: custom HIR blocks / `HirOp::Mir` escape hatches only when needed.

LLaMA-3.2 prefill is the reference migration: `llama32/flow.rs` in the
model graph builders in the separate model-builders repo (see root README).

Legacy / low-level path — `GraphModule` directly:

```rust
use rlx_ir::{Graph, GraphModule, Shape, DType};

// Fusion-first HIR (recommended)
let module = Graph::define("layer", |m| {
    let x = m.input("x", Shape::new(&[2, 128], DType::F32));
    let w = m.param("w", Shape::new(&[128, 128], DType::F32));
    m.linear(x, w, None, None, Shape::new(&[2, 128], DType::F32))
});

// Primitive MIR (legacy)
let mut g = Graph::new("raw");
// … build with g.matmul / g.binary …
let module = g.module();

// Pipeline
let result = CompilePipeline::default().compile_module(module)?;
// or: Session::new(Device::Cpu).compile_module(module)?;
```

- `Graph::define` / `GraphModule::define` — HIR-stage higher-order builder
- `GraphModule::mir` / `Graph::module` — wrap MIR graphs
- `GraphModule::block` — named nested HIR blocks (`hir.named` alias)
- `GraphModule::lower` — HIR → MIR; `Deref` to `Graph` once at MIR/LIR
- `GraphModule::inspect` — stage-aware text dump

## Migration

Existing code keeps using `Graph` — it **is** MIR. New model code should
prefer `Graph::define` / `GraphModule` and call `Session::compile_module`
or `CompilePipeline::compile_module`. `compile_hir` / `compile_graph`
remain for direct stage entry.

## Autodiff (training)

AD runs on **MIR** only (`rlx-autodiff`). Use `grad_with_loss_module` on a
`GraphModule` at HIR or MIR stage, or `prepare_graph_for_ad` then
`grad_with_loss`. Do not feed LIR into AD.

- HIR `FusionPolicy::for_autodiff()` — primitive lowering (less unfuse).
- HIR `Direct` + `rlx_fusion::unfuse_fused_for_autodiff` via `prepare_graph_for_ad`.

Compile forward (full fusion) and backward (AD + cleanup passes only) via
`rlx-compile::CompilePipeline::compile_training` (feature `training` on
`rlx-compile`). Backward `Op::Param` nodes alias the forward weight arena
offsets — weights are not stored twice. Inference uses `compile_module` /
`compile_hir` only. `rlx-opt` re-exports all three crates.

## What doesn't change

- `Op` remains the MIR vocabulary; HIR adds block ops on top, not a
  replacement enum.
- LIR does not embed backend thunks — those stay in `rlx-cpu` /
  `rlx-metal` (device LIR).
