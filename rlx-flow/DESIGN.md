# rlx-flow design

## Goals

1. **Better DX** — model builders never import `HirModule`, `FusionPolicy`, or `Op`.
2. **Better performance** — blocks emit fusion-first HIR (`FusionPolicy::Direct` by default).
3. **Backend leverage** — `CompileProfile` selects fusion target, pass toggles, and precision per device.
4. **Validation** — profile can enable `assert_fusion_clean`; blocks use canonical fused shapes.
5. **Escape hatch** — `FlowStage::Custom` + `rlx_flow::escape::Emit` for tier-2 IR when blocks + config are insufficient. Promote stable patterns into new blocks.

## Flexibility model

```text
Arch recipe (Llama32Flow)     Generic ModelFlow
        │                            │
        ├─ .layer(|ctx| …)           ├─ .repeat_layers(|i| any FlowStage)
        ├─ .before_layers / .after   ├─ .sequence / .when / .custom
        └─ .patch_flow(|flow| …)     └─ .raw_stage / ModelRecipe trait
                    │                            │
                    └──────────┬─────────────────┘
                               ▼
                         BuiltModel → compile
```

Recipes provide defaults; hooks and custom stages cover future arch variants (MoE, cross-attn, vision towers, new norms) without forking the compiler.

## Pipeline

```text
ModelFlow (blocks + profile)
    → HirModule (internal, fusion policy from profile)
    → MirModule (via GraphModule::lower / CompilePipeline)
    → LirModule → backend thunks
```

`BuiltModel` carries `params`, `CompileProfile`, and optional extra outputs (KV taps, etc.).

## Fluent DSL (`src/dsl.rs`)

`ModelFlow` methods chain like a builder — sugar over `FlowStage`:

```rust
ModelFlow::new("model")
    .profile_prefill()
    .input("tokens", shape)
    .token_embed()
    .repeat_layers(n, |i| /* layer */)
    .final_norm(eps)
    .lm_head(vocab, hidden, tied)
    .build(&mut weights)?;
```

Arch-specific recipes (e.g. `Llama32Flow` in downstream graph builders) wrap the same blocks with config-aware defaults.

## CompileProfile

Loaded from `*.rlx.toml` or Rust presets (`CompileProfile::llama32_prefill()`).

Maps to runtime `CompileOptions` via [`ModelExecutionConfig`] + model-builder `flow_bridge::compile_options_for()` (implemented in the model-builders repo).

## Execution variant (shader-component pattern)

[`ModelComponent`](../../rlx-ir/src/component.rs) in `rlx-ir` bundles variant, kernel dispatch,
compilation mode (eager/lazy/AOT), profile key, quant, and layer-composition fingerprint.
[`ModelExecutionConfig`](src/execution.rs) pairs that component with an [`ExecutionPreset`].

Three-step host compile ([`ModelCompilePipeline`](../../rlx-runtime/src/model_pipeline.rs)):

1. `build_template()` — symbolic HIR → LIR template  
2. `specialize_template(binding)` — concrete shapes + buffer plan  
3. `compile_lir()` — backend executable  

Use `get_or_compile_component` / `binding_manifest_for_component` for specialized layouts.
[`BindingManifest::weight_blocks`](../../rlx-ir/src/binding_manifest.rs) groups params by prefix.

Reflection: [`ModelReflection`](../../rlx-runtime/src/reflect.rs) (`load_hir_template`, `layout_for_component`).

Stage interfaces: [`AttentionStage`](src/stage_interfaces.rs), [`FfnStage`](src/stage_interfaces.rs), [`NormStage`](src/stage_interfaces.rs).

Composite stacks: [`LayerComposition`](src/composite.rs) (`Homogeneous` / `Pair` — Slang light-array pattern).

HIR extensions: [`FlowExtensionPlan`](src/extension.rs) + [`rlx_ir::hir_extension`](../../rlx-ir/src/hir_extension.rs).

Attention interfaces: [`AttentionStage`](src/stage_interfaces.rs) on
[`SelfAttnPrefillStage`](src/blocks/self_attn.rs),
[`LlamaDecodeLayerStage`](src/blocks/llama_decode_layer.rs),
[`Qwen3DecodeLayerStage`](src/blocks/qwen3_decode_layer.rs) via [`attention_stage.rs`](src/blocks/attention_stage.rs).

Qwen35 runner: `Qwen35CompileCache::with_aot` in the in-tree Qwen3.5 builder +
[`CompilationMode::Aot`](../../rlx-ir/src/component.rs) persist specialized LIR to disk.

## Multi-stream models (FLUX, …)

Generic primitives in `rlx-flow`:

- **`bind_inputs_to_streams`** — map declared graph inputs (`hidden`, `encoder`, …) into named streams.
- **`dual_stream(name, a, b, f)`** — transform two streams in place; arch plugins emit HIR via `Emit`.
- **`plugin_named` / `PluginStage`** — type-erased arch blocks live in downstream crates, not new `FlowStage` enum variants.

Convention stream ids: `stream::id::IMG`, `stream::id::TXT`, `stream::id::MAIN` (any string works).

Arch recipes (`Flux2Flow`, `Qwen35Flow`) compose these; fused composites stay in `rlx-ir` / existing HIR builders.

## Adding a block

1. Add a stage struct under `src/blocks/`.
2. Implement `BlockStage::emit(&self, ctx: &mut FlowCtx, input: FlowValue) -> Result<FlowValue>`.
3. Add a `FlowStage` variant.
4. Document weight key conventions.

Promote repeated hand-wired subgraphs from in-tree model builders into blocks — do not add new `HirGraphExt` wiring in model code.
