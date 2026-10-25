# rlx-stablehlo

Export an RLX graph as **StableHLO MLIR** — the input format of
[IREE](https://iree.dev) (`iree-compile --iree-input-type=stablehlo`) and of
MLIR toolchains built on it.

```rust
let m = rlx_stablehlo::export(graph, &rlx_stablehlo::StableHloOptions::default())?;
std::fs::write("model.mlir", &m.text)?;
// m.args: inputs then params, in call order; m.results: output shapes
```

```sh
iree-compile --iree-input-type=stablehlo \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  model.mlir -o model.vmfb
iree-run-module --module=model.vmfb --device=local-task --function=main \
  --input=@x.npy --input=@w.npy --output=@y.npy
```

Or through the runtime: `rlx_runtime::export::export_graph(&g,
ExportTarget::StableHlo, &ExportOptions::new(dir))` (feature `stablehlo`), which
also writes `model.signature.json`.

## How it works

The graph goes through the TPU backend's own preparation and rlx → HLO
lowering, unchanged; this crate prints the resulting `HloModuleProto` as
StableHLO (generic op form, tuples flattened into multiple results, constants
as exact hex). One lowering, two sinks — op coverage and semantics are shared
with `rlx-tpu`, and the IREE round trip below doubles as a correctness check of
that lowering on any machine.

- **Layout:** NCHW / OIHW convolutions export as-is — StableHLO carries the
  layout in `dimension_numbers`, so no transposes are inserted.
- **Fused ops** (attention, norms, RoPE, …) export as primitives. IREE does not
  accept `stablehlo.composite`.
- **Weights** are function arguments by default. With
  `StableHloOptions::default().with_parameters("model")` they become IREE
  parameters instead (`util.global` + `#flow.parameter.named<"model"::name>`),
  loaded at run time from a file — `rlx_stablehlo::write_safetensors` writes
  one, and `iree-run-module --parameters=model=weights.safetensors` reads it.
  The function then takes only the graph's inputs.
- **Int8:** an `I8` weight with `Op::Dequantize` crosses the boundary as
  `tensor<…xi8>` (a quarter of the f32 bytes) and is dequantized in the
  module; `Quantize`/`Dequantize` pairs export as integer arithmetic. StableHLO
  `!quant.uniform` types are not emitted: IREE (3.11) cannot lower
  `uniform_quantize`/`uniform_dequantize`, and no other consumer here takes
  them.
- **Not supported:** symbolic shapes (bind them first), graphs that need
  host-executed segments, ops lowered to `custom-call`, run-time GGUF weights.
- **Several shapes in one module:** `export_specializations(graph, &specs,
  &opts)` exports one symbolic graph at several `DimBinding`s as functions
  `@main_<suffix>` (prefill buckets, say `seq16` / `seq128` / `seq512`); with
  parameters, every function shares one weight file.

## Tests

`tests/iree_roundtrip.rs` exports MLP, NCHW conv + pool, embedding /
LayerNorm / shape ops, RmsNorm, a batched matmul, an int8 per-channel weight,
an int8 quantize/dequantize pair, and Llama and BERT layers built through
`rlx-flow` — some also with weights as a safetensors parameter file; compiles them with `iree-compile` (llvm-cpu); runs
them with `iree-run-module`; and compares with rlx CPU. On a mismatch it
re-runs with every op exposed and names the first op that diverges. It skips
when IREE is not on `PATH` (`IREE_COMPILE` / `IREE_RUN_MODULE` override;
`RLX_REQUIRE_DEVICE=1` makes the skip a failure).
## License

MIT OR Apache-2.0.
