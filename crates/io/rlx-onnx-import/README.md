# rlx-onnx-import

ONNX → [`rlx-ir`](../../core/rlx-ir) HIR lowering used by [`rlx-onnx`](../rlx-onnx) native execution.

## Features

- Load RLX bundles (`manifest.json` + `graph.json` + `weights.safetensors`) or raw `.onnx`
- Graph rewrites: `DynamicQuantizeLinear` input aliasing, `ConvInteger` / `MatMulInteger` → f32 ops
- Broad ONNX op lowering (see `src/bin/report.rs` `LOWERED_OPS`)
- Model output names are kept (`HirModule::output_names` → the graph's named
  outputs), so `CompiledGraph::output_index("logits")` finds them after fusion
- Negative `Pad` crops, as ONNX defines it

## CLI

```bash
cargo run -p rlx-onnx-import --bin rlx-onnx-import-report -- /path/to/bundle
cargo run -p rlx-onnx-import --features runtime --bin bundle-compile --release
```

`examples/transpose_census.rs` counts the transposes `SimplifyTransposes`
(`CompileOptions::simplify_transposes`, off by default) removes from an
imported model; `--run` also times each model before and after the pass and
reports the largest output difference.

Set `RLX_ONNX_BUNDLE` for `bundle-compile`. Use `--quantize-bundle` with the report tool
for quant fusion rewrites (`ImportOptions::quant_bundle()`).

## Tests

```bash
cargo test -p rlx-onnx-import
```

Optional raw ONNX tests: set `RLX_ONNX_TEST_MODEL` to a `.onnx` path.

## License

MIT OR Apache-2.0.
