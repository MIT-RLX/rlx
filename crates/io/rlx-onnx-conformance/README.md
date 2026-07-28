# rlx-onnx-conformance

ONNX op-level conformance harness — compares ONNX Runtime reference outputs
against RLX import + execute paths from [`rlx-onnx-import`](../rlx-onnx-import).

Used to track lowering coverage and numeric alignment as the ONNX importer
grows. Not a user-facing inference crate; workspace / CI validation only.

## What's here

- **`harness`** — `compare_tensors`, `ConformanceResult`, shared atol
  (`DEFAULT_ATOL = 1e-4`).
- **`synthetic`** — per-op synthetic ONNX graphs with known outputs.
- **`onnx_op_registry`** — op coverage checklist for bundled ONNX exports.
- **`backend_runner`** — run imported HIR through RLX CPU (and helpers).
- **`rlx-onnx-coverage`** binary — coverage dashboard over lowered ops.

## Run tests

```bash
cargo test -p rlx-onnx-conformance
```

## Coverage dashboard

```bash
cargo run -p rlx-onnx-conformance --bin rlx-onnx-coverage
```

## Dependencies

Pulls `ort` on desktop targets for the reference side. iOS / Android builds
skip ORT in the harness (`OrtSession::from_bytes` bails).

## License

MIT OR Apache-2.0.
