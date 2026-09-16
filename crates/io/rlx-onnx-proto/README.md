# rlx-onnx-proto

Vendored pure-Rust ONNX protobuf types — no `protoc`, no build script.

`src/onnx.rs` is the checked-in `rust-protobuf` output for the ONNX schema.
The crates.io `onnx` crate code-generates the same module at build time by
shelling out to an external `protoc` binary; carrying the generated source
instead keeps `cargo build` hermetic and toolchain-free, which matters for the
[`rlx-onnx-import`](../rlx-onnx-import/) path on machines that have no
protobuf compiler installed.

The only runtime dependency is the pure-Rust `protobuf` crate.

## Compatibility

The module layout mirrors the old `onnx` crate — `onnx::onnx::ModelProto` and
friends — so consumers only change the crate name.

## Install

```toml
[dependencies]
rlx-onnx-proto = "0.2"
```

## Quickstart

```rust
use rlx_onnx_proto::onnx::ModelProto;

let model = ModelProto::new();
assert_eq!(model.get_ir_version(), 0);
```

## Regenerating

Only needed when the ONNX schema itself changes. Run `rust-protobuf` 1.x over
`onnx.proto`, then rewrite the bare `::std::any::Any` trait-object references
to `dyn ::std::any::Any` so the output builds on a modern edition.

## License

MIT OR Apache-2.0.
