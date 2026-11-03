# rlx-onnx-proto

Vendored pure-Rust ONNX protobuf types — no `protoc`, no build script.

`src/onnx.rs` is the checked-in `rust-protobuf` 3.7.2 output for the ONNX schema
in `onnx.proto`.
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
assert_eq!(model.ir_version(), 0);
```

## Regenerating

Only needed when the ONNX schema or the `protobuf` version changes:

```bash
cargo run --manifest-path crates/io/rlx-onnx-proto/regen/Cargo.toml
```

`regen/` runs `protobuf-codegen` with its pure-Rust parser (no `protoc`) over
`onnx.proto` and rewrites `src/onnx.rs`. Keep the `protobuf` pin in
`Cargo.toml` equal to the codegen version in `regen/Cargo.toml` — the generated
code refuses to build against any other runtime version.

`onnx.proto` is the schema the original 1.x bindings were generated from,
recovered from their embedded file descriptor, so field numbers and types are
unchanged. It predates `external_data` / `data_location` (TensorProto fields 13
and 14); `rlx-onnx-import` reads those from each tensor's unknown fields.

## License


MIT OR Apache-2.0.
