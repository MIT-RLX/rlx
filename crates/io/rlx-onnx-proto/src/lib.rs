// Pure-Rust ONNX protobuf types — checked-in rust-protobuf output, no protoc.
// Mirrors the layout of the old `onnx` crate (`onnx::onnx::ModelProto`, …) so
// consumers only change the crate name.
extern crate protobuf;

pub mod onnx;
