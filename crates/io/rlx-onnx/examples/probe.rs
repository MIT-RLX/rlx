// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Run one ONNX model on one raw f32 buffer and write the raw f32 result.
//!
//! ```text
//! cargo run -p rlx-onnx --features <backend> --example probe -- \
//!     <model.onnx> <input.f32> <output.f32> [input_name] [device]
//! ```
//!
//! Deliberately minimal, and deliberately not `rlx-onnx-run`: that fills inputs
//! with zeros, which cannot answer the only question worth asking of a new
//! backend — does it compute the same numbers as a reference runtime. Feed this
//! and ONNX Runtime the same bytes and compare the outputs.

use std::collections::HashMap;

use rlx_onnx::{OnnxCompileLevel, OnnxModel, OnnxTensor};
use rlx_runtime::Device;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: probe <model.onnx> <input.f32> <output.f32> [input_name] [device]");
        std::process::exit(2);
    }
    let name = args.get(3).cloned().unwrap_or_else(|| "image".into());
    let device = match args.get(4).map(String::as_str) {
        None | Some("cpu") => Device::Cpu,
        Some("cuda") => Device::Cuda,
        Some("rocm") => Device::Rocm,
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("vulkan") => Device::Vulkan,
        Some(other) => panic!("unknown device {other:?}"),
    };

    let bytes = std::fs::read(&args[1]).expect("reading the input");
    let input: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut model = OnnxModel::load_native(&args[0], device, OnnxCompileLevel::Level3, 1)
        .expect("loading the model");
    let mut feed = HashMap::new();
    feed.insert(name, OnnxTensor::F32(input));
    let out = model.run(&feed).expect("inference");
    let OnnxTensor::F32(values) = out.into_iter().next().expect("an output") else {
        panic!("expected a float output");
    };
    let raw: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&args[2], raw).expect("writing the output");
    println!("{} values → {}", values.len(), args[2]);
}
