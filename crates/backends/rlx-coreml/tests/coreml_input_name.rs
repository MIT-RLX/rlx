// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(any(target_os = "macos", target_os = "ios"))]
use rlx_coreml::{ComputeUnits, CoremlExecutable};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};

fn try_name(name: &str, dims: &[usize]) {
    let mut g = Graph::new("name_probe");
    let shape = Shape::new(dims, DType::F32);
    let x = g.input(name, shape.clone());
    let y = g.activation(Activation::Relu, x, shape);
    g.set_outputs(vec![y]);
    let mut ex = CoremlExecutable::compile_with_units(g, ComputeUnits::CpuAndGpu);
    let n: usize = dims.iter().product();
    let data = vec![1.0f32; n];
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ex.run(&[(name, &data)]))) {
        Ok(Ok(_)) => eprintln!("OK name={name} dims={dims:?}"),
        Ok(Err(e)) => eprintln!("ERR name={name}: {e}"),
        Err(_) => eprintln!("PANIC name={name} dims={dims:?}"),
    }
}

#[test]
fn qk_rotated_empty_input_name() {
    try_name("qk_rotated_empty", &[2, 200, 64]);
    try_name("qk_rotated_zeros", &[2, 200, 64]);
    try_name("qk", &[2, 200, 64]);
    try_name("empty", &[2, 200, 64]);
    try_name("rope_cos", &[1, 200, 64]);
    // Scalar-ish TTS controls (Kitten `speed`) must load on CoreML.
    try_name("speed", &[1]);
}
