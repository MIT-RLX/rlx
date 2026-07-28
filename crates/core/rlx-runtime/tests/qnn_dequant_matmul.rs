// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session-level `DequantMatMul` on `Device::Hexagon` (host-dequant → MatMul).

#![cfg(feature = "qnn")]

use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn qnn_available() -> bool {
    std::env::var_os("RLX_QNN_BACKEND_LIB").is_some() || std::env::var_os("QNN_SDK_ROOT").is_some()
}

#[test]
fn dequant_matmul_q8_0_matches_cpu() {
    if !qnn_available() {
        eprintln!("skip: no QNN backend");
        return;
    }

    let (m, k, n) = (2usize, 64, 8);
    let w_nk: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.45)
        .collect();
    let packed = rlx_gguf::quantize(&w_nk, GgmlType::Q8_0).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    let build = || {
        let mut g = Graph::new("dqmm");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wp = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
        let y = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ8_0,
            },
            vec![xi, wp],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    cpu.set_param_typed("w_packed", &packed, DType::U8);
    let cpu_out = cpu.run(&[("x", x.as_slice())]);

    let mut hex = Session::new(Device::Hexagon).compile(build());
    hex.set_param_typed("w_packed", &packed, DType::U8);
    let hex_out = hex.run(&[("x", x.as_slice())]);

    let md = cpu_out[0]
        .iter()
        .zip(&hex_out[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(md < 1e-3, "Hexagon DequantMatMul Q8_0 vs CPU max_diff {md}");
    eprintln!("Session(Device::Hexagon) DequantMatMul Q8_0 OK (max_diff={md:.2e})");
}
