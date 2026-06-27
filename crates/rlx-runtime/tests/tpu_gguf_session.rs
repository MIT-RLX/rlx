//! End-to-end TPU GGUF bake via [`Session::compile_with`] +
//! [`CompileOptions::quant_param_bindings`].

#![cfg(feature = "tpu")]

use std::collections::HashMap;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn run_tpu_gguf_session_case(scheme: QuantScheme, ggml: rlx_gguf::GgmlType, k: usize, n: usize) {
    if !rlx_runtime::is_available(Device::Tpu) {
        eprintln!("TPU unavailable, skipping");
        return;
    }

    let w_f32: Vec<f32> = (0..k * n).map(|i| 0.035 * (i as f32).sin()).collect();
    let packed = rlx_gguf::quantize(&w_f32, ggml).expect("quantize");
    let w_ref = rlx_cpu::dequant_cache::gguf_weight_f32(0, &packed, k, n, scheme);
    let x: Vec<f32> = (0..k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    let mut expected = vec![0f32; n];
    for c in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += x[i] * w_ref[c * k + i];
        }
        expected[c] = acc;
    }

    let mut g = Graph::new("tpu_gguf_session");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("weights", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut tpu = Session::new(Device::Tpu).compile(g);
    tpu.set_param_typed("weights", &packed, DType::U8);
    let actual = tpu.run(&[("x", x.as_slice())]).pop().unwrap();

    for i in 0..expected.len() {
        let rel = (actual[i] - expected[i]).abs() / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-3,
            "TPU {scheme:?} mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

fn run_tpu_gguf_session_with_bindings(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    k: usize,
    n: usize,
) {
    if !rlx_runtime::is_available(Device::Tpu) {
        eprintln!("TPU unavailable, skipping");
        return;
    }

    let w_f32: Vec<f32> = (0..k * n).map(|i| 0.035 * (i as f32).sin()).collect();
    let packed = rlx_gguf::quantize(&w_f32, ggml).expect("quantize");
    let w_ref = rlx_cpu::dequant_cache::gguf_weight_f32(0, &packed, k, n, scheme);
    let x: Vec<f32> = (0..k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    let mut expected = vec![0f32; n];
    for c in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += x[i] * w_ref[c * k + i];
        }
        expected[c] = acc;
    }

    let mut g = Graph::new("tpu_gguf_session");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("weights", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut bindings = HashMap::new();
    bindings.insert("weights".to_string(), packed);
    let opts = CompileOptions::default().quant_param_bindings(bindings);

    let mut tpu = Session::new(Device::Tpu).compile_with(g, &opts);
    let actual = tpu.run(&[("x", x.as_slice())]).pop().unwrap();

    for i in 0..expected.len() {
        let rel = (actual[i] - expected[i]).abs() / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-3,
            "TPU {scheme:?} mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn tpu_gguf_q4_0_session_set_param_typed() {
    run_tpu_gguf_session_case(QuantScheme::GgufQ4_0, rlx_gguf::GgmlType::Q4_0, 32, 4);
}

#[test]
fn tpu_gguf_q8_0_session_quant_bindings() {
    run_tpu_gguf_session_with_bindings(QuantScheme::GgufQ8_0, rlx_gguf::GgmlType::Q8_0, 32, 4);
}

#[test]
fn tpu_gguf_q4_0_session_reupload_set_param_typed() {
    if !rlx_runtime::is_available(Device::Tpu) {
        eprintln!("TPU unavailable, skipping");
        return;
    }
    let k = 32usize;
    let n = 4usize;
    let x: Vec<f32> = (0..k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

    let mut g = Graph::new("tpu_gguf_reupload");
    let x_in = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("weights", Shape::new(&[72], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4_0,
        },
        vec![x_in, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let w_a: Vec<f32> = (0..k * n).map(|i| 0.1 * (i as f32).sin()).collect();
    let packed_a = rlx_gguf::quantize(&w_a, rlx_gguf::GgmlType::Q4_0).unwrap();
    let w_b: Vec<f32> = (0..k * n).map(|i| 0.2 * (i as f32).cos()).collect();
    let packed_b = rlx_gguf::quantize(&w_b, rlx_gguf::GgmlType::Q4_0).unwrap();

    let mut tpu = Session::new(Device::Tpu).compile(g);
    tpu.set_param_typed("weights", &packed_a, DType::U8);
    let out_a = tpu.run(&[("x", x.as_slice())]).pop().unwrap();
    tpu.set_param_typed("weights", &packed_b, DType::U8);
    let out_b = tpu.run(&[("x", x.as_slice())]).pop().unwrap();

    let ref_a = {
        let w_ref =
            rlx_cpu::dequant_cache::gguf_weight_f32(0, &packed_a, k, n, QuantScheme::GgufQ4_0);
        let mut acc = vec![0f32; n];
        for c in 0..n {
            for i in 0..k {
                acc[c] += x[i] * w_ref[c * k + i];
            }
        }
        acc
    };
    let ref_b = {
        let w_ref =
            rlx_cpu::dequant_cache::gguf_weight_f32(0, &packed_b, k, n, QuantScheme::GgufQ4_0);
        let mut acc = vec![0f32; n];
        for c in 0..n {
            for i in 0..k {
                acc[c] += x[i] * w_ref[c * k + i];
            }
        }
        acc
    };

    for i in 0..n {
        assert!((out_a[i] - ref_a[i]).abs() / ref_a[i].abs().max(1.0) < 1e-3);
        assert!((out_b[i] - ref_b[i]).abs() / ref_b[i].abs().max(1.0) < 1e-3);
        assert!(
            (out_a[i] - out_b[i]).abs() > 1e-6,
            "reupload should change output"
        );
    }
}
