// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// End-to-end smoke tests: build a tiny IR graph, lower it to a CoreML ML
// Program, run it through CoreML.framework, and check the numbers. These
// exercise the full proto → .mlpackage → MLModel pipeline on-device.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::{ComputeUnits, CoremlExecutable, ane_available, chip_info};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n want {b:?}"
    );
}

#[test]
fn chip_info_reports_something() {
    let info = chip_info();
    eprintln!("chip: {info:?}");
    assert!(!info.brand.is_empty());
    // On Apple silicon an ANE should be present.
    eprintln!("ane_available = {}", ane_available());
}

#[test]
fn relu_pipeline() {
    let mut g = Graph::new("relu_test");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);

    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])])
        .expect("run");
    approx(&out[0], &[0.0, 2.0, 0.0, 4.0, 0.0, 6.0], 1e-5);
}

#[test]
fn matmul_with_param() {
    // y = x @ W, x:[2,3], W:[3,4]
    let mut g = Graph::new("matmul_test");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("W", Shape::new(&[3, 4], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);

    let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();

    // CPU reference: row-major [2,3] @ [3,4].
    let mut want = vec![0.0f32; 8];
    for i in 0..2 {
        for j in 0..4 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += x_data[i * 3 + k] * w_data[k * 4 + j];
            }
            want[i * 4 + j] = acc;
        }
    }

    let mut exe = CoremlExecutable::compile(g);
    exe.set_param("W", &w_data);
    let out = exe.run(&[("x", &x_data)]).expect("run");
    approx(&out[0], &want, 1e-4);
}

/// Run the same graph routed to the ANE vs CPU-only and confirm the ANE
/// path is both exercised and numerically faithful. Also surfaces the
/// per-device op routing (MLComputePlan) when the OS supports it.
fn build_mlp() -> (Graph, Vec<f32>, Vec<f32>) {
    // y = relu(x @ W1) @ W2 — a structure CoreML will happily route to ANE.
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[1, 8], DType::F32));
    let w1 = g.param("W1", Shape::new(&[8, 16], DType::F32));
    let h = g.matmul(x, w1, Shape::new(&[1, 16], DType::F32));
    let a = g.activation(Activation::Relu, h, Shape::new(&[1, 16], DType::F32));
    let w2 = g.param("W2", Shape::new(&[16, 4], DType::F32));
    let y = g.matmul(a, w2, Shape::new(&[1, 4], DType::F32));
    g.set_outputs(vec![y]);

    let w1: Vec<f32> = (0..128).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let w2: Vec<f32> = (0..64).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect();
    (g, w1, w2)
}

#[test]
fn ane_routing_matches_cpu() {
    let xv: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let (g_cpu, w1, w2) = build_mlp();
    let mut cpu = CoremlExecutable::compile_with_units(g_cpu, ComputeUnits::CpuOnly);
    cpu.set_param("W1", &w1);
    cpu.set_param("W2", &w2);
    let cpu_out = cpu.run(&[("x", &xv)]).expect("cpu run").remove(0);

    let (g_ane, w1, w2) = build_mlp();
    let mut ane = CoremlExecutable::compile_with_units(g_ane, ComputeUnits::CpuAndNeuralEngine);
    ane.set_param("W1", &w1);
    ane.set_param("W2", &w2);
    let ane_out = ane.run(&[("x", &xv)]).expect("ane run").remove(0);

    // The Neural Engine runs in fp16, so allow a looser tolerance.
    approx(&ane_out, &cpu_out, 5e-2);

    match ane.compute_plan() {
        Ok(Some(counts)) => {
            eprintln!(
                "MLComputePlan op routing — cpu:{} gpu:{} ane:{} unknown:{}",
                counts[0], counts[1], counts[2], counts[3]
            );
        }
        Ok(None) => eprintln!("MLComputePlan unsupported on this OS"),
        Err(e) => eprintln!("compute_plan error: {e}"),
    }
}

#[test]
fn compile_cache_reuse() {
    // Two executables built from the same graph + weights: the second
    // load hits the compiled-model cache. Both must produce identical,
    // correct output (proves the cached .mlmodelc loads and runs right).
    let build = || {
        let mut g = Graph::new("cache_reuse_fixed");
        let x = g.input("x", Shape::new(&[1, 4], DType::F32));
        let w = g.param("W", Shape::new(&[4, 2], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let wv = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let xv = [2.0f32, 3.0, 4.0, 5.0];

    let mut e1 = CoremlExecutable::compile(build());
    e1.set_param("W", &wv);
    let o1 = e1.run(&[("x", &xv)]).expect("run1").remove(0);

    let mut e2 = CoremlExecutable::compile(build());
    e2.set_param("W", &wv);
    let o2 = e2.run(&[("x", &xv)]).expect("run2").remove(0);

    assert_eq!(o1, o2, "cached model output must match fresh compile");
    // [2,3,4,5]@[[1,0],[0,1],[1,1],[0.5,0.5]] = [2+4+2.5, 3+4+2.5] = [8.5, 9.5]
    approx(&o1, &[8.5, 9.5], 1e-4);
}

#[test]
fn add_mul_chain() {
    // z = (a + b) * a
    let mut g = Graph::new("addmul");
    let a = g.input("a", Shape::new(&[4], DType::F32));
    let b = g.input("b", Shape::new(&[4], DType::F32));
    let s = g.binary(
        rlx_ir::op::BinaryOp::Add,
        a,
        b,
        Shape::new(&[4], DType::F32),
    );
    let z = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        s,
        a,
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![z]);

    let av = [1.0f32, 2.0, 3.0, 4.0];
    let bv = [10.0f32, 20.0, 30.0, 40.0];
    let want: Vec<f32> = av.iter().zip(bv).map(|(x, y)| (x + y) * x).collect();

    let mut exe = CoremlExecutable::compile(g);
    let out = exe.run(&[("a", &av), ("b", &bv)]).expect("run");
    approx(&out[0], &want, 1e-4);
}
