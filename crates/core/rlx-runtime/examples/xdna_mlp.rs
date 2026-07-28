// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// End-to-end demo: a 2-layer quantized MLP whose matmuls run on the AMD XDNA
// NPU (`Device::Xdna`), activations on the host, verified vs a CPU f32 reference.
//
//   y = relu(x @ W1) @ W2
//
//   RLX_XDNA_SHIM=… RLX_XDNA_XCLBIN=… RLX_XDNA_INSTS=… RLX_XDNA_GEMM=512,512,512 \
//   LD_LIBRARY_PATH=<xrt lib> \
//   cargo run -p rlx-runtime --features xdna --example xdna_mlp

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

// Layer sizes: x[M,D0] · W1[D0,D1] -> relu -> · W2[D1,D2].
const M: usize = 128;
const D0: usize = 384;
const D1: usize = 512;
const D2: usize = 256;

fn matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn matmul_host(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += av * b[kk * n + j];
            }
        }
    }
    c
}

fn relu(v: &mut [f32]) {
    for x in v {
        *x = x.max(0.0);
    }
}

fn main() {
    if !is_available(Device::Xdna) {
        eprintln!(
            "Device::Xdna not available — set RLX_XDNA_SHIM/XCLBIN/INSTS/GEMM (+ LD_LIBRARY_PATH)."
        );
        std::process::exit(2);
    }

    // Fractional weights with per-channel outliers (the realistic quant case).
    let x: Vec<f32> = (0..M * D0).map(|i| (i as f32 * 0.021).sin()).collect();
    let w1: Vec<f32> = (0..D0 * D1)
        .map(|i| {
            (i as f32 * 0.013).cos() * 0.4 * if (i % D1).is_multiple_of(8) { 6.0 } else { 1.0 }
        })
        .collect();
    let w2: Vec<f32> = (0..D1 * D2)
        .map(|i| (i as f32 * 0.009).sin() * 0.3)
        .collect();

    // Run one matmul on the NPU (Device::Xdna): quantize→NPU→dequant, arbitrary shape.
    let npu_matmul = |a: &[f32], w: &[f32], m: usize, k: usize, n: usize| -> Vec<f32> {
        let mut s = Session::new(Device::Xdna).compile(matmul_graph(m, k, n));
        s.set_param("W", w);
        s.run(&[("x", a)]).into_iter().next().unwrap()
    };

    // ── NPU MLP: both matmuls on the NPU, relu on the host ──
    let mut h = npu_matmul(&x, &w1, M, D0, D1);
    relu(&mut h);
    let y_npu = npu_matmul(&h, &w2, M, D1, D2);

    // ── CPU f32 reference ──
    let mut h_ref = matmul_host(&x, &w1, M, D0, D1);
    relu(&mut h_ref);
    let y_ref = matmul_host(&h_ref, &w2, M, D1, D2);

    // Relative L2 error (dominated by int8 quantization of the two layers).
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let diff: Vec<f32> = y_npu.iter().zip(&y_ref).map(|(a, b)| a - b).collect();
    let rel = l2(&diff) / l2(&y_ref).max(1e-9);

    if rel < 0.10 {
        println!(
            "rlx Device::Xdna MLP  relu(x@W1)@W2  ({M}x{D0} · {D0}x{D1} → relu → · {D1}x{D2}): \
             PASS ✓  both matmuls on NPU; end-to-end vs CPU f32 = {:.2}% (int8, 2 layers)",
            rel * 100.0
        );
    } else {
        println!(
            "rlx Device::Xdna MLP: FAIL ✗ — rel error {:.2}% too high",
            rel * 100.0
        );
        std::process::exit(1);
    }
}
