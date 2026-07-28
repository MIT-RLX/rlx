// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::DequantMatMul { scheme: MxFp4x2Block }` on CUDA vs a CPU reference.
//! Decode the packed two-level residual E2M1 weight into f32 [n,k] scratch
//! (`mxfp4x2_dequant_nk`), then `matmul_bt` x·Wᵀ. No-ops when CUDA is absent.

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::quant::QuantScheme;
use rlx_ir::residual::{residual_dequantize, residual_quantize};
use rlx_ir::{DType, Graph, GraphExt, Op, ScaledFormat, Shape};

#[test]
fn dequant_matmul_mxfp4x2_cuda_matches_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let (m, k, n) = (2usize, 32usize, 4usize);
    let group = k; // one MX block per column → nblk = 1
    let nblk = k.div_ceil(group);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect(); // [k,n]

    let plane = (k * n).div_ceil(2);
    let mut w_bytes = vec![0u8; 2 * plane];
    let (mut s0, mut s1) = (vec![0f32; nblk * n], vec![0f32; nblk * n]);
    let mut w_dq = vec![0f32; k * n];
    for j in 0..n {
        let col: Vec<f32> = (0..k).map(|p| w[p * n + j]).collect();
        let rb = residual_quantize(&col, ScaledFormat::F4E2M1, 2);
        s0[j] = rb.scales[0];
        s1[j] = rb.scales[1];
        let dq = residual_dequantize(&rb);
        for p in 0..k {
            let elem = p * n + j;
            let byte = elem / 2;
            let shift: u32 = if elem & 1 == 0 { 0 } else { 4 };
            let mask: u8 = 0x0Fu8 << shift;
            w_bytes[byte] = (w_bytes[byte] & !mask) | ((rb.codes[0][p] & 0x0F) << shift);
            w_bytes[plane + byte] = (w_bytes[plane + byte] & !mask) | ((rb.codes[1][p] & 0x0F) << shift);
            w_dq[elem] = dq[p];
        }
    }
    let mut scales = s0.clone();
    scales.extend_from_slice(&s1); // [s0 | s1]

    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[r * k + p] * w_dq[p * n + c];
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("dq_mxfp4x2_cuda");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_param = g.param("w_q", Shape::new(&[2 * plane], DType::U8));
    let scale_param = g.param("scale", Shape::new(&[2 * nblk * n], DType::F32));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::MxFp4x2Block { group_size: group as u32 },
        },
        vec![x_in, w_param, scale_param],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param_bytes("w_q", &w_bytes);
    exe.set_param("scale", &scales);
    let out = exe.run(&[("x", &x)]);
    let max_abs = out[0]
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "MxFp4x2 DequantMatMul CUDA mismatch (max|Δ|={max_abs}): got {:?} want {expected:?}",
        out[0]
    );
}
