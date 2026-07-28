// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// End-to-end op wiring for `Op::DequantMatMul { scheme: MxFp4x2Block }` — the
// two-level residual E2M1 (double-word-style) quantized GEMM. The op takes
// three inputs: x [m,k] f32, a packed weight `[plane0|plane1]` (E2M1 nibbles,
// 2/byte, [k,n] row-major), and `[s0|s1]` f32 scales per (k/group, n) block.
// Reference: decode the same weight via `rlx_ir::residual`, then plain matmul.

use rlx_ir::quant::QuantScheme;
use rlx_ir::residual::{residual_dequantize, residual_quantize};
use rlx_ir::*;
use rlx_runtime::{Device, Session};

#[test]
fn dequant_matmul_mxfp4x2_cpu_end_to_end() {
    let (m, k, n) = (2usize, 32usize, 3usize);
    let group = k; // one MX block per column → nblk = 1
    let nblk = k.div_ceil(group);

    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect(); // [k,n]

    // Quantize each column's K-block to a two-level residual E2M1 code, packing
    // the two nibble planes and the two f32 scale sets exactly as the CPU kernel
    // reads them back.
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

    // Reference: plain matmul of x with the residual-decoded weight.
    let mut expected = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * w_dq[p * n + j];
            }
            expected[i * n + j] = acc;
        }
    }

    let mut g = Graph::new("dq_mxfp4x2");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_q = g.param("w_q", Shape::new(&[2 * plane], DType::U8));
    let scale = g.param("scale", Shape::new(&[2 * nblk * n], DType::F32));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::MxFp4x2Block { group_size: group as u32 },
        },
        vec![x_in, w_q, scale],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_q", &w_bytes, DType::U8);
    compiled.set_param("scale", &scales);
    let outputs = compiled.run(&[("x", x.as_slice())]);
    let actual = outputs.into_iter().next().unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        assert!(diff < 1e-3, "at {i}: got {} expected {} (diff {diff})", actual[i], expected[i]);
    }
}
