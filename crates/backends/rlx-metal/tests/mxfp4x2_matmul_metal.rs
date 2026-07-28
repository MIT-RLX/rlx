// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::DequantMatMul { scheme: MxFp4x2Block }` — Metal vs CPU parity.
//! The fused `dequant_matmul_mxfp4x2` MSL kernel decodes the packed two-level
//! residual E2M1 weight ([plane0|plane1] nibbles + [s0|s1] f32 scales) and
//! matmuls with x in one pass. No-ops when Metal is unavailable.

#![cfg(target_os = "macos")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::residual::residual_quantize;
use rlx_ir::{DType, Graph, Op, ScaledFormat, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn dequant_matmul_mxfp4x2_metal_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
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
    for j in 0..n {
        let col: Vec<f32> = (0..k).map(|p| w[p * n + j]).collect();
        let rb = residual_quantize(&col, ScaledFormat::F4E2M1, 2);
        s0[j] = rb.scales[0];
        s1[j] = rb.scales[1];
        for p in 0..k {
            let elem = p * n + j;
            let byte = elem / 2;
            let shift: u32 = if elem & 1 == 0 { 0 } else { 4 };
            let mask: u8 = 0x0Fu8 << shift;
            w_bytes[byte] = (w_bytes[byte] & !mask) | ((rb.codes[0][p] & 0x0F) << shift);
            w_bytes[plane + byte] =
                (w_bytes[plane + byte] & !mask) | ((rb.codes[1][p] & 0x0F) << shift);
        }
    }
    let mut scales = s0.clone();
    scales.extend_from_slice(&s1); // [s0 | s1]

    let build = || {
        let mut g = Graph::new("dq_mxfp4x2_metal");
        let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
        let w_q = g.param("w_q", Shape::new(&[2 * plane], DType::U8));
        let scale = g.param("scale", Shape::new(&[2 * nblk * n], DType::F32));
        let y = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::MxFp4x2Block {
                    group_size: group as u32,
                },
            },
            vec![x_in, w_q, scale],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };

    let run = |device| {
        let mut c = Session::new(device).compile(build());
        c.set_param_typed("w_q", &w_bytes, DType::U8);
        c.set_param("scale", &scales);
        c.run(&[("x", x.as_slice())]).remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    let max_abs = metal
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "MxFp4x2 DequantMatMul Metal vs CPU mismatch (max|Δ|={max_abs}): {metal:?} vs {cpu:?}"
    );
}
