// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Block-quantized int8 / int4 `Op::DequantMatMul` on Metal vs CPU. These ran
//! as a host fallback over unified memory; they now have native MSL kernels
//! (`dequant_matmul_int8` / `_int4`) — the GPU path for running calibrated
//! AWQ/GPTQ weights. Outputs must match the CPU reference.

#![cfg(target_os = "macos")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run_case(scheme: QuantScheme, m: usize, k: usize, n: usize, block: usize) {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let n_blocks = k.div_ceil(block);
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let scales: Vec<f32> = (0..n_blocks * n)
        .map(|i| 0.02 + 0.001 * (i % 7) as f32)
        .collect();
    let zp: Vec<f32> = vec![0.0; n_blocks * n]; // symmetric

    // Quantized weight bytes per scheme.
    let (w_bytes, is_int4) = match scheme {
        QuantScheme::Int4Block { .. } => {
            // u4 nibbles in [0,15], packed 2 per byte (low first).
            let nibs: Vec<u8> = (0..k * n).map(|i| ((i * 5 + 3) % 16) as u8).collect();
            let mut packed = vec![0u8; (k * n).div_ceil(2)];
            for (idx, &nv) in nibs.iter().enumerate() {
                packed[idx >> 1] |= nv << ((idx & 1) * 4);
            }
            (packed, true)
        }
        _ => {
            // i8 in [-100,100] reinterpreted as bytes.
            let w: Vec<i8> = (0..k * n)
                .map(|i| (((i * 37 + 11) % 201) as i32 - 100) as i8)
                .collect();
            (w.iter().map(|&v| v as u8).collect(), false)
        }
    };

    let mut g = Graph::new("dq_int");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_dt = if is_int4 { DType::U8 } else { DType::I8 };
    let w = g.param("w", Shape::new(&[w_bytes.len()], w_dt));
    let s = g.param("scale", Shape::new(&[n_blocks, n], DType::F32));
    let z = g.param("zp", Shape::new(&[n_blocks, n], DType::F32));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w, s, z],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &w_bytes, w_dt);
        c.set_param_typed("scale", &f32_bytes(&scales), DType::F32);
        c.set_param_typed("zp", &f32_bytes(&zp), DType::F32);
        c.run(&[("x", x.as_slice())]).remove(0)
    };

    let metal = run(Device::Metal);
    let cpu = run(Device::Cpu);
    assert_eq!(metal.len(), m * n);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("dequant_matmul {scheme:?}: max_abs={max_abs:.6}");
    assert!(max_abs < 1e-3, "{scheme:?} Metal vs CPU max_abs={max_abs}");
}

#[test]
fn metal_dequant_matmul_int8_matches_cpu() {
    run_case(QuantScheme::Int8Block { block_size: 32 }, 2, 64, 8, 32);
}

#[test]
fn metal_dequant_matmul_int4_matches_cpu() {
    run_case(QuantScheme::Int4Block { block_size: 32 }, 2, 64, 8, 32);
}

#[test]
fn metal_dequant_matmul_int8_gemv() {
    // m = 1 (decode shape) — one thread per output column.
    run_case(QuantScheme::Int8Block { block_size: 16 }, 1, 32, 12, 16);
}
