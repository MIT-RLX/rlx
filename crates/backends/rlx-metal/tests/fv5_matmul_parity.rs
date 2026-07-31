// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end FV5 (Neutrino-8B ternary) `Op::DequantMatMul` — Metal vs CPU.
//! Exercises the full lower → dispatch → in-shader dequant → matmul path with a
//! synthetic FV5 pack (FV5 has no float quantizer; packs are produced offline).

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Pack one FV5 block (104 bytes) from 256 five-value codes in {-2,-1,0,1,2}.
fn pack_fv5_block(codes: &[i8], s_lo: f32, s_hi: f32) -> Vec<u8> {
    let mut b = vec![0u8; 104];
    b[0..4].copy_from_slice(&s_lo.to_le_bytes());
    b[4..8].copy_from_slice(&s_hi.to_le_bytes());
    for (j, &c) in codes.iter().enumerate() {
        let (byte, bit) = (j / 8, 1u8 << (j % 8));
        let (p, ng, hi) = match c {
            1 => (true, false, false),
            2 => (true, false, true),
            -1 => (false, true, false),
            -2 => (false, true, true),
            _ => (false, false, false),
        };
        if p {
            b[8 + byte] |= bit;
        }
        if ng {
            b[40 + byte] |= bit;
        }
        if hi {
            b[72 + byte] |= bit;
        }
    }
    b
}

fn pack_fv5b_block(qs: &[i8], s: f32) -> Vec<u8> {
    let mut b = vec![0u8; 260];
    b[0..4].copy_from_slice(&s.to_le_bytes());
    for (i, &q) in qs.iter().enumerate() {
        b[4 + i] = q as u8;
    }
    b
}

fn run(scheme: QuantScheme, packed: &[u8], m: usize, k: usize, n: usize) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return None;
    }
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let mut g = Graph::new("metal_fv5_dq");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let go = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };
    let cpu = go(Device::Cpu);
    let gpu = go(Device::Metal);
    Some(
        cpu.iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max),
    )
}

// Both cases run inside ONE `#[test]` so they execute serially — concurrent
// MPSMatrixMultiplication from multiple test threads SIGABRTs ("A and B index
// not found") on the shared Metal device (same reason the sibling GGUF prefill
// parity test collapses its cases into one test).
#[test]
fn metal_fv5_fv5b_dequant_matmul_matches_cpu() {
    let (m, k, n) = (4usize, 256usize, 8usize);

    let mut fv5 = Vec::new();
    for row in 0..n {
        let codes: [i8; 256] = std::array::from_fn(|j| match (j + row) % 5 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => -1,
            _ => -2,
        });
        fv5.extend_from_slice(&pack_fv5_block(&codes, 0.05, 0.2));
    }
    if let Some(max_abs) = run(QuantScheme::GgufFV5, &fv5, m, k, n) {
        eprintln!("metal FV5 matmul: max_abs={max_abs:.6e}");
        assert!(max_abs <= 1e-4, "metal FV5 max_abs {max_abs} > 1e-4");
    } else {
        return;
    }

    let mut fv5b = Vec::new();
    for row in 0..n {
        let qs: [i8; 256] =
            std::array::from_fn(|i| ((i as i32 * 7 + row as i32) % 251 - 125) as i8);
        fv5b.extend_from_slice(&pack_fv5b_block(&qs, 0.03));
    }
    let max_abs = run(QuantScheme::GgufFV5B, &fv5b, m, k, n).expect("Metal available");
    eprintln!("metal FV5B matmul: max_abs={max_abs:.6e}");
    assert!(max_abs <= 1e-4, "metal FV5B max_abs {max_abs} > 1e-4");
}
