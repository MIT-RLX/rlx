// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
// End-to-end test: run `Op::DequantMatMul { scheme: GgufQ8K }` against
// a manually-built graph and compare against the reference path of
// "dequant the weight bytes to f32, then plain matmul." Both should
// produce identical (modulo dequant order) outputs.

use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

const QK_K: usize = 256;

/// Build one Q8_K block (276 bytes / 256 elements):
///   f32 d                  (4 bytes)
///   i8 qs[256]             (256 bytes)
///   i16 bsums[16]          (32 bytes, only used by Q8_K×Q8_K accum;
///                           plain dequant ignores them)
fn build_one_q8_k_block(scale: f32, qs: &[i8; QK_K]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(276);
    bytes.extend_from_slice(&scale.to_le_bytes());
    for &q in qs {
        bytes.push(q as u8);
    }
    for _ in 0..(QK_K / 16) {
        bytes.extend_from_slice(&0i16.to_le_bytes());
    }
    bytes
}

#[test]
fn dequant_matmul_q8k_matches_dequant_then_matmul() {
    // Weight: [k, n] = [256, 4], packed as 4 Q8_K blocks (one per
    // output column n). Each block has scale=0.0625 and qs[i]=i-128
    // (covers the full i8 range).
    let k = 256;
    let _n = 4;
    let scale = 0.0625f32;
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);

    // Build the packed weight buffer. Layout: 4 super-blocks of 276
    // bytes each — first block holds column 0's 256 weights, etc.
    // Wait — Q8_K's super-block is 256 elements. For a [k=256, n=4]
    // weight in row-major, the 256*4 elements are laid out
    // (k0,n0)(k0,n1)(k0,n2)(k0,n3)(k1,n0)... — one block per 256
    // consecutive elements means each block spans 64 rows × 4 cols,
    // not 256 rows × 1 col. To keep the test simple, use a single
    // column (n=1) so each block maps cleanly to one column's full
    // 256 rows.
    let n = 1; // override
    let total = k * n;
    let n_blocks = total / QK_K;
    assert_eq!(n_blocks, 1);
    let packed = build_one_q8_k_block(scale, &qs);

    // Reference dequant.
    let w_ref = rlx_gguf::dequant_q8_k(&packed, total).unwrap();
    assert_eq!(w_ref.len(), total);
    // Sanity: scale * (i-128).
    for i in 0..QK_K {
        assert!((w_ref[i] - scale * (qs[i] as f32)).abs() < 1e-6);
    }

    // Input x: [m, k] = [2, 256], arbitrary values.
    let m = 2;
    let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001 - 0.5).collect();

    // Reference: pure CPU matmul x @ w_ref → [m, n].
    let mut expected = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w_ref[kk * n + j];
            }
            expected[i * n + j] = acc;
        }
    }

    // Build the rlx graph: Op::DequantMatMul { scheme: GgufQ8K }.
    let mut g = Graph::new("dq_matmul_q8k");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    // Weight: U8 byte tensor with `packed.len()` elements.
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_packed],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let outputs = compiled.run(&[("x", x.as_slice())]);
    let actual = outputs.into_iter().next().unwrap();
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        assert!(
            diff < 1e-3,
            "mismatch at {i}: got {} expected {} (diff {})",
            actual[i],
            expected[i],
            diff
        );
    }
}

/// Layout regression for n>1 — see `run_q8k_layout_case`.
#[test]
fn dequant_matmul_q8k_correct_layout_for_n_gt_1() {
    run_q8k_layout_case(Device::Cpu);
}

#[test]
fn dequant_matmul_q6k_runs_without_panicking() {
    // Q6_K block: [128 ql + 64 qh + 16 i8 scales + 2 (f16 d)] = 210
    // bytes / 256 elements. Hand-built with d=1, every scale=1,
    // every 6-bit quant value = 32 (which decodes to 0 after the
    // -32 bias) → output all zeros.
    let ql_len = QK_K / 2;
    let qh_len = QK_K / 4;
    let sc_len = QK_K / 16;
    let mut packed = Vec::with_capacity(ql_len + qh_len + sc_len + 2);
    packed.resize(ql_len, 0u8); // low nibbles = 0
    packed.resize(ql_len + qh_len, 0xAAu8); // high 2 bits = 2 each
    packed.extend(std::iter::repeat_n(1u8, sc_len));
    packed.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

    // [k=256, n=1] @ [k=256] = [m=1, n=1]
    let k = 256;
    let n = 1;
    let m = 1;
    let x = vec![1.0f32; m * k];

    let mut g = Graph::new("dq_matmul_q6k");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ6K,
        },
        vec![x_in, w_packed],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let outputs = compiled.run(&[("x", x.as_slice())]);
    let actual = outputs.into_iter().next().unwrap();
    assert_eq!(actual.len(), 1);
    assert!(
        actual[0].abs() < 1e-4,
        "Q6_K decoded weight should be all zeros, got {}",
        actual[0]
    );
}

fn run_q8k_layout_case(device: Device) {
    let k = 256;
    let n = 4;
    let m = 2;

    let mut packed = Vec::with_capacity(n * 292);
    let scale = 1.0f32;
    for j in 0..n {
        packed.extend_from_slice(&scale.to_le_bytes());
        for i in 0..QK_K {
            let v = (j as i32 * 1000) + (i as i32 - 128);
            let q = v.clamp(-128, 127) as i8;
            packed.push(q as u8);
        }
        for _ in 0..(QK_K / 16) {
            packed.extend_from_slice(&0i16.to_le_bytes());
        }
    }
    assert_eq!(packed.len(), n * 292);

    let w_ref = rlx_gguf::dequant_q8_k(&packed, k * n).unwrap();
    assert_eq!(w_ref.len(), k * n);

    let x: Vec<f32> = (0..(m * k)).map(|i| 0.01 * (i as f32 + 1.0)).collect();

    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w_ref[c * k + i];
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("q8k_layout");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_packed],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let outputs = compiled.run(&[("x", x.as_slice())]);
    let actual = outputs.into_iter().next().unwrap();
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        let rel = diff / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-4,
            "{device:?} layout-bug regression at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_q8k_metal_matches_cpu() {
    run_q8k_layout_case(Device::Metal);
}

fn run_q4k_case(device: Device) {
    use half::f16;
    const K_SCALE_SIZE: usize = 12;
    let mut packed = Vec::new();
    packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
    packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
    let mut scales = [0u8; K_SCALE_SIZE];
    for s in &mut scales[0..4] {
        *s = 0x01;
    }
    packed.extend_from_slice(&scales);
    packed.extend(std::iter::repeat_n(0x77u8, QK_K / 2));

    let k = 256;
    let n = 1;
    let m = 2;
    let w_ref = rlx_gguf::dequant_q4_k(&packed, k * n).unwrap();
    let x: Vec<f32> = (0..(m * k)).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w_ref[c * k + i];
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("q4k_matmul");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![x_in, w_packed],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled.run(&[("x", x.as_slice())]).pop().unwrap();
    for i in 0..actual.len() {
        let rel = (actual[i] - expected[i]).abs() / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-3,
            "{device:?} q4k mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}

#[test]
fn dequant_matmul_q4k_matches_reference() {
    run_q4k_case(Device::Cpu);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_matmul_q4k_metal_matches_cpu() {
    run_q4k_case(Device::Metal);
}
