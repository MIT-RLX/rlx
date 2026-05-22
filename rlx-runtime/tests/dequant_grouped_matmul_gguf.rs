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
//! End-to-end test: `Op::DequantGroupedMatMul { scheme: GgufQ8K }` on a
//! synthetic MoE expert stack (no real GGUF file). Compares the in-graph
//! op against per-token expert dequant + matmul reference.

use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

const QK_K: usize = 256;

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

/// Packed expert stack: `num_experts` slabs of shape `[n, k]` in GGUF Q8_K layout.
fn build_q8k_expert_stack(
    num_experts: usize,
    k: usize,
    n: usize,
    expert_scales: &[f32],
) -> Vec<u8> {
    assert_eq!(expert_scales.len(), num_experts);
    assert_eq!((k * n) % QK_K, 0);
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);
    let mut packed = Vec::with_capacity(num_experts * n * 292);
    for &scale in expert_scales {
        for _ in 0..n {
            packed.extend_from_slice(&build_one_q8_k_block(scale, &qs));
        }
    }
    packed
}

fn reference_grouped_q8k(
    x: &[f32],
    packed: &[u8],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
) -> Vec<f32> {
    let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        let e = expert_idx[row] as usize;
        assert!(e < num_experts);
        let w_ref = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * w_ref[c * k + i];
            }
            out[row * n + c] = acc;
        }
    }
    out
}

fn run_grouped_q8k_case(device: Device) {
    let k = 256;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    // Distinct scales so routing to the wrong expert is obvious.
    let expert_scales = [0.25f32, 0.5, 1.0];
    let packed = build_q8k_expert_stack(num_experts, k, n, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    // Non-contiguous expert ids exercise sort + unpermute in the kernel.
    let expert_idx = vec![1.0, 0.0, 2.0, 1.0, 0.0];
    let expected = reference_grouped_q8k(&x, &packed, &expert_idx, m, k, n, num_experts);

    let mut g = Graph::new("dq_grouped_matmul_q8k");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_packed = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_packed, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let session = Session::new(device);
    let mut compiled = session.compile(g);
    compiled.set_param_typed("w_packed", &packed, DType::U8);
    let actual = compiled
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        let rel = diff / expected[i].abs().max(1.0);
        assert!(
            rel < 1e-4,
            "{device:?} grouped Q8K mismatch at {i}: got {} expected {} (rel {:.2e})",
            actual[i],
            expected[i],
            rel
        );
    }
}

#[test]
fn dequant_grouped_matmul_q8k_matches_per_expert_reference() {
    run_grouped_q8k_case(Device::Cpu);
}

/// Same synthetic stack through F32 `GroupedMatMul` must match packed path.
#[test]
fn dequant_grouped_matmul_q8k_matches_f32_grouped_matmul() {
    let k = 256;
    let n = 4;
    let m = 5;
    let num_experts = 3;
    let expert_scales = [0.25f32, 0.5, 1.0];
    let packed = build_q8k_expert_stack(num_experts, k, n, &expert_scales);
    let x: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let expert_idx = vec![1.0, 0.0, 2.0, 1.0, 0.0];

    let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
    let mut w_f32 = vec![0f32; num_experts * k * n];
    for e in 0..num_experts {
        let deq = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
        // GroupedMatMul uses sgemm with B stored row-major [k, n].
        for i in 0..k {
            for j in 0..n {
                w_f32[e * k * n + i * n + j] = deq[j * k + i];
            }
        }
    }

    let mut g_packed = Graph::new("dq_gmm_packed");
    let x_in = g_packed.input("x", Shape::new(&[m, k], DType::F32));
    let w_p = g_packed.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g_packed.input("expert_idx", Shape::new(&[m], DType::F32));
    let y_packed = g_packed.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_p, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g_packed.set_outputs(vec![y_packed]);

    let mut g_f32 = Graph::new("dq_gmm_f32");
    let x2 = g_f32.input("x", Shape::new(&[m, k], DType::F32));
    let w2 = g_f32.param("w", Shape::new(&[num_experts, k, n], DType::F32));
    let idx2 = g_f32.input("expert_idx", Shape::new(&[m], DType::F32));
    let y_f32 = g_f32.add_node(
        Op::GroupedMatMul,
        vec![x2, w2, idx2],
        Shape::new(&[m, n], DType::F32),
    );
    g_f32.set_outputs(vec![y_f32]);

    let session = Session::new(Device::Cpu);
    let mut exe_packed = session.compile(g_packed);
    exe_packed.set_param_typed("w_packed", &packed, DType::U8);
    let packed_out = exe_packed
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    let mut exe_f32 = session.compile(g_f32);
    exe_f32.set_param("w", &w_f32);
    let f32_out = exe_f32
        .run(&[("x", x.as_slice()), ("expert_idx", expert_idx.as_slice())])
        .pop()
        .unwrap();

    for i in 0..packed_out.len() {
        let diff = (packed_out[i] - f32_out[i]).abs();
        let rel = diff / f32_out[i].abs().max(1.0);
        assert!(
            rel < 1e-4,
            "packed vs F32 GroupedMatMul at {i}: {} vs {} (rel {:.2e})",
            packed_out[i],
            f32_out[i],
            rel
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dequant_grouped_matmul_q8k_metal_matches_cpu() {
    run_grouped_q8k_case(Device::Metal);
}
