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
//! vmap smoke for batched `Op::DequantGroupedMatMul`.

use rlx_autodiff::vmap;
use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

const QK_K: usize = 256;

fn build_q8k_stack(e: usize, k: usize, n: usize) -> Vec<u8> {
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);
    let mut packed = Vec::new();
    for _ in 0..e {
        for _ in 0..n {
            packed.extend_from_slice(&0.5f32.to_le_bytes());
            for &q in &qs {
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
    }
    let _ = k;
    packed
}

#[test]
fn vmap_dequant_grouped_matmul_runs() {
    let k = 256;
    let n = 2;
    let m = 2;
    let batch = 2;
    let num_experts = 2;
    let packed = build_q8k_stack(num_experts, k, n);

    let mut g = Graph::new("vmap_dq_gmm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let idx = g.input("idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x, w, idx],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let batched = vmap(&g, &["x", "idx"], batch);
    let session = Session::new(Device::Cpu);
    let mut exe = session.compile(batched);
    exe.set_param_typed("w", &packed, DType::U8);

    let x_val: Vec<f32> = (0..batch * m * k).map(|i| 0.001 * i as f32).collect();
    let idx_val: Vec<f32> = (0..batch * m).map(|i| (i % num_experts) as f32).collect();
    let out = exe.run(&[("x", &x_val), ("idx", &idx_val)]);
    assert_eq!(out[0].len(), batch * m * n);
    assert!(out[0].iter().all(|v| v.is_finite()));
}
