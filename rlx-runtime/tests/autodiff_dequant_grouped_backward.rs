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
//! Autodiff smoke for `Op::DequantGroupedMatMul` (dx via DequantMoEWeights + GroupedMatMul VJP).

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const QK_K: usize = 256;

fn build_q8k_expert_stack(num_experts: usize, k: usize, n: usize, scales: &[f32]) -> Vec<u8> {
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);
    let mut packed = Vec::new();
    for &scale in scales {
        for _ in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for &q in &qs {
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
    }
    let _ = (num_experts, k);
    packed
}

#[test]
fn dequant_grouped_matmul_dx_is_nonzero() {
    let k = 256;
    let n = 4;
    let m = 3;
    let num_experts = 2;
    let packed = build_q8k_expert_stack(num_experts, k, n, &[0.5, 1.0]);

    let mut g = Graph::new("dq_gmm_ad");
    let x = g.param("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
    let idx = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x, w, idx],
        Shape::new(&[m, n], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(m * n) as i64],
        },
        vec![y],
        Shape::new(&[m * n], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let session = Session::new(Device::Cpu);
    let mut exe = session.compile(bwd);

    let x_val: Vec<f32> = (0..m * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let idx_val = vec![0.0, 1.0, 0.0];
    exe.set_param("x", &x_val);
    exe.set_param_typed("w_packed", &packed, DType::U8);

    let outs = exe.run(&[("expert_idx", idx_val.as_slice()), ("d_output", &[1.0f32])]);
    let loss = outs[0][0];
    assert!(loss.is_finite() && loss.abs() > 1e-6, "loss={loss}");
    let dx = &outs[1];
    assert_eq!(dx.len(), x_val.len());
    assert!(
        dx.iter().any(|v| v.abs() > 1e-6),
        "dx should be non-zero for DequantGroupedMatMul: {dx:?}"
    );
}
