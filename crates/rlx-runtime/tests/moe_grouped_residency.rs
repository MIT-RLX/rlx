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
//! GroupedMatMul is lossless under TIDE residency masks (CPU).

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn grouped_matmul_same_output_all_vs_partial_residency() {
    let m = 4usize;
    let k = 3usize;
    let n = 2usize;
    let num_experts = 3usize;

    let mut g = Graph::new("moe_gmm");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[num_experts, k, n], DType::F32));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let out = g.add_node(
        Op::GroupedMatMul,
        vec![x_in, w, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![out]);

    let mut w_data = vec![0f32; num_experts * k * n];
    for e in 0..num_experts {
        for i in 0..k * n {
            w_data[e * k * n + i] = (e as f32 + 1.0) * 0.1 + i as f32 * 0.01;
        }
    }
    let x: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.05).collect();
    let expert_idx = vec![0.0, 1.0, 1.0, 2.0];

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param("w", &w_data);

    let all_resident = vec![true, true, true];
    let partial = vec![false, true, true];

    compiled.set_moe_resident_experts(&all_resident);
    let out_all = compiled.run(&[("x", &x), ("expert_idx", &expert_idx)])[0].clone();

    compiled.set_moe_resident_experts(&partial);
    let out_part = compiled.run(&[("x", &x), ("expert_idx", &expert_idx)])[0].clone();

    assert_eq!(out_all.len(), out_part.len());
    for (i, (a, b)) in out_all.iter().zip(out_part.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "mismatch at {i}: all={a} partial={b}");
    }
}
