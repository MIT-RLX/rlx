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

//! VJP of `Op::Sort` vs finite differences. Loss = Σ position-weighted sorted
//! values, so the gradient genuinely tests the scatter-by-argsort permutation.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const N: usize = 6;

fn loss_graph(g: &mut Graph, x: NodeId) -> NodeId {
    let s = Shape::new(&[N], DType::F32);
    let y = g.sort(x, 0, false, s.clone());
    // Distinct position weights c_k = k+1 so the loss depends on sorted order.
    let cw: Vec<f32> = (0..N).map(|k| (k as f32) + 1.0).collect();
    let c_bytes: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = g.add_node(Op::Constant { data: c_bytes }, vec![], s.clone());
    let yc = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, c], s);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![yc],
        Shape::from_dims(&[], DType::F32),
    )
}

fn loss_at(xv: &[f32]) -> f32 {
    let mut g = Graph::new("l");
    let x = g.input("x", Shape::new(&[N], DType::F32));
    let loss = loss_graph(&mut g, x);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("x", xv)])
        .pop()
        .unwrap()[0]
}

#[test]
fn sort_vjp_matches_fd() {
    // Distinct, well-separated values → small perturbations never reorder.
    let x: Vec<f32> = vec![3.0, 1.0, 4.0, 1.5, -2.0, 0.7];
    let mut g = Graph::new("g");
    let xi = g.param("x", Shape::new(&[N], DType::F32));
    let loss = loss_graph(&mut g, xi);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[xi]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("x", &x);
    let dx = c.run(&[("d_output", &[1.0f32])])[1].clone();

    let eps = 1e-3f32;
    for k in 0..N {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[k] += eps;
        xm[k] -= eps;
        let fd = (loss_at(&xp) - loss_at(&xm)) / (2.0 * eps);
        assert!(
            (fd - dx[k]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "dx[{k}]: analytic {} vs FD {fd}",
            dx[k]
        );
    }
}
