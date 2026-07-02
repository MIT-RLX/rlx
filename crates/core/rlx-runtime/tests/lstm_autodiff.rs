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
//! Backprop-through-time for `Op::Lstm`: `prepare_graph_for_ad` unfuses
//! the op into primitives, so `grad_with_loss` differentiates it for
//! free. We check ∂(Σ y)/∂x against central finite differences.

use rlx_autodiff::{grad_with_loss, prepare_graph_for_ad};
use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const B: usize = 1;
const S: usize = 3;
const IN: usize = 2;
const H: usize = 3;

fn run_bwd(bwd: Graph, slots: &[(&str, &[f32])], out_idx: usize) -> Vec<f32> {
    let (bwd, remap) = run_with_remap(bwd);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = rlx_opt::memory::plan_memory(&bwd);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&bwd, &arena);
    for (name, data) in slots {
        let id = bwd
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name: n } if n == name))
            .map(|n| n.id)
            .expect(name);
        let off = arena.byte_offset(r(id));
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out_id = r(bwd.outputs[out_idx]);
    let n = bwd.node(out_id).shape.num_elements().unwrap();
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Σ over all outputs of the reference LSTM — the scalar loss.
fn reference_loss(x: &[f32], w_ih: &[f32], w_hh: &[f32], bias: &[f32]) -> f32 {
    let four_h = 4 * H;
    let mut loss = 0.0f32;
    for bi in 0..B {
        let mut hs = [0f32; H];
        let mut cs = [0f32; H];
        for t in 0..S {
            let xt = &x[(bi * S + t) * IN..(bi * S + t + 1) * IN];
            let mut z = vec![0f32; four_h];
            for (r, zr) in z.iter_mut().enumerate() {
                let mut acc = bias[r];
                for (j, &xj) in xt.iter().enumerate() {
                    acc += w_ih[r * IN + j] * xj;
                }
                for (j, &hj) in hs.iter().enumerate() {
                    acc += w_hh[r * H + j] * hj;
                }
                *zr = acc;
            }
            for k in 0..H {
                let i_g = sigmoid(z[k]);
                let f_g = sigmoid(z[H + k]);
                let g_g = z[2 * H + k].tanh();
                let o_g = sigmoid(z[3 * H + k]);
                let c_new = f_g * cs[k] + i_g * g_g;
                cs[k] = c_new;
                let h_new = o_g * c_new.tanh();
                hs[k] = h_new;
                loss += h_new;
            }
        }
    }
    loss
}

#[test]
fn lstm_bptt_matches_finite_differences() {
    let four_h = 4 * H;
    let f = DType::F32;
    let x: Vec<f32> = (0..B * S * IN)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let w_ih: Vec<f32> = (0..four_h * IN)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.05)
        .collect();
    let w_hh: Vec<f32> = (0..four_h * H)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.05)
        .collect();
    let bias: Vec<f32> = (0..four_h).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();

    // Build loss = Σ lstm(x).
    let mut g = Graph::new("lstm_loss");
    let x_in = g.input("x", Shape::new(&[B, S, IN], f));
    let wih = g.input("w_ih", Shape::new(&[four_h, IN], f));
    let whh = g.input("w_hh", Shape::new(&[four_h, H], f));
    let bs = g.input("bias", Shape::new(&[four_h], f));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: H,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x_in, wih, whh, bs],
        Shape::new(&[B, S, H], f),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1, 2],
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);

    let prep = prepare_graph_for_ad(g);
    let bwd = grad_with_loss(&prep, &[x_in]);
    let dx = run_bwd(
        bwd,
        &[
            ("x", &x),
            ("w_ih", &w_ih),
            ("w_hh", &w_hh),
            ("bias", &bias),
            ("d_output", &[1.0]),
        ],
        1,
    );
    assert_eq!(dx.len(), x.len());

    // Central finite differences on the reference loss.
    let eps = 1e-3f32;
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (reference_loss(&xp, &w_ih, &w_hh, &bias)
            - reference_loss(&xm, &w_ih, &w_hh, &bias))
            / (2.0 * eps);
        assert!(
            (dx[i] - fd).abs() < 2e-2,
            "BPTT grad mismatch at x[{i}]: autodiff {} vs finite-diff {}",
            dx[i],
            fd
        );
    }
}
