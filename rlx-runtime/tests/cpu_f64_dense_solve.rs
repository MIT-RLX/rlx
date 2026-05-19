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

//! Hello Resistor end-to-end at f64 through the public Session /
//! CompiledGraph API. Validates the F64 path through `run_typed`
//! and `set_param_typed` — direct byte writes into the CPU arena
//! with no f32 widening (which would lose precision).
//!
//! Pairs with `cpu_grad_finite_difference.rs`: that test exercises
//! AD on f32 transformer-flavored ops; this one exercises AD on the
//! Circulax-flavored DenseSolve path at full f64 precision.

#![cfg(feature = "cpu")]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_opt::autodiff_fwd::jvp;
use rlx_runtime::jacfwd::jacfwd;
use rlx_runtime::{Device, Session};

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    assert!(
        bytes.len().is_multiple_of(8),
        "byte length {} not divisible by 8",
        bytes.len()
    );
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn find_by_name(graph: &Graph, want: &str) -> NodeId {
    for node in graph.nodes() {
        let name = match &node.op {
            Op::Input { name } => Some(name.as_str()),
            Op::Param { name } => Some(name.as_str()),
            _ => None,
        };
        if name == Some(want) {
            return node.id;
        }
    }
    panic!("no node named {want:?}");
}

#[test]
fn f64_forward_dense_solve_via_run_typed() {
    // x = solve(A, b) with A SPD tridiagonal 3x3.
    let mut g = Graph::new("solve_runtime");
    let a = g.input("A", Shape::new(&[3, 3], DType::F64));
    let b = g.input("b", Shape::new(&[3], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[3], DType::F64));
    g.set_outputs(vec![x]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];

    let outs = compiled.run_typed(&[
        ("A", &f64s_to_bytes(&a_data), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
    ]);

    assert_eq!(outs.len(), 1);
    let (bytes, dtype) = &outs[0];
    assert_eq!(*dtype, DType::F64);
    let x_got = bytes_to_f64s(bytes);

    // Verify A·x ≈ b (residual check — tighter than comparing element-
    // wise to a hand-computed solution; works for any well-conditioned A).
    let mut residual = [0.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            residual[i] += a_data[i * 3 + j] * x_got[j];
        }
    }
    for i in 0..3 {
        assert!(
            (residual[i] - b_data[i]).abs() < 1e-10,
            "residual[{i}] = {} vs b {}",
            residual[i],
            b_data[i]
        );
    }
}

#[test]
fn f64_jvp_dense_solve_via_run_typed() {
    // Forward: x = solve(A, b). JVP w.r.t. b alone.
    // Output: [primal_x, tangent_x] where tangent_x = solve(A, t_b).
    let n = 3usize;
    let mut g = Graph::new("jvp_runtime");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    let jg = jvp(&g, &[b]);
    let mut compiled = Session::new(Device::Cpu).compile(jg);

    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];
    let tb: [f64; 3] = [0.5, -0.25, 1.0];

    let outs = compiled.run_typed(&[
        ("A", &f64s_to_bytes(&a_data), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("tangent_b", &f64s_to_bytes(&tb), DType::F64),
    ]);
    assert_eq!(outs.len(), 2, "[primal_x, tangent_x]");
    let _primal_x = bytes_to_f64s(&outs[0].0);
    let tangent_x = bytes_to_f64s(&outs[1].0);

    // Closed form: t_x = solve(A, t_b).
    let mut a_ref = a_data;
    let mut tb_ref = tb;
    let info = rlx_cpu::blas::dgesv(&mut a_ref, &mut tb_ref, n, 1);
    assert_eq!(info, 0);
    for i in 0..n {
        assert!(
            (tangent_x[i] - tb_ref[i]).abs() < 1e-10,
            "t_x[{i}]: AD={} ref={}",
            tangent_x[i],
            tb_ref[i]
        );
    }
}

#[test]
fn f64_jacfwd_recovers_inverse_for_dense_solve() {
    // x = solve(A, b)   ⇒   ∂x/∂b = A⁻¹.
    // Build the JVP graph perturbing b, hand it to `jacfwd`, and
    // assert the materialized Jacobian matches `np.linalg.inv(A)`
    // (well, `dgesv(A, I)`).
    let n = 3usize;
    let mut g = Graph::new("jac_inverse_runtime");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    let jg = jvp(&g, &[b]);
    let mut compiled = Session::new(Device::Cpu).compile(jg);

    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];

    let a_bytes = f64s_to_bytes(&a_data);
    let b_bytes = f64s_to_bytes(&b_data);

    let jacs = jacfwd(
        &mut compiled,
        &[("A", &a_bytes, DType::F64), ("b", &b_bytes, DType::F64)],
        "b",
        &[n],
        DType::F64,
    );

    assert_eq!(jacs.len(), 1, "one primal output ⇒ one Jacobian");
    let jac = &jacs[0];
    assert_eq!(jac.output_size, n);
    assert_eq!(jac.wrt_size, n);
    assert_eq!(jac.dtype, DType::F64);

    // Compute reference inverse via the same dgesv: solve(A, I) = A⁻¹.
    let mut a_ref = a_data;
    let mut rhs = [0.0_f64; 9];
    for i in 0..n {
        rhs[i * n + i] = 1.0;
    } // identity, row-major
    // dgesv expects rhs as [N, nrhs] row-major (column index = nrhs).
    // Here nrhs = N. The dgesv wrapper handles row→col→row internally.
    let info = rlx_cpu::blas::dgesv(&mut a_ref, &mut rhs, n, n);
    assert_eq!(info, 0);
    // `rhs` now holds A⁻¹ row-major, same layout as the Jacobian.
    let jac_data = jac.as_f64();
    for i in 0..n * n {
        assert!(
            (jac_data[i] - rhs[i]).abs() < 1e-10,
            "jac[{i}] = {} vs A⁻¹[{i}] = {}",
            jac_data[i],
            rhs[i]
        );
    }
}

#[test]
fn f64_hello_resistor_gradient_via_run_typed() {
    // Forward: A param, b input, x = solve(A, b), loss = sum(x).
    // Reverse: grad_with_loss → bwd graph with [loss, dA, db].
    // Inputs to the bwd graph: A, b, d_output (seed = 1.0).
    let mut g = Graph::new("hello_resistor_runtime");
    let n = 3usize;
    let a = g.param("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    let loss = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    assert_eq!(bwd.outputs.len(), 3, "[loss, dA, db]");

    // The bwd graph still has A as a Param (forward mirror) — Session
    // calls Param "params" and Input "inputs". Since grad_with_loss
    // copies Param nodes verbatim, A is still a Param in bwd; we
    // upload via set_param_typed. b and d_output are Inputs.
    let _a_bwd = find_by_name(&bwd, "A"); // Param — via set_param_typed
    let _b_bwd = find_by_name(&bwd, "b"); // Input — via run_typed
    let _d_out = find_by_name(&bwd, "d_output");

    let mut compiled = Session::new(Device::Cpu).compile(bwd);

    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];
    let d_out: [f64; 1] = [1.0];

    compiled.set_param_typed("A", &f64s_to_bytes(&a_data), DType::F64);

    let outs = compiled.run_typed(&[
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&d_out), DType::F64),
    ]);

    assert_eq!(outs.len(), 3);
    for (_, dt) in &outs {
        assert_eq!(*dt, DType::F64, "all outputs should be F64");
    }
    let loss_out = bytes_to_f64s(&outs[0].0);
    let da_out = bytes_to_f64s(&outs[1].0);
    let db_out = bytes_to_f64s(&outs[2].0);
    assert_eq!(loss_out.len(), 1);
    assert_eq!(da_out.len(), n * n);
    assert_eq!(db_out.len(), n);

    // Reference: solve forward, solve transpose for db, outer product for dA.
    let mut a_ref = a_data;
    let mut b_ref = b_data;
    let info = rlx_cpu::blas::dgesv(&mut a_ref, &mut b_ref, n, 1);
    assert_eq!(info, 0);
    let x_ref = b_ref;
    let loss_ref: f64 = x_ref.iter().sum();

    let mut at = [0.0_f64; 9];
    for i in 0..n {
        for j in 0..n {
            at[i * n + j] = a_data[j * n + i];
        }
    }
    let mut ones = [1.0_f64; 3];
    let info = rlx_cpu::blas::dgesv(&mut at, &mut ones, n, 1);
    assert_eq!(info, 0);
    let db_ref = ones;

    let mut da_ref = [0.0_f64; 9];
    for i in 0..n {
        for j in 0..n {
            da_ref[i * n + j] = -db_ref[i] * x_ref[j];
        }
    }

    assert!(
        (loss_out[0] - loss_ref).abs() < 1e-10,
        "loss: got {} want {}",
        loss_out[0],
        loss_ref
    );
    for i in 0..n {
        assert!(
            (db_out[i] - db_ref[i]).abs() < 1e-10,
            "db[{i}]: got {} want {}",
            db_out[i],
            db_ref[i]
        );
    }
    for i in 0..n * n {
        assert!(
            (da_out[i] - da_ref[i]).abs() < 1e-10,
            "dA[{i}]: got {} want {}",
            da_out[i],
            da_ref[i]
        );
    }
}
