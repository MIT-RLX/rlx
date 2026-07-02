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

//! New sparse-package improvements: PCG (Jacobi-preconditioned CG)
//! and graph-level CSR-values transpose.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};
use rlx_sparse::SparseTensor;

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn const_i32(g: &mut Graph, xs: &[i32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::I32),
    )
}
fn const_f64(g: &mut Graph, xs: &[f64]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F64),
    )
}

fn build_tridiag_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

fn build_nonsym_4() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let values = vec![5.0, -1.0, -2.0, 4.0, -1.0, -2.0, 4.0, -1.0, -2.0, 3.0];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    (values, col_idx, row_ptr)
}

// ── Transpose Values ──────────────────────────────────────────────

#[test]
fn sparse_transpose_values_recovers_a_when_symmetric() {
    rlx_sparse::register();
    // Symmetric A: transpose should yield bit-exact same values
    // (in the same nnz order, since the pattern is symmetric).
    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let nnz = values.len();
    // Compute the transposed pattern (structural).
    let (cit, rpt) = rlx_sparse::csr_transpose_pattern(&col_idx, &row_ptr, n, n);
    // For symmetric tridiag the pattern is symmetric so cit/rpt should
    // equal col_idx/row_ptr — but they're NOT in the same order
    // because the transposed walk visits entries column-by-column.
    // The point of this test isn't pattern equality; it's that
    // applying transpose to symmetric A reproduces A's value-set.

    let mut g = Graph::new("transpose_values");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let v_t = a.transpose_values(&mut g, cit_n, rpt_n);
    g.set_outputs(vec![v_t]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let v_t_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(v_t_got.len(), nnz);

    // Densify both A (original) and Aᵀ (from transpose_values + transposed pattern)
    // and compare. They must equal as dense matrices since A is symmetric.
    let mut a_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a_dense[r * n + col_idx[k] as usize] = values[k];
        }
    }
    let mut at_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in rpt[r] as usize..rpt[r + 1] as usize {
            at_dense[r * n + cit[k] as usize] = v_t_got[k];
        }
    }
    for i in 0..n * n {
        assert!(
            (a_dense[i] - at_dense[i]).abs() < 1e-12,
            "Aᵀ[{i}] = {} vs A[{i}] = {} (symmetric → equal)",
            at_dense[i],
            a_dense[i]
        );
    }
}

#[test]
fn sparse_transpose_values_correctly_transposes_nonsymmetric() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let (cit, rpt) = rlx_sparse::csr_transpose_pattern(&col_idx, &row_ptr, n, n);

    let mut g = Graph::new("transpose_values_nonsym");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let cit_n = const_i32(&mut g, &cit);
    let rpt_n = const_i32(&mut g, &rpt);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let v_t = a.transpose_values(&mut g, cit_n, rpt_n);
    g.set_outputs(vec![v_t]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let v_t_got = bytes_to_f64s(&outs[0].0);

    // Densify both and check Aᵀ[i, j] == A[j, i].
    let mut a_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            a_dense[r * n + col_idx[k] as usize] = values[k];
        }
    }
    let mut at_dense = vec![0f64; n * n];
    for r in 0..n {
        for k in rpt[r] as usize..rpt[r + 1] as usize {
            at_dense[r * n + cit[k] as usize] = v_t_got[k];
        }
    }
    for i in 0..n {
        for j in 0..n {
            assert!(
                (at_dense[i * n + j] - a_dense[j * n + i]).abs() < 1e-12,
                "Aᵀ[{i},{j}] = {} vs A[{j},{i}] = {}",
                at_dense[i * n + j],
                a_dense[j * n + i]
            );
        }
    }
}

#[test]
fn sparse_transpose_values_vjp_self_inverse() {
    // VJP of transpose is itself transpose. Verify by FD parity
    // through a graph that does (transpose ∘ sum-loss) and gets
    // dL/d(values).
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_nonsym_4();
    let n = 4;
    let nnz = values.len();
    let (cit, rpt) = rlx_sparse::csr_transpose_pattern(&col_idx, &row_ptr, n, n);

    let build = || {
        let mut g = Graph::new("trans_grad");
        let v = g.input("values", Shape::new(&[nnz], DType::F64));
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let cit_n = const_i32(&mut g, &cit);
        let rpt_n = const_i32(&mut g, &rpt);
        let a = SparseTensor::from_csr(v, ci, rp, n, n);
        let v_t = a.transpose_values(&mut g, cit_n, rpt_n);
        let loss = g.sum(v_t, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, v)
    };

    let (g, v_in) = build();
    let bwd = grad_with_loss(&g, &[v_in]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("values", &f64s_to_bytes(&values), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dv = bytes_to_f64s(&outs[1].0);
    assert_eq!(dv.len(), nnz);

    // Loss = sum(values_T) = sum(values_T) — every entry contributes
    // exactly once to the loss (it's permuted, not duplicated). So
    // dL/d(values[k]) should be 1.0 for every k.
    for k in 0..nnz {
        assert!(
            (dv[k] - 1.0).abs() < 1e-12,
            "transpose VJP should give dL/dvalues[{k}] = 1, got {}",
            dv[k]
        );
    }
}

// ── PCG ───────────────────────────────────────────────────────────

#[test]
fn pcg_solve_forward_matches_lu_solve() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let _nnz = values.len();

    let mut g = Graph::new("pcg_vs_lu");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);

    let x_lu = a.solve(&mut g, b);
    let x_pcg = a.pcg_solve(&mut g, b, /*max_iter=*/ 100, /*tol=*/ 1e-12);
    g.set_outputs(vec![x_lu, x_pcg]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[("b", &f64s_to_bytes(&b_data), DType::F64)]);
    let x_lu = bytes_to_f64s(&outs[0].0);
    let x_pcg = bytes_to_f64s(&outs[1].0);

    for i in 0..n {
        assert!(
            (x_lu[i] - x_pcg[i]).abs() < 1e-9,
            "x_lu[{i}] = {}, x_pcg[{i}] = {}",
            x_lu[i],
            x_pcg[i]
        );
    }
}

#[test]
fn pcg_converges_faster_than_cg_on_ill_conditioned() {
    // Construct a 4×4 SPD matrix whose diagonal magnitudes vary by
    // orders — Jacobi preconditioning should help measurably. Test
    // that PCG converges in few iterations to the right answer
    // (matching plain CG with many more iterations).
    rlx_sparse::register();
    // Diagonal entries: 1000, 100, 10, 1 (4 orders of magnitude).
    // Off-diagonal coupling: ±1 (small relative to diagonals).
    let values = vec![
        1000.0_f64, -1.0, -1.0, 100.0, -1.0, -1.0, 10.0, -1.0, -1.0, 1.0,
    ];
    let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let row_ptr = vec![0, 2, 5, 8, 10];
    let n = 4;

    let build = |max_iter: u32, use_pcg: bool| {
        let mut g = Graph::new("ill_cond");
        let v = const_f64(&mut g, &values);
        let ci = const_i32(&mut g, &col_idx);
        let rp = const_i32(&mut g, &row_ptr);
        let b = const_f64(&mut g, &[1.0, 1.0, 1.0, 1.0]);
        let a = SparseTensor::from_csr(v, ci, rp, n, n);
        let x = if use_pcg {
            a.pcg_solve(&mut g, b, max_iter, 1e-10)
        } else {
            a.cg_solve(&mut g, b, max_iter, 1e-10)
        };
        g.set_outputs(vec![x]);
        g
    };

    // Ground truth via direct LU.
    let mut g_lu = Graph::new("lu_truth");
    let v = const_f64(&mut g_lu, &values);
    let ci = const_i32(&mut g_lu, &col_idx);
    let rp = const_i32(&mut g_lu, &row_ptr);
    let b = const_f64(&mut g_lu, &[1.0, 1.0, 1.0, 1.0]);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x_lu = a.solve(&mut g_lu, b);
    g_lu.set_outputs(vec![x_lu]);
    let truth = bytes_to_f64s(&Session::new(Device::Cpu).compile(g_lu).run_typed(&[])[0].0);

    let pcg_4 = bytes_to_f64s(
        &Session::new(Device::Cpu)
            .compile(build(4, true))
            .run_typed(&[])[0]
            .0,
    );
    // PCG should be very close after just 4 iterations.
    let pcg_err: f64 = pcg_4
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        pcg_err < 1e-6,
        "PCG after 4 iters should be near truth (got err={pcg_err})"
    );

    // Plain CG with same iteration budget should be much further off
    // on this ill-conditioned matrix (κ ≈ 1000).
    let cg_4 = bytes_to_f64s(
        &Session::new(Device::Cpu)
            .compile(build(4, false))
            .run_typed(&[])[0]
            .0,
    );
    let cg_err: f64 = cg_4
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    // CG should be at least 10× worse than PCG given the conditioning.
    assert!(
        cg_err > pcg_err * 10.0,
        "PCG should outperform plain CG on ill-conditioned (cg_err={cg_err}, pcg_err={pcg_err})"
    );
}

#[test]
fn pcg_vjp_db_matches_finite_differences() {
    rlx_sparse::register();
    let (values, col_idx, row_ptr) = build_tridiag_4();
    let n = 4;
    let _nnz = values.len();

    let mut g = Graph::new("pcg_grad");
    let v = const_f64(&mut g, &values);
    let ci = const_i32(&mut g, &col_idx);
    let rp = const_i32(&mut g, &row_ptr);
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.pcg_solve(&mut g, b, 200, 1e-14);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[b]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let b_data = [1.0_f64, 2.0, 3.0, 4.0];
    let outs = compiled.run_typed(&[
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    let h = 1e-7;
    let mut fd = vec![0f64; n];
    for i in 0..n {
        let mut bp = b_data.to_vec();
        bp[i] += h;
        let mut bm = b_data.to_vec();
        bm[i] -= h;
        let lp = run_pcg_loss(&values, &col_idx, &row_ptr, &bp);
        let lm = run_pcg_loss(&values, &col_idx, &row_ptr, &bm);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db[i] - fd[i]).abs() < 1e-5,
            "pcg db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

fn run_pcg_loss(values: &[f64], col_idx: &[i32], row_ptr: &[i32], b: &[f64]) -> f64 {
    let n = b.len();
    let mut g = Graph::new("pcg_fwd");
    let v = const_f64(&mut g, values);
    let ci = const_i32(&mut g, col_idx);
    let rp = const_i32(&mut g, row_ptr);
    let bn = const_f64(&mut g, b);
    let a = SparseTensor::from_csr(v, ci, rp, n, n);
    let x = a.pcg_solve(&mut g, bn, 200, 1e-14);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    bytes_to_f64s(&compiled.run_typed(&[])[0].0)[0]
}
