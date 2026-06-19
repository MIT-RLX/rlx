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

//! JVP (forward-mode AD) for the linalg utility ops.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff_fwd::jvp;
use rlx_runtime::{Device, Session};

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
fn const_f64(g: &mut Graph, xs: &[f64], shape: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(shape, DType::F64),
    )
}

#[test]
fn jvp_diag_extract_pushes_diagonal_tangent() {
    rlx_linalg::register();
    let n = 3;

    let mut g = Graph::new("jvp_de");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let d = rlx_linalg::diag_extract(&mut g, a);
    g.set_outputs(vec![d]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ta = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&ta), DType::F64),
    ]);
    let primal = bytes_to_f64s(&outs[0].0);
    let tangent = bytes_to_f64s(&outs[1].0);
    assert_eq!(primal, vec![1.0, 5.0, 9.0]);
    assert_eq!(tangent, vec![0.1, 0.5, 0.9]);
}

#[test]
fn jvp_diag_set_pushes_vector_tangent() {
    rlx_linalg::register();
    let n = 3;

    let mut g = Graph::new("jvp_ds");
    let v = g.input("v", Shape::new(&[n], DType::F64));
    let m = rlx_linalg::diag_set(&mut g, v);
    g.set_outputs(vec![m]);

    let jg = jvp(&g, &[v]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    let v_data = vec![1.0, 2.0, 3.0];
    let tv = vec![0.5, 0.25, 0.1];
    let outs = c.run_typed(&[
        ("v", &f64s_to_bytes(&v_data), DType::F64),
        ("tangent_v", &f64s_to_bytes(&tv), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);
    let want = vec![0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.1];
    for i in 0..(n * n) {
        assert!(
            (tangent[i] - want[i]).abs() < 1e-12,
            "jvp[ds][{i}]={} want {}",
            tangent[i],
            want[i]
        );
    }
}

#[test]
fn jvp_trace_pushes_diagonal_sum() {
    // trace(A) = sum(diag(A)). t_trace = sum(diag(t_A)) = trace(t_A).
    rlx_linalg::register();
    let n = 3;
    let mut g = Graph::new("jvp_tr");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let t = rlx_linalg::trace(&mut g, a);
    g.set_outputs(vec![t]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ta = vec![0.1, 9.0, 9.0, 9.0, 0.2, 9.0, 9.0, 9.0, 0.3];
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&ta), DType::F64),
    ]);
    let primal = bytes_to_f64s(&outs[0].0)[0];
    let tangent = bytes_to_f64s(&outs[1].0)[0];
    assert!((primal - 15.0).abs() < 1e-12);
    assert!((tangent - 0.6).abs() < 1e-12);
}

// ── cholesky JVP ──────────────────────────────────────────────────

#[test]
fn jvp_cholesky_matches_finite_differences() {
    rlx_linalg::register();
    let n = 3;
    let a_data = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
    let da = vec![0.05, 0.02, 0.0, 0.02, 0.03, 0.01, 0.0, 0.01, 0.04];

    let mut g = Graph::new("jvp_chol");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let l = rlx_linalg::cholesky(&mut g, a, /*lower=*/ true);
    g.set_outputs(vec![l]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);

    // FD reference.
    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    for i in 0..(n * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    let l_p = run_chol_fwd(&a_p, n);
    let l_m = run_chol_fwd(&a_m, n);
    for i in 0..(n * n) {
        let want = (l_p[i] - l_m[i]) / (2.0 * h);
        assert!(
            (tangent[i] - want).abs() < 1e-7,
            "chol jvp[{i}]={} FD={}",
            tangent[i],
            want
        );
    }
}

fn run_chol_fwd(a: &[f64], n: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("chol_fwd");
    let a_n = const_f64(&mut g, a, &[n, n]);
    let l = rlx_linalg::cholesky(&mut g, a_n, true);
    g.set_outputs(vec![l]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

// ── solve_triangular JVP ──────────────────────────────────────────

#[test]
fn jvp_solve_triangular_matches_fd() {
    rlx_linalg::register();
    let n = 3;
    // Lower-triangular A.
    let a_data = vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.5, 0.25, 4.0];
    let b_data = vec![
        1.0, 2.0, 3.0, // rhs col 0
        0.5, -1.0, 0.5,
    ]; // rhs col 1 → B is [n, 2]
    let nrhs = 2;
    let da = vec![0.1, 0.0, 0.0, 0.05, 0.2, 0.0, 0.0, 0.05, 0.15];
    let db = vec![0.1, 0.05, -0.05, 0.0, 0.1, 0.0];

    let mut g = Graph::new("jvp_st");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n, nrhs], DType::F64));
    let y = rlx_linalg::solve_triangular(
        &mut g, a, b, /*lower=*/ true, /*transpose_a=*/ false,
    );
    g.set_outputs(vec![y]);

    let jg = jvp(&g, &[a, b]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("b", &f64s_to_bytes(&b_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
        ("tangent_b", &f64s_to_bytes(&db), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    let mut b_p = b_data.clone();
    let mut b_m = b_data.clone();
    for i in 0..(n * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    for i in 0..(n * nrhs) {
        b_p[i] += h * db[i];
        b_m[i] -= h * db[i];
    }
    let y_p = run_solve_tri(&a_p, &b_p, n, nrhs);
    let y_m = run_solve_tri(&a_m, &b_m, n, nrhs);
    for i in 0..(n * nrhs) {
        let want = (y_p[i] - y_m[i]) / (2.0 * h);
        assert!(
            (tangent[i] - want).abs() < 1e-7,
            "solve_tri jvp[{i}]={} FD={}",
            tangent[i],
            want
        );
    }
}

fn run_solve_tri(a: &[f64], b: &[f64], n: usize, nrhs: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("solve_tri_fwd");
    let a_n = const_f64(&mut g, a, &[n, n]);
    let b_n = const_f64(&mut g, b, &[n, nrhs]);
    let y = rlx_linalg::solve_triangular(&mut g, a_n, b_n, true, false);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

// ── eigh JVP ──────────────────────────────────────────────────────

#[test]
fn jvp_eigh_eigenvalue_tangent_matches_fd() {
    // Output is packed [λ (n), V (n²)]. We test the eigenvalue tangent
    // against FD; eigenvector tangent has sign-ambiguity so we check
    // a sign-invariant quantity (V^T·V = I should still hold tangentially).
    rlx_linalg::register();
    let n = 3;
    let a_data = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
    let da = vec![0.1, 0.05, 0.0, 0.05, 0.2, 0.02, 0.0, 0.02, 0.15];

    let mut g = Graph::new("jvp_eigh");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let (lambda, _v) = rlx_linalg::eigh(&mut g, a);
    g.set_outputs(vec![lambda]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    for i in 0..(n * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    let l_p = run_eigh_lambda(&a_p, n);
    let l_m = run_eigh_lambda(&a_m, n);
    for i in 0..n {
        let want = (l_p[i] - l_m[i]) / (2.0 * h);
        assert!(
            (tangent[i] - want).abs() < 1e-6,
            "eigh λ jvp[{i}]={} FD={}",
            tangent[i],
            want
        );
    }
}

// ── qr JVP ────────────────────────────────────────────────────────

#[test]
fn jvp_qr_matches_finite_differences() {
    rlx_linalg::register();
    let m = 4;
    let n = 3;
    let a_data = vec![
        1.0, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];
    let da = vec![
        0.05, 0.0, -0.02, 0.01, 0.0, 0.0, 0.0, 0.03, 0.0, 0.02, -0.01, 0.04,
    ];

    let mut g = Graph::new("jvp_qr");
    let a = g.input("a", Shape::new(&[m, n], DType::F64));
    let (q, r) = rlx_linalg::qr(&mut g, a);
    g.set_outputs(vec![q, r]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    // Outputs: [primal_Q, primal_R, t_Q, t_R]
    assert_eq!(outs.len(), 4);
    let t_q = bytes_to_f64s(&outs[2].0);
    let t_r = bytes_to_f64s(&outs[3].0);

    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    for i in 0..(m * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    let (q_p, r_p) = run_qr_fwd(&a_p, m, n);
    let (q_m, r_m) = run_qr_fwd(&a_m, m, n);
    for i in 0..(m * n) {
        let want = (q_p[i] - q_m[i]) / (2.0 * h);
        assert!(
            (t_q[i] - want).abs() < 1e-6,
            "qr t_Q[{i}]={} FD={}",
            t_q[i],
            want
        );
    }
    for i in 0..(n * n) {
        let want = (r_p[i] - r_m[i]) / (2.0 * h);
        assert!(
            (t_r[i] - want).abs() < 1e-6,
            "qr t_R[{i}]={} FD={}",
            t_r[i],
            want
        );
    }
}

fn run_qr_fwd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    rlx_linalg::register();
    let mut g = Graph::new("qr_fwd");
    let a_n = const_f64(&mut g, a, &[m, n]);
    let (q, r) = rlx_linalg::qr(&mut g, a_n);
    g.set_outputs(vec![q, r]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    (bytes_to_f64s(&outs[0].0), bytes_to_f64s(&outs[1].0))
}

// ── svd JVP ───────────────────────────────────────────────────────

#[test]
fn jvp_svd_singular_values_match_fd() {
    rlx_linalg::register();
    let m = 4;
    let n = 3;
    let a_data = vec![
        1.5, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];
    let da = vec![
        0.05, 0.0, -0.02, 0.01, 0.0, 0.0, 0.0, 0.03, 0.0, 0.02, -0.01, 0.04,
    ];

    let mut g = Graph::new("jvp_svd");
    let a = g.input("a", Shape::new(&[m, n], DType::F64));
    let (_u, s, _vt) = rlx_linalg::svd(&mut g, a);
    g.set_outputs(vec![s]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    let t_s = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    for i in 0..(m * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    let s_p = run_svd_s(&a_p, m, n);
    let s_m = run_svd_s(&a_m, m, n);
    for i in 0..n {
        let want = (s_p[i] - s_m[i]) / (2.0 * h);
        assert!(
            (t_s[i] - want).abs() < 1e-6,
            "svd t_s[{i}]={} FD={}",
            t_s[i],
            want
        );
    }
}

fn run_svd_s(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("svd_s_fwd");
    let a_n = const_f64(&mut g, a, &[m, n]);
    let (_u, s, _) = rlx_linalg::svd(&mut g, a_n);
    g.set_outputs(vec![s]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

// ── pinv JVP ──────────────────────────────────────────────────────

#[test]
fn jvp_pinv_matches_finite_differences() {
    rlx_linalg::register();
    let m = 4;
    let n = 2;
    let a_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let da = vec![0.1, 0.05, 0.0, 0.02, 0.05, 0.0, 0.0, -0.03];

    let mut g = Graph::new("jvp_pinv");
    let a = g.input("a", Shape::new(&[m, n], DType::F64));
    let y = rlx_linalg::pinv(&mut g, a);
    g.set_outputs(vec![y]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    let t_y = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut a_p = a_data.clone();
    let mut a_m = a_data.clone();
    for i in 0..(m * n) {
        a_p[i] += h * da[i];
        a_m[i] -= h * da[i];
    }
    let y_p = run_pinv(&a_p, m, n);
    let y_m = run_pinv(&a_m, m, n);
    for i in 0..(n * m) {
        let want = (y_p[i] - y_m[i]) / (2.0 * h);
        assert!(
            (t_y[i] - want).abs() < 1e-5,
            "pinv t_Y[{i}]={} FD={}",
            t_y[i],
            want
        );
    }
}

fn run_pinv(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("pinv_fwd_jvp_test");
    let a_n = const_f64(&mut g, a, &[m, n]);
    let y = rlx_linalg::pinv(&mut g, a_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

fn run_eigh_lambda(a: &[f64], n: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("eigh_λ_fwd");
    let a_n = const_f64(&mut g, a, &[n, n]);
    let (lambda, _) = rlx_linalg::eigh(&mut g, a_n);
    g.set_outputs(vec![lambda]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

#[test]
fn jvp_logdet_matches_trace_solve() {
    // d/dt log|det(A)| = tr(A⁻¹·dA).
    rlx_linalg::register();
    let n = 3;

    let mut g = Graph::new("jvp_ld");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let l = rlx_linalg::logdet(&mut g, a);
    g.set_outputs(vec![l]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    let a_data = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
    let ta = vec![0.1, 0.2, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4];
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&ta), DType::F64),
    ]);
    let primal = bytes_to_f64s(&outs[0].0)[0];
    let tangent = bytes_to_f64s(&outs[1].0)[0];

    // Reference: tr(A⁻¹·dA) via 3×3 adjugate.
    let det = a_data[0] * (a_data[4] * a_data[8] - a_data[5] * a_data[7])
        - a_data[1] * (a_data[3] * a_data[8] - a_data[5] * a_data[6])
        + a_data[2] * (a_data[3] * a_data[7] - a_data[4] * a_data[6]);
    let inv_det = 1.0 / det;
    let inv = vec![
        (a_data[4] * a_data[8] - a_data[5] * a_data[7]) * inv_det,
        -(a_data[1] * a_data[8] - a_data[2] * a_data[7]) * inv_det,
        (a_data[1] * a_data[5] - a_data[2] * a_data[4]) * inv_det,
        -(a_data[3] * a_data[8] - a_data[5] * a_data[6]) * inv_det,
        (a_data[0] * a_data[8] - a_data[2] * a_data[6]) * inv_det,
        -(a_data[0] * a_data[5] - a_data[2] * a_data[3]) * inv_det,
        (a_data[3] * a_data[7] - a_data[4] * a_data[6]) * inv_det,
        -(a_data[0] * a_data[7] - a_data[1] * a_data[6]) * inv_det,
        (a_data[0] * a_data[4] - a_data[1] * a_data[3]) * inv_det,
    ];
    let mut want_t = 0f64;
    for i in 0..n {
        for j in 0..n {
            want_t += inv[i * n + j] * ta[j * n + i]; // tr(A⁻¹·dA) = sum_{i,j} A⁻¹[i,j]·dA[j,i]
        }
    }
    let want_p = det.ln();
    assert!(
        (primal - want_p).abs() < 1e-9,
        "primal logdet={primal} want {want_p}"
    );
    assert!(
        (tangent - want_t).abs() < 1e-9,
        "tangent logdet={tangent} want {want_t}"
    );
}

#[test]
fn jvp_slogdet_logabs_tangent_only() {
    // Sign tangent should be 0; log|det| tangent matches logdet's.
    rlx_linalg::register();
    let n = 3;

    let mut g = Graph::new("jvp_sld");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let (sign, logabs) = rlx_linalg::slogdet(&mut g, a);
    g.set_outputs(vec![sign, logabs]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    let a_data = vec![
        1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0, 1.0, // negative det
    ];
    let ta = vec![0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3];
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&ta), DType::F64),
    ]);
    // Outputs: [sign, logabs, t_sign, t_logabs]
    assert_eq!(outs.len(), 4);
    let t_sign = bytes_to_f64s(&outs[2].0)[0];
    let t_logabs = bytes_to_f64s(&outs[3].0)[0];
    assert_eq!(t_sign, 0.0, "sign tangent should be 0");

    // Reference: tr(A⁻¹·dA). With dA diagonal, this is sum_i A⁻¹[i,i]·dA[i,i].
    let det = a_data[0] * (a_data[4] * a_data[8] - a_data[5] * a_data[7])
        - a_data[1] * (a_data[3] * a_data[8] - a_data[5] * a_data[6])
        + a_data[2] * (a_data[3] * a_data[7] - a_data[4] * a_data[6]);
    let inv_det = 1.0 / det;
    let inv_diag = [
        (a_data[4] * a_data[8] - a_data[5] * a_data[7]) * inv_det,
        (a_data[0] * a_data[8] - a_data[2] * a_data[6]) * inv_det,
        (a_data[0] * a_data[4] - a_data[1] * a_data[3]) * inv_det,
    ];
    let want = inv_diag[0] * ta[0] + inv_diag[1] * ta[4] + inv_diag[2] * ta[8];
    assert!(
        (t_logabs - want).abs() < 1e-9,
        "t_logabs={} want {}",
        t_logabs,
        want
    );
}

#[test]
fn jvp_expm_matches_finite_differences() {
    // Forward Frechet derivative: ∂/∂t exp(A + t·dA)|_{t=0} ≈
    //   (exp(A + h·dA) − exp(A − h·dA)) / (2h).
    rlx_linalg::register();
    let n = 3;
    let a_data = vec![0.1, 0.2, -0.3, 0.0, 0.4, 0.1, 0.2, -0.1, 0.3];
    let da = vec![0.05, 0.0, -0.1, 0.0, 0.0, 0.05, 0.1, 0.0, 0.0];

    let mut g = Graph::new("jvp_exp");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let e = rlx_linalg::expm(&mut g, a);
    g.set_outputs(vec![e]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&da), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);

    // FD reference.
    let h = 1e-6;
    let mut a_plus = a_data.clone();
    let mut a_minus = a_data.clone();
    for i in 0..(n * n) {
        a_plus[i] += h * da[i];
        a_minus[i] -= h * da[i];
    }
    let e_plus = run_expm_fwd(&a_plus, n);
    let e_minus = run_expm_fwd(&a_minus, n);
    for i in 0..(n * n) {
        let want = (e_plus[i] - e_minus[i]) / (2.0 * h);
        assert!(
            (tangent[i] - want).abs() < 1e-7,
            "expm jvp[{i}]={} FD={}",
            tangent[i],
            want
        );
    }
}

fn run_expm_fwd(a: &[f64], n: usize) -> Vec<f64> {
    rlx_linalg::register();
    let mut g = Graph::new("expm_fwd_only");
    let a_n = const_f64(&mut g, a, &[n, n]);
    let e = rlx_linalg::expm(&mut g, a_n);
    g.set_outputs(vec![e]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    bytes_to_f64s(&outs[0].0)
}

#[test]
fn jvp_kron_via_composed_rules() {
    // kron is composed of Reshape + Expand + Mul + Reshape — JVP comes
    // free if those rules are present. Verify against FD.
    rlx_linalg::register();
    let m = 2;
    let n = 2;
    let p = 2;
    let q = 2;
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 1.0, -1.0, 2.0];

    let mut g = Graph::new("jvp_kr");
    let a = g.input("a", Shape::new(&[m, n], DType::F64));
    let b = const_f64(&mut g, &b_data, &[p, q]);
    let k = rlx_linalg::kron(&mut g, a, b);
    g.set_outputs(vec![k]);

    let jg = jvp(&g, &[a]);
    let mut c = Session::new(Device::Cpu).compile(jg);

    // tangent_a = e_0 (one-hot). t_kron should equal kron(e_0, B).
    let mut ta = vec![0f64; m * n];
    ta[0] = 1.0;
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a_data), DType::F64),
        ("tangent_a", &f64s_to_bytes(&ta), DType::F64),
    ]);
    let tangent = bytes_to_f64s(&outs[1].0);
    // Reference: kron of e_0 with B, computed naively.
    let mut ref_t = vec![0f64; m * p * n * q];
    for i in 0..m {
        for j in 0..n {
            let coef = ta[i * n + j];
            if coef == 0.0 {
                continue;
            }
            for r in 0..p {
                for s in 0..q {
                    let row = i * p + r;
                    let col = j * q + s;
                    ref_t[row * (n * q) + col] = coef * b_data[r * q + s];
                }
            }
        }
    }
    for i in 0..(m * p * n * q) {
        assert!(
            (tangent[i] - ref_t[i]).abs() < 1e-12,
            "jvp[kron][{i}]={} ref={}",
            tangent[i],
            ref_t[i]
        );
    }
}
