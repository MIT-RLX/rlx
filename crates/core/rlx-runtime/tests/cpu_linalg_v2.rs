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

//! Linalg improvements: logdet (forward + VJP) and SVD VJP.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
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

fn build_spd_3() -> Vec<f64> {
    vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]
}

// ── logdet ────────────────────────────────────────────────────────

#[test]
fn logdet_forward_matches_lu_log_diag() {
    rlx_linalg::register();
    let n = 3;
    let a = build_spd_3();

    let mut g = Graph::new("logdet_fwd");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let ld = rlx_linalg::logdet(&mut g, a_n);
    g.set_outputs(vec![ld]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    let ld_got = bytes_to_f64s(&outs[0].0)[0];

    // Reference: compute eigenvalues, sum logs.
    // For build_spd_3 (4×3 + 0.5 + 0.5 + 0.25 + 0.25 ...), use a
    // direct numerical reference via determinant expansion.
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let want = det.ln();
    assert!(
        (ld_got - want).abs() < 1e-10,
        "logdet got {} vs ln(det)={}",
        ld_got,
        want
    );
}

#[test]
fn logdet_vjp_matches_inverse() {
    // dlogdet(A)/dA = A⁻¹ (for SPD A, also = A⁻ᵀ).
    rlx_linalg::register();
    let n = 3;
    let a = build_spd_3();

    let build = || {
        let mut g = Graph::new("logdet_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let ld = rlx_linalg::logdet(&mut g, a_n);
        g.set_outputs(vec![ld]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da = bytes_to_f64s(&outs[1].0);
    assert_eq!(da.len(), n * n);

    // Reference: A⁻¹ via direct adjugate formula for 3×3.
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let inv_det = 1.0 / det;
    let inv = vec![
        (a[4] * a[8] - a[5] * a[7]) * inv_det,
        -(a[1] * a[8] - a[2] * a[7]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        -(a[3] * a[8] - a[5] * a[6]) * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        -(a[0] * a[5] - a[2] * a[3]) * inv_det,
        (a[3] * a[7] - a[4] * a[6]) * inv_det,
        -(a[0] * a[7] - a[1] * a[6]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det,
    ];
    for i in 0..(n * n) {
        assert!(
            (da[i] - inv[i]).abs() < 1e-9,
            "logdet dA[{i}] = {} vs A⁻¹[{i}] = {}",
            da[i],
            inv[i]
        );
    }
}

#[test]
fn logdet_in_quadratic_loss_matches_fd() {
    // L = logdet(A); dA via FD with symmetric perturbation.
    rlx_linalg::register();
    let n = 3;
    let a = build_spd_3();

    let build = || {
        let mut g = Graph::new("logdet_quad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let ld = rlx_linalg::logdet(&mut g, a_n);
        g.set_outputs(vec![ld]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut fd = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut ap = a.clone();
            ap[i * n + j] += h;
            let mut am = a.clone();
            am[i * n + j] -= h;
            if i != j {
                ap[j * n + i] += h;
                am[j * n + i] -= h;
            }
            let lp = run_logdet(&ap);
            let lm = run_logdet(&am);
            fd[i * n + j] = (lp - lm) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..n {
            let want_fd = fd[i * n + j];
            let got_vjp = if i == j {
                da_vjp[i * n + j]
            } else {
                da_vjp[i * n + j] + da_vjp[j * n + i]
            };
            assert!(
                (got_vjp - want_fd).abs() < 1e-5,
                "logdet dA[{i},{j}]: VJP={} FD={}",
                got_vjp,
                want_fd
            );
        }
    }
}

fn run_logdet(a: &[f64]) -> f64 {
    rlx_linalg::register();
    let n = (a.len() as f64).sqrt() as usize;
    let mut g = Graph::new("logdet_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let ld = rlx_linalg::logdet(&mut g, a_n);
    g.set_outputs(vec![ld]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── SVD VJP ───────────────────────────────────────────────────────

#[test]
fn svd_vjp_singular_values_only_loss_matches_fd() {
    // L = sum(s). Since s_i are the singular values, dL/ds = 1.
    // dL/dU = 0, dL/dVᵀ = 0. Closed form: dL/dA = U · I · Vᵀ.
    // FD-test against the actual VJP.
    rlx_linalg::register();
    let m = 4;
    let n = 3;
    let a0: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, // perturbed last entry
        11.0, 12.0, 14.0,
    ];

    let build = || {
        let mut g = Graph::new("svd_grad");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let (_u, s, _vt) = rlx_linalg::svd(&mut g, a_n);
        let loss = g.sum(s, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    assert_eq!(da_vjp.len(), m * n);

    // FD parity (no symmetry constraint; A is general).
    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_svd_s_sum(&ap, m, n);
        let lm = run_svd_s_sum(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-5,
            "svd dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn run_svd_s_sum(a: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("svd_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let (_, s, _) = rlx_linalg::svd(&mut g, a_n);
    let loss = g.sum(s, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

#[test]
fn svd_vjp_with_u_v_gradients_matches_fd() {
    // Loss that depends on U and Vᵀ (not just s) — exercises the
    // F-mediated subspace term. L = sum(U) + 2·sum(s) + 3·sum(Vᵀ).
    rlx_linalg::register();
    let m = 4;
    let n = 3;
    let a0: Vec<f64> = vec![
        1.5, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];

    let build = || {
        let mut g = Graph::new("svd_full_grad");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let (u, s, vt) = rlx_linalg::svd(&mut g, a_n);
        let u_flat = g.reshape_(u, vec![(m * n) as i64]);
        let vt_flat = g.reshape_(vt, vec![(n * n) as i64]);
        let su = g.sum(u_flat, vec![0], false);
        let ss = g.sum(s, vec![0], false);
        let sv = g.sum(vt_flat, vec![0], false);
        // Use scalar broadcast multiply to weight s and vt contributions.
        // Just use Add of scaled tensors via constants.
        let two = const_f64(&mut g, &[2.0], &[1]);
        let three = const_f64(&mut g, &[3.0], &[1]);
        let s_w = g.binary(
            rlx_ir::op::BinaryOp::Mul,
            ss,
            two,
            Shape::new(&[1], DType::F64),
        );
        let vt_w = g.binary(
            rlx_ir::op::BinaryOp::Mul,
            sv,
            three,
            Shape::new(&[1], DType::F64),
        );
        let s1 = g.add(su, s_w);
        let loss = g.add(s1, vt_w);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_svd_full_loss(&ap, m, n);
        let lm = run_svd_full_loss(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    // SVD VJP can have looser numerical agreement than scalar ops
    // because it composes multiple matmuls. Tolerance ~1e-4.
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-4,
            "svd full dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn run_svd_full_loss(a: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("svd_full_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let (u, s, vt) = rlx_linalg::svd(&mut g, a_n);
    let u_flat = g.reshape_(u, vec![(m * n) as i64]);
    let vt_flat = g.reshape_(vt, vec![(n * n) as i64]);
    let su = g.sum(u_flat, vec![0], false);
    let ss = g.sum(s, vec![0], false);
    let sv = g.sum(vt_flat, vec![0], false);
    let two = const_f64(&mut g, &[2.0], &[1]);
    let three = const_f64(&mut g, &[3.0], &[1]);
    let s_w = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        ss,
        two,
        Shape::new(&[1], DType::F64),
    );
    let vt_w = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        sv,
        three,
        Shape::new(&[1], DType::F64),
    );
    let s1 = g.add(su, s_w);
    let loss = g.add(s1, vt_w);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── slogdet ───────────────────────────────────────────────────────

fn build_negdet_3() -> Vec<f64> {
    // Has negative determinant (row swap from SPD). det = -29.5 (approx).
    vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0, 1.0]
}

#[test]
fn slogdet_forward_matches_lu_sign_logabs() {
    rlx_linalg::register();
    let n = 3;
    let a = build_negdet_3();
    let mut g = Graph::new("slogdet_fwd");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let (sign, logabs) = rlx_linalg::slogdet(&mut g, a_n);
    g.set_outputs(vec![sign, logabs]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let s_got = bytes_to_f64s(&outs[0].0)[0];
    let l_got = bytes_to_f64s(&outs[1].0)[0];
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let want_sign = det.signum();
    let want_log = det.abs().ln();
    assert!(
        (s_got - want_sign).abs() < 1e-12,
        "sign got {s_got} want {want_sign}"
    );
    assert!(
        (l_got - want_log).abs() < 1e-10,
        "log|det| got {l_got} want {want_log}"
    );
}

#[test]
fn slogdet_singular_returns_zero_sign() {
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![
        1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // row 2 = 2·row 0 → singular
        7.0, 8.0, 9.0,
    ];
    let mut g = Graph::new("slogdet_sing");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let (sign, logabs) = rlx_linalg::slogdet(&mut g, a_n);
    g.set_outputs(vec![sign, logabs]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let s_got = bytes_to_f64s(&outs[0].0)[0];
    let l_got = bytes_to_f64s(&outs[1].0)[0];
    assert_eq!(s_got, 0.0, "sign should be 0 for singular");
    assert!(
        l_got.is_infinite() && l_got < 0.0,
        "log|det| should be -inf for singular, got {l_got}"
    );
}

#[test]
fn slogdet_vjp_matches_inverse_transpose() {
    // dlogabsdet(A)/dA = A⁻ᵀ (general, not just SPD).
    rlx_linalg::register();
    let n = 3;
    let a = build_negdet_3();
    let build = || {
        let mut g = Graph::new("slogdet_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (_sign, logabs) = rlx_linalg::slogdet(&mut g, a_n);
        g.set_outputs(vec![logabs]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da = bytes_to_f64s(&outs[1].0);
    // Reference: A⁻ᵀ via 3×3 adjugate, transposed.
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let inv_det = 1.0 / det;
    // A⁻¹[i,j] then transpose ⇒ A⁻ᵀ[i,j] = A⁻¹[j,i].
    let inv = vec![
        (a[4] * a[8] - a[5] * a[7]) * inv_det,
        -(a[1] * a[8] - a[2] * a[7]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        -(a[3] * a[8] - a[5] * a[6]) * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        -(a[0] * a[5] - a[2] * a[3]) * inv_det,
        (a[3] * a[7] - a[4] * a[6]) * inv_det,
        -(a[0] * a[7] - a[1] * a[6]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det,
    ];
    for i in 0..n {
        for j in 0..n {
            let want = inv[j * n + i]; // transpose
            let got = da[i * n + j];
            assert!(
                (got - want).abs() < 1e-9,
                "slogdet dA[{i},{j}] = {} vs A⁻ᵀ = {}",
                got,
                want
            );
        }
    }
}

// ── pinv / lstsq ──────────────────────────────────────────────────

#[test]
fn pinv_square_matches_inverse() {
    rlx_linalg::register();
    let n = 3;
    let a = build_negdet_3();
    let mut g = Graph::new("pinv_sq");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let y = rlx_linalg::pinv(&mut g, a_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let y_got = bytes_to_f64s(&outs[0].0);

    // Reference: A⁻¹ via 3×3 adjugate.
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let inv_det = 1.0 / det;
    let inv = vec![
        (a[4] * a[8] - a[5] * a[7]) * inv_det,
        -(a[1] * a[8] - a[2] * a[7]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        -(a[3] * a[8] - a[5] * a[6]) * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        -(a[0] * a[5] - a[2] * a[3]) * inv_det,
        (a[3] * a[7] - a[4] * a[6]) * inv_det,
        -(a[0] * a[7] - a[1] * a[6]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det,
    ];
    for i in 0..(n * n) {
        assert!(
            (y_got[i] - inv[i]).abs() < 1e-8,
            "pinv[{i}]={} A⁻¹={}",
            y_got[i],
            inv[i]
        );
    }
}

#[test]
fn pinv_overdetermined_satisfies_pinv_axioms() {
    // For full-rank m>n: A·Y·A = A and Y·A = I (col-rank-n case).
    rlx_linalg::register();
    let m = 4;
    let n = 2;
    let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let mut g = Graph::new("pinv_over");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let y = rlx_linalg::pinv(&mut g, a_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let y_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(y_got.len(), n * m);
    // Check Y·A = I_n.
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..m {
                acc += y_got[i * m + l] * a[l * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (acc - want).abs() < 1e-9,
                "Y·A[{i},{j}]={} want {}",
                acc,
                want
            );
        }
    }
    // Check A·Y·A = A.
    let mut ay = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..n {
                acc += a[i * n + l] * 0.0;
                let _ = acc;
            }
            // ... using y_got
            let s = 0f64;
            for l in 0..n {
                // A·Y first computes m×m ; we want A·Y·A which is m×n. Direct:
                // (A·Y·A)[i,j] = sum_p sum_q A[i,p]·Y[p,q]·A[q,j].
                // Rewrite below.
                let _ = l;
            }
            ay[i * n + j] = s;
        }
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..n {
                for q in 0..m {
                    acc += a[i * n + p] * y_got[p * m + q] * a[q * n + j];
                }
            }
            assert!(
                (acc - a[i * n + j]).abs() < 1e-9,
                "A·Y·A[{i},{j}]={} A={}",
                acc,
                a[i * n + j]
            );
        }
    }
}

#[test]
fn lstsq_square_matches_solve() {
    rlx_linalg::register();
    let n = 3;
    let a = build_negdet_3();
    let b: Vec<f64> = vec![5.0, 7.0, 2.0];
    let mut g = Graph::new("lstsq_sq");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let b_n = const_f64(&mut g, &b, &[n]);
    let x = rlx_linalg::lstsq(&mut g, a_n, b_n);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);
    // Verify A·x ≈ b.
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += a[i * n + j] * x_got[j];
        }
        assert!((acc - b[i]).abs() < 1e-8, "A·x[{i}]={} b={}", acc, b[i]);
    }
}

#[test]
fn lstsq_overdetermined_normal_equations() {
    // x = pinv(A)·b should satisfy Aᵀ·A·x = Aᵀ·b.
    rlx_linalg::register();
    let m = 5;
    let n = 2;
    let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0];
    let b: Vec<f64> = vec![1.0, 2.0, 4.0, 0.5, 3.0];
    let mut g = Graph::new("lstsq_over");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let b_n = const_f64(&mut g, &b, &[m]);
    let x = rlx_linalg::lstsq(&mut g, a_n, b_n);
    g.set_outputs(vec![x]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let x_got = bytes_to_f64s(&outs[0].0);
    // Aᵀ·A and Aᵀ·b
    let mut ata = vec![0f64; n * n];
    let mut atb = vec![0f64; n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..m {
                acc += a[l * n + i] * a[l * n + j];
            }
            ata[i * n + j] = acc;
        }
        let mut acc = 0f64;
        for l in 0..m {
            acc += a[l * n + i] * b[l];
        }
        atb[i] = acc;
    }
    // Verify Aᵀ·A·x = Aᵀ·b
    for i in 0..n {
        let mut acc = 0f64;
        for j in 0..n {
            acc += ata[i * n + j] * x_got[j];
        }
        assert!(
            (acc - atb[i]).abs() < 1e-8,
            "(AᵀA·x)[{i}]={} (Aᵀb)[{i}]={}",
            acc,
            atb[i]
        );
    }
}

#[test]
fn lstsq_vjp_b_matches_pinv_transpose() {
    // dL/db = pinv(A)ᵀ · dL/dx. With L = sum(x), dL/dx = 1, so dL/db
    // should equal column sums of pinv(A) (i.e., pinv(A)ᵀ · 1).
    rlx_linalg::register();
    let m = 3;
    let n = 3;
    let a = build_negdet_3();
    let b: Vec<f64> = vec![5.0, 7.0, 2.0];

    let build = || {
        let mut g = Graph::new("lstsq_grad_b");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let b_n = g.input("b", Shape::new(&[m], DType::F64));
        let x = rlx_linalg::lstsq(&mut g, a_n, b_n);
        let loss = g.sum(x, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n, b_n)
    };
    let (g, _a_n, b_n) = build();
    let bwd = grad_with_loss(&g, &[b_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("b", &f64s_to_bytes(&b), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    // Reference: for square A, pinv = A⁻¹, so pinv(A)ᵀ·1 = A⁻ᵀ·1.
    // Compute via solving Aᵀ·y = 1.
    let h = 1e-6;
    let mut fd = vec![0f64; m];
    for i in 0..m {
        let mut bp = b.clone();
        bp[i] += h;
        let mut bm = b.clone();
        bm[i] -= h;
        let lp = run_lstsq_loss(&a, &bp, m, n);
        let lm = run_lstsq_loss(&a, &bm, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..m {
        assert!(
            (db[i] - fd[i]).abs() < 1e-6,
            "lstsq dL/db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

#[test]
fn lstsq_vjp_a_overdetermined_matches_fd() {
    rlx_linalg::register();
    let m = 4;
    let n = 2;
    let a0: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let b: Vec<f64> = vec![1.0, 2.0, 4.0, 0.5];

    let build = || {
        let mut g = Graph::new("lstsq_grad_a");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let b_n = g.input("b", Shape::new(&[m], DType::F64));
        let x = rlx_linalg::lstsq(&mut g, a_n, b_n);
        let loss = g.sum(x, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n, b_n)
    };
    let (g, a_n, _b_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("b", &f64s_to_bytes(&b), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_lstsq_loss(&ap, &b, m, n);
        let lm = run_lstsq_loss(&am, &b, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-5,
            "lstsq dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn run_lstsq_loss(a: &[f64], b: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("lstsq_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let b_n = g.input("b", Shape::new(&[m], DType::F64));
    let x = rlx_linalg::lstsq(&mut g, a_n, b_n);
    let loss = g.sum(x, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(a), DType::F64),
        ("b", &f64s_to_bytes(b), DType::F64),
    ]);
    bytes_to_f64s(&outs[0].0)[0]
}

#[test]
fn pinv_vjp_matches_fd() {
    // Loss = sum(Y) where Y = pinv(A); compare VJP vs. FD.
    rlx_linalg::register();
    let m = 4;
    let n = 2;
    let a0: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];

    let build = || {
        let mut g = Graph::new("pinv_grad");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let y = rlx_linalg::pinv(&mut g, a_n);
        let y_flat = g.reshape_(y, vec![(n * m) as i64]);
        let loss = g.sum(y_flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_pinv_sum(&ap, m, n);
        let lm = run_pinv_sum(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-4,
            "pinv dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

// ── expm ──────────────────────────────────────────────────────────

#[test]
fn expm_zero_is_identity() {
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![0.0; n * n];
    let mut g = Graph::new("expm_zero");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let e = rlx_linalg::expm(&mut g, a_n);
    g.set_outputs(vec![e]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let e_got = bytes_to_f64s(&outs[0].0);
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (e_got[i * n + j] - want).abs() < 1e-12,
                "expm(0)[{i},{j}]={} want {}",
                e_got[i * n + j],
                want
            );
        }
    }
}

#[test]
fn expm_diagonal_matches_componentwise_exp() {
    rlx_linalg::register();
    let n = 3;
    let d = [0.5, -1.0, 2.0];
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        a[i * n + i] = d[i];
    }
    let mut g = Graph::new("expm_diag");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let e = rlx_linalg::expm(&mut g, a_n);
    g.set_outputs(vec![e]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let e_got = bytes_to_f64s(&outs[0].0);
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { d[i].exp() } else { 0.0 };
            assert!(
                (e_got[i * n + j] - want).abs() < 1e-10,
                "expm(diag)[{i},{j}]={} want {}",
                e_got[i * n + j],
                want
            );
        }
    }
}

#[test]
fn expm_skew_is_orthogonal() {
    // For skew-symmetric A (Aᵀ = -A), exp(A) is orthogonal: exp(A)·exp(A)ᵀ = I.
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![0.0, 0.4, -0.7, -0.4, 0.0, 0.5, 0.7, -0.5, 0.0];
    let mut g = Graph::new("expm_skew");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let e = rlx_linalg::expm(&mut g, a_n);
    g.set_outputs(vec![e]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let e_got = bytes_to_f64s(&outs[0].0);
    // Check E·Eᵀ = I.
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..n {
                acc += e_got[i * n + l] * e_got[j * n + l];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (acc - want).abs() < 1e-10,
                "(E·Eᵀ)[{i},{j}]={} want {}",
                acc,
                want
            );
        }
    }
}

#[test]
fn expm_large_norm_scaling_squaring() {
    // ||A||_1 > θ_13 ≈ 5.37 → exercises scaling/squaring path.
    rlx_linalg::register();
    let n = 2;
    let a: Vec<f64> = vec![3.0, 4.0, 1.0, 2.0]; // ||·||_1 = 6
    let mut g = Graph::new("expm_big");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let e = rlx_linalg::expm(&mut g, a_n);
    g.set_outputs(vec![e]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let e_got = bytes_to_f64s(&outs[0].0);
    // Reference via series: too slow; instead check via spectral
    // decomposition: A = [[3,4],[1,2]]; eigvals λ_± = (5 ± √17)/2.
    // exp(A) = V·diag(exp(λ))·V⁻¹. Compute analytically.
    let tr: f64 = 5.0;
    let det: f64 = 3.0 * 2.0 - 4.0 * 1.0;
    let disc = tr * tr - 4.0 * det;
    let sd = disc.sqrt();
    let l1 = (tr + sd) / 2.0;
    let l2 = (tr - sd) / 2.0;
    // A = α·I + β·M where via Cayley-Hamilton: exp(A) = c0·I + c1·A.
    // Actually use Sylvester formula:
    //   exp(A) = (exp(l1)·(A - l2·I) - exp(l2)·(A - l1·I)) / (l1 - l2)
    let mut want = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let aij = a[i * n + j];
            let id = if i == j { 1.0 } else { 0.0 };
            let term1 = l1.exp() * (aij - l2 * id);
            let term2 = l2.exp() * (aij - l1 * id);
            want[i * n + j] = (term1 - term2) / (l1 - l2);
        }
    }
    for i in 0..(n * n) {
        assert!(
            (e_got[i] - want[i]).abs() < 1e-9,
            "expm[{i}]={} want {}",
            e_got[i],
            want[i]
        );
    }
}

#[test]
fn expm_vjp_matches_fd() {
    rlx_linalg::register();
    let n = 3;
    let a0: Vec<f64> = vec![0.1, 0.2, -0.3, 0.0, 0.4, 0.1, 0.2, -0.1, 0.3];

    let build = || {
        let mut g = Graph::new("expm_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let e = rlx_linalg::expm(&mut g, a_n);
        let e_flat = g.reshape_(e, vec![(n * n) as i64]);
        let loss = g.sum(e_flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };
    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut fd = vec![0f64; n * n];
    for i in 0..(n * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_expm_sum(&ap, n);
        let lm = run_expm_sum(&am, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(n * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-5,
            "expm dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn run_expm_sum(a: &[f64], n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("expm_fwd_sum");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let e = rlx_linalg::expm(&mut g, a_n);
    let e_flat = g.reshape_(e, vec![(n * n) as i64]);
    let loss = g.sum(e_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

fn run_pinv_sum(a: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("pinv_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let y = rlx_linalg::pinv(&mut g, a_n);
    let y_flat = g.reshape_(y, vec![(n * m) as i64]);
    let loss = g.sum(y_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}
