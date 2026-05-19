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

//! Linalg VJP tests — finite-difference parity for solve_triangular,
//! cholesky, eigh, qr. SVD VJP is deferred (degenerate-singular-value
//! handling warrants its own turn).
//!
//! Each test builds a small loss `L = f(op(A))`, runs autodiff to
//! get `dL/dA`, and compares against a central-difference reference.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
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

fn build_spd_3() -> Vec<f64> {
    // Small SPD matrix (3×3). Diagonally dominant + symmetric.
    vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]
}

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

// ── solve_triangular VJP ──────────────────────────────────────────

#[test]
fn solve_triangular_vjp_db_matches_fd() {
    rlx_linalg::register();
    let n = 3;
    // Lower-triangular A (well-conditioned).
    let a: Vec<f64> = vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, -0.5, 1.0, 4.0];
    let b = vec![1.0_f64, 2.0, 3.0];

    let build = |b_data: &[f64]| {
        let mut g = Graph::new("trsm_grad");
        let mut a_bytes = Vec::with_capacity(n * n * 8);
        for &x in &a {
            a_bytes.extend_from_slice(&x.to_le_bytes());
        }
        let a_n = g.add_node(
            Op::Constant { data: a_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let b_n = g.input("b", Shape::new(&[n], DType::F64));
        let x = rlx_linalg::solve_triangular(&mut g, a_n, b_n, true, false);
        let loss = g.sum(x, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, b_n, b_data.to_vec())
    };

    let (g, b_n, _) = build(&b);
    let bwd = grad_with_loss(&g, &[b_n]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("b", &f64s_to_bytes(&b), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let db = bytes_to_f64s(&outs[1].0);

    // FD reference.
    let h = 1e-7;
    let mut fd = vec![0f64; n];
    for i in 0..n {
        let mut bp = b.clone();
        bp[i] += h;
        let mut bm = b.clone();
        bm[i] -= h;
        fd[i] = (run_loss(&build(&bp).0, &bp) - run_loss(&build(&bm).0, &bm)) / (2.0 * h);
    }
    for i in 0..n {
        assert!(
            (db[i] - fd[i]).abs() < 1e-6,
            "trsm db[{i}]: VJP={} FD={}",
            db[i],
            fd[i]
        );
    }
}

fn run_loss(g: &Graph, b_data: &[f64]) -> f64 {
    let mut compiled = Session::new(Device::Cpu).compile(g.clone());
    let outs = compiled.run_typed(&[("b", &f64s_to_bytes(b_data), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── Cholesky VJP ──────────────────────────────────────────────────

#[test]
fn cholesky_vjp_da_matches_fd() {
    rlx_linalg::register();
    let n = 3;
    let a0 = build_spd_3();

    // Loss: sum of all entries of L (the Cholesky factor).
    let build = |a_in: &[f64]| {
        let mut g = Graph::new("chol_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let l = rlx_linalg::cholesky(&mut g, a_n, true);
        let l_flat = g.reshape_(l, vec![(n * n) as i64]);
        let loss = g.sum(l_flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n, a_in.to_vec())
    };

    let (g, a_n, _) = build(&a0);
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    assert_eq!(da_vjp.len(), n * n);

    // FD reference: perturb each entry of A, re-run forward, central-diff.
    // Cholesky requires A symmetric. To preserve that, perturb both A[i,j]
    // and A[j,i] by h/2 each (off-diagonal); for diagonal, perturb by h.
    // The expected gradient is the symmetric "free-A" gradient — i.e.
    // dL/dA where the user thinks of A as a general matrix and Cholesky
    // is implicitly defined on the symmetric part. JAX returns the
    // symmetric VJP; we match.
    let h = 1e-6;
    let mut fd = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut ap = a0.clone();
            ap[i * n + j] += h;
            // Symmetrize the perturbation if i != j so Cholesky is valid.
            if i != j {
                ap[j * n + i] += h;
            }
            let mut am = a0.clone();
            am[i * n + j] -= h;
            if i != j {
                am[j * n + i] -= h;
            }
            let lp = run_loss_a(&a0, &ap);
            let lm = run_loss_a(&a0, &am);
            // FD picks up the symmetric contribution; for off-diagonals
            // the VJP returns dL/dA[i,j] + dL/dA[j,i] = symmetric grad,
            // halved per entry. Match by summing both entries of FD.
            let two_h = 2.0 * h;
            fd[i * n + j] = (lp - lm) / two_h;
        }
    }

    // Compare. The closed-form returns 0.5(Q + Qᵀ) which is symmetric.
    // FD perturbing A[i,j] with implicit symmetrization computes
    // ∂L/∂A[i,j] + ∂L/∂A[j,i] for off-diagonal — twice the symmetric
    // VJP entry. For diag the FD matches the VJP entry directly.
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
                "chol dA[{i},{j}]: VJP_combined={} FD={}",
                got_vjp,
                want_fd
            );
        }
    }
}

fn run_loss_a(a_template: &[f64], a_input: &[f64]) -> f64 {
    let _ = a_template;
    let n = (a_input.len() as f64).sqrt() as usize;
    rlx_linalg::register();
    let mut g = Graph::new("chol_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let l = rlx_linalg::cholesky(&mut g, a_n, true);
    let l_flat = g.reshape_(l, vec![(n * n) as i64]);
    let loss = g.sum(l_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("a", &f64s_to_bytes(a_input), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── Eigh VJP ──────────────────────────────────────────────────────

#[test]
fn eigh_vjp_eigenvalue_loss_matches_fd() {
    // Loss = sum(eigenvalues). Closed-form gradient: dL/dA = I (the
    // identity matrix) since trace-of-A = sum of eigenvalues and
    // d/dA[i,j] of trace is δ_{ij}. Let's verify the VJP returns
    // approximately I (up to symmetric averaging of off-diagonals).
    rlx_linalg::register();
    let n = 3;
    let a0 = build_spd_3();

    let build = || {
        let mut g = Graph::new("eigh_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (eigvals, _eigvecs) = rlx_linalg::eigh(&mut g, a_n);
        let loss = g.sum(eigvals, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };

    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    assert_eq!(da_vjp.len(), n * n);

    // FD reference (symmetric perturbation).
    let h = 1e-6;
    let mut fd = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut ap = a0.clone();
            let mut am = a0.clone();
            ap[i * n + j] += h;
            am[i * n + j] -= h;
            if i != j {
                ap[j * n + i] += h;
                am[j * n + i] -= h;
            }
            let lp = eigh_eigenvalue_sum_loss(&ap);
            let lm = eigh_eigenvalue_sum_loss(&am);
            fd[i * n + j] = (lp - lm) / (2.0 * h);
        }
    }
    // Closed form: trace gradient is identity; symmetric VJP returns
    // I directly on diagonal, 0 on off-diag (symmetrized to 0+0).
    // FD with symmetric perturbation: diagonal gives +1 (matches);
    // off-diag gives 0 (matches symmetrized VJP which is 0 here).
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
                "eigh dA[{i},{j}]: VJP={} FD={}",
                got_vjp,
                want_fd
            );
        }
    }
}

fn eigh_eigenvalue_sum_loss(a_input: &[f64]) -> f64 {
    rlx_linalg::register();
    let n = (a_input.len() as f64).sqrt() as usize;
    let mut g = Graph::new("eigh_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let (eigvals, _) = rlx_linalg::eigh(&mut g, a_n);
    let loss = g.sum(eigvals, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("a", &f64s_to_bytes(a_input), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── QR VJP ────────────────────────────────────────────────────────

#[test]
fn qr_vjp_loss_matches_fd() {
    // Loss = sum of all entries of R. For thin QR the gradient should
    // satisfy the standard FD relationship. Use a 4×3 (m≥n) matrix.
    rlx_linalg::register();
    let m: usize = 4;
    let n: usize = 3;
    let a0: Vec<f64> = vec![
        1.0, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];

    let build = || {
        let mut g = Graph::new("qr_grad");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let (_q, r) = rlx_linalg::qr(&mut g, a_n);
        let r_flat = g.reshape_(r, vec![(n * n) as i64]);
        let loss = g.sum(r_flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, a_n)
    };

    let (g, a_n) = build();
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled.run_typed(&[
        ("a", &f64s_to_bytes(&a0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let da_vjp = bytes_to_f64s(&outs[1].0);
    assert_eq!(da_vjp.len(), m * n);

    // FD (no symmetry constraint; A is general).
    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = qr_r_sum_loss(&ap, m, n);
        let lm = qr_r_sum_loss(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-5,
            "qr dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn qr_r_sum_loss(a_input: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("qr_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let (_q, r) = rlx_linalg::qr(&mut g, a_n);
    let r_flat = g.reshape_(r, vec![(n * n) as i64]);
    let loss = g.sum(r_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[("a", &f64s_to_bytes(a_input), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

#[allow(dead_code)]
fn _ensure_matmul_use() {
    let _ = matmul(&[0.0], &[0.0], 1, 1, 1);
}

// ── Stronger eigh / qr VJP coverage ───────────────────────────────

#[test]
fn eigh_vjp_eigenvector_loss_matches_fd() {
    // Loss = sum(V). Exercises the F-matrix subspace term (off-diag
    // contribution that vanishes for eigenvalue-only losses).
    rlx_linalg::register();
    let n = 3;
    let a0 = build_spd_3();

    let build = || {
        let mut g = Graph::new("eigh_grad_v");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (_eigvals, eigvecs) = rlx_linalg::eigh(&mut g, a_n);
        let v_flat = g.reshape_(eigvecs, vec![(n * n) as i64]);
        let loss = g.sum(v_flat, vec![0], false);
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
    for i in 0..n {
        for j in 0..n {
            let mut ap = a0.clone();
            ap[i * n + j] += h;
            let mut am = a0.clone();
            am[i * n + j] -= h;
            if i != j {
                ap[j * n + i] += h;
                am[j * n + i] -= h;
            }
            let lp = eigh_v_sum_loss(&ap);
            let lm = eigh_v_sum_loss(&am);
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
                (got_vjp - want_fd).abs() < 1e-4,
                "eigh-V dA[{i},{j}]: VJP={} FD={}",
                got_vjp,
                want_fd
            );
        }
    }
}

#[test]
fn eigh_vjp_mixed_loss_matches_fd() {
    // Loss = sum(λ) + 2·sum(V). Exercises both diag and F-mediated terms.
    rlx_linalg::register();
    let n = 3;
    let a0 = build_spd_3();

    let build = || {
        let mut g = Graph::new("eigh_grad_mix");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (eigvals, eigvecs) = rlx_linalg::eigh(&mut g, a_n);
        let v_flat = g.reshape_(eigvecs, vec![(n * n) as i64]);
        let sl = g.sum(eigvals, vec![0], false);
        let sv = g.sum(v_flat, vec![0], false);
        let two = const_f64(&mut g, &[2.0], &[1]);
        let svw = g.binary(
            rlx_ir::op::BinaryOp::Mul,
            sv,
            two,
            Shape::new(&[1], DType::F64),
        );
        let loss = g.add(sl, svw);
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
    for i in 0..n {
        for j in 0..n {
            let mut ap = a0.clone();
            ap[i * n + j] += h;
            let mut am = a0.clone();
            am[i * n + j] -= h;
            if i != j {
                ap[j * n + i] += h;
                am[j * n + i] -= h;
            }
            let lp = eigh_mixed_loss(&ap);
            let lm = eigh_mixed_loss(&am);
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
                (got_vjp - want_fd).abs() < 1e-4,
                "eigh-mix dA[{i},{j}]: VJP={} FD={}",
                got_vjp,
                want_fd
            );
        }
    }
}

fn eigh_v_sum_loss(a: &[f64]) -> f64 {
    rlx_linalg::register();
    let n = (a.len() as f64).sqrt() as usize;
    let mut g = Graph::new("eigh_v_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let (_, eigvecs) = rlx_linalg::eigh(&mut g, a_n);
    let v_flat = g.reshape_(eigvecs, vec![(n * n) as i64]);
    let loss = g.sum(v_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

fn eigh_mixed_loss(a: &[f64]) -> f64 {
    rlx_linalg::register();
    let n = (a.len() as f64).sqrt() as usize;
    let mut g = Graph::new("eigh_mix_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let (eigvals, eigvecs) = rlx_linalg::eigh(&mut g, a_n);
    let v_flat = g.reshape_(eigvecs, vec![(n * n) as i64]);
    let sl = g.sum(eigvals, vec![0], false);
    let sv = g.sum(v_flat, vec![0], false);
    let two = const_f64(&mut g, &[2.0], &[1]);
    let svw = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        sv,
        two,
        Shape::new(&[1], DType::F64),
    );
    let loss = g.add(sl, svw);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

#[test]
fn qr_vjp_q_loss_matches_fd() {
    // Loss = sum(Q). Exercises the copytril path of qr_backward
    // (which only fires when dL/dQ is non-zero).
    rlx_linalg::register();
    let m: usize = 4;
    let n: usize = 3;
    let a0: Vec<f64> = vec![
        1.0, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];

    let build = || {
        let mut g = Graph::new("qr_grad_q");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let (q, _r) = rlx_linalg::qr(&mut g, a_n);
        let q_flat = g.reshape_(q, vec![(m * n) as i64]);
        let loss = g.sum(q_flat, vec![0], false);
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
        let lp = qr_q_sum_loss(&ap, m, n);
        let lm = qr_q_sum_loss(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-4,
            "qr-Q dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

#[test]
fn qr_vjp_mixed_loss_matches_fd() {
    // Loss = sum(Q) + 3·sum(R). Exercises both backward paths together.
    rlx_linalg::register();
    let m: usize = 4;
    let n: usize = 3;
    let a0: Vec<f64> = vec![
        1.0, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];

    let build = || {
        let mut g = Graph::new("qr_grad_mix");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let (q, r) = rlx_linalg::qr(&mut g, a_n);
        let q_flat = g.reshape_(q, vec![(m * n) as i64]);
        let r_flat = g.reshape_(r, vec![(n * n) as i64]);
        let sq = g.sum(q_flat, vec![0], false);
        let sr = g.sum(r_flat, vec![0], false);
        let three = const_f64(&mut g, &[3.0], &[1]);
        let srw = g.binary(
            rlx_ir::op::BinaryOp::Mul,
            sr,
            three,
            Shape::new(&[1], DType::F64),
        );
        let loss = g.add(sq, srw);
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
        let lp = qr_mixed_loss(&ap, m, n);
        let lm = qr_mixed_loss(&am, m, n);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da_vjp[i] - fd[i]).abs() < 1e-4,
            "qr-mix dA[{i}]: VJP={} FD={}",
            da_vjp[i],
            fd[i]
        );
    }
}

fn qr_q_sum_loss(a: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("qr_q_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let (q, _) = rlx_linalg::qr(&mut g, a_n);
    let q_flat = g.reshape_(q, vec![(m * n) as i64]);
    let loss = g.sum(q_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

fn qr_mixed_loss(a: &[f64], m: usize, n: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("qr_mix_fwd");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let (q, r) = rlx_linalg::qr(&mut g, a_n);
    let q_flat = g.reshape_(q, vec![(m * n) as i64]);
    let r_flat = g.reshape_(r, vec![(n * n) as i64]);
    let sq = g.sum(q_flat, vec![0], false);
    let sr = g.sum(r_flat, vec![0], false);
    let three = const_f64(&mut g, &[3.0], &[1]);
    let srw = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        sr,
        three,
        Shape::new(&[1], DType::F64),
    );
    let loss = g.add(sq, srw);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

fn const_f64(g: &mut Graph, xs: &[f64], shape: &[usize]) -> rlx_ir::NodeId {
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
