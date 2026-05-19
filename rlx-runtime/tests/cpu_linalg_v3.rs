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

//! Linalg utilities: trace, diag_extract, diag_set, kron, polar.

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

// ── diag_extract / diag_set ───────────────────────────────────────

#[test]
fn diag_extract_pulls_diagonal() {
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut g = Graph::new("diag_x");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let d = rlx_linalg::diag_extract(&mut g, a_n);
    g.set_outputs(vec![d]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let d_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(d_got, vec![1.0, 5.0, 9.0]);
}

#[test]
fn diag_set_builds_diagonal_matrix() {
    rlx_linalg::register();
    let v: Vec<f64> = vec![2.0, 3.0, 5.0];
    let n = v.len();
    let mut g = Graph::new("diag_s");
    let v_n = const_f64(&mut g, &v, &[n]);
    let m = rlx_linalg::diag_set(&mut g, v_n);
    g.set_outputs(vec![m]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let m_got = bytes_to_f64s(&outs[0].0);
    let expected = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
    for i in 0..(n * n) {
        assert!(
            (m_got[i] - expected[i]).abs() < 1e-12,
            "diag_set[{i}]={} want {}",
            m_got[i],
            expected[i]
        );
    }
}

#[test]
fn diag_extract_vjp_is_diag_set() {
    // Loss = sum(diag(A)) = trace(A). dL/dA = I.
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];

    let build = || {
        let mut g = Graph::new("diag_x_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let d = rlx_linalg::diag_extract(&mut g, a_n);
        let loss = g.sum(d, vec![0], false);
        g.set_outputs(vec![loss]);
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
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (da[i * n + j] - want).abs() < 1e-12,
                "trace dA[{i},{j}]={} want {}",
                da[i * n + j],
                want
            );
        }
    }
}

#[test]
fn diag_set_vjp_is_diag_extract() {
    // Loss = sum(M) where M = diag_set(v). M[i,j] = v[i] if i==j; loss = sum_i v[i].
    // dL/dv[i] = 1.
    rlx_linalg::register();
    let n = 3;
    let v0: Vec<f64> = vec![2.0, 3.0, 5.0];

    let build = || {
        let mut g = Graph::new("diag_s_grad");
        let v_n = g.input("v", Shape::new(&[n], DType::F64));
        let m = rlx_linalg::diag_set(&mut g, v_n);
        let m_flat = g.reshape_(m, vec![(n * n) as i64]);
        let loss = g.sum(m_flat, vec![0], false);
        g.set_outputs(vec![loss]);
        (g, v_n)
    };
    let (g, v_n) = build();
    let bwd = grad_with_loss(&g, &[v_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("v", &f64s_to_bytes(&v0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dv = bytes_to_f64s(&outs[1].0);
    for i in 0..n {
        assert!((dv[i] - 1.0).abs() < 1e-12, "dv[{i}]={}", dv[i]);
    }
}

// ── trace ─────────────────────────────────────────────────────────

#[test]
fn trace_returns_sum_of_diagonal() {
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut g = Graph::new("trace_fwd");
    let a_n = const_f64(&mut g, &a, &[n, n]);
    let t = rlx_linalg::trace(&mut g, a_n);
    g.set_outputs(vec![t]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let t_got = bytes_to_f64s(&outs[0].0)[0];
    assert!((t_got - 15.0).abs() < 1e-12, "trace={t_got} want 15");
}

#[test]
fn trace_vjp_is_identity() {
    rlx_linalg::register();
    let n = 3;
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let build = || {
        let mut g = Graph::new("trace_grad");
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let t = rlx_linalg::trace(&mut g, a_n);
        g.set_outputs(vec![t]);
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
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (da[i * n + j] - want).abs() < 1e-12,
                "trace dA[{i},{j}]={}",
                da[i * n + j]
            );
        }
    }
}

// ── kron ──────────────────────────────────────────────────────────

#[test]
fn kron_matches_naive_definition() {
    rlx_linalg::register();
    let m = 2;
    let n = 3;
    let p = 2;
    let q = 2;
    let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f64> = vec![1.0, 0.0, 0.0, -1.0];
    let mut g = Graph::new("kron_fwd");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let b_n = const_f64(&mut g, &b, &[p, q]);
    let k = rlx_linalg::kron(&mut g, a_n, b_n);
    g.set_outputs(vec![k]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let k_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(k_got.len(), m * p * n * q);

    // Reference via naive definition.
    let mut k_ref = vec![0f64; (m * p) * (n * q)];
    for i in 0..m {
        for j in 0..n {
            for r in 0..p {
                for s in 0..q {
                    let row = i * p + r;
                    let col = j * q + s;
                    k_ref[row * (n * q) + col] = a[i * n + j] * b[r * q + s];
                }
            }
        }
    }
    for i in 0..(m * p * n * q) {
        assert!(
            (k_got[i] - k_ref[i]).abs() < 1e-12,
            "kron[{i}]={} ref={}",
            k_got[i],
            k_ref[i]
        );
    }
}

#[test]
fn kron_vjp_via_autodiff_matches_fd() {
    // Loss = sum(kron(A, B)). VJP comes free from Reshape + Mul autodiff.
    rlx_linalg::register();
    let m = 2;
    let n = 2;
    let p = 2;
    let q = 2;
    let a0: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f64> = vec![0.5, 1.0, -1.0, 2.0];

    let build = || {
        let mut g = Graph::new("kron_grad");
        let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
        let b_n = const_f64(&mut g, &b, &[p, q]);
        let k = rlx_linalg::kron(&mut g, a_n, b_n);
        let k_flat = g.reshape_(k, vec![((m * p) * (n * q)) as i64]);
        let loss = g.sum(k_flat, vec![0], false);
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
    let da = bytes_to_f64s(&outs[1].0);

    let h = 1e-6;
    let mut fd = vec![0f64; m * n];
    for i in 0..(m * n) {
        let mut ap = a0.clone();
        ap[i] += h;
        let mut am = a0.clone();
        am[i] -= h;
        let lp = run_kron_sum_loss(&ap, &b, m, n, p, q);
        let lm = run_kron_sum_loss(&am, &b, m, n, p, q);
        fd[i] = (lp - lm) / (2.0 * h);
    }
    for i in 0..(m * n) {
        assert!(
            (da[i] - fd[i]).abs() < 1e-6,
            "kron dA[{i}]: VJP={} FD={}",
            da[i],
            fd[i]
        );
    }
}

fn run_kron_sum_loss(a: &[f64], b: &[f64], m: usize, n: usize, p: usize, q: usize) -> f64 {
    rlx_linalg::register();
    let mut g = Graph::new("kron_loss");
    let a_n = g.input("a", Shape::new(&[m, n], DType::F64));
    let b_n = const_f64(&mut g, b, &[p, q]);
    let k = rlx_linalg::kron(&mut g, a_n, b_n);
    let k_flat = g.reshape_(k, vec![((m * p) * (n * q)) as i64]);
    let loss = g.sum(k_flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(a), DType::F64)]);
    bytes_to_f64s(&outs[0].0)[0]
}

// ── polar ─────────────────────────────────────────────────────────

#[test]
fn polar_decomposition_orthogonal_and_spd() {
    rlx_linalg::register();
    let m = 4;
    let n = 3;
    let a: Vec<f64> = vec![
        1.0, 0.5, -0.25, 0.0, 2.0, 1.0, -1.5, 0.25, 3.0, 2.0, 1.0, 0.5,
    ];

    let mut g = Graph::new("polar_fwd");
    let a_n = const_f64(&mut g, &a, &[m, n]);
    let (u, h) = rlx_linalg::polar(&mut g, a_n);
    g.set_outputs(vec![u, h]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[]);
    let u_got = bytes_to_f64s(&outs[0].0);
    let h_got = bytes_to_f64s(&outs[1].0);
    assert_eq!(u_got.len(), m * n);
    assert_eq!(h_got.len(), n * n);

    // U must be orthogonal: Uᵀ·U = I_n.
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..m {
                acc += u_got[l * n + i] * u_got[l * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (acc - want).abs() < 1e-9,
                "(UᵀU)[{i},{j}]={} want {}",
                acc,
                want
            );
        }
    }
    // H must be symmetric.
    for i in 0..n {
        for j in 0..n {
            assert!(
                (h_got[i * n + j] - h_got[j * n + i]).abs() < 1e-9,
                "H not symmetric at [{i},{j}]"
            );
        }
    }
    // U·H must equal A.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for l in 0..n {
                acc += u_got[i * n + l] * h_got[l * n + j];
            }
            assert!(
                (acc - a[i * n + j]).abs() < 1e-9,
                "(U·H)[{i},{j}]={} A={}",
                acc,
                a[i * n + j]
            );
        }
    }
}
