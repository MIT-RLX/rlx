// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end tests for the core Riemannian / SPD-manifold ops
//! (`Op::BiMap` / `Op::ReEig` / `Op::LogEig` / `Op::SpdBatchNorm` /
//! `Op::SpdKarcherMean`, plus the arbitrary-base maps `Op::SpdLogMap` /
//! `Op::SpdExpMap` / `Op::SpdParallelTransport`, the weighted barycentre
//! `Op::SpdKarcherMeanWeighted`, and batched `Op::SpdMatrixFnBatch`) through
//! the full `Session::compile` + execute stack on `Device::Cpu`.
//!
//! Forward parity uses trivial (diagonal / manual-matmul) references so
//! the checks are eigendecomposition-free. Gradients are verified by
//! central finite differences taken *through the same Session-run
//! forward graph* — proving the autodiff wiring + kernel execution end
//! to end. (The raw Daleckii–Krein kernels are additionally FD-checked
//! in `rlx-cpu`'s `spd::tests`.)

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
    g.add_node(
        Op::Constant {
            data: f64s_to_bytes(xs),
        },
        vec![],
        Shape::new(shape, DType::F64),
    )
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
fn diag(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        m[i * n + i] = vals[i];
    }
    m
}
/// Deterministic symmetric matrix (reproducible FD direction).
fn sym(n: usize, seed: f64) -> Vec<f64> {
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = ((i as f64 * 3.0 + j as f64 * 1.7 + seed).sin()) * 0.5;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}
/// Deterministic SPD matrix `M·M + (n+1)·I`.
fn spd(n: usize, seed: f64) -> Vec<f64> {
    let m = sym(n, seed);
    let mut a = matmul(&m, &m, n, n, n);
    for i in 0..n {
        a[i * n + i] += (n + 1) as f64;
    }
    a
}

// ── Forward parity ───────────────────────────────────────────────

#[test]
fn reeig_forward_floors_diagonal() {
    let n = 4;
    let x = diag(&[0.05, 0.5, 2.0, 7.0]);
    let eps = 0.25f32;
    let mut g = Graph::new("reeig_fwd");
    let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
    let y = g.reeig(x_n, eps);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &f64s_to_bytes(&x), DType::F64)]);
    let y = bytes_to_f64s(&outs[0].0);
    let want = diag(&[0.25, 0.5, 2.0, 7.0]); // 0.05 floored to eps
    for i in 0..n * n {
        assert!(
            (y[i] - want[i]).abs() < 1e-9,
            "reeig[{i}] {} vs {}",
            y[i],
            want[i]
        );
    }
}

#[test]
fn logeig_forward_diagonal() {
    let n = 3;
    let x = diag(&[1.0, std::f64::consts::E, 4.0]);
    let mut g = Graph::new("logeig_fwd");
    let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
    let y = g.logeig(x_n, 1e-12);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &f64s_to_bytes(&x), DType::F64)]);
    let y = bytes_to_f64s(&outs[0].0);
    let want = diag(&[0.0, 1.0, 4.0_f64.ln()]);
    for i in 0..n * n {
        assert!(
            (y[i] - want[i]).abs() < 1e-9,
            "logeig[{i}] {} vs {}",
            y[i],
            want[i]
        );
    }
}

#[test]
fn bimap_forward_matches_manual() {
    let (m, n) = (2, 3);
    let w = vec![1.0, 0.5, -0.25, 2.0, -1.0, 0.75];
    let x = spd(n, 0.3);
    let mut g = Graph::new("bimap_fwd");
    let w_n = g.input("w", Shape::new(&[m, n], DType::F64));
    let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
    let y = g.bimap(w_n, x_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("w", &f64s_to_bytes(&w), DType::F64),
        ("x", &f64s_to_bytes(&x), DType::F64),
    ]);
    let y = bytes_to_f64s(&outs[0].0);
    let wx = matmul(&w, &x, m, n, n);
    let mut want = vec![0f64; m * m]; // W·X·Wᵀ
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..n {
                s += wx[i * n + k] * w[j * n + k];
            }
            want[i * m + j] = s;
        }
    }
    for i in 0..m * m {
        assert!(
            (y[i] - want[i]).abs() < 1e-9,
            "bimap[{i}] {} vs {}",
            y[i],
            want[i]
        );
    }
}

#[test]
fn karcher_mean_of_identicals() {
    let n = 3;
    let batch = 3;
    let one = spd(n, 0.7);
    let mut x = vec![0f64; batch * n * n];
    for bi in 0..batch {
        x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&one);
    }
    let mut g = Graph::new("karcher_fwd");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let m = g.spd_karcher_mean(x_n, 50, 1e-12);
    g.set_outputs(vec![m]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &f64s_to_bytes(&x), DType::F64)]);
    let m = bytes_to_f64s(&outs[0].0);
    for i in 0..n * n {
        assert!(
            (m[i] - one[i]).abs() < 1e-6,
            "karcher[{i}] {} vs {}",
            m[i],
            one[i]
        );
    }
}

#[test]
fn spd_batch_norm_forward_diagonal() {
    // Diagonal X_i, mean, G → Y_i[k,k] = (g_k / m_k) · x_i[k,k].
    let n = 3;
    let batch = 2;
    let mean = diag(&[2.0, 4.0, 1.0]);
    let gmat = diag(&[3.0, 1.0, 5.0]);
    let x0 = diag(&[1.5, 2.5, 0.5]);
    let x1 = diag(&[4.0, 1.0, 3.0]);
    let mut x = vec![0f64; batch * n * n];
    x[..n * n].copy_from_slice(&x0);
    x[n * n..].copy_from_slice(&x1);
    let mut g = Graph::new("spdbn_fwd");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let mean_n = const_f64(&mut g, &mean, &[n, n]);
    let g_n = g.input("g", Shape::new(&[n, n], DType::F64));
    let y = g.spd_batch_norm_transport(x_n, mean_n, g_n, 1e-12);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("g", &f64s_to_bytes(&gmat), DType::F64),
    ]);
    let y = bytes_to_f64s(&outs[0].0);
    let gk = [3.0, 1.0, 5.0];
    let mk = [2.0, 4.0, 1.0];
    for (bi, xi) in [x0, x1].iter().enumerate() {
        for k in 0..n {
            let want = gk[k] / mk[k] * xi[k * n + k];
            let got = y[bi * n * n + k * n + k];
            assert!((got - want).abs() < 1e-9, "spdbn[{bi},{k}] {got} vs {want}");
        }
    }
}

#[test]
fn spd_karcher_mean_weighted_of_identicals() {
    // The weighted barycentre of identical points is that point, for any
    // weights — exercises the [X[B,n,n], w[B]] two-input plumbing.
    let n = 3;
    let batch = 3;
    let one = spd(n, 0.7);
    let mut x = vec![0f64; batch * n * n];
    for bi in 0..batch {
        x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&one);
    }
    let weights = vec![0.2f64, 0.5, 0.3];
    let mut g = Graph::new("karcher_w_fwd");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let w_n = g.input("w", Shape::new(&[batch], DType::F64));
    let m = g.spd_karcher_mean_weighted(x_n, w_n, 50, 1e-12);
    g.set_outputs(vec![m]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("w", &f64s_to_bytes(&weights), DType::F64),
    ]);
    let m = bytes_to_f64s(&outs[0].0);
    for i in 0..n * n {
        assert!(
            (m[i] - one[i]).abs() < 1e-6,
            "karcher_w[{i}] {} vs {}",
            m[i],
            one[i]
        );
    }
}

#[test]
fn spd_log_exp_map_identity_base_diagonal() {
    // At base = I the AIRM maps reduce to logm / expm, elementwise on a
    // diagonal spectrum — an eigendecomposition-free reference.
    let n = 3;
    let ident = diag(&[1.0, 1.0, 1.0]);
    let x = diag(&[1.0, std::f64::consts::E, 4.0]);
    let want_log = diag(&[0.0, 1.0, 4.0_f64.ln()]);

    // log_map(I, x) == diag(log λ).
    let mut g = Graph::new("log_map_fwd");
    let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
    let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
    let y = g.spd_log_map(b_n, x_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("base", &f64s_to_bytes(&ident), DType::F64),
        ("x", &f64s_to_bytes(&x), DType::F64),
    ]);
    let y = bytes_to_f64s(&outs[0].0);
    for i in 0..n * n {
        assert!(
            (y[i] - want_log[i]).abs() < 1e-9,
            "log_map[{i}] {} vs {}",
            y[i],
            want_log[i]
        );
    }

    // exp_map(I, log_map(I, x)) roundtrips back to x.
    let mut g = Graph::new("exp_map_fwd");
    let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
    let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
    let z = g.spd_exp_map(b_n, v_n);
    g.set_outputs(vec![z]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("base", &f64s_to_bytes(&ident), DType::F64),
        ("v", &f64s_to_bytes(&want_log), DType::F64),
    ]);
    let z = bytes_to_f64s(&outs[0].0);
    for i in 0..n * n {
        assert!(
            (z[i] - x[i]).abs() < 1e-9,
            "exp_map[{i}] {} vs {}",
            z[i],
            x[i]
        );
    }
}

#[test]
fn spd_parallel_transport_diagonal() {
    // Diagonal P, Q, V make E = (Q P^{-1})^{1/2} diagonal, so
    // Γ_{P→Q}(V)[k,k] = V[k,k] · Q[k,k] / P[k,k] and off-diagonals vanish —
    // an exact eigendecomposition-free check of the real transport formula.
    let n = 3;
    let pk = [2.0, 4.0, 1.0];
    let qk = [6.0, 1.0, 3.0];
    let vk = [1.5, 2.5, 0.5];
    let p = diag(&pk);
    let q = diag(&qk);
    let v = diag(&vk);
    let mut g = Graph::new("transport_fwd");
    let p_n = g.input("p", Shape::new(&[n, n], DType::F64));
    let q_n = g.input("q", Shape::new(&[n, n], DType::F64));
    let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
    let y = g.spd_parallel_transport(p_n, q_n, v_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("p", &f64s_to_bytes(&p), DType::F64),
        ("q", &f64s_to_bytes(&q), DType::F64),
        ("v", &f64s_to_bytes(&v), DType::F64),
    ]);
    let y = bytes_to_f64s(&outs[0].0);
    for r in 0..n {
        for col in 0..n {
            let want = if r == col { vk[r] * qk[r] / pk[r] } else { 0.0 };
            let got = y[r * n + col];
            assert!(
                (got - want).abs() < 1e-9,
                "transport[{r},{col}] {got} vs {want}"
            );
        }
    }
}

#[test]
fn spd_matrix_fn_batch_logm_diagonal() {
    // Batched logm over a stack of diagonal SPD matrices ⇒ per-slice
    // diag(log λ). Verifies the [B,n,n] → [B,n,n] batched plumbing.
    let n = 2;
    let batch = 2;
    let s0 = diag(&[1.0, 4.0]);
    let s1 = diag(&[std::f64::consts::E, 9.0]);
    let mut x = vec![0f64; batch * n * n];
    x[..n * n].copy_from_slice(&s0);
    x[n * n..].copy_from_slice(&s1);
    let mut g = Graph::new("matfn_batch_fwd");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let y = g.spd_logm_batch(x_n);
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &f64s_to_bytes(&x), DType::F64)]);
    let y = bytes_to_f64s(&outs[0].0);
    let want0 = diag(&[0.0, 4.0_f64.ln()]);
    let want1 = diag(&[1.0, 9.0_f64.ln()]);
    for i in 0..n * n {
        assert!(
            (y[i] - want0[i]).abs() < 1e-9,
            "matfn0[{i}] {} vs {}",
            y[i],
            want0[i]
        );
        assert!(
            (y[n * n + i] - want1[i]).abs() < 1e-9,
            "matfn1[{i}] {} vs {}",
            y[n * n + i],
            want1[i]
        );
    }
}

// ── Gradient parity (autodiff through Session vs central FD) ──────

/// Run a scalar-output graph and return the loss value.
fn run_scalar(g: Graph, inputs: &[(&str, &[f64])]) -> f64 {
    let typed: Vec<(&str, Vec<u8>, DType)> = inputs
        .iter()
        .map(|(name, data)| (*name, f64s_to_bytes(data), DType::F64))
        .collect();
    let refs: Vec<(&str, &[u8], DType)> = typed
        .iter()
        .map(|(n, b, d)| (*n, b.as_slice(), *d))
        .collect();
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&refs);
    bytes_to_f64s(&outs[0].0)[0]
}

fn frob(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn reeig_grad_matches_fd() {
    let n = 4;
    let x0 = spd(n, 0.9);
    let eps = 3.0f32; // inside the spectrum → exercises the rectifier kink
    let build_loss = |g: &mut Graph| -> NodeId {
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.reeig(x_n, eps);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        x_n
    };
    let mut g = Graph::new("reeig_grad");
    let x_n = build_loss(&mut g);
    let bwd = grad_with_loss(&g, &[x_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let grad = bytes_to_f64s(&outs[1].0);

    // Directional FD along a symmetric direction.
    let v = sym(n, 4.0);
    let scalar = |x: &[f64]| {
        let mut g = Graph::new("reeig_loss");
        build_loss(&mut g);
        run_scalar(g, &[("x", x)])
    };
    let h = 1e-6;
    let xp: Vec<f64> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
    let xm: Vec<f64> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
    let fd = (scalar(&xp) - scalar(&xm)) / (2.0 * h);
    let analytic = frob(&grad, &v);
    assert!(
        (analytic - fd).abs() / (fd.abs() + 1e-6) < 1e-4,
        "reeig grad: analytic {analytic} vs fd {fd}"
    );
}

#[test]
fn logeig_grad_matches_fd() {
    let n = 4;
    let x0 = spd(n, 1.3);
    let build_loss = |g: &mut Graph| -> NodeId {
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.logeig(x_n, 1e-12);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        x_n
    };
    let mut g = Graph::new("logeig_grad");
    let x_n = build_loss(&mut g);
    let bwd = grad_with_loss(&g, &[x_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let grad = bytes_to_f64s(&outs[1].0);
    let v = sym(n, 5.0);
    let scalar = |x: &[f64]| {
        let mut g = Graph::new("logeig_loss");
        build_loss(&mut g);
        run_scalar(g, &[("x", x)])
    };
    let h = 1e-6;
    let xp: Vec<f64> = x0.iter().zip(&v).map(|(a, b)| a + h * b).collect();
    let xm: Vec<f64> = x0.iter().zip(&v).map(|(a, b)| a - h * b).collect();
    let fd = (scalar(&xp) - scalar(&xm)) / (2.0 * h);
    let analytic = frob(&grad, &v);
    assert!(
        (analytic - fd).abs() / (fd.abs() + 1e-6) < 1e-4,
        "logeig grad: analytic {analytic} vs fd {fd}"
    );
}

#[test]
fn bimap_grad_matches_fd() {
    let (m, n) = (2, 3);
    let w0 = vec![1.0, 0.5, -0.25, 2.0, -1.0, 0.75];
    let x0 = spd(n, 0.4);
    let build_loss = |g: &mut Graph| -> (NodeId, NodeId) {
        let w_n = g.input("w", Shape::new(&[m, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.bimap(w_n, x_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (w_n, x_n)
    };
    let mut g = Graph::new("bimap_grad");
    let (w_n, x_n) = build_loss(&mut g);
    let bwd = grad_with_loss(&g, &[w_n, x_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("w", &f64s_to_bytes(&w0), DType::F64),
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dw = bytes_to_f64s(&outs[1].0);
    let dx = bytes_to_f64s(&outs[2].0);

    let scalar = |w: &[f64], x: &[f64]| {
        let mut g = Graph::new("bimap_loss");
        build_loss(&mut g);
        run_scalar(g, &[("w", w), ("x", x)])
    };
    let h = 1e-6;
    // dW check (arbitrary direction — W is unconstrained).
    let vw = sym(m.max(n), 6.0)[..m * n].to_vec();
    let wp: Vec<f64> = w0.iter().zip(&vw).map(|(a, b)| a + h * b).collect();
    let wm: Vec<f64> = w0.iter().zip(&vw).map(|(a, b)| a - h * b).collect();
    let fd_w = (scalar(&wp, &x0) - scalar(&wm, &x0)) / (2.0 * h);
    assert!(
        (frob(&dw, &vw) - fd_w).abs() / (fd_w.abs() + 1e-6) < 1e-4,
        "bimap dW: {} vs {fd_w}",
        frob(&dw, &vw)
    );
    // dX check (symmetric direction — X is SPD).
    let vx = sym(n, 7.0);
    let xp: Vec<f64> = x0.iter().zip(&vx).map(|(a, b)| a + h * b).collect();
    let xm: Vec<f64> = x0.iter().zip(&vx).map(|(a, b)| a - h * b).collect();
    let fd_x = (scalar(&w0, &xp) - scalar(&w0, &xm)) / (2.0 * h);
    assert!(
        (frob(&dx, &vx) - fd_x).abs() / (fd_x.abs() + 1e-6) < 1e-4,
        "bimap dX: {} vs {fd_x}",
        frob(&dx, &vx)
    );
}

#[test]
fn spd_batch_norm_grad_matches_fd() {
    let n = 3;
    let batch = 2;
    let mean = spd(n, 5.0); // detached constant
    let g0 = spd(n, 6.0);
    let mut x0 = vec![0f64; batch * n * n];
    for bi in 0..batch {
        let s = spd(n, bi as f64 + 0.3);
        x0[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
    }
    let mean_c = mean.clone();
    let build_loss = move |g: &mut Graph| -> (NodeId, NodeId) {
        let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
        let mean_n = const_f64(g, &mean_c, &[n, n]);
        let g_n = g.input("g", Shape::new(&[n, n], DType::F64));
        let y = g.spd_batch_norm_transport(x_n, mean_n, g_n, 1e-12);
        let loss = g.sum(y, vec![0, 1, 2], false);
        g.set_outputs(vec![loss]);
        (x_n, g_n)
    };
    let mut g = Graph::new("spdbn_grad");
    let (x_n, g_n) = build_loss(&mut g);
    let bwd = grad_with_loss(&g, &[x_n, g_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("g", &f64s_to_bytes(&g0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let dx = bytes_to_f64s(&outs[1].0);
    let dg = bytes_to_f64s(&outs[2].0);

    let scalar = |x: &[f64], gm: &[f64]| {
        let mut g = Graph::new("spdbn_loss");
        build_loss(&mut g);
        run_scalar(g, &[("x", x), ("g", gm)])
    };
    let h = 1e-6;
    // dX (symmetric per-slice direction).
    let mut vx = vec![0f64; batch * n * n];
    for bi in 0..batch {
        let s = sym(n, bi as f64 + 8.0);
        vx[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
    }
    let xp: Vec<f64> = x0.iter().zip(&vx).map(|(a, b)| a + h * b).collect();
    let xm: Vec<f64> = x0.iter().zip(&vx).map(|(a, b)| a - h * b).collect();
    let fd_x = (scalar(&xp, &g0) - scalar(&xm, &g0)) / (2.0 * h);
    assert!(
        (frob(&dx, &vx) - fd_x).abs() / (fd_x.abs() + 1e-6) < 1e-4,
        "spdbn dX: {} vs {fd_x}",
        frob(&dx, &vx)
    );
    // dG (symmetric direction).
    let vg = sym(n, 9.0);
    let gp: Vec<f64> = g0.iter().zip(&vg).map(|(a, b)| a + h * b).collect();
    let gm: Vec<f64> = g0.iter().zip(&vg).map(|(a, b)| a - h * b).collect();
    let fd_g = (scalar(&x0, &gp) - scalar(&x0, &gm)) / (2.0 * h);
    assert!(
        (frob(&dg, &vg) - fd_g).abs() / (fd_g.abs() + 1e-6) < 1e-4,
        "spdbn dG: {} vs {fd_g}",
        frob(&dg, &vg)
    );
}

/// Central-FD directional check of one input's gradient, driving the scalar
/// loss through the same Session-run forward graph.
fn fd_grad_check(
    grad: &[f64],
    dir: &[f64],
    at: &[f64],
    scalar_at: impl Fn(&[f64]) -> f64,
    name: &str,
) {
    let h = 1e-6;
    let ap: Vec<f64> = at.iter().zip(dir).map(|(a, b)| a + h * b).collect();
    let am: Vec<f64> = at.iter().zip(dir).map(|(a, b)| a - h * b).collect();
    let fd = (scalar_at(&ap) - scalar_at(&am)) / (2.0 * h);
    let an = frob(grad, dir);
    assert!(
        (an - fd).abs() / (fd.abs() + 1e-6) < 1e-4,
        "{name}: analytic {an} vs fd {fd}"
    );
}

#[test]
fn spd_log_map_grad_matches_fd() {
    let n = 3;
    let base0 = spd(n, 0.7);
    let x0 = spd(n, 2.1);
    let build = |g: &mut Graph| -> (NodeId, NodeId) {
        let base_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.spd_log_map(base_n, x_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (base_n, x_n)
    };
    let mut g = Graph::new("log_map_grad");
    let (base_n, x_n) = build(&mut g);
    let bwd = grad_with_loss(&g, &[base_n, x_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("base", &f64s_to_bytes(&base0), DType::F64),
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_base = bytes_to_f64s(&outs[1].0);
    let d_x = bytes_to_f64s(&outs[2].0);
    let scalar = |base: &[f64], x: &[f64]| {
        let mut g = Graph::new("log_map_loss");
        build(&mut g);
        run_scalar(g, &[("base", base), ("x", x)])
    };
    fd_grad_check(&d_x, &sym(n, 4.0), &x0, |x| scalar(&base0, x), "log_map dX");
    fd_grad_check(
        &d_base,
        &sym(n, 5.0),
        &base0,
        |b| scalar(b, &x0),
        "log_map dBase",
    );
}

#[test]
fn spd_exp_map_grad_matches_fd() {
    let n = 3;
    let base0 = spd(n, 0.8);
    let v0 = sym(n, 1.4); // small symmetric tangent
    let build = |g: &mut Graph| -> (NodeId, NodeId) {
        let base_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
        let y = g.spd_exp_map(base_n, v_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (base_n, v_n)
    };
    let mut g = Graph::new("exp_map_grad");
    let (base_n, v_n) = build(&mut g);
    let bwd = grad_with_loss(&g, &[base_n, v_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("base", &f64s_to_bytes(&base0), DType::F64),
        ("v", &f64s_to_bytes(&v0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_base = bytes_to_f64s(&outs[1].0);
    let d_v = bytes_to_f64s(&outs[2].0);
    let scalar = |base: &[f64], v: &[f64]| {
        let mut g = Graph::new("exp_map_loss");
        build(&mut g);
        run_scalar(g, &[("base", base), ("v", v)])
    };
    fd_grad_check(&d_v, &sym(n, 4.1), &v0, |v| scalar(&base0, v), "exp_map dV");
    fd_grad_check(
        &d_base,
        &sym(n, 5.1),
        &base0,
        |b| scalar(b, &v0),
        "exp_map dBase",
    );
}

#[test]
fn spd_parallel_transport_grad_matches_fd() {
    let n = 3;
    let from0 = spd(n, 0.6);
    let to0 = spd(n, 1.9);
    let v0 = sym(n, 2.5);
    let build = |g: &mut Graph| -> (NodeId, NodeId, NodeId) {
        let f_n = g.input("from", Shape::new(&[n, n], DType::F64));
        let t_n = g.input("to", Shape::new(&[n, n], DType::F64));
        let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
        let y = g.spd_parallel_transport(f_n, t_n, v_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (f_n, t_n, v_n)
    };
    let mut g = Graph::new("transport_grad");
    let (f_n, t_n, v_n) = build(&mut g);
    let bwd = grad_with_loss(&g, &[f_n, t_n, v_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("from", &f64s_to_bytes(&from0), DType::F64),
        ("to", &f64s_to_bytes(&to0), DType::F64),
        ("v", &f64s_to_bytes(&v0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_from = bytes_to_f64s(&outs[1].0);
    let d_to = bytes_to_f64s(&outs[2].0);
    let d_v = bytes_to_f64s(&outs[3].0);
    let scalar = |from: &[f64], to: &[f64], v: &[f64]| {
        let mut g = Graph::new("transport_loss");
        build(&mut g);
        run_scalar(g, &[("from", from), ("to", to), ("v", v)])
    };
    fd_grad_check(
        &d_from,
        &sym(n, 4.0),
        &from0,
        |f| scalar(f, &to0, &v0),
        "transport dFrom",
    );
    fd_grad_check(
        &d_to,
        &sym(n, 5.0),
        &to0,
        |t| scalar(&from0, t, &v0),
        "transport dTo",
    );
    fd_grad_check(
        &d_v,
        &sym(n, 6.0),
        &v0,
        |v| scalar(&from0, &to0, v),
        "transport dV",
    );
}

#[test]
fn spd_matrix_fn_batch_grad_matches_fd() {
    let n = 3;
    let batch = 3;
    let mut x0 = vec![0f64; batch * n * n];
    for bi in 0..batch {
        x0[bi * n * n..(bi + 1) * n * n].copy_from_slice(&spd(n, bi as f64 * 0.5 + 0.2));
    }
    let build = |g: &mut Graph| -> NodeId {
        let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
        let y = g.spd_logm_batch(x_n);
        let loss = g.sum(y, vec![0, 1, 2], false);
        g.set_outputs(vec![loss]);
        x_n
    };
    let mut g = Graph::new("matfn_batch_grad");
    let x_n = build(&mut g);
    let bwd = grad_with_loss(&g, &[x_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let d_x = bytes_to_f64s(&outs[1].0);
    let scalar = |x: &[f64]| {
        let mut g = Graph::new("matfn_batch_loss");
        build(&mut g);
        run_scalar(g, &[("x", x)])
    };
    let mut dir = vec![0f64; batch * n * n];
    for bi in 0..batch {
        dir[bi * n * n..(bi + 1) * n * n].copy_from_slice(&sym(n, bi as f64 + 7.0));
    }
    fd_grad_check(&d_x, &dir, &x0, scalar, "matrix_fn_batch dX");
}

/// Well-separated SPD (distinct diagonal + tiny coupling).
fn eig_mat(n: usize) -> Vec<f64> {
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = if i == j {
                (i as f64 + 1.0) * 2.0
            } else {
                0.05 * ((i + j) as f64).cos()
            };
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

#[test]
fn eigh_forward_reconstructs() {
    let n = 4;
    let a = eig_mat(n);
    let mut g = Graph::new("eigh_fwd");
    let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
    let (lam, u) = g.eigh(a_n);
    g.set_outputs(vec![lam, u]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("a", &f64s_to_bytes(&a), DType::F64)]);
    let lam = bytes_to_f64s(&outs[0].0);
    let u = bytes_to_f64s(&outs[1].0);
    // A = U diag(λ) Uᵀ, λ ascending.
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += u[i * n + k] * lam[k] * u[j * n + k];
            }
            assert!((s - a[i * n + j]).abs() < 1e-9, "eigh recon[{i},{j}]");
        }
    }
    assert!(
        lam.windows(2).all(|w| w[0] <= w[1] + 1e-12),
        "λ not ascending"
    );
}

#[test]
fn eigh_grad_sum_eigenvalues_is_identity() {
    // Σλ = trace(A) ⇒ ∂/∂A = I. Exact end-to-end check of the Eigh VJP through
    // Session (autodiff → EighBackward → kernel).
    let n = 4;
    let a = eig_mat(n);
    let build = |g: &mut Graph| -> NodeId {
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (lam, _u) = g.eigh(a_n);
        let loss = g.sum(lam, vec![0], false);
        g.set_outputs(vec![loss]);
        a_n
    };
    let mut g = Graph::new("eigh_grad");
    let a_n = build(&mut g);
    let bwd = grad_with_loss(&g, &[a_n]);
    let mut c = Session::new(Device::Cpu).compile(bwd);
    let outs = c.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let grad = bytes_to_f64s(&outs[1].0);
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (grad[i * n + j] - want).abs() < 1e-6,
                "eigh grad[{i},{j}]={} want {want}",
                grad[i * n + j]
            );
        }
    }
}

#[test]
fn eigh_batch_forward_reconstructs() {
    let n = 3;
    let batch = 2;
    let mut x = vec![0f64; batch * n * n];
    for bi in 0..batch {
        let m = eig_mat(n);
        x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&m);
    }
    let mut g = Graph::new("eigh_batch_fwd");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let (lam, u) = g.eigh_batch(x_n);
    g.set_outputs(vec![lam, u]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &f64s_to_bytes(&x), DType::F64)]);
    let lam = bytes_to_f64s(&outs[0].0); // [batch, n]
    let u = bytes_to_f64s(&outs[1].0); // [batch, n, n]
    for bi in 0..batch {
        let a = &x[bi * n * n..(bi + 1) * n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += u[bi * n * n + i * n + k] * lam[bi * n + k] * u[bi * n * n + j * n + k];
                }
                assert!(
                    (s - a[i * n + j]).abs() < 1e-9,
                    "eigh_batch recon[{bi},{i},{j}]"
                );
            }
        }
    }
}

#[test]
fn spd_batch_norm_training_builder_normalizes_to_g() {
    // The composite training builder = batch Karcher mean → transport.
    // Karcher-mean equivariance under congruence ⇒ the Fréchet mean of
    // the normalized batch equals the learnable bias G. End-to-end
    // through Session (karcher + transport + a second karcher on Y).
    let n = 3;
    let batch = 4;
    let gmat = spd(n, 3.0);
    let mut x = vec![0f64; batch * n * n];
    for bi in 0..batch {
        let s = spd(n, bi as f64 * 0.9 + 0.2);
        x[bi * n * n..(bi + 1) * n * n].copy_from_slice(&s);
    }
    let mut g = Graph::new("spdbn_train");
    let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
    let g_n = g.input("g", Shape::new(&[n, n], DType::F64));
    let (y, _mean) = g.spd_batch_norm(x_n, g_n, 1e-12, 50, 1e-10);
    let ymean = g.spd_karcher_mean(y, 50, 1e-10);
    g.set_outputs(vec![ymean]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("g", &f64s_to_bytes(&gmat), DType::F64),
    ]);
    let ym = bytes_to_f64s(&outs[0].0);
    let err = ym
        .iter()
        .zip(&gmat)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-3,
        "SPD-BN training builder didn't normalize to G: err {err}"
    );
}

#[test]
fn spd_ops_rejected_by_backend_without_support() {
    // A backend that doesn't claim the SPD ops must reject them at
    // legalize time (→ users pin SPDNet graphs to Device::Cpu, like FFT).
    let n = 3;
    let mut g = Graph::new("reeig_pin");
    let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
    let y = g.reeig(x_n, 1e-5);
    g.set_outputs(vec![y]);
    // Narrow/Reshape supported, ReEig NOT → the SPD op must be flagged.
    let res = rlx_opt::legalize_for_backend(&g, &[rlx_ir::OpKind::Narrow, rlx_ir::OpKind::Reshape]);
    assert!(
        res.is_err(),
        "expected legalize to reject unsupported ReEig"
    );
}
