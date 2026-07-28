// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-ROCm parity for the core Riemannian / SPD-manifold ops, which run on
//! the CPU host-fallback path (`rlx_rocm::spd` / `rlx_rocm::spd_host`, D2H →
//! CPU F64 reference → H2D). The SPD ops are F64 and have no ROCm kernel; on
//! ROCm they run the SAME `rlx-cpu` thunk kernels the CPU backend uses
//! (widening the f32 arena values to f64 in a one-op CPU graph), then write the
//! f32 result back to the device arena.
//!
//! NOTE: real compute runs ONLY when a ROCm/HIP device is reachable (Linux +
//! AMD driver). On a driverless host (this macOS dev box, CI without an AMD
//! GPU) every test is a graceful no-op via `rlx_rocm::is_available()`. These
//! assertions have therefore NOT been executed on this host — they are for the
//! user's Linux AMD rigs.
//!
//! References are eigendecomposition-free (diagonal inputs / manual matmuls),
//! so no `rlx` umbrella / Session dependency is needed.

use rlx_ir::{DType, Graph, Shape};
use rlx_rocm::RocmExecutable;

fn s(dims: &[usize]) -> Shape {
    // SPD ops are F64 in the IR (the CPU kernels require f64); the ROCm arena
    // stays f32 and the host fallback widens on the fly.
    Shape::new(dims, DType::F64)
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// Row-major `n×n` diagonal matrix from its diagonal, f32.
fn diagf(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = vals[i];
    }
    m
}

/// ReEig on a diagonal SPD matrix floors each diagonal entry at `eps`; the
/// off-diagonal stays zero. Reference is exact (no eigendecomposition needed).
#[test]
fn reeig_forward_floors_diagonal() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping reeig_forward_floors_diagonal");
        return;
    }
    let n = 4usize;
    let eps = 0.25f32;
    // Diagonal SPD X = diag(0.05, 0.5, 2.0, 7.0).
    let diag = [0.05f32, 0.5, 2.0, 7.0];
    let mut x = vec![0.0f32; n * n];
    for i in 0..n {
        x[i * n + i] = diag[i];
    }
    let mut g = Graph::new("reeig");
    let x_n = g.input("x", s(&[n, n]));
    let y = g.reeig(x_n, eps); // narrows the packed [Y,λ,U] to [n,n]
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();

    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        want[i * n + i] = diag[i].max(eps);
    }
    assert!(
        close(&got, &want, 1e-4),
        "ReEig ROCm vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// BiMap `Y = W·X·Wᵀ` against a manual f32 matmul reference.
#[test]
fn bimap_forward_matches_manual() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping bimap_forward_matches_manual");
        return;
    }
    let (m, n) = (2usize, 3usize);
    let w = vec![0.4f32, 0.1, 0.2, 0.3, 0.5, 0.15];
    // Symmetric SPD-ish X (diagonally dominant).
    let x = vec![4.0f32, 0.5, 0.3, 0.5, 3.0, 0.4, 0.3, 0.4, 2.5];
    let mut g = Graph::new("bimap");
    let w_n = g.input("w", s(&[m, n]));
    let x_n = g.input("x", s(&[n, n]));
    let y = g.bimap(w_n, x_n);
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe.run(&[("w", &w), ("x", &x)]).into_iter().next().unwrap();

    // want = W · X · Wᵀ
    let mut wx = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += w[i * n + k] * x[k * n + j];
            }
            wx[i * n + j] = acc;
        }
    }
    let mut want = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += wx[i * n + k] * w[j * n + k];
            }
            want[i * m + j] = acc;
        }
    }
    assert!(
        close(&got, &want, 1e-3),
        "BiMap ROCm vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// Weighted Karcher barycentre of identical points is that point (any weights).
#[test]
fn karcher_mean_weighted_of_identicals() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping karcher_mean_weighted");
        return;
    }
    let (n, batch) = (3usize, 3usize);
    let one = diagf(&[2.0, 3.0, 5.0]);
    let mut x = Vec::new();
    for _ in 0..batch {
        x.extend_from_slice(&one);
    }
    let weights = vec![0.2f32, 0.5, 0.3];
    let mut g = Graph::new("karcher_w");
    let x_n = g.input("x", s(&[batch, n, n]));
    let w_n = g.input("w", s(&[batch]));
    let m = g.spd_karcher_mean_weighted(x_n, w_n, 50, 1e-10);
    g.set_outputs(vec![m]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe
        .run(&[("x", &x), ("w", &weights)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &one, 1e-3),
        "weighted Karcher ROCm vs reference:\n got={got:?}\n want={one:?}"
    );
}

/// log_map(I, X) on a diagonal spectrum reduces to diag(log λ).
#[test]
fn log_map_identity_base_diagonal() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping log_map");
        return;
    }
    let n = 3usize;
    let ident = diagf(&[1.0, 1.0, 1.0]);
    let x = diagf(&[1.0, std::f32::consts::E, 4.0]);
    let mut g = Graph::new("log_map");
    let b_n = g.input("base", s(&[n, n]));
    let x_n = g.input("x", s(&[n, n]));
    let y = g.spd_log_map(b_n, x_n);
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("x", &x)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[0.0, 1.0, 4.0f32.ln()]);
    assert!(
        close(&got, &want, 1e-3),
        "log_map ROCm vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// exp_map(I, V) on a diagonal tangent reduces to diag(exp v).
#[test]
fn exp_map_identity_base_diagonal() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping exp_map");
        return;
    }
    let n = 3usize;
    let ident = diagf(&[1.0, 1.0, 1.0]);
    let v = diagf(&[0.0, 1.0, 4.0f32.ln()]);
    let mut g = Graph::new("exp_map");
    let b_n = g.input("base", s(&[n, n]));
    let v_n = g.input("v", s(&[n, n]));
    let y = g.spd_exp_map(b_n, v_n);
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("v", &v)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[1.0, std::f32::consts::E, 4.0]);
    assert!(
        close(&got, &want, 1e-3),
        "exp_map ROCm vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Parallel transport of a diagonal tangent between diagonal base points:
/// `Γ_{P→Q}(V)[k,k] = V[k,k]·Q[k,k]/P[k,k]`.
#[test]
fn parallel_transport_diagonal() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping parallel_transport");
        return;
    }
    let n = 3usize;
    let pk = [2.0f32, 4.0, 1.0];
    let qk = [6.0f32, 1.0, 3.0];
    let vk = [1.5f32, 2.5, 0.5];
    let mut g = Graph::new("transport");
    let p_n = g.input("p", s(&[n, n]));
    let q_n = g.input("q", s(&[n, n]));
    let v_n = g.input("v", s(&[n, n]));
    let y = g.spd_parallel_transport(p_n, q_n, v_n);
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe
        .run(&[("p", &diagf(&pk)), ("q", &diagf(&qk)), ("v", &diagf(&vk))])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[
        vk[0] * qk[0] / pk[0],
        vk[1] * qk[1] / pk[1],
        vk[2] * qk[2] / pk[2],
    ]);
    assert!(
        close(&got, &want, 1e-3),
        "parallel_transport ROCm vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Batched logm over a stack of diagonal matrices ⇒ per-slice diag(log λ).
#[test]
fn matrix_fn_batch_logm_diagonal() {
    if !rlx_rocm::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm device — skipping matrix_fn_batch");
        return;
    }
    let (n, batch) = (2usize, 2usize);
    let mut x = diagf(&[1.0, 4.0]);
    x.extend(diagf(&[std::f32::consts::E, 9.0]));
    let mut g = Graph::new("logm_batch");
    let x_n = g.input("x", s(&[batch, n, n]));
    let y = g.spd_logm_batch(x_n);
    g.set_outputs(vec![y]);
    let mut exe = RocmExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();
    let mut want = diagf(&[0.0, 4.0f32.ln()]);
    want.extend(diagf(&[1.0, 9.0f32.ln()]));
    assert!(
        close(&got, &want, 1e-3),
        "logm_batch ROCm vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Native hipSOLVER `SsyevjBatched` forward path (`Step::EighNative`): reconstruct
/// `A = U diag(λ) Uᵀ` from the GPU-native eigendecomposition and check it matches
/// the input — validates the solver output *and* the col-major→packed `[λ ∥ U]`
/// assemble kernel (single + batched). Skips when HIP or hipSOLVER is unavailable.
#[test]
fn eigh_native_forward_reconstructs() {
    if !rlx_rocm::is_available() || !rlx_rocm::eigh_native::is_available() {
        eprintln!("[rlx-rocm spd] no ROCm/hipSOLVER — skipping eigh_native");
        return;
    }
    let mk = |n: usize| -> Vec<f32> {
        let mut a = vec![0f32; n * n];
        for i in 0..n {
            for j in i..n {
                let v = if i == j {
                    (i as f32 + 1.0) * 2.0
                } else {
                    0.05 * ((i + j) as f32).cos()
                };
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    };
    let recon_ok = |lam: &[f32], u: &[f32], a: &[f32], n: usize, off_u: usize, off_l: usize| {
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    s += u[off_u + i * n + k] * lam[off_l + k] * u[off_u + j * n + k];
                }
                assert!(
                    (s - a[i * n + j]).abs() < 2e-3,
                    "native eigh recon[{i},{j}] {s} vs {}",
                    a[i * n + j]
                );
            }
        }
        assert!(
            lam[off_l..off_l + n]
                .windows(2)
                .all(|w| w[0] <= w[1] + 1e-3),
            "λ not ascending"
        );
    };

    // Single (batch=1) — routes through Op::Eigh → EighNative.
    let n = 5usize;
    let a = mk(n);
    let mut g = Graph::new("eigh_native1");
    let a_n = g.input("a", s(&[n, n]));
    let (lam, u) = g.eigh(a_n);
    g.set_outputs(vec![lam, u]);
    let outs = RocmExecutable::compile(g).run(&[("a", &a)]);
    recon_ok(&outs[0], &outs[1], &a, n, 0, 0);

    // Batched — Op::EighBatch → EighNative (one launch, whole batch).
    let (n, batch) = (6usize, 3usize);
    let mut x = Vec::new();
    for _ in 0..batch {
        x.extend(mk(n));
    }
    let mut g = Graph::new("eigh_native_batch");
    let x_n = g.input("x", s(&[batch, n, n]));
    let (lam, u) = g.eigh_batch(x_n);
    g.set_outputs(vec![lam, u]);
    let outs = RocmExecutable::compile(g).run(&[("x", &x)]);
    let a1 = mk(n);
    for b in 0..batch {
        recon_ok(&outs[0], &outs[1], &a1, n, b * n * n, b * n);
    }
}

#[test]
fn unavailable_is_graceful() {
    // Never panics regardless of host.
    let _ = rlx_rocm::is_available();
    let _ = rlx_rocm::eigh_native::is_available();
}
