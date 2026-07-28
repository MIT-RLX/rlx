// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Stiefel-manifold optimizer tests. Two guarantees:
//   1. Feasibility — after many steps, `W·Wᵀ` stays `= I_m` (to 1e-6).
//   2. Optimality — minimizing `L(W) = −tr(Wᵀ A W)` for a fixed SPD `A`
//      decreases the loss ~monotonically and converges to the sum of
//      the `m` largest eigenvalues of `A` (the analytic optimum on
//      St(m,n)).
// All data is generated from fixed deterministic patterns (no RNG).

use rlx_optim::{Optimizer, Stiefel};

const M: usize = 3; // orthonormal rows
const N: usize = 6; // ambient dimension (m ≤ n → wide / row-orthonormal)

// ── Small dense helpers (row-major) ─────────────────────────────────

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// `A · Bᵀ` with `A: p×q`, `B: r×q` → `p×r`.
fn matmul_bt(a: &[f32], b: &[f32], p: usize, q: usize, r: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; p * r];
    for i in 0..p {
        for j in 0..r {
            let mut s = 0.0f32;
            for t in 0..q {
                s += a[i * q + t] * b[j * q + t];
            }
            c[i * r + j] = s;
        }
    }
    c
}

/// Transpose a `rows × cols` matrix.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// `‖W·Wᵀ − I_m‖_F`.
fn orthonormality_err(w: &[f32], m: usize, n: usize) -> f32 {
    let wwt = matmul_bt(w, w, m, n, m);
    let mut s = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            let target = if i == j { 1.0 } else { 0.0 };
            let d = (wwt[i * m + j] - target) as f64;
            s += d * d;
        }
    }
    s.sqrt() as f32
}

// ── Fixed deterministic test data ───────────────────────────────────

/// A fixed SPD matrix `A = Bᵀ·B + N·I` (N×N), diagonally dominant so its
/// spectrum is well-separated. Deterministic — no RNG.
fn spd_a() -> Vec<f32> {
    // A pseudo-"random" but fully deterministic dense B (N×N).
    let mut b = vec![0.0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let k = (i * N + j) as f32;
            b[i * N + j] = (0.37 * k + 1.0).sin() * 0.5 + 0.2 * ((k * 0.11).cos());
        }
    }
    // A = Bᵀ·B + N·I  (symmetric positive-definite).
    let bt = transpose(&b, N, N);
    let mut a = matmul(&bt, &b, N, N, N);
    for i in 0..N {
        a[i * N + i] += N as f32;
    }
    // Symmetrize to kill any f32 round-off asymmetry.
    for i in 0..N {
        for j in (i + 1)..N {
            let avg = 0.5 * (a[i * N + j] + a[j * N + i]);
            a[i * N + j] = avg;
            a[j * N + i] = avg;
        }
    }
    a
}

/// A fixed semi-orthogonal starting point `W₀` (M×N, row-orthonormal):
/// take the first M rows of an identity-ish pattern and orthonormalize.
fn w0() -> Vec<f32> {
    let mut w = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let k = (i * N + j) as f32;
            w[i * N + j] = (0.5 * k + 0.3).cos() + if i == j { 1.0 } else { 0.0 };
        }
    }
    // Gram–Schmidt the rows so W₀·W₀ᵀ = I_M exactly (start ON manifold).
    for i in 0..M {
        for j in 0..i {
            let mut dot = 0.0f32;
            for c in 0..N {
                dot += w[i * N + c] * w[j * N + c];
            }
            for c in 0..N {
                w[i * N + c] -= dot * w[j * N + c];
            }
        }
        let mut nrm = 0.0f64;
        for c in 0..N {
            nrm += (w[i * N + c] as f64).powi(2);
        }
        let inv = 1.0 / (nrm.sqrt() as f32);
        for c in 0..N {
            w[i * N + c] *= inv;
        }
    }
    w
}

/// Euclidean gradient of `L(W) = −tr(Wᵀ A W)` w.r.t. `W` (M×N).
/// `∂L/∂W = −2·(A·Wᵀ)ᵀ = −2·W·A` (using A symmetric). Here W is M×N and
/// A is N×N, so `W·A` is M×N.
fn grad_neg_trace(w: &[f32], a: &[f32]) -> Vec<f32> {
    let wa = matmul(w, a, M, N, N); // M×N
    wa.iter().map(|x| -2.0 * x).collect()
}

/// Loss `L(W) = −tr(Wᵀ A W) = −Σ_i (row_i · A · row_iᵀ)`.
fn loss_neg_trace(w: &[f32], a: &[f32]) -> f32 {
    let wa = matmul(w, a, M, N, N); // M×N
    // tr(Wᵀ A W) = Σ_{i,j} W[i,j]·(A·Wᵀ)[j,i] = Σ (W ∘ (W·A)).
    let mut tr = 0.0f32;
    for i in 0..M * N {
        tr += w[i] * wa[i];
    }
    -tr
}

/// Analytic optimum of `min −tr(Wᵀ A W)` over St(M,N): `−Σ` of the M
/// largest eigenvalues of A. Computed here via a tiny cyclic-Jacobi
/// eigensolver on the symmetric N×N `A` (deterministic).
fn optimal_loss(a: &[f32]) -> f32 {
    let mut m = a.to_vec();
    // Cyclic Jacobi to diagonalize; we only need the eigenvalues.
    for _ in 0..100 {
        let mut off = 0.0f32;
        for p in 0..N {
            for q in (p + 1)..N {
                off += m[p * N + q].powi(2);
            }
        }
        if off.sqrt() < 1e-9 {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = m[p * N + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let app = m[p * N + p];
                let aqq = m[q * N + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                m[p * N + p] = app - t * apq;
                m[q * N + q] = aqq + t * apq;
                m[p * N + q] = 0.0;
                m[q * N + p] = 0.0;
                for r in 0..N {
                    if r != p && r != q {
                        let arp = m[r * N + p];
                        let arq = m[r * N + q];
                        m[r * N + p] = c * arp - s * arq;
                        m[r * N + q] = s * arp + c * arq;
                        m[p * N + r] = m[r * N + p];
                        m[q * N + r] = m[r * N + q];
                    }
                }
            }
        }
    }
    let mut eig: Vec<f32> = (0..N).map(|i| m[i * N + i]).collect();
    eig.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    -eig.iter().take(M).sum::<f32>()
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn stiefel_preserves_orthonormality() {
    let a = spd_a();
    let mut w = w0();
    assert!(
        orthonormality_err(&w, M, N) < 1e-6,
        "starting point not on manifold"
    );
    let mut opt = Stiefel::new(0.05);
    let shape = [M, N];
    for _ in 0..500 {
        let g = grad_neg_trace(&w, &a);
        opt.step("W", &shape, &mut w, &g);
    }
    let e = orthonormality_err(&w, M, N);
    assert!(e < 1e-6, "‖W·Wᵀ − I‖ drifted to {e}");
}

#[test]
fn stiefel_momentum_preserves_orthonormality() {
    let a = spd_a();
    let mut w = w0();
    let mut opt = Stiefel::new(0.02).with_momentum(0.9);
    let shape = [M, N];
    for _ in 0..500 {
        let g = grad_neg_trace(&w, &a);
        opt.step("W", &shape, &mut w, &g);
    }
    let e = orthonormality_err(&w, M, N);
    assert!(e < 1e-6, "‖W·Wᵀ − I‖ (momentum) drifted to {e}");
}

#[test]
fn stiefel_descends_and_converges() {
    let a = spd_a();
    let opt_star = optimal_loss(&a);
    let mut w = w0();
    let mut opt = Stiefel::new(0.05);
    let shape = [M, N];

    let mut prev = loss_neg_trace(&w, &a);
    let first = prev;
    // Monotone (non-increasing) descent, modulo a tiny tolerance for
    // f32 round-off in the retraction.
    for _ in 0..3000 {
        let g = grad_neg_trace(&w, &a);
        opt.step("W", &shape, &mut w, &g);
        let cur = loss_neg_trace(&w, &a);
        assert!(
            cur <= prev + 1e-4,
            "loss increased: {prev} → {cur} (Δ {})",
            cur - prev
        );
        prev = cur;
    }
    let final_loss = loss_neg_trace(&w, &a);
    assert!(
        final_loss < first,
        "no overall progress: {first} → {final_loss}"
    );
    // Converged near the analytic optimum (sum of top-M eigenvalues).
    // Riemannian SGD converges linearly, so the tail is a small f32
    // residual; 1e-4 is comfortably below the ~1e-3 reached at 800
    // steps and hit well before the 3000-step budget.
    let gap = (final_loss - opt_star).abs();
    assert!(
        gap < 1e-4,
        "did not reach optimum: final {final_loss}, optimal {opt_star}, gap {gap}"
    );
}

#[test]
fn stiefel_deterministic() {
    // Same inputs → bit-identical trajectory (no hidden RNG).
    let run = || {
        let a = spd_a();
        let mut w = w0();
        let mut opt = Stiefel::new(0.05).with_momentum(0.5);
        let shape = [M, N];
        for _ in 0..100 {
            let g = grad_neg_trace(&w, &a);
            opt.step("W", &shape, &mut w, &g);
        }
        w
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "Stiefel step is not deterministic");
}

#[test]
fn stiefel_non_matrix_falls_back_to_sgd() {
    // 1-D parameter must not be treated as a Stiefel point — plain SGD.
    let mut opt = Stiefel::new(0.5);
    let target = vec![1.0f32, -2.0, 3.0, 0.5];
    let mut x = vec![0.0f32; target.len()];
    let shape = [target.len()];
    for _ in 0..200 {
        let g: Vec<f32> = x.iter().zip(&target).map(|(xi, ti)| xi - ti).collect();
        opt.step("v", &shape, &mut x, &g);
    }
    let e: f32 = x
        .iter()
        .zip(&target)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(e < 1e-4, "SGD fallback residual {e}");
}
