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

//! CPU-vs-CUDA parity for the core Riemannian / SPD-manifold ops, which run on
//! the CPU host-fallback path (`rlx_cuda::spd` / `rlx_cuda::spd_host`, D2H →
//! CPU F64 reference → H2D). The SPD ops are F64 and have no CUDA kernel; on
//! CUDA they run the SAME `rlx-cpu` thunk kernels the CPU backend uses
//! (widening the f32 arena values to f64 in a one-op CPU graph), then write the
//! f32 result back to the device arena.
//!
//! NOTE: real compute runs ONLY when a CUDA device is reachable (Linux +
//! NVIDIA driver). On a driverless host (this macOS dev box, CI without a GPU)
//! every test is a graceful no-op via `rlx_cuda::is_available()`. These
//! assertions have therefore NOT been executed on this host — they are for the
//! user's Linux CUDA rigs.
//!
//! References are eigendecomposition-free (diagonal inputs / manual matmuls),
//! so no `rlx` umbrella / Session dependency is needed.

use rlx_cuda::CudaExecutable;
use rlx_ir::{DType, Graph, NodeId, Shape};

fn s(dims: &[usize]) -> Shape {
    // SPD ops are F64 in the IR (the CPU kernels require f64); the CUDA arena
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

// Helpers for the gradient parity test (F64 CPU reference vs f32 CUDA surface).
fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn as_f32(xs: &[f64]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
/// Deterministic diagonally-dominant SPD matrix (f64).
fn spd(n: usize, seed: f64) -> Vec<f64> {
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = ((i as f64 * 1.3 + j as f64 * 0.7 + seed).sin()) * 0.2;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] += n as f64 + 1.0 + i as f64;
    }
    a
}

/// ReEig on a diagonal SPD matrix floors each diagonal entry at `eps`; the
/// off-diagonal stays zero. Reference is exact (no eigendecomposition needed).
#[test]
fn reeig_forward_floors_diagonal() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping reeig_forward_floors_diagonal");
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
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();

    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        want[i * n + i] = diag[i].max(eps);
    }
    assert!(
        close(&got, &want, 1e-4),
        "ReEig CUDA vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// BiMap `Y = W·X·Wᵀ` against a manual f32 matmul reference.
#[test]
fn bimap_forward_matches_manual() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping bimap_forward_matches_manual");
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
    let mut exe = CudaExecutable::compile(g);
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
        "BiMap CUDA vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// Weighted Karcher barycentre of identical points is that point (any weights).
#[test]
fn karcher_mean_weighted_of_identicals() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping karcher_mean_weighted");
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
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("x", &x), ("w", &weights)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &one, 1e-3),
        "weighted Karcher CUDA vs reference:\n got={got:?}\n want={one:?}"
    );
}

/// log_map(I, X) on a diagonal spectrum reduces to diag(log λ).
#[test]
fn log_map_identity_base_diagonal() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping log_map");
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
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("x", &x)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[0.0, 1.0, 4.0f32.ln()]);
    assert!(
        close(&got, &want, 1e-3),
        "log_map CUDA vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// exp_map(I, V) on a diagonal tangent reduces to diag(exp v).
#[test]
fn exp_map_identity_base_diagonal() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping exp_map");
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
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("v", &v)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[1.0, std::f32::consts::E, 4.0]);
    assert!(
        close(&got, &want, 1e-3),
        "exp_map CUDA vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Parallel transport of a diagonal tangent between diagonal base points:
/// `Γ_{P→Q}(V)[k,k] = V[k,k]·Q[k,k]/P[k,k]`.
#[test]
fn parallel_transport_diagonal() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping parallel_transport");
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
    let mut exe = CudaExecutable::compile(g);
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
        "parallel_transport CUDA vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Batched logm over a stack of diagonal matrices ⇒ per-slice diag(log λ).
#[test]
fn matrix_fn_batch_logm_diagonal() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping matrix_fn_batch");
        return;
    }
    let (n, batch) = (2usize, 2usize);
    let mut x = diagf(&[1.0, 4.0]);
    x.extend(diagf(&[std::f32::consts::E, 9.0]));
    let mut g = Graph::new("logm_batch");
    let x_n = g.input("x", s(&[batch, n, n]));
    let y = g.spd_logm_batch(x_n);
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();
    let mut want = diagf(&[0.0, 4.0f32.ln()]);
    want.extend(diagf(&[1.0, 9.0f32.ln()]));
    assert!(
        close(&got, &want, 1e-3),
        "logm_batch CUDA vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Differentiate through `log_map` and check the **gradient** matches the CPU
/// backend on real CUDA hardware — exercises `SpdLogMapBackward` on the CUDA
/// host-delegation path (backward op is F64, host-delegated like the forward).
#[test]
fn log_map_grad_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping log_map_grad");
        return;
    }
    use rlx_ir::GraphExt;
    use rlx_runtime::{Device, Session};
    let n = 3usize;
    let base = spd(n, 0.7);
    let x = spd(n, 2.1);
    let build = |g: &mut Graph| -> (NodeId, NodeId) {
        let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.spd_log_map(b_n, x_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (b_n, x_n)
    };
    let mut fg = Graph::new("log_map_fwd");
    let (b_n, x_n) = build(&mut fg);
    let bwd = rlx_opt::autodiff::grad_with_loss(&fg, &[b_n, x_n]); // [loss, d_base, d_x]

    // CPU reference (true f64).
    let mut cs = Session::new(Device::Cpu).compile(bwd.clone());
    let cpu_outs = cs.run_typed(&[
        ("base", &f64s_to_bytes(&base), DType::F64),
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let cpu_dx = as_f32(&bytes_to_f64s(&cpu_outs[2].0));

    // CUDA (f32 surface; SPD subgraph widened, backward host-delegated).
    let mut exe = CudaExecutable::compile(bwd);
    let gpu_outs = exe.run(&[
        ("base", &as_f32(&base)),
        ("x", &as_f32(&x)),
        ("d_output", &[1.0f32]),
    ]);
    let err = max_abs(&cpu_dx, &gpu_outs[2]);
    eprintln!("log_map grad: max_abs={err:.6}");
    assert!(err < 1e-3, "log_map grad CUDA vs CPU max_abs={err}");
}

/// Differentiable symmetric eigendecomposition on real CUDA hardware:
/// `Σλ = trace(A) ⇒ ∂/∂A = I`. Exercises `Op::Eigh` + `Op::EighBackward` on the
/// CUDA path, and checks the gradient equals identity vs the CPU backend.
#[test]
fn eigh_grad_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping eigh_grad");
        return;
    }
    use rlx_ir::GraphExt;
    use rlx_runtime::{Device, Session};
    let n = 4usize;
    // well-separated SPD
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
    let build = |g: &mut Graph| -> NodeId {
        let a_n = g.input("a", Shape::new(&[n, n], DType::F64));
        let (lam, _u) = g.eigh(a_n);
        let loss = g.sum(lam, vec![0], false);
        g.set_outputs(vec![loss]);
        a_n
    };
    let mut fg = Graph::new("eigh_fwd");
    let a_n = build(&mut fg);
    let bwd = rlx_opt::autodiff::grad_with_loss(&fg, &[a_n]); // [loss, dA]

    let mut cs = Session::new(Device::Cpu).compile(bwd.clone());
    let cpu = cs.run_typed(&[
        ("a", &f64s_to_bytes(&a), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let cpu_da = as_f32(&bytes_to_f64s(&cpu[1].0));

    let mut exe = CudaExecutable::compile(bwd);
    let gpu = exe.run(&[("a", &as_f32(&a)), ("d_output", &[1.0f32])]);
    let gpu_da = &gpu[1];

    // Both equal identity, and agree with each other.
    let mut ident = vec![0f32; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    eprintln!(
        "eigh grad: vs-cpu={:.6} vs-I={:.6}",
        max_abs(&cpu_da, gpu_da),
        max_abs(gpu_da, &ident)
    );
    assert!(max_abs(&cpu_da, gpu_da) < 1e-3, "eigh grad CUDA vs CPU");
    assert!(max_abs(gpu_da, &ident) < 1e-3, "eigh grad CUDA != identity");
}

/// Native cuSOLVER `SsyevjBatched` forward path (`Step::EighNative`): reconstruct
/// `A = U diag(λ) Uᵀ` from the GPU-native eigendecomposition and check it matches
/// the input — validates the solver output *and* the col-major→packed `[λ ∥ U]`
/// assemble kernel (single + batched).
#[test]
fn eigh_native_forward_reconstructs() {
    if !rlx_cuda::is_available() {
        eprintln!("[rlx-cuda spd] no CUDA device — skipping eigh_native");
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
    let outs = CudaExecutable::compile(g).run(&[("a", &a)]);
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
    let outs = CudaExecutable::compile(g).run(&[("x", &x)]);
    let a1 = mk(n);
    for b in 0..batch {
        recon_ok(&outs[0], &outs[1], &a1, n, b * n * n, b * n);
    }
}

#[test]
fn unavailable_is_graceful() {
    // Never panics regardless of host.
    let _ = rlx_cuda::is_available();
}
