// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-Vulkan parity for the core Riemannian / SPD-manifold ops, which run
//! on the CPU host-fallback path (`rlx_vulkan::spd`). The SPD ops are F64 and
//! have no SPIR-V kernel; on Vulkan they run the SAME `rlx-cpu` thunk kernels
//! the CPU backend uses (widening the f32 arena values to f64 in a one-op CPU
//! graph), and write the f32 result back to the mapped arena.
//!
//! Runs real compute only when a Vulkan device is reachable (e.g. Linux +
//! native driver, or macOS + MoltenVK via `VK_ICD_FILENAMES`). On a driverless
//! host (macOS without MoltenVK, CI) it is a graceful no-op — the memory note
//! [[vulkan_backend]] documents that Apple Silicon has no Vulkan loader by
//! default, so this parity is validated on the wgpu backend (Metal) instead;
//! see `rlx-wgpu/tests/spd_host_parity.rs`, which exercises the identical
//! host-fallback wiring.
//!
//! References are eigendecomposition-free (diagonal inputs / manual matmuls),
//! so no `rlx` umbrella / Session dependency is needed.

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;

fn s(dims: &[usize]) -> Shape {
    // SPD ops are F64 in the IR (the CPU kernels require f64); the Vulkan
    // arena stays f32 and the host fallback widens on the fly.
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
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping reeig_forward_floors_diagonal");
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
    let mut exe = VulkanExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();

    let mut want = vec![0.0f32; n * n];
    for i in 0..n {
        want[i * n + i] = diag[i].max(eps);
    }
    assert!(
        close(&got, &want, 1e-4),
        "ReEig Vulkan vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// BiMap `Y = W·X·Wᵀ` against a manual f32 matmul reference.
#[test]
fn bimap_forward_matches_manual() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping bimap_forward_matches_manual");
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
    let mut exe = VulkanExecutable::compile(g);
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
        "BiMap Vulkan vs reference mismatch:\n got={got:?}\n want={want:?}"
    );
}

/// Weighted Karcher barycentre of identical points is that point (any weights).
#[test]
fn karcher_mean_weighted_of_identicals() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping karcher_mean_weighted");
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
    let mut exe = VulkanExecutable::compile(g);
    let got = exe
        .run(&[("x", &x), ("w", &weights)])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &one, 1e-3),
        "weighted Karcher Vulkan vs reference:\n got={got:?}\n want={one:?}"
    );
}

/// log_map(I, X) on a diagonal spectrum reduces to diag(log λ).
#[test]
fn log_map_identity_base_diagonal() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping log_map");
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
    let mut exe = VulkanExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("x", &x)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[0.0, 1.0, 4.0f32.ln()]);
    assert!(
        close(&got, &want, 1e-3),
        "log_map Vulkan vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// exp_map(I, V) on a diagonal tangent reduces to diag(exp v).
#[test]
fn exp_map_identity_base_diagonal() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping exp_map");
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
    let mut exe = VulkanExecutable::compile(g);
    let got = exe
        .run(&[("base", &ident), ("v", &v)])
        .into_iter()
        .next()
        .unwrap();
    let want = diagf(&[1.0, std::f32::consts::E, 4.0]);
    assert!(
        close(&got, &want, 1e-3),
        "exp_map Vulkan vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Parallel transport of a diagonal tangent between diagonal base points:
/// `Γ_{P→Q}(V)[k,k] = V[k,k]·Q[k,k]/P[k,k]`.
#[test]
fn parallel_transport_diagonal() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping parallel_transport");
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
    let mut exe = VulkanExecutable::compile(g);
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
        "parallel_transport Vulkan vs reference:\n got={got:?}\n want={want:?}"
    );
}

/// Batched logm over a stack of diagonal matrices ⇒ per-slice diag(log λ).
#[test]
fn matrix_fn_batch_logm_diagonal() {
    if !rlx_vulkan::is_available() {
        eprintln!("[rlx-vulkan spd] no Vulkan device — skipping matrix_fn_batch");
        return;
    }
    let (n, batch) = (2usize, 2usize);
    let mut x = diagf(&[1.0, 4.0]);
    x.extend(diagf(&[std::f32::consts::E, 9.0]));
    let mut g = Graph::new("logm_batch");
    let x_n = g.input("x", s(&[batch, n, n]));
    let y = g.spd_logm_batch(x_n);
    g.set_outputs(vec![y]);
    let mut exe = VulkanExecutable::compile(g);
    let got = exe.run(&[("x", &x)]).into_iter().next().unwrap();
    let mut want = diagf(&[0.0, 4.0f32.ln()]);
    want.extend(diagf(&[1.0, 9.0f32.ln()]));
    assert!(
        close(&got, &want, 1e-3),
        "logm_batch Vulkan vs reference:\n got={got:?}\n want={want:?}"
    );
}

#[test]
fn unavailable_is_graceful() {
    // Never panics regardless of host (mirrors smoke.rs).
    let _ = rlx_vulkan::is_available();
}
