// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! CPU-vs-Metal parity for the core Riemannian / SPD-manifold ops, which run on
//! the CPU host-fallback path (`rlx_metal::spd`). The SPD ops are F64 and have
//! no MSL eigen kernel; on the native Metal backend they run the SAME
//! `rlx-cpu` thunk kernels the CPU backend uses (against the unified-memory
//! arena, widening the arena's f32 tensors to f64 in a one-op CPU graph), and
//! write the f32 result back. Mirrors `rlx-vulkan/tests/spd_host_parity.rs` and
//! `rlx-wgpu/tests/spd_host_parity.rs`.
//!
//! The graph is F64, but a `Session` feeds it f32 values through the f32
//! surface (`run(&[(&str, &[f32])])`). The Metal arena is f32-uniform for the
//! SPD subgraph (widened at compile time), so that feed is exact for Metal.
//! The comparison target is the **CPU SPD reference** (`rlx_cpu::spd`) —
//! bit-for-bit the same kernels Metal delegates to — evaluated directly in F64.
//! (`Session(Device::Cpu).run(&[f32])` can't be the target: the CPU backend's
//! f32 input surface doesn't widen f32→f64 for a genuinely-F64 graph, so its
//! F64 output would be garbage; the direct `rlx_cpu::spd` call IS the CPU
//! ground truth those Metal thunks run.)
//!
//! Guarded with `rlx_metal::is_available()` so it is a graceful no-op on a
//! headless / non-Metal host (CI).

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// SPD ops are F64 in the IR (the CPU kernels `expect_f64`); the Metal arena is
/// widened to f32 for the SPD subgraph and the host fallback widens on the fly.
fn s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F64)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Run `g` on Metal via the f32 surface (the SPD subgraph's arena is
/// f32-uniform) and return output 0.
fn run_metal(g: &Graph, feeds: &[(&str, &[f32])]) -> Vec<f32> {
    let mut m = Session::new(Device::Metal).compile(g.clone());
    m.run(feeds).remove(0)
}

fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Deterministic diagonally-dominant (hence SPD) symmetric `n×n` matrix, f32
/// row-major — no RNG, reproducible.
fn spd_f32(n: usize, seed: f32) -> Vec<f32> {
    let mut a = vec![0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let v = ((i as f32 * 1.3 + j as f32 * 0.7 + seed).sin()) * 0.2;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] += n as f32 + 1.0 + i as f32;
    }
    a
}

/// BiMap `Y = W · X · Wᵀ` — Metal host-fallback vs the CPU SPD reference.
#[test]
fn bimap_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping bimap_metal_matches_cpu");
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

    let metal = run_metal(&g, &[("w", &w), ("x", &x)]);
    let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let cpu = f64_to_f32(&rlx_cpu::spd::bimap(&wf, &xf, m, n));

    let d = max_abs(&metal, &cpu);
    eprintln!("bimap: max_abs={d:.6}\n metal={metal:?}\n cpu={cpu:?}");
    assert!(d < 1e-3, "BiMap Metal vs CPU max_abs={d}");
}

/// ReEig (eigenvalue rectification, the SPD ReLU) — Metal vs the CPU reference.
#[test]
fn reeig_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping reeig_metal_matches_cpu");
        return;
    }
    let n = 4usize;
    let eps = 0.25f32;
    // Symmetric SPD-ish X (diagonally dominant); the low eigenvalue is floored.
    let x = vec![
        0.10f32, 0.02, 0.01, 0.00, //
        0.02, 0.50, 0.03, 0.02, //
        0.01, 0.03, 2.00, 0.10, //
        0.00, 0.02, 0.10, 7.00, //
    ];

    let mut g = Graph::new("reeig");
    let x_n = g.input("x", s(&[n, n]));
    let y = g.reeig(x_n, eps); // narrows the packed [Y,λ,U] to [n,n]
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("x", &x)]);
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let cpu = f64_to_f32(&rlx_cpu::spd::reeig(&xf, n, eps as f64));

    let d = max_abs(&metal, &cpu);
    eprintln!("reeig: max_abs={d:.6}\n metal={metal:?}\n cpu={cpu:?}");
    assert!(d < 1e-3, "ReEig Metal vs CPU max_abs={d}");
}

/// LogEig (matrix log to the tangent space) — Metal vs the CPU reference.
#[test]
fn logeig_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping logeig_metal_matches_cpu");
        return;
    }
    let n = 3usize;
    let eps = 1e-4f32;
    // Symmetric SPD X (diagonally dominant, positive eigenvalues).
    let x = vec![
        2.0f32, 0.3, 0.2, //
        0.3, 3.0, 0.4, //
        0.2, 0.4, 5.0, //
    ];

    let mut g = Graph::new("logeig");
    let x_n = g.input("x", s(&[n, n]));
    let y = g.logeig(x_n, eps);
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("x", &x)]);
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let cpu = f64_to_f32(&rlx_cpu::spd::logeig(&xf, n, eps as f64));

    let d = max_abs(&metal, &cpu);
    eprintln!("logeig: max_abs={d:.6}\n metal={metal:?}\n cpu={cpu:?}");
    assert!(d < 1e-3, "LogEig Metal vs CPU max_abs={d}");
}

/// Weighted Karcher barycentre — Metal host-fallback vs the CPU reference.
#[test]
fn karcher_mean_weighted_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping karcher_mean_weighted");
        return;
    }
    let (n, batch) = (3usize, 4usize);
    let mut x = Vec::new();
    for bi in 0..batch {
        x.extend(spd_f32(n, bi as f32 * 0.5 + 0.2));
    }
    let weights = vec![0.4f32, 0.1, 0.3, 0.2];

    let mut g = Graph::new("karcher_w");
    let x_n = g.input("x", s(&[batch, n, n]));
    let w_n = g.input("w", s(&[batch]));
    let m = g.spd_karcher_mean_weighted(x_n, w_n, 50, 1e-10);
    g.set_outputs(vec![m]);

    let metal = run_metal(&g, &[("x", &x), ("w", &weights)]);
    let covs: Vec<Vec<f64>> = (0..batch)
        .map(|bi| to_f64(&x[bi * n * n..(bi + 1) * n * n]))
        .collect();
    let cpu = f64_to_f32(&rlx_cpu::spd::karcher_mean_weighted(
        &covs,
        &to_f64(&weights),
        n,
        50,
        1e-10,
    ));
    let d = max_abs(&metal, &cpu);
    eprintln!("karcher_weighted: max_abs={d:.6}");
    assert!(d < 1e-3, "weighted Karcher Metal vs CPU max_abs={d}");
}

/// AIRM log map at an arbitrary base — Metal vs the CPU reference.
#[test]
fn log_map_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping log_map");
        return;
    }
    let n = 3usize;
    let base = spd_f32(n, 0.3);
    let x = spd_f32(n, 1.1);

    let mut g = Graph::new("log_map");
    let b_n = g.input("base", s(&[n, n]));
    let x_n = g.input("x", s(&[n, n]));
    let y = g.spd_log_map(b_n, x_n);
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("base", &base), ("x", &x)]);
    let cpu = f64_to_f32(&rlx_cpu::spd::log_map(&to_f64(&base), &to_f64(&x), n));
    let d = max_abs(&metal, &cpu);
    eprintln!("log_map: max_abs={d:.6}");
    assert!(d < 1e-3, "log_map Metal vs CPU max_abs={d}");
}

/// AIRM exp map at an arbitrary base — Metal vs the CPU reference.
#[test]
fn exp_map_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping exp_map");
        return;
    }
    let n = 3usize;
    let base = spd_f32(n, 0.5);
    // A symmetric tangent vector (small, so Exp stays well-conditioned).
    let v = vec![0.2f32, 0.1, -0.05, 0.1, 0.3, 0.08, -0.05, 0.08, 0.15];

    let mut g = Graph::new("exp_map");
    let b_n = g.input("base", s(&[n, n]));
    let v_n = g.input("v", s(&[n, n]));
    let y = g.spd_exp_map(b_n, v_n);
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("base", &base), ("v", &v)]);
    let cpu = f64_to_f32(&rlx_cpu::spd::exp_map(&to_f64(&base), &to_f64(&v), n));
    let d = max_abs(&metal, &cpu);
    eprintln!("exp_map: max_abs={d:.6}");
    assert!(d < 1e-3, "exp_map Metal vs CPU max_abs={d}");
}

/// AIRM parallel transport — Metal vs the CPU reference.
#[test]
fn parallel_transport_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping parallel_transport");
        return;
    }
    let n = 3usize;
    let from = spd_f32(n, 0.4);
    let to = spd_f32(n, 1.7);
    let v = vec![0.2f32, 0.1, -0.05, 0.1, 0.3, 0.08, -0.05, 0.08, 0.15];

    let mut g = Graph::new("transport");
    let f_n = g.input("from", s(&[n, n]));
    let t_n = g.input("to", s(&[n, n]));
    let v_n = g.input("v", s(&[n, n]));
    let y = g.spd_parallel_transport(f_n, t_n, v_n);
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("from", &from), ("to", &to), ("v", &v)]);
    let cpu = f64_to_f32(&rlx_cpu::spd::parallel_transport(
        &to_f64(&from),
        &to_f64(&to),
        &to_f64(&v),
        n,
    ));
    let d = max_abs(&metal, &cpu);
    eprintln!("parallel_transport: max_abs={d:.6}");
    assert!(d < 1e-3, "parallel_transport Metal vs CPU max_abs={d}");
}

/// Batched matrix logarithm — Metal vs the CPU reference.
#[test]
fn matrix_fn_batch_metal_matches_cpu() {
    if !rlx_metal::is_available() {
        eprintln!("[rlx-metal spd] no Metal device — skipping matrix_fn_batch");
        return;
    }
    let (n, batch) = (3usize, 3usize);
    let mut x = Vec::new();
    for bi in 0..batch {
        x.extend(spd_f32(n, bi as f32 * 0.6 + 0.1));
    }

    let mut g = Graph::new("logm_batch");
    let x_n = g.input("x", s(&[batch, n, n]));
    let y = g.spd_logm_batch(x_n);
    g.set_outputs(vec![y]);

    let metal = run_metal(&g, &[("x", &x)]);
    let covs: Vec<Vec<f64>> = (0..batch)
        .map(|bi| to_f64(&x[bi * n * n..(bi + 1) * n * n]))
        .collect();
    let cpu = f64_to_f32(&rlx_cpu::spd::logm_batch(&covs, n).concat());
    let d = max_abs(&metal, &cpu);
    eprintln!("logm_batch: max_abs={d:.6}");
    assert!(d < 1e-3, "logm_batch Metal vs CPU max_abs={d}");
}

#[test]
fn unavailable_is_graceful() {
    // Never panics regardless of host.
    let _ = rlx_metal::is_available();
}
