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

//! CPU-vs-wgpu parity for the core Riemannian / SPD-manifold ops, which run on
//! the CPU host-fallback path (`rlx_wgpu::spd_host`). The SPD ops are F64 and
//! have no WGSL kernel; on wgpu they read the arena span back, run the SAME
//! `rlx-cpu` thunk kernels the CPU backend uses (widening the f32 arena bytes
//! to f64), and write the f32 result back. This asserts the wgpu forward
//! output matches the CPU reference (`Device::Cpu`, run in true f64) within
//! f32 tolerance.
//!
//! The SPD graph I/O is F64 on the CPU backend (`run_typed`), but the wgpu
//! executable API is f32-only; feeding the same numeric values, the wgpu path
//! widens f32→f64 internally, so results match to f32 precision.
//!
//! Runs only when a wgpu device is present (Metal / MoltenVK / lavapipe);
//! otherwise a graceful no-op.

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

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
fn as_f32(xs: &[f64]) -> Vec<f32> {
    xs.iter().map(|&x| x as f32).collect()
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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// CPU reference (true f64) for a single-output SPD graph built by `build`.
fn cpu_ref(build: impl Fn(&mut Graph), inputs: &[(&str, &[f64])]) -> Vec<f32> {
    use rlx::prelude::*;
    let mut g = Graph::new("spd_ref");
    build(&mut g);
    let typed: Vec<(&str, Vec<u8>, DType)> = inputs
        .iter()
        .map(|(n, d)| (*n, f64s_to_bytes(d), DType::F64))
        .collect();
    let refs: Vec<(&str, &[u8], DType)> = typed
        .iter()
        .map(|(n, b, d)| (*n, b.as_slice(), *d))
        .collect();
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&refs);
    as_f32(&bytes_to_f64s(&outs[0].0))
}

/// wgpu output (f32 API, widens to f64 in the host fallback) for the same graph.
fn wgpu_out(build: impl Fn(&mut Graph), inputs: &[(&str, &[f64])]) -> Vec<f32> {
    let mut g = Graph::new("spd_wgpu");
    build(&mut g);
    let f32_inputs: Vec<(&str, Vec<f32>)> = inputs.iter().map(|(n, d)| (*n, as_f32(d))).collect();
    let refs: Vec<(&str, &[f32])> = f32_inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let mut exe = WgpuExecutable::compile(g);
    exe.run(&refs).into_iter().next().unwrap()
}

#[test]
fn reeig_forward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping reeig_forward_matches_cpu");
        return;
    }
    let n = 4;
    let x = spd(n, 0.9);
    let build = |g: &mut Graph| {
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.reeig(x_n, 0.5);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("x", &x)]);
    let got = wgpu_out(build, &[("x", &x)]);
    let err = max_abs(&want, &got);
    assert!(
        err < 1e-3,
        "ReEig wgpu vs cpu max_abs={err:.3e}\n cpu={want:?}\n gpu={got:?}"
    );
}

#[test]
fn logeig_forward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping logeig_forward_matches_cpu");
        return;
    }
    let n = 4;
    let x = spd(n, 1.3);
    let build = |g: &mut Graph| {
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.logeig(x_n, 1e-6);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("x", &x)]);
    let got = wgpu_out(build, &[("x", &x)]);
    let err = max_abs(&want, &got);
    assert!(
        err < 1e-3,
        "LogEig wgpu vs cpu max_abs={err:.3e}\n cpu={want:?}\n gpu={got:?}"
    );
}

#[test]
fn bimap_forward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping bimap_forward_matches_cpu");
        return;
    }
    let (m, n) = (2usize, 3usize);
    let w = vec![1.0f64, 0.5, -0.25, 2.0, -1.0, 0.75];
    let x = spd(n, 0.3);
    let build = |g: &mut Graph| {
        let w_n = g.input("w", Shape::new(&[m, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.bimap(w_n, x_n);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("w", &w), ("x", &x)]);
    let got = wgpu_out(build, &[("w", &w), ("x", &x)]);
    let err = max_abs(&want, &got);
    assert!(
        err < 1e-3,
        "BiMap wgpu vs cpu max_abs={err:.3e}\n cpu={want:?}\n gpu={got:?}"
    );
}

#[test]
fn karcher_mean_weighted_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping karcher_mean_weighted");
        return;
    }
    let (n, batch) = (3usize, 4usize);
    let mut x = Vec::new();
    for bi in 0..batch {
        x.extend(spd(n, bi as f64 * 0.5 + 0.2));
    }
    let weights = vec![0.4f64, 0.1, 0.3, 0.2];
    let build = |g: &mut Graph| {
        let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
        let w_n = g.input("w", Shape::new(&[batch], DType::F64));
        let m = g.spd_karcher_mean_weighted(x_n, w_n, 50, 1e-10);
        g.set_outputs(vec![m]);
    };
    let want = cpu_ref(build, &[("x", &x), ("w", &weights)]);
    let got = wgpu_out(build, &[("x", &x), ("w", &weights)]);
    let err = max_abs(&want, &got);
    assert!(err < 1e-3, "weighted Karcher wgpu vs cpu max_abs={err:.3e}");
}

#[test]
fn log_map_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping log_map");
        return;
    }
    let n = 3usize;
    let base = spd(n, 0.3);
    let x = spd(n, 1.1);
    let build = |g: &mut Graph| {
        let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.spd_log_map(b_n, x_n);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("base", &base), ("x", &x)]);
    let got = wgpu_out(build, &[("base", &base), ("x", &x)]);
    let err = max_abs(&want, &got);
    assert!(err < 1e-3, "log_map wgpu vs cpu max_abs={err:.3e}");
}

#[test]
fn exp_map_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping exp_map");
        return;
    }
    let n = 3usize;
    let base = spd(n, 0.5);
    let v = sym(n, 2.0); // symmetric tangent
    let build = |g: &mut Graph| {
        let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
        let y = g.spd_exp_map(b_n, v_n);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("base", &base), ("v", &v)]);
    let got = wgpu_out(build, &[("base", &base), ("v", &v)]);
    let err = max_abs(&want, &got);
    assert!(err < 1e-3, "exp_map wgpu vs cpu max_abs={err:.3e}");
}

#[test]
fn parallel_transport_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping parallel_transport");
        return;
    }
    let n = 3usize;
    let from = spd(n, 0.4);
    let to = spd(n, 1.7);
    let v = sym(n, 2.0);
    let build = |g: &mut Graph| {
        let f_n = g.input("from", Shape::new(&[n, n], DType::F64));
        let t_n = g.input("to", Shape::new(&[n, n], DType::F64));
        let v_n = g.input("v", Shape::new(&[n, n], DType::F64));
        let y = g.spd_parallel_transport(f_n, t_n, v_n);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("from", &from), ("to", &to), ("v", &v)]);
    let got = wgpu_out(build, &[("from", &from), ("to", &to), ("v", &v)]);
    let err = max_abs(&want, &got);
    assert!(
        err < 1e-3,
        "parallel_transport wgpu vs cpu max_abs={err:.3e}"
    );
}

#[test]
fn matrix_fn_batch_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping matrix_fn_batch");
        return;
    }
    let (n, batch) = (3usize, 3usize);
    let mut x = Vec::new();
    for bi in 0..batch {
        x.extend(spd(n, bi as f64 * 0.6 + 0.1));
    }
    let build = |g: &mut Graph| {
        let x_n = g.input("x", Shape::new(&[batch, n, n], DType::F64));
        let y = g.spd_logm_batch(x_n);
        g.set_outputs(vec![y]);
    };
    let want = cpu_ref(build, &[("x", &x)]);
    let got = wgpu_out(build, &[("x", &x)]);
    let err = max_abs(&want, &got);
    assert!(err < 1e-3, "logm_batch wgpu vs cpu max_abs={err:.3e}");
}

/// Differentiate through `log_map` and check the **gradient** matches the CPU
/// backend — exercises the new `SpdLogMapBackward` op on the GPU host-delegation
/// path (the backward op is F64, host-delegated exactly like the forward).
#[test]
fn log_map_grad_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[spd_host_parity] no wgpu device — skipping log_map_grad");
        return;
    }
    use rlx::prelude::*;
    let n = 3usize;
    let base = spd(n, 0.7);
    let x = spd(n, 2.1);
    let build = |g: &mut Graph| {
        let b_n = g.input("base", Shape::new(&[n, n], DType::F64));
        let x_n = g.input("x", Shape::new(&[n, n], DType::F64));
        let y = g.spd_log_map(b_n, x_n);
        let loss = g.sum(y, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        (b_n, x_n)
    };
    let mut fg = Graph::new("log_map_fwd");
    let (b_n, x_n) = build(&mut fg);
    let bwd = rlx::opt::grad_with_loss(&fg, &[b_n, x_n]); // outputs [loss, d_base, d_x]

    // CPU reference (true f64).
    let mut cs = Session::new(Device::Cpu).compile(bwd.clone());
    let cpu_outs = cs.run_typed(&[
        ("base", &f64s_to_bytes(&base), DType::F64),
        ("x", &f64s_to_bytes(&x), DType::F64),
        ("d_output", &f64s_to_bytes(&[1.0]), DType::F64),
    ]);
    let cpu_dx = as_f32(&bytes_to_f64s(&cpu_outs[2].0));

    // wgpu (f32 surface; SPD subgraph widened, backward host-delegated).
    let mut exe = WgpuExecutable::compile(bwd);
    let gpu_outs = exe.run(&[
        ("base", &as_f32(&base)),
        ("x", &as_f32(&x)),
        ("d_output", &[1.0f32]),
    ]);
    let err = max_abs(&cpu_dx, &gpu_outs[2]);
    assert!(err < 1e-3, "log_map grad wgpu vs cpu max_abs={err:.3e}");
}
