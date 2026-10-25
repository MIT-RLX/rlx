// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The native batched ReEig/LogEig custom ops (`rlx_cpu::spd_batched`) against
//! two independent references: a host-f64 eigendecomposition for the forward,
//! and autodiff through the graph-primitive Jacobi eigensolver
//! (`spd_*_batched`) for the gradient — so packing, batch indexing, the
//! eigenvector convention and the VJP wiring are all checked.

#![cfg(feature = "cpu")]

use rlx_cpu::spd_batched::{BatchedSpectral, spectral_batched};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const B: usize = 5;
const N: usize = 6;

/// `B` distinct SPD matrices `[B,N,N]`, eigenvalues spread over ~3 decades.
fn batch() -> Vec<f64> {
    let mut out = Vec::with_capacity(B * N * N);
    for b in 0..B {
        let mut m = vec![0f64; N * N];
        for i in 0..N {
            for j in 0..N {
                let mut s = 0.0;
                for k in 0..N {
                    let a = (((i * 7 + k * 3 + b * 5 + 1) % 11) as f64) - 5.0;
                    let c = (((j * 7 + k * 3 + b * 5 + 1) % 11) as f64) - 5.0;
                    s += a * c * 0.1f64.powi(k as i32 % 3);
                }
                m[i * N + j] = s;
            }
            m[i * N + i] += 0.05 * (b + 1) as f64;
        }
        out.extend(m);
    }
    out
}

const KINDS: [BatchedSpectral; 5] = [
    BatchedSpectral::Re,
    BatchedSpectral::Log,
    BatchedSpectral::Sqrt,
    BatchedSpectral::InvSqrt,
    BatchedSpectral::Pow(0.7),
];

fn eps_for(kind: BatchedSpectral) -> f64 {
    // Re and Log: a floor that clamps part of the spectrum, so f′ = 0 below it
    // is exercised (Log at ε = 1.0 is ReEig∘LogEig folded, as SPDNet uses it)
    match kind {
        BatchedSpectral::Re | BatchedSpectral::Log => 1.0,
        _ => 1e-6,
    }
}

/// `f(max(λ, ε))` on the host, for the f64 references.
fn host_f(kind: BatchedSpectral, eps: f64) -> impl Fn(f64) -> f64 {
    move |l: f64| {
        let m = l.max(eps);
        match kind {
            BatchedSpectral::Re => m,
            BatchedSpectral::Log => m.ln(),
            BatchedSpectral::Sqrt => m.sqrt(),
            BatchedSpectral::InvSqrt => 1.0 / m.sqrt(),
            BatchedSpectral::Pow(p) => m.powf(p),
        }
    }
}

fn weights() -> Vec<f32> {
    (0..B * N * N)
        .map(|i| (((i * 13) % 17) as f32 - 8.0) / 8.0)
        .collect()
}

/// `loss = Σ W ⊙ f(X)` via the native op or the Jacobi builder.
fn graph(kind: BatchedSpectral, native: bool) -> (Graph, NodeId, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("spd_batched");
    let x = g.input("x", Shape::new(&[B, N, N], f));
    let eps = eps_for(kind);
    let y = if native {
        spectral_batched(&mut g, x, B, N, eps, kind)
    } else {
        match kind {
            BatchedSpectral::Re => g.spd_reeig_batched(x, B, N, 12, eps),
            BatchedSpectral::Log => g.spd_logeig_batched(x, B, N, 12, eps),
            BatchedSpectral::Sqrt => g.spd_sqrt_batched(x, B, N, 12, eps),
            BatchedSpectral::InvSqrt => g.spd_invsqrt_batched(x, B, N, 12, eps),
            BatchedSpectral::Pow(p) => g.spd_poweig_batched(x, B, N, 12, eps, p),
        }
    };
    let wb: Vec<u8> = weights().iter().flat_map(|v| v.to_le_bytes()).collect();
    let w = g.add_node(Op::Constant { data: wb }, vec![], Shape::new(&[B, N, N], f));
    let wy = g.mul(w, y);
    let flat = g.reshape_(wy, vec![(B * N * N) as i64]);
    let loss = g.reduce(flat, ReduceOp::Sum, vec![0], false, Shape::new(&[1], f));
    (g, y, loss)
}

fn forward(kind: BatchedSpectral, native: bool, x: &[f32]) -> Vec<f32> {
    forward_on(Device::Cpu, kind, native, x)
}

fn forward_on(dev: Device, kind: BatchedSpectral, native: bool, x: &[f32]) -> Vec<f32> {
    let (mut g, y, _) = graph(kind, native);
    g.set_outputs(vec![y]);
    let mut s = Session::new(dev).compile(g);
    s.run(&[("x", x)]).remove(0)
}

fn gradient(kind: BatchedSpectral, native: bool, x: &[f32]) -> Vec<f32> {
    gradient_on(Device::Cpu, kind, native, x)
}

fn gradient_on(dev: Device, kind: BatchedSpectral, native: bool, x: &[f32]) -> Vec<f32> {
    let (mut g, _, loss) = graph(kind, native);
    g.set_outputs(vec![loss]);
    let xid = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "x"))
        .unwrap()
        .id;
    let bwd = rlx_autodiff::grad_with_loss(&g, &[xid]);
    let mut s = Session::new(dev).compile(bwd);
    s.run(&[("x", x), ("d_output", &[1.0f32])]).pop().unwrap()
}

fn rel(a: &[f32], b: &[f64]) -> f64 {
    let num = a
        .iter()
        .zip(b)
        .map(|(&p, &q)| (p as f64 - q).abs())
        .fold(0.0, f64::max);
    num / b.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[test]
fn native_forward_matches_a_host_f64_eigendecomposition() {
    let x64 = batch();
    let x: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    for kind in KINDS {
        let eps = eps_for(kind);
        // reference from the f32-rounded input, in f64
        let want: Vec<f64> = x
            .chunks(N * N)
            .flat_map(|m| {
                let m: Vec<f64> = m.iter().map(|&v| v as f64).collect();
                rlx_cpu::spd::matrix_fn(&m, N, host_f(kind, eps))
            })
            .collect();
        let got = forward(kind, true, &x);
        let r = rel(&got, &want);
        eprintln!("{kind:?} forward rel {r:.2e}");
        assert!(r < 1e-6, "{kind:?}: forward rel {r}");
    }
}

#[test]
fn native_gradient_matches_autodiff_through_the_jacobi_solver() {
    let x: Vec<f32> = batch().iter().map(|&v| v as f32).collect();
    for kind in KINDS {
        let native = gradient(kind, true, &x);
        let jacobi = gradient(kind, false, &x);
        assert_eq!(native.len(), B * N * N);
        // Compare symmetric parts: on symmetric inputs only `dX + dXᵀ` is
        // defined, and autodiff through Jacobi splits it between the two
        // triangles differently (the raw entries differ by ~0.85 here, while
        // both agree with finite differences: see the test below).
        let sym = |g: &[f32]| -> Vec<f64> {
            (0..B * N * N)
                .map(|k| {
                    let (b, i, j) = (k / (N * N), (k / N) % N, k % N);
                    0.5 * (g[k] as f64 + g[b * N * N + j * N + i] as f64)
                })
                .collect()
        };
        let native_sym: Vec<f32> = sym(&native).iter().map(|&v| v as f32).collect();
        let r = rel(&native_sym, &sym(&jacobi));
        eprintln!("{kind:?} gradient rel vs Jacobi {r:.2e}");
        assert!(r < 1e-3, "{kind:?}: gradient rel {r}");
        // the Loewner adjoint is symmetric per matrix
        for m in native.chunks(N * N) {
            for i in 0..N {
                for j in 0..N {
                    assert!(
                        (m[i * N + j] - m[j * N + i]).abs() <= 1e-5 * (1.0 + m[i * N + j].abs())
                    );
                }
            }
        }
    }
}

/// Central differences in f64 along SYMMETRIC perturbations `E_ij + E_ji` —
/// the only directions that stay on the symmetric matrices the op is defined
/// on. The matching gradient component is `dX_ij + dX_ji` (`dX_ii` on the
/// diagonal).
fn fd_symmetric(kind: BatchedSpectral, x: &[f32]) -> Vec<(usize, usize, usize, f64)> {
    let eps = eps_for(kind);
    let w = weights();
    let loss = |m: &[f64], b: usize| -> f64 {
        let y = rlx_cpu::spd::matrix_fn(m, N, host_f(kind, eps));
        y.iter()
            .zip(&w[b * N * N..(b + 1) * N * N])
            .map(|(a, &c)| a * c as f64)
            .sum()
    };
    let mut out = Vec::new();
    for b in 0..B {
        let m: Vec<f64> = x[b * N * N..(b + 1) * N * N]
            .iter()
            .map(|&v| v as f64)
            .collect();
        for (i, j) in [(0usize, 0usize), (1, 3), (2, 5), (4, 4)] {
            let h = 1e-6;
            let (mut p, mut q) = (m.clone(), m.clone());
            p[i * N + j] += h;
            q[i * N + j] -= h;
            if i != j {
                p[j * N + i] += h;
                q[j * N + i] -= h;
            }
            out.push((b, i, j, (loss(&p, b) - loss(&q, b)) / (2.0 * h)));
        }
    }
    out
}

#[test]
fn native_gradient_matches_finite_differences() {
    let x: Vec<f32> = batch().iter().map(|&v| v as f32).collect();
    for kind in KINDS {
        let native = gradient(kind, true, &x);
        let jacobi = gradient(kind, false, &x);
        let comp = |g: &[f32], b: usize, i: usize, j: usize| -> f64 {
            let at = |r: usize, c: usize| g[b * N * N + r * N + c] as f64;
            if i == j {
                at(i, i)
            } else {
                at(i, j) + at(j, i)
            }
        };
        let (mut worst_n, mut worst_j, mut scale) = (0f64, 0f64, 0f64);
        for (b, i, j, fd) in fd_symmetric(kind, &x) {
            worst_n = worst_n.max((comp(&native, b, i, j) - fd).abs());
            worst_j = worst_j.max((comp(&jacobi, b, i, j) - fd).abs());
            scale = scale.max(fd.abs());
        }
        eprintln!(
            "{kind:?}: |native − FD| {:.2e}, |Jacobi − FD| {:.2e} (scale {scale:.2e})",
            worst_n / scale,
            worst_j / scale
        );
        assert!(
            worst_n / scale < 1e-3,
            "{kind:?}: native gradient off FD by {}",
            worst_n / scale
        );
    }
}

/// GPU backends run the op through its CPU kernel (host staging), so a graph
/// using it is not CPU-only. Same numbers as the CPU run.
#[test]
fn gpu_backends_run_the_op_through_the_host_kernel() {
    let x: Vec<f32> = batch().iter().map(|&v| v as f32).collect();
    for dev in [Device::Metal, Device::Mlx, Device::Gpu] {
        if common::skip_unless(dev) {
            eprintln!("{dev:?} unavailable — skip");
            continue;
        }
        let _gpu = common::serialize_gpu();
        for kind in KINDS {
            let (fc, fg) = (forward(kind, true, &x), forward_on(dev, kind, true, &x));
            let (gc, gg) = (gradient(kind, true, &x), gradient_on(dev, kind, true, &x));
            let d = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, f32::max)
            };
            eprintln!(
                "{dev:?} {kind:?}: forward Δ {:.2e}, gradient Δ {:.2e}",
                d(&fc, &fg),
                d(&gc, &gg)
            );
            assert!(d(&fc, &fg) < 1e-4 && d(&gc, &gg) < 1e-3, "{dev:?} {kind:?}");
        }
    }
}
