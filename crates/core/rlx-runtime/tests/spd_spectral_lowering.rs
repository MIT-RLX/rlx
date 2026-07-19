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

//! SPD-manifold spectral-op lowering: proves the f32 graph-primitive Jacobi
//! eigensolver (`rewrite`'s `LowerSpectral`, replacing the CPU-only f64 LAPACK
//! `Op::ReEig` / `Op::LogEig`) matches an independent host-f64 eigendecomposition
//! reference — on CPU (pass applied directly) and on Metal (backend auto-rewrite
//! at compile time). This is the mechanism that gives SPDNet / TensorCSPNet /
//! GraphCSPNet / TSMNet real GPU execution.

#![cfg(feature = "cpu")]

use rlx_fusion::lower_spectral::LowerSpectral;
use rlx_fusion::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 6;
const EPS: f64 = 1e-4;

/// Deterministic well-conditioned SPD matrix `[N,N]` (row-major, f64).
fn spd_matrix() -> Vec<f64> {
    let n = N;
    let mut b = vec![0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            b[i * n + k] = (((i * 7 + k * 3 + 1) % 11) as f64) - 5.0;
        }
    }
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0f64;
            for k in 0..n {
                s += b[i * n + k] * b[j * n + k];
            }
            m[i * n + j] = s;
        }
        m[i * n + i] += n as f64; // ensure positive-definite / well-conditioned
    }
    m
}

/// Host-f64 reference: symmetric eigendecomposition via classic Jacobi, then
/// reconstruct `V·diag(f(λ))·Vᵀ` with `f = log∘max(·,eps)` when `log`, else
/// `max(·,eps)`.
fn spectral_ref(a: &[f64], log: bool) -> Vec<f64> {
    let n = N;
    let mut m = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        // find largest off-diagonal
        let (mut p, mut q, mut off) = (0, 1, 0.0);
        for i in 0..n {
            for j in i + 1..n {
                let a = m[i * n + j].abs();
                if a > off {
                    off = a;
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-15 {
            break;
        }
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let theta = 0.5 * (aqq - app) / apq;
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // apply J on both sides: M <- Jᵀ M J, V <- V J
        for k in 0..n {
            let mkp = m[k * n + p];
            let mkq = m[k * n + q];
            m[k * n + p] = c * mkp - s * mkq;
            m[k * n + q] = s * mkp + c * mkq;
        }
        for k in 0..n {
            let mpk = m[p * n + k];
            let mqk = m[q * n + k];
            m[p * n + k] = c * mpk - s * mqk;
            m[q * n + k] = s * mpk + c * mqk;
        }
        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }
    // eigenvalues on the diagonal of m; f(λ)
    let mut f = vec![0f64; n];
    for i in 0..n {
        let lam = m[i * n + i].max(EPS);
        f[i] = if log { lam.ln() } else { lam };
    }
    // Y = V diag(f) Vᵀ
    let mut y = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0f64;
            for k in 0..n {
                s += v[i * n + k] * f[k] * v[j * n + k];
            }
            y[i * n + j] = s;
        }
    }
    y
}

/// Generic host-f64 symmetric eigendecomposition (classic Jacobi) → (λ, V) for
/// arbitrary `n`. `V` is row-major, columns = eigenvectors.
fn jacobi_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..200 {
        let (mut p, mut q, mut off) = (0, 1, 0.0);
        for i in 0..n {
            for j in i + 1..n {
                let av = m[i * n + j].abs();
                if av > off {
                    off = av;
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-16 {
            break;
        }
        let (app, aqq, apq) = (m[p * n + p], m[q * n + q], m[p * n + q]);
        let theta = 0.5 * (aqq - app) / apq;
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..n {
            let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
            m[k * n + p] = c * mkp - s * mkq;
            m[k * n + q] = s * mkp + c * mkq;
        }
        for k in 0..n {
            let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
            m[p * n + k] = c * mpk - s * mqk;
            m[q * n + k] = s * mpk + c * mqk;
        }
        for k in 0..n {
            let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }
    let lam = (0..n).map(|i| m[i * n + i]).collect();
    (lam, v)
}

/// Host `V·diag(f(λ))·Vᵀ` for an arbitrary spectral function.
fn matfn(a: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (lam, v) = jacobi_eig(a, n);
    let fl: Vec<f64> = lam.iter().map(|&l| f(l)).collect();
    let mut y = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0f64;
            for k in 0..n {
                s += v[i * n + k] * fl[k] * v[j * n + k];
            }
            y[i * n + j] = s;
        }
    }
    y
}

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, nn: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * nn];
    for i in 0..m {
        for j in 0..nn {
            let mut s = 0f64;
            for p in 0..k {
                s += a[i * k + p] * b[p * nn + j];
            }
            c[i * nn + j] = s;
        }
    }
    c
}

fn const_f32(g: &mut Graph, xs: &[f64], dims: &[usize]) -> NodeId {
    let bytes: Vec<u8> = xs.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(dims, DType::F32),
    )
}

/// Build an f32 graph: `logeig`/`reeig` of the baked SPD matrix → `[N,N]` output.
fn build_graph(a: &[f64], log: bool) -> Graph {
    let mut g = Graph::new("spd_spectral");
    let x = const_f32(&mut g, a, &[N, N]);
    let y = if log {
        g.logeig(x, EPS as f32)
    } else {
        g.reeig(x, EPS as f32)
    };
    g.set_outputs(vec![y]);
    g
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0.0, f64::max)
}

fn run(dev: Device, g: Graph) -> Vec<f32> {
    let mut c = Session::new(dev).compile(g);
    c.finalize_params();
    c.run(&[]).into_iter().next().expect("one output")
}

/// CPU: apply `LowerSpectral` directly (f32 ReEig/LogEig → Jacobi primitives),
/// then run. Proves the pass output matches the host-f64 reference.
#[test]
fn logeig_lowering_cpu() {
    let a = spd_matrix();
    let g = LowerSpectral.run(build_graph(&a, true));
    assert!(
        !g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::LogEig { .. } | Op::ReEig { .. })),
        "LowerSpectral must eliminate all f32 spectral ops"
    );
    let out = run(Device::Cpu, g);
    let refy = spectral_ref(&a, true);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("logeig CPU: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999, "logeig CPU cos={cos}");
    assert!(mx < 1e-2, "logeig CPU max_abs={mx}");
}

#[test]
fn reeig_lowering_cpu() {
    let a = spd_matrix();
    let g = LowerSpectral.run(build_graph(&a, false));
    let out = run(Device::Cpu, g);
    let refy = spectral_ref(&a, false);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("reeig CPU: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999, "reeig CPU cos={cos}");
    assert!(mx < 1e-2, "reeig CPU max_abs={mx}");
}

/// Regression: a DIAGONAL SPD matrix has every off-diagonal `a_pq == 0` exactly,
/// which the signed denominator floor must survive (else `den==0` → tau=NaN).
/// This is the tsmnet-on-GPU failure mode (near-diagonal transported matrices).
#[test]
fn logeig_lowering_diagonal_no_nan() {
    let n = N;
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        a[i * n + i] = (i + 2) as f64; // distinct positive diagonal, zero off-diagonals
    }
    let out = run(Device::Cpu, LowerSpectral.run(build_graph(&a, true)));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "diagonal logeig produced NaN/Inf: {out:?}"
    );
    let refy = spectral_ref(&a, true);
    let cos = cosine(&out, &refy);
    eprintln!("logeig diagonal CPU: cos={cos:.9}");
    assert!(cos > 0.99999, "logeig diagonal cos={cos}");
}

/// Metal: compile the f32 graph WITH `Op::LogEig` still present — the Metal
/// backend's own `legalize_or_rewrite` fires `LowerSpectral` at compile time.
/// Proves the SPD spectral op runs natively on GPU (no CPU host-fallback).
#[cfg(feature = "metal")]
#[test]
fn logeig_lowering_metal() {
    let a = spd_matrix();
    let out = run(Device::Metal, build_graph(&a, true));
    let refy = spectral_ref(&a, true);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("logeig Metal: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999, "logeig Metal cos={cos}");
    assert!(mx < 1e-2, "logeig Metal max_abs={mx}");
}

#[cfg(feature = "metal")]
#[test]
fn reeig_lowering_metal() {
    let a = spd_matrix();
    let out = run(Device::Metal, build_graph(&a, false));
    let refy = spectral_ref(&a, false);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("reeig Metal: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999, "reeig Metal cos={cos}");
    assert!(mx < 1e-2, "reeig Metal max_abs={mx}");
}

/// wgpu: same end-to-end backend auto-rewrite proof on the cross-platform GPU path.
#[cfg(feature = "gpu")]
#[test]
fn logeig_lowering_wgpu() {
    let a = spd_matrix();
    let out = run(Device::Gpu, build_graph(&a, true));
    let refy = spectral_ref(&a, true);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("logeig wgpu: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.9999, "logeig wgpu cos={cos}");
}

#[cfg(feature = "gpu")]
#[test]
fn reeig_lowering_wgpu() {
    let a = spd_matrix();
    let out = run(Device::Gpu, build_graph(&a, false));
    let refy = spectral_ref(&a, false);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("reeig wgpu: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.9999, "reeig wgpu cos={cos}");
}

/// MLX: minimal single-Scan SPD op (n=6) — isolates whether MLX's Op::Scan path
/// works at all vs only failing on the many-nested-Scan model graphs.
#[cfg(feature = "mlx")]
#[test]
fn logeig_lowering_mlx() {
    let a = spd_matrix();
    let out = run(Device::Mlx, build_graph(&a, true));
    let refy = spectral_ref(&a, true);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("logeig MLX: cos={cos:.9} max_abs={mx:.3e}");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "logeig MLX produced NaN/Inf"
    );
    assert!(cos > 0.9999, "logeig MLX cos={cos}");
}

// ── Batched eigensolver ([B,n,n], one scan for the whole batch) ─────────────

fn batched_mats() -> [Vec<f64>; 3] {
    [spd_matrix(), spd_k(5), spd_k(7)]
}

fn build_batched_graph(mats: &[Vec<f64>; 3], log: bool) -> Graph {
    let (batch, n) = (3usize, N);
    let mut a = vec![0f64; batch * n * n];
    for (b, m) in mats.iter().enumerate() {
        a[b * n * n..(b + 1) * n * n].copy_from_slice(m);
    }
    let mut g = Graph::new("spd_batched");
    let x = const_f32(&mut g, &a, &[batch, n, n]);
    let y = if log {
        g.spd_logeig_batched(x, batch, n, 15, EPS)
    } else {
        g.spd_reeig_batched(x, batch, n, 15, EPS)
    };
    g.set_outputs(vec![y]);
    g
}

fn check_batched(out: &[f32], mats: &[Vec<f64>; 3], log: bool, tag: &str) {
    let n = N;
    assert!(out.iter().all(|v| v.is_finite()), "{tag}: NaN/Inf");
    for (b, m) in mats.iter().enumerate() {
        let refy = if log {
            matfn(m, n, |l| l.max(EPS).ln())
        } else {
            matfn(m, n, |l| l.max(EPS))
        };
        let slice = &out[b * n * n..(b + 1) * n * n];
        let cos = cosine(slice, &refy);
        eprintln!("{tag} b{b}: cos={cos:.9}");
        assert!(cos > 0.99999, "{tag} b{b} cos={cos}");
    }
}

#[test]
fn logeig_batched_cpu() {
    let mats = batched_mats();
    let out = run(Device::Cpu, build_batched_graph(&mats, true));
    check_batched(&out, &mats, true, "batched logeig CPU");
}

#[test]
fn reeig_batched_cpu() {
    let mats = batched_mats();
    let out = run(Device::Cpu, build_batched_graph(&mats, false));
    check_batched(&out, &mats, false, "batched reeig CPU");
}

#[cfg(feature = "metal")]
#[test]
fn logeig_batched_metal() {
    let mats = batched_mats();
    let out = run(Device::Metal, build_batched_graph(&mats, true));
    check_batched(&out, &mats, true, "batched logeig Metal");
}

/// Regression for the batched-matmul broadcast fix: `mm([1,m,k], [B,k,n])` must
/// broadcast the batch-1 lhs across all B batches (it used to over-read → garbage
/// for b>0). Exercises the backend's NATIVE batched matmul directly (not via a
/// scan). `bcast_lhs` picks which operand is broadcast.
fn bmm_broadcast_check(dev: Device, bcast_lhs: bool, tag: &str) {
    if !rlx_runtime::is_available(dev) {
        eprintln!("skip {tag}: {dev:?} unavailable");
        return;
    }
    let (batch, m, k, nn) = (3usize, 2usize, 4usize, 5usize);
    let mut g = Graph::new("bmm_bcast");
    let out;
    let (lhs, rhs);
    if bcast_lhs {
        lhs = (0..m * k)
            .map(|i| (i as f64) * 0.1 - 0.3)
            .collect::<Vec<_>>();
        rhs = (0..batch * k * nn)
            .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.2)
            .collect::<Vec<_>>();
        let l = const_f32(&mut g, &lhs, &[1, m, k]);
        let r = const_f32(&mut g, &rhs, &[batch, k, nn]);
        let y = g.mm(l, r);
        g.set_outputs(vec![y]);
        out = run(dev, g);
    } else {
        lhs = (0..batch * m * k)
            .map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.15)
            .collect::<Vec<_>>();
        rhs = (0..k * nn)
            .map(|i| (i as f64) * 0.07 - 0.2)
            .collect::<Vec<_>>();
        let l = const_f32(&mut g, &lhs, &[batch, m, k]);
        let r = const_f32(&mut g, &rhs, &[1, k, nn]);
        let y = g.mm(l, r);
        g.set_outputs(vec![y]);
        out = run(dev, g);
    }
    for b in 0..batch {
        let refc = if bcast_lhs {
            matmul(&lhs, &rhs[b * k * nn..(b + 1) * k * nn], m, k, nn)
        } else {
            matmul(&lhs[b * m * k..(b + 1) * m * k], &rhs, m, k, nn)
        };
        let slice = &out[b * m * nn..(b + 1) * m * nn];
        let mx = max_abs(slice, &refc);
        eprintln!("{tag} b{b}: max_abs={mx:.3e}");
        assert!(
            slice.iter().all(|v| v.is_finite()) && mx < 1e-4,
            "{tag} b{b} max_abs={mx}"
        );
    }
}

#[test]
fn batched_matmul_broadcast_cpu() {
    bmm_broadcast_check(Device::Cpu, true, "bmm bcast-lhs CPU");
    bmm_broadcast_check(Device::Cpu, false, "bmm bcast-rhs CPU");
}

#[cfg(feature = "metal")]
#[test]
fn batched_matmul_broadcast_metal() {
    bmm_broadcast_check(Device::Metal, true, "bmm bcast-lhs Metal");
    bmm_broadcast_check(Device::Metal, false, "bmm bcast-rhs Metal");
}

#[cfg(feature = "gpu")]
#[test]
fn batched_matmul_broadcast_wgpu() {
    bmm_broadcast_check(Device::Gpu, true, "bmm bcast-lhs wgpu");
    bmm_broadcast_check(Device::Gpu, false, "bmm bcast-rhs wgpu");
}

#[cfg(feature = "mlx")]
#[test]
fn batched_matmul_broadcast_mlx() {
    bmm_broadcast_check(Device::Mlx, true, "bmm bcast-lhs MLX");
    bmm_broadcast_check(Device::Mlx, false, "bmm bcast-rhs MLX");
}

#[cfg(feature = "cuda")]
#[test]
fn batched_matmul_broadcast_cuda() {
    bmm_broadcast_check(Device::Cuda, true, "bmm bcast-lhs CUDA");
    bmm_broadcast_check(Device::Cuda, false, "bmm bcast-rhs CUDA");
}

// ── BiMap (Y = W·X·Wᵀ) — the second SPD forward op ──────────────────────────

const BM_M: usize = 4;

/// Deterministic BiMap weight `[BM_M, N]`.
fn bimap_w() -> Vec<f64> {
    let mut w = vec![0f64; BM_M * N];
    for i in 0..BM_M {
        for j in 0..N {
            w[i * N + j] = (((i * 5 + j * 2 + 1) % 7) as f64 - 3.0) * 0.3;
        }
    }
    w
}

fn build_bimap_graph(w: &[f64], x: &[f64]) -> Graph {
    let mut g = Graph::new("spd_bimap");
    let wn = const_f32(&mut g, w, &[BM_M, N]);
    let xn = const_f32(&mut g, x, &[N, N]);
    let y = g.bimap(wn, xn);
    g.set_outputs(vec![y]);
    g
}

fn bimap_ref(w: &[f64], x: &[f64]) -> Vec<f64> {
    let wx = matmul(w, x, BM_M, N, N); // [m,n]
    let mut wt = vec![0f64; N * BM_M];
    for i in 0..BM_M {
        for j in 0..N {
            wt[j * BM_M + i] = w[i * N + j];
        }
    }
    matmul(&wx, &wt, BM_M, N, BM_M) // [m,m]
}

#[test]
fn bimap_lowering_cpu() {
    let (w, x) = (bimap_w(), spd_matrix());
    let g = LowerSpectral.run(build_bimap_graph(&w, &x));
    assert!(!g.nodes().iter().any(|n| matches!(n.op, Op::BiMap)));
    let out = run(Device::Cpu, g);
    let refy = bimap_ref(&w, &x);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("bimap CPU: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999 && mx < 1e-3, "bimap CPU cos={cos} mx={mx}");
}

#[cfg(feature = "metal")]
#[test]
fn bimap_lowering_metal() {
    let (w, x) = (bimap_w(), spd_matrix());
    let out = run(Device::Metal, build_bimap_graph(&w, &x));
    let refy = bimap_ref(&w, &x);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("bimap Metal: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.99999 && mx < 1e-3, "bimap Metal cos={cos} mx={mx}");
}

// ── SpdBatchNorm transport (Y_i = G^½·M^{-½}·X_i·M^{-½}·G^½) ─────────────────

const BN_BATCH: usize = 3;

/// Deterministic SPD matrix `[N,N]` parameterized by a seed (mean, g, batch).
fn spd_k(seed: usize) -> Vec<f64> {
    let n = N;
    let mut b = vec![0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            b[i * n + k] = (((i * 3 + k * 5 + seed * 2 + 1) % 9) as f64) - 4.0;
        }
    }
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0f64;
            for k in 0..n {
                s += b[i * n + k] * b[j * n + k];
            }
            m[i * n + j] = s;
        }
        m[i * n + i] += (n + seed) as f64;
    }
    m
}

fn build_spdbn_graph(x: &[f64], mean: &[f64], gg: &[f64]) -> Graph {
    let mut g = Graph::new("spd_bn");
    let xn = const_f32(&mut g, x, &[BN_BATCH, N, N]);
    let mn = const_f32(&mut g, mean, &[N, N]);
    let gn = const_f32(&mut g, gg, &[N, N]);
    let y = g.spd_batch_norm_transport(xn, mn, gn, EPS as f32);
    g.set_outputs(vec![y]);
    g
}

fn spdbn_ref(x: &[f64], mean: &[f64], gg: &[f64]) -> Vec<f64> {
    let ms = matfn(mean, N, |l| 1.0 / l.max(EPS).sqrt());
    let gs = matfn(gg, N, |l| l.max(EPS).sqrt());
    let mut out = vec![0f64; BN_BATCH * N * N];
    for bi in 0..BN_BATCH {
        let xi = &x[bi * N * N..(bi + 1) * N * N];
        let c = matmul(&matmul(&ms, xi, N, N, N), &ms, N, N, N);
        let y = matmul(&matmul(&gs, &c, N, N, N), &gs, N, N, N);
        out[bi * N * N..(bi + 1) * N * N].copy_from_slice(&y);
    }
    out
}

fn spdbn_inputs() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mean = spd_k(1);
    let gg = spd_k(2);
    let mut x = vec![0f64; BN_BATCH * N * N];
    for bi in 0..BN_BATCH {
        x[bi * N * N..(bi + 1) * N * N].copy_from_slice(&spd_k(bi + 3));
    }
    (x, mean, gg)
}

#[test]
fn spdbn_lowering_cpu() {
    let (x, mean, gg) = spdbn_inputs();
    let g = LowerSpectral.run(build_spdbn_graph(&x, &mean, &gg));
    assert!(
        !g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SpdBatchNorm { .. }))
    );
    let out = run(Device::Cpu, g);
    let refy = spdbn_ref(&x, &mean, &gg);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("spdbn CPU: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.9999, "spdbn CPU cos={cos} mx={mx}");
}

#[cfg(feature = "metal")]
#[test]
fn spdbn_lowering_metal() {
    let (x, mean, gg) = spdbn_inputs();
    let out = run(Device::Metal, build_spdbn_graph(&x, &mean, &gg));
    let refy = spdbn_ref(&x, &mean, &gg);
    let (cos, mx) = (cosine(&out, &refy), max_abs(&out, &refy));
    eprintln!("spdbn Metal: cos={cos:.9} max_abs={mx:.3e}");
    assert!(cos > 0.9999, "spdbn Metal cos={cos} mx={mx}");
}
