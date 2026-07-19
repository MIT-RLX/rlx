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

//! Backend-agnostic **symmetric-eigendecomposition spectral layers** built from
//! primitive ops + `Op::Scan` — the GPU-capable lowering for the opaque,
//! CPU-only (f64 LAPACK) `Op::ReEig` / `Op::LogEig` SPD-manifold kernels.
//!
//! SPDNet-family models (`spdnet`, `tensorcspnet`, `graphcspnet`, `tsmnet`) map
//! covariance descriptors through the SPD manifold with ReEig
//! (`Y = U·max(ε,Σ)·Uᵀ`, the SPD ReLU) and LogEig (`Y = U·log(Σ)·Uᵀ`, the tangent
//! projection). The native kernels compute the eigendecomposition with LAPACK in
//! f64 and only exist on CPU. This module expresses the same math as a graph of
//! primitives (`mm`/`add`/`sub`/`mul`/`div`/`sqrt`/`neg`/`narrow`/`concat`/
//! `transpose`/`Activation::Log` + `Op::Scan`) so it lowers to **every backend**.
//!
//! **Eigensolver = graph-based cyclic Jacobi via `Op::Scan`.** A fully-unrolled
//! Jacobi explodes the graph (`sweeps · n²/2` rotations → GPU shader compile is
//! super-linear in node count). Instead `Op::Scan` compiles the one-sweep body
//! **once** and iterates it → the graph stays compact and compiles+runs at
//! `n = 20+`. Each `p<q` rotation applies `J = I + (c−1)(E_pp+E_qq) + s·E_pq −
//! s·E_qp` (basis-const matrices scaled by `[1,1]` scalars) as `A ← Jᵀ·A·J`,
//! `V ← V·J`.
//!
//! **Precision.** Plain f32 is sufficient — an f32 Jacobi eigensolver matches the
//! f64 LAPACK reference to cos = 1.0 even at condition 1e7 for the small SPD
//! matrices these models use, so no f64 / double-single is needed. A **signed
//! denominator floor** (`|den| ≥ 1e-6`, sign preserved) is required in f32 or an
//! off-diagonal `a_pq ≈ 0` overflows `tau` → `NaN`. Constants are emitted in the
//! input node's dtype, so an f64 graph (CPU) stays f64 and an f32 graph (GPU)
//! stays f32.

use crate::infer::GraphExt;
use crate::op::{Activation, Op};
use crate::{DType, Graph, NodeId, Shape};

/// Default cyclic-Jacobi sweep count. Every GPU backend host-falls-back
/// `Op::Scan` (and the opt-in unroll bakes one round-block per sweep), so unlike
/// a native on-device scan each extra sweep is real host compute / graph nodes —
/// margin is NOT free. Measured convergence to cos = 1.0: spdnet/tsmnet/
/// tensorcspnet at ≤6, graphcspnet (spd_dim 36, worst case) at 5 (cos 0.999987 at
/// 4). Cyclic Jacobi converges QUADRATICALLY, so sweeps past the convergence
/// point are near-no-ops on already-diagonal matrices — measured cost is ~linear
/// in sweeps (tensorcspnet/coreml ~8 s/sweep), and cos is already 1.0 at 6 for
/// every SPD crate (0.999995 at 4). 6 = full convergence + 1 sweep of headroom,
/// ~2.5× faster than the old 15 (and ~25% faster than 8). Under-convergence
/// surfaces as a failing parity test, never silent drift, so this is safe to
/// trim. Override with `RLX_SPD_JACOBI_SWEEPS`.
pub const SPD_JACOBI_SWEEPS: u32 = 6;

/// Effective sweep count, honoring the `RLX_SPD_JACOBI_SWEEPS` env override.
pub fn spd_jacobi_sweeps() -> u32 {
    std::env::var("RLX_SPD_JACOBI_SWEEPS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(SPD_JACOBI_SWEEPS)
}

/// Emit `xs` as little-endian bytes in `dtype` (f64 verbatim, everything else as f32).
fn const_bytes(xs: &[f64], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F64 => xs.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => xs.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect(),
    }
}

/// A constant matrix `[dims]` in `dtype`.
fn cmat(g: &mut Graph, xs: &[f64], dims: &[usize], dtype: DType) -> NodeId {
    g.add_node(
        Op::Constant {
            data: const_bytes(xs, dtype),
        },
        vec![],
        Shape::new(dims, dtype),
    )
}

/// A `[1,1]` constant scalar in `dtype`.
fn cscalar(g: &mut Graph, v: f64, dtype: DType) -> NodeId {
    cmat(g, &[v], &[1, 1], dtype)
}

/// Round-robin (circle-method) 1-factorization of `K_ne` (`ne` even): `ne-1`
/// rounds, each a set of `ne/2` **disjoint** `(p,q)` pairs covering all vertices.
/// Over the rounds every `p<q` pair appears exactly once — one full cyclic-Jacobi
/// sweep, but split so each round's rotations act on disjoint index pairs and can
/// be applied **simultaneously** (they commute).
fn round_robin_rounds(ne: usize) -> Vec<Vec<(usize, usize)>> {
    let mut players: Vec<usize> = (0..ne).collect();
    let mut rounds = Vec::with_capacity(ne - 1);
    for _ in 0..ne - 1 {
        let pairs: Vec<(usize, usize)> = (0..ne / 2)
            .map(|i| {
                let (a, b) = (players[i], players[ne - 1 - i]);
                (a.min(b), a.max(b))
            })
            .collect();
        rounds.push(pairs);
        // Circle method: keep players[0] fixed, rotate players[1..] by one.
        let last = players[ne - 1];
        for i in (2..ne).rev() {
            players[i] = players[i - 1];
        }
        players[1] = last;
    }
    rounds
}

/// Diagonal of a `[k,k]` matrix as a `[k,1]` column via `sum(m ⊙ I_k, axis1)`
/// (last-axis reduce → lowers everywhere).
fn diag_k(g: &mut Graph, m: NodeId, ident_k: NodeId) -> NodeId {
    let d = g.mul(m, ident_k);
    g.sum(d, vec![1], true) // [k,1]
}

/// **One parallel round** of `k = ne/2` disjoint Jacobi rotations, encoded by the
/// per-round selection matrices `spp/sqq` (`[n,k]`, column `i` = one-hot of pair
/// `i`'s `p`/`q` row). All `k` rotations are computed from the same `av` and
/// applied as ONE combined orthogonal `J` — the constant-size `Op::Scan` body.
/// Extraction: `a_pp = diag(sppᵀ·av·spp)` etc. (vectorised over the `k` pairs);
/// `J = I + spp·diag(c−1)·sppᵀ + sqq·diag(c−1)·sqqᵀ + spp·diag(s)·sqqᵀ −
/// sqq·diag(s)·sppᵀ`. Numerically robust in f32 via the signed floor + `sbias`.
#[allow(clippy::too_many_arguments)]
fn one_round(
    g: &mut Graph,
    av: NodeId,
    vv: NodeId,
    spp: NodeId,
    sqq: NodeId,
    ident_n: NodeId,
    ident_k: NodeId,
    dt: DType,
) -> (NodeId, NodeId) {
    let one = cscalar(g, 1.0, dt);
    let two = cscalar(g, 2.0, dt);
    let tiny = cscalar(g, 1e-30, dt);
    let sbias = cscalar(g, 1e-12, dt);
    let floor = cscalar(g, 1e-6, dt);

    let sppt = g.transpose_(spp, vec![1, 0]); // [k,n]
    let sqqt = g.transpose_(sqq, vec![1, 0]);
    let tpp = g.mm(sppt, av); // [k,n]
    let m_pp = g.mm(tpp, spp); // [k,k]
    let app = diag_k(g, m_pp, ident_k); // [k,1]
    let tqq = g.mm(sqqt, av);
    let m_qq = g.mm(tqq, sqq);
    let aqq = diag_k(g, m_qq, ident_k);
    let m_pq = g.mm(tpp, sqq); // sppᵀ·av·sqq
    let apq = diag_k(g, m_pq, ident_k);

    // Per-pair c,s (vectorised over the [k,1] columns; scalars broadcast).
    let num = g.sub(aqq, app);
    let den0 = g.mul(two, apq);
    let den0b = g.add(den0, sbias); // sbias breaks exact-zero → sign +1 (no NaN)
    let den0sq = g.mul(den0b, den0b);
    let den0abs = g.sqrt(den0sq);
    let den0abst = g.add(den0abs, tiny);
    let sgnden = g.div(den0b, den0abst);
    let sfloor = g.mul(sgnden, floor);
    let den = g.add(den0, sfloor);
    let tau = g.div(num, den);
    let tau2 = g.mul(tau, tau);
    let atau = g.sqrt(tau2);
    let satau = g.add(atau, tiny);
    let sgn = g.div(tau, satau);
    let tau2p1 = g.add(tau2, one);
    let rt = g.sqrt(tau2p1);
    let tden = g.add(atau, rt);
    let t = g.div(sgn, tden);
    let t2 = g.mul(t, t);
    let t2p1 = g.add(t2, one);
    let rc = g.sqrt(t2p1);
    let c = g.div(one, rc);
    let s = g.mul(t, c);
    let cm1 = g.sub(c, one);
    let negs = g.neg(s);

    // Diagonal [k,k] matrices from the [k,1] vectors, then scatter to [n,n] via
    // the selection matmuls.
    let d_cm1 = g.mul(cm1, ident_k); // [k,k] diag(c−1)
    let d_s = g.mul(s, ident_k);
    let d_negs = g.mul(negs, ident_k);
    let pp_diag = {
        let l = g.mm(spp, d_cm1);
        g.mm(l, sppt)
    };
    let qq_diag = {
        let l = g.mm(sqq, d_cm1);
        g.mm(l, sqqt)
    };
    let pq = {
        let l = g.mm(spp, d_s);
        g.mm(l, sqqt)
    };
    let qp = {
        let l = g.mm(sqq, d_negs);
        g.mm(l, sppt)
    };
    let j1 = g.add(ident_n, pp_diag);
    let j2 = g.add(j1, qq_diag);
    let j3 = g.add(j2, pq);
    let jj = g.add(j3, qp);
    let jt = g.transpose_(jj, vec![1, 0]);
    let jta = g.mm(jt, av);
    let av2 = g.mm(jta, jj);
    let vv2 = g.mm(vv, jj);
    (av2, vv2)
}

/// Constant-size parallel-round scan body: inputs `[carry, spp, sqq]` (carry
/// first, then the 2 per-round `xs`).
fn round_body(n: usize, k: usize, dt: DType) -> Graph {
    let mut body = Graph::new("jacobi_round");
    let carry = body.input("carry", Shape::new(&[2 * n, n], dt));
    let spp = body.input("spp", Shape::new(&[n, k], dt));
    let sqq = body.input("sqq", Shape::new(&[n, k], dt));
    let a_in = body.narrow_(carry, 0, 0, n);
    let v_in = body.narrow_(carry, 0, n, n);
    let mut ident = vec![0f64; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let ident_n = cmat(&mut body, &ident, &[n, n], dt);
    let mut idk = vec![0f64; k * k];
    for i in 0..k {
        idk[i * k + i] = 1.0;
    }
    let ident_k = cmat(&mut body, &idk, &[k, k], dt);
    let (a_out, v_out) = one_round(&mut body, a_in, v_in, spp, sqq, ident_n, ident_k, dt);
    let out = body.concat_(vec![a_out, v_out], 0);
    body.set_outputs(vec![out]);
    body
}

/// Run the **parallel** Jacobi eigensolver on symmetric `a` (`[n,n]`) and return
/// `(av, vv)`: `av`'s diagonal holds the eigenvalues, `vv`'s columns the
/// eigenvectors.
///
/// Each sweep is `ne-1` **rounds** (round-robin 1-factorization, `ne = n` rounded
/// up to even), and each round applies `ne/2` disjoint rotations at once. So one
/// sweep is `ne-1` sequential `Op::Scan` steps instead of `n(n-1)/2` — an ~`n/2`×
/// reduction in the dependent-dispatch count that dominates GPU runtime for large
/// `n`. The body is constant-size (compiles once); per-round selection matrices
/// `spp/sqq` (`[ne-1,n,k]`, one sweep's worth, tiny) drive it. Odd `n` pads to
/// `ne = n+1` with a phantom index whose zero selector columns make it a no-op.
fn eigensolve(g: &mut Graph, a: NodeId, n: usize, sweeps: u32, dt: DType) -> (NodeId, NodeId) {
    let mut ident = vec![0f64; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    let iv = cmat(g, &ident, &[n, n], dt);
    if n < 2 {
        return (a, iv); // 1×1: eigenvalue = a, eigenvector = 1.
    }
    let mut carry = g.concat_(vec![a, iv], 0); // [2n, n] = A over V(=I)

    let ne = if n.is_multiple_of(2) { n } else { n + 1 };
    let k = ne / 2;
    let rounds = round_robin_rounds(ne);
    let nr = rounds.len(); // ne - 1

    // Native UNROLLED path (RLX_SPD_UNROLL=1): apply each round's `one_round`
    // DIRECTLY on the parent graph instead of through `Op::Scan`. Op::Scan has no
    // native GPU kernel — it host-falls-back to the CPU executor, so every one of
    // the `nr·sweeps` steps round-trips GPU→CPU→GPU. That sync dominates runtime
    // for large batch / large `n` (tensorcspnet, graphcspnet n=36). Unrolling keeps
    // the whole eigensolve on-device (bigger graph, but the matmul kernel is reused
    // and there are no per-round host round-trips). Numerically identical to the
    // scan path — it calls the SAME `one_round`. Opt-in so the default small-graph
    // scan is unchanged for very large models.
    //
    // DEFAULT = single-scan (below): it works on ALL backends including CoreML
    // (one host segment instead of `sweeps` separately-compiled MIL segments —
    // tensorcspnet/graphcspnet go from "never finishes on CoreML" to ~3 min), and
    // its wall-time is competitive with the native unroll on GPU (tensorcspnet:
    // wgpu 40 s single-scan vs 59 s unroll). The native UNROLL (no `Op::Scan`,
    // fully on-device) stays available for GPU-only workloads via
    // `RLX_SPD_UNROLL=1`, but it is NOT the default because its ~n²·sweeps node
    // count blows up CoreML's MIL compile (~125k nodes).
    if std::env::var("RLX_SPD_UNROLL").as_deref() == Ok("1") {
        let mut idk = vec![0f64; k * k];
        for i in 0..k {
            idk[i * k + i] = 1.0;
        }
        let ident_k = cmat(g, &idk, &[k, k], dt);
        let sel: Vec<(NodeId, NodeId)> = rounds
            .iter()
            .map(|round| {
                let mut sp = vec![0f64; n * k];
                let mut sq = vec![0f64; n * k];
                for (i, &(p, q)) in round.iter().enumerate() {
                    if p < n {
                        sp[p * k + i] = 1.0;
                    }
                    if q < n {
                        sq[q * k + i] = 1.0;
                    }
                }
                (cmat(g, &sp, &[n, k], dt), cmat(g, &sq, &[n, k], dt))
            })
            .collect();
        // `iv` (the [n,n] identity built above) doubles as the read-only `ident_n`
        // constant and the eigenvector accumulator's initial value.
        let (mut av, mut vv) = (a, iv);
        for _ in 0..sweeps {
            for &(spp_r, sqq_r) in &sel {
                let (a2, v2) = one_round(g, av, vv, spp_r, sqq_r, iv, ident_k, dt);
                av = a2;
                vv = v2;
            }
        }
        return (av, vv);
    }

    // Per-round selection stacks TILED over all sweeps into [nr·sweeps, n, k], so
    // the whole eigensolve is a SINGLE `Op::Scan` (length nr·sweeps) rather than
    // `sweeps` separate scans. This matters most on CoreML: every `Op::Scan`
    // host-splits the graph into its own separately-compiled+loaded MIL segment,
    // so `sweeps` scans → ~sweeps+1 model compiles (minutes for tensorcspnet/
    // graphcspnet); one scan → ~2 segments. It also collapses the per-sweep
    // GPU↔host transfers of the host-fallback scan to a single round-trip on
    // every GPU backend. Numerically identical — same rounds in the same order.
    // Column i one-hots pair i's p/q row (phantom index == n → zero selector →
    // no-op rotation).
    let total = nr * sweeps as usize;
    let mut spp = vec![0f64; total * n * k];
    let mut sqq = vec![0f64; total * n * k];
    for s in 0..sweeps as usize {
        for (r, round) in rounds.iter().enumerate() {
            let base = (s * nr + r) * n * k;
            for (i, &(p, q)) in round.iter().enumerate() {
                if p < n {
                    spp[base + p * k + i] = 1.0;
                }
                if q < n {
                    sqq[base + q * k + i] = 1.0;
                }
            }
        }
    }
    let spp_c = cmat(g, &spp, &[total, n, k], dt);
    let sqq_c = cmat(g, &sqq, &[total, n, k], dt);
    let body = round_body(n, k, dt);
    carry = g.scan_with_xs(carry, &[spp_c, sqq_c], body, total as u32);
    let av = g.narrow_(carry, 0, 0, n);
    let vv = g.narrow_(carry, 0, n, n);
    (av, vv)
}

/// Spectral matrix function applied to the (floored) eigenvalues — the tail
/// after `max(λ, eps)`. Selects which SPD matrix function a spectral layer
/// computes: `Re` = ReEig (`max`), `Log` = LogEig (`log∘max`), `Sqrt` = matrix
/// square root (`√∘max`, `G^{1/2}`), `InvSqrt` = inverse square root
/// (`1/√∘max`, `M^{-1/2}`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpectralFn {
    Re,
    Log,
    Sqrt,
    InvSqrt,
}

/// Diagonal `diag(f(λ_i))` matrix `[n,n]` for the selected [`SpectralFn`].
/// Also returns the per-eigenvalue `[1]` raw-λ parts (for packing the
/// backward-reuse buffer).
fn spectral_fmat(
    g: &mut Graph,
    av: NodeId,
    n: usize,
    eps: f64,
    f: SpectralFn,
    dt: DType,
) -> (NodeId, Vec<NodeId>) {
    let epsn = cscalar(g, eps, dt);
    let half = cscalar(g, 0.5, dt);
    let one = cscalar(g, 1.0, dt);
    let mut terms: Vec<NodeId> = Vec::with_capacity(n);
    let mut lam_parts: Vec<NodeId> = Vec::with_capacity(n);
    for i in 0..n {
        let ri = g.narrow_(av, 0, i, 1); // [1,n]
        let lam = g.narrow_(ri, 1, i, 1); // [1,1]
        lam_parts.push(g.reshape_(lam, vec![1])); // raw eigenvalue λ_i
        // max(λ, eps) = 0.5·((λ+eps) + |λ−eps|)
        let sumv = g.add(lam, epsn);
        let diff = g.sub(lam, epsn);
        let d2 = g.mul(diff, diff);
        let ad = g.sqrt(d2);
        let sad = g.add(sumv, ad);
        let mx = g.mul(half, sad);
        let fl = match f {
            SpectralFn::Re => mx,
            SpectralFn::Log => g.activation(Activation::Log, mx, Shape::new(&[1, 1], dt)),
            SpectralFn::Sqrt => g.sqrt(mx),
            SpectralFn::InvSqrt => {
                let s = g.sqrt(mx);
                g.div(one, s)
            }
        };
        let mut eii = vec![0f64; n * n];
        eii[i * n + i] = 1.0;
        let eii = cmat(g, &eii, &[n, n], dt);
        terms.push(g.mul(fl, eii));
    }
    let mut fmat = terms[0];
    for t in &terms[1..] {
        fmat = g.add(fmat, *t);
    }
    (fmat, lam_parts)
}

/// `V · diag(f(max(λ, eps))) · Vᵀ` of symmetric `a` (`[n,n]`) for an arbitrary
/// spectral matrix function `f` — the general SPD matrix-function builder used to
/// lower ReEig / LogEig / matrix-sqrt (`G^{1/2}`) / inverse-sqrt (`M^{-1/2}`).
pub fn spectral_matfn(
    g: &mut Graph,
    a: NodeId,
    n: usize,
    sweeps: u32,
    eps: f64,
    f: SpectralFn,
) -> NodeId {
    let dt = g.shape(a).dtype();
    let (av, vv) = eigensolve(g, a, n, sweeps, dt);
    let (fmat, _) = spectral_fmat(g, av, n, eps, f, dt);
    let vt = g.transpose_(vv, vec![1, 0]);
    let vf = g.mm(vv, fmat);
    g.mm(vf, vt)
}

// ── Batched eigensolver over `[B,n,n]` ──────────────────────────────────────
//
// The SPD models eigendecompose MANY independent matrices (per batch × channel).
// Running them as separate `Op::Scan`s is slow — every GPU backend host-falls-
// back `Op::Scan` (runs on the CPU host executor), so B separate scans = B×
// single-matrix host loops. Batching them into ONE scan over `[B,n,n]` cuts the
// host-scan iteration count by ~B and turns the per-round work into BLAS-batched
// matmuls. `rlx` `mm` broadcasts leading (batch) dims, so the same parallel
// round runs over the whole batch; per-round selection matrices `[n,k]` are
// shared (broadcast) across the batch. The batched spectral tail is also simpler
// (no per-eigenvalue loop): `diag(av)` = `sum(av ⊙ I, last-axis)` → `[B,n,1]`.

fn ident_mat3(g: &mut Graph, n: usize, dt: DType) -> NodeId {
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    cmat(g, &m, &[1, n, n], dt)
}

/// Shared Jacobi `(c−1, s, −s)` from `(app, aqq, apq)` — broadcast-shape agnostic
/// (`[1,1]` single or `[B,k,1]` batched). Signed floor + `sbias` exact-zero fix.
fn jacobi_cs(
    g: &mut Graph,
    app: NodeId,
    aqq: NodeId,
    apq: NodeId,
    dt: DType,
) -> (NodeId, NodeId, NodeId) {
    let one = cscalar(g, 1.0, dt);
    let two = cscalar(g, 2.0, dt);
    let tiny = cscalar(g, 1e-30, dt);
    let sbias = cscalar(g, 1e-12, dt);
    let floor = cscalar(g, 1e-6, dt);
    let num = g.sub(aqq, app);
    let den0 = g.mul(two, apq);
    let den0b = g.add(den0, sbias);
    let den0sq = g.mul(den0b, den0b);
    let den0abs = g.sqrt(den0sq);
    let den0abst = g.add(den0abs, tiny);
    let sgnden = g.div(den0b, den0abst);
    let sfloor = g.mul(sgnden, floor);
    let den = g.add(den0, sfloor);
    let tau = g.div(num, den);
    let tau2 = g.mul(tau, tau);
    let atau = g.sqrt(tau2);
    let satau = g.add(atau, tiny);
    let sgn = g.div(tau, satau);
    let tau2p1 = g.add(tau2, one);
    let rt = g.sqrt(tau2p1);
    let tden = g.add(atau, rt);
    let t = g.div(sgn, tden);
    let t2 = g.mul(t, t);
    let t2p1 = g.add(t2, one);
    let rc = g.sqrt(t2p1);
    let c = g.div(one, rc);
    let s = g.mul(t, c);
    let cm1 = g.sub(c, one);
    let negs = g.neg(s);
    (cm1, s, negs)
}

/// One parallel round over a BATCH `av/vv: [B,n,n]`. `spp/sqq: [n,k]` are shared
/// (broadcast) across the batch; `ident_n:[n,n]`, `ident_k:[k,k]`.
#[allow(clippy::too_many_arguments)]
fn one_round_b(
    g: &mut Graph,
    av: NodeId,
    vv: NodeId,
    spp: NodeId,
    sqq: NodeId,
    ident_n: NodeId,
    ident_k: NodeId,
    dt: DType,
) -> (NodeId, NodeId) {
    // spp/sqq/ident_n/ident_k are rank-3 `[1,·,·]` (batch-broadcast against `av`).
    let sppt = g.transpose_(spp, vec![0, 2, 1]); // [1,k,n]
    let sqqt = g.transpose_(sqq, vec![0, 2, 1]);
    let tpp = g.mm(sppt, av); // [B,k,n]
    let m_pp = g.mm(tpp, spp); // [B,k,k]
    let app = {
        let d = g.mul(m_pp, ident_k);
        g.sum(d, vec![2], true)
    }; // [B,k,1]
    let tqq = g.mm(sqqt, av);
    let m_qq = g.mm(tqq, sqq);
    let aqq = {
        let d = g.mul(m_qq, ident_k);
        g.sum(d, vec![2], true)
    };
    let m_pq = g.mm(tpp, sqq);
    let apq = {
        let d = g.mul(m_pq, ident_k);
        g.sum(d, vec![2], true)
    };
    let (cm1, s, negs) = jacobi_cs(g, app, aqq, apq, dt);
    let d_cm1 = g.mul(cm1, ident_k); // [B,k,k] diag
    let d_s = g.mul(s, ident_k);
    let d_negs = g.mul(negs, ident_k);
    let pp_diag = {
        let l = g.mm(spp, d_cm1);
        g.mm(l, sppt)
    };
    let qq_diag = {
        let l = g.mm(sqq, d_cm1);
        g.mm(l, sqqt)
    };
    let pq = {
        let l = g.mm(spp, d_s);
        g.mm(l, sqqt)
    };
    let qp = {
        let l = g.mm(sqq, d_negs);
        g.mm(l, sppt)
    };
    let j1 = g.add(ident_n, pp_diag);
    let j2 = g.add(j1, qq_diag);
    let j3 = g.add(j2, pq);
    let jj = g.add(j3, qp);
    let jt = g.transpose_(jj, vec![0, 2, 1]); // batched transpose
    let jta = g.mm(jt, av);
    let av2 = g.mm(jta, jj);
    let vv2 = g.mm(vv, jj);
    (av2, vv2)
}

/// Constant-size batched round scan body: inputs `[carry [B,2n,n], spp, sqq]`.
fn round_body_b(batch: usize, n: usize, k: usize, dt: DType) -> Graph {
    let mut body = Graph::new("jacobi_round_b");
    let carry = body.input("carry", Shape::new(&[batch, 2 * n, n], dt));
    let spp = body.input("spp", Shape::new(&[n, k], dt));
    let sqq = body.input("sqq", Shape::new(&[n, k], dt));
    let a_in = body.narrow_(carry, 1, 0, n); // [B,n,n]
    let v_in = body.narrow_(carry, 1, n, n);
    // Per-step selectors + identities are rank-3 `[1,·,·]`, broadcast across the
    // batch by the (now broadcast-aware) batched matmul.
    let _ = batch;
    let spp3 = body.reshape_(spp, vec![1, n as i64, k as i64]);
    let sqq3 = body.reshape_(sqq, vec![1, n as i64, k as i64]);
    let ident_n = ident_mat3(&mut body, n, dt);
    let ident_k = ident_mat3(&mut body, k, dt);
    let (a_out, v_out) = one_round_b(&mut body, a_in, v_in, spp3, sqq3, ident_n, ident_k, dt);
    let out = body.concat_(vec![a_out, v_out], 1); // [B,2n,n]
    body.set_outputs(vec![out]);
    body
}

/// Batched parallel-Jacobi eigensolver: `a: [B,n,n]` → `(av, vv): [B,n,n]`.
fn eigensolve_b(
    g: &mut Graph,
    a: NodeId,
    batch: usize,
    n: usize,
    sweeps: u32,
    dt: DType,
) -> (NodeId, NodeId) {
    // iv = identity per batch, baked as a full [B,n,n] constant (avoid a 1-vs-B
    // broadcast — rlx batched matmul/broadcast mishandles a batch-1 operand).
    let mut ivdata = vec![0f64; batch * n * n];
    for b in 0..batch {
        for i in 0..n {
            ivdata[b * n * n + i * n + i] = 1.0;
        }
    }
    let iv = cmat(g, &ivdata, &[batch, n, n], dt);
    if n < 2 {
        return (a, iv);
    }
    let mut carry = g.concat_(vec![a, iv], 1); // [B, 2n, n]

    let ne = if n.is_multiple_of(2) { n } else { n + 1 };
    let k = ne / 2;
    let rounds = round_robin_rounds(ne);
    let nr = rounds.len();
    let mut spp = vec![0f64; nr * n * k];
    let mut sqq = vec![0f64; nr * n * k];
    for (r, round) in rounds.iter().enumerate() {
        for (i, &(p, q)) in round.iter().enumerate() {
            if p < n {
                spp[r * n * k + p * k + i] = 1.0;
            }
            if q < n {
                sqq[r * n * k + q * k + i] = 1.0;
            }
        }
    }
    let spp_c = cmat(g, &spp, &[nr, n, k], dt);
    let sqq_c = cmat(g, &sqq, &[nr, n, k], dt);
    let xs = [spp_c, sqq_c];
    for _ in 0..sweeps {
        let body = round_body_b(batch, n, k, dt);
        carry = g.scan_with_xs(carry, &xs, body, nr as u32);
    }
    let av = g.narrow_(carry, 1, 0, n);
    let vv = g.narrow_(carry, 1, n, n);
    (av, vv)
}

/// `V·diag(f(max(λ,eps)))·Vᵀ` over a BATCH `a: [B,n,n]` — the batched analogue of
/// [`spectral_matfn`]. One scan for the whole batch → far fewer host-scan loops.
pub fn spectral_matfn_batched(
    g: &mut Graph,
    a: NodeId,
    batch: usize,
    n: usize,
    sweeps: u32,
    eps: f64,
    f: SpectralFn,
) -> NodeId {
    let dt = g.shape(a).dtype();
    let ident_n = ident_mat3(g, n, dt); // [1,n,n], broadcast in the elementwise muls
    let (av, vv) = eigensolve_b(g, a, batch, n, sweeps, dt);
    // diag(av) → [B,n,1]; apply f(max(·,eps)); rebuild diag [B,n,n].
    let epsn = cscalar(g, eps, dt);
    let half = cscalar(g, 0.5, dt);
    let one = cscalar(g, 1.0, dt);
    let diag_av = {
        let d = g.mul(av, ident_n);
        g.sum(d, vec![2], true)
    }; // [B,n,1]
    let sumv = g.add(diag_av, epsn);
    let diff = g.sub(diag_av, epsn);
    let d2 = g.mul(diff, diff);
    let ad = g.sqrt(d2);
    let sad = g.add(sumv, ad);
    let mx = g.mul(half, sad); // max(λ,eps) [B,n,1]
    let fl = match f {
        SpectralFn::Re => mx,
        SpectralFn::Log => g.activation(Activation::Log, mx, Shape::new(&[batch, n, 1], dt)),
        SpectralFn::Sqrt => g.sqrt(mx),
        SpectralFn::InvSqrt => {
            let s = g.sqrt(mx);
            g.div(one, s)
        }
    };
    let fmat = g.mul(fl, ident_n); // [B,n,n] diag(f)
    let vt = g.transpose_(vv, vec![0, 2, 1]);
    let vf = g.mm(vv, fmat);
    g.mm(vf, vt)
}

impl Graph {
    /// Batched graph-primitive **ReEig** over `x: [B,n,n]` — GPU-capable, one scan
    /// for the whole batch. See [`spectral_matfn_batched`].
    pub fn spd_reeig_batched(
        &mut self,
        x: NodeId,
        batch: usize,
        n: usize,
        sweeps: u32,
        eps: f64,
    ) -> NodeId {
        spectral_matfn_batched(self, x, batch, n, sweeps, eps, SpectralFn::Re)
    }

    /// Batched graph-primitive **LogEig** over `x: [B,n,n]`.
    pub fn spd_logeig_batched(
        &mut self,
        x: NodeId,
        batch: usize,
        n: usize,
        sweeps: u32,
        eps: f64,
    ) -> NodeId {
        spectral_matfn_batched(self, x, batch, n, sweeps, eps, SpectralFn::Log)
    }
}

/// BiMap SPDNet layer `Y = W · X · Wᵀ` as two matmuls — the GPU-portable
/// replacement for `Op::BiMap`. `w` is `[m,n]`, `x` is `[n,n]`; output `[m,m]`.
pub fn bimap(g: &mut Graph, w: NodeId, x: NodeId) -> NodeId {
    let wt = g.transpose_(w, vec![1, 0]);
    let wx = g.mm(w, x);
    g.mm(wx, wt)
}

/// SPD batch-norm transport `Y_i = G^{1/2} (M^{-1/2} X_i M^{-1/2}) G^{1/2}` as
/// graph primitives — the GPU-portable replacement for `Op::SpdBatchNorm`.
/// `x` is `[batch,n,n]`, `mean` and `g_` are `[n,n]`; output matches `x`.
pub fn spd_batch_norm_transport(
    g: &mut Graph,
    x: NodeId,
    mean: NodeId,
    g_: NodeId,
    n: usize,
    batch: usize,
    sweeps: u32,
    eps: f64,
) -> NodeId {
    let ms = spectral_matfn(g, mean, n, sweeps, eps, SpectralFn::InvSqrt); // M^{-1/2}
    let gs = spectral_matfn(g, g_, n, sweeps, eps, SpectralFn::Sqrt); // G^{1/2}
    let mut slices = Vec::with_capacity(batch);
    for bi in 0..batch {
        let xi = g.narrow_(x, 0, bi, 1);
        let xi = g.reshape_(xi, vec![n as i64, n as i64]);
        let mx = g.mm(ms, xi); // Ms Xi
        let ci = g.mm(mx, ms); // Ms Xi Ms
        let gc = g.mm(gs, ci); // Gs Ci
        let yi = g.mm(gc, gs); // Gs Ci Gs
        slices.push(g.reshape_(yi, vec![1, n as i64, n as i64]));
    }
    g.concat_(slices, 0)
}

/// `V · diag(max(λ, eps)) · Vᵀ` — the **ReEig** eigenvalue-rectification SPD
/// layer (`[n,n]` → `[n,n]`). GPU-portable replacement for `Op::ReEig`.
pub fn spectral_reeig(g: &mut Graph, a: NodeId, n: usize, sweeps: u32, eps: f64) -> NodeId {
    spectral_matfn(g, a, n, sweeps, eps, SpectralFn::Re)
}

/// `V · diag(log max(λ, eps)) · Vᵀ` — the **LogEig** matrix-log / tangent-space
/// SPD layer (`[n,n]` → `[n,n]`). GPU-portable replacement for `Op::LogEig`.
pub fn spectral_logeig(g: &mut Graph, a: NodeId, n: usize, sweeps: u32, eps: f64) -> NodeId {
    spectral_matfn(g, a, n, sweeps, eps, SpectralFn::Log)
}

/// Packed `[2n²+n]` buffer `Y ∥ λ ∥ U` matching the native `Op::ReEig` /
/// `Op::LogEig` output layout (so the downstream `Narrow(0,0,n²)` that the
/// manifold builder emits selects `Y` unchanged). `log` picks LogEig vs ReEig.
/// This is what the [`Op::ReEig`]/[`Op::LogEig`] → primitive lowering emits.
pub fn spectral_packed(
    g: &mut Graph,
    a: NodeId,
    n: usize,
    sweeps: u32,
    eps: f64,
    log: bool,
) -> NodeId {
    let dt = g.shape(a).dtype();
    let (av, vv) = eigensolve(g, a, n, sweeps, dt);
    let f = if log { SpectralFn::Log } else { SpectralFn::Re };
    let (fmat, lam_parts) = spectral_fmat(g, av, n, eps, f, dt);
    let vt = g.transpose_(vv, vec![1, 0]);
    let vf = g.mm(vv, fmat);
    let y = g.mm(vf, vt); // [n,n]
    let y_flat = g.reshape_(y, vec![(n * n) as i64]); // Y  [n²]
    let lam = g.concat_(lam_parts, 0); // λ  [n]
    let u_flat = g.reshape_(vv, vec![(n * n) as i64]); // U  [n²]
    g.concat_(vec![y_flat, lam, u_flat], 0) // [2n²+n]
}

impl Graph {
    /// Graph-primitive **ReEig** (`V·diag(max(λ,eps))·Vᵀ`) via the Jacobi
    /// eigensolver — a GPU-capable drop-in for [`Graph::reeig`]. `x` is `[n,n]`.
    pub fn spectral_reeig(&mut self, x: NodeId, sweeps: u32, eps: f64) -> NodeId {
        let n = self.shape(x).dim(0).unwrap_static();
        spectral_reeig(self, x, n, sweeps, eps)
    }

    /// Graph-primitive **LogEig** (`V·diag(log max(λ,eps))·Vᵀ`) via the Jacobi
    /// eigensolver — a GPU-capable drop-in for [`Graph::logeig`]. `x` is `[n,n]`.
    pub fn spectral_logeig(&mut self, x: NodeId, sweeps: u32, eps: f64) -> NodeId {
        let n = self.shape(x).dim(0).unwrap_static();
        spectral_logeig(self, x, n, sweeps, eps)
    }
}
