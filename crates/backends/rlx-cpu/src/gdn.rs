// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gated-DeltaNet BLAS micro-kernels (Tier C.10).

const MAX_STATE: usize = 128;

/// One recurrent timestep using BLAS (n ≤ 128).
#[inline]
pub fn gdn_step_blas(
    s_mat: &mut [f32],
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    g_t: f32,
    beta_t: f32,
    out_row: &mut [f32],
    sk_buf: &mut [f32],
    n: usize,
    scale: f32,
) {
    debug_assert!(n <= MAX_STATE);
    crate::blas::sscal(s_mat, g_t.exp());
    crate::blas::sgemv_at(s_mat, k_row, sk_buf, n, 1.0, 0.0);
    for j in 0..n {
        sk_buf[j] = (v_row[j] - sk_buf[j]) * beta_t;
    }
    crate::blas::sger(s_mat, k_row, sk_buf, n, 1.0);
    crate::blas::sgemv_at(s_mat, q_row, out_row, n, scale, 0.0);
}

/// One recurrent timestep with a **per-channel** log-gate (Kimi-K3 KDA): the
/// state decays per key-row `S[i, j] *= exp(g_row[i])` instead of by a single
/// scalar `exp(g_t)`. `g_row` has length `n` (one log-gate per key channel).
/// Everything else matches [`gdn_step_blas`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_blas_pc(
    s_mat: &mut [f32],
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    g_row: &[f32],
    beta_t: f32,
    out_row: &mut [f32],
    sk_buf: &mut [f32],
    n: usize,
    scale: f32,
) {
    debug_assert!(n <= MAX_STATE);
    debug_assert!(g_row.len() >= n);
    // S[i, j] *= exp(g_row[i]) — decay the key dimension (rows) per channel.
    for i in 0..n {
        let a = g_row[i].exp();
        let row = &mut s_mat[i * n..i * n + n];
        for x in row.iter_mut() {
            *x *= a;
        }
    }
    crate::blas::sgemv_at(s_mat, k_row, sk_buf, n, 1.0, 0.0);
    for j in 0..n {
        sk_buf[j] = (v_row[j] - sk_buf[j]) * beta_t;
    }
    crate::blas::sger(s_mat, k_row, sk_buf, n, 1.0);
    crate::blas::sgemv_at(s_mat, q_row, out_row, n, scale, 0.0);
}

pub const GDN_MAX_STATE: usize = MAX_STATE;

// ── Backward ────────────────────────────────────────────────────────────────

/// Per-head buffers for [`gdn_backward_head`], reused across heads.
///
/// The reverse scan needs the state at every timestep. `s_hist` holds
/// `(seq + 1)` copies of the `[n, n]` state — `s_hist[t]` is the state
/// *entering* timestep `t`, so `s_hist[t + 1]` is `S_t`. That is
/// `O(seq · n²)` per head rather than per `(batch, head)` pair, because heads
/// are processed one at a time; at `n = 128, seq = 24` it is 1.6 MB.
///
/// Reconstructing the states backwards from the final one instead — dividing
/// out `exp(g)` — would avoid the buffer, but the decay is `< 1`, so undoing it
/// amplifies rounding by `∏ 1/exp(g)`, which over a long prompt is unbounded.
pub struct GdnBackwardScratch {
    /// `(seq + 1) · n · n`
    pub s_hist: Vec<f32>,
    /// `seq · n` — `m_t = kᵀ P_t`, kept so the backward need not redo the matvec.
    pub m_hist: Vec<f32>,
    /// Working `[n, n]` buffers.
    pub p_mat: Vec<f32>,
    pub ds_mat: Vec<f32>,
    /// Working `[n]` rows.
    pub u_row: Vec<f32>,
    pub du_row: Vec<f32>,
    pub dm_row: Vec<f32>,
}

impl GdnBackwardScratch {
    pub fn new(seq: usize, n: usize) -> Self {
        Self {
            s_hist: vec![0.0; (seq + 1) * n * n],
            m_hist: vec![0.0; seq * n],
            p_mat: vec![0.0; n * n],
            ds_mat: vec![0.0; n * n],
            u_row: vec![0.0; n],
            du_row: vec![0.0; n],
            dm_row: vec![0.0; n],
        }
    }
}

/// One head's contiguous inputs / outputs for [`gdn_backward_head`].
///
/// Rows are `[seq, n]` (or `[seq]` for a per-head gate and beta); the caller
/// gathers them out of the `[B, S, H, N]` layout so the scan itself indexes
/// contiguously.
pub struct GdnHeadIo<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    /// `[seq · n]` when `gate_per_channel`, else `[seq]`.
    pub g: &'a [f32],
    pub beta: &'a [f32],
    pub dy: &'a [f32],
    /// `[n · n]` initial state, or `None` for a zero start.
    pub init_state: Option<&'a [f32]>,
}

/// Gradient outputs for one head, same row layout as [`GdnHeadIo`].
pub struct GdnHeadGrads<'a> {
    pub dq: &'a mut [f32],
    pub dk: &'a mut [f32],
    pub dv: &'a mut [f32],
    pub dg: &'a mut [f32],
    pub dbeta: &'a mut [f32],
    /// `[n · n]`, written only when the forward carried a state in.
    pub dstate: Option<&'a mut [f32]>,
}

/// Reverse scan of the gated delta-net for one head.
///
/// Forward, with `S` row-major `[key, value]` and `A = exp(g)`:
///
/// ```text
///   P   = A ⊙ S_{t-1}          (per-head A is scalar; per-channel decays row i)
///   m   = kᵀP
///   u   = (v − m) · β
///   S_t = P + k ⊗ u
///   y   = c · qᵀS_t            c = 1/√n
/// ```
///
/// Backward, accumulating `dS` from later timesteps:
///
/// ```text
///   dq  = c · S_t · dy         dS += c · q ⊗ dy
///   dk += dS · u               du  = kᵀ dS
///   dv += β · du               dβ += ⟨v − m, du⟩        dm = −β · du
///   dk += P · dm               dS += k ⊗ dm             (dS is now dP)
///   dg  = A ⊙ ⟨S_{t-1}, dP⟩    dS_{t-1} = A ⊙ dP
/// ```
///
/// `dk` and `dS` each take two contributions, and the second pair must be added
/// only after `du` has been read out of `dS` — the order above is load-bearing.
#[allow(clippy::too_many_arguments)]
pub fn gdn_backward_head(
    io: &GdnHeadIo<'_>,
    grads: &mut GdnHeadGrads<'_>,
    seq: usize,
    n: usize,
    gate_per_channel: bool,
    scratch: &mut GdnBackwardScratch,
) {
    debug_assert!(n <= MAX_STATE);
    let nn = n * n;
    let scale = 1.0f32 / (n as f32).sqrt();

    // ── Forward, recording the state entering each timestep ──
    let (s_hist, m_hist) = (&mut scratch.s_hist, &mut scratch.m_hist);
    match io.init_state {
        Some(init) => s_hist[..nn].copy_from_slice(&init[..nn]),
        None => s_hist[..nn].fill(0.0),
    }
    for t in 0..seq {
        let (prev, cur) = s_hist[t * nn..(t + 2) * nn].split_at_mut(nn);
        cur.copy_from_slice(prev);
        // P = A ⊙ S_{t-1}
        if gate_per_channel {
            let g_row = &io.g[t * n..t * n + n];
            for i in 0..n {
                let a = g_row[i].exp();
                for x in cur[i * n..i * n + n].iter_mut() {
                    *x *= a;
                }
            }
        } else {
            crate::blas::sscal(cur, io.g[t].exp());
        }
        let k_row = &io.k[t * n..t * n + n];
        let v_row = &io.v[t * n..t * n + n];
        let m_row = &mut m_hist[t * n..t * n + n];
        // m = kᵀP
        crate::blas::sgemv_at(cur, k_row, m_row, n, 1.0, 0.0);
        // u = (v − m)·β, then S_t = P + k ⊗ u
        let beta_t = io.beta[t];
        for j in 0..n {
            scratch.u_row[j] = (v_row[j] - m_row[j]) * beta_t;
        }
        crate::blas::sger(cur, k_row, &scratch.u_row[..n], n, 1.0);
    }

    // ── Reverse ──
    let ds = &mut scratch.ds_mat;
    ds.fill(0.0);
    for t in (0..seq).rev() {
        let k_row = &io.k[t * n..t * n + n];
        let v_row = &io.v[t * n..t * n + n];
        let q_row = &io.q[t * n..t * n + n];
        let dy_row = &io.dy[t * n..t * n + n];
        let m_row = &m_hist[t * n..t * n + n];
        let beta_t = io.beta[t];

        let s_prev = &s_hist[t * nn..t * nn + nn];
        let s_cur = &s_hist[(t + 1) * nn..(t + 1) * nn + nn];

        // P = A ⊙ S_{t-1}, recomputed rather than stored.
        let p = &mut scratch.p_mat;
        p.copy_from_slice(s_prev);
        if gate_per_channel {
            let g_row = &io.g[t * n..t * n + n];
            for i in 0..n {
                let a = g_row[i].exp();
                for x in p[i * n..i * n + n].iter_mut() {
                    *x *= a;
                }
            }
        } else {
            crate::blas::sscal(p, io.g[t].exp());
        }

        // y = c·qᵀS_t  ⇒  dq = c·S_t·dy ; dS += c·q ⊗ dy
        crate::blas::sgemv_nn(
            s_cur,
            dy_row,
            &mut grads.dq[t * n..t * n + n],
            n,
            n,
            scale,
            0.0,
        );
        crate::blas::sger(ds, q_row, dy_row, n, scale);

        // S_t = P + k ⊗ u  ⇒  dk += dS·u ; du = kᵀdS   (both read dS as-is)
        for j in 0..n {
            scratch.u_row[j] = (v_row[j] - m_row[j]) * beta_t;
        }
        {
            let dk_row = &mut grads.dk[t * n..t * n + n];
            crate::blas::sgemv_nn(ds, &scratch.u_row[..n], dk_row, n, n, 1.0, 1.0);
        }
        crate::blas::sgemv_at(ds, k_row, &mut scratch.du_row[..n], n, 1.0, 0.0);

        // u = (v − m)·β  ⇒  dv += β·du ; dβ += ⟨v − m, du⟩ ; dm = −β·du
        let mut dbeta_acc = 0.0f32;
        for j in 0..n {
            let du = scratch.du_row[j];
            grads.dv[t * n + j] += beta_t * du;
            dbeta_acc += (v_row[j] - m_row[j]) * du;
            scratch.dm_row[j] = -beta_t * du;
        }
        grads.dbeta[t] += dbeta_acc;

        // m = kᵀP  ⇒  dk += P·dm ; dP += k ⊗ dm  (dS becomes dP here)
        {
            let dk_row = &mut grads.dk[t * n..t * n + n];
            crate::blas::sgemv_nn(p, &scratch.dm_row[..n], dk_row, n, n, 1.0, 1.0);
        }
        crate::blas::sger(ds, k_row, &scratch.dm_row[..n], n, 1.0);

        // P = A ⊙ S_{t-1}  ⇒  dA = ⟨S_{t-1}, dP⟩ ; dS_{t-1} = A ⊙ dP ; dg = dA·A
        if gate_per_channel {
            let g_row = &io.g[t * n..t * n + n];
            for i in 0..n {
                let a = g_row[i].exp();
                let mut acc = 0.0f32;
                let row = i * n;
                for j in 0..n {
                    acc += s_prev[row + j] * ds[row + j];
                    ds[row + j] *= a;
                }
                grads.dg[t * n + i] += acc * a;
            }
        } else {
            let a = io.g[t].exp();
            let mut acc = 0.0f32;
            for idx in 0..nn {
                acc += s_prev[idx] * ds[idx];
                ds[idx] *= a;
            }
            grads.dg[t] += acc * a;
        }
    }

    if let Some(dstate) = grads.dstate.as_deref_mut() {
        dstate[..nn].copy_from_slice(&ds[..nn]);
    }
}

#[cfg(test)]
mod backward_tests {
    use super::*;

    const SEQ: usize = 5;
    const N: usize = 6;

    fn hashed(seed: u64, i: usize) -> f32 {
        let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 32;
        ((x >> 40) as f32) / 8_388_608.0 - 1.0
    }

    struct Case {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        g: Vec<f32>,
        beta: Vec<f32>,
        dy: Vec<f32>,
    }

    fn case(gate_per_channel: bool) -> Case {
        let gl = if gate_per_channel { SEQ * N } else { SEQ };
        Case {
            q: (0..SEQ * N).map(|i| 0.4 * hashed(1, i)).collect(),
            k: (0..SEQ * N).map(|i| 0.4 * hashed(2, i)).collect(),
            v: (0..SEQ * N).map(|i| 0.4 * hashed(3, i)).collect(),
            // Decay must be < 1, so the log-gate is negative.
            g: (0..gl).map(|i| -0.4 + 0.15 * hashed(4, i)).collect(),
            beta: (0..SEQ).map(|i| 0.5 + 0.2 * hashed(5, i)).collect(),
            dy: (0..SEQ * N)
                .map(|i| 0.5 + 0.25 * ((i % 7) as f32))
                .collect(),
        }
    }

    /// Reference forward, straight from the documented recurrence.
    fn forward(c: &Case, gate_per_channel: bool) -> Vec<f32> {
        let mut s = vec![0.0f32; N * N];
        let mut out = vec![0.0f32; SEQ * N];
        let mut sk = vec![0.0f32; N];
        let scale = 1.0f32 / (N as f32).sqrt();
        for t in 0..SEQ {
            let (q, k, v) = (
                &c.q[t * N..t * N + N],
                &c.k[t * N..t * N + N],
                &c.v[t * N..t * N + N],
            );
            let o = &mut out[t * N..t * N + N];
            if gate_per_channel {
                gdn_step_blas_pc(
                    &mut s,
                    q,
                    k,
                    v,
                    &c.g[t * N..t * N + N],
                    c.beta[t],
                    o,
                    &mut sk,
                    N,
                    scale,
                );
            } else {
                gdn_step_blas(&mut s, q, k, v, c.g[t], c.beta[t], o, &mut sk, N, scale);
            }
        }
        out
    }

    /// `Σ dy·y` — the scalar the finite differences probe.
    fn probe(c: &Case, gate_per_channel: bool) -> f64 {
        forward(c, gate_per_channel)
            .iter()
            .zip(&c.dy)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    }

    fn run_backward(
        c: &Case,
        gate_per_channel: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let gl = c.g.len();
        let (mut dq, mut dk, mut dv) = (vec![0.0; SEQ * N], vec![0.0; SEQ * N], vec![0.0; SEQ * N]);
        let (mut dg, mut dbeta) = (vec![0.0; gl], vec![0.0; SEQ]);
        let io = GdnHeadIo {
            q: &c.q,
            k: &c.k,
            v: &c.v,
            g: &c.g,
            beta: &c.beta,
            dy: &c.dy,
            init_state: None,
        };
        let mut grads = GdnHeadGrads {
            dq: &mut dq,
            dk: &mut dk,
            dv: &mut dv,
            dg: &mut dg,
            dbeta: &mut dbeta,
            dstate: None,
        };
        let mut scratch = GdnBackwardScratch::new(SEQ, N);
        gdn_backward_head(&io, &mut grads, SEQ, N, gate_per_channel, &mut scratch);
        (dq, dk, dv, dg, dbeta)
    }

    fn check(gate_per_channel: bool) {
        let c = case(gate_per_channel);
        let (dq, dk, dv, dg, dbeta) = run_backward(&c, gate_per_channel);

        let eps = 1e-3f32;
        let mut worst = 0.0f64;
        // Every input, one coordinate at a time, against central differences.
        let mut check_one =
            |name: &str, analytic: &[f32], pick: &dyn Fn(&mut Case) -> *mut f32, len: usize| {
                for i in 0..len {
                    let mut plus = case(gate_per_channel);
                    let mut minus = case(gate_per_channel);
                    unsafe {
                        *pick(&mut plus).add(i) += eps;
                        *pick(&mut minus).add(i) -= eps;
                    }
                    let fd = (probe(&plus, gate_per_channel) - probe(&minus, gate_per_channel))
                        / (2.0 * eps as f64);
                    let delta = (fd - analytic[i] as f64).abs();
                    worst = worst.max(delta);
                    assert!(
                        delta < 2e-3,
                        "{name}[{i}]: kernel {} vs finite-difference {fd}",
                        analytic[i]
                    );
                }
                let mag = analytic.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(mag > 1e-4, "{name} is ~zero (max {mag}) — check is vacuous");
            };

        check_one("dq", &dq, &|c: &mut Case| c.q.as_mut_ptr(), SEQ * N);
        check_one("dk", &dk, &|c: &mut Case| c.k.as_mut_ptr(), SEQ * N);
        check_one("dv", &dv, &|c: &mut Case| c.v.as_mut_ptr(), SEQ * N);
        check_one("dg", &dg, &|c: &mut Case| c.g.as_mut_ptr(), dg.len());
        check_one("dbeta", &dbeta, &|c: &mut Case| c.beta.as_mut_ptr(), SEQ);
        eprintln!(
            "gate_per_channel={gate_per_channel}: worst |kernel - finite difference| = {worst:.2e}"
        );
    }

    #[test]
    fn per_head_gate_backward_matches_finite_differences() {
        check(false);
    }

    #[test]
    fn per_channel_gate_backward_matches_finite_differences() {
        check(true);
    }
}
