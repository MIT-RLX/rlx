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

//! Parity test for `Op::GatedDeltaNet` — the linear-attention scan
//! used by the Qwen3.5/3.6 trunk. Compares the CPU thunk against a
//! straight Rust transcription of the autoregressive recurrence
//! from `llama.cpp / src/models/delta-net-base.cpp`
//! (`build_delta_net_autoregressive`).
//!
//! Specifically we verify the same math llama.cpp executes:
//!
//! ```text
//!   q[t] *= 1 / sqrt(n_state)
//!   S[h] *= exp(g[t,h])                       # scalar gate
//!   sk[h,j]   = Σ_i S[h, i, j] * k[t,h,i]
//!   d[h,j]    = (v[t,h,j] - sk[h,j]) * beta[t,h]
//!   S[h,i,j] += k[t,h,i] * d[h,j]             # outer-product accum
//!   o[t,h,j]  = Σ_i S[h, i, j] * q[t,h,i]
//! ```

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_gdn_graph(b: usize, s: usize, h: usize, n: usize) -> Graph {
    let mut g = Graph::new("gdn");
    let bshn = Shape::new(&[b, s, h, n], DType::F32);
    let bsh = Shape::new(&[b, s, h], DType::F32);
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let g_in = g.input("g", bsh.clone());
    let beta = g.input("beta", bsh);
    let y = g.gated_delta_net(q, k, v, g_in, beta, n, bshn);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn cpu_gated_delta_net_matches_reference_recurrence() {
    let (b, s, h, n) = (1, 4, 2, 3);

    // Deterministic-but-non-trivial inputs. g is small-negative so
    // exp(g) stays in (0,1) (the realistic "leaky" gate regime).
    let nqkv = b * s * h * n;
    let ngb = b * s * h;
    let q_data: Vec<f32> = (0..nqkv).map(|i| 0.05 + 0.03 * (i as f32)).collect();
    let k_data: Vec<f32> = (0..nqkv).map(|i| 0.10 + 0.02 * (i as f32)).collect();
    let v_data: Vec<f32> = (0..nqkv).map(|i| 0.30 + 0.05 * (i as f32)).collect();
    let g_data: Vec<f32> = (0..ngb).map(|i| -0.20 - 0.01 * (i as f32)).collect();
    let beta_data: Vec<f32> = (0..ngb).map(|i| 0.40 + 0.02 * (i as f32)).collect();

    let g_native = build_gdn_graph(b, s, h, n);
    let session = Session::new(Device::Cpu);
    let mut native = session.compile(g_native);
    let native_out = native.run(&[
        ("q", &q_data),
        ("k", &k_data),
        ("v", &v_data),
        ("g", &g_data),
        ("beta", &beta_data),
    ]);

    // Scalar reference recurrence — straight transcription of
    // delta-net-base.cpp build_delta_net_autoregressive into
    // row-major Rust.
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut want = vec![0f32; nqkv];
    // state[h, i, j]
    let mut state = vec![0f32; h * n * n];
    let mut sk = vec![0f32; n];

    for bi in 0..b {
        for st in state.iter_mut() {
            *st = 0.0;
        }
        for ti in 0..s {
            let step_qkv = bi * s * h * n + ti * h * n;
            let step_gb = bi * s * h + ti * h;
            for hi in 0..h {
                let q_row = &q_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let k_row = &k_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let v_row = &v_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let g_t = g_data[step_gb + hi];
                let beta_t = beta_data[step_gb + hi];

                let s_base = hi * n * n;
                let s_mat = &mut state[s_base..s_base + n * n];

                // gate
                let g_exp = g_t.exp();
                for v in s_mat.iter_mut() {
                    *v *= g_exp;
                }
                // sk[j]
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * k_row[i];
                    }
                    sk[j] = acc;
                }
                // d = (v - sk) * beta
                for j in 0..n {
                    sk[j] = (v_row[j] - sk[j]) * beta_t;
                }
                // S += outer(k, d)
                for i in 0..n {
                    for j in 0..n {
                        s_mat[i * n + j] += k_row[i] * sk[j];
                    }
                }
                // o[j] = scale * Σ_i S[i, j] * q[i]
                let out_row = &mut want[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * q_row[i];
                    }
                    out_row[j] = acc * scale;
                }
            }
        }
    }

    let got = &native_out[0];
    assert_eq!(
        got.len(),
        want.len(),
        "GatedDeltaNet output length mismatch: got {} want {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let abs_err = (g - w).abs();
        let rel_err = abs_err / (w.abs().max(1e-6));
        assert!(
            abs_err < 1e-5 || rel_err < 1e-5,
            "GatedDeltaNet parity diverges at idx {i}: native {g} vs reference {w} \
             (abs {abs_err:e}, rel {rel_err:e})"
        );
    }
}

/// State must reset between batches. Re-run with batch=2 where the
/// second batch is a copy of the first; per-batch outputs must
/// match.
#[test]
fn cpu_gated_delta_net_resets_state_between_batches() {
    let (h, n, s) = (2, 3, 4);

    let nqkv_1 = s * h * n;
    let ngb_1 = s * h;
    let q1: Vec<f32> = (0..nqkv_1).map(|i| 0.05 + 0.03 * (i as f32)).collect();
    let k1: Vec<f32> = (0..nqkv_1).map(|i| 0.10 + 0.02 * (i as f32)).collect();
    let v1: Vec<f32> = (0..nqkv_1).map(|i| 0.30 + 0.05 * (i as f32)).collect();
    let g1: Vec<f32> = (0..ngb_1).map(|i| -0.20 - 0.01 * (i as f32)).collect();
    let b1: Vec<f32> = (0..ngb_1).map(|i| 0.40 + 0.02 * (i as f32)).collect();

    let session = Session::new(Device::Cpu);

    // Single-batch reference run.
    let g1g = build_gdn_graph(1, s, h, n);
    let mut single = session.compile(g1g);
    let single_out = single.run(&[
        ("q", &q1),
        ("k", &k1),
        ("v", &v1),
        ("g", &g1),
        ("beta", &b1),
    ]);

    // Two-batch run, both batches identical to the single-batch input.
    let mut q2 = q1.clone();
    q2.extend_from_slice(&q1);
    let mut k2 = k1.clone();
    k2.extend_from_slice(&k1);
    let mut v2 = v1.clone();
    v2.extend_from_slice(&v1);
    let mut gd2 = g1.clone();
    gd2.extend_from_slice(&g1);
    let mut bd2 = b1.clone();
    bd2.extend_from_slice(&b1);

    let g2g = build_gdn_graph(2, s, h, n);
    let mut multi = session.compile(g2g);
    let multi_out = multi.run(&[
        ("q", &q2),
        ("k", &k2),
        ("v", &v2),
        ("g", &gd2),
        ("beta", &bd2),
    ]);

    let single_y = &single_out[0];
    let multi_y = &multi_out[0];
    assert_eq!(multi_y.len(), 2 * single_y.len());

    for (i, (m0, s0)) in multi_y[..nqkv_1].iter().zip(single_y.iter()).enumerate() {
        assert!(
            (m0 - s0).abs() < 1e-6,
            "batch0 row {i}: multi={m0} single={s0}"
        );
    }
    for (i, (m1, s0)) in multi_y[nqkv_1..].iter().zip(single_y.iter()).enumerate() {
        assert!(
            (m1 - s0).abs() < 1e-6,
            "batch1 row {i} (should equal batch0): multi={m1} single={s0}"
        );
    }
}
