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

//! Parity test: CPU's native SelectiveScan thunk against a scalar
//! reference of the same Mamba SSM recurrence
//!
//!   state_t = exp(δ_t * A) * state_{t-1} + δ_t * B_t * x_t
//!   y_t     = sum_n( C_t * state_t )
//!
//! Same math the autodiff time-loop unfuse implements (see
//! `rlx_opt::autodiff::unfuse_fused_for_autodiff`), so passing here
//! transitively confirms the autodiff path produces correct outputs
//! before any gradient walk runs.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_ssm_graph(b: usize, s: usize, h: usize, n: usize) -> Graph {
    let mut g = Graph::new("ssm");
    let bsh = Shape::new(&[b, s, h], DType::F32);
    let hn = Shape::new(&[h, n], DType::F32);
    let bsn = Shape::new(&[b, s, n], DType::F32);
    let x = g.input("x", bsh.clone());
    let delta = g.input("delta", bsh.clone());
    let a = g.input("a", hn);
    let b_in = g.input("b", bsn.clone());
    let c_in = g.input("c", bsn);
    let y = g.selective_scan(x, delta, a, b_in, c_in, n, bsh);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn cpu_selective_scan_native_matches_recurrence() {
    let (b, s, h, n) = (1, 4, 2, 3);

    // Deterministic-but-non-trivial inputs. Δ in (0, 0.5) so exp(Δ A)
    // stays bounded for negative-leaning A, which is the realistic
    // Mamba regime.
    let nx = b * s * h;
    let nd = b * s * h;
    let na = h * n;
    let nb = b * s * n;
    let xs: Vec<f32> = (0..nx).map(|i| 0.1 + 0.05 * (i as f32)).collect();
    let delta: Vec<f32> = (0..nd).map(|i| 0.1 + 0.02 * (i as f32)).collect();
    let a_data: Vec<f32> = (0..na).map(|i| -0.5 + 0.1 * (i as f32)).collect();
    let b_data: Vec<f32> = (0..nb).map(|i| 0.1 + 0.03 * (i as f32)).collect();
    let c_data: Vec<f32> = (0..nb).map(|i| 0.2 + 0.04 * (i as f32)).collect();

    // Native CPU run: standard compile path uses the SelectiveScan thunk.
    let g_native = build_ssm_graph(b, s, h, n);
    let session = Session::new(Device::Cpu);
    let mut native = session.compile(g_native);
    let native_out = native.run(&[
        ("x", &xs),
        ("delta", &delta),
        ("a", &a_data),
        ("b", &b_data),
        ("c", &c_data),
    ]);

    // Scalar reference. Same recurrence as the thunk and the unfuse.
    let mut want = vec![0f32; b * s * h];
    let mut state = vec![0f32; h * n];
    for bi in 0..b {
        for v in state.iter_mut() {
            *v = 0.0;
        }
        for si in 0..s {
            for ci in 0..h {
                let d = delta[bi * s * h + si * h + ci];
                let xv = xs[bi * s * h + si * h + ci];
                let mut acc = 0.0f32;
                for ni in 0..n {
                    let da = (d * a_data[ci * n + ni]).exp();
                    state[ci * n + ni] =
                        da * state[ci * n + ni] + d * b_data[bi * s * n + si * n + ni] * xv;
                    acc += c_data[bi * s * n + si * n + ni] * state[ci * n + ni];
                }
                want[bi * s * h + si * h + ci] = acc;
            }
        }
    }

    let got = &native_out[0];
    assert_eq!(
        got.len(),
        want.len(),
        "SelectiveScan output length mismatch: got {} want {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let abs_err = (g - w).abs();
        let rel_err = abs_err / (w.abs().max(1e-6));
        assert!(
            abs_err < 1e-5 || rel_err < 1e-5,
            "SelectiveScan parity diverges at idx {i}: native {g} vs scalar reference {w} (abs {abs_err:e}, rel {rel_err:e})"
        );
    }
}
