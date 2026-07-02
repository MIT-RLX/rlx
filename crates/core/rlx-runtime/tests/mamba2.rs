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
//! `Op::Mamba2` (SSD scalar-decay SSM) vs an independent recurrent
//! reference. Decomposes via `unfuse`, so this also exercises the
//! MLX/CoreML/TPU and autodiff path.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const H: usize = 2;
const P: usize = 3;
const N: usize = 4;

fn seq(n: usize, m: usize, off: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % m) as f32 - off) * scale).collect()
}

/// Recurrent reference: S[P,N] per (batch, head); see Op::Mamba2.
fn ref_mamba2(x: &[f32], dt: &[f32], a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; B * S * H * P];
    for bi in 0..B {
        for hi in 0..H {
            let mut s = [0f32; P * N];
            for t in 0..S {
                let dt_t = dt[(bi * S + t) * H + hi];
                let da = (dt_t * a[hi]).exp();
                let x_off = ((bi * S + t) * H + hi) * P;
                let bc_off = ((bi * S + t) * H + hi) * N;
                for p in 0..P {
                    let dtx = dt_t * x[x_off + p];
                    for nn in 0..N {
                        s[p * N + nn] = da * s[p * N + nn] + dtx * b[bc_off + nn];
                    }
                }
                for p in 0..P {
                    let mut acc = 0f32;
                    for nn in 0..N {
                        acc += s[p * N + nn] * c[bc_off + nn];
                    }
                    y[x_off + p] = acc;
                }
            }
        }
    }
    y
}

#[test]
fn mamba2_matches_recurrent_reference() {
    let f = DType::F32;
    let x = seq(B * S * H * P, 13, 6.0, 0.06);
    // dt ≥ 0 (caller-softplus'd); keep modest so exp(dt·a) stays sane.
    let dt: Vec<f32> = (0..B * S * H)
        .map(|i| 0.2 + 0.05 * (i % 5) as f32)
        .collect();
    // a < 0 decay per head.
    let a: Vec<f32> = (0..H).map(|i| -0.5 - 0.3 * i as f32).collect();
    let b = seq(B * S * H * N, 11, 5.0, 0.05);
    let c = seq(B * S * H * N, 7, 3.0, 0.05);

    let expected = ref_mamba2(&x, &dt, &a, &b, &c);

    let mut g = Graph::new("mamba2");
    let xi = g.input("x", Shape::new(&[B, S, H, P], f));
    let dti = g.input("dt", Shape::new(&[B, S, H], f));
    let ai = g.input("a", Shape::new(&[H], f));
    let bi = g.input("b", Shape::new(&[B, S, H, N], f));
    let ci = g.input("c", Shape::new(&[B, S, H, N], f));
    let y = g.add_node(
        Op::Mamba2 {
            head_dim: P,
            state_size: N,
        },
        vec![xi, dti, ai, bi, ci],
        Shape::new(&[B, S, H, P], f),
    );
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let actual = compiled
        .run(&[
            ("x", x.as_slice()),
            ("dt", dt.as_slice()),
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("c", c.as_slice()),
        ])
        .pop()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            (actual[i] - expected[i]).abs() < 1e-4,
            "Mamba2 mismatch at {i}: {} vs {}",
            actual[i],
            expected[i]
        );
    }
}
