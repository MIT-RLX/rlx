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

//! Metal vs CPU parity for LayerNorm / GroupNorm backward.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol || (x.is_nan() && y.is_nan()))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn metal_layer_norm_backward_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (rows, h, eps) = (4usize, 16usize, 1e-5f32);
    let mut g = Graph::new("ln_bwd");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.layer_norm_backward_input(x, gamma, dy, -1, eps);
    let dgamma = g.layer_norm_backward_gamma(x, dy, Shape::new(&[h], DType::F32), -1, eps);
    g.set_outputs(vec![dx, dgamma]);

    let xv: Vec<f32> = (0..rows * h)
        .map(|i| (((i + 1) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let gv: Vec<f32> = (0..h)
        .map(|i| 0.8 + (((i + 3) % 11) as f32) * 0.05)
        .collect();
    let dyv: Vec<f32> = (0..rows * h)
        .map(|i| (((i + 7) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let feed = [
        ("x", xv.as_slice()),
        ("gamma", gv.as_slice()),
        ("dy", dyv.as_slice()),
    ];

    let want: Vec<f32> = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&feed)
        .into_iter()
        .flatten()
        .collect();
    let got: Vec<f32> = Session::new(Device::Metal)
        .compile(g)
        .run(&feed)
        .into_iter()
        .flatten()
        .collect();
    let err = max_abs(&got, &want);
    assert!(
        close(&got, &want, 1e-4),
        "LayerNormBackward mismatch max_abs={err}:\n got={got:?}\nwant={want:?}"
    );
}

#[test]
fn metal_group_norm_backward_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (n, c, h, w, groups) = (1usize, 8usize, 4usize, 4usize, 2usize);
    let mut g = Graph::new("gn_bwd");
    let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let gamma = g.input("gamma", Shape::new(&[c], DType::F32));
    let beta = g.input("beta", Shape::new(&[c], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c, h, w], DType::F32));
    let dx = g.group_norm_backward_input(x, gamma, beta, dy, groups, 1e-5);
    let dgamma = g.group_norm_backward_gamma(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    let dbeta = g.group_norm_backward_beta(x, dy, Shape::new(&[c], DType::F32), groups, 1e-5);
    g.set_outputs(vec![dx, dgamma, dbeta]);

    let xv: Vec<f32> = (0..n * c * h * w)
        .map(|i| (((i + 1) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let gv: Vec<f32> = (0..c)
        .map(|i| (((i + 3) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let bv: Vec<f32> = (0..c)
        .map(|i| (((i + 5) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let dyv: Vec<f32> = (0..n * c * h * w)
        .map(|i| (((i + 7) % 19) as f32 - 9.0) * 0.07)
        .collect();
    let feed = [
        ("x", xv.as_slice()),
        ("gamma", gv.as_slice()),
        ("beta", bv.as_slice()),
        ("dy", dyv.as_slice()),
    ];

    let want: Vec<f32> = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&feed)
        .into_iter()
        .flatten()
        .collect();
    let got: Vec<f32> = Session::new(Device::Metal)
        .compile(g)
        .run(&feed)
        .into_iter()
        .flatten()
        .collect();
    let err = max_abs(&got, &want);
    assert!(
        close(&got, &want, 1e-4),
        "GroupNormBackward mismatch max_abs={err}:\n got={got:?}\nwant={want:?}"
    );
}
