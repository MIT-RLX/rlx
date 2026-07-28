// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs Metal LayerNorm parity (EEG-DINO encoder block shape).

#![cfg(all(feature = "metal", target_os = "macos"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_layernorm_encoder_rows() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let rows = 191usize;
    let h = 200usize;
    let x: Vec<f32> = (0..rows * h)
        .map(|i| (i as f32 * 0.017).sin() * 0.2)
        .collect();
    let gamma: Vec<f32> = (0..h).map(|i| 1.0 + 0.001 * i as f32).collect();
    let beta: Vec<f32> = (0..h).map(|i| -0.0005 * i as f32).collect();

    let mut g = Graph::new("ln");
    let x_in = g.input("x", Shape::new(&[rows, h], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[h], DType::F32));
    let b_p = g.param("beta", Shape::new(&[h], DType::F32));
    let y = g.layer_norm(x_in, g_p, b_p, -1, 1e-5, Shape::new(&[rows, h], DType::F32));
    g.set_outputs(vec![y]);

    let opts = rlx_runtime::CompileOptions::default();
    let mut cpu_sess = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    cpu_sess.set_param("gamma", &gamma);
    cpu_sess.set_param("beta", &beta);
    let cpu = cpu_sess.run(&[("x", &x)]).remove(0);

    let mut metal_sess = Session::new(Device::Metal).compile_with(g, &opts);
    metal_sess.set_param("gamma", &gamma);
    metal_sess.set_param("beta", &beta);
    let metal = metal_sess.run(&[("x", &x)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("layernorm [{rows},{h}] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "Metal LayerNorm max_abs={max_abs}");
}

#[test]
fn metal_layernorm_small() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let rows = 4usize;
    let h = 8usize;
    let x: Vec<f32> = (0..rows * h).map(|i| i as f32 * 0.1 - 2.0).collect();
    let gamma = vec![1.0; h];
    let beta = vec![0.0; h];
    let mut g = Graph::new("ln_s");
    let x_in = g.input("x", Shape::new(&[rows, h], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[h], DType::F32));
    let b_p = g.param("beta", Shape::new(&[h], DType::F32));
    let y = g.layer_norm(x_in, g_p, b_p, -1, 1e-5, Shape::new(&[rows, h], DType::F32));
    g.set_outputs(vec![y]);
    let opts = rlx_runtime::CompileOptions::default();
    let mut cpu_sess = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    cpu_sess.set_param("gamma", &gamma);
    cpu_sess.set_param("beta", &beta);
    let cpu = cpu_sess.run(&[("x", &x)]).remove(0);
    let mut metal_sess = Session::new(Device::Metal).compile_with(g, &opts);
    metal_sess.set_param("gamma", &gamma);
    metal_sess.set_param("beta", &beta);
    let metal = metal_sess.run(&[("x", &x)]).remove(0);
    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("layernorm small [{rows},{h}] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4);
}
