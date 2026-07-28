// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs Metal parity for NCHW GroupNorm (EEG-DINO patch-embed shape).

#![cfg(all(feature = "metal", target_os = "macos"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_group_norm_eegdino_patch_shape() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // After conv1 in patch embed: [1, 25, 190, 8], groups=5
    let n = 1usize;
    let c = 25usize;
    let h = 190usize;
    let w = 8usize;
    let num_groups = 5usize;
    let eps = 1e-5f32;
    let x: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32 * 0.013).sin() * 0.4)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + 0.01 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.005 * i as f32).collect();

    let mut g = Graph::new("gn");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[c], DType::F32));
    let b_p = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x_in, g_p, b_p, num_groups, eps);
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
    eprintln!("group_norm [1,25,190,8] g=5 max_abs={max_abs:.6}");
    assert!(
        max_abs < 1e-4,
        "Metal GroupNorm diverges from CPU: max_abs={max_abs}"
    );
}

#[test]
fn metal_group_norm_tiny() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let (n, c, h, w, num_groups) = (1, 4, 2, 2, 2);
    let x: Vec<f32> = vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ];
    let gamma = vec![1.; c];
    let beta = vec![0.; c];
    let mut g = Graph::new("gn_tiny");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[c], DType::F32));
    let b_p = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x_in, g_p, b_p, num_groups, 1e-5);
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
    eprintln!("tiny group_norm max_abs={max_abs:.6}");
    for (i, (a, b)) in cpu.iter().zip(metal.iter()).enumerate() {
        let d = (a - b).abs();
        if d > 1e-4 {
            eprintln!("  diff[{i}] cpu={a} metal={b}");
        }
    }
    assert!(max_abs < 1e-4, "tiny gn max_abs={max_abs}");
}
