// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs Metal parity for Conv2d (EEG-DINO patch conv1).

#![cfg(all(feature = "metal", target_os = "macos"))]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn metal_conv2d_patch_conv1() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let n = 1usize;
    let c_in = 1usize;
    let c_out = 25usize;
    let h = 190usize;
    let w = 72usize; // padded width after concat
    let x: Vec<f32> = (0..n * c_in * h * w)
        .map(|i| (i as f32 * 0.011).sin() * 0.3)
        .collect();
    let weight: Vec<f32> = (0..c_out * c_in * 49)
        .map(|i| (i as f32 * 0.02).cos() * 0.1)
        .collect();
    let mut g = Graph::new("conv1");
    let x_in = g.input("x", Shape::new(&[n, c_in, h, w], DType::F32));
    let w_p = g.param("w", Shape::new(&[c_out, c_in, 1, 49], DType::F32));
    let y = g.conv2d(x_in, w_p, [1, 49], [1, 25], [0, 0], [1, 1], 1);
    g.set_outputs(vec![y]);

    let opts = rlx_runtime::CompileOptions::default();
    let mut cpu_sess = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    cpu_sess.set_param("w", &weight);
    let cpu = cpu_sess.run(&[("x", &x)]).remove(0);

    let mut metal_sess = Session::new(Device::Metal).compile_with(g, &opts);
    metal_sess.set_param("w", &weight);
    let metal = metal_sess.run(&[("x", &x)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("conv2d patch conv1 max_abs={max_abs:.6}");
    assert!(max_abs < 1e-3, "Metal conv2d max_abs={max_abs}");
}
