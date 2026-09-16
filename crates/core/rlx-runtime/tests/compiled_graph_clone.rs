// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — CompiledGraph::clone parity across backends.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn cumsum_graph() -> Graph {
    let mut g = Graph::new("cumsum_clone");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.cumsum(x, 0, false, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn cpu_compiled_graph_clone_matches_original() {
    let _gpu = common::serialize_gpu();
    let g = cumsum_graph();
    let mut a = Session::new(Device::Cpu).compile(g);
    let mut b = a.clone();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let out_a = a.run(&[("x", &input)]);
    let out_b = b.run(&[("x", &input)]);
    assert_eq!(out_a[0], vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(out_b, out_a);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_compiled_graph_clone_matches_original() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        return;
    }
    let g = cumsum_graph();
    let mut a = Session::new(Device::Metal).compile(g);
    let mut b = a.clone();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let out_a = a.run(&[("x", &input)]);
    let out_b = b.run(&[("x", &input)]);
    assert_eq!(out_a[0], vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(out_b, out_a);
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_compiled_graph_clone_matches_original() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Mlx, "mlx") {
        return;
    }
    let g = cumsum_graph();
    let mut a = Session::new(Device::Mlx).compile(g);
    let mut b = a.clone();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let out_a = a.run(&[("x", &input)]);
    let out_b = b.run(&[("x", &input)]);
    assert_eq!(out_a[0], vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(out_b, out_a);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_compiled_graph_clone_matches_original() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let g = cumsum_graph();
    let mut a = Session::new(Device::Gpu).compile(g);
    let mut b = a.clone();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let out_a = a.run(&[("x", &input)]);
    let out_b = b.run(&[("x", &input)]);
    assert_eq!(out_a[0], vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(out_b, out_a);
}
