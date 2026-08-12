// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ScatterElements / GatherNd / GatherElements parity — CPU vs host-delegating GPU.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, ScatterNdReduction, Shape};
use rlx_runtime::{Device, Session};

fn run_scatter_elements(device: Device) -> Vec<f32> {
    let mut g = Graph::new("sel");
    let data = g.input("data", Shape::new(&[2, 4], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 4], DType::F32));
    let updates = g.input("updates", Shape::new(&[2, 4], DType::F32));
    let y = g.scatter_elements(data, indices, updates, 1, ScatterNdReduction::None);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[
            ("data", &[1.0f32; 8][..]),
            ("indices", &[0.0f32, 2.0, 0.0, 2.0, 1.0, 3.0, 1.0, 3.0][..]),
            (
                "updates",
                &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0][..],
            ),
        ])
        .pop()
        .unwrap()
}

fn run_gather_nd(device: Device) -> Vec<f32> {
    let mut g = Graph::new("gnd");
    let data = g.input("data", Shape::new(&[4, 2], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 1], DType::F32));
    let y = g.gather_nd(data, indices, 0, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let data_v: Vec<f32> = (0..8).map(|i| i as f32).collect();
    Session::new(device)
        .compile(g)
        .run(&[("data", &data_v[..]), ("indices", &[1.0f32, 3.0][..])])
        .pop()
        .unwrap()
}

fn run_gather_elements(device: Device) -> Vec<f32> {
    let mut g = Graph::new("gel");
    let data = g.input("data", Shape::new(&[2, 4], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 4], DType::F32));
    let y = g.gather_elements(data, indices, 1);
    g.set_outputs(vec![y]);
    let data_v: Vec<f32> = (0..8).map(|i| (i as f32) + 1.0).collect();
    Session::new(device)
        .compile(g)
        .run(&[
            ("data", &data_v[..]),
            ("indices", &[0.0f32, 2.0, 1.0, 3.0, 1.0, 0.0, 3.0, 2.0][..]),
        ])
        .pop()
        .unwrap()
}

#[test]
fn scatter_elements_none_cpu() {
    let out = run_scatter_elements(Device::Cpu);
    // Row0: cols 0,2 overwritten by 10,30 (and again 30,40 → last write wins)
    // indices row0 = [0,2,0,2] updates [10,20,30,40] → col0=30, col2=40
    // Row1: [1,3,1,3] / [50,60,70,80] → col1=70, col3=80
    assert_eq!(out[0], 30.0);
    assert_eq!(out[1], 1.0);
    assert_eq!(out[2], 40.0);
    assert_eq!(out[3], 1.0);
    assert_eq!(out[4], 1.0);
    assert_eq!(out[5], 70.0);
    assert_eq!(out[6], 1.0);
    assert_eq!(out[7], 80.0);
}

#[test]
fn gather_nd_cpu() {
    let out = run_gather_nd(Device::Cpu);
    assert_eq!(out, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn gather_elements_cpu() {
    let out = run_gather_elements(Device::Cpu);
    // data = [[1,2,3,4],[5,6,7,8]], axis=1
    // idx row0 [0,2,1,3] → [1,3,2,4]; row1 [1,0,3,2] → [6,5,8,7]
    assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 8.0, 7.0]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn indexing_ops_metal_matches_cpu() {
    assert_eq!(
        run_scatter_elements(Device::Cpu),
        run_scatter_elements(Device::Metal)
    );
    assert_eq!(run_gather_nd(Device::Cpu), run_gather_nd(Device::Metal));
    assert_eq!(
        run_gather_elements(Device::Cpu),
        run_gather_elements(Device::Metal)
    );
}
