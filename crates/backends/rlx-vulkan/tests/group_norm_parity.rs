// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native SPIR-V GroupNorm vs CPU, on blocks large enough to matter.
//!
//! The shader used to give each (batch, group) one invocation and walk the
//! whole block serially, three times. InstanceNorm over a DAN page — one
//! group per channel, ~1M elements each, 16 channels — then ran as 16
//! threads: on RADV the dispatch outlived the GPU watchdog
//! (`ERROR_DEVICE_LOST`), and on NVIDIA the single running f32 sum drifted
//! by ~1e-3. Now one workgroup reduces each block cooperatively.
//!
//! An offset of 100 on a small-variance signal is what makes an unpaired
//! f32 accumulation visible: the mean is large next to the spread, so any
//! error in it survives the subtraction.

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;

mod common;

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn graph(n: usize, c: usize, h: usize, w: usize, groups: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("group_norm");
    let x = g.input("x", Shape::new(&[n, c, h, w], f));
    let gamma = g.input("gamma", Shape::new(&[c], f));
    let beta = g.input("beta", Shape::new(&[c], f));
    let y = g.group_norm(x, gamma, beta, groups, 1e-3);
    g.set_outputs(vec![y]);
    g
}

fn check(n: usize, c: usize, h: usize, w: usize, groups: usize) {
    let len = n * c * h * w;
    let x: Vec<f32> = (0..len)
        .map(|i| 100.0 + ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..c).map(|i| i as f32 * -0.05).collect();
    let inputs: [(&str, &[f32]); 3] = [("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let want = {
        use rlx::prelude::*;
        Session::new(Device::Cpu)
            .compile(graph(n, c, h, w, groups))
            .run(&inputs)
            .remove(0)
    };
    let got = VulkanExecutable::compile(graph(n, c, h, w, groups))
        .run(&inputs)
        .remove(0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // Outputs are unit-scale; 1e-4 is ~10 ulps of the normalized values and
    // far below the 1e-3 the serial sum drifted by.
    assert!(
        worst < 1e-4,
        "group_norm [{n},{c},{h},{w}] groups={groups}: max |vk − cpu| = {worst:e}"
    );
}

#[test]
fn instance_norm_over_a_large_plane() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    // One group per channel, 250K elements per group.
    check(1, 4, 400, 625, 4);
}

#[test]
fn grouped_channels_and_batch() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    // Two channels per group, a block that is not a multiple of 256, batch 2.
    check(2, 6, 37, 53, 3);
}
