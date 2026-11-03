// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Under `rlx_ir::reproducible` automatic device selection takes the CPU,
//! whatever GPUs the host has; off, it is the cost model's choice.
//! (Own test binary: the switch is process-wide.)

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::Device;

#[test]
fn reproducible_runs_pick_the_cpu() {
    let mut g = Graph::new("pick");
    let x = g.input("x", Shape::new(&[256, 256], DType::F32));
    let y = g.mm(x, x);
    g.set_outputs(vec![y]);
    rlx_ir::reproducible::set_reproducible(true);
    assert_eq!(rlx_runtime::cost::fastest_device_for(&g), Device::Cpu);
    assert_eq!(rlx_runtime::fastest_device(), Device::Cpu);
    rlx_ir::reproducible::set_reproducible(false);
    let free = rlx_runtime::cost::fastest_device_for(&g);
    assert!(rlx_runtime::available_devices().contains(&free), "{free:?}");
}
