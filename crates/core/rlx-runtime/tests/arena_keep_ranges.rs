// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU executable zeros its scratch arena before every run, keeping the
//! persistent bytes (params, constants, resident handles). The kept ranges
//! are computed once and reused, so they must be recomputed when a handle
//! becomes resident after the first run: otherwise the reset zeros the
//! resident data the run is meant to read.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn graph() -> Graph {
    let mut g = Graph::new("keep_ranges");
    let kv = g.input("kv", Shape::new(&[4], DType::F32));
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let out = g.add(kv, x);
    g.set_outputs(vec![out]);
    g
}

#[test]
fn a_handle_made_resident_after_the_first_run_is_kept() {
    let mut c = Session::new(Device::Cpu).compile(graph());
    // first run: `kv` is an ordinary input, so its bytes are scratch
    let out = c.run(&[("kv", &[9.0; 4]), ("x", &[1.0; 4])]);
    assert_eq!(out[0], vec![10.0; 4]);
    // now resident: written once, not rewritten per run, never zeroed
    assert!(c.bind_gpu_handle("kv", &[1.0, 2.0, 3.0, 4.0]));
    for _ in 0..2 {
        let out = c.run(&[("x", &[0.5; 4])]);
        assert_eq!(
            out[0],
            vec![1.5, 2.5, 3.5, 4.5],
            "the reset zeroed the resident handle"
        );
    }
}
