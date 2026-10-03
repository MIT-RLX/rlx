// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `RLX_KEEP_ELEMENTWISE_REGIONS=1` must not be able to build an inexpressible
//! region.
//!
//! That flag removes the unfuse pass for CPU, MLX, Metal **and** wgpu at once
//! (`stages::pipeline_for`), without touching their fusion limits. CPU asks for
//! `FusionLimits::UNBOUNDED` precisely because it normally unfuses — so with the
//! flag set, an arbitrarily long chain reached a region interpreter whose
//! scratch is a fixed `[0f32; 32]` indexed by chain step:
//!
//! ```text
//! panicked at rlx-cpu/src/thunk/ops/elementwise.rs:
//! index out of bounds: the len is 32 but the index is 32
//! ```
//!
//! Limit resolution now lives in one shared `resolve_fusion_limits`, which
//! bounds the caps whenever regions will survive. It had been two copies of the
//! same line — this crate's and rlx-compile's — and a clamp added to one simply
//! never ran on the path the runtime takes.
//!
//! The test drives the flag through the process environment because that is the
//! only way the runtime reads it.

use rlx_ir::{DType, Graph, Shape, op::BinaryOp};
use rlx_runtime::{Device, Session};

const N: usize = 256;
/// Past `FusionLimits::GPU_NATIVE.max_elementwise_steps` (32), which is what
/// the CPU interpreter's scratch is sized for.
const STEPS: usize = 48;

/// Alternating `+d` / `-d` from 1.0, so the exact answer is 1.0 for even
/// `STEPS` regardless of how the chain is split into regions.
fn chain() -> Graph {
    let s = Shape::new(&[N], DType::F32);
    let mut g = Graph::new("kept_region");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let mut cur = x;
    for i in 0..STEPS {
        let op = if i % 2 == 0 {
            BinaryOp::Add
        } else {
            BinaryOp::Sub
        };
        cur = g.binary(op, cur, y, s.clone());
    }
    g.set_outputs(vec![cur]);
    g
}

#[test]
fn keeping_regions_cannot_overrun_the_cpu_interpreter() {
    // SAFETY: single-threaded test; no other thread reads the environment here.
    unsafe { std::env::set_var("RLX_KEEP_ELEMENTWISE_REGIONS", "1") };

    let xs = vec![1.0f32; N];
    let ys = vec![0.25f32; N];
    let out = Session::new(Device::Cpu)
        .compile(chain())
        .run(&[("x", &xs), ("y", &ys)])
        .pop()
        .expect("one output");

    unsafe { std::env::remove_var("RLX_KEEP_ELEMENTWISE_REGIONS") };

    assert_eq!(out.len(), N);
    for (i, v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "element {i}: expected 1.0, got {v} — {STEPS} alternating ±0.25 steps from 1.0"
        );
    }
}
