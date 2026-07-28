// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end: `Op::Reduce{Sum}` on Metal (which has no f64) routed through the
//! double-single accumulation kernel via `RLX_METAL_DW_SUM`, recovering the
//! correctly-rounded sum where plain f32 accumulation loses it.

#![cfg(all(feature = "metal", target_os = "macos"))]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

/// `1e8 + 100000·1 − 1e8`, true sum = 100000. Each `+1` is below `1e8`'s f32
/// ulp (=8), so plain f32 accumulation returns ~0.
fn ill_conditioned() -> Vec<f32> {
    let mut x = vec![1e8f32];
    x.extend(std::iter::repeat_n(1.0f32, 100_000));
    x.push(-1e8f32);
    x
}

fn sum_on_metal(x: &[f32]) -> f32 {
    let mut g = Graph::new("sum");
    let xi = g.input("x", Shape::new(&[x.len()], DType::F32));
    let s = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![xi],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![s]);
    Session::new(Device::Metal)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()[0]
}

#[test]
fn metal_reduce_sum_double_single_precision_mode() {
    if !is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let x = ill_conditioned();
    let truth = 100_000.0f32;

    // Default: plain f32 accumulation.
    rlx_ir::env::unset("RLX_METAL_DW_SUM");
    let naive = sum_on_metal(&x);

    // Opt-in: double-single accumulation.
    rlx_ir::env::set("RLX_METAL_DW_SUM", "1");
    let dw = sum_on_metal(&x);
    rlx_ir::env::unset("RLX_METAL_DW_SUM");

    eprintln!("Op::Reduce{{Sum}} on Metal:  naive f32 = {naive}  |  dw = {dw}  (true = {truth})");
    assert!(
        (naive - truth).abs() > 1.0,
        "plain f32 reduce should lose it, got {naive}"
    );
    assert!(
        (dw - truth).abs() < 1.0,
        "double-single reduce should recover ~{truth}, got {dw}"
    );
}
