// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Executables compiled under one sharing scope on Metal hold one copy of a
//! weight they both read (`rlx_metal::shared_weights`), though their params
//! and layouts differ: each computes what it computes alone, the second's
//! copy is the first's pages, and it outlives the first.

#![cfg(all(feature = "metal", target_vendor = "apple"))]

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

const K: usize = 256;
const N: usize = 512; // 512 KiB of f32: past the external-weight threshold

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + seed * 11) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

/// `x · shared` (then `· extra` when given, a second weight only it has).
fn graph(extra: bool) -> Graph {
    let mut g = Graph::new(if extra { "two" } else { "one" });
    let x = g.input("x", Shape::new(&[4, K], DType::F32));
    let w = g.param("shared", Shape::new(&[K, N], DType::F32));
    let mut y = g.mm(x, w);
    if extra {
        let e = g.param("extra", Shape::new(&[N, N], DType::F32));
        y = g.mm(y, e);
    }
    g.set_outputs(vec![y]);
    g
}

fn compile(device: Device, extra: bool, bind: bool) -> CompiledGraph {
    let mut c = Session::new(device).compile(graph(extra));
    if bind {
        c.set_param("shared", &values(K * N, 1));
    }
    if extra {
        c.set_param("extra", &values(N * N, 2));
    }
    c
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

#[test]
fn a_shared_weight_is_one_copy_across_executables() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    // Weights this small stay in the activation arena unless asked.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    let x = values(4 * K, 3);
    let want_one = compile(Device::Cpu, false, true)
        .run(&[("x", &x)])
        .remove(0);
    let want_two = compile(Device::Cpu, true, true).run(&[("x", &x)]).remove(0);

    let scope = format!("test:{}", std::process::id());
    let (mut one, mut two) = rlx_runtime::with_param_sharing_scope(&scope, || {
        // The second never binds `shared`: it reads the first's copy.
        (
            compile(Device::Metal, false, true),
            compile(Device::Metal, true, false),
        )
    });
    // A binder may skip what the second maps from the first.
    assert!(two.param_already_set("shared") && !two.param_already_set("extra"));
    assert!(!one.param_already_set("shared"));
    let got_one = one.run(&[("x", &x)]).remove(0);
    let got_two = two.run(&[("x", &x)]).remove(0);
    assert!(
        max_diff(&got_one, &want_one) < 1e-3,
        "{}",
        max_diff(&got_one, &want_one)
    );
    assert!(
        max_diff(&got_two, &want_two) < 1e-2,
        "{}",
        max_diff(&got_two, &want_two)
    );

    // A new value through either is the other's too: one copy.
    one.set_param("shared", &values(K * N, 4));
    let mut cpu = compile(Device::Cpu, true, false);
    cpu.set_param("shared", &values(K * N, 4));
    let want_two = cpu.run(&[("x", &x)]).remove(0);
    drop(one);
    // …and it outlives the executable that made it.
    let got_two = two.run(&[("x", &x)]).remove(0);
    assert!(
        max_diff(&got_two, &want_two) < 1e-2,
        "{}",
        max_diff(&got_two, &want_two)
    );

    // Without a scope nothing is shared.
    let mut alone = compile(Device::Metal, true, false);
    assert!(!alone.param_already_set("shared"));
    let got = alone.run(&[("x", &x)]).remove(0);
    assert!(
        got.iter().all(|v| *v == 0.0),
        "an unbound weight read another's"
    );
}
