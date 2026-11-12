// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fused QKV on Metal stores its combined weight once, as a pack param that
//! `set_param` of each part writes (`rlx_metal` `weight_packs`), instead of a
//! per-run concat of the parts. These check it computes what the CPU does,
//! follows a later write of one part, shares across executables of a sharing
//! scope, and matches the concat it replaces.

#![cfg(all(feature = "metal", target_vendor = "apple"))]

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};

const K: usize = 256;
const N: [usize; 3] = [256, 128, 128];
const PARTS: [&str; 3] = ["q", "k", "v"];

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + seed * 11) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

/// `x·q`, `x·k`, `x·v` summed per row block: three matmuls on one input,
/// which the fusion runs as one over a combined weight.
fn graph() -> Graph {
    let mut g = Graph::new("qkv");
    let x = g.input("x", Shape::new(&[4, K], DType::F32));
    let outs = PARTS
        .iter()
        .zip(N)
        .map(|(name, n)| {
            let w = g.param(name, Shape::new(&[K, n], DType::F32));
            g.mm(x, w)
        })
        .collect();
    g.set_outputs(outs);
    g
}

fn compile(device: Device, seed: usize, bind: bool) -> CompiledGraph {
    let mut c = Session::new(device).compile(graph());
    if bind {
        for (i, (name, n)) in PARTS.iter().zip(N).enumerate() {
            c.set_param(name, &values(K * n, seed + i));
        }
    }
    c
}

fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (o, (a, b)) in got.iter().zip(want).enumerate() {
        let d = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(d < 1e-2, "{what}: output {o} off by {d}");
    }
}

struct Env;

impl Drop for Env {
    fn drop(&mut self) {
        for k in ["RLX_METAL_WEIGHT_SPLIT_MIN", "RLX_METAL_NO_WEIGHT_PACKS"] {
            rlx_ir::env::unset(k);
        }
    }
}

#[test]
fn packed_qkv_matches_cpu_and_follows_part_writes() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let _env = Env;
    let x = values(4 * K, 9);
    for split_min in [None, Some("1")] {
        // Inline in the arena, then in the weight buffer.
        match split_min {
            Some(v) => rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", v),
            None => rlx_ir::env::unset("RLX_METAL_WEIGHT_SPLIT_MIN"),
        }
        let what = format!("split min {split_min:?}");
        let mut metal = compile(Device::Metal, 1, true);
        let mut cpu = compile(Device::Cpu, 1, true);
        assert_close(&metal.run(&[("x", &x)]), &cpu.run(&[("x", &x)]), &what);
        // One part rewritten after a run: the others keep their values.
        metal.set_param("k", &values(K * N[1], 40));
        cpu.set_param("k", &values(K * N[1], 40));
        assert_close(&metal.run(&[("x", &x)]), &cpu.run(&[("x", &x)]), &format!("{what}, k rewritten"));
    }
}

#[test]
fn pack_and_concat_agree() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let _env = Env;
    let x = values(4 * K, 9);
    let packed = compile(Device::Metal, 1, true).run(&[("x", &x)]);
    rlx_ir::env::set("RLX_METAL_NO_WEIGHT_PACKS", "1");
    let concat = compile(Device::Metal, 1, true).run(&[("x", &x)]);
    assert_close(&packed, &concat, "pack vs concat");
}

#[test]
fn a_pack_is_shared_across_executables() {
    if common::skip_unless(Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Metal);
    let _env = Env;
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    let x = values(4 * K, 9);
    let want = compile(Device::Cpu, 1, true).run(&[("x", &x)]);
    let scope = format!("packs:{}", std::process::id());
    let (one, mut two) = rlx_runtime::with_param_sharing_scope(&scope, || {
        (compile(Device::Metal, 1, true), compile(Device::Metal, 1, false))
    });
    assert!(PARTS.iter().all(|p| two.param_already_set(p)), "the pack was not mapped");
    drop(one);
    assert_close(&two.run(&[("x", &x)]), &want, "second executable");
}
