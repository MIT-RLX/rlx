// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Outputs past the first must come back with real values on Metal.
//!
//! A decode graph emits `logits` plus the appended K/V per layer, and the
//! caller keeps the K/V as its cache. If anything but output 0 reads back as
//! zeros, decode still produces plausible logits — attention over a cache whose
//! newest row is zero is not obviously broken — and the model quietly repeats
//! itself. That is hard to spot from the output and trivial to spot here.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use std::sync::{Mutex, MutexGuard};

static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct MetalTestGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl MetalTestGuard {
    fn new() -> Self {
        Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for MetalTestGuard {
    fn drop(&mut self) {
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

/// `out0` is a reduction (the "logits"); `out1` is the appended cache, shaped
/// the way a decode graph shapes it: past rows concatenated with one new row.
fn build(past: usize, dim: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("secondary_output");
    let p = g.input("past", Shape::new(&[past, dim], f));
    let x = g.input("x", Shape::new(&[1, dim], f));
    // One computed row, so the new row is not merely a copy of an input.
    let scaled = g.binary(rlx_ir::op::BinaryOp::Add, x, x, Shape::new(&[1, dim], f));
    let cat = g.add_node(
        Op::Concat { axis: 0 },
        vec![p, scaled],
        Shape::new(&[past + 1, dim], f),
    );
    let summed = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![0],
            keep_dim: true,
        },
        vec![cat],
        Shape::new(&[1, dim], f),
    );
    g.set_outputs(vec![summed, cat]);
    g
}

fn run_on(device: Device, past: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut c = Session::new(device).compile(build(past, dim));
    let past_vals: Vec<f32> = (0..past * dim).map(|i| 1.0 + i as f32).collect();
    let x: Vec<f32> = (0..dim).map(|i| 100.0 + i as f32).collect();
    c.run(&[("past", &past_vals), ("x", &x)])
}

#[test]
fn concat_output_after_logits_is_not_zeroed() {
    let _guard = MetalTestGuard::new();
    let (past, dim) = (4usize, 3usize);
    let cpu = run_on(Device::Cpu, past, dim);
    let metal = run_on(Device::Metal, past, dim);

    assert_eq!(metal.len(), 2, "metal returned {} outputs", metal.len());
    for (i, (m, c)) in metal.iter().zip(&cpu).enumerate() {
        assert_eq!(m.len(), c.len(), "output {i}: length");
        let bad = m.iter().zip(c).position(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            bad.is_none(),
            "output {i} differs at {:?}\n  metal: {:?}\n  cpu:   {:?}",
            bad,
            m,
            c
        );
    }
}

/// The same graph re-run with different inputs, from one compiled instance —
/// the decode loop's actual usage. A cached compiled graph that returns the
/// previous run's secondary output looks like a model that lags a token behind.
#[test]
fn secondary_output_updates_on_a_second_run() {
    let _guard = MetalTestGuard::new();
    let (past, dim) = (4usize, 3usize);
    let mut c = Session::new(Device::Metal).compile(build(past, dim));
    let past_vals: Vec<f32> = (0..past * dim).map(|i| 1.0 + i as f32).collect();

    let first = {
        let x: Vec<f32> = (0..dim).map(|i| 100.0 + i as f32).collect();
        c.run(&[("past", &past_vals), ("x", &x)])[1].clone()
    };
    let second = {
        let x: Vec<f32> = (0..dim).map(|i| 500.0 + i as f32).collect();
        c.run(&[("past", &past_vals), ("x", &x)])[1].clone()
    };
    let new_row = |v: &Vec<f32>| v[past * dim..].to_vec();
    assert_ne!(
        new_row(&first),
        new_row(&second),
        "the appended row did not change when the input did"
    );
    // 2x of 500.. — pins that it is the *current* input, not just "different".
    let want: Vec<f32> = (0..dim).map(|i| 2.0 * (500.0 + i as f32)).collect();
    assert_eq!(new_row(&second), want, "second run's appended row");
}

/// The pattern a Metal decode step actually uses: ask for the logits only, then
/// pull the one appended K/V row out of an output that was *not* requested.
///
/// Reading only output 0 is the whole point — a full K/V readback at every token
/// costs more than the decode — so `read_output_row` has to be able to reach an
/// output that selective readback skipped. If it cannot, it returns zeros, and a
/// decode loop stores a cache whose newest row is empty.
#[test]
fn read_output_row_works_after_selective_readback() {
    let _guard = MetalTestGuard::new();
    let (past, dim) = (4usize, 3usize);
    let mut c = Session::new(Device::Metal).compile(build(past, dim));
    let past_vals: Vec<f32> = (0..past * dim).map(|i| 1.0 + i as f32).collect();
    let x: Vec<f32> = (0..dim).map(|i| 100.0 + i as f32).collect();

    let outs = c.run_read_outputs(&[("past", &past_vals), ("x", &x)], Some(&[0]));
    assert!(!outs.is_empty(), "no logits returned");

    let row = c
        .read_output_row(1, past, dim)
        .expect("read_output_row on the cache output");
    let want: Vec<f32> = (0..dim).map(|i| 2.0 * (100.0 + i as f32)).collect();
    assert_eq!(
        row, want,
        "appended row read back wrong after logits-only readback"
    );
}

/// Mid-axis concat as a graph output, which is the shape a KV cache has.
///
/// A decode graph appends the new token's K/V by concatenating `[B, past, H, D]`
/// with `[B, 1, H, D]` on axis 1 — a *middle* axis, so the copy is strided
/// rather than a pair of contiguous blocks. The outermost-axis case is the easy
/// one and is covered above; this is the one the model depends on.
#[test]
fn mid_axis_concat_output_includes_the_appended_slice() {
    let _guard = MetalTestGuard::new();
    let f = DType::F32;
    let (b, past, h, d) = (1usize, 4usize, 2usize, 3usize);

    let mut g = Graph::new("mid_axis_concat");
    let p = g.input("past", Shape::new(&[b, past, h, d], f));
    let x = g.input("x", Shape::new(&[b, 1, h, d], f));
    let doubled = g.binary(
        rlx_ir::op::BinaryOp::Add,
        x,
        x,
        Shape::new(&[b, 1, h, d], f),
    );
    let cat = g.add_node(
        Op::Concat { axis: 1 },
        vec![p, doubled],
        Shape::new(&[b, past + 1, h, d], f),
    );
    g.set_outputs(vec![cat]);

    let past_vals: Vec<f32> = (0..b * past * h * d).map(|i| 1.0 + i as f32).collect();
    let x_vals: Vec<f32> = (0..b * h * d).map(|i| 100.0 + i as f32).collect();

    let run = |device: Device| {
        let mut c = Session::new(device).compile({
            let mut g2 = Graph::new("mid_axis_concat");
            let p = g2.input("past", Shape::new(&[b, past, h, d], f));
            let x = g2.input("x", Shape::new(&[b, 1, h, d], f));
            let doubled = g2.binary(
                rlx_ir::op::BinaryOp::Add,
                x,
                x,
                Shape::new(&[b, 1, h, d], f),
            );
            let cat = g2.add_node(
                Op::Concat { axis: 1 },
                vec![p, doubled],
                Shape::new(&[b, past + 1, h, d], f),
            );
            g2.set_outputs(vec![cat]);
            g2
        });
        c.run(&[("past", &past_vals), ("x", &x_vals)])
            .into_iter()
            .next()
            .expect("concat output")
    };
    let _ = &g;

    let cpu = run(Device::Cpu);
    let metal = run(Device::Metal);
    let row = h * d;
    let appended = |v: &Vec<f32>| v[past * row..].to_vec();
    assert_eq!(metal.len(), cpu.len(), "output length");
    assert_eq!(
        appended(&metal),
        appended(&cpu),
        "appended slice differs\n  metal: {:?}\n  cpu:   {:?}",
        appended(&metal),
        appended(&cpu)
    );
    assert_eq!(metal, cpu, "whole concat differs");
}
