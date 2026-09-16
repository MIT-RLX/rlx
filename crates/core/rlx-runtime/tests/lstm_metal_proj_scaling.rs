// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Does hoisting `bias + W_ih·x` out of the LSTM recurrence actually pay?
//!
//! The recurrence is sequential and runs in one threadgroup per batch item, so any
//! work left inside it is serialized on a single GPU core. `lstm_input_proj` moves
//! the input term into a fully parallel prepass, which makes a falsifiable
//! prediction: **runtime should be roughly flat in `input_size`**, because growing
//! `in_sz` only grows the parallel pass, not the serial one. If the input term were
//! still inside the loop, time would grow linearly with `in_sz`.
//!
//! Run with `--nocapture`; ignored by default since it is a timing observation, not
//! a correctness assertion (the machine may be loaded).
#![cfg(all(feature = "cpu", feature = "metal"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::time::Instant;

mod common;

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 5000.0 - 0.1)
        .collect()
}

fn build(b: usize, s: usize, inp: usize, h: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("lstm_proj_scaling");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input("w_ih", Shape::new(&[4 * h * inp], f));
    let whh = g.input("w_hh", Shape::new(&[4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![out]);
    g
}

/// Best-of-N wall time for one `Op::Lstm` run, warm.
fn time_ms(dev: Device, b: usize, s: usize, inp: usize, h: usize) -> f64 {
    let xd = mk(b * s * inp, 1);
    let wihd = mk(4 * h * inp, 2);
    let whhd = mk(4 * h * h, 3);
    let bd = mk(4 * h, 4);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let mut c = Session::new(dev).compile(build(b, s, inp, h));
    for _ in 0..3 {
        let _ = c.run(&slots);
    }
    let mut best = f64::MAX;
    for _ in 0..10 {
        let t = Instant::now();
        let _ = c.run(&slots);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore = "timing observation, not a correctness gate"]
fn lstm_metal_runtime_is_flat_in_input_size() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    let (b, s, h) = (1usize, 256usize, 128usize);
    println!("Op::Lstm  batch={b} seq={s} hidden={h}, sweeping input_size:");
    let mut first = 0.0;
    for (i, &inp) in [64usize, 128, 256, 512, 1024].iter().enumerate() {
        let ms = time_ms(Device::Metal, b, s, inp, h);
        if i == 0 {
            first = ms;
        }
        println!(
            "  in_sz={inp:<5} {ms:7.3} ms   ({:.2}x the in_sz=64 time)",
            ms / first
        );
    }
    println!(
        "flat => the input projection is hoisted out of the serial recurrence;\n\
         linear in in_sz => it is still inside the timestep loop."
    );
}
