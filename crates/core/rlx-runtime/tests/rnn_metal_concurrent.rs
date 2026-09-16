// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multi-layer LSTM / GRU / Elman RNN on Metal under a **Concurrent** encoder.
//!
//! `encode_lstm`/`encode_gru`/`encode_rnn` each issue several dispatches per op:
//! one per (layer, direction), ping-ponging intermediate layer outputs through an
//! in-arena scratch pair, and — for LSTM — an input-projection prepass the
//! recurrence then reads. Those are read-after-write dependencies *inside a single
//! thunk*, which `concurrent_barrier_set` (which reasons between thunks) cannot
//! see. On a Serial encoder the ordering is implicit, so the hazard is invisible;
//! with `RLX_METAL_CONCURRENT=1` it is real and the helpers must order themselves
//! via `intra_thunk_barrier`.
//!
//! Run the whole file under `RLX_METAL_CONCURRENT=1` to exercise that path. Adding
//! `RLX_METAL_CONCURRENT_NOBARRIER=1` drops the barriers and should make these
//! fail — that is what shows the barriers are load-bearing rather than decorative.
//!
//! Weights are deliberately contractive: an LSTM driven by unit-scale random
//! weights is chaotic, so a well-conditioned recurrence is what lets this assert a
//! real tolerance instead of measuring conditioning (see `lstm_metal_diagnose.rs`).
#![cfg(all(feature = "cpu", feature = "metal"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const RECURRENT_SCALE: f32 = 0.1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Lstm,
    Gru,
    Rnn,
}

fn mk(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
                * scale
        })
        .collect()
}

/// Per-layer input width: `inp` for layer 0, `dirs·hidden` afterwards.
fn wih_total(kind: Kind, layers: usize, dirs: usize, inp: usize, h: usize) -> usize {
    let gates = if kind == Kind::Rnn {
        1
    } else if kind == Kind::Gru {
        3
    } else {
        4
    };
    (0..layers)
        .map(|l| {
            let in_l = if l == 0 { inp } else { dirs * h };
            dirs * gates * h * in_l
        })
        .sum()
}

fn build(
    kind: Kind,
    b: usize,
    s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
) -> Graph {
    let f = DType::F32;
    let dirs = if bidir { 2 } else { 1 };
    let gates = match kind {
        Kind::Lstm => 4,
        Kind::Gru => 3,
        Kind::Rnn => 1,
    };
    let mut g = Graph::new("rnn_concurrent");
    let x = g.input("x", Shape::new(&[b, s, inp], f));
    let wih = g.input(
        "w_ih",
        Shape::new(&[wih_total(kind, layers, dirs, inp, h)], f),
    );
    let whh = g.input("w_hh", Shape::new(&[layers * dirs * gates * h * h], f));
    let mut ins = vec![x, wih, whh];
    // GRU carries separate input/hidden biases; LSTM and Elman RNN take one each.
    let bias_n = layers * dirs * gates * h;
    ins.push(g.input("b_ih", Shape::new(&[bias_n], f)));
    if kind == Kind::Gru {
        ins.push(g.input("b_hh", Shape::new(&[bias_n], f)));
    }
    let out_shape = Shape::new(&[b, s, dirs * h], f);
    let op = match kind {
        Kind::Lstm => Op::Lstm {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
        },
        Kind::Gru => Op::Gru {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
        },
        Kind::Rnn => Op::Rnn {
            hidden_size: h,
            num_layers: layers,
            bidirectional: bidir,
            carry: false,
            relu: false,
        },
    };
    let y = g.add_node(op, ins, out_shape);
    g.set_outputs(vec![y]);
    g
}

fn run(
    dev: Device,
    kind: Kind,
    b: usize,
    s: usize,
    inp: usize,
    h: usize,
    layers: usize,
    bidir: bool,
) -> Vec<f32> {
    let dirs = if bidir { 2 } else { 1 };
    let gates = match kind {
        Kind::Lstm => 4,
        Kind::Gru => 3,
        Kind::Rnn => 1,
    };
    let xd = mk(b * s * inp, 1, 1.0);
    let wihd = mk(wih_total(kind, layers, dirs, inp, h), 2, RECURRENT_SCALE);
    let whhd = mk(layers * dirs * gates * h * h, 3, RECURRENT_SCALE);
    let bias_n = layers * dirs * gates * h;
    let bihd = mk(bias_n, 4, RECURRENT_SCALE);
    let bhhd = mk(bias_n, 5, RECURRENT_SCALE);
    let mut slots: Vec<(&str, &[f32])> = vec![
        ("x", &xd),
        ("w_ih", &wihd),
        ("w_hh", &whhd),
        ("b_ih", &bihd),
    ];
    if kind == Kind::Gru {
        slots.push(("b_hh", &bhhd));
    }
    let mut c = Session::new(dev).compile(build(kind, b, s, inp, h, layers, bidir));
    c.run(&slots).pop().unwrap()
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Multi-layer, both directions, at the shape a wide speech LSTM uses (h=128,
/// s=256) — the configuration where a missing intra-thunk barrier races.
#[test]
fn multilayer_recurrent_matches_cpu_under_concurrent_encoder() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    if !rlx_ir::env::flag("RLX_METAL_CONCURRENT") {
        eprintln!("note: RLX_METAL_CONCURRENT unset — Serial encoder, ordering is implicit");
    }
    let mut failures = Vec::new();
    for kind in [Kind::Lstm, Kind::Gru, Kind::Rnn] {
        for layers in [2usize, 3] {
            for bidir in [false, true] {
                let (b, s, inp, h) = (1usize, 256usize, 64usize, 128usize);
                let cpu = run(Device::Cpu, kind, b, s, inp, h, layers, bidir);
                let gpu = run(Device::Metal, kind, b, s, inp, h, layers, bidir);
                let nans = gpu.iter().filter(|v| !v.is_finite()).count();
                let d = max_delta(&cpu, &gpu);
                let tag = format!("{kind:?} layers={layers} bidir={bidir}");
                println!("  {tag:<34} max_abs={d:.3e} nans={nans}");
                if nans > 0 || d > 1e-4 {
                    failures.push(format!("{tag}: max_abs={d:.3e} nans={nans}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "multi-layer recurrent diverged on Metal:\n{}",
        failures.join("\n")
    );
}

/// A surviving hazard shows up as run-to-run instability, which a pure ordering
/// bug would produce and fp reassociation would not.
#[test]
fn multilayer_lstm_is_deterministic_under_concurrent_encoder() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("skip: no Metal device");
        return;
    }
    for layers in [2usize, 3] {
        let a = run(Device::Metal, Kind::Lstm, 1, 256, 64, 128, layers, true);
        let b = run(Device::Metal, Kind::Lstm, 1, 256, 64, 128, layers, true);
        let c = run(Device::Metal, Kind::Lstm, 1, 256, 64, 128, layers, true);
        let d = max_delta(&a, &b).max(max_delta(&b, &c));
        println!("  layers={layers}: metal-vs-metal max_abs={d:.3e}");
        assert_eq!(
            d, 0.0,
            "layers={layers}: non-deterministic — a surviving race"
        );
    }
}
