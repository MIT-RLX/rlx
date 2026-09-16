// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Lstm { carry: true }` must thread `hn`/`cn` back into `h0`/`c0`.
//!
//! Carry's whole purpose is decode threading: `h0`/`c0` are persistent buffers,
//! the op overwrites them with the final state, and the next call continues the
//! sequence. Seeding *without* writing back is silently wrong — every step
//! restarts from the same state, outputs stay plausible, and nothing fails.
//!
//! That is exactly what Metal, MLX and wgpu did before this (found via
//! `rlx-ten-vad`, whose
//! streaming VAD scored `max|Δ| 0.52` with 64 decision flips off-CPU while CPU
//! was correct). The existing `gru_carry_native` test could not catch it: it
//! compares one run's *output*, and the writeback is only observable across
//! two runs.
//!
//! The check here is device-independent: stepping a `seq = 1` graph `N` times
//! must reproduce a single `seq = N` run of the same weights. Anything that
//! fails to advance the state diverges from step 2 onwards.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const B: usize = 1;
const INP: usize = 5;
const H: usize = 16;
const STEPS: usize = 4;

fn mk(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 0.6 - 0.3
        })
        .collect()
}

struct Weights {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    bias: Vec<f32>,
    x: Vec<f32>,
}

impl Weights {
    fn new() -> Self {
        Self {
            w_ih: mk(4 * H * INP, 2),
            w_hh: mk(4 * H * H, 3),
            bias: mk(4 * H, 4),
            x: mk(B * STEPS * INP, 1),
        }
    }
}

/// `seq` timesteps in one shot, no carry — the ground truth.
fn one_shot(dev: Device, w: &Weights) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("lstm_one_shot");
    let x = g.input("x", Shape::new(&[B, STEPS, INP], f));
    let wi = g.input("w_ih", Shape::new(&[4 * H * INP], f));
    let wh = g.input("w_hh", Shape::new(&[4 * H * H], f));
    let bs = g.input("bias", Shape::new(&[4 * H], f));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: H,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, wi, wh, bs],
        Shape::new(&[B, STEPS, H], f),
    );
    g.set_outputs(vec![y]);
    Session::new(dev)
        .compile(g)
        .run(&[
            ("x", &w.x),
            ("w_ih", &w.w_ih),
            ("w_hh", &w.w_hh),
            ("bias", &w.bias),
        ])
        .remove(0)
}

/// One timestep per call, state carried in params across calls.
fn stepped(dev: Device, w: &Weights) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("lstm_stepped");
    let x = g.input("x", Shape::new(&[B, 1, INP], f));
    let wi = g.input("w_ih", Shape::new(&[4 * H * INP], f));
    let wh = g.input("w_hh", Shape::new(&[4 * H * H], f));
    let bs = g.input("bias", Shape::new(&[4 * H], f));
    // `[L*D, batch, hidden]`, persistent across runs — this is the buffer the
    // op is contracted to overwrite.
    let h0 = g.param("h0", Shape::new(&[1, B, H], f));
    let c0 = g.param("c0", Shape::new(&[1, B, H], f));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: H,
            num_layers: 1,
            bidirectional: false,
            carry: true,
        },
        vec![x, wi, wh, bs, h0, c0],
        Shape::new(&[B, 1, H], f),
    );
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(dev).compile(g);
    compiled.set_param("h0", &[0.0; B * H]);
    compiled.set_param("c0", &[0.0; B * H]);
    compiled.finalize_params();

    let mut out = Vec::with_capacity(STEPS * B * H);
    for t in 0..STEPS {
        let xt = &w.x[t * INP..(t + 1) * INP];
        let mut y = compiled
            .run(&[
                ("x", xt),
                ("w_ih", &w.w_ih),
                ("w_hh", &w.w_hh),
                ("bias", &w.bias),
            ])
            .remove(0);
        out.append(&mut y);
    }
    out
}

fn check(dev: Device) {
    let w = Weights::new();
    let want = one_shot(dev, &w);
    let got = stepped(dev, &w);
    assert_eq!(got.len(), want.len(), "{dev:?}: output length");

    let maxd = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Step 0 always agrees (both start from zero state); a missing writeback
    // shows up from step 1 on, so report where it first diverges.
    let first_bad = got
        .iter()
        .zip(&want)
        .position(|(a, b)| (a - b).abs() > 1e-4)
        .map(|i| i / H);
    eprintln!("[lstm-carry] {dev:?}: max|Δ| = {maxd:.3e} first bad step = {first_bad:?}");
    assert!(
        maxd < 1e-4,
        "{dev:?}: carry did not thread state — {STEPS} single steps diverge from one \
         {STEPS}-step run by {maxd:.3e}, first at step {first_bad:?}. `hn`/`cn` are \
         contracted to overwrite `h0`/`c0` in place."
    );
}

#[test]
fn carry_threads_state_on_cpu() {
    check(Device::Cpu);
}

#[test]
fn carry_threads_state_on_gpu_backends() {
    for dev in [
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Cuda,
        Device::Rocm,
        Device::Vulkan,
        Device::Ane,
    ] {
        if common::skip_unless(dev) {
            continue;
        }
        let _guard = common::GpuTestGuard::acquire(dev);
        check(dev);
    }
}
