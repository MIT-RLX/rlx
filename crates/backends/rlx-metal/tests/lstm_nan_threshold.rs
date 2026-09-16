// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Where exactly does the native MSL LSTM start returning NaN?**
//!
//! `lstm_cpu_parity` fails with NaN at `hidden >= 40`; forcing the host path
//! (`RLX_METAL_LSTM_CPU=1`) is bit-exact (`max|Δ| = 0.000e0`) at every size, so
//! the native kernel is the culprit and the dispatch gate is admitting shapes it
//! cannot serve.
//!
//! The obvious candidates are ruled out by reading: the gate phase strides
//! (`for r = tid; r < 4h; r += tg_size`) so a narrow threadgroup is handled, and
//! `LSTM_MAX_H` is 1024 so `h_sh` is not overflowed at these widths. This finds
//! the real boundary so the gate can be set from a measurement rather than a
//! guess.
//!
//! Reports rather than asserts a specific threshold — the point is the number,
//! and the pass/fail guard lives in `lstm_cpu_parity`.
//!
//!     cargo test -p rlx-metal --test lstm_nan_threshold -- --nocapture

// `!(maxd < tol)` rather than `maxd >= tol`, deliberately: every comparison
// against NaN is false, so the `>=` form reports a NaN run as PASSING. This file
// exists for a kernel that returned NaN, so that is the one failure mode it must
// not be able to miss. Same reasoning (and the same allow) as `lstm_cpu_parity`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * scale
        })
        .collect()
}

fn run(device: Device, s: usize, h: usize) -> Vec<f32> {
    let mut g = Graph::new("lstm_thresh");
    let x = g.input("x", Shape::new(&[1, s, h], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * h * h], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * h * h], F32));
    let bias = g.param("bias", Shape::new(&[4 * h], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![x, wih, whh, bias],
        Shape::new(&[1, s, h], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param("w_ih", &fill(4 * h * h, 1, 0.1));
    c.set_param("w_hh", &fill(4 * h * h, 2, 0.1));
    c.set_param("bias", &fill(4 * h, 3, 0.1));
    c.finalize_params();
    c.run(&[("x", &fill(s * h, 4, 1.0))]).remove(0)
}

#[test]
fn report_the_hidden_width_where_native_lstm_breaks() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("metal not available — skipped");
        return;
    }
    println!("  h    seq  nan     max|Δ| vs CPU");
    let mut first_bad: Option<usize> = None;
    for h in [8usize, 16, 24, 31, 32, 33, 34, 36, 40, 48, 56, 64, 96, 128] {
        let s = 64;
        let cpu = run(Device::Cpu, s, h);
        let met = run(Device::Metal, s, h);
        let nan = met.iter().filter(|v| v.is_nan()).count();
        let maxd = cpu
            .iter()
            .zip(&met)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("  {h:3}  {s:4}  {nan:5}  {maxd:.3e}");
        if (nan > 0 || !(maxd < 1e-5)) && first_bad.is_none() {
            first_bad = Some(h);
        }
    }
    match first_bad {
        Some(h) => println!("\n  first bad hidden width: {h}"),
        None => println!("\n  no bad width found in the swept range"),
    }

    // Which PASS is wrong? The kernel runs `lstm_input_proj` (bias + W_ih·x for
    // every timestep) and then the sequential recurrence. At t=0 the recurrence
    // term is W_hh·h_prev with h_prev = 0, so timestep 0's output depends on the
    // projection ALONE. A wrong t=0 implicates the projection; a correct t=0 that
    // drifts later implicates the recurrence (or the ordering between them).
    for (h, s) in [
        (96usize, 8usize),
        (96, 16),
        (96, 32),
        (96, 64),
        (48, 32),
        (48, 64),
        (48, 128),
    ] {
        let cpu = run(Device::Cpu, s, h);
        let met = run(Device::Metal, s, h);
        let first_bad_idx = cpu
            .iter()
            .zip(&met)
            .position(|(a, b)| b.is_nan() || (a - b).abs() > 1e-5);
        {
            let bad: Vec<(usize, usize)> = met
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == 0.0)
                .map(|(i, _)| (i / h, i % h))
                .collect();
            if !bad.is_empty() {
                println!(
                    "  h={h:3} s={s:4}: first zero at t={} (of {s}), {} zeros",
                    bad[0].0,
                    bad.len()
                );
            }
        }
        {
            // Per-timestep max|Δ|: is the error GROWING (accumulation) or flat
            // then sudden (a discrete event at one step)?
            let traj: Vec<(usize, f32)> = (0..s)
                .map(|t| {
                    let d = (0..h)
                        .map(|u| (cpu[t * h + u] - met[t * h + u]).abs())
                        .fold(0f32, f32::max);
                    (t, d)
                })
                .collect();
            let interesting: Vec<String> = traj
                .iter()
                .filter(|(t, d)| *t % 8 == 0 || *d > 1e-5)
                .take(14)
                .map(|(t, d)| format!("t{t}:{d:.1e}"))
                .collect();
            println!("  h={h:3} s={s:4} trajectory: {}", interesting.join(" "));
        }
        let first_zero = met.iter().position(|v| *v == 0.0);
        let zeros = met.iter().filter(|v| **v == 0.0).count();
        println!(
            "  h={h:3}: len={} first_zero={:?} zeros={zeros} (bytes to first_zero={:?})",
            met.len(),
            first_zero,
            first_zero.map(|i| i * 4)
        );
        match first_bad_idx {
            None => println!("  h={h:3} s={s:4}: all timesteps agree"),
            Some(i) => println!(
                "  h={h:3}: first mismatch at elem {i} = timestep {}, unit {} \
                 (cpu {:.4} vs metal {:.4})",
                i / h,
                i % h,
                cpu[i],
                met[i]
            ),
        }
    }
}
#[test]
fn is_the_bad_output_written_as_zero_or_not_written_at_all() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("metal not available — skipped");
        return;
    }
    // One executable, two DIFFERENT inputs. Their corrupted windows sit at
    // different timesteps. If a bad slot holds the PREVIOUS run's value, the
    // write was dropped; if it holds exactly 0.0, the kernel computed a zero.
    const H: usize = 48;
    const S: usize = 64;
    let mut g = Graph::new("lstm_write");
    let xi = g.input("x", Shape::new(&[1, S, H], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * H * H], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * H * H], F32));
    let bias = g.param("bias", Shape::new(&[4 * H], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: H,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![xi, wih, whh, bias],
        Shape::new(&[1, S, H], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Metal).compile(g);
    c.set_param("w_ih", &fill(4 * H * H, 1, 0.1));
    c.set_param("w_hh", &fill(4 * H * H, 2, 0.1));
    c.set_param("bias", &fill(4 * H, 3, 0.1));
    c.finalize_params();

    let xa = fill(S * H, 4, 1.0);
    let xb = fill(S * H, 77, 1.0);
    let run_a1 = c.run(&[("x", &xa)]).remove(0);
    let run_b = c.run(&[("x", &xb)]).remove(0);

    let zeros_b: Vec<usize> = run_b
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 0.0)
        .map(|(i, _)| i)
        .collect();
    println!("  run B has {} exact zeros", zeros_b.len());
    for &i in zeros_b.iter().take(6) {
        println!(
            "    idx {i} (t={}, unit={}): runB={:.6}  runA_same_slot={:.6}",
            i / H,
            i % H,
            run_b[i],
            run_a1[i]
        );
    }
    println!("  => runA nonzero at those slots means the write was DROPPED, not computed as 0");
}
