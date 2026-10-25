// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::GatedDeltaNet` on MLX against the CPU reference.
//!
//! Per-head gates lower to MLX's fused `fast::gated_delta_update` kernel
//! (gate passed as `exp(g)`, state transposed to its `[B, H, Dv, Dk]`, the
//! `1/√n` query scale applied to the output); per-channel gates keep the
//! unrolled loop. Shapes include Qwen3.5's (32 heads × 128) and a sequence
//! long enough that the kernel's tiling is exercised, plus odd sizes that
//! MLX routes to its own fallback.
#![cfg(feature = "mlx")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn data(n: usize, seed: usize, lo: f32, span: f32) -> Vec<f32> {
    (0..n)
        .map(|i| lo + span * (((i * 7919 + seed * 104729) % 1000) as f32) / 1000.0)
        .collect()
}

#[test]
fn gated_delta_net_matches_cpu() {
    if common::skip_unless_available(Device::Mlx, "mlx") {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(Device::Mlx);
    for (b, s, h, n) in [
        (1usize, 1usize, 16usize, 128usize),
        (1, 37, 32, 128),
        (2, 5, 3, 64),
        (1, 3, 2, 7),
    ] {
        for per_channel in [false, true] {
            for carry in [false, true] {
                let f = DType::F32;
                let bshn = Shape::new(&[b, s, h, n], f);
                let gate_shape = if per_channel {
                    bshn.clone()
                } else {
                    Shape::new(&[b, s, h], f)
                };
                let mut g = Graph::new("gdn");
                let q = g.input("q", bshn.clone());
                let k = g.input("k", bshn.clone());
                let v = g.input("v", bshn.clone());
                let gi = g.input("g", gate_shape.clone());
                let beta = g.input("beta", Shape::new(&[b, s, h], f));
                let y = match (carry, per_channel) {
                    (false, false) => g.gated_delta_net(q, k, v, gi, beta, n, bshn.clone()),
                    (false, true) => g.gated_delta_net_pc(q, k, v, gi, beta, n, bshn.clone()),
                    (true, pc) => {
                        let st = g.input("state", Shape::new(&[b, h, n, n], f));
                        if pc {
                            g.gated_delta_net_carry_pc(q, k, v, gi, beta, st, n, bshn.clone())
                        } else {
                            g.gated_delta_net_carry(q, k, v, gi, beta, st, n, bshn.clone())
                        }
                    }
                };
                g.set_outputs(vec![y]);
                let nq = b * s * h * n;
                let feeds_owned = [
                    ("q", data(nq, 1, -0.5, 1.0)),
                    ("k", data(nq, 2, -0.1, 0.2)),
                    ("v", data(nq, 3, -1.0, 2.0)),
                    ("g", data(gate_shape.num_elements().unwrap(), 4, -0.4, 0.35)),
                    ("beta", data(b * s * h, 5, 0.1, 0.8)),
                    ("state", data(b * h * n * n, 6, -0.2, 0.4)),
                ];
                let feeds: Vec<(&str, &[f32])> = feeds_owned
                    .iter()
                    .filter(|(name, _)| carry || *name != "state")
                    .map(|(name, d)| (*name, d.as_slice()))
                    .collect();
                let want = Session::new(Device::Cpu).compile(g.clone()).run(&feeds);
                let got = Session::new(Device::Mlx).compile(g).run(&feeds);
                let d = want[0]
                    .iter()
                    .zip(&got[0])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let scale = want[0].iter().fold(1f32, |a, v| a.max(v.abs()));
                eprintln!(
                    "b={b} s={s} h={h} n={n} per_channel={per_channel} carry={carry}: max |cpu - mlx| = {d:.2e}"
                );
                assert!(
                    d <= 1e-5 * scale,
                    "b={b} s={s} h={h} n={n} per_channel={per_channel} carry={carry}: \
                     max |cpu - mlx| = {d} (scale {scale})"
                );
            }
        }
    }
}
