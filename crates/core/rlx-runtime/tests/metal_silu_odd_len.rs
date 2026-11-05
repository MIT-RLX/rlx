// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal SiLU on lengths that are not a multiple of 4.
//!
//! Those skip the vectorised `silu_inplace4` and reach `silu_inplace`, a
//! data-pointer kernel the encoder used to bind with the arena-base ABI of
//! the generated GELU kernels: SiLU landed on the arena's first bytes and the
//! tensor kept its input values (errors of 0.3-3 vs CPU, e.g. a 6-channel
//! squeeze-excite gate in EfficientNet). The arena bytes it hit are checked
//! too, via a second output that must come back unchanged.
#![cfg(feature = "metal")]

mod common;

use common::skip_unless_available;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn silu_matches_cpu_for_every_tail_length() {
    if skip_unless_available(Device::Metal, "metal_silu_odd_len") {
        return;
    }
    for n in [1usize, 2, 3, 5, 6, 7, 9, 10, 1023, 1030] {
        let build = || {
            let mut g = Graph::new("silu");
            let keep = g.input("keep", Shape::new(&[64], DType::F32));
            let x = g.input("x", Shape::new(&[1, n, 1, 1], DType::F32));
            let y = g.silu(x);
            let k = g.relu(keep);
            g.set_outputs(vec![y, k]);
            g
        };
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let keep: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();
        let run = |dev| {
            Session::new(dev)
                .compile(build())
                .run(&[("keep", &keep), ("x", &xs)])
        };
        let (cpu, metal) = (run(Device::Cpu), run(Device::Metal));
        for (out, (a, b)) in cpu.iter().zip(&metal).enumerate() {
            let d = a
                .iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            assert!(d <= 1e-6, "n={n} output {out}: max |diff| {d}");
        }
    }
}
