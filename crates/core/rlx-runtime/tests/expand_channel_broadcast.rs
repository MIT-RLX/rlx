// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `[C]` vector broadcast to `[N, C, H, W]` through reshape + `Expand`
//! (how a conv bias reaches its feature map) equals the plain broadcast, for
//! shapes with size-1 axes too: `[N, C, 1, W]` reaches the CPU backend's
//! per-channel fill with a stride on its size-1 axis.

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn broadcast(c: usize, target: [usize; 4]) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("expand_channel");
    let b = g.input("b", Shape::new(&[c], f));
    let b4 = g.reshape(b, vec![1, c as i64, 1, 1], Shape::new(&[1, c, 1, 1], f));
    let out = g.add_node(
        Op::Expand {
            target_shape: target.iter().map(|&d| d as i64).collect(),
        },
        vec![b4],
        Shape::new(&target, f),
    );
    g.set_outputs(vec![out]);
    let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.25 - 1.0).collect();
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("b", &bias)])
        .remove(0)
}

#[test]
fn a_channel_vector_broadcasts_to_every_position() {
    for (c, target) in [
        (16, [32, 16, 1, 257]),
        (40, [32, 40, 1, 232]),
        (8, [32, 8, 8, 257]),
        (3, [1, 3, 1, 1]),
        (5, [2, 5, 4, 1]),
        (16, [1, 16, 1, 65]),
    ] {
        let got = broadcast(c, target);
        let [n, _, h, w] = target;
        assert_eq!(got.len(), n * c * h * w);
        for (k, v) in got.iter().enumerate() {
            let ch = (k / (h * w)) % c;
            assert_eq!(
                v.to_bits(),
                (ch as f32 * 0.25 - 1.0).to_bits(),
                "target {target:?} at {k}"
            );
        }
    }
}
