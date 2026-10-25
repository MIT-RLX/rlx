// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu greedy `Op::Sample` (the argmax kernel, a workgroup per row) against a
//! host argmax at LM-head size: 248320 logits (Qwen3.5's vocabulary), several
//! rows, all-negative rows, and ties — which go to the smaller index.

#![cfg(feature = "gpu")]

mod common;

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn host_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

#[test]
fn greedy_sample_matches_host_argmax() {
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let _gpu = common::serialize_gpu();
    for (rows, vocab) in [(1usize, 248_320usize), (3, 248_320), (5, 1000), (2, 1)] {
        let mut logits: Vec<f32> = (0..rows * vocab)
            .map(|i| (((i * 7919 + 13) % 100_003) as f32) / 1000.0 - 60.0)
            .collect();
        for r in 0..rows {
            let row = &mut logits[r * vocab..(r + 1) * vocab];
            match r % 3 {
                // A tie between two late positions: the smaller index wins.
                0 if vocab > 10 => {
                    row[vocab - 7] = 500.0;
                    row[vocab - 3] = 500.0;
                }
                // All negative.
                1 => row.iter_mut().for_each(|v| *v = -1000.0 - v.abs()),
                _ => {}
            }
        }
        let mut g = Graph::new("greedy");
        let x = g.input("logits", Shape::new(&[rows, 1, vocab], DType::F32));
        let y = g.sample(x, 0, 1.0, 1.0, 0, Shape::new(&[rows, 1], DType::F32));
        g.set_outputs(vec![y]);
        let got = Session::new(Device::Gpu)
            .compile(g)
            .run(&[("logits", &logits)]);
        for r in 0..rows {
            let want = host_argmax(&logits[r * vocab..(r + 1) * vocab]);
            assert_eq!(
                got[0][r] as usize, want,
                "rows={rows} vocab={vocab} row {r}"
            );
        }
    }
}
