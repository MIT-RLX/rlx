// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::TopK` on every available backend returns the `k` largest along the
//! last axis **in order**: value descending, ties to the smaller index. The
//! set alone is not the contract — o-means reads its probe cells in rank
//! order, and MLX once returned the right set in index order.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::Session;

mod common;

fn reference(x: &[f32], cols: usize, k: usize) -> Vec<f32> {
    x.chunks_exact(cols)
        .flat_map(|row| {
            let mut idx: Vec<usize> = (0..cols).collect();
            idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            idx.into_iter().take(k).map(|i| i as f32)
        })
        .collect()
}

#[test]
fn topk_is_sorted_with_ties_to_the_smaller_index() {
    let _gpu = common::serialize_gpu();
    let (rows, cols) = (6, 32);
    let mut s = 3u64;
    let mut x: Vec<f32> = (0..rows * cols)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Few distinct values: plenty of ties, negatives included.
            ((s >> 59) as f32) - 8.0
        })
        .collect();
    // An all-tied row, and a row with the largest values late.
    x[cols..2 * cols].fill(1.5);
    for (j, v) in x[2 * cols..3 * cols].iter_mut().enumerate() {
        *v = j as f32;
    }
    let mut failed = Vec::new();
    for k in [1, 5, 16, 32] {
        let want = reference(&x, cols, k);
        for dev in rlx_runtime::available_devices() {
            let mut g = Graph::new("topk");
            let inp = g.input("x", Shape::new(&[rows, cols], DType::F32));
            let y = g.add_node(
                Op::TopK { k },
                vec![inp],
                Shape::new(&[rows, k], DType::F32),
            );
            g.set_outputs(vec![y]);
            let got = Session::new(dev)
                .compile(g)
                .run(&[("x", &x)])
                .pop()
                .unwrap();
            if got != want {
                let row = got
                    .chunks(k)
                    .zip(want.chunks(k))
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                eprintln!(
                    "{dev:?} k={k} row {row}: got {:?}, want {:?}",
                    &got[row * k..(row + 1) * k],
                    &want[row * k..(row + 1) * k]
                );
                failed.push((dev, k));
            }
        }
    }
    assert!(failed.is_empty(), "TopK order wrong on {failed:?}");
}
