// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct forward check for `Op::Slice` at every step, against the exact answer.
//!
//! Exists because `fd_backward_gate` flagged `slice_positive_step` on ROCm with a
//! finite difference of exactly zero — which says the sliced elements had no
//! influence on the output, i.e. the FORWARD was wrong, not the gradient. This
//! isolates that: no autodiff, no cotangents, just slice-vs-truth.
//!
//! `RLX_PROBE_DEVICE=rocm|cuda|metal|gpu|cpu` (default cpu).

use rlx_ir::{DType, Graph, Op, Shape};

fn main() {
    let dev = rlx_ir::env::var("RLX_PROBE_DEVICE")
        .and_then(|s| rlx_runtime::parse_device(&s).ok())
        .unwrap_or(rlx_runtime::Device::Cpu);
    if !rlx_runtime::is_available(dev) {
        println!("{dev:?} unavailable");
        return;
    }
    let n = 8usize;
    let mut bad = 0usize;
    // (step, start, len) chosen so every index stays inside [0, n).
    for &(step, start, len) in &[
        (1i64, 0usize, 4usize),
        (2, 0, 4),
        (3, 0, 3),
        (-1, 7, 4),
        (-2, 7, 4),
    ] {
        let mut g = Graph::new("slice");
        let x = g.input("x", Shape::new(&[n], DType::F32));
        let y = g.add_node(
            Op::Slice {
                axis: 0,
                start,
                len,
                step,
            },
            vec![x],
            Shape::new(&[len], DType::F32),
        );
        g.set_outputs(vec![y]);
        let xv: Vec<f32> = (0..n).map(|i| i as f32 * 10.0).collect();
        let want: Vec<f32> = (0..len)
            .map(|j| {
                let idx = start as i64 + j as i64 * step;
                assert!(
                    (0..n as i64).contains(&idx),
                    "probe bug: step={step} start={start} len={len} indexes {idx}"
                );
                xv[idx as usize]
            })
            .collect();
        let got = rlx_runtime::Session::new(dev).compile(g).run(&[("x", &xv)])[0].clone();
        let ok =
            got.len() == want.len() && got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6);
        if !ok {
            bad += 1;
        }
        println!(
            "step={step:>3} start={start} len={len}  want={want:?}  got={got:?}  {}",
            if ok { "OK" } else { "*** WRONG ***" }
        );
    }
    println!("{} of 5 step configurations wrong on {dev:?}", bad);
    if bad > 0 {
        std::process::exit(1);
    }
}
