// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AveragePool` when the last window overhangs its input.
//!
//! ONNX `ceil_mode = 1` rounds the window count UP, so the final window can
//! start inside the input and run off its end: 259 frames with kernel and
//! stride 100 gives three windows, the last covering frames 200..300 of which
//! only 59 exist. Two things used to go wrong there, and both were silent:
//!
//!   * the kernel's no-padding fast path assumed every window is in bounds and
//!     indexed past the buffer (a panic at best, neighbouring data at worst);
//!   * the mean divided by the nominal window size, scaling that last window by
//!     59/100 against the reference. ONNX's default is `count_include_pad = 0`,
//!     and an overhang is not padding — those positions do not exist.
//!
//! ChatterBox's speaker encoder pools exactly this way, and the second bug
//! survived the first: the shapes looked right and the embedding was wrong.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};

/// Average-pool `[1, 1, 1, len]` with a `kernel`/`stride` window, asking for
/// `out_len` windows (as `ceil_mode` would).
fn avg_pool_row(data: &[f32], kernel: usize, stride: usize, out_len: usize) -> Vec<f32> {
    let len = data.len();
    let mut g = Graph::new("pool");
    let x = g.input("x", Shape::new(&[1, 1, 1, len], DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: vec![1, kernel],
            stride: vec![1, stride],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, out_len], DType::F32),
    );
    g.set_outputs(vec![y]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    arena.slice_mut(x)[..len].copy_from_slice(data);
    execute_thunks(&sched, arena.raw_buf_mut());
    arena.slice(y)[..out_len].to_vec()
}

#[test]
fn the_overhanging_window_averages_only_what_exists() {
    // 259 "frames", kernel and stride 100 -> 3 windows under ceil_mode.
    let data: Vec<f32> = (0..259).map(|i| i as f32).collect();
    let out = avg_pool_row(&data, 100, 100, 3);

    let mean = |a: usize, b: usize| data[a..b].iter().sum::<f32>() / (b - a) as f32;
    assert!((out[0] - mean(0, 100)).abs() < 1e-3, "window 0: {}", out[0]);
    assert!(
        (out[1] - mean(100, 200)).abs() < 1e-3,
        "window 1: {}",
        out[1]
    );
    // The window that runs off the end: 59 real frames, divided by 59 — not by
    // the nominal 100, which would scale it by 0.59.
    assert!(
        (out[2] - mean(200, 259)).abs() < 1e-3,
        "overhanging window: got {}, want {} (dividing by the full kernel \
         would give {})",
        out[2],
        mean(200, 259),
        data[200..259].iter().sum::<f32>() / 100.0
    );
}

#[test]
fn fully_contained_windows_are_unchanged() {
    // The common case must keep taking the fast path and stay exact.
    let data: Vec<f32> = (0..300).map(|i| (i % 7) as f32).collect();
    let out = avg_pool_row(&data, 100, 100, 3);
    for (w, chunk) in out.iter().zip(data.chunks(100)) {
        let want = chunk.iter().sum::<f32>() / 100.0;
        assert!((w - want).abs() < 1e-4, "got {w}, want {want}");
    }
}
