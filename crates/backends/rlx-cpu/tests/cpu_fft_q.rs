// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end CPU test for `Op::FftQ`, the fixed-point transform.
//!
//! The kernel itself is covered in `rlx-ir`; this checks the graph path —
//! that the op compiles to a thunk, reads and writes `I32` in the 2N-block
//! layout, and honours the scaling policy it was built with.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::fft::{FftNorm, FftQScale};
use rlx_ir::{DType, Graph, Op, Shape};

/// Build `fft_q` over one row of `n` complex points and run it on CPU.
fn run(data: &[i32], n: usize, inverse: bool, scale: FftQScale) -> Vec<i32> {
    let mut g = Graph::new("fft_q");
    let x = g.input("x", Shape::new(&[2 * n], DType::I32));
    let y = g.fft_q(x, inverse, FftNorm::Backward, scale);
    g.set_outputs(vec![y]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    for node in g.nodes() {
        if let Op::Input { .. } = &node.op {
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off).cast::<i32>();
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let off = arena.byte_offset(g.outputs[0]);
    unsafe {
        let p = arena.raw_buf_mut().as_ptr().add(off).cast::<i32>();
        (0..2 * n).map(|i| *p.add(i)).collect()
    }
}

fn tone(n: usize, bin: usize, amp: f64) -> Vec<i32> {
    let mut v = vec![0i32; 2 * n];
    for t in 0..n {
        v[t] = (amp * (2.0 * std::f64::consts::PI * bin as f64 * t as f64 / n as f64).cos()) as i32;
    }
    v
}

#[test]
fn a_tone_lands_in_its_bin() {
    let (n, bin) = (256usize, 20usize);
    let out = run(&tone(n, bin, 12000.0), n, false, FftQScale::None);
    let peak = (0..n / 2)
        .max_by_key(|&k| {
            let (r, i) = (i64::from(out[k]), i64::from(out[n + k]));
            r * r + i * i
        })
        .unwrap();
    assert_eq!(peak, bin, "peak at {peak}, expected {bin}");
    // A real cosine puts half its energy in each of ±f, so |X| = amp·n/2.
    let mag = {
        let (r, i) = (f64::from(out[bin]), f64::from(out[n + bin]));
        (r * r + i * i).sqrt()
    };
    let want = 12000.0 * n as f64 / 2.0;
    assert!((mag - want).abs() / want < 1e-3, "|X| {mag} vs {want}");
}

#[test]
fn round_trips_through_the_graph() {
    let n = 128usize;
    let x = tone(n, 7, 9000.0);
    let fwd = run(&x, n, false, FftQScale::None);
    let back = run(&fwd, n, true, FftQScale::None);
    // Unnormalised both ways, so the round trip scales by n.
    let peak = x.iter().map(|v| v.abs()).max().unwrap() as f64 * n as f64;
    for t in 0..n {
        let want = f64::from(x[t]) * n as f64;
        assert!(
            (f64::from(back[t]) - want).abs() / peak < 1e-3,
            "sample {t}: {} vs {want}",
            back[t]
        );
    }
}

/// The scaling policy reaches the kernel: `PerStage` divides the result by `n`.
#[test]
fn the_scaling_policy_survives_the_graph() {
    let n = 256usize;
    let x = tone(n, 11, 12000.0);
    let unscaled = run(&x, n, false, FftQScale::None);
    let staged = run(&x, n, false, FftQScale::PerStage);
    let mag = |v: &[i32], k: usize| {
        let (r, i) = (f64::from(v[k]), f64::from(v[n + k]));
        (r * r + i * i).sqrt()
    };
    let ratio = mag(&unscaled, 11) / mag(&staged, 11);
    assert!(
        (ratio - n as f64).abs() / (n as f64) < 0.02,
        "per-stage should be {n}x smaller, got {ratio:.1}x"
    );
}
