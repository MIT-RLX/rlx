// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ArgMax / ArgMin on ROCm against the CPU: a middle axis (SynthSeg's labels,
//! a thread an output), long last-axis rows (a block a row) and short ones,
//! with ties, infinities and NaNs, where the first index of the strict best
//! must win as in the CPU's left-to-right scan.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn data(n: usize, seed: u32, levels: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Few levels: many ties.
            ((s >> 9) % levels) as f32
        })
        .collect()
}

fn arg(device: Device, dims: &[usize], axis: usize, max: bool, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("argreduce");
    let i = g.input("x", Shape::new(dims, DType::F32));
    let mut out: Vec<usize> = dims.to_vec();
    out[axis] = 1;
    let shape = Shape::new(&out, DType::F32);
    let y = if max {
        g.argmax(i, axis, true, shape)
    } else {
        g.argmin(i, axis, true, shape)
    };
    g.set_outputs(vec![y]);
    Session::new(device).compile(g).run(&[("x", x)]).remove(0)
}

#[test]
fn argreduce_matches_cpu() {
    for (dims, axis) in [
        (&[1usize, 33, 16, 16, 16][..], 1),
        (&[2, 7, 300][..], 1),
        (&[3, 5000][..], 1),
        (&[64, 50][..], 1),
        (&[4, 1][..], 1),
    ] {
        let n = dims.iter().product();
        let mut x = data(n, n as u32, 9);
        // Specials on the first rows: a NaN first element, a NaN later,
        // and infinities.
        let reduced = dims[axis];
        let inner: usize = dims[axis + 1..].iter().product();
        if reduced > 2 {
            x[0] = f32::NAN;
            x[inner + 1.min(inner - 1)] = f32::INFINITY;
            x[2 * inner] = f32::NEG_INFINITY;
            x[inner] = f32::NAN;
        }
        for max in [true, false] {
            let want = arg(Device::Cpu, dims, axis, max, &x);
            let got = arg(Device::Rocm, dims, axis, max, &x);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            eprintln!(
                "{dims:?} axis {axis} {}: {bad} of {} differ",
                if max { "max" } else { "min" },
                want.len()
            );
            assert_eq!(bad, 0, "{dims:?} axis {axis} max={max}");
        }
    }
}
