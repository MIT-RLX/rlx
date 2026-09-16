// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Is a `set_param` write observed by the very next `run`?
//!
//! Sets no environment itself, so the caller controls `RLX_CUDA_EXEC_MODE` /
//! `RLX_CUDA_WHOLE_GRAPH_CAPTURE` and the answer is attributable to the config.
//!
//! `w = k·I` and `x` all ones, so the output should be exactly `k` everywhere.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 64;

fn main() {
    let mut g = Graph::new("scale");
    let x = g.input("x", Shape::new(&[N, N], DType::F32));
    let w = g.param("w", Shape::new(&[N, N], DType::F32));
    let y = g.mm(x, w);
    g.set_outputs(vec![y]);

    let mut c = Session::new(Device::Cuda).compile(g);
    let xv = vec![1.0f32; N * N];
    let mut bad = 0;
    for step in 1..=8u32 {
        let k = step as f32;
        let mut wd = vec![0.0f32; N * N];
        for i in 0..N {
            wd[i * N + i] = k;
        }
        c.set_param("w", &wd);
        let out = c.run(&[("x", &xv)]).pop().unwrap();
        let got = out[0];
        let ok = (got - k).abs() <= 1e-4 * k;
        if !ok {
            bad += 1;
        }
        println!(
            "  step {step}: want {k}, got {got}{}",
            if ok { "" } else { "   <-- STALE" }
        );
    }
    println!("{} of 8 steps stale", bad);
}
