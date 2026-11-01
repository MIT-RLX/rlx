// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Device::OpenGl` claims every op its plan can run — native WebGL kernels
//! plus the rlx-cpu host fallback — so a session reaches `build_plan` with ops
//! like `Svd` / `Sample` instead of failing legalization. Ops that also have a
//! native lowering (`GroupNorm`, fusible matmul chains) must still come out
//! right now that the session no longer lowers them first.

#![cfg(feature = "opengl")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, SvdPart};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn vals(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

#[test]
fn opengl_session_runs_host_and_native_ops() {
    let mut g = Graph::new("webgl_session");
    // Host fallback.
    let a = g.input("a", sh(&[4, 3]));
    let s = g.svd(a, SvdPart::S, sh(&[3]));
    let l = g.input("l", sh(&[2, 6]));
    let tok = g.sample(l, 3, 1.0, 1.0, 5, sh(&[2]));
    // Native lowerings: GroupNorm, and a matmul → GELU → LayerNorm chain.
    let x = g.input("x", sh(&[1, 4, 3, 3]));
    let gam = g.input("gam", sh(&[4]));
    let bet = g.input("bet", sh(&[4]));
    let gn = g.group_norm(x, gam, bet, 2, 1e-5);
    let h = g.input("h", sh(&[2, 4]));
    let w = g.input("w", sh(&[4, 4]));
    let mm = g.matmul(h, w, sh(&[2, 4]));
    let act = g.activation(Activation::Gelu, mm, sh(&[2, 4]));
    let ln = g.ln(act, gam, bet, 1e-5);
    g.set_outputs(vec![s, tok, gn, ln]);

    let inputs: Vec<(&str, Vec<f32>)> = vec![
        ("a", vals(12, |i| (i as f32 * 0.7).sin())),
        ("l", vals(12, |i| (i as f32 * 0.3).cos())),
        ("x", vals(36, |i| (i as f32 * 0.41).sin())),
        ("gam", vals(4, |i| 0.5 + 0.25 * i as f32)),
        ("bet", vals(4, |i| 0.1 * i as f32)),
        ("h", vals(8, |i| i as f32 * 0.2 - 0.7)),
        ("w", vals(16, |i| 0.1 * (i % 5) as f32 - 0.2)),
    ];
    let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let cpu = Session::new(Device::Cpu).compile(g.clone()).run(&feed);
    let gl = Session::new(Device::OpenGl).compile(g).run(&feed);
    assert_eq!(cpu.len(), gl.len());
    for (oi, (r, w)) in cpu.iter().zip(&gl).enumerate() {
        assert_eq!(r.len(), w.len(), "output {oi} length");
        for (a, b) in r.iter().zip(w) {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}: cpu={a} webgl={b}"
            );
        }
    }
}
