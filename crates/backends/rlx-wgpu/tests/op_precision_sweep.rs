// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Which op in a transformer block loses precision on wgpu?
//!
//! `rlx-brainbert`'s relative L2 is ~240x worse on wgpu than on CPU/Metal/MLX
//! while `cos = 1.000000` — no structural error, just accumulation. Matmul is
//! bit-exact (see `matmul_accum_precision`), so this checks the rest of the
//! block at the model's own 768 width.

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let den: f64 = a
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}

fn probe(tag: &str, build: impl Fn(&mut Graph, rlx_ir::NodeId) -> rlx_ir::NodeId) -> Option<f64> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return None;
    }
    let (rows, d) = (128usize, 768usize);
    let mut g = Graph::new(tag);
    let x = g.input("x", Shape::new(&[rows, d], DType::F32));
    let y = build(&mut g, x);
    g.set_outputs(vec![y]);
    // A large DC offset is what makes a one-pass variance cancel, and pre-norm
    // residual streams carry exactly that.
    let xs: Vec<f32> = (0..rows * d)
        .map(|i| ((i as f32) * 0.013).sin() * 3.0 + 12.0)
        .collect();
    let run = |dev: Device| -> Vec<f32> {
        Session::new(dev)
            .compile(g.clone())
            .run(&[("x", xs.as_slice())])
            .remove(0)
    };
    Some(rel_l2(&run(Device::Cpu), &run(Device::Gpu)))
}

#[test]
fn transformer_block_ops_keep_cpu_precision() {
    let d = 768usize;
    let cases: Vec<(
        &str,
        Box<dyn Fn(&mut Graph, rlx_ir::NodeId) -> rlx_ir::NodeId>,
    )> = vec![
        (
            "layer_norm",
            Box::new(move |g: &mut Graph, x| {
                let gm = g.param("g", Shape::new(&[d], DType::F32));
                let bt = g.param("b", Shape::new(&[d], DType::F32));
                g.add_node(
                    Op::LayerNorm {
                        axis: -1,
                        eps: 1e-5,
                    },
                    vec![x, gm, bt],
                    Shape::new(&[128, d], DType::F32),
                )
            }),
        ),
        ("gelu", Box::new(|g: &mut Graph, x| g.gelu(x))),
        ("silu", Box::new(|g: &mut Graph, x| g.silu(x))),
        ("softmax", Box::new(|g: &mut Graph, x| g.sm(x, 1))),
        ("tanh", Box::new(|g: &mut Graph, x| g.tanh(x))),
    ];
    let mut bad = Vec::new();
    for (tag, f) in cases {
        let Some(r) = probe(tag, |g, x| f(g, x)) else {
            eprintln!("skip: wgpu unavailable");
            return;
        };
        eprintln!("{tag:<12} rel_l2 = {r:.3e}");
        if r > 1e-5 {
            bad.push((tag, r));
        }
    }
    assert!(bad.is_empty(), "wgpu loses precision in: {bad:?}");
}
