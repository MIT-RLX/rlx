// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! How precise is wgpu's f32 matmul at transformer widths, against CPU?
//!
//! `rlx-brainbert` agrees with CPU to `cos = 1.000000` on every stage yet its
//! `head_out` relative L2 is 4.8e-4 on wgpu where CPU, Metal and MLX all sit at
//! ~2e-6 — a ~240x gap that accumulates through a 768-wide, 12-head transformer.
//! There is no structural error; the question is purely how the reduction is
//! accumulated. This measures that directly at the model's own K.

use rlx_ir::{DType, Graph, GraphExt, Shape};
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

fn case(m: usize, k: usize, n: usize) -> Option<f64> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let mut g = Graph::new("mm_prec");
    let a = g.input("a", Shape::new(&[m, k], DType::F32));
    let b = g.input("b", Shape::new(&[k, n], DType::F32));
    let y = g.mm(a, b);
    g.set_outputs(vec![y]);
    // Zero-mean inputs: cancellation is what exposes accumulation order, and a
    // transformer's activations are roughly zero-mean after LayerNorm.
    let av: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let bv: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.013).cos()).collect();
    let run = |d: Device| -> Vec<f32> {
        Session::new(d)
            .compile(g.clone())
            .run(&[("a", av.as_slice()), ("b", bv.as_slice())])
            .remove(0)
    };
    Some(rel_l2(&run(Device::Cpu), &run(Device::Gpu)))
}

#[test]
fn matmul_relative_error_at_transformer_widths() {
    // BrainBERT: d_h = 768, ffn = 3072, head_dim = 64.
    for (m, k, n) in [
        (64usize, 64usize, 64usize),
        (128, 768, 768),
        (128, 768, 3072),
    ] {
        let Some(r) = case(m, k, n) else { return };
        eprintln!("matmul [{m}x{k}]·[{k}x{n}]: rel_l2 = {r:.3e}");
        assert!(
            r < 1e-5,
            "wgpu matmul rel_l2 {r:.3e} at K={k} — an f32 reduction should be \
             far tighter than this; the accumulation is losing precision"
        );
    }
}
