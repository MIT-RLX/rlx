// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bucketed decode SDPA on wgpu vs CPU (`MaskKind::Custom`, Lq=1).

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_decode_attn(b: usize, lq: usize, lk: usize, nh: usize, dh: usize) -> Graph {
    let f = DType::F32;
    let hs = nh * dh;
    let mut g = Graph::new("decode_custom_mask_attn");
    let q = g.input("q", Shape::new(&[b, lq, hs], f));
    let k = g.input("k", Shape::new(&[b, lk, hs], f));
    let v = g.input("v", Shape::new(&[b, lk, hs], f));
    let mask = g.input("mask", Shape::new(&[b, lk], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
            score_scale: Some(1.0),
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, lq, hs], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn wgpu_bucketed_decode_custom_mask_matches_cpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let (b, lq, lk, nh, dh) = (1, 1, 33, 4, 256);
    let hs = nh * dh;
    let nq = b * lq * hs;
    let nk = b * lk * hs;

    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin()).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.013).cos()).collect();
    let v: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.011).sin()).collect();

    let past_seq = 17usize;
    let upper = 32usize;
    let mut mask = vec![0f32; lk];
    for (i, slot) in mask.iter_mut().enumerate() {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }

    let g = build_decode_attn(b, lq, lk, nh, dh);
    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[("q", &q), ("k", &k), ("v", &v), ("mask", &mask)])
            .remove(0)
    };
    let cpu = run(Device::Cpu);
    let gpu = run(Device::Gpu);
    let d = max_abs(&cpu, &gpu);
    eprintln!("wgpu decode custom-mask attn max_abs={d:.6e}");
    assert!(d < 1e-2, "wgpu decode attention max_abs {d}");
}
