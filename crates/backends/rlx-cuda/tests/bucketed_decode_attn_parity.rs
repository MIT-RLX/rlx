// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.
//
// Bucketed decode SDPA on CUDA must match CPU (custom mask + concat KV).

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
            score_scale: None,
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
fn cuda_bucketed_decode_custom_mask_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }

    let (b, lq, lk, nh, dh) = (1, 1, 33, 16, 128);
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

    let graph = build_decode_attn(b, lq, lk, nh, dh);
    let inputs = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
        ("mask", mask.as_slice()),
    ];

    let mut cuda = Session::new(Device::Cuda).compile(graph.clone());
    let cuda_out = cuda.run(&inputs).remove(0);

    let mut cpu = Session::new(Device::Cpu).compile(graph);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &cuda_out);
    eprintln!("bucketed decode attn cuda vs cpu max_abs={d:.6}");
    assert!(
        d < 1e-3,
        "bucketed decode custom-mask attention diverged (max_abs={d})"
    );
}

#[test]
fn cuda_causal_decode_attn_hd128_matches_cpu() {
    if !rlx_cuda::is_available() {
        eprintln!("skip: CUDA unavailable");
        return;
    }

    let (b, lq, lk, nh, dh) = (1, 1, 37, 16, 128);
    let hs = nh * dh;
    let nq = b * lq * hs;
    let nk = b * lk * hs;

    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin()).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.013).cos()).collect();
    let v: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.011).sin()).collect();

    let f = DType::F32;
    let mut g = Graph::new("decode_causal_attn");
    let q_in = g.input("q", Shape::new(&[b, lq, hs], f));
    let k_in = g.input("k", Shape::new(&[b, lk, hs], f));
    let v_in = g.input("v", Shape::new(&[b, lk, hs], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q_in, k_in, v_in],
        Shape::new(&[b, lq, hs], f),
    );
    g.set_outputs(vec![y]);

    let inputs = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
    ];

    let mut cuda = Session::new(Device::Cuda).compile(g.clone());
    let cuda_out = cuda.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &cuda_out);
    eprintln!("causal decode attn lk=37 cuda vs cpu max_abs={d:.6}");
    assert!(d < 1e-3, "causal decode attention diverged (max_abs={d})");
}
