// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `fold_strided_bmm`: multi-head attention whose Q / Kᵀ / V are
//! `narrow → reshape → transpose → reshape` copies of one qkv projection. On
//! Metal-backed wgpu the two batched matmuls become `rlx_wgpu.strided_bmm`
//! nodes that read the qkv tensor through strided views (no narrow /
//! transpose copies). The softmax sits BETWEEN the two matmuls, so the qkv
//! tensor must stay alive past it — the rewrite makes it the matmuls' input so
//! the planner knows. Compared against CPU at ragged sizes (head_dim 25, 191
//! tokens) and with the attention output also returned, so a dropped copy that
//! something else still needed would show up.

use rlx_ir::infer::GraphExt as _;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn attention(b: usize, s: usize, h: usize, dh: usize, keep_q: bool, merge: bool) -> Graph {
    let hd = h * dh;
    let mut g = Graph::new("mha");
    let x = g.input("x", Shape::new(&[b, s, hd], F));
    let w = g.input("w", Shape::new(&[hd, 3 * hd], F));
    let qkv = g.mm(x, w); // [b, s, 3hd]
    let bsh = vec![b as i64, s as i64, h as i64, dh as i64];
    let q0 = g.narrow_(qkv, 2, 0, hd);
    let k0 = g.narrow_(qkv, 2, hd, hd);
    let v0 = g.narrow_(qkv, 2, 2 * hd, hd);
    let q = g.reshape_(q0, bsh.clone());
    let k = g.reshape_(k0, bsh.clone());
    let v = g.reshape_(v0, bsh);
    let bh = (b * h) as i64;
    let qh = g.transpose_(q, vec![0, 2, 1, 3]);
    let qh = g.reshape_(qh, vec![bh, s as i64, dh as i64]);
    let kt = g.transpose_(k, vec![0, 2, 3, 1]);
    let kt = g.reshape_(kt, vec![bh, dh as i64, s as i64]);
    let vh = g.transpose_(v, vec![0, 2, 1, 3]);
    let vh = g.reshape_(vh, vec![bh, s as i64, dh as i64]);
    let scores = g.mm(qh, kt);
    let p = g.softmax(scores, -1, Shape::new(&[b * h, s, s], F));
    let ctx = g.mm(p, vh);
    // `merge`: heads back to [b, s, h·dh] — the matmul writes that layout itself.
    let ctx = if merge {
        let c4 = g.reshape_(ctx, vec![b as i64, h as i64, s as i64, dh as i64]);
        let ct = g.transpose_(c4, vec![0, 2, 1, 3]);
        g.reshape_(ct, vec![b as i64, s as i64, hd as i64])
    } else {
        ctx
    };
    // `keep_q`: the Q head copy is ALSO a graph output, so it must survive.
    g.set_outputs(if keep_q { vec![ctx, qh] } else { vec![ctx] });
    g
}

fn check(b: usize, s: usize, h: usize, dh: usize, keep_q: bool, merge: bool) {
    let hd = h * dh;
    let x: Vec<f32> = noise(b * s * hd, 1).iter().map(|v| v * 0.5).collect();
    let w: Vec<f32> = noise(hd * 3 * hd, 2).iter().map(|v| v * 0.2).collect();
    let inputs: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];
    let build = || attention(b, s, h, dh, keep_q, merge);
    let cpu = Session::new(Device::Cpu).compile(build()).run(&inputs);
    let gpu = Session::new(Device::Gpu).compile(build()).run(&inputs);
    assert_eq!(cpu.len(), gpu.len());
    for (o, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert_eq!(c.len(), g.len());
        let scale = c.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let worst = c
            .iter()
            .zip(g)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst / scale < 1e-5,
            "mha b{b} s{s} h{h} dh{dh} keep_q={keep_q} merge={merge} output {o}: rel err {:.3e}",
            worst / scale
        );
    }
}

#[test]
fn strided_attention_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    check(2, 191, 8, 25, false, false); // EEG-DINO Small shape class
    check(1, 37, 3, 16, false, false);
    check(2, 33, 4, 8, true, false); // a head copy that is also an output
    check(2, 191, 8, 25, false, true); // merged-heads output written in place
    check(2, 33, 4, 8, true, true);
}
