// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Attention` with a logit soft-cap on the `[B, S, H·D]` layout, against
//! an f64 reference. Only the head-major (`[B, H, S, D]`) path used to apply
//! the cap; the sequential, row-blocked and strided paths ran uncapped.
//!
//! The reference follows the op's definition, which is the order the models'
//! own implementations use: scores `q·k·scale`, then `c · tanh(s / c)`, then
//! the mask (a masked key's score becomes `-1e9`; an additive bias is added),
//! then softmax. A masked key thus gets weight 0, not `exp(-c)`. The cap of 5
//! is small against scores of up to ±16, so a path that skips it, or applies it
//! after the mask, is far off.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};

const NH: usize = 2;
const DH: usize = 64;
const CAP: f32 = 5.0;

fn val(seed: usize, i: usize) -> f32 {
    let z = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5
}

/// Bias value for (head, query, key): drops every third key past the first.
fn bias(h: usize, q: usize, k: usize) -> f32 {
    if k > 0 && (k + q + h).is_multiple_of(3) {
        -1e4
    } else {
        0.0
    }
}

fn run(seq: usize, mask: MaskKind) -> f64 {
    let f = DType::F32;
    let hs = NH * DH;
    let mut g = Graph::new("attn_softcap");
    let q = g.input("q", Shape::new(&[1, seq, hs], f));
    let k = g.input("k", Shape::new(&[1, seq, hs], f));
    let v = g.input("v", Shape::new(&[1, seq, hs], f));
    let mut ins = vec![q, k, v];
    if matches!(mask, MaskKind::Bias) {
        ins.push(g.input("bias", Shape::new(&[1, NH, seq, seq], f)));
    }
    let out = g.add_node(
        Op::Attention {
            num_heads: NH,
            head_dim: DH,
            v_head_dim: None,
            mask_kind: mask,
            score_scale: Some(1.0),
            attn_logit_softcap: Some(CAP),
        },
        ins,
        Shape::new(&[1, seq, hs], f),
    );
    g.set_outputs(vec![out]);
    // Inputs scaled so q·k reaches about ±16.
    let qd: Vec<f32> = (0..seq * hs).map(|i| 2.0 * val(1, i)).collect();
    let kd: Vec<f32> = (0..seq * hs).map(|i| 2.0 * val(2, i)).collect();
    let vd: Vec<f32> = (0..seq * hs).map(|i| val(3, i)).collect();
    let bd: Vec<f32> = (0..NH * seq * seq)
        .map(|i| bias(i / (seq * seq), (i / seq) % seq, i % seq))
        .collect();

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    for node in g.nodes() {
        let Op::Input { name } = &node.op else {
            continue;
        };
        let data = match name.as_str() {
            "q" => &qd,
            "k" => &kd,
            "v" => &vd,
            _ => &bd,
        };
        let off = arena.byte_offset(node.id);
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let got: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(arena.byte_offset(out)) as *const f32;
        std::slice::from_raw_parts(p, seq * hs).to_vec()
    };

    let cap = f64::from(CAP);
    let mut worst = 0f64;
    for h in 0..NH {
        for qi in 0..seq {
            let qrow = &qd[qi * hs + h * DH..][..DH];
            let sc: Vec<f64> = (0..seq)
                .map(|ki| {
                    let krow = &kd[ki * hs + h * DH..][..DH];
                    let s: f64 = qrow
                        .iter()
                        .zip(krow)
                        .map(|(&a, &c)| f64::from(a) * f64::from(c))
                        .sum();
                    let s = cap * (s / cap).tanh();
                    match mask {
                        MaskKind::Causal if ki > qi => -1e9,
                        MaskKind::Bias => s + f64::from(bias(h, qi, ki)),
                        _ => s,
                    }
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for d in 0..DH {
                let want: f64 = (0..seq)
                    .map(|ki| w[ki] * f64::from(vd[ki * hs + h * DH + d]))
                    .sum::<f64>()
                    / z;
                worst = worst.max((f64::from(got[qi * hs + h * DH + d]) - want).abs());
            }
        }
    }
    worst
}

#[test]
fn softcap_sequential_path() {
    // At most `sdpa_seq_threshold` (32) rows: the sequential NEON path.
    for mask in [MaskKind::None, MaskKind::Causal, MaskKind::Bias] {
        let w = run(24, mask);
        assert!(w < 1e-5, "{mask:?}: max |d| {w:e}");
    }
}

#[test]
fn softcap_strided_path() {
    // Past the threshold: the strided / parallel path.
    for mask in [MaskKind::None, MaskKind::Causal, MaskKind::Bias] {
        let w = run(160, mask);
        assert!(w < 1e-5, "{mask:?}: max |d| {w:e}");
    }
}
