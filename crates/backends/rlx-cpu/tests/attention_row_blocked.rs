// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Long-sequence attention takes the row-blocked path (score matrix past
//! `RLX_SDPA_BLOCK_MIN`, 4M): batch 2, a key-padding mask, rows on both sides
//! of block boundaries, checked against an f64 reference that skips masked
//! keys. A wrong block row offset, batch offset or head offset shows up as a
//! wrong number.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Op, Shape};

const B: usize = 2;
const S: usize = 2100; // 2100² = 4.41M scores per head
const NH: usize = 2;
const DH: usize = 64; // >= 64: the SME route takes both products

fn val(seed: usize, i: usize) -> f32 {
    let z = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5
}

#[test]
fn row_blocked_attention_matches_reference() {
    let f = DType::F32;
    let hs = NH * DH;
    let mut g = Graph::new("attn_rows");
    let q = g.input("q", Shape::new(&[B, S, hs], f));
    let k = g.input("k", Shape::new(&[B, S, hs], f));
    let v = g.input("v", Shape::new(&[B, S, hs], f));
    let mask = g.input("mask", Shape::new(&[B, S], f));
    let out = g.attention(q, k, v, mask, NH, DH, Shape::new(&[B, S, hs], f));
    g.set_outputs(vec![out]);
    let qd: Vec<f32> = (0..B * S * hs).map(|i| val(1, i)).collect();
    let kd: Vec<f32> = (0..B * S * hs).map(|i| val(2, i)).collect();
    let vd: Vec<f32> = (0..B * S * hs).map(|i| val(3, i)).collect();
    let valid = |b: usize, ki: usize| !(ki * 7 + b).is_multiple_of(11);
    let md: Vec<f32> = (0..B * S)
        .map(|i| if valid(i / S, i % S) { 1.0 } else { 0.0 })
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
            _ => &md,
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
        std::slice::from_raw_parts(p, B * S * hs).to_vec()
    };
    let scale = 1.0 / (DH as f64).sqrt();
    let mut worst = 0f64;
    for b in 0..B {
        for h in 0..NH {
            for &qi in &[
                0usize,
                1,
                31,
                32,
                33,
                499,
                500,
                1000,
                1023,
                1024,
                2047,
                2048,
                S - 1,
            ] {
                let qrow = &qd[(b * S + qi) * hs + h * DH..][..DH];
                let sc: Vec<f64> = (0..S)
                    .map(|ki| {
                        if !valid(b, ki) {
                            return f64::NEG_INFINITY;
                        }
                        let krow = &kd[(b * S + ki) * hs + h * DH..][..DH];
                        qrow.iter()
                            .zip(krow)
                            .map(|(&a, &c)| a as f64 * c as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|&x| (x - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..DH {
                    let want: f64 = (0..S)
                        .map(|ki| w[ki] * vd[(b * S + ki) * hs + h * DH + d] as f64)
                        .sum::<f64>()
                        / z;
                    let g = got[(b * S + qi) * hs + h * DH + d] as f64;
                    worst = worst.max((g - want).abs());
                }
            }
        }
    }
    assert!(worst < 1e-5, "max |d| {worst:e}");
}
