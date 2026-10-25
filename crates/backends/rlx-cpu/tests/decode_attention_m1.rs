// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-query (decode) attention with a key-padding mask and GQA, the shape a
//! bucketed KV cache produces: real keys, then padding, then the new token's
//! key in the bucket's last row. Checked against an f64 reference that only
//! ever looks at unmasked keys, so a kernel that lets a padded key in — or
//! mis-maps a query head onto its shared KV head — shows up as a wrong number.
//! The padded rows hold large garbage rather than zeros to make that visible.
//!
//! Key counts on both sides of the CPU's sequential-path threshold (32), so
//! both the short and the long decode routes are covered.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Op, Shape};

const NH: usize = 4;
const NKV: usize = 2;
const DH: usize = 16;

fn val(seed: usize, i: usize) -> f32 {
    let z = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5
}

fn run(ks: usize, real: usize) {
    let f = DType::F32;
    let mut g = Graph::new("decode_attn");
    let q = g.input("q", Shape::new(&[1, 1, NH * DH], f));
    let k = g.input("k", Shape::new(&[1, ks, NKV * DH], f));
    let v = g.input("v", Shape::new(&[1, ks, NKV * DH], f));
    let mask = g.input("mask", Shape::new(&[1, ks], f));
    let out = g.attention(q, k, v, mask, NH, DH, Shape::new(&[1, 1, NH * DH], f));
    g.set_outputs(vec![out]);

    // Keys 0..real and the last row are real; the rest is padding.
    let valid = |ki: usize| ki < real || ki == ks - 1;
    let qd: Vec<f32> = (0..NH * DH).map(|i| val(1, i)).collect();
    let pad = |ki: usize, x: f32| if valid(ki) { x } else { 1e3 };
    let kd: Vec<f32> = (0..ks * NKV * DH)
        .map(|i| pad(i / (NKV * DH), val(2, i)))
        .collect();
    let vd: Vec<f32> = (0..ks * NKV * DH)
        .map(|i| pad(i / (NKV * DH), val(3, i)))
        .collect();
    let md: Vec<f32> = (0..ks)
        .map(|ki| if valid(ki) { 1.0 } else { 0.0 })
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
        std::slice::from_raw_parts(p, NH * DH).to_vec()
    };

    let scale = 1.0 / (DH as f64).sqrt();
    for h in 0..NH {
        let kh = h / (NH / NKV);
        let keys: Vec<usize> = (0..ks).filter(|&ki| valid(ki)).collect();
        let s: Vec<f64> = keys
            .iter()
            .map(|&ki| {
                (0..DH)
                    .map(|d| qd[h * DH + d] as f64 * kd[ki * NKV * DH + kh * DH + d] as f64)
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let m = s.iter().cloned().fold(f64::MIN, f64::max);
        let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
        for d in 0..DH {
            let want: f64 = keys
                .iter()
                .zip(&s)
                .map(|(&ki, x)| (x - m).exp() / z * vd[ki * NKV * DH + kh * DH + d] as f64)
                .sum();
            let g = got[h * DH + d] as f64;
            assert!(
                (g - want).abs() < 1e-5,
                "ks={ks} head {h} dim {d}: {g} vs {want}"
            );
        }
    }
}

#[test]
fn decode_attention_ignores_padded_keys_short() {
    run(20, 7);
}

#[test]
fn decode_attention_ignores_padded_keys_long() {
    run(40, 13);
}
