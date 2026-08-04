// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity guard for the packed-U8 **Constant** index path through
//! `Cast(u8→i64) → Gather` on Metal. This is the SynthMatMul-VJP reconstruction
//! (`rows = codebook[indices]`) that NaN-ed rlx-tiny on Metal: the u8 index
//! `Op::Constant` is stored 1-byte-packed, but the backward `Cast(u8→i64)` was
//! only routed to the true-1-byte-width host cast when its source was an
//! `Op::Param` — a u8 `Op::Constant` fell into the f32-wide `CastTruncF32` fast
//! path and read the packed bytes as 4-byte f32 → garbage indices → the Gather
//! reconstructed weights from the wrong rows (gradients exploded to 1e22 → NaN).
//!
//! Fix: `packed_int_src` in `thunk/compile.rs` now covers `Op::Constant` too.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const NE: usize = 256; // codebook rows (entries)
const D: usize = 4; // entry width
const P: usize = 4096; // number of gathered indices

fn gather_ref(codebook: &[f32], idx: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; P * D];
    for (i, &row) in idx.iter().enumerate() {
        let r = row as usize;
        for t in 0..D {
            out[i * D + t] = codebook[r * D + t];
        }
    }
    out
}

#[test]
fn metal_cast_u8_constant_gather_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // u8 index Constant (as rlx-tiny bakes SynthMatMul indices).
    let idx: Vec<u8> = (0..P).map(|i| ((i * 7 + 13) % NE) as u8).collect();
    // Non-trivial codebook so a wrong row is detectable and no row is all-zero.
    let codebook: Vec<f32> = (0..NE * D)
        .map(|i| (i as f32 * 0.017).sin() + 0.5)
        .collect();

    let mut g = Graph::new("cast_u8_const_gather");
    let cb = g.input("codebook", Shape::new(&[NE, D], DType::F32));
    let idx_u8 = g.add_node(
        Op::Constant { data: idx.clone() },
        vec![],
        Shape::new(&[P], DType::U8),
    );
    let idx_i64 = g.add_node(
        Op::Cast { to: DType::I64 },
        vec![idx_u8],
        Shape::new(&[P], DType::I64),
    );
    let rows = g.add_node(
        Op::Gather { axis: 0 },
        vec![cb, idx_i64],
        Shape::new(&[P, D], DType::F32),
    );
    g.set_outputs(vec![rows]);

    let mut m = Session::new(Device::Metal).compile(g);
    let out = m.run(&[("codebook", codebook.as_slice())]).remove(0);

    let reference = gather_ref(&codebook, &idx);

    assert!(
        out.iter().all(|v| v.is_finite()),
        "Metal Cast(u8-const)->Gather produced non-finite values"
    );
    let mut max_err = 0f32;
    for (a, b) in out.iter().zip(reference.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-5,
        "Metal Cast(u8-const)->Gather != CPU oracle (max_err {max_err:.3e}); \
         the packed u8 index constant was mis-read as f32"
    );
}
