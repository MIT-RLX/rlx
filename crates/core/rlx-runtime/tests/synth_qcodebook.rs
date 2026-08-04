// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Low-precision codebook for `Op::SynthMatMul` via `synth_matmul_qcodebook`:
//! the centroids are stored as fp8 / fp4 / nvf4 / custom `fpXmYeZ` codes and
//! decoded to f32 (Op::ScaledDequantize) before the synth matmul — reusing the
//! whole ScaledFormat/lowp_codec system with no new kernel. Verifies each format
//! tracks the f32-codebook result within its (format-dependent) quant error.

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape, SynthKind};
use rlx_runtime::{Device, Session};

const M: usize = 4;
const K: usize = 32;
const N: usize = 6;
const D: usize = 2;
const NE: usize = 8;

fn kb() -> usize {
    K / D
}
fn x_data() -> Vec<f32> {
    (0..M * K).map(|i| (i as f32 * 0.1).sin()).collect()
}
fn cb_data() -> Vec<f32> {
    (0..NE * D).map(|i| (i as f32 * 0.3).cos() * 0.8).collect()
}
fn idx_data() -> Vec<u8> {
    (0..N * kb()).map(|i| (i % NE) as u8).collect()
}
fn kind() -> SynthKind {
    SynthKind::Codebook {
        entry_dim: D as u32,
        num_entries: NE as u32,
    }
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb + 1e-12)
}

fn run(fmt: Option<ScaledFormat>) -> Vec<f32> {
    let mut g = Graph::new("qcb");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let cb = g.input("cb", Shape::new(&[NE, D], DType::F32));
    let idx = g.param("idx", Shape::new(&[N, kb()], DType::U8));
    let out_shape = Shape::new(&[M, N], DType::F32);
    let y = match fmt {
        None => g.synth_matmul(x, idx, cb, kind(), out_shape),
        Some(f) => {
            // Store the codebook as `f` codes (round-trip the f32 input through
            // the format), then decode + synth-matmul via the convenience builder.
            let (codes, scale) = g.scaled_quantize(cb, f, ScaleLayout::PerTensor);
            g.synth_matmul_qcodebook(
                x,
                idx,
                codes,
                scale,
                kind(),
                f,
                ScaleLayout::PerTensor,
                out_shape,
            )
        }
    };
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param_typed("idx", &idx_data(), DType::U8);
    c.run(&[("x", &x_data()), ("cb", &cb_data())])
        .pop()
        .unwrap()
}

#[test]
fn qcodebook_formats_track_f32() {
    let reference = run(None);
    // (format, min cosine vs the f32-codebook result). fp8 is near-exact; fp4
    // (16 levels) is coarse; custom f4e3m0 (exp-heavy 4-bit) coarser still.
    let cases = [
        (ScaledFormat::F8E4M3, 0.97f32),
        (ScaledFormat::F8E5M2, 0.95),
        (ScaledFormat::F4E2M1, 0.80),
        (ScaledFormat::custom(3, 0), 0.50), // f4e3m0
    ];
    for (fmt, tol) in cases {
        let out = run(Some(fmt));
        assert_eq!(out.len(), M * N);
        assert!(out.iter().all(|v| v.is_finite()), "{fmt:?}: non-finite");
        let cos = cosine(&reference, &out);
        eprintln!("qcodebook {fmt:?}: cos_vs_f32={cos:.4}");
        assert!(cos >= tol, "{fmt:?}: cosine {cos} < {tol}");
    }
}
