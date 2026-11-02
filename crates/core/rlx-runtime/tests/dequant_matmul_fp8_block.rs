// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::DequantMatMul` with `Fp8E4m3B128` / `Fp8E5m2B128` — FP8 weights with
//! an f32 scale per 128 along K, the HF `fp8` / DeepSeek-V3 block layout — on
//! every available backend, against `x · (decode(code) · scale)ᵀ` from
//! `rlx_ir::lowp_codec`. Covers decode (`m = 1`, the GEMV paths), the 2–4-row
//! verify GEMV and prefill shapes, and every finite code of each format
//! (subnormals, ±max, −0).

use rlx_ir::lowp_codec::fp8_b128;
use rlx_ir::quant::{QuantScheme, ScaledFormat};
use rlx_ir::*;
use rlx_runtime::{Device, Session};

mod common;

/// Packed `[n, k]` weight, its f32 reference, from every finite code of `fmt`
/// cycled with per-(row, block) scales.
fn weight(fmt: ScaledFormat, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let finite: Vec<u8> = (0..=255u8).filter(|&c| fmt.decode(c).is_finite()).collect();
    let codes: Vec<u8> = (0..n * k)
        .map(|i| finite[(i * 31 + i / k) % finite.len()])
        .collect();
    let nb = k / fp8_b128::QK;
    // Scales well away from powers of two, so a dropped or misplaced scale
    // shows up.
    let scales: Vec<f32> = (0..n * nb)
        .map(|t| 0.0013 * (1.0 + ((t * 7) % 13) as f32))
        .collect();
    let packed = fp8_b128::pack(&codes, &scales, n, k).unwrap();
    let w = (0..n * k)
        .map(|t| fmt.decode(codes[t]) * scales[(t / k) * nb + (t % k) / fp8_b128::QK])
        .collect();
    (packed, w)
}

fn run_case(device: Device, scheme: QuantScheme, m: usize, k: usize, n: usize) {
    let fmt = fp8_b128::format(scheme).unwrap();
    let (packed, w) = weight(fmt, n, k);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32) * 0.02 - 1.0)
        .collect();

    let mut g = Graph::new("fp8_block");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_q = g.param("w_q", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w_q],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w_q", &packed, DType::U8);
    let got = c.run(&[("x", &x)]).pop().unwrap();
    assert_eq!(got.len(), m * n);

    let mut max_rel = 0f64;
    for r in 0..m {
        for col in 0..n {
            let want: f64 = (0..k)
                .map(|i| x[r * k + i] as f64 * w[col * k + i] as f64)
                .sum();
            let scale: f64 = (0..k)
                .map(|i| (x[r * k + i] as f64 * w[col * k + i] as f64).abs())
                .sum();
            let v = got[r * n + col] as f64;
            let rel = (v - want).abs() / scale.max(1e-6);
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-5,
                "{device:?} {scheme} m={m} k={k} n={n} [{r}, {col}]: {v} vs {want} (rel {rel:.2e})"
            );
        }
    }
    eprintln!("{device:?} {scheme} m={m} k={k} n={n} max_rel={max_rel:.2e}");
}

const DEVICES: &[Device] = &[
    Device::Cpu,
    Device::Metal,
    Device::Mlx,
    Device::Gpu,
    Device::Vulkan,
    Device::Cuda,
    Device::Rocm,
];

#[test]
fn fp8_block_dequant_matmul_matches_reference_on_every_backend() {
    let _gpu = common::serialize_gpu();
    for &device in DEVICES {
        if common::skip_unless(device) {
            continue;
        }
        for scheme in [QuantScheme::Fp8E4m3B128, QuantScheme::Fp8E5m2B128] {
            for (m, k, n) in [
                (1, 128, 5),
                (1, 1024, 37),
                (2, 256, 11),
                (4, 384, 9),
                (17, 512, 64),
            ] {
                run_case(device, scheme, m, k, n);
            }
        }
    }
}

/// `Op::DequantGroupedMatMul` (MoE) over a stack of block-FP8 expert slabs:
/// row `r` uses expert `expert_idx[r]`'s `[n, k]` weight. Distinct per-expert
/// scales make a mis-routed row obvious; out-of-order ids exercise the sort /
/// unpermute paths.
fn run_grouped_case(device: Device, scheme: QuantScheme, m: usize, k: usize, n: usize) {
    let fmt = fp8_b128::format(scheme).unwrap();
    let experts = 3;
    let mut packed = Vec::new();
    let mut ws = Vec::new();
    for e in 0..experts {
        let (p, w) = weight(fmt, n, k);
        let s = (e + 1) as f32;
        // Rescale expert `e` by rewriting each block's f32 scale.
        let mut p = p;
        for b in p.as_chunks_mut::<{ fp8_b128::BLOCK_BYTES }>().0 {
            let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * s;
            b[..4].copy_from_slice(&v.to_le_bytes());
        }
        packed.extend_from_slice(&p);
        ws.push(w.iter().map(|v| v * s).collect::<Vec<f32>>());
    }
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 37 % 101) as f32) * 0.02 - 1.0)
        .collect();
    let idx: Vec<f32> = (0..m).map(|r| ((r * 2 + 1) % experts) as f32).collect();

    let mut g = Graph::new("fp8_block_grouped");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_q = g.param("w_q", Shape::new(&[packed.len()], DType::U8));
    let idx_in = g.input("expert_idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::DequantGroupedMatMul { scheme },
        vec![x_in, w_q, idx_in],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("w_q", &packed, DType::U8);
    let got = c.run(&[("x", &x), ("expert_idx", &idx)]).pop().unwrap();
    for r in 0..m {
        let w = &ws[idx[r] as usize];
        for col in 0..n {
            let terms = (0..k).map(|i| x[r * k + i] as f64 * w[col * k + i] as f64);
            let want: f64 = terms.clone().sum();
            let scale: f64 = terms.map(f64::abs).sum();
            let v = got[r * n + col] as f64;
            let rel = (v - want).abs() / scale.max(1e-6);
            assert!(
                rel < 1e-5,
                "{device:?} grouped {scheme} m={m} [{r}, {col}]: {v} vs {want} (rel {rel:.2e})"
            );
        }
    }
    eprintln!("{device:?} grouped {scheme} m={m} k={k} n={n} ok");
}

#[test]
fn fp8_block_dequant_grouped_matmul_matches_reference_on_every_backend() {
    let _gpu = common::serialize_gpu();
    for &device in DEVICES {
        if common::skip_unless(device) {
            continue;
        }
        for scheme in [QuantScheme::Fp8E4m3B128, QuantScheme::Fp8E5m2B128] {
            for m in [1, 2, 7] {
                run_grouped_case(device, scheme, m, 256, 24);
            }
        }
    }
}

/// Two executables of one model under a param-sharing scope: on Metal a large
/// block-FP8 weight moves to the shared weight buffer, so the decode GEMV
/// (`m = 1`) and the dequant-to-scratch prefill path (`m = 4`) both read it
/// from there, through one copy. The weight is above the externalization
/// threshold (256 KiB); a path that still read the activation arena at the
/// weight's offset would produce garbage.
#[test]
fn fp8_block_weight_shared_across_executables() {
    let _gpu = common::serialize_gpu();
    // The split normally waits for half a gigabyte of weights.
    rlx_ir::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "1");
    for &device in DEVICES {
        if common::skip_unless(device) {
            continue;
        }
        for scheme in [QuantScheme::Fp8E4m3B128, QuantScheme::Fp8E5m2B128] {
            let fmt = fp8_b128::format(scheme).unwrap();
            let (k, n) = (1024, 320);
            let (packed, w) = weight(fmt, n, k);
            let label = format!("fp8-shared-{scheme}");
            let mut graphs = Vec::new();
            for m in [1usize, 4] {
                let mut g = Graph::new("fp8_block_shared");
                let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
                let w_q = g.param("w_q", Shape::new(&[packed.len()], DType::U8));
                let y = g.add_node(
                    Op::DequantMatMul { scheme },
                    vec![x_in, w_q],
                    Shape::new(&[m, n], DType::F32),
                );
                g.set_outputs(vec![y]);
                let mut c = rlx_runtime::with_param_sharing_scope(&label, || {
                    Session::new(device).compile(g)
                });
                if !c.param_already_set("w_q") {
                    c.set_param_typed("w_q", &packed, DType::U8);
                }
                graphs.push((m, c));
            }
            for (m, c) in &mut graphs {
                let m = *m;
                let x: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 37 % 101) as f32) * 0.02 - 1.0)
                    .collect();
                let got = c.run(&[("x", &x)]).pop().unwrap();
                for r in 0..m {
                    for col in 0..n {
                        let terms = (0..k).map(|i| x[r * k + i] as f64 * w[col * k + i] as f64);
                        let want: f64 = terms.clone().sum();
                        let scale: f64 = terms.map(f64::abs).sum();
                        let v = got[r * n + col] as f64;
                        let rel = (v - want).abs() / scale.max(1e-6);
                        assert!(
                            rel < 1e-5,
                            "{device:?} shared {scheme} m={m} [{r}, {col}]: {v} vs {want} (rel {rel:.2e})"
                        );
                    }
                }
            }
            eprintln!("{device:?} shared {scheme} ok");
        }
    }
}
