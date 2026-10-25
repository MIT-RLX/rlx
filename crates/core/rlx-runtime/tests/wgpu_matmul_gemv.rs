// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu `m ≤ 4` matmuls (decode GEMVs) take `matmul_gemv`; check it against
//! the CPU backend across the cases its indexing has to get right: K not a
//! multiple of its 8-way split, N not a multiple of its 32-column tile, every
//! row count it handles, batched and broadcast right-hand sides, and a fused
//! bias + activation epilogue.

#![cfg(feature = "gpu")]

mod common;

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7919 + seed * 104729) % 1000) as f32) / 1000.0 - 0.5)
        .collect()
}

fn compare(g: Graph, feeds: &[(&str, Vec<f32>)]) {
    let f: Vec<(&str, &[f32])> = feeds.iter().map(|(n, d)| (*n, d.as_slice())).collect();
    let want = Session::new(Device::Cpu).compile(g.clone()).run(&f);
    let got = Session::new(Device::Gpu).compile(g).run(&f);
    for (o, (a, b)) in want.iter().zip(&got).enumerate() {
        assert_eq!(a.len(), b.len(), "output {o} length");
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-4, "output {o}: max |cpu - wgpu| = {d}");
    }
}

#[test]
fn gemv_rows_and_odd_shapes() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    for m in 1..=4usize {
        for (k, n) in [(37usize, 70usize), (256, 33), (1024, 1)] {
            let mut g = Graph::new("gemv");
            let x = g.input("x", sh(&[m, k]));
            let w = g.input("w", sh(&[k, n]));
            let y = g.matmul(x, w, sh(&[m, n]));
            g.set_outputs(vec![y]);
            compare(g, &[("x", data(m * k, 1)), ("w", data(k * n, 2))]);
        }
    }
}

#[test]
fn gemv_batched_and_broadcast() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    let (b, m, k, n) = (3usize, 2usize, 45usize, 40usize);
    // Matched batch: [B, M, K] x [B, K, N].
    let mut g = Graph::new("gemv_batched");
    let x = g.input("x", sh(&[b, m, k]));
    let w = g.input("w", sh(&[b, k, n]));
    let y = g.matmul(x, w, sh(&[b, m, n]));
    g.set_outputs(vec![y]);
    compare(g, &[("x", data(b * m * k, 3)), ("w", data(b * k * n, 4))]);
    // Broadcast rhs: [B, M, K] x [K, N] (leading dims fold into M = 6 > 4,
    // so this one checks the tiled path is still chosen and right).
    let mut g = Graph::new("gemv_broadcast");
    let x = g.input("x", sh(&[b, m, k]));
    let w = g.input("w", sh(&[k, n]));
    let y = g.matmul(x, w, sh(&[b, m, n]));
    g.set_outputs(vec![y]);
    compare(g, &[("x", data(b * m * k, 5)), ("w", data(k * n, 6))]);
}

#[test]
fn gemv_bias_and_activation() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    let (m, k, n) = (1usize, 300usize, 129usize);
    let mut g = Graph::new("gemv_bias_act");
    let x = g.input("x", sh(&[m, k]));
    let w = g.param("w", sh(&[k, n]));
    let bias = g.param("b", sh(&[n]));
    let y = g.matmul(x, w, sh(&[m, n]));
    let y = g.binary(BinaryOp::Add, y, bias, sh(&[m, n]));
    let y = g.activation(Activation::Silu, y, sh(&[m, n]));
    g.set_outputs(vec![y]);
    let (wd, bd) = (data(k * n, 7), data(n, 8));
    let xs = data(m * k, 9);
    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let mut gpu = Session::new(Device::Gpu).compile(g);
    for c in [&mut cpu, &mut gpu] {
        c.set_param("w", &wd);
        c.set_param("b", &bd);
    }
    let (a, b) = (cpu.run(&[("x", &xs)]), gpu.run(&[("x", &xs)]));
    let d = a[0]
        .iter()
        .zip(&b[0])
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(d < 1e-4, "max |cpu - wgpu| = {d}");
}

/// False where no wgpu adapter exists (a skip, or a failure under
/// `RLX_REQUIRE_DEVICE`).
fn common_gpu() -> bool {
    !common::skip_unless_available(Device::Gpu, "wgpu")
}

/// bf16 kept packed (`QuantScheme::GgufBF16`) rides the GGUF GEMV with a
/// wgpu-local scheme id. Even `k` takes the two-per-word read, odd `k` the
/// byte path; m > 1 runs one GEMV per row. Checked against a host matmul over
/// the same bf16 values (exact as f32) and against the CPU backend.
#[test]
fn bf16_dequant_matmul() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    use rlx_ir::quant::QuantScheme;
    for (m, k, n) in [(1usize, 64usize, 37usize), (3, 64, 33), (2, 45, 20)] {
        let x = data(m * k, 11);
        // Weights as bf16, GGUF row order [n, k].
        let w_bf16: Vec<u16> = data(n * k, 12)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let w_bytes: Vec<u8> = w_bf16.iter().flat_map(|h| h.to_le_bytes()).collect();
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = (0..k)
                    .map(|i| x[r * k + i] * f32::from_bits(u32::from(w_bf16[c * k + i]) << 16))
                    .sum();
            }
        }
        let mut g = Graph::new("bf16_dequant");
        let xi = g.input("x", sh(&[m, k]));
        let wi = g.param("w", Shape::new(&[n * k * 2], DType::U8));
        let y = g.add_node(
            rlx_ir::Op::DequantMatMul {
                scheme: QuantScheme::GgufBF16,
            },
            vec![xi, wi],
            sh(&[m, n]),
        );
        g.set_outputs(vec![y]);
        for dev in [Device::Cpu, Device::Gpu] {
            let mut c = Session::new(dev).compile(g.clone());
            c.set_param_typed("w", &w_bytes, DType::U8);
            let got = c.run(&[("x", &x)]);
            let d = got[0]
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(d < 1e-4, "{dev:?} m={m} k={k} n={n}: max |d| = {d}");
        }
    }
}

/// The GGUF block GEMVs (Q4_K, Q6_K, Q8_0 — the cooperative kernel, one
/// workgroup per output column) against an exact host reference
/// (`rlx_gguf::dequant_typed`, then an f64 dot), on blocks of random quants
/// with small finite scales; odd `n`, and m > 1 rows — including more rows
/// than one workgroup's 8 and a partial last group (the prefill case). The CPU backend is not
/// the reference: for some shapes it quantizes the activations to 8 bits
/// (Q8_K dot), ~0.2% off by design.
#[test]
fn gguf_block_gemv_matches_reference() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    use rlx_gguf::GgmlType;
    use rlx_ir::quant::QuantScheme;
    // (scheme, ggml type, elements per block, bytes per block, f16 scale offsets)
    let schemes: [(QuantScheme, GgmlType, usize, usize, &[usize]); 10] = [
        (QuantScheme::GgufQ4K, GgmlType::Q4K, 256, 144, &[0, 2]),
        (QuantScheme::GgufQ5K, GgmlType::Q5K, 256, 176, &[0, 2]),
        (QuantScheme::GgufQ6K, GgmlType::Q6K, 256, 210, &[208]),
        (QuantScheme::GgufQ8_0, GgmlType::Q8_0, 32, 34, &[0]),
        (QuantScheme::GgufQ4_0, GgmlType::Q4_0, 32, 18, &[0]),
        (QuantScheme::GgufQ4_1, GgmlType::Q4_1, 32, 20, &[0, 2]),
        (QuantScheme::GgufQ5_0, GgmlType::Q5_0, 32, 22, &[0]),
        (QuantScheme::GgufQ5_1, GgmlType::Q5_1, 32, 24, &[0, 2]),
        (QuantScheme::GgufIQ4NL, GgmlType::IQ4NL, 32, 18, &[0]),
        (QuantScheme::GgufIQ4XS, GgmlType::IQ4XS, 256, 136, &[0]),
    ];
    for (scheme, ty, be, bb, scale_at) in schemes {
        for (m, k, n) in [
            (1usize, 512usize, 37usize),
            (3, 1024, 70),
            (1, 256, 1),
            (11, 512, 37),
            (20, 256, 33),
        ] {
            let blocks = n * k / be;
            let mut w = vec![0u8; blocks * bb];
            for (i, b) in w.iter_mut().enumerate() {
                *b = ((i * 7919 + 13) % 256) as u8;
            }
            for blk in 0..blocks {
                for &at in scale_at {
                    // f16 ~0.01, sign varied by block.
                    let h: u16 = 0x211e | (((blk & 1) as u16) << 15);
                    w[blk * bb + at..blk * bb + at + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
            let x = data(m * k, 21);
            let wf = rlx_gguf::dequant_typed(ty, &w, n * k, "w").expect("dequant");
            let want: Vec<f32> = (0..m * n)
                .map(|o| {
                    let (r, j) = (o / n, o % n);
                    (0..k)
                        .map(|i| f64::from(x[r * k + i]) * f64::from(wf[j * k + i]))
                        .sum::<f64>() as f32
                })
                .collect();
            let mut g = Graph::new("gguf_gemv");
            let xi = g.input("x", sh(&[m, k]));
            let wi = g.param("w", Shape::new(&[w.len()], DType::U8));
            let y = g.add_node(
                rlx_ir::Op::DequantMatMul { scheme },
                vec![xi, wi],
                sh(&[m, n]),
            );
            g.set_outputs(vec![y]);
            let scale = want.iter().fold(1f32, |a, v| a.max(v.abs()));
            for dev in [Device::Cpu, Device::Gpu] {
                let mut c = Session::new(dev).compile(g.clone());
                c.set_param_typed("w", &w, DType::U8);
                let got = c.run(&[("x", &x)]).remove(0);
                let d = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{dev:?} {scheme:?} m={m} k={k} n={n}: max |ref - out| = {d:.2e}");
                if dev == Device::Gpu {
                    assert!(
                        d <= 1e-5 * scale,
                        "{scheme:?} m={m} k={k} n={n}: max |ref - wgpu| = {d} (scale {scale})"
                    );
                }
            }
        }
    }
}

/// Two executables over the same packed GGUF weight (a decode graph and a
/// prefill-shaped one) share one GPU copy: the second adopts the first's
/// weight buffer and computes the same as with its own upload. A third
/// executable of the same layout that is NOT told to share keeps its own
/// buffer, so writing different weights into it leaves the first untouched.
#[test]
fn gguf_weight_buffer_shared_between_executables() {
    if !common_gpu() {
        return;
    }
    let _gpu = common::serialize_gpu();
    use rlx_ir::quant::QuantScheme;
    let (k, n) = (512usize, 37usize);
    let blocks = n * k / 256;
    let weights = |seed: usize| {
        let mut w = vec![0u8; blocks * 144];
        for (i, b) in w.iter_mut().enumerate() {
            *b = ((i * 7919 + 13 + seed) % 256) as u8;
        }
        for blk in 0..blocks {
            for at in [0, 2] {
                let h: u16 = 0x211e | (((blk & 1) as u16) << 15);
                w[blk * 144 + at..blk * 144 + at + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
        w
    };
    let graph = |m: usize| {
        let mut g = Graph::new("gguf_shared");
        let xi = g.input("x", sh(&[m, k]));
        let wi = g.param("w", Shape::new(&[blocks * 144], DType::U8));
        let bi = g.param("b", sh(&[n]));
        let y = g.add_node(
            rlx_ir::Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K,
            },
            vec![xi, wi],
            sh(&[m, n]),
        );
        let z = g.binary(BinaryOp::Add, y, bi, sh(&[m, n]));
        g.set_outputs(vec![z]);
        g
    };
    let (w, w_other) = (weights(0), weights(1));
    let bias = data(n, 5);
    let run_own = |m: usize, w: &[u8], x: &[f32]| {
        let mut c = Session::new(Device::Gpu).compile(graph(m));
        c.set_param_typed("w", w, DType::U8);
        c.set_param("b", &bias);
        c.run(&[("x", x)]).remove(0)
    };
    let (x1, x3) = (data(k, 21), data(3 * k, 22));
    let want1 = run_own(1, &w, &x1);
    let want3 = run_own(3, &w, &x3);

    let mut first = Session::new(Device::Gpu).compile(graph(1));
    first.set_param_typed("w", &w, DType::U8);
    first.set_param("b", &bias);
    let mut second = Session::new(Device::Gpu).compile(graph(3));
    second.set_param("b", &bias);
    assert!(
        second.share_params_from(&first),
        "same layout: should share"
    );
    assert_eq!(second.run(&[("x", &x3)]).remove(0), want3);

    let mut other = Session::new(Device::Gpu).compile(graph(1));
    other.set_param_typed("w", &w_other, DType::U8);
    other.set_param("b", &bias);
    let other_out = other.run(&[("x", &x1)]).remove(0);
    assert_ne!(other_out, want1, "different weights should differ");
    assert_eq!(first.run(&[("x", &x1)]).remove(0), want1);
    assert_eq!(second.run(&[("x", &x3)]).remove(0), want3);

    // A different weight layout is refused.
    let mut g = Graph::new("gguf_other_layout");
    let xi = g.input("x", sh(&[1, k]));
    let wi = g.param("w", Shape::new(&[18 * k / 256 * 144], DType::U8));
    let y = g.add_node(
        rlx_ir::Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4K,
        },
        vec![xi, wi],
        sh(&[1, 18]),
    );
    g.set_outputs(vec![y]);
    let mut odd = Session::new(Device::Gpu).compile(g);
    assert!(
        !odd.share_params_from(&first),
        "different layout: must not share"
    );
}
