// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Rope` execution on Metal vs CPU, packed `[B, S, H*D]`, head_dim 128
//! (Voxtral LM dims). Other tests cover attention/rmsnorm/swiglu but RoPE-apply
//! was only checked by formula inspection — this actually runs the kernel.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_rope(b: usize, s: usize, h: usize, d: usize) -> Graph {
    build_rope_styled(b, s, h, d, rlx_ir::RopeStyle::NeoX)
}

fn build_rope_styled(b: usize, s: usize, h: usize, d: usize, style: rlx_ir::RopeStyle) -> Graph {
    let f = DType::F32;
    let w = h * d;
    let half = d / 2;
    let mut g = Graph::new("rope");
    let x = g.input("x", Shape::new(&[b, s, w], f));
    // cos/sin tables: one row of `head_dim/2` per sequence position.
    let cos = g.input("cos", Shape::new(&[s, half], f));
    let sin = g.input("sin", Shape::new(&[s, half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim: d,
            n_rot: d,
            style,
        },
        vec![x, cos, sin],
        Shape::new(&[b, s, w], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_rope_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, s, h, d) = (1, 64, 32, 128);
    let half = d / 2;
    let x: Vec<f32> = (0..b * s * h * d)
        .map(|i| ((i as f32) * 0.0007).sin())
        .collect();
    // Realistic rotary tables: theta_j = pos * base^(-2j/d).
    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for p in 0..s {
        for j in 0..half {
            let freq = 1.0f32 / (100_000_000.0f32).powf((2 * j) as f32 / d as f32);
            let ang = p as f32 * freq;
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }

    let g = build_rope(b, s, h, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!("rope (hd=128): max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}");
    assert!(max_abs < 1e-4, "rope max_abs={max_abs}");
}

/// GPT-J / llama.cpp-NORM interleaved RoPE (`RopeStyle::GptJ`): rotated pairs
/// are adjacent `(2i, 2i+1)` rather than NeoX rotate-half `(i, i+d/2)`. GGUF
/// Llama weights need this flavor; the Metal kernel must match CPU and produce
/// a result distinct from NeoX. Mirrors rlx-mlx `rope_gptj_interleaved_*`.
#[test]
fn metal_rope_gptj_interleaved_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, s, h, d) = (1, 64, 32, 128);
    let half = d / 2;
    let x: Vec<f32> = (0..b * s * h * d)
        .map(|i| ((i as f32) * 0.0007).sin())
        .collect();
    let mut cos = vec![0f32; s * half];
    let mut sin = vec![0f32; s * half];
    for p in 0..s {
        for j in 0..half {
            let freq = 1.0f32 / (100_000_000.0f32).powf((2 * j) as f32 / d as f32);
            let ang = p as f32 * freq;
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }

    let g = build_rope_styled(b, s, h, d, rlx_ir::RopeStyle::GptJ);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("gptj rope (hd=128): max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "gptj rope Metal vs CPU max_abs={max_abs}");

    // Interleaved must differ from NeoX: same inputs, NeoX graph on CPU, then
    // confirm the Metal GptJ output is not (numerically) the NeoX result.
    let g_neox = build_rope(b, s, h, d);
    let mut cn = Session::new(Device::Cpu).compile(g_neox);
    let cpu_neox = cn.run(&[("x", &x), ("cos", &cos), ("sin", &sin)]).remove(0);
    let differ = cpu_neox
        .iter()
        .zip(metal.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(differ, "GptJ output identical to NeoX ⇒ style ignored");
}

/// Ragged batched decode: `seq = 1` but the cos/sin tables carry **one row per
/// batch element** (each sequence at its own absolute position). Metal must
/// apply per-token RoPE and match the CPU reference — guards the
/// `cos_per_token` kernel path that makes ragged continuous batching correct on
/// Metal.
#[test]
fn metal_rope_ragged_per_token_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, h, d) = (4usize, 8usize, 64usize);
    let half = d / 2;
    let f = DType::F32;
    let w = h * d;

    let mut g = Graph::new("rope_ragged");
    let x = g.input("x", Shape::new(&[b, 1, w], f)); // decode: seq = 1
    let cos = g.input("cos", Shape::new(&[b, half], f)); // one row per batch token
    let sin = g.input("sin", Shape::new(&[b, half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim: d,
            n_rot: d,
            style: rlx_ir::RopeStyle::NeoX,
        },
        vec![x, cos, sin],
        Shape::new(&[b, 1, w], f),
    );
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = (0..b * w).map(|i| ((i as f32) * 0.0013).cos()).collect();
    // Each batch element sits at a DIFFERENT absolute position.
    let positions = [3usize, 7, 1, 12];
    let mut cos_d = vec![0f32; b * half];
    let mut sin_d = vec![0f32; b * half];
    for (bi, &p) in positions.iter().enumerate() {
        for j in 0..half {
            let freq = 1.0f32 / (1_000_000.0f32).powf((2 * j) as f32 / d as f32);
            let ang = p as f32 * freq;
            cos_d[bi * half + j] = ang.cos();
            sin_d[bi * half + j] = ang.sin();
        }
    }

    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m
        .run(&[("x", &x_data), ("cos", &cos_d), ("sin", &sin_d)])
        .remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c
        .run(&[("x", &x_data), ("cos", &cos_d), ("sin", &sin_d)])
        .remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("ragged rope: max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "ragged rope Metal vs CPU max_abs={max_abs}");

    // Sanity: batch elements at different positions must NOT all be identical
    // (i.e. per-token really happened, not a shared row).
    let row0 = &metal[0..w];
    let row1 = &metal[w..2 * w];
    let differ = row0.iter().zip(row1).any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(differ, "rows identical ⇒ per-token RoPE not applied");
}

/// Partial rotary: `n_rot < head_dim`, the shape Gemma 4 global layers,
/// DeepSeek-V4 MLA and MiniMax-H3 all use.
///
/// The cos/sin table holds exactly `n_rot/2` angles per token. Striding it by
/// `head_dim/2` instead reads into the *next* token's angles from position 1
/// onward — every position but the first comes out wrong, and the failure is a
/// plausible-looking rotation rather than a crash.
fn build_rope_partial(b: usize, s: usize, h: usize, d: usize, n_rot: usize) -> Graph {
    let f = DType::F32;
    let w = h * d;
    let rot_half = n_rot / 2;
    let mut g = Graph::new("rope_partial");
    let x = g.input("x", Shape::new(&[b, s, w], f));
    // One row of `n_rot/2` per position — NOT head_dim/2.
    let cos = g.input("cos", Shape::new(&[s, rot_half], f));
    let sin = g.input("sin", Shape::new(&[s, rot_half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim: d,
            n_rot,
            style: rlx_ir::RopeStyle::NeoX,
        },
        vec![x, cos, sin],
        Shape::new(&[b, s, w], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_partial_rope_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // 96 of 128 (MiniMax-H3 DiT) and 48 of 64 (its video VAE decoder).
    for (d, n_rot) in [(128usize, 96usize), (64, 48), (128, 64)] {
        let (b, s, h) = (1usize, 16usize, 4usize);
        let rot_half = n_rot / 2;
        let x: Vec<f32> = (0..b * s * h * d)
            .map(|i| ((i as f32) * 0.0011).sin())
            .collect();
        let mut cos = vec![0f32; s * rot_half];
        let mut sin = vec![0f32; s * rot_half];
        for p in 0..s {
            for j in 0..rot_half {
                let freq = 1.0f32 / (10_000.0f32).powf((2 * j) as f32 / n_rot as f32);
                let ang = p as f32 * freq;
                cos[p * rot_half + j] = ang.cos();
                sin[p * rot_half + j] = ang.sin();
            }
        }
        let inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("cos", &cos), ("sin", &sin)];

        let mut cpu = Session::new(Device::Cpu).compile(build_rope_partial(b, s, h, d, n_rot));
        let want = cpu.run(&inputs).remove(0);
        let mut gpu = Session::new(Device::Metal).compile(build_rope_partial(b, s, h, d, n_rot));
        let got = gpu.run(&inputs).remove(0);

        assert_eq!(got.len(), want.len());
        let max = want
            .iter()
            .zip(&got)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max < 1e-5,
            "partial rope head_dim={d} n_rot={n_rot}: max abs diff {max} vs CPU"
        );
        // The pass-through tail must be copied verbatim.
        for t in 0..s {
            for hi in 0..h {
                for j in n_rot..d {
                    let i = ((t * h) + hi) * d + j;
                    assert_eq!(got[i], x[i], "channel {j} should pass through unrotated");
                }
            }
        }
    }
}
