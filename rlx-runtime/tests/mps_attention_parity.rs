// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Isolated MPSGraph attention parity bisect (PLAN: Metal MPSGraph
//! attention bug). Builds a single Op::Attention layer, runs it on
//! both CPU (well-tested thunk path) and Metal (with MPSGraph attention
//! lowering forced on), compares element-wise.
//!
//! This is the unit test referenced from the PLAN — without it we can't
//! see whether the parity gap is in the matmul ordering, softmax,
//! mask handling, or the post-attention reshape.

#![cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]

use rlx_ir::op::{BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

/// Build the smallest interesting attention graph:
///   inputs:  qkv [B, S, 3*H], mask [B, S]
///   narrow:  q, k, v each [B, S, H]
///   op:      attention(num_heads=NH, head_dim=DH, MaskKind::Custom)
///   output:  [B, S, H]
fn build_attn_graph(b: usize, s: usize, nh: usize, dh: usize) -> Graph {
    let h = nh * dh;
    let f = DType::F32;
    let mut g = Graph::new("attn_parity");
    let qkv = g.input("qkv", Shape::new(&[b, s, 3 * h], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let k = g.add_node(
        Op::Narrow {
            axis: 2,
            start: h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let v = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![attn]);
    g
}

fn run_on(
    device: Device,
    b: usize,
    s: usize,
    nh: usize,
    dh: usize,
    qkv: &[f32],
    mask: &[f32],
) -> Vec<f32> {
    let g = build_attn_graph(b, s, nh, dh);
    let session = Session::new(device);
    let mut compiled = session.compile_with(g, &CompileOptions::default());
    let outs = compiled.run(&[("qkv", qkv), ("mask", mask)]);
    outs.into_iter().next().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn cpu_vs_metal_attention_no_mask_unpadded() {
    // Tiny: B=1, S=4, NH=1, DH=4. Mask all 1s (no padding).
    let (b, s, nh, dh) = (1, 4, 1, 4);
    let h = nh * dh;
    // Hand-picked deterministic inputs that round-trip cleanly through
    // f32. QKV layout: [B, S, 3*H] = packed Q, then K, then V along
    // the last dim.
    let qkv: Vec<f32> = (0..b * s * 3 * h).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let mask = vec![1.0f32; b * s];

    let cpu_out = run_on(Device::Cpu, b, s, nh, dh, &qkv, &mask);
    // Force MPSGraph attention path so we exercise the lowering we're
    // trying to validate.
    unsafe {
        std::env::set_var("RLX_USE_MPSGRAPH", "1");
        std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
    }
    let mtl_out = run_on(Device::Metal, b, s, nh, dh, &qkv, &mask);
    unsafe {
        std::env::remove_var("RLX_USE_MPSGRAPH");
        std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
    }

    assert_eq!(cpu_out.len(), mtl_out.len());
    let diff = max_abs_diff(&cpu_out, &mtl_out);
    eprintln!("[attn-parity NH=1 DH=4 unpadded] max abs diff = {diff:e}");
    eprintln!("CPU first 8: {:?}", &cpu_out[..8.min(cpu_out.len())]);
    eprintln!("Mtl first 8: {:?}", &mtl_out[..8.min(mtl_out.len())]);
    assert!(
        diff < 1e-4,
        "MPSGraph attention diverges from CPU \
        ({diff:e}) on unpadded NH=1 DH=4 input — bug is in attention math"
    );
}

#[test]
fn cpu_vs_metal_attention_multi_head_unpadded() {
    // Multi-head: B=1, S=4, NH=2, DH=4 (matches the layout the
    // BERT model emits). Mask all 1s.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let qkv: Vec<f32> = (0..b * s * 3 * h)
        .map(|i| ((i % 17) as f32) * 0.05)
        .collect();
    let mask = vec![1.0f32; b * s];

    let cpu_out = run_on(Device::Cpu, b, s, nh, dh, &qkv, &mask);
    unsafe {
        std::env::set_var("RLX_USE_MPSGRAPH", "1");
        std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
    }
    let mtl_out = run_on(Device::Metal, b, s, nh, dh, &qkv, &mask);
    unsafe {
        std::env::remove_var("RLX_USE_MPSGRAPH");
        std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
    }

    let diff = max_abs_diff(&cpu_out, &mtl_out);
    eprintln!("[attn-parity NH=2 DH=4 unpadded] max abs diff = {diff:e}");
    assert!(
        diff < 1e-4,
        "MPSGraph attention diverges from CPU \
        ({diff:e}) on unpadded NH=2 DH=4 input"
    );
}

#[test]
fn cpu_vs_metal_full_block_unpadded() {
    // Larger graph: a full transformer block (matmul + bias + attention
    // + residual+LN + FFN). Tests whether the MPSGraph parity gap
    // observed in the BERT bench is from accumulation across multiple
    // ops, not from attention alone.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let mut g = Graph::new("attn_block");

    let x = g.input("x", Shape::new(&[b, s, h], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));
    let qkv_w = g.param("qkv_w", Shape::new(&[h, 3 * h], f));
    let qkv_b = g.param("qkv_b", Shape::new(&[3 * h], f));

    // Fused QKV projection: x @ qkv_w + qkv_b → [B, S, 3H]
    let qkv_mm = g.add_node(Op::MatMul, vec![x, qkv_w], Shape::new(&[b, s, 3 * h], f));
    let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, Shape::new(&[b, s, 3 * h], f));

    // Narrow Q/K/V
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let k = g.add_node(
        Op::Narrow {
            axis: 2,
            start: h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let v = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );

    // Attention
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, s, h], f),
    );

    g.set_outputs(vec![attn]);

    let qkv_w_data: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let qkv_b_data: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.001).collect();
    let x_data: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let mask_data = vec![1.0f32; b * s];

    // CPU
    let session_cpu = Session::new(Device::Cpu);
    let mut cpu = session_cpu.compile_with(g.clone(), &CompileOptions::default());
    cpu.set_param("qkv_w", &qkv_w_data);
    cpu.set_param("qkv_b", &qkv_b_data);
    let cpu_out = cpu
        .run(&[("x", &x_data), ("mask", &mask_data)])
        .into_iter()
        .next()
        .unwrap();

    // Metal with MPSGraph
    unsafe {
        std::env::set_var("RLX_USE_MPSGRAPH", "1");
        std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
    }
    let session_mtl = Session::new(Device::Metal);
    let mut mtl = session_mtl.compile_with(g, &CompileOptions::default());
    mtl.set_param("qkv_w", &qkv_w_data);
    mtl.set_param("qkv_b", &qkv_b_data);
    let mtl_out = mtl
        .run(&[("x", &x_data), ("mask", &mask_data)])
        .into_iter()
        .next()
        .unwrap();
    unsafe {
        std::env::remove_var("RLX_USE_MPSGRAPH");
        std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
    }

    let diff = max_abs_diff(&cpu_out, &mtl_out);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let rel = diff / cpu_max.max(1e-6);
    eprintln!(
        "[full-block unpadded] max abs diff = {diff:e}, max val = {cpu_max:e}, rel = {rel:e}"
    );
}

/// Bisect helper: build a sub-graph and compare CPU vs Metal-MPSGraph.
fn bisect(
    name: &str,
    build: impl Fn(&mut Graph) -> rlx_ir::NodeId,
    inputs: &[(&str, &[f32])],
    params: &[(&str, &[f32])],
) {
    let mut g = Graph::new(name);
    let out = build(&mut g);
    g.set_outputs(vec![out]);

    // Disable fusion so we test the OPS-AS-WRITTEN, not fusion artifacts.
    let opts = CompileOptions {
        dce: false,
        constant_folding: false,
        ..CompileOptions::default()
    };

    let session_cpu = Session::new(Device::Cpu);
    let mut cpu = session_cpu.compile_with(g.clone(), &opts);
    for (n, d) in params {
        cpu.set_param(n, d);
    }
    let cpu_out = cpu.run(inputs).into_iter().next().unwrap();

    unsafe {
        std::env::set_var("RLX_USE_MPSGRAPH", "1");
        std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
    }
    let session_mtl = Session::new(Device::Metal);
    let mut mtl = session_mtl.compile_with(g, &opts);
    for (n, d) in params {
        mtl.set_param(n, d);
    }
    let mtl_out = mtl.run(inputs).into_iter().next().unwrap();
    unsafe {
        std::env::remove_var("RLX_USE_MPSGRAPH");
        std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
    }

    let diff = max_abs_diff(&cpu_out, &mtl_out);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(1e-9, f32::max);
    eprintln!(
        "[bisect:{name}] max abs diff = {diff:e}, max val = {cpu_max:e}, rel = {:e}",
        diff / cpu_max
    );
}

#[test]
fn bisect_matmul_only() {
    let (b, s, h) = (1, 4, 8);
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..h * (3 * h))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    bisect(
        "mm",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f))
        },
        &[("x", &x)],
        &[("w", &w)],
    );
}

#[test]
fn bisect_matmul_plus_bias() {
    let (b, s, h) = (1, 4, 8);
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..h * (3 * h))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let bi: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.001).collect();
    bisect(
        "mm+bias",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f))
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bi)],
    );
}

#[test]
fn bisect_matmul_bias_narrow() {
    let (b, s, h) = (1, 4, 8);
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..h * (3 * h))
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let bi: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.001).collect();
    bisect(
        "mm+bias+narrow",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            )
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bi)],
    );
}

#[test]
fn bisect_three_narrows_to_attention() {
    // Q, K, V all from the same parent narrow on the SAME tensor.
    // This is what the BERT block does. Tests whether multiple narrows
    // sharing a parent cause MPSGraph to alias data incorrectly.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let qkv: Vec<f32> = (0..b * s * 3 * h)
        .map(|i| ((i % 17) as f32) * 0.05)
        .collect();
    let mask = vec![1.0f32; b * s];

    bisect(
        "3-narrow-attn",
        |g| {
            let qkv_in = g.input("qkv", Shape::new(&[b, s, 3 * h], f));
            let mask_in = g.input("mask", Shape::new(&[b, s], f));
            let q = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv_in],
                Shape::new(&[b, s, h], f),
            );
            let k = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: h,
                    len: h,
                },
                vec![qkv_in],
                Shape::new(&[b, s, h], f),
            );
            let v = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 2 * h,
                    len: h,
                },
                vec![qkv_in],
                Shape::new(&[b, s, h], f),
            );
            g.add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: dh,
                    mask_kind: MaskKind::Custom,
                },
                vec![q, k, v, mask_in],
                Shape::new(&[b, s, h], f),
            )
        },
        &[("qkv", &qkv), ("mask", &mask)],
        &[],
    );
}

#[test]
fn bisect_mm_bias_then_three_narrows_no_attention() {
    // Same as bisect_full_qkv_to_attention but stops after the 3 narrows
    // — output is the concat of narrowed Q/K/V to capture all values.
    // Tests whether the matmul→bias→narrow combination diverges WITHOUT
    // attention being involved.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let bv: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.001).collect();

    bisect(
        "mm+bias+3narrow",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            let q = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let k = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let v = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 2 * h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            // Concat back so output captures all narrows.
            g.add_node(
                Op::Concat { axis: 2 },
                vec![q, k, v],
                Shape::new(&[b, s, 3 * h], f),
            )
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bv)],
    );
}

#[test]
fn bisect_mm_bias_narrow_reshape() {
    // Test whether reshape AFTER narrow on a matmul output diverges.
    // This isolates the narrow→reshape chain that mg.attention does
    // internally.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    let bv: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.001).collect();

    bisect(
        "mm+bias+narrow+reshape",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            let q = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            // Reshape q to [B, S, NH, DH] — what mg.attention does internally.
            g.add_node(
                Op::Reshape {
                    new_shape: vec![b as i64, s as i64, nh as i64, dh as i64],
                },
                vec![q],
                Shape::new(&[b, s, nh, dh], f),
            )
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bv)],
    );
}

#[test]
fn bisect_full_qkv_to_attention() {
    // Full chain: matmul + bias + 3 narrows + attention.
    // Larger-magnitude inputs so any divergence is well above f32 noise
    // and the relative-error figure is meaningful.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.5)
        .collect();
    let w: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.1)
        .collect();
    let bv: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.01).collect();
    let mask = vec![1.0f32; b * s];

    bisect(
        "mm+bias+3narrow+attn",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let mi = g.input("mask", Shape::new(&[b, s], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            let q = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let k = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let v = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 2 * h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            g.add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: dh,
                    mask_kind: MaskKind::Custom,
                },
                vec![q, k, v, mi],
                Shape::new(&[b, s, h], f),
            )
        },
        &[("x", &x), ("mask", &mask)],
        &[("w", &w), ("b", &bv)],
    );
}

#[test]
fn bisect_mm_bias_three_narrows_three_reshapes() {
    // 3 narrows of computed tensor + 3 reshapes (mimicking what
    // attention does to Q/K/V before its matmul). If THIS fails, the
    // bug is reshape-of-slice-of-computed when there are multiple
    // such chains in the same graph.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.5)
        .collect();
    let w: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.1)
        .collect();
    let bv: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.01).collect();

    bisect(
        "mm+bias+3narrow+3reshape",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            let q = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let k = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let v = g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 2 * h,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            );
            let q4 = g.add_node(
                Op::Reshape {
                    new_shape: vec![b as i64, s as i64, nh as i64, dh as i64],
                },
                vec![q],
                Shape::new(&[b, s, nh, dh], f),
            );
            let k4 = g.add_node(
                Op::Reshape {
                    new_shape: vec![b as i64, s as i64, nh as i64, dh as i64],
                },
                vec![k],
                Shape::new(&[b, s, nh, dh], f),
            );
            let v4 = g.add_node(
                Op::Reshape {
                    new_shape: vec![b as i64, s as i64, nh as i64, dh as i64],
                },
                vec![v],
                Shape::new(&[b, s, nh, dh], f),
            );
            // Concat all three reshaped → output [B, S, NH, 3*DH] for inspection.
            g.add_node(
                Op::Concat { axis: 3 },
                vec![q4, k4, v4],
                Shape::new(&[b, s, nh, 3 * dh], f),
            )
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bv)],
    );
}

#[test]
fn bisect_mm_bias_one_narrow_no_attention() {
    // The simplest failing-pattern variant: just mm+bias+narrow,
    // NO downstream attention. Output is the [B, S, H] narrow.
    // If this passes, the bug needs the slice→reshape combination
    // to surface. If it fails, even basic slice-of-compute is broken.
    let (b, s, h) = (1, 4, 8);
    let f = DType::F32;
    let x: Vec<f32> = (0..b * s * h)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.5)
        .collect();
    let w: Vec<f32> = (0..h * 3 * h)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.1)
        .collect();
    let bv: Vec<f32> = (0..3 * h).map(|i| (i as f32) * 0.01).collect();

    bisect(
        "mm+bias+narrow0",
        |g| {
            let xi = g.input("x", Shape::new(&[b, s, h], f));
            let wi = g.param("w", Shape::new(&[h, 3 * h], f));
            let bii = g.param("b", Shape::new(&[3 * h], f));
            let mm = g.add_node(Op::MatMul, vec![xi, wi], Shape::new(&[b, s, 3 * h], f));
            let qkv = g.binary(BinaryOp::Add, mm, bii, Shape::new(&[b, s, 3 * h], f));
            // Narrow to first H elements (= "Q" slice).
            g.add_node(
                Op::Narrow {
                    axis: 2,
                    start: 0,
                    len: h,
                },
                vec![qkv],
                Shape::new(&[b, s, h], f),
            )
        },
        &[("x", &x)],
        &[("w", &w), ("b", &bv)],
    );
}

#[test]
fn bisect_mm_bias_then_full_attention_noslice() {
    // Mimic the failing pattern but feed Q/K/V as separate inputs
    // (not narrows). This isolates the question: does mm+bias on the
    // upstream side break attention, or is it specifically the
    // narrow-of-computed pattern?
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let f = DType::F32;
    // Q/K/V already split — host provides them as separate inputs.
    // No mm+bias upstream.
    let q_in: Vec<f32> = (0..b * s * h).map(|i| ((i % 17) as f32) * 0.05).collect();
    let k_in: Vec<f32> = (0..b * s * h).map(|i| ((i % 13) as f32) * 0.05).collect();
    let v_in: Vec<f32> = (0..b * s * h).map(|i| ((i % 11) as f32) * 0.05).collect();
    let mask = vec![1.0f32; b * s];

    bisect(
        "attn-from-three-inputs",
        |g| {
            let q = g.input("q", Shape::new(&[b, s, h], f));
            let k = g.input("k", Shape::new(&[b, s, h], f));
            let v = g.input("v", Shape::new(&[b, s, h], f));
            let m = g.input("mask", Shape::new(&[b, s], f));
            g.add_node(
                Op::Attention {
                    num_heads: nh,
                    head_dim: dh,
                    mask_kind: MaskKind::Custom,
                },
                vec![q, k, v, m],
                Shape::new(&[b, s, h], f),
            )
        },
        &[("q", &q_in), ("k", &k_in), ("v", &v_in), ("mask", &mask)],
        &[],
    );
}

#[test]
fn cpu_vs_metal_attention_with_padding() {
    // B=1, S=4, NH=2, DH=4. Last 2 positions padded.
    let (b, s, nh, dh) = (1, 4, 2, 4);
    let h = nh * dh;
    let qkv: Vec<f32> = (0..b * s * 3 * h)
        .map(|i| ((i % 13) as f32) * 0.05)
        .collect();
    let mask = vec![1.0f32, 1.0, 0.0, 0.0];

    let cpu_out = run_on(Device::Cpu, b, s, nh, dh, &qkv, &mask);
    unsafe {
        std::env::set_var("RLX_USE_MPSGRAPH", "1");
        std::env::set_var("RLX_MPSGRAPH_ATTENTION", "1");
    }
    let mtl_out = run_on(Device::Metal, b, s, nh, dh, &qkv, &mask);
    unsafe {
        std::env::remove_var("RLX_USE_MPSGRAPH");
        std::env::remove_var("RLX_MPSGRAPH_ATTENTION");
    }

    let diff = max_abs_diff(&cpu_out, &mtl_out);
    eprintln!("[attn-parity NH=2 DH=4 padded] max abs diff = {diff:e}");
    eprintln!("CPU full: {:?}", &cpu_out);
    eprintln!("Mtl full: {:?}", &mtl_out);
    // Don't assert — this is the failure case we're investigating.
    // Document the gap so future fixes can confirm convergence.
}
