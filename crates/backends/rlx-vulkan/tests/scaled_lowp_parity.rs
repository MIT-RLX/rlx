// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native `scaled_lowp_*.comp` (quant-scale / quantize / dequantize /
//! decode-GEMM) vs the rlx-cpu oracle, across every `ScaledFormat` and layout.
//!
//! Vulkan was routing all four to the CPU host fallback. The grouped (MoE)
//! variant next door already had a native decode kernel, but it is MXFP4-only,
//! so this is a full codec port (`lowp_codec.inc`) — and a codec port fails
//! quietly: a mis-decoded exponent bias produces plausible numbers, not an
//! error.
//!
//! Same two-gate split as the wgpu twin, for the same reason. Round-trip is
//! elementwise, so it must be **bit-exact**; the GEMM accumulates over a 16-wide
//! tile against rlx-cpu's straight loop, so it gets a relative tolerance. Both
//! fast-math hazards the wgpu port surfaced (`exp2` and bare `/`) are handled in
//! `lowp_codec.inc`; this is what proves it on a second driver.

use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape};
use rlx_vulkan::backend::VulkanExecutable;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Skip when no Vulkan device is present.
///
/// `rlx_ir::env::skip_unless_device` rather than a bare
/// `if !is_available() { return }`: the bare form reports `ok` on a rig with no
/// device, so a CI box that lost its Vulkan driver would look green. This one
/// honours `RLX_REQUIRE_DEVICE=1` and fails instead. (A local `fn available()`
/// wrapper hides the same problem from `require_device_coverage` without fixing
/// it.)
fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn formats() -> Vec<(&'static str, ScaledFormat)> {
    vec![
        ("e4m3", ScaledFormat::F8E4M3),
        ("e5m2", ScaledFormat::F8E5M2),
        ("e4m3fnuz", ScaledFormat::F8E4M3Fnuz),
        ("e5m2fnuz", ScaledFormat::F8E5M2Fnuz),
        ("e2m3", ScaledFormat::F6E2M3),
        ("e3m2", ScaledFormat::F6E3M2),
        ("e2m1", ScaledFormat::F4E2M1),
        ("custom f4e3m0", ScaledFormat::custom(3, 0)),
    ]
}

fn layouts() -> Vec<(&'static str, ScaleLayout)> {
    vec![
        ("per_tensor", ScaleLayout::PerTensor),
        ("mx_e8m0", ScaleLayout::mx()),
        ("nvfp4", ScaleLayout::nvfp4()),
    ]
}

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

fn cpu(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    use rlx::prelude::*;
    Session::new(Device::Cpu).compile(g).run(inputs).remove(0)
}

fn vk(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    VulkanExecutable::compile(g).run(inputs).remove(0)
}

fn roundtrip(rows: usize, cols: usize, fmt: ScaledFormat, layout: ScaleLayout) -> Graph {
    let mut g = Graph::new("lowp_rt");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let (codes, scale) = g.scaled_quantize(x, fmt, layout);
    let y = g.scaled_dequantize(codes, scale, fmt, layout);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn quantize_dequantize_roundtrip_is_bit_exact() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // Not a multiple of the MX block (32) or the NVFP4 group (16): the ragged
    // last block is exercised rather than assumed away.
    let (rows, cols) = (5usize, 40usize);
    let x = wave(rows * cols, 0.21, 3.0);
    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            let want = cpu(roundtrip(rows, cols, fmt, layout), &[("x", &x)]);
            let got = vk(roundtrip(rows, cols, fmt, layout), &[("x", &x)]);
            assert_eq!(got.len(), want.len(), "{fname}/{lname}: length");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fname}/{lname}: element {i} — vulkan {a} vs cpu {b} (input {})",
                    x[i]
                );
            }
        }
    }
}

#[test]
fn roundtrip_survives_the_hard_inputs() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // Zeros, denormals, values far past every format's max (the encoder's
    // saturate path), and exact negatives.
    let mut x = vec![
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        -1e-30,
        1e30,
        -1e30,
        6.0,
        -6.0,
        448.0,
        -448.0,
        57344.0,
        -57344.0,
        0.5,
        -0.5,
    ];
    x.extend(wave(16, 0.9, 900.0));
    let (rows, cols) = (2usize, 16usize);
    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            let want = cpu(roundtrip(rows, cols, fmt, layout), &[("x", &x)]);
            let got = vk(roundtrip(rows, cols, fmt, layout), &[("x", &x)]);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fname}/{lname}: element {i} — vulkan {a} vs cpu {b} (input {})",
                    x[i]
                );
            }
        }
    }
}

fn matmul(
    m: usize,
    k: usize,
    n: usize,
    fmt: ScaledFormat,
    layout: ScaleLayout,
    bias: bool,
) -> Graph {
    let mut g = Graph::new("lowp_mm");
    // `scaled_matmul_bias` takes f32 and inserts the quant-scale + quantize
    // chain itself, so one graph exercises all four kernels.
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.input("w", Shape::new(&[n, k], DType::F32));
    let b = if bias {
        Some(g.input("b", Shape::new(&[n], DType::F32)))
    } else {
        None
    };
    let y = g.scaled_matmul_bias(x, w, b, fmt, layout);
    g.set_outputs(vec![y]);
    g
}

fn close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let rel = (a - b).abs() / scale;
        worst = worst.max(rel);
        assert!(
            rel <= 1e-5,
            "{what}: element {i} — vulkan {a} vs cpu {b} (rel {rel:e})"
        );
    }
    eprintln!("{what}: max rel diff {worst:.2e}");
}

#[test]
fn scaled_matmul_decode_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // m and n straddle the 16x16 tile so the edge guards run; k is a multiple of
    // neither the tile nor the MX block.
    let (m, k, n) = (18usize, 40usize, 20usize);
    let x = wave(m * k, 0.13, 1.5);
    let w = wave(n * k, 0.07, 1.2);
    let b = wave(n, 0.31, 0.4);
    for (fname, fmt) in formats() {
        for (lname, layout) in layouts() {
            for bias in [false, true] {
                let mut inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w)];
                if bias {
                    inputs.push(("b", &b));
                }
                let want = cpu(matmul(m, k, n, fmt, layout, bias), &inputs);
                let got = vk(matmul(m, k, n, fmt, layout, bias), &inputs);
                close(&format!("{fname}/{lname}/bias={bias}"), &got, &want);
            }
        }
    }
}

#[test]
fn scaled_matmul_decode_handles_a_single_tile() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // Everything smaller than one 16x16 tile — the all-edges case.
    let (m, k, n) = (3usize, 7usize, 5usize);
    let x = wave(m * k, 0.41, 2.0);
    let w = wave(n * k, 0.17, 1.7);
    for (fname, fmt) in formats() {
        let layout = ScaleLayout::mx();
        let want = cpu(matmul(m, k, n, fmt, layout, false), &[("x", &x), ("w", &w)]);
        let got = vk(matmul(m, k, n, fmt, layout, false), &[("x", &x), ("w", &w)]);
        close(&format!("{fname}/single-tile"), &got, &want);
    }
}
