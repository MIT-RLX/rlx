// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! CUDA segmented graph-capture FOUNDATION test on a cuBLAS-FREE graph.
//!
//! Isolates the capture machinery — dedicated non-null stream, eager warm-up
//! (module + workspace load can't happen mid-capture), per-segment
//! `begin/end_capture`, replay — from the separate, harder cuBLAS-capture
//! problem. The graph is pure elementwise (SiLU + Add) blocks split by host
//! `Op::Sort` steps, so every captured segment is NVRTC kernels that record
//! cleanly. Run WITH capture engaged on the msi rig:
//!
//!   RLX_CUDA_SEGMENTED_CAPTURE=1 RLX_CUDA_SEGMENTED_CAPTURE_ENGAGE=1 \
//!   RLX_CUDA_EXEC_MODE=graph RLX_CUDA_CAPTURE_DEBUG=1 \
//!     cargo test -p rlx-runtime --features cuda \
//!       --test cuda_segmented_capture_elementwise -- --nocapture
//!
//! Off the flags it is a plain eager correctness test; no-ops without CUDA.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const F: DType = DType::F32;

fn target() -> Device {
    match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    }
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 0.4 - 0.2
        })
        .collect()
}

/// cuBLAS-free hybrid: `layers` elementwise blocks `h = h + SiLU(h) * scale`
/// (unary + binary NVRTC kernels only) split by host `Op::Sort` steps.
fn elementwise_graph(m: usize, d: usize, layers: usize, host_every: usize) -> Graph {
    let mut g = Graph::new("elementwise_hybrid");
    let s = Shape::new(&[m, d], F);
    let x = g.input("x", s.clone());
    let scale = g.input("scale", s.clone());
    let mut h = x;
    for l in 0..layers {
        let a = g.activation(Activation::Silu, h, s.clone());
        let sa = g.binary(BinaryOp::Mul, a, scale, s.clone());
        h = g.binary(BinaryOp::Add, h, sa, s.clone());
        if (l + 1) % host_every == 0 && l + 1 < layers {
            h = g.sort(h, 1, false, s.clone());
        }
    }
    g.set_outputs(vec![h]);
    g
}

#[test]
fn cuda_elementwise_segmented_capture_matches_cpu() {
    let dev = target();
    if !is_available(dev) {
        eprintln!("skip cuda_segmented_capture_elementwise ({dev:?} unavailable)");
        return;
    }
    let (m, d, layers, host_every) = (32, 128, 9, 3);
    let x = seeded(m * d, 11);
    let scale = seeded(m * d, 22);
    let feed: [(&str, &[f32]); 2] = [("x", &x), ("scale", &scale)];

    let cpu = Session::new(Device::Cpu)
        .compile(elementwise_graph(m, d, layers, host_every))
        .run(&feed)
        .remove(0);

    // warm-up → capture → replay (under the engage flag); all eager otherwise.
    let mut cuda = Session::new(dev).compile(elementwise_graph(m, d, layers, host_every));
    let o1 = cuda.run(&feed).remove(0);
    let o2 = cuda.run(&feed).remove(0);
    let o3 = cuda.run(&feed).remove(0);

    assert_eq!(o1.len(), m * d);
    let maxd = |o: &[f32]| {
        o.iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let (d1, d2, d3) = (maxd(&o1), maxd(&o2), maxd(&o3));
    eprintln!("[elem] max|CUDA-CPU| warmup={d1:.3e} capture={d2:.3e} replay={d3:.3e}");
    assert!(
        d1 < 1e-4 && d2 < 1e-4 && d3 < 1e-4,
        "elementwise diverged from CPU"
    );
    // Foundation invariant: warm-up == capture == replay, bit-exact.
    assert_eq!(o1, o2, "warm-up != capture (segmented determinism)");
    assert_eq!(o2, o3, "capture != replay (segmented determinism)");
}
