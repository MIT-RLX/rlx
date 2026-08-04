// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end prefill timing for `Op::SynthMatMul` on Metal: the reconstruct→MPS
//! path (default, m>8) vs the fused kernel (`RLX_METAL_SYNTH_MPS_DISABLE=1`).
//! Run: cargo run --release --example synth_prefill_bench -p rlx-runtime --features metal

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use rlx_ir::{DType, Graph, Shape, SynthKind};
    use rlx_runtime::{Device, Session};
    use std::time::Instant;

    if !rlx_runtime::is_available(Device::Metal) {
        println!("no Metal device");
        return;
    }
    let (m, k, n, d, ne) = (256usize, 2048usize, 2048usize, 4usize, 256usize);
    let kb = k / d;
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let cb: Vec<f32> = (0..ne * d).map(|i| (i as f32 * 0.01).cos()).collect();
    let idx: Vec<u8> = (0..n * kb).map(|i| (i % ne) as u8).collect();

    let build = || {
        let mut g = Graph::new("synth_prefill");
        let xn = g.input("x", Shape::new(&[m, k], DType::F32));
        let cn = g.input("cb", Shape::new(&[ne, d], DType::F32));
        let idn = g.param("idx", Shape::new(&[n, kb], DType::U8));
        let y = g.synth_matmul(
            xn,
            idn,
            cn,
            SynthKind::Codebook {
                entry_dim: d as u32,
                num_entries: ne as u32,
            },
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);
        g
    };

    let time = |label: &str| -> f64 {
        let mut c = Session::new(Device::Metal).compile(build());
        c.set_param_typed("idx", &idx, DType::U8);
        for _ in 0..5 {
            let _ = c.run(&[("x", &x), ("cb", &cb)]);
        }
        let iters = 30;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = c.run(&[("x", &x), ("cb", &cb)]);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("  {label:<24} {ms:7.3} ms");
        ms
    };

    println!("SynthMatMul prefill {m}×{k}×{n} (d={d}) on Metal:");
    // Default: recon→MPS.
    unsafe {
        std::env::remove_var("RLX_METAL_SYNTH_MPS_DISABLE");
    }
    let mps = time("recon→MPS (default)");
    // Fused kernel.
    unsafe {
        std::env::set_var("RLX_METAL_SYNTH_MPS_DISABLE", "1");
    }
    let fused = time("fused (MPS disabled)");
    unsafe {
        std::env::remove_var("RLX_METAL_SYNTH_MPS_DISABLE");
    }
    println!("  → recon→MPS is {:.2}× faster than fused", fused / mps);
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    println!("Metal only (macOS + --features metal)");
}
