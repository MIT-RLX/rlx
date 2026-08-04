// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precision sweep for `Op::SynthMatMul`'s **codebook** stored in low-precision
//! minifloats — fp8 / fp6 / fp4 and the custom `fNeXmY` family — via
//! `synth_matmul_qcodebook` (ScaledQuantize → ScaledDequantize → SynthMatMul, all
//! Metal-native). A `macro_rules!` sweep generates one measured row per format so
//! the variations line up side by side: bit width, realized codebook footprint,
//! accuracy (cosine vs the f32 codebook), and on-device forward time.
//!
//! The point it makes honestly: the codebook is tiny (a few KB, L1-resident), so
//! the format is a **footprint + accuracy** knob, NOT a speed knob — every format
//! runs at ~the same forward time as the f32 codebook. The weight-bandwidth win of
//! SynthMatMul comes from the u8 *indices* (16× fewer bytes than a dense weight),
//! not from the codebook's dtype.
//!
//! Run: cargo run --release --example synth_precision_bench -p rlx-runtime --features metal

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use rlx_ir::{DType, Graph, ScaleLayout, ScaledFormat, Shape, SynthKind};
    use rlx_runtime::{Device, Session, is_available};
    use std::time::Instant;

    // Prefill-ish shape: the matmul dominates, the codebook stays tiny.
    const M: usize = 256; // rows (batch·seq)
    const K: usize = 2048; // contraction
    const N: usize = 2048; // outputs
    const D: usize = 4; // entry_dim (weights per code)
    const NE: usize = 256; // codebook entries (full u8 palette)
    const WARMUP: usize = 8;
    const ITERS: usize = 40;

    let kb = K / D;
    let dev = if is_available(Device::Metal) {
        Device::Metal
    } else {
        eprintln!("Metal unavailable — falling back to CPU");
        Device::Cpu
    };

    // Deterministic inputs. The codebook is smooth so fp-quant tracks it well.
    let x: Vec<f32> = (0..M * K).map(|i| (i as f32 * 0.011).sin() * 1.1).collect();
    let indices: Vec<u8> = (0..N * kb).map(|i| (i % NE) as u8).collect();
    let codebook: Vec<f32> = (0..NE * D)
        .map(|i| (i as f32 * 0.017).cos() * 0.9)
        .collect();

    let kind = SynthKind::Codebook {
        entry_dim: D as u32,
        num_entries: NE as u32,
    };
    let out_shape = Shape::new(&[M, N], DType::F32);

    // Build a compiled graph for one codebook format (None = f32 baseline).
    let compile = |fmt: Option<ScaledFormat>| {
        let mut g = Graph::new("synth_prec");
        let xn = g.input("x", Shape::new(&[M, K], DType::F32));
        let cbn = g.input("cb", Shape::new(&[NE, D], DType::F32));
        let idx = g.param("idx", Shape::new(&[N, kb], DType::U8));
        let y = match fmt {
            None => g.synth_matmul(xn, idx, cbn, kind, out_shape.clone()),
            Some(f) => {
                let (codes, scale) = g.scaled_quantize(cbn, f, ScaleLayout::PerTensor);
                g.synth_matmul_qcodebook(
                    xn,
                    idx,
                    codes,
                    scale,
                    kind,
                    f,
                    ScaleLayout::PerTensor,
                    out_shape.clone(),
                )
            }
        };
        g.set_outputs(vec![y]);
        let mut c = Session::new(dev).compile(g);
        c.set_param_typed("idx", &indices, DType::U8);
        c
    };

    // Warm, then take the MIN over 3 timed blocks of ITERS passes each — min is the
    // least-contended sample, closest to uncontended peak (robust to machine load).
    let bench = |fmt: Option<ScaledFormat>| -> (f64, Vec<f32>) {
        let mut c = compile(fmt);
        let mut out = vec![];
        for _ in 0..WARMUP {
            out = c.run(&[("x", &x), ("cb", &codebook)]).pop().unwrap();
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                out = c.run(&[("x", &x), ("cb", &codebook)]).pop().unwrap();
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        (best, out)
    };

    // f64 accumulation so the score never drifts past 1.0 on 500k-element sums.
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        d / (na * nb + 1e-12)
    }

    // Codebook reconstruction fidelity: dequant(quant(cb)) vs cb, relative L2 error.
    // This is the *discriminating* precision metric — the output cosine washes the
    // f8/f4 gap out because the K-long matmul averages the per-code error away.
    let cb_recon_relerr = |fmt: ScaledFormat| -> f64 {
        let mut g = Graph::new("cb_roundtrip");
        let cbn = g.input("cb", Shape::new(&[NE, D], DType::F32));
        let (codes, scale) = g.scaled_quantize(cbn, fmt, ScaleLayout::PerTensor);
        let deq = g.scaled_dequantize(codes, scale, fmt, ScaleLayout::PerTensor);
        g.set_outputs(vec![deq]);
        let mut c = Session::new(Device::Cpu).compile(g);
        let recon = c.run(&[("cb", &codebook)]).pop().unwrap();
        let num: f64 = codebook
            .iter()
            .zip(&recon)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum();
        let den: f64 = codebook.iter().map(|a| (*a as f64).powi(2)).sum();
        (num / den).sqrt()
    };

    // f32-codebook baseline (the reference the low-precision rows are scored against).
    let (base_ms, reference) = bench(None);
    let cb_bytes_f32 = NE * D * 4;

    println!("device: {dev:?}   shape {M}×{K}×{N}  (entry_dim={D}, entries={NE})\n");
    println!(
        "  dense f32 weight would be {} — SynthMatMul stores {} indices + a {}-entry codebook\n",
        fmt_bytes(K * N * 4),
        fmt_bytes(N * kb),
        NE
    );
    println!(
        "{:>10} {:>4} {:>11} {:>11} {:>10} {:>9} {:>8}",
        "codebook", "bits", "cb_bytes", "cb_recon_e", "cos_vs_f32", "fwd_ms", "vs_f32"
    );
    println!("{}", "-".repeat(70));
    println!(
        "{:>10} {:>4} {:>11} {:>11} {:>10.5} {:>9.3} {:>7.2}x",
        "f32",
        32,
        fmt_bytes(cb_bytes_f32),
        "0",
        1.0,
        base_ms,
        1.0
    );

    // The macro: one measured, printed row per low-precision format. `bits` is the
    // minifloat width (1+exp+mant); the realized code footprint is 1 byte/code in
    // rlx's software-decode path (+ a per-tensor scale), so all rows are 4× smaller
    // than f32 regardless of `bits` — `bits` is the *accuracy grid*, not storage.
    macro_rules! sweep {
        ($($label:literal => $fmt:expr, $bits:expr);+ $(;)?) => {{
            $(
                let (ms, out) = bench(Some($fmt));
                let cos = cosine(&reference, &out);
                let recon_e = cb_recon_relerr($fmt);
                let cb_bytes = NE * D + 4; // 1 byte/code + PerTensor scale
                println!(
                    "{:>10} {:>4} {:>11} {:>11.5} {:>10.5} {:>9.3} {:>7.2}x",
                    $label, $bits, fmt_bytes(cb_bytes), recon_e, cos, ms, base_ms / ms
                );
            )+
        }};
    }

    sweep! {
        "f8 e4m3"  => ScaledFormat::F8E4M3,        8;  // OCP fp8, wide mantissa
        "f8 e5m2"  => ScaledFormat::F8E5M2,        8;  // OCP fp8, wide range
        "f6 e2m3"  => ScaledFormat::F6E2M3,        6;  // MX fp6
        "f6 e3m2"  => ScaledFormat::F6E3M2,        6;  // MX fp6, wider range
        "f4 e2m1"  => ScaledFormat::F4E2M1,        4;  // fp4 (NVFP4/MXFP4 grid)
        "f5 e3m1"  => ScaledFormat::custom(3, 1),  5;  // custom fNeXmY
        "f4 e3m0"  => ScaledFormat::custom(3, 0),  4;  // signed power-of-two (log)
        "f3 e2m0"  => ScaledFormat::custom(2, 0),  3;  // 3-bit minifloat
    }

    println!(
        "\nnote: precision is a FOOTPRINT + ACCURACY knob, not a speed knob. cb_recon_e (the\n\
         codebook round-trip error) is the real precision ladder — driven by mantissa bits:\n\
         *m3 ≈ 0.022, *m2 ≈ 0.044, *m1 ≈ 0.09, *m0 ≈ 0.15 — while cos_vs_f32 stays ≥ 0.9999\n\
         because the K-long matmul averages per-code error away. fwd_ms carries per-call\n\
         Session readback overhead (2 MB D2H each run), so treat it as format-INDEPENDENT at\n\
         the kernel level (the matmul is identical f32 after a negligible {}-byte codebook\n\
         dequant); for clean kernel timing see `synth_roofline`. The bandwidth win is the u8\n\
         indices ({} vs a {} dense weight), not the codebook dtype.",
        fmt_bytes(cb_bytes_f32),
        fmt_bytes(N * kb),
        fmt_bytes(K * N * 4)
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    eprintln!(
        "build with: cargo run --release --example synth_precision_bench -p rlx-runtime --features metal"
    );
}
