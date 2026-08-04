// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! M-sweep for `Op::SynthMatMul` on Metal — locate the crossover between the
//! fused decode kernel (reconstruct-in-loop) and reconstruct→MPS, and measure how
//! far each sits from the dense-f32-MPS ceiling (the speed you'd get if you gave up
//! compression and stored the full weight). The gap between the best synth path and
//! the dense ceiling at each M is the "cost of compression" a better-tiled fused
//! kernel would try to close.
//!
//! Run: cargo run --release --example synth_m_sweep_bench -p rlx-runtime --features metal

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use rlx_ir::{DType, Graph, Shape, SynthKind};
    use rlx_runtime::{Device, Session, is_available};
    use std::time::Instant;

    const K: usize = 2048;
    const N: usize = 2048;
    const D: usize = 4;
    const NE: usize = 256;
    const WARMUP: usize = 6;
    const ITERS: usize = 30;
    let ms_list = [1usize, 4, 8, 16, 32, 64, 128, 256];

    if !is_available(Device::Metal) {
        eprintln!("Metal unavailable");
        return;
    }
    let kb = K / D;
    let kind = SynthKind::Codebook {
        entry_dim: D as u32,
        num_entries: NE as u32,
    };
    let indices: Vec<u8> = (0..N * kb).map(|i| (i % NE) as u8).collect();
    let codebook: Vec<f32> = (0..NE * D)
        .map(|i| (i as f32 * 0.017).cos() * 0.9)
        .collect();

    // Dense weight the synth op *avoids* storing: W[k, j] = codebook[indices[j,b], t]
    // with k = b*D+t. Laid out [K, N] so a plain matmul x[M,K] @ W[K,N] = [M,N].
    let mut wdense = vec![0f32; K * N];
    for j in 0..N {
        for b in 0..kb {
            let code = indices[j * kb + b] as usize;
            for t in 0..D {
                wdense[(b * D + t) * N + j] = codebook[code * D + t];
            }
        }
    }

    // Time `iters` runs, min over 3 blocks (robust to load).
    fn timed(
        c: &mut rlx_runtime::CompiledGraph,
        inputs: &[(&str, &[f32])],
        warm: usize,
        iters: usize,
    ) -> f64 {
        for _ in 0..warm {
            let _ = c.run(inputs);
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = c.run(inputs);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / iters as f64);
        }
        best
    }

    println!("SynthMatMul M-sweep on Metal  (K={K}, N={N}, d={D}, entries={NE})\n");
    println!(
        "{:>4} {:>9} {:>11} {:>9} {:>9} {:>9} {:>11} {:>12}",
        "M", "fused", "recon→MPS", "TILED", "TILED_f16", "denseMPS", "best", "best/dense"
    );
    println!("{}", "-".repeat(86));

    for &m in &ms_list {
        let x: Vec<f32> = (0..m * K).map(|i| (i as f32 * 0.011).sin() * 1.1).collect();
        let out_shape = Shape::new(&[m, N], DType::F32);

        // synth graph (indices param set once).
        let mut g = Graph::new("synth");
        let xn = g.input("x", Shape::new(&[m, K], DType::F32));
        let cbn = g.input("cb", Shape::new(&[NE, D], DType::F32));
        let idx = g.param("idx", Shape::new(&[N, kb], DType::U8));
        let y = g.synth_matmul(xn, idx, cbn, kind, out_shape.clone());
        g.set_outputs(vec![y]);
        let mut cs = Session::new(Device::Metal).compile(g);
        cs.set_param_typed("idx", &indices, DType::U8);
        let syn_inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("cb", &codebook)];

        // recon→MPS (default for M>8; split-K for M≤8).
        unsafe { std::env::remove_var("RLX_METAL_SYNTH_MPS_DISABLE") };
        unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED") };
        let recon_ms = timed(&mut cs, &syn_inputs, WARMUP, ITERS);
        // fused-forced (reconstruct-in-loop kernel for all M).
        unsafe { std::env::set_var("RLX_METAL_SYNTH_MPS_DISABLE", "1") };
        let fused_ms = timed(&mut cs, &syn_inputs, WARMUP, ITERS);
        unsafe { std::env::remove_var("RLX_METAL_SYNTH_MPS_DISABLE") };
        // NEW: threadgroup-tiled fused kernel (takes priority when set).
        unsafe { std::env::set_var("RLX_METAL_SYNTH_TILED", "1") };
        let tiled_ms = timed(&mut cs, &syn_inputs, WARMUP, ITERS);
        // f16 (relaxed-precision) tiled variant: simdgroup_half8x8 MMAs.
        unsafe { std::env::set_var("RLX_METAL_SYNTH_TILED_F16", "1") };
        let tiled_h_ms = timed(&mut cs, &syn_inputs, WARMUP, ITERS);
        unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED_F16") };
        unsafe { std::env::remove_var("RLX_METAL_SYNTH_TILED") };

        // dense f32 matmul ceiling: x[M,K] @ Wdense[K,N]. The weight is a PARAM set
        // once (resident) — NOT fed per run — so we don't charge a 16 MB H2D upload
        // to every timed iteration (that would inflate the ceiling; a real inference
        // keeps the weight resident).
        let mut gd = Graph::new("dense");
        let xnd = gd.input("x", Shape::new(&[m, K], DType::F32));
        let wnd = gd.param("w", Shape::new(&[K, N], DType::F32));
        let yd = gd.matmul(xnd, wnd, out_shape.clone());
        gd.set_outputs(vec![yd]);
        let mut cd = Session::new(Device::Metal).compile(gd);
        cd.set_param("w", &wdense);
        let dense_ms = timed(&mut cd, &[("x", &x)], WARMUP, ITERS);

        let best_syn = fused_ms.min(recon_ms).min(tiled_ms).min(tiled_h_ms);
        let best_label = if best_syn == tiled_h_ms {
            "TILED_f16"
        } else if best_syn == tiled_ms {
            "TILED"
        } else if best_syn == recon_ms {
            "recon→MPS"
        } else {
            "fused"
        };
        println!(
            "{:>4} {:>9.3} {:>11.3} {:>9.3} {:>9.3} {:>9.3} {:>11} {:>10.2}x",
            m,
            fused_ms,
            recon_ms,
            tiled_ms,
            tiled_h_ms,
            dense_ms,
            best_label,
            best_syn / dense_ms
        );
    }

    println!(
        "\ncomp_cost_vs_dense = best synth path / dense-f32-MPS. >1 means compression costs\n\
         speed at that M; the gap is what a threadgroup-tiled fused kernel would target.\n\
         (dense-MPS is the no-compression ceiling — it stores the full {} weight; synth\n\
         stores {} of indices + a {}-entry codebook.)",
        {
            let b = K * N * 4;
            if b >= 1 << 20 {
                format!("{:.1} MB", b as f64 / (1 << 20) as f64)
            } else {
                format!("{b} B")
            }
        },
        {
            let b = N * kb;
            if b >= 1 << 20 {
                format!("{:.1} MB", b as f64 / (1 << 20) as f64)
            } else {
                format!("{:.1} KB", b as f64 / (1 << 10) as f64)
            }
        },
        NE
    );
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    eprintln!("build with: --features metal on macOS");
}
