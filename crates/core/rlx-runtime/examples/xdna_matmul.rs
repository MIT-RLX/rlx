// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Verify `Device::Xdna` runs a real (quantized) rlx matmul GRAPH on the NPU.
//
// Uses fractional f32 data, so the NPU path exercises real per-tensor symmetric
// INT8 quantization. Checks two things:
//   1. NPU output == a CPU quant-reference (same i8 math) — validates the NPU +
//      quant plumbing (should match to f32 dequant rounding).
//   2. quant error of that reference vs the true f32 matmul — quantization quality.
//
//   RLX_XDNA_SHIM=… RLX_XDNA_XCLBIN=… RLX_XDNA_INSTS=… RLX_XDNA_GEMM=512,512,512 \
//   LD_LIBRARY_PATH=<xrt lib> M=512 K=512 N=512 \
//   cargo run -p rlx-runtime --features xdna --example xdna_matmul

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn dim(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn mkn() -> (usize, usize, usize) {
    (dim("M", 512), dim("K", 512), dim("N", 512))
}

fn build() -> Graph {
    let (m, k, n) = mkn();
    let mut g = Graph::new("xdna_mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Per-row symmetric int8 (matches the backend's per-token activation quant).
fn q_per_row(v: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![1.0f32; rows];
    for r in 0..rows {
        let amax = v[r * cols..r * cols + cols].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if amax == 0.0 {
            continue;
        }
        s[r] = amax / 127.0;
        for c in 0..cols {
            q[r * cols + c] = (v[r * cols + c] / s[r]).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

/// Per-col symmetric int8 (matches the backend's per-channel weight quant).
fn q_per_col(v: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut s = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            s[c] = s[c].max(v[r * cols + c].abs());
        }
    }
    for sc in s.iter_mut() {
        *sc = if *sc == 0.0 { 1.0 } else { *sc / 127.0 };
    }
    let mut q = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            q[r * cols + c] = (v[r * cols + c] / s[c]).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

fn main() {
    let (m, k, n) = mkn();
    // Fractional data → genuinely exercises quantization (not integer-exact).
    // W has OUTLIER output channels (every 8th column ~8× larger) — the case
    // per-channel quant handles well and per-tensor quant does not.
    let xv: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin()).collect();
    let wv: Vec<f32> = (0..k * n)
        .map(|i| (i as f32 * 0.011).cos() * 0.5 * if (i % n) % 8 == 0 { 8.0 } else { 1.0 })
        .collect();

    if !is_available(Device::Xdna) {
        eprintln!("Device::Xdna not available — set RLX_XDNA_SHIM/XCLBIN/INSTS/GEMM (+ LD_LIBRARY_PATH).");
        std::process::exit(2);
    }

    // NPU — the quantized matmul graph.
    let mut npu = Session::new(Device::Xdna).compile(build());
    npu.set_param("W", &wv);
    let y_npu = npu.run(&[("x", xv.as_slice())]).into_iter().next().unwrap();

    // CPU quant-reference: per-token/per-channel int8 + i32 matmul + dequant.
    let (xq, sx) = q_per_row(&xv, m, k);
    let (wq, sw) = q_per_col(&wv, k, n);
    let mut acc = vec![0i64; m * n];
    let mut y_true = vec![0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let (aq, af) = (xq[i * k + kk] as i64, xv[i * k + kk]);
            for j in 0..n {
                acc[i * n + j] += aq * wq[kk * n + j] as i64;
                y_true[i * n + j] += af * wv[kk * n + j];
            }
        }
    }
    let y_ref: Vec<f32> = (0..m * n)
        .map(|idx| acc[idx] as f32 * sx[idx / n] * sw[idx % n])
        .collect();

    // 1) NPU vs quant-ref (should match — same integer math + dequant).
    let max_abs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let npu_err = max_abs(&y_npu, &y_ref);
    let scale = y_ref.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
    let npu_ok = npu_err < 1e-3 * scale;

    // 2) quant quality: relative L2 of the quant-ref vs the true f32 matmul.
    let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rel_of = |y: &[f32]| {
        let d: Vec<f32> = y.iter().zip(&y_true).map(|(a, b)| a - b).collect();
        l2(&d) / l2(&y_true).max(1e-9)
    };
    let rel = rel_of(&y_ref); // per-channel/per-token (what the NPU does)

    // Contrast: a whole-tensor (per-tensor) quantization of the same data.
    let amax = |v: &[f32]| v.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let (sxt, swt) = (amax(&xv) / 127.0, amax(&wv) / 127.0);
    let qt = |v: &[f32], s: f32| -> Vec<i8> {
        v.iter().map(|&x| (x / s).round().clamp(-127., 127.) as i8).collect()
    };
    let (xt, wt) = (qt(&xv, sxt), qt(&wv, swt));
    let mut y_pt = vec![0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a = xt[i * k + kk] as i64;
            for j in 0..n {
                y_pt[i * n + j] += (a * wt[kk * n + j] as i64) as f32;
            }
        }
    }
    for y in &mut y_pt {
        *y *= sxt * swt;
    }
    let rel_pt = rel_of(&y_pt);

    // Warm per-call latency (exercises the resident-weight fast path when the
    // matmul fits one tile; set RLX_XDNA_NO_RESIDENT=1 to compare the re-upload).
    let iters = dim("ITERS", 50);
    let t = std::time::Instant::now();
    for _ in 0..iters {
        let _ = npu.run(&[("x", xv.as_slice())]);
    }
    let per_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let resident = std::env::var("RLX_XDNA_NO_RESIDENT").is_err();

    if npu_ok {
        println!(
            "rlx Device::Xdna quantized matmul {m}x{k}x{n}: PASS ✓  \
             NPU==quant-ref (max_err {npu_err:.2e}); int8 err vs f32: per-channel {:.2}% \
             (vs per-tensor {:.2}%); warm {per_us:.0} us/call ({})",
            rel * 100.0,
            rel_pt * 100.0,
            if resident { "resident weight" } else { "re-upload W" }
        );
    } else {
        println!(
            "rlx Device::Xdna quantized matmul {m}x{k}x{n}: FAIL ✗ — NPU vs quant-ref \
             max_err {npu_err:.2e} (scale {scale:.1e})"
        );
        std::process::exit(1);
    }
}
