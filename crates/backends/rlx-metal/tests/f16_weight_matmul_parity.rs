// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Root-cause probe for the qwen35 vision F16-weight regression.
//!
//! `A_f32 @ W_f16` (the `b_f16` sgemm path) MUST equal `A_f32 @ W_f32` when `W`
//! holds the SAME f16-representable values — the F16 kernel just promotes W→f32
//! and accumulates in f32, so only accumulation-order f32 rounding (~1e-5 rel)
//! may differ. A large divergence means the f16-weight GEMM is buggy (the vision
//! tower's blue→"green" corruption), not precision. Shapes are the actual ViT
//! matmuls (m=1024 → `sgemm_wide8x64_f16w`; the merger m=256).

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn matmul_out(
    m: usize,
    k: usize,
    n: usize,
    wdt: DType,
    device: Device,
    x: &[f32],
    w: &[f32],
) -> Vec<f32> {
    let mut g = Graph::new("f16w_probe");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wp = g.param("w", Shape::new(&[k, n], wdt));
    let y = g.matmul(xi, wp, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param("w", w); // f32 → node dtype (F16 converts on upload)
    c.run(&[("x", x)]).remove(0)
}

// One #[test] so Metal MPS calls stay serial (see sibling parity tests).
#[test]
fn f16_weight_matmul_matches_f32_weight() {
    // qkv, ffn_down, merger — the ViT weight matmuls.
    let mut failures: Vec<String> = Vec::new();
    for (m, k, n) in [
        (1024usize, 1024usize, 3072usize),
        (1024, 4096, 1024),
        (256, 4096, 2560),
    ] {
        // x is the F32 activation (unchanged between runs). w holds
        // f16-representable values so f32-weight == f16-weight-promoted exactly.
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.0013).sin() * 0.5)
            .collect();
        let w: Vec<f32> = (0..k * n)
            .map(|i| f16r(((i as f32) * 0.0007).cos() * 0.08))
            .collect();

        for device in [Device::Cpu, Device::Metal] {
            if device == Device::Metal && !rlx_runtime::is_available(Device::Metal) {
                eprintln!("skip Metal (unavailable)");
                continue;
            }
            let f32w = matmul_out(m, k, n, DType::F32, device, &x, &w);
            let f16w = matmul_out(m, k, n, DType::F16, device, &x, &w);
            let maxval = f32w
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            let maxerr = f32w
                .iter()
                .zip(&f16w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rel = maxerr / maxval;
            let verdict = if rel <= 2e-3 { "OK" } else { "DIVERGES ✗" };
            eprintln!(
                "[f16w] {device:?} {m}x{k}x{n}: maxerr={maxerr:.5} maxval={maxval:.4} rel={rel:.3e} {verdict}"
            );
            if rel > 2e-3 {
                failures.push(format!("{device:?} {m}x{k}x{n} rel={rel:.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "f16-weight matmul diverges from f32: {failures:?}"
    );
}
