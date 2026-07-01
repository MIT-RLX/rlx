//! Packed GGUF `Op::DequantMatMul` prefill parity (m > 1) — wgpu vs CPU.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run_case(
    scheme: QuantScheme,
    ggml: rlx_gguf::GgmlType,
    m: usize,
    k: usize,
    n: usize,
) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }

    let w_row: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.011).sin() * 0.5)
        .collect();
    let packed = rlx_gguf::quantize(&w_row, ggml).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let mut g = Graph::new("wgpu_gguf_dq_prefill");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.set_param_typed("w", &packed, DType::U8);
        c.run(&[("x", x.as_slice())]).remove(0)
    };

    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    assert_eq!(gpu.len(), m * n);
    assert_eq!(cpu.len(), m * n);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("wguf_dequant_matmul {scheme:?} m={m} k={k} n={n}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_gguf_dequant_matmul_prefill_matches_cpu() {
    let cases: &[(QuantScheme, rlx_gguf::GgmlType, usize, usize, usize, f32)] = &[
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            4,
            256,
            8,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            256,
            640,
            1e-2,
        ),
        (
            QuantScheme::GgufQ4K,
            rlx_gguf::GgmlType::Q4K,
            21,
            640,
            2048,
            1e-2,
        ),
        (
            QuantScheme::GgufQ6K,
            rlx_gguf::GgmlType::Q6K,
            21,
            2048,
            640,
            1e-2,
        ),
        // Gemma 3 270M unsloth GGUF: q/k Q5_0, v Q8_0 (hidden=640).
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            1024,
            1e-2,
        ),
        (
            QuantScheme::GgufQ5_0,
            rlx_gguf::GgmlType::Q5_0,
            21,
            640,
            256,
            1e-2,
        ),
        (
            QuantScheme::GgufQ8_0,
            rlx_gguf::GgmlType::Q8_0,
            21,
            640,
            256,
            1e-2,
        ),
    ];
    for (scheme, ggml, m, k, n, tol) in cases {
        let Some(max_abs) = run_case(*scheme, *ggml, *m, *k, *n) else {
            return;
        };
        assert!(
            max_abs <= *tol,
            "wgpu prefill {scheme:?} m={m} max_abs {max_abs} > {tol}"
        );
    }
}
