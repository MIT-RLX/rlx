// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — BERT-QKV baseline: CPU vs CoopF16Vk vs wide F32 (cosine + column errors).

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use std::sync::Mutex;

static COOP_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn require_vk() -> bool {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => return false,
    };
    rlx_wgpu::device::coop_discrete_backend()
        && !rlx_ir::env::flag("RLX_WGPU_NO_COOP_F16_VK")
        && rlx_wgpu::device::coop_f16_16x16_supported()
        && rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&dev.device).is_some()
}

fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn matmul_f16_round(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                let av = half::f16::from_f32(a[i * k + kk]).to_f32();
                let bv = half::f16::from_f32(b[kk * n + j]).to_f32();
                s += av * bv;
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn run_matmul(n: usize, use_coop: bool, a_data: &[f32], b_data: &[f32]) -> Vec<f32> {
    const M: usize = 96;
    const K: usize = 384;
    let mut g = Graph::new(if use_coop {
        "bert_baseline_coop"
    } else {
        "bert_baseline_wide"
    });
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, n], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, n], DType::F32));
    g.set_outputs(vec![c]);
    if use_coop {
        unsafe {
            std::env::remove_var("RLX_WGPU_NO_COOP_F16_VK");
        }
    } else {
        unsafe {
            std::env::set_var("RLX_WGPU_NO_COOP_F16_VK", "1");
        }
    }
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", b_data);
    let out = exe.run(&[("a", a_data)])[0].clone();
    if !use_coop {
        unsafe {
            std::env::remove_var("RLX_WGPU_NO_COOP_F16_VK");
        }
    }
    out
}

fn baseline_n(n: usize, gentle: bool) {
    const M: usize = 96;
    const K: usize = 384;
    let a_data: Vec<f32> = if gentle {
        (0..M * K).map(|x| 0.05 * (x as f32 * 0.01).sin()).collect()
    } else {
        (0..M * K).map(|x| 0.1 * (x as f32).sin()).collect()
    };
    let b_data: Vec<f32> = if gentle {
        (0..K * n).map(|x| 0.05 * (x as f32 * 0.02).cos()).collect()
    } else {
        (0..K * n).map(|x| 0.1 * (x as f32).cos()).collect()
    };
    let ref_f32 = matmul_f32(&a_data, &b_data, M, K, n);
    let ref_f16 = matmul_f16_round(&a_data, &b_data, M, K, n);
    let coop = run_matmul(n, true, &a_data, &b_data);
    let wide = run_matmul(n, false, &a_data, &b_data);

    let abs_max = ref_f32.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let tag = if gentle { "gentle" } else { "bert" };
    eprintln!("=== BERT baseline ({tag}) M={M} K={K} N={n} ===");
    eprintln!(
        "ref_f32 abs_max={abs_max}  ref_f16 max|Δ|={}",
        max_diff(&ref_f32, &ref_f16)
    );
    eprintln!(
        "coop  max|Δ|={} rel={} cosine={}",
        max_diff(&ref_f32, &coop),
        max_diff(&ref_f32, &coop) / abs_max.max(1e-30),
        cosine(&ref_f32, &coop)
    );
    eprintln!(
        "wide  max|Δ|={} rel={} cosine={}",
        max_diff(&ref_f32, &wide),
        max_diff(&ref_f32, &wide) / abs_max.max(1e-30),
        cosine(&ref_f32, &wide)
    );
    let coop_f16_diff = max_diff(&ref_f16, &coop);
    eprintln!(
        "coop vs f16-ref max|Δ|={} cosine={}",
        coop_f16_diff,
        cosine(&ref_f16, &coop)
    );

    if gentle {
        assert!(
            coop_f16_diff < 5e-4,
            "N={n} ({tag}): coop vs f16-ref max|Δ| = {coop_f16_diff}"
        );
        if n > 768 {
            assert!(
                max_diff(&wide, &coop) < 1e-4,
                "N={n}: large-N coop should match wide f32 (max|Δ|={})",
                max_diff(&wide, &coop)
            );
        }
    } else {
        assert!(
            max_diff(&wide, &coop) < 1e-4,
            "N={n} ({tag}): oscillating B should auto-wide (coop vs wide max|Δ| = {})",
            max_diff(&wide, &coop)
        );
        assert!(
            cosine(&ref_f32, &coop) > 0.9999,
            "N={n} ({tag}): coop auto-wide vs f32-ref cosine = {}",
            cosine(&ref_f32, &coop)
        );
    }
}

#[test]
fn coop_f16_vk_bert_qkv_baseline() {
    let _g = COOP_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    if !require_vk() {
        eprintln!("skip bert baseline (no CoopF16Vk adapter)");
        return;
    }
    baseline_n(1152, true);
    baseline_n(384, false);
    baseline_n(1152, false);
}
