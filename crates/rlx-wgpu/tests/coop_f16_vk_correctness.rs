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

//
// Correctness tests for Vulkan/DX12 cooperative-matrix matmul (16×16 f16).

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;
use std::sync::Mutex;

/// wgpu + env-var probes are process-global; serialize these tests.
static COOP_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct CoopTestGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
impl CoopTestGuard {
    fn new() -> Self {
        Self(COOP_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

fn require_coop_f16_vk_test() -> bool {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return false;
        }
    };
    if !rlx_wgpu::device::coop_discrete_backend() {
        eprintln!(
            "CoopF16Vk requires Vulkan/DX12; skipping on {:?}",
            dev.backend
        );
        return false;
    }
    if rlx_ir::env::flag("RLX_WGPU_NO_COOP_F16_VK") {
        eprintln!("RLX_WGPU_NO_COOP_F16_VK set, skipping");
        return false;
    }
    if !rlx_wgpu::device::coop_f16_16x16_supported() {
        eprintln!("adapter lacks 16×16 f16 cooperative-matrix support, skipping");
        return false;
    }
    if rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&dev.device).is_none() {
        eprintln!("matmul_coop_f16_vulkan kernel unavailable, skipping");
        return false;
    }
    true
}

fn matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

fn max_diff(expected: &[f32], out: &[f32]) -> f32 {
    expected
        .iter()
        .zip(out.iter())
        .map(|(e, o)| (e - o).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn coop_f16_vk_ones_matmul_sum_k() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 256;
    let mut g = Graph::new("coop_f16_vk_ones");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &vec![1.0_f32; K * N]);
    let outs = exe.run(&[("a", vec![1.0_f32; M * K].as_slice())]);
    let v = outs[0][0];
    eprintln!("ones matmul corner = {v}, want {K}");
    assert!((v - K as f32).abs() < 1.0, "expected {K}, got {v}");
}

#[test]
fn coop_f16_vk_correct_256_cube() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 256;

    let mut g = Graph::new("coop_f16_vk_256");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|i| 0.01 * ((i % 17) as f32 - 8.0)).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| 0.02 * ((i % 13) as f32 - 6.0)).collect();

    let expected = matmul_reference(&a_data, &b_data, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let outs = exe.run(&[("a", a_data.as_slice())]);
    let out = &outs[0];

    let diff = max_diff(&expected, out);
    let abs_max = expected.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "CoopF16Vk 256³ max|Δ| = {diff}, rel = {}",
        diff / abs_max.max(1e-30)
    );
    assert!(
        diff < 6e-2,
        "CoopF16Vk matmul diverges: max|Δ|={diff} rel={}",
        diff / abs_max.max(1e-30)
    );
}

#[test]
fn coop_f16_vk_large_k_384() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 96;
    const K: usize = 384;
    const N: usize = 384;

    let mut g = Graph::new("coop_f16_vk_k384");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|x| 0.05 * (x as f32 * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..K * N).map(|x| 0.05 * (x as f32 * 0.02).cos()).collect();
    let expected = matmul_reference(&a_data, &b_data, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let out = &exe.run(&[("a", a_data.as_slice())])[0];
    let diff = max_diff(&expected, out);
    let abs_max = expected.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "CoopF16Vk K=384 max|Δ| = {diff}, rel = {}",
        diff / abs_max.max(1e-30)
    );
    assert!(diff < 8e-2, "CoopF16Vk K=384 diverges: max|Δ|={diff}");
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

#[test]
fn coop_f16_vk_correct_at_minilm_qkv() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 96;
    const K: usize = 384;
    const N: usize = 1152;

    let mut g = Graph::new("coop_f16_vk_bertk");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|x| 0.05 * (x as f32 * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..K * N).map(|x| 0.05 * (x as f32 * 0.02).cos()).collect();
    let expected = matmul_f16_round(&a_data, &b_data, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let out = &exe.run(&[("a", a_data.as_slice())])[0];
    let diff = max_diff(&expected, out);
    let abs_max = expected.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "CoopF16Vk QKV N=1152 (gentle) max|Δ| vs f16-ref = {diff}, rel = {}",
        diff / abs_max.max(1e-30)
    );
    assert!(
        diff < 5e-4,
        "CoopF16Vk K=384 N=1152 diverges from f16-ref: max|Δ|={diff} rel={}",
        diff / abs_max.max(1e-30)
    );
}

#[test]
fn coop_f16_vk_bert_sin_qkv() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 96;
    const K: usize = 384;
    const N: usize = 1152;

    let mut g = Graph::new("coop_f16_vk_bert_sin");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|x| 0.1 * (x as f32).sin()).collect();
    let b_data: Vec<f32> = (0..K * N).map(|x| 0.1 * (x as f32).cos()).collect();
    let expected = matmul_reference(&a_data, &b_data, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let out = &exe.run(&[("a", a_data.as_slice())])[0];
    let diff = max_diff(&expected, out);
    let abs_max = expected.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "CoopF16Vk BERT sin/cos QKV (auto wide) max|Δ| vs f32-ref = {diff}, rel = {}",
        diff / abs_max.max(1e-30)
    );
    assert!(
        diff < 1e-4,
        "CoopF16Vk BERT sin QKV should auto-wide to f32: max|Δ|={diff} rel={}",
        diff / abs_max.max(1e-30)
    );
}

#[test]
fn coop_f16_vk_computed_b_matmul() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;

    let mut g = Graph::new("coop_f16_vk_comp_b");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b_in = g.input("b_in", Shape::new(&[K, N], DType::F32));
    let b = g.activation(Activation::Relu, b_in, Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|i| (i as f32 * 0.003).sin()).collect();
    let b_in_data: Vec<f32> = (0..K * N).map(|i| i as f32 * 0.004 - 0.2).collect();
    let b_ref: Vec<f32> = b_in_data.iter().map(|&v| v.max(0.0)).collect();
    let expected = matmul_reference(&a_data, &b_ref, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    let out = &exe.run(&[("a", a_data.as_slice()), ("b_in", b_in_data.as_slice())])[0];
    let diff = max_diff(&expected, out);
    assert!(
        diff < 8e-2,
        "CoopF16Vk computed-B matmul diverges: max|Δ|={diff}"
    );
}

#[test]
fn coop_f16_vk_activation_operand_a() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;

    let mut g = Graph::new("coop_f16_vk_act_a");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let w = g.param("w", Shape::new(&[K, N], DType::F32));
    let a = g.activation(Activation::Relu, x, Shape::new(&[M, K], DType::F32));
    let c = g.matmul(a, w, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let x_data: Vec<f32> = (0..M * K).map(|i| i as f32 * 0.01 - 0.5).collect();
    let w_data: Vec<f32> = (0..K * N).map(|i| 0.01 * (i as f32).sin()).collect();
    let a_ref: Vec<f32> = x_data.iter().map(|&v| v.max(0.0)).collect();
    let expected = matmul_reference(&a_ref, &w_data, M, K, N);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &w_data);
    let out = &exe.run(&[("x", x_data.as_slice())])[0];
    let diff = max_diff(&expected, out);
    assert!(
        diff < 8e-2,
        "CoopF16Vk activation-A matmul diverges: max|Δ|={diff}"
    );
}

#[test]
fn coop_f16_vk_input_upload_skip_preserves_result() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;

    let mut g = Graph::new("coop_f16_vk_skip");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data: Vec<f32> = (0..M * K).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i as f32 * 0.002).cos()).collect();

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &b_data);
    let first = exe.run(&[("a", a_data.as_slice())])[0].clone();
    let second = exe.run(&[("a", a_data.as_slice())])[0].clone();
    let diff = max_diff(&first, &second);
    assert!(
        diff == 0.0,
        "identical input re-run diverged after upload skip: max|Δ|={diff}"
    );
}

#[test]
fn coop_f16_vk_bench_vs_wide_dispatch_only() {
    let _guard = CoopTestGuard::new();
    if !require_coop_f16_vk_test() {
        return;
    }
    use rlx_ir::Tick;
    const M: usize = 1024;
    const K: usize = 1024;
    const N: usize = 1024;

    let mut g = Graph::new("coop_f16_vk_bench");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let a_data = vec![0.01_f32; M * K];
    let b_data = vec![0.01_f32; K * N];

    let mut exe_coop = WgpuExecutable::compile(g.clone());
    exe_coop.set_param("b", &b_data);

    unsafe {
        std::env::set_var("RLX_WGPU_NO_COOP_F16_VK", "1");
    }
    let mut exe_wide = WgpuExecutable::compile(g);
    exe_wide.set_param("b", &b_data);
    unsafe {
        std::env::remove_var("RLX_WGPU_NO_COOP_F16_VK");
    }

    unsafe {
        std::env::set_var("RLX_BENCH_DISPATCH_ONLY", "1");
    }
    let _ = exe_coop.run(&[("a", a_data.as_slice())]);
    let t0 = Tick::now();
    for _ in 0..5 {
        let _ = exe_coop.run(&[("a", a_data.as_slice())]);
    }
    let coop_us = Tick::now().elapsed_us(t0) / 5.0;

    let _ = exe_wide.run(&[("a", a_data.as_slice())]);
    let t1 = Tick::now();
    for _ in 0..5 {
        let _ = exe_wide.run(&[("a", a_data.as_slice())]);
    }
    let wide_us = Tick::now().elapsed_us(t1) / 5.0;
    unsafe {
        std::env::remove_var("RLX_BENCH_DISPATCH_ONLY");
    }

    eprintln!("CoopF16Vk 1024³ dispatch-only ≈ {coop_us} µs vs wide_nv ≈ {wide_us} µs");
}
