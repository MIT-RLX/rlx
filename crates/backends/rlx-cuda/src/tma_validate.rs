// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! On-device validation of the Hopper TMA GEMM path. What actually runs
//! depends on the box:
//!   - ANY CUDA device (incl. Ampere): NVRTC really compiles `matmul_tma.cu`
//!     for `compute_90a` (validates all the hand-written inline PTX),
//!     `cuTensorMapEncodeTiled` succeeds against a live driver, and the
//!     cc-probe / `tma_arch` gate behave.
//!   - sm_90 only: the numeric TMA-vs-CPU GEMM parity actually executes.
//! No-ops entirely when CUDA is absent (Mac / CI).

use std::sync::Mutex;

use crate::backend::{CudaExecutable, TMA_BK, TMA_BM, device_cc, tma_arch, tma_encode_tiled_2d};
use crate::config::{CudaRuntimeConfig, install_runtime_config, reload_runtime_config};
use crate::device::cuda_context;
use cudarc::driver::DevicePtr;

// Serializes the tests that mutate the process-wide runtime config so parallel
// execution can't interleave `install_runtime_config` calls.
static CFG_LOCK: Mutex<()> = Mutex::new(());

/// The real payoff of running on a CUDA box without sm_90 silicon: NVRTC must
/// accept the TMA kernel for the Hopper virtual arch, which validates the
/// inline PTX (mbarrier / `cp.async.bulk.tensor` / `cvta`).
/// NVRTC must accept a TMA kernel source for the Hopper virtual arch, and the
/// PTX must actually target sm_90a and carry the bulk-tensor copy. This is the
/// real win on a non-sm_90 CUDA box: it validates the hand-written inline PTX.
fn assert_nvrtc_compiles_for_hopper(name: &str, src: &str) {
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_90a"),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)
        .unwrap_or_else(|e| panic!("{name} must NVRTC-compile for compute_90a: {e}"));
    let out = ptx.to_src();
    assert!(out.contains("sm_90a"), "{name}: PTX should target sm_90a");
    assert!(
        out.contains("cp.async.bulk.tensor"),
        "{name}: PTX should carry the TMA bulk-tensor copy"
    );
    eprintln!(
        "[tma_validate] NVRTC compiled {name} for compute_90a ({} bytes PTX)",
        out.len()
    );
}

#[test]
fn nvrtc_compiles_matmul_tma_for_compute_90a() {
    if !crate::is_available() {
        return;
    }
    assert_nvrtc_compiles_for_hopper("matmul_tma", crate::kernels::MATMUL_TMA_CU);
}

#[test]
fn nvrtc_compiles_matmul_bt_tma_for_compute_90a() {
    if !crate::is_available() {
        return;
    }
    assert_nvrtc_compiles_for_hopper("matmul_bt_tma", crate::kernels::MATMUL_BT_TMA_CU);
}

/// `cuTensorMapEncodeTiled` (dyn-loaded via cudarc) behaves per device support.
/// The encode is NOT arch-independent: the driver returns
/// `CUDA_ERROR_NOT_SUPPORTED` on pre-Hopper (no TMA unit), so we assert success
/// on sm_90+ and graceful `None` (→ cuBLAS fallback) below it. Either way this
/// exercises the real host builder against a live driver.
#[test]
fn tma_host_encode_matches_device_support() {
    if !crate::is_available() {
        return;
    }
    let ctx = cuda_context().expect("cuda context");
    let (major, _) = device_cc(&ctx);
    let stream = ctx.default_stream();
    // A [dim1=64, dim0=16] f32 region so the [BK,BM]=[16,64] box fits; the
    // allocator hands back a 256B-aligned base (TMA needs ≥16B).
    let buf = stream.alloc_zeros::<f32>(16 * 64).expect("device alloc");
    let (ptr, _guard) = buf.device_ptr(&stream);
    // dim0=K=16 (contiguous), dim1=M=64, row pitch = 16*4 = 64B (multiple of 16).
    let map = unsafe { tma_encode_tiled_2d(ptr, 16, 64, TMA_BK, TMA_BM, 16 * 4) };
    if major >= 9 {
        assert!(
            map.is_some(),
            "cuTensorMapEncodeTiled must succeed for a valid f32 tile on sm_90+"
        );
        eprintln!("[tma_validate] cuTensorMapEncodeTiled succeeded on sm_90 device");
    } else {
        assert!(
            map.is_none(),
            "pre-Hopper encode must degrade to None (driver returns NOT_SUPPORTED)"
        );
        eprintln!(
            "[tma_validate] host encode correctly unsupported on sm_{major}x → graceful fallback"
        );
    }
}

/// The compute-capability probe must be sane and `tma_arch` must gate exactly:
/// `None` unless `RLX_CUDA_TMA` is set AND the device is sm_90.
#[test]
fn cc_probe_and_tma_arch_gate() {
    if !crate::is_available() {
        return;
    }
    let _guard = CFG_LOCK.lock().unwrap();
    let ctx = cuda_context().expect("cuda context");
    let (major, minor) = device_cc(&ctx);
    assert!(major >= 1, "cc probe returned garbage: {major}.{minor}");
    eprintln!("[tma_validate] device compute capability = {major}.{minor}");

    // Flag OFF → never selects TMA, whatever the arch.
    let mut off = CudaRuntimeConfig::from_env();
    off.tma = false;
    install_runtime_config(off);
    assert!(
        tma_arch(&ctx).is_none(),
        "tma_arch must be None when RLX_CUDA_TMA is off"
    );

    // Flag ON → Some("compute_90a") iff sm_90, else portable fallback (None).
    let mut on = CudaRuntimeConfig::from_env();
    on.tma = true;
    install_runtime_config(on);
    let arch = tma_arch(&ctx);
    if major == 9 {
        assert_eq!(arch, Some("compute_90a"), "sm_90 must select the TMA arch");
    } else {
        assert!(
            arch.is_none(),
            "non-Hopper (sm_{major}{minor}) must fall back to portable, got {arch:?}"
        );
    }

    reload_runtime_config();
}

/// Numeric TMA-vs-CPU GEMM parity. Only executes on sm_90 (the dispatch gates
/// the TMA kernel off elsewhere); on any lower arch it prints a skip and
/// returns. This is the scaffold that will validate the kernel end-to-end the
/// first time an H100 is reachable.
#[test]
fn tma_gemm_matches_cpu_on_hopper() {
    if !crate::is_available() {
        return;
    }
    let _guard = CFG_LOCK.lock().unwrap();
    let ctx = cuda_context().expect("cuda context");
    let (major, _) = device_cc(&ctx);
    if major < 9 {
        eprintln!("[tma_validate] skipping TMA numeric parity: needs sm_90, have sm_{major}x");
        return;
    }

    use rlx_ir::{DType, Graph, Op, Shape};
    let (m, k, n) = (96usize, 128usize, 64usize); // K,N %4 == 0 → TMA-eligible
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.03).collect();

    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += a[r * k + p] * b[p * n + c];
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("tma_gemm_parity");
    let a_in = g.input("a", Shape::new(&[m, k], DType::F32));
    let b_param = g.param("b", Shape::new(&[k, n], DType::F32));
    let y = g.add_node(
        Op::MatMul,
        vec![a_in, b_param],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut cfg = CudaRuntimeConfig::from_env();
    cfg.tma = true;
    install_runtime_config(cfg);

    let mut exe = CudaExecutable::compile(g);
    exe.set_param("b", &b);
    let out = exe.run(&[("a", &a)]);
    reload_runtime_config();

    let max_abs = out[0]
        .iter()
        .zip(&expected)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-2, "TMA GEMM mismatch max|Δ| = {max_abs}");
    eprintln!("[tma_validate] TMA GEMM parity OK (max|Δ| = {max_abs})");
}
