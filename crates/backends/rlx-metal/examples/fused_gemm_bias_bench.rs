// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Option (b): a register-blocked simdgroup GEMM with the bias+ReLU epilogue
//! FOLDED into the store (one dispatch) vs MPS GEMM + a SEPARATE bias+ReLU
//! dispatch (two dispatches — what MPS forces, since it can't fuse). Measures the
//! end-to-end win on the kernels we're required to hand-roll. `C = relu(A·B + bias)`.

use metal::MTLSize;
use rlx_metal::device::metal_device;
use rlx_metal::mps_blas::encode_mps_sgemm_t;
use std::time::Instant;

const KSRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// Register-blocked GEMM (64×64 tile, 4×4 simdgroups, 2×2 acc each) with the
// bias+ReLU epilogue folded through a threadgroup staging tile → one dispatch.
kernel void gemm_rb_bias(
    device const float* A    [[buffer(0)]],   // [M,K]
    device const float* B    [[buffer(1)]],   // [K,N]
    device const float* bias [[buffer(2)]],   // [N]
    device float* C          [[buffer(3)]],   // [M,N]
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    const uint KT = 16;
    uint sg_row = sgid / 4, sg_col = sgid % 4;
    uint rbase = tgid.y * 64, cbase = tgid.x * 64;
    threadgroup float A_tg[64 * 16];
    threadgroup float B_tg[16 * 64];
    threadgroup float C_tg[64 * 64];
    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = c00, c10 = c00, c11 = c00;
    uint lin = sgid * 32 + slid;
    for (uint kk = 0; kk < K; kk += KT) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + lin;
            uint ar = idx / KT, ac = idx % KT;
            A_tg[idx] = A[(rbase + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + lin;
            uint br = idx / 64, bc = idx % 64;
            B_tg[br * 64 + bc] = B[(kk + br) * N + (cbase + bc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ki = 0; ki < KT; ki += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &A_tg[(sg_row * 16 + 0) * KT + ki], KT);
            simdgroup_load(a1, &A_tg[(sg_row * 16 + 8) * KT + ki], KT);
            simdgroup_load(b0, &B_tg[ki * 64 + sg_col * 16 + 0], 64);
            simdgroup_load(b1, &B_tg[ki * 64 + sg_col * 16 + 8], 64);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint cr = sg_row * 16, cc = sg_col * 16;
    simdgroup_store(c00, &C_tg[(cr + 0) * 64 + cc + 0], 64);
    simdgroup_store(c01, &C_tg[(cr + 0) * 64 + cc + 8], 64);
    simdgroup_store(c10, &C_tg[(cr + 8) * 64 + cc + 0], 64);
    simdgroup_store(c11, &C_tg[(cr + 8) * 64 + cc + 8], 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative write-out with the fused epilogue: 64*64=4096 / 512 = 8 each.
    for (uint i = 0; i < 8; ++i) {
        uint idx = i * 512 + lin;
        uint r = idx / 64, cn = idx % 64;
        uint gc = cbase + cn;
        float v = C_tg[idx] + bias[gc];
        C[(rbase + r) * N + gc] = max(v, 0.0f);
    }
}
// The SEPARATE epilogue MPS is forced to run as a second dispatch.
kernel void bias_relu(device const float* tmp [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device float* C [[buffer(2)]], constant uint& N [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    float v = tmp[gid] + bias[gid % N];
    C[gid] = max(v, 0.0f);
}
"#;

fn main() {
    let dev = metal_device().expect("no Metal device");
    let lib = dev
        .device
        .new_library_with_source(KSRC, &metal::CompileOptions::new())
        .expect("compile");
    let pso_fused = dev
        .device
        .new_compute_pipeline_state_with_function(&lib.get_function("gemm_rb_bias", None).unwrap())
        .unwrap();
    let pso_ep = dev
        .device
        .new_compute_pipeline_state_with_function(&lib.get_function("bias_relu", None).unwrap())
        .unwrap();
    println!("device: {}\n", dev.name);
    let align = |x: usize| (x + 255) & !255;
    const NITER: usize = 50;
    const WARM: usize = 8;

    println!(
        "{:>14}  {:>22}  {:>9}  {:>8}",
        "shape m×k×n", "path", "avg_ms", "max|err|"
    );
    println!("{}", "-".repeat(60));

    for (m, kk, n) in [(4096usize, 192usize, 192usize), (4096, 768, 768)] {
        let (x_b, w_b, bias_b, c_b) = (m * kk * 4, kk * n * 4, n * 4, m * n * 4);
        let x_o = 0usize;
        let w_o = align(x_o + x_b);
        let bias_o = align(w_o + w_b);
        let c_o = align(bias_o + bias_b); // fused output
        let tmp_o = align(c_o + c_b); // MPS gemm output
        let c2_o = align(tmp_o + c_b); // MPS+epilogue output
        let buf = dev.alloc_shared(c2_o + c_b);
        let (mut x, mut w, mut bias) = (vec![0f32; m * kk], vec![0f32; kk * n], vec![0f32; n]);
        unsafe {
            let base = buf.contents() as *mut u8;
            for i in 0..m * kk {
                x[i] = ((i * 13 + 7) % 23) as f32 / 23.0;
                *(base.add(x_o) as *mut f32).add(i) = x[i];
            }
            for i in 0..kk * n {
                w[i] = ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5;
                *(base.add(w_o) as *mut f32).add(i) = w[i];
            }
            for i in 0..n {
                bias[i] = (i % 7) as f32 * 0.1 - 0.3;
                *(base.add(bias_o) as *mut f32).add(i) = bias[i];
            }
        }
        // CPU reference: relu(x·W + bias).
        let mut r = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..kk {
                    acc += x[i * kk + p] * w[p * n + j];
                }
                r[i * n + j] = (acc + bias[j]).max(0.0);
            }
        }
        let err = |off: usize| unsafe {
            let base = buf.contents() as *const u8;
            (0..m * n)
                .map(|i| (*(base.add(off) as *const f32).add(i) - r[i]).abs())
                .fold(0f32, f32::max)
        };

        // (b) fused: one dispatch.
        let enc_fused = |cb: &metal::CommandBufferRef| {
            let e = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            e.set_compute_pipeline_state(&pso_fused);
            for (i, o) in [x_o, w_o, bias_o, c_o].iter().enumerate() {
                e.set_buffer(i as u64, Some(&buf), *o as u64);
            }
            for (i, v) in [m as u32, kk as u32, n as u32].iter().enumerate() {
                e.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
            }
            e.dispatch_thread_groups(
                MTLSize {
                    width: (n / 64) as u64,
                    height: (m / 64) as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
        };
        // MPS GEMM (x·W, no transpose) + SEPARATE bias+relu dispatch.
        let enc_mps = |cb: &metal::CommandBufferRef| {
            encode_mps_sgemm_t(cb, &buf, x_o, w_o, tmp_o, m, kk, n, false, false);
            let e = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            e.set_compute_pipeline_state(&pso_ep);
            e.set_buffer(0, Some(&buf), tmp_o as u64);
            e.set_buffer(1, Some(&buf), bias_o as u64);
            e.set_buffer(2, Some(&buf), c2_o as u64);
            let nn = n as u32;
            e.set_bytes(3, 4, &nn as *const u32 as *const _);
            e.dispatch_threads(
                MTLSize {
                    width: (m * n) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            e.end_encoding();
        };
        let fused_ms = time(dev, WARM, NITER, enc_fused);
        let fused_err = err(c_o);
        let mps_ms = time(dev, WARM, NITER, enc_mps);
        let mps_err = err(c2_o);
        println!(
            "{:>14}  {:>22}  {:>9.4}  {:>8.1e}",
            format!("{m}×{kk}×{n}"),
            "fused-RB (1 dispatch)",
            fused_ms,
            fused_err
        );
        println!(
            "{:>14}  {:>22}  {:>9.4}  {:>8.1e}",
            "", "MPS + bias (2 dispatch)", mps_ms, mps_err
        );
        println!(
            "{:>14}  {:>22}  {:>8.2}× fused win",
            "",
            "",
            mps_ms / fused_ms
        );
    }
}

fn time(
    dev: &rlx_metal::device::MetalDevice,
    warm: usize,
    n: usize,
    enc: impl Fn(&metal::CommandBufferRef),
) -> f64 {
    for _ in 0..warm {
        let cb = dev.queue.new_command_buffer();
        enc(cb);
        cb.commit();
        cb.wait_until_completed();
    }
    let cb = dev.queue.new_command_buffer();
    let t0 = Instant::now();
    for _ in 0..n {
        enc(cb);
    }
    cb.commit();
    cb.wait_until_completed();
    t0.elapsed().as_secs_f64() * 1e3 / n as f64
}
