// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! RE-informed experiment: a 32×32 simdgroup-matrix tiled GEMM that RECONSTRUCTS
//! the weight tile into threadgroup memory from codebook+indices (no materialized
//! W, no W DRAM read) vs the decompose path (GPU reconstruct → MPS sgemm).
//! `y = x[m,k] · Ŵ[k,n]`, Ŵ[p,j] = codebook[indices[j, p/d]][p%d]. All f32.

use metal::MTLSize;
use rlx_metal::device::metal_device;
use rlx_metal::mps_blas::encode_mps_sgemm_t;
use std::time::Instant;

const D: usize = 4;
const NE: usize = 256;

const KSRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// A=x[m,k] loaded; B=Ŵ[k,n] reconstructed in threadgroup; C=y[m,n]. 32×32 tiles,
// 16 simdgroups (512 threads), 8×8 simdgroup MMA. m,k,n multiples of 32.
kernel void synth_matmul_simd(
    device const float* A        [[buffer(0)]],
    device const uchar* indices  [[buffer(1)]],
    device const float* codebook [[buffer(2)]],
    device float* C              [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;
    uint tg_row_base = tgid.y * 32;
    uint tg_col_base = tgid.x * 32;
    uint nb = K / d;
    threadgroup float A_tg[32 * 32];
    threadgroup float B_tg[32 * 32];
    simdgroup_float8x8 a, b;
    simdgroup_float8x8 c = simdgroup_float8x8(0.0f);
    uint linear = sgid * 32 + slid; // [0,512)
    for (uint kk = 0; kk < K; kk += 32) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint ar = idx / 32, ac = idx % 32;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;
            uint br = idx / 32, bc = idx % 32;
            uint p = kk + br;             // K index
            uint j = tg_col_base + bc;    // N index (output col)
            uint code = uint(indices[j * nb + (p / d)]);
            B_tg[idx] = codebook[code * d + (p % d)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ki = 0; ki < 32; ki += 8) {
            simdgroup_load(a, &A_tg[sg_row * 8 * 32 + ki], 32);
            simdgroup_load(b, &B_tg[ki * 32 + sg_col * 8], 32);
            simdgroup_multiply_accumulate(c, a, b, c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint out_row = tg_row_base + sg_row * 8;
    uint out_col = tg_col_base + sg_col * 8;
    simdgroup_store(c, &C[out_row * N + out_col], N);
}

// Register-blocked: 64×64 output tile, 4×4 simdgroups, each accumulates a 2×2 grid
// of 8×8 (16×16 output) in 4 registers → 4× the MMA per loaded tile (MPS-style
// arithmetic intensity). Still synchronous loads. m,n mult of 64; k mult of 16.
kernel void synth_matmul_simd_rb(
    device const float* A        [[buffer(0)]],
    device const uchar* indices  [[buffer(1)]],
    device const float* codebook [[buffer(2)]],
    device float* C              [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint slid  [[thread_index_in_simdgroup]]
) {
    const uint KT = 16;
    uint sg_row = sgid / 4;   // [0,4)
    uint sg_col = sgid % 4;   // [0,4)
    uint tg_row_base = tgid.y * 64;
    uint tg_col_base = tgid.x * 64;
    uint nb = K / d;
    threadgroup float A_tg[64 * 16];
    threadgroup float B_tg[16 * 64];
    simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = c00, c10 = c00, c11 = c00;
    uint linear = sgid * 32 + slid; // [0,512)
    for (uint kk = 0; kk < K; kk += KT) {
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;   // [0,1024)
            uint ar = idx / KT, ac = idx % KT;
            A_tg[idx] = A[(tg_row_base + ar) * K + (kk + ac)];
        }
        for (uint i = 0; i < 2; ++i) {
            uint idx = i * 512 + linear;   // [0,1024)
            uint br = idx / 64, bc = idx % 64;
            uint p = kk + br;
            uint j = tg_col_base + bc;
            uint code = uint(indices[j * nb + (p / d)]);
            B_tg[br * 64 + bc] = codebook[code * d + (p % d)];
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
    uint or0 = tg_row_base + sg_row * 16;
    uint oc0 = tg_col_base + sg_col * 16;
    simdgroup_store(c00, &C[(or0 + 0) * N + oc0 + 0], N);
    simdgroup_store(c01, &C[(or0 + 0) * N + oc0 + 8], N);
    simdgroup_store(c10, &C[(or0 + 8) * N + oc0 + 0], N);
    simdgroup_store(c11, &C[(or0 + 8) * N + oc0 + 8], N);
}
"#;

fn main() {
    let dev = metal_device().expect("no Metal device");
    let k = rlx_metal::kernels::kernels();
    println!("device: {}\n", dev.name);
    let lib = dev
        .device
        .new_library_with_source(KSRC, &metal::CompileOptions::new())
        .expect("compile");
    let func = lib.get_function("synth_matmul_simd", None).unwrap();
    let simd_pso = dev
        .device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();
    let rb_func = lib.get_function("synth_matmul_simd_rb", None).unwrap();
    let rb_pso = dev
        .device
        .new_compute_pipeline_state_with_function(&rb_func)
        .unwrap();

    let align = |x: usize| (x + 255) & !255;
    const NITER: usize = 50;
    const WARM: usize = 8;

    println!(
        "{:>16}  {:>16}  {:>9}  {:>9}  {:>8}",
        "shape m×k×n", "kernel", "avg_ms", "GFLOP/s", "max|err|"
    );
    println!("{}", "-".repeat(66));

    // Real synth training shapes (batch·seq × in × out), all multiples of 32.
    for (m, kk, n) in [
        (4096usize, 192usize, 192usize),
        (4096, 768, 192),
        (4096, 192, 768),
    ] {
        let nb = kk / D;
        let x_b = m * kk * 4;
        let idx_b = n * nb;
        let cb_b = NE * D * 4;
        let w_b = kk * n * 4; // reconstructed dense weight scratch (decompose path)
        let dst_b = m * n * 4;
        let (x_o, idx_o) = (0usize, align(x_b));
        let cb_o = align(idx_o + idx_b);
        let w_o = align(cb_o + cb_b);
        let dst_o = align(w_o + w_b);
        let dst2_o = align(dst_o + dst_b);
        let buf = dev.alloc_shared(dst2_o + dst_b);
        // Fill inputs deterministically + build CPU reference.
        let mut x = vec![0f32; m * kk];
        let mut idx = vec![0u8; n * nb];
        let mut cb = vec![0f32; NE * D];
        unsafe {
            let base = buf.contents() as *mut u8;
            for i in 0..m * kk {
                let v = ((i * 13 + 7) % 23) as f32 / 23.0;
                x[i] = v;
                *(base.add(x_o) as *mut f32).add(i) = v;
            }
            for i in 0..n * nb {
                let c = ((i * 7 + 1) % NE) as u8;
                idx[i] = c;
                *base.add(idx_o).add(i) = c;
            }
            for i in 0..NE * D {
                let v = ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5;
                cb[i] = v;
                *(base.add(cb_o) as *mut f32).add(i) = v;
            }
        }
        // CPU reference: y = x · Ŵ, Ŵ[p,j] = cb[idx[j*nb + p/D]][p%D].
        let mut refy = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..kk {
                    let code = idx[j * nb + p / D] as usize;
                    acc += x[i * kk + p] * cb[code * D + p % D];
                }
                refy[i * n + j] = acc;
            }
        }
        let flops = 2.0 * (m * kk * n) as f64;
        let maxerr = |off: usize| -> f32 {
            let base = buf.contents() as *const u8;
            (0..m * n)
                .map(|i| unsafe { (*(base.add(off) as *const f32).add(i) - refy[i]).abs() })
                .fold(0f32, f32::max)
        };

        // ── (1) fused reconstruct-in-tile simdgroup GEMM (one dispatch) ──
        let enc_simd = |cb: &metal::CommandBufferRef| {
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            enc.set_compute_pipeline_state(&simd_pso);
            enc.set_buffer(0, Some(&buf), x_o as u64);
            enc.set_buffer(1, Some(&buf), idx_o as u64);
            enc.set_buffer(2, Some(&buf), cb_o as u64);
            enc.set_buffer(3, Some(&buf), dst_o as u64);
            for (i, v) in [m as u32, kk as u32, n as u32, D as u32].iter().enumerate() {
                enc.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
                MTLSize {
                    width: (n / 32) as u64,
                    height: (m / 32) as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        };
        let simd_ms = time(dev, WARM, NITER, enc_simd);
        let simd_err = maxerr(dst_o);

        // ── (1b) register-blocked (64×64 tile, 2×2 accumulators/simdgroup) ──
        let enc_rb = |cb: &metal::CommandBufferRef| {
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            enc.set_compute_pipeline_state(&rb_pso);
            enc.set_buffer(0, Some(&buf), x_o as u64);
            enc.set_buffer(1, Some(&buf), idx_o as u64);
            enc.set_buffer(2, Some(&buf), cb_o as u64);
            enc.set_buffer(3, Some(&buf), dst_o as u64);
            for (i, v) in [m as u32, kk as u32, n as u32, D as u32].iter().enumerate() {
                enc.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_thread_groups(
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
            enc.end_encoding();
        };
        let rb_ms = time(dev, WARM, NITER, enc_rb);
        let rb_err = maxerr(dst_o);

        // ── (2) decompose: GPU reconstruct Ŵᵀ[n,k] → scratch, then MPS sgemm
        //        (transposeRight so x·(Ŵᵀ)ᵀ = x·Ŵ). ──
        let enc_dec = |cb: &metal::CommandBufferRef| {
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            enc.set_compute_pipeline_state(&k.synth_reconstruct);
            enc.set_buffer(0, Some(&buf), 0);
            for (i, v) in [idx_o as u64, cb_o as u64, w_o as u64].iter().enumerate() {
                enc.set_bytes((i + 1) as u64, 8, v as *const u64 as *const _);
            }
            for (i, v) in [kk as u32, n as u32, D as u32].iter().enumerate() {
                enc.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_threads(
                MTLSize {
                    width: nb as u64,
                    height: n as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 8,
                    height: 8,
                    depth: 1,
                },
            );
            enc.end_encoding();
            encode_mps_sgemm_t(cb, &buf, x_o, w_o, dst2_o, m, kk, n, false, true);
        };
        let dec_ms = time(dev, WARM, NITER, enc_dec);
        let dec_err = maxerr(dst2_o);

        let gf = |ms: f64| flops / 1e9 / (ms / 1e3);
        println!(
            "{:>16}  {:>16}  {:>9.4}  {:>9.0}  {:>8.1e}",
            format!("{m}×{kk}×{n}"),
            "simd-fused",
            simd_ms,
            gf(simd_ms),
            simd_err
        );
        println!(
            "{:>16}  {:>16}  {:>9.4}  {:>9.0}  {:>8.1e}",
            "",
            "simd-fused-RB",
            rb_ms,
            gf(rb_ms),
            rb_err
        );
        println!(
            "{:>16}  {:>16}  {:>9.4}  {:>9.0}  {:>8.1e}",
            "",
            "recon+MPS",
            dec_ms,
            gf(dec_ms),
            dec_err
        );
        println!(
            "{:>16}  {:>16}  naive {:>5.2}×   RB {:>5.2}×  (vs recon+MPS)",
            "",
            "speedup",
            dec_ms / simd_ms,
            dec_ms / rb_ms
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
