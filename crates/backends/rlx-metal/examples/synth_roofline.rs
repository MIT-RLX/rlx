// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Roofline probe for `Op::SynthMatMul` (codebook weight-synthesis matmul) vs a
//! plain f32 matmul (`metal_sgemm`), in both f32 and f16 (half x/dst) — to see
//! where the native kernels sit (bandwidth- vs compute-bound) and the f16 gain.
//!
//! Run: cargo run --release --example synth_roofline -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use rlx_metal::blas::metal_sgemm;
    use rlx_metal::device::{MetalDevice, metal_device};
    use rlx_metal::kernels::Kernels;
    use std::time::Instant;

    let dev = metal_device().expect("no Metal device");
    let k = rlx_metal::kernels::kernels();
    println!("device: {}\n", dev.name);

    const D: usize = 4; // entry_dim
    const NE: usize = 256; // num_entries (8-bit codes)
    const N_ITER: usize = 50;
    const WARMUP: usize = 5;
    let align16 = |x: usize| (x + 15) & !15;

    // Time the split-K (M≤8) / one-per-output (M>8) synth kernel in f32 or f16
    // (half x/dst). Returns (avg_ms, bytes_moved).
    fn measure_synth(
        dev: &MetalDevice,
        k: &Kernels,
        m: usize,
        kk: usize,
        n: usize,
        half: bool,
        n_iter: usize,
        warmup: usize,
        align16: &dyn Fn(usize) -> usize,
    ) -> (f64, f64, Vec<f32>) {
        let esz = if half { 2 } else { 4 }; // x/dst element size
        let x_bytes = m * kk * esz;
        let idx_bytes = n * (kk / D);
        let cb_bytes = NE * D * 4;
        let dst_bytes = m * n * esz;
        let x_off = 0usize;
        let idx_off = align16(x_off + x_bytes);
        let cb_off = align16(idx_off + idx_bytes);
        let dst_off = align16(cb_off + cb_bytes);
        let buffer = dev.alloc_shared(dst_off + dst_bytes);
        unsafe {
            let base = buffer.contents() as *mut u8;
            for i in 0..m * kk {
                let v = ((i * 13 + 7) % 23) as f32 / 23.0;
                if half {
                    *(base.add(x_off) as *mut u16).add(i) = half::f16::from_f32(v).to_bits();
                } else {
                    *(base.add(x_off) as *mut f32).add(i) = v;
                }
            }
            let ip = base.add(idx_off);
            for i in 0..idx_bytes {
                *ip.add(i) = ((i * 7 + 1) % NE) as u8;
            }
            let cp = base.add(cb_off) as *mut f32;
            for i in 0..NE * D {
                *cp.add(i) = ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5;
            }
        }
        let split_k = m <= 8;
        let pso = match (split_k, half) {
            (true, false) => &k.synth_matmul_codebook,
            (true, true) => &k.synth_matmul_codebook_h,
            (false, false) => &k.synth_matmul_codebook_mm,
            (false, true) => &k.synth_matmul_codebook_mm_h,
        };
        let encode = |enc: &metal::ComputeCommandEncoderRef| {
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(&buffer), 0);
            let vals: [u64; 4] = [x_off as u64, idx_off as u64, cb_off as u64, dst_off as u64];
            for (i, v) in vals.iter().enumerate() {
                enc.set_bytes((i + 1) as u64, 8, v as *const u64 as *const _);
            }
            let scalars: [u32; 5] = [kk as u32, n as u32, D as u32, NE as u32, m as u32];
            for (i, v) in scalars.iter().enumerate() {
                enc.set_bytes((i + 5) as u64, 4, v as *const u32 as *const _);
            }
            let (grid, tg) = if split_k {
                (
                    metal::MTLSize {
                        width: 32,
                        height: n as u64,
                        depth: m as u64,
                    },
                    metal::MTLSize {
                        width: 32,
                        height: 8u64.min(n as u64).max(1),
                        depth: 1,
                    },
                )
            } else {
                let tgh = 8u64.min(m as u64).max(1);
                let tgw = (256 / tgh).min(n as u64).max(1);
                (
                    metal::MTLSize {
                        width: n as u64,
                        height: m as u64,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: tgw,
                        height: tgh,
                        depth: 1,
                    },
                )
            };
            enc.dispatch_threads(grid, tg);
        };
        for _ in 0..warmup {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            encode(enc);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let cb = dev.queue.new_command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
        let t0 = Instant::now();
        for _ in 0..n_iter {
            encode(enc);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n_iter as f64;
        let bytes = (x_bytes + idx_bytes + cb_bytes + dst_bytes) as f64;
        let out: Vec<f32> = unsafe {
            let base = buffer.contents() as *const u8;
            (0..m * n)
                .map(|i| {
                    if half {
                        half::f16::from_bits(*(base.add(dst_off) as *const u16).add(i)).to_f32()
                    } else {
                        *(base.add(dst_off) as *const f32).add(i)
                    }
                })
                .collect()
        };
        (ms, bytes, out)
    }

    let cases = [
        (1usize, 4096usize, 4096usize),
        (1, 4096, 11008),
        (256, 4096, 4096),
    ];

    println!(
        "{:>18}  {:>11}  {:>9}  {:>9}  {:>8}",
        "shape (m×k×n)", "kernel", "avg_ms", "GFLOP/s", "GB/s"
    );
    println!("{}", "-".repeat(62));

    for (m, kk, n) in cases {
        let flops = 2.0 * (m * kk * n) as f64;
        let gflops = |ms: f64| flops / 1e9 / (ms / 1e3);
        let gbps = |bytes: f64, ms: f64| bytes / 1e9 / (ms / 1e3);

        // f32 matmul baseline
        let sg_bytes = ((m * kk + kk * n + m * n) * 4) as f64;
        let sg_ms = {
            let arena = (m * kk + kk * n + m * n) * 4;
            let buffer = dev.alloc_shared(arena);
            unsafe {
                let p = buffer.contents() as *mut f32;
                for i in 0..m * kk {
                    *p.add(i) = ((i * 13 + 7) % 23) as f32 / 23.0;
                }
                for i in 0..kk * n {
                    *p.add(m * kk + i) = ((i * 17 + 3) % 31) as f32 / 31.0;
                }
            }
            let (a_off, b_off, c_off) = (0usize, m * kk * 4, (m * kk + kk * n) * 4);
            for _ in 0..WARMUP {
                let cb = dev.queue.new_command_buffer();
                let enc =
                    cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
                metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, kk, n);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            let t0 = Instant::now();
            for _ in 0..N_ITER {
                metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, kk, n);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            t0.elapsed().as_secs_f64() * 1e3 / N_ITER as f64
        };

        let (syn32_ms, syn32_bytes, out32) =
            measure_synth(dev, k, m, kk, n, false, N_ITER, WARMUP, &align16);
        let (syn16_ms, syn16_bytes, out16) =
            measure_synth(dev, k, m, kk, n, true, N_ITER, WARMUP, &align16);

        // f16 correctness: the macro-generated f16 kernel must match f32 within
        // f16 rounding (x/dst are half; codebook + accumulation stay f32).
        let err = out32
            .iter()
            .zip(&out16)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = out32.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
        assert!(
            err / scale < 5e-2,
            "f16 kernel diverges from f32 (m={m}): rel err {}",
            err / scale
        );

        // The two "reconstruct → real GEMM" prefill paths share a reconstruction
        // cost (build dense [k,n] f32 weight from u8 indices + f32 codebook), then
        // differ only in the GEMM: AMX (Accelerate, CPU) vs MPS (GPU, = f32-sgemm).
        let amx_ms = {
            let a: Vec<f32> = (0..m * kk)
                .map(|i| ((i * 13 + 7) % 23) as f32 / 23.0)
                .collect();
            let b: Vec<f32> = (0..kk * n)
                .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0)
                .collect();
            let mut c = vec![0f32; m * n];
            for _ in 0..WARMUP {
                rlx_cpu::blas::sgemm(&a, &b, &mut c, m, kk, n);
            }
            let t0 = Instant::now();
            for _ in 0..N_ITER {
                rlx_cpu::blas::sgemm(&a, &b, &mut c, m, kk, n);
            }
            t0.elapsed().as_secs_f64() * 1e3 / N_ITER as f64
        };
        let recon_ms = {
            let kb = kk / D;
            let idx: Vec<u8> = (0..n * kb).map(|i| ((i * 7 + 1) % NE) as u8).collect();
            let cbk: Vec<f32> = (0..NE * D)
                .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0 - 0.5)
                .collect();
            let mut w = vec![0f32; kk * n];
            // Cache-friendly: reconstruct Wᵀ as [n, k] contiguous (matches the
            // [n, k/d] index layout) — for the GEMM use a transposed-B sgemm.
            // (The stride-n [k,n] write is the cache-hostile pattern to avoid.)
            let recon = |w: &mut [f32]| {
                for j in 0..n {
                    let wj = &mut w[j * kk..j * kk + kk];
                    for bb in 0..kb {
                        let code = idx[j * kb + bb] as usize;
                        for t in 0..D {
                            wj[bb * D + t] = cbk[code * D + t];
                        }
                    }
                }
            };
            for _ in 0..WARMUP {
                recon(&mut w);
            }
            let t0 = Instant::now();
            for _ in 0..N_ITER {
                recon(&mut w);
            }
            t0.elapsed().as_secs_f64() * 1e3 / N_ITER as f64
        };

        let shape = format!("{}×{}×{}", m, kk, n);
        println!(
            "{:>18}  {:>11}  {:>9.3}  {:>9.0}  {:>8.0}",
            shape,
            "f32-sgemm",
            sg_ms,
            gflops(sg_ms),
            gbps(sg_bytes, sg_ms)
        );
        println!(
            "{:>18}  {:>11}  {:>9.3}  {:>9.0}  {:>8.0}",
            "",
            "synth-f32",
            syn32_ms,
            gflops(syn32_ms),
            gbps(syn32_bytes, syn32_ms)
        );
        println!(
            "{:>18}  {:>11}  {:>9.3}  {:>9.0}  {:>8.0}",
            "",
            "synth-f16",
            syn16_ms,
            gflops(syn16_ms),
            gbps(syn16_bytes, syn16_ms)
        );
        println!(
            "{:>18}  {:>11}  {:>9.3}  {:>9.0}  {:>8.0}",
            "",
            "amx-sgemm",
            amx_ms,
            gflops(amx_ms),
            gbps(sg_bytes, amx_ms)
        );
        println!(
            "{:>18}  reconstruct(host)={:.3}ms  |  recon+MPS={:.3}ms  recon+AMX={:.3}ms  |  fused-synth={:.3}ms",
            "",
            recon_ms,
            recon_ms + sg_ms,
            recon_ms + amx_ms,
            syn32_ms,
        );
        println!();
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal only on macOS");
}
