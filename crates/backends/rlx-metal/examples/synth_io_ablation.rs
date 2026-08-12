// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! IO-lever ablation + stacking for `Op::SynthMatMul`. Self-contained: compiles its
//! own MSL variants and controls the data format so each lever is isolated.
//!
//! DECODE (M=1, bandwidth-bound) — the indices are the dominant DRAM traffic:
//!   L2 = sub-byte (4-bit) packed indices (num_entries<=16)  → halve the index read
//!   L4 = f16 codebook                                        → halve the (tiny) codebook
//!   ablated as base / +L2 / +L4 / +L2+L4 (stacked).
//! PREFILL (M large) — the recon→MPS weight roundtrip is the traffic:
//!   L1 = f16 reconstruct → MPS f16 (halve the 16 MB scratch write+read + faster GEMM)
//!   L3 = double-buffered fused tiled kernel (hide the synchronous-load stall)
//!
//! Run: cargo run --release --example synth_io_ablation -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use half::f16;
    use rlx_metal::mtl::MTLSize;
    use std::time::Instant;

    let dev = rlx_metal::device::metal_device().expect("no Metal device");
    let device = &dev.device;
    let queue = &dev.queue;
    println!("device: {}\n", dev.name);

    // ---- decode shapes: indices dominate (num_entries<=16 so 4-bit is valid) ----
    const K: usize = 4096;
    const N: usize = 4096;
    const D: usize = 4;
    const NE: usize = 16;
    const WARMUP: usize = 8;
    const ITERS: usize = 60;
    let nb = K / D; // 1024 blocks; even → clean 4-bit packing

    // ---- decode MSL: base + 3 lever variants (all M=1 split-K, 32-lane simd_sum) ----
    let decode_src = r#"
#include <metal_stdlib>
using namespace metal;
#define HEAD \
    device float* arena [[buffer(0)]], constant ulong& x_off [[buffer(1)]], \
    constant ulong& idx_off [[buffer(2)]], constant ulong& cb_off [[buffer(3)]], \
    constant ulong& dst_off [[buffer(4)]], constant uint& k_dim [[buffer(5)]], \
    constant uint& n_dim [[buffer(6)]], constant uint& d [[buffer(7)]], \
    uint3 gid [[thread_position_in_grid]]

kernel void gemv_base(HEAD) {
    uint split = gid.x, j = gid.y; if (j >= n_dim) return;
    uint nb = k_dim / d;
    device const float* x   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {
        uint code = uint(idx[b]);
        device const float* c = cb + (ulong)code*d; uint base = b*d;
        for (uint t=0;t<d;++t) acc += x[base+t]*c[t];
    }
    acc = simd_sum(acc); if (split==0) dst[j] = acc;
}
kernel void gemv_idx4(HEAD) {              // L2: 4-bit packed indices (2 codes/byte)
    uint split = gid.x, j = gid.y; if (j >= n_dim) return;
    uint nb = k_dim / d;
    device const float* x   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*(nb>>1);
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {
        uint code = (uint(idx[b>>1]) >> ((b&1u)*4u)) & 0xFu;
        device const float* c = cb + (ulong)code*d; uint base = b*d;
        for (uint t=0;t<d;++t) acc += x[base+t]*c[t];
    }
    acc = simd_sum(acc); if (split==0) dst[j] = acc;
}
kernel void gemv_cbf16(HEAD) {             // L4: f16 codebook
    uint split = gid.x, j = gid.y; if (j >= n_dim) return;
    uint nb = k_dim / d;
    device const float* x   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*nb;
    device const half*  cb  = (device const half*)((device const char*)arena + cb_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {
        uint code = uint(idx[b]);
        device const half* c = cb + (ulong)code*d; uint base = b*d;
        for (uint t=0;t<d;++t) acc += x[base+t]*float(c[t]);
    }
    acc = simd_sum(acc); if (split==0) dst[j] = acc;
}
kernel void gemv_vec(HEAD) {               // LATENCY: float4 x + codebook loads (d==4), uchar4 index
    uint split = gid.x, j = gid.y; if (j >= n_dim) return;
    uint nb = k_dim / d;
    device const float4* x  = (device const float4*)((device const char*)arena + x_off);
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*nb;
    device const float4* cb = (device const float4*)((device const char*)arena + cb_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {          // one float4 dot per block (d==4)
        uint code = uint(idx[b]);
        acc += dot(x[b], cb[code]);
    }
    acc = simd_sum(acc); if (split==0) dst[j] = acc;
}
kernel void gemv_idx4_cbf16(HEAD) {        // L2+L4 stacked
    uint split = gid.x, j = gid.y; if (j >= n_dim) return;
    uint nb = k_dim / d;
    device const float* x   = (device const float*)((device const char*)arena + x_off);
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*(nb>>1);
    device const half*  cb  = (device const half*)((device const char*)arena + cb_off);
    device float* dst = (device float*)((device char*)arena + dst_off);
    float acc = 0.0f;
    for (uint b = split; b < nb; b += 32u) {
        uint code = (uint(idx[b>>1]) >> ((b&1u)*4u)) & 0xFu;
        device const half* c = cb + (ulong)code*d; uint base = b*d;
        for (uint t=0;t<d;++t) acc += x[base+t]*float(c[t]);
    }
    acc = simd_sum(acc); if (split==0) dst[j] = acc;
}
"#;
    let lib = device
        .new_library_with_source(decode_src, &rlx_metal::mtl::CompileOptions::new())
        .expect("decode MSL compile");
    let pipe = |name: &str| {
        let f = lib.get_function(name, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };
    let (p_base, p_idx4, p_cbf16, p_idx4_cbf16, p_vec) = (
        pipe("gemv_base"),
        pipe("gemv_idx4"),
        pipe("gemv_cbf16"),
        pipe("gemv_idx4_cbf16"),
        pipe("gemv_vec"),
    );

    // ---- data (host) ----
    let x: Vec<f32> = (0..K).map(|i| (i as f32 * 0.01).sin()).collect();
    let codes: Vec<u8> = (0..N * nb).map(|i| (i % NE) as u8).collect(); // 0..15
    let cb_f32: Vec<f32> = (0..NE * D).map(|i| (i as f32 * 0.1).cos() * 0.7).collect();

    // packed 4-bit indices: 2 codes per byte
    let mut codes4 = vec![0u8; N * nb / 2];
    for (i, chunk) in codes.chunks(2).enumerate() {
        codes4[i] = (chunk[0] & 0xF) | ((chunk[1] & 0xF) << 4);
    }
    let cb_f16: Vec<f16> = cb_f32.iter().map(|v| f16::from_f32(*v)).collect();

    // ---- arena layout (16-byte aligned offsets) ----
    let align = |b: usize| (b + 15) & !15;
    let x_off = 0usize;
    let idx_off = align(x_off + x.len() * 4);
    let idx4_off = align(idx_off + codes.len());
    let cbf32_off = align(idx4_off + codes4.len());
    let cbf16_off = align(cbf32_off + cb_f32.len() * 4);
    let dst_off = align(cbf16_off + cb_f16.len() * 2);
    let total = align(dst_off + N * 4);

    let buffer = device.new_buffer(
        total as u64,
        rlx_metal::mtl::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = buffer.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, base.add(x_off), x.len() * 4);
        std::ptr::copy_nonoverlapping(codes.as_ptr(), base.add(idx_off), codes.len());
        std::ptr::copy_nonoverlapping(codes4.as_ptr(), base.add(idx4_off), codes4.len());
        std::ptr::copy_nonoverlapping(
            cb_f32.as_ptr() as *const u8,
            base.add(cbf32_off),
            cb_f32.len() * 4,
        );
        std::ptr::copy_nonoverlapping(
            cb_f16.as_ptr() as *const u8,
            base.add(cbf16_off),
            cb_f16.len() * 2,
        );
    }

    let run =
        |pipe: &rlx_metal::mtl::ComputePipelineState, idxo: usize, cbo: usize| -> (f64, Vec<f32>) {
            // Encode `count` dispatches into ONE command buffer, single commit+wait —
            // amortizes the ~0.1 ms commit/wait so we time the KERNEL, not dispatch.
            let batch = |count: usize| {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(&buffer), 0);
                for (i, v) in [x_off as u64, idxo as u64, cbo as u64, dst_off as u64]
                    .iter()
                    .enumerate()
                {
                    enc.set_bytes(1 + i as u64, 8, v as *const u64 as *const _);
                }
                for (i, v) in [K as u32, N as u32, D as u32].iter().enumerate() {
                    enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
                }
                for _ in 0..count {
                    enc.dispatch_threads(
                        MTLSize {
                            width: 32,
                            height: N as u64,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };
            batch(WARMUP);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = Instant::now();
                batch(ITERS);
                best = best.min(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
            }
            let out = unsafe {
                let p = (buffer.contents() as *const u8).add(dst_off) as *const f32;
                (0..N).map(|i| *p.add(i)).collect()
            };
            (best, out)
        };

    // CPU reference for correctness.
    let mut want = vec![0f32; N];
    for j in 0..N {
        let mut acc = 0f32;
        for b in 0..nb {
            let code = codes[j * nb + b] as usize;
            for t in 0..D {
                acc += x[b * D + t] * cb_f32[code * D + t];
            }
        }
        want[j] = acc;
    }
    let relerr = |o: &[f32]| {
        let s = want.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        o.iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / s
    };

    let (base_ms, o_base) = run(&p_base, idx_off, cbf32_off);
    let (l2_ms, o_l2) = run(&p_idx4, idx4_off, cbf32_off);
    let (l4_ms, o_l4) = run(&p_cbf16, idx_off, cbf16_off);
    let (l24_ms, o_l24) = run(&p_idx4_cbf16, idx4_off, cbf16_off);

    let idx_b = |bits: usize| N * nb * bits / 8;
    let cb_b = |bytes: usize| NE * D * bytes;
    let kb = |b: usize| b as f64 / 1024.0;
    println!("=== DECODE (M=1, K={K}, N={N}, d={D}, entries={NE}) — indices dominate ===\n");
    println!(
        "{:>14} {:>10} {:>9} {:>9} {:>9} {:>8}",
        "variant", "idx", "codebook", "relerr", "ms", "speedup"
    );
    println!("{}", "-".repeat(64));
    let row = |name: &str, ib: usize, cbb: usize, err: f32, ms: f64| {
        println!(
            "{:>14} {:>8.0} KB {:>7.0} B {:>9.1e} {:>9.3} {:>7.2}x",
            name,
            kb(ib),
            cbb as f64,
            err,
            ms,
            base_ms / ms
        );
    };
    row("base (u8,f32)", idx_b(8), cb_b(4), relerr(&o_base), base_ms);
    row("L2 idx4", idx_b(4), cb_b(4), relerr(&o_l2), l2_ms);
    row("L4 cb_f16", idx_b(8), cb_b(2), relerr(&o_l4), l4_ms);
    row("L2+L4", idx_b(4), cb_b(2), relerr(&o_l24), l24_ms);
    println!(
        "\ndecode read footprint: base {:.1} KB → L2+L4 {:.1} KB ({:.2}x less)",
        kb(idx_b(8) + cb_b(4)),
        kb(idx_b(4) + cb_b(2)),
        (idx_b(8) + cb_b(4)) as f64 / (idx_b(4) + cb_b(2)) as f64
    );

    // ===================== LATENCY (decode is latency-bound) =====================
    // (a) vectorized loads (float4 x + codebook, d==4) vs scalar — hide memory latency
    //     inside the kernel. (b) per-dispatch (own command buffer + commit/wait) vs
    //     batched (N dispatches, one commit/wait) — isolates the kernel-LAUNCH tax that
    //     dominates real decode (dozens of kernels/token). This is the #1 lever: ICB /
    //     graph capture amortizes it across the whole step.
    let run_ub = |pipe: &rlx_metal::mtl::ComputePipelineState| -> f64 {
        let one = |pipe: &rlx_metal::mtl::ComputePipelineState| {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&buffer), 0);
            for (i, v) in [
                x_off as u64,
                idx_off as u64,
                cbf32_off as u64,
                dst_off as u64,
            ]
            .iter()
            .enumerate()
            {
                enc.set_bytes(1 + i as u64, 8, v as *const u64 as *const _);
            }
            for (i, v) in [K as u32, N as u32, D as u32].iter().enumerate() {
                enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
            }
            enc.dispatch_threads(
                MTLSize {
                    width: 32,
                    height: N as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 8,
                    depth: 1,
                },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        for _ in 0..WARMUP {
            one(pipe);
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                one(pipe);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        best
    };
    let (vec_ms, o_vec) = run(&p_vec, idx_off, cbf32_off);
    let base_ub = run_ub(&p_base);
    println!("\n=== LATENCY (decode M=1) ===\n");
    println!("{:>28} {:>9} {:>8}", "variant", "ms", "speedup");
    println!("{}", "-".repeat(48));
    println!(
        "{:>28} {:>9.3} {:>7.2}x",
        "scalar loads (batched)", base_ms, 1.0
    );
    println!(
        "{:>28} {:>9.3} {:>7.2}x  relerr={:.0e}",
        "float4 loads (batched)",
        vec_ms,
        base_ms / vec_ms,
        relerr(&o_vec)
    );
    println!(
        "{:>28} {:>9.3} {:>7.2}x",
        "scalar loads (per-dispatch)",
        base_ub,
        base_ms / base_ub
    );
    println!(
        "\nlaunch tax = per-dispatch − batched = {:.3} − {:.3} = {:.3} ms/dispatch ({:.0}% of the naive time)",
        base_ub,
        base_ms,
        base_ub - base_ms,
        (base_ub - base_ms) / base_ub * 100.0
    );
    println!(
        "→ a real decode step is dozens of kernels; ICB/graph-capture amortizes that tax across all of them."
    );

    // ============================ PREFILL (L1) ============================
    // recon→MPS moves the reconstructed weight through DRAM (write scratch + MPS
    // read). L1 reconstructs f16 → half the 2·k·n·? roundtrip + MPS f16 GEMM.
    const MP: usize = 256;
    const KP: usize = 2048;
    const NP: usize = 2048;
    let nbp = KP / D;
    let recon_src = r#"
#include <metal_stdlib>
using namespace metal;
// Reconstruct dense W[k,n] (row-major, stride n) from u8 indices + f32 codebook.
kernel void recon_f32(device float* arena [[buffer(0)]],
    constant ulong& idx_off [[buffer(1)]], constant ulong& cb_off [[buffer(2)]],
    constant ulong& w_off [[buffer(3)]], constant uint& k_dim [[buffer(4)]],
    constant uint& n_dim [[buffer(5)]], constant uint& d [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint nb = k_dim / d; uint blk = gid.x, j = gid.y; if (blk>=nb||j>=n_dim) return;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device float* w = (device float*)((device char*)arena + w_off);
    uint code = uint(idx[blk]); device const float* c = cb + (ulong)code*d;
    for (uint t=0;t<d;++t) w[(ulong)(blk*d+t)*n_dim + j] = c[t];
}
kernel void recon_f16(device float* arena [[buffer(0)]],
    constant ulong& idx_off [[buffer(1)]], constant ulong& cb_off [[buffer(2)]],
    constant ulong& w_off [[buffer(3)]], constant uint& k_dim [[buffer(4)]],
    constant uint& n_dim [[buffer(5)]], constant uint& d [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint nb = k_dim / d; uint blk = gid.x, j = gid.y; if (blk>=nb||j>=n_dim) return;
    device const uchar* idx = (device const uchar*)arena + idx_off + (ulong)j*nb;
    device const float* cb  = (device const float*)((device const char*)arena + cb_off);
    device half* w = (device half*)((device char*)arena + w_off);
    uint code = uint(idx[blk]); device const float* c = cb + (ulong)code*d;
    for (uint t=0;t<d;++t) w[(ulong)(blk*d+t)*n_dim + j] = (half)c[t];
}
kernel void castf32f16(device float* arena [[buffer(0)]], constant ulong& s_off [[buffer(1)]],
    constant ulong& d_off [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid>=n) return;
    device const float* s = (device const float*)((device const char*)arena + s_off);
    device half* d = (device half*)((device char*)arena + d_off); d[gid] = (half)s[gid];
}
kernel void castf16f32(device float* arena [[buffer(0)]], constant ulong& s_off [[buffer(1)]],
    constant ulong& d_off [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid>=n) return;
    device const half* s = (device const half*)((device const char*)arena + s_off);
    device float* d = (device float*)((device char*)arena + d_off); d[gid] = float(s[gid]);
}
"#;
    let rlib = device
        .new_library_with_source(recon_src, &rlx_metal::mtl::CompileOptions::new())
        .unwrap();
    let rpipe = |n: &str| {
        let f = rlib.get_function(n, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };
    let (p_rf32, p_rf16, p_cf, p_cb2) = (
        rpipe("recon_f32"),
        rpipe("recon_f16"),
        rpipe("castf32f16"),
        rpipe("castf16f32"),
    );

    let xp: Vec<f32> = (0..MP * KP).map(|i| (i as f32 * 0.003).sin()).collect();
    let idxp: Vec<u8> = (0..NP * nbp).map(|i| (i % NE) as u8).collect();
    let a2 = |b: usize| (b + 255) & !255;
    let xp_off = 0usize;
    let idxp_off = a2(xp_off + xp.len() * 4);
    let cbp_off = a2(idxp_off + idxp.len());
    let w32_off = a2(cbp_off + cb_f32.len() * 4);
    let w16_off = a2(w32_off + KP * NP * 4);
    let dstp_off = a2(w16_off + KP * NP * 2);
    let xp16_off = a2(dstp_off + MP * NP * 4); // f16 activation scratch (for MPS-f16 path)
    let dst16_off = a2(xp16_off + MP * KP * 2);
    let ptot = a2(dst16_off + MP * NP * 2);
    let pbuf = device.new_buffer(
        ptot as u64,
        rlx_metal::mtl::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let base = pbuf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(xp.as_ptr() as *const u8, base.add(xp_off), xp.len() * 4);
        std::ptr::copy_nonoverlapping(idxp.as_ptr(), base.add(idxp_off), idxp.len());
        std::ptr::copy_nonoverlapping(
            cb_f32.as_ptr() as *const u8,
            base.add(cbp_off),
            cb_f32.len() * 4,
        );
    }
    let recon_dispatch = |enc: &rlx_metal::mtl::ComputeCommandEncoderRef,
                          pipe: &rlx_metal::mtl::ComputePipelineState,
                          w_off: usize| {
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&pbuf), 0);
        for (i, v) in [idxp_off as u64, cbp_off as u64, w_off as u64]
            .iter()
            .enumerate()
        {
            enc.set_bytes(1 + i as u64, 8, v as *const u64 as *const _);
        }
        for (i, v) in [KP as u32, NP as u32, D as u32].iter().enumerate() {
            enc.set_bytes(4 + i as u64, 4, v as *const u32 as *const _);
        }
        enc.dispatch_threads(
            MTLSize {
                width: nbp as u64,
                height: NP as u64,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 8,
                depth: 1,
            },
        );
    };
    // CPU reference for prefill correctness (W in [k,n] layout → y = x·W).
    let mut wantp = vec![0f32; MP * NP];
    for r in 0..MP {
        for j in 0..NP {
            let mut acc = 0f32;
            for b in 0..nbp {
                let code = idxp[j * nbp + b] as usize;
                for t in 0..D {
                    acc += xp[r * KP + b * D + t] * cb_f32[code * D + t];
                }
            }
            wantp[r * NP + j] = acc;
        }
    }
    let relp = |o: &[f32]| {
        let s = wantp.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        o.iter()
            .zip(&wantp)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / s
    };
    let read_dst = || unsafe {
        let p = (pbuf.contents() as *const u8).add(dstp_off) as *const f32;
        (0..MP * NP).map(|i| *p.add(i)).collect::<Vec<f32>>()
    };

    let cast = |enc: &rlx_metal::mtl::ComputeCommandEncoderRef,
                pipe: &rlx_metal::mtl::ComputePipelineState,
                s: usize,
                d: usize,
                n: usize| {
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&pbuf), 0);
        for (i, v) in [s as u64, d as u64].iter().enumerate() {
            enc.set_bytes(1 + i as u64, 8, v as *const u64 as *const _);
        }
        let nn = n as u32;
        enc.set_bytes(3, 4, &nn as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    };
    // 0 = f32 recon→MPS-f32; 1 = f16 recon→mixed sgemm_f16w (no casts); 2 = MPS-f16 + x/dst casts.
    let prefill = |mode: u32| -> (f64, Vec<f32>) {
        let once = || {
            let cbuf = queue.new_command_buffer();
            let enc = cbuf.new_compute_command_encoder();
            match mode {
                0 => {
                    recon_dispatch(enc, &p_rf32, w32_off);
                    enc.end_encoding();
                    rlx_metal::mps_blas::encode_mps_sgemm(
                        cbuf, &pbuf, xp_off, w32_off, dstp_off, MP, KP, NP,
                    );
                }
                1 => {
                    recon_dispatch(enc, &p_rf16, w16_off);
                    rlx_metal::blas::metal_sgemm_f16w(
                        enc, &pbuf, xp_off, w16_off, dstp_off, MP, KP, NP,
                    );
                    enc.end_encoding();
                }
                _ => {
                    cast(enc, &p_cf, xp_off, xp16_off, MP * KP); // x → f16
                    recon_dispatch(enc, &p_rf16, w16_off);
                    enc.end_encoding();
                    rlx_metal::mps_blas::encode_mps_hgemm(
                        cbuf, &pbuf, xp16_off, w16_off, dst16_off, MP, KP, NP,
                    );
                    let e2 = cbuf.new_compute_command_encoder();
                    cast(e2, &p_cb2, dst16_off, dstp_off, MP * NP);
                    e2.end_encoding();
                }
            }
            cbuf.commit();
            cbuf.wait_until_completed();
        };
        for _ in 0..WARMUP {
            once();
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                once();
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        (best, read_dst())
    };
    let (base_p, o_bp) = prefill(0);
    let (l1w_p, o_l1w) = prefill(1);
    let (l1m_p, o_l1m) = prefill(2);
    println!("\n=== PREFILL (M={MP}, K={KP}, N={NP}) — recon→GEMM weight roundtrip ===\n");
    println!(
        "{:>28} {:>10} {:>9} {:>9} {:>8}",
        "variant", "scratch", "relerr", "ms", "speedup"
    );
    println!("{}", "-".repeat(68));
    println!(
        "{:>28} {:>7.1} MB {:>9.1e} {:>9.3} {:>7.2}x",
        "base recon f32→MPS f32",
        (KP * NP * 4) as f64 / 1048576.0,
        relp(&o_bp),
        base_p,
        1.0
    );
    println!(
        "{:>28} {:>7.1} MB {:>9.1e} {:>9.3} {:>7.2}x",
        "recon f16→sgemm_f16w (no cast)",
        (KP * NP * 2) as f64 / 1048576.0,
        relp(&o_l1w),
        l1w_p,
        base_p / l1w_p
    );
    println!(
        "{:>28} {:>7.1} MB {:>9.1e} {:>9.3} {:>7.2}x",
        "L1 recon f16→MPS-f16 (+casts)",
        (KP * NP * 2) as f64 / 1048576.0,
        relp(&o_l1m),
        l1m_p,
        base_p / l1m_p
    );
    let l1_p = l1m_p;

    // ======================= L3: double-buffer the fused tiled kernel =======================
    // f16 64×64/2×2 register-blocked, reconstruct B on-chip. `_sb` = single-buffer
    // (synchronous load + barrier, like production); `_db` = double-buffer (prefetch
    // next K-panel into a ping-pong buffer, then compute — hide the load stall WITHOUT
    // the async-copy intrinsic MPS uses). No dense weight in DRAM (reads only indices).
    let tiled_src = r#"
#include <metal_stdlib>
using namespace metal;
#define THEAD \
    device float* arena [[buffer(0)]], constant ulong& x_off [[buffer(1)]], \
    constant ulong& idx_off [[buffer(2)]], constant ulong& cb_off [[buffer(3)]], \
    constant ulong& dst_off [[buffer(4)]], constant uint& k_dim [[buffer(5)]], \
    constant uint& n_dim [[buffer(6)]], constant uint& d [[buffer(7)]], \
    constant uint& m_dim [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], \
    uint sgid [[simdgroup_index_in_threadgroup]], uint slid [[thread_index_in_simdgroup]]
#define LOAD_TILES(AT, BT) { \
    for (uint i=0;i<4;++i){ uint id=i*512+lin; uint ar=id/32,ac=id%32; uint r=trb+ar,kc=kk+ac; \
        AT[id]=(r<m_dim&&kc<k_dim)?(half)xb[(ulong)r*k_dim+kc]:(half)0.0h; } \
    for (uint i=0;i<4;++i){ uint id=i*512+lin; uint br=id/64,bc=id%64; uint kr=kk+br,j=tcb+bc; half w=(half)0.0h; \
        if(j<n_dim&&kr<k_dim){uint bl=kr/d,t=kr%d; uint co=uint(ib[(ulong)j*nb+bl]); w=(half)cb[(ulong)co*d+t];} \
        BT[id]=w; } }
#define MMA(AT, BT) for (uint kin=0;kin<32;kin+=8){ \
    simdgroup_half8x8 a0,a1,b0,b1; \
    simdgroup_load(a0,&AT[(sgr*16+0)*32+kin],32); simdgroup_load(a1,&AT[(sgr*16+8)*32+kin],32); \
    simdgroup_load(b0,&BT[kin*64+sgc*16+0],64);   simdgroup_load(b1,&BT[kin*64+sgc*16+8],64); \
    simdgroup_multiply_accumulate(c00,a0,b0,c00); simdgroup_multiply_accumulate(c01,a0,b1,c01); \
    simdgroup_multiply_accumulate(c10,a1,b0,c10); simdgroup_multiply_accumulate(c11,a1,b1,c11); }
#define STORE \
    threadgroup float st[16*64]; device float* ds=(device float*)((device char*)arena+dst_off); \
    { simdgroup_store(c00,&st[sgid*64],8); threadgroup_barrier(mem_flags::mem_threadgroup); uint rb=trb+sgr*16,cbse=tcb+sgc*16; \
      for(uint e=slid;e<64;e+=32){uint rr=e/8,cc=e%8; if(rb+rr<m_dim&&cbse+cc<n_dim) ds[(ulong)(rb+rr)*n_dim+cbse+cc]=st[sgid*64+rr*8+cc];} threadgroup_barrier(mem_flags::mem_threadgroup);} \
    { simdgroup_store(c01,&st[sgid*64],8); threadgroup_barrier(mem_flags::mem_threadgroup); uint rb=trb+sgr*16,cbse=tcb+sgc*16+8; \
      for(uint e=slid;e<64;e+=32){uint rr=e/8,cc=e%8; if(rb+rr<m_dim&&cbse+cc<n_dim) ds[(ulong)(rb+rr)*n_dim+cbse+cc]=st[sgid*64+rr*8+cc];} threadgroup_barrier(mem_flags::mem_threadgroup);} \
    { simdgroup_store(c10,&st[sgid*64],8); threadgroup_barrier(mem_flags::mem_threadgroup); uint rb=trb+sgr*16+8,cbse=tcb+sgc*16; \
      for(uint e=slid;e<64;e+=32){uint rr=e/8,cc=e%8; if(rb+rr<m_dim&&cbse+cc<n_dim) ds[(ulong)(rb+rr)*n_dim+cbse+cc]=st[sgid*64+rr*8+cc];} threadgroup_barrier(mem_flags::mem_threadgroup);} \
    { simdgroup_store(c11,&st[sgid*64],8); threadgroup_barrier(mem_flags::mem_threadgroup); uint rb=trb+sgr*16+8,cbse=tcb+sgc*16+8; \
      for(uint e=slid;e<64;e+=32){uint rr=e/8,cc=e%8; if(rb+rr<m_dim&&cbse+cc<n_dim) ds[(ulong)(rb+rr)*n_dim+cbse+cc]=st[sgid*64+rr*8+cc];} threadgroup_barrier(mem_flags::mem_threadgroup);}
#define PRE \
    uint sgr=sgid/4,sgc=sgid%4; uint trb=tgid.y*64,tcb=tgid.x*64; uint nb=k_dim/d; uint lin=sgid*32+slid; \
    device const float* xb=(device const float*)((device const char*)arena+x_off); \
    device const uchar* ib=(device const uchar*)arena+idx_off; \
    device const float* cb=(device const float*)((device const char*)arena+cb_off); \
    simdgroup_float8x8 c00=simdgroup_float8x8(0.0f),c01=simdgroup_float8x8(0.0f),c10=simdgroup_float8x8(0.0f),c11=simdgroup_float8x8(0.0f);

kernel void tiled_sb(THEAD) {
    PRE
    threadgroup half A[64*32]; threadgroup half B[32*64];
    for (uint kk=0; kk<k_dim; kk+=32) {
        LOAD_TILES(A, B)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MMA(A, B)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    STORE
}
kernel void tiled_db(THEAD) {
    PRE
    threadgroup half A[2][64*32]; threadgroup half B[2][32*64];
    uint kk = 0;
    { LOAD_TILES(A[0], B[0]) }              // prologue: prefetch first panel
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint kt = 0;
    for (kk = 0; kk < k_dim; kk += 32, ++kt) {
        uint cur = kt & 1u, nxt = (kt + 1u) & 1u;
        uint kknext = kk + 32;
        if (kknext < k_dim) {              // prefetch NEXT panel into the other buffer
            uint kk = kknext;              // shadow for LOAD_TILES
            LOAD_TILES(A[nxt], B[nxt])
        }
        MMA(A[cur], B[cur])                // compute CURRENT while the prefetch is in flight
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    STORE
}
"#;
    let tlib = device
        .new_library_with_source(tiled_src, &rlx_metal::mtl::CompileOptions::new())
        .expect("tiled MSL");
    let tpipe = |n: &str| {
        let f = tlib.get_function(n, None).unwrap();
        device.new_compute_pipeline_state_with_function(&f).unwrap()
    };
    let (p_sb, p_db) = (tpipe("tiled_sb"), tpipe("tiled_db"));
    let run_tiled = |pipe: &rlx_metal::mtl::ComputePipelineState| -> (f64, Vec<f32>) {
        let batch = |count: usize| {
            let cbuf = queue.new_command_buffer();
            let enc = cbuf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&pbuf), 0);
            for (i, v) in [
                xp_off as u64,
                idxp_off as u64,
                cbp_off as u64,
                dstp_off as u64,
            ]
            .iter()
            .enumerate()
            {
                enc.set_bytes(1 + i as u64, 8, v as *const u64 as *const _);
            }
            for (i, v) in [KP as u32, NP as u32, D as u32, MP as u32]
                .iter()
                .enumerate()
            {
                enc.set_bytes(5 + i as u64, 4, v as *const u32 as *const _);
            }
            for _ in 0..count {
                enc.dispatch_thread_groups(
                    MTLSize {
                        width: (NP as u64).div_ceil(64),
                        height: (MP as u64).div_ceil(64),
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 16,
                        depth: 1,
                    },
                );
            }
            enc.end_encoding();
            cbuf.commit();
            cbuf.wait_until_completed();
        };
        batch(WARMUP);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            batch(ITERS);
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        let out = unsafe {
            let p = (pbuf.contents() as *const u8).add(dstp_off) as *const f32;
            (0..MP * NP).map(|i| *p.add(i)).collect()
        };
        (best, out)
    };
    // CPU reference for the tiled prefill.
    let mut wantp = vec![0f32; MP * NP];
    for r in 0..MP {
        for j in 0..NP {
            let mut acc = 0f32;
            for b in 0..nbp {
                let code = idxp[j * nbp + b] as usize;
                for t in 0..D {
                    acc += xp[r * KP + b * D + t] * cb_f32[code * D + t];
                }
            }
            wantp[r * NP + j] = acc;
        }
    }
    let relp = |o: &[f32]| {
        let s = wantp.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        o.iter()
            .zip(&wantp)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / s
    };
    let (sb_ms, o_sb) = run_tiled(&p_sb);
    let (db_ms, o_db) = run_tiled(&p_db);
    println!("\n=== L3: fused tiled f16 (reads only indices, no dense-weight DRAM) ===\n");
    println!(
        "{:>22} {:>9} {:>9} {:>8}",
        "variant", "relerr", "ms", "vs sb"
    );
    println!("{}", "-".repeat(52));
    println!(
        "{:>22} {:>9.1e} {:>9.3} {:>7.2}x",
        "single-buffer",
        relp(&o_sb),
        sb_ms,
        1.0
    );
    println!(
        "{:>22} {:>9.1e} {:>9.3} {:>7.2}x",
        "L3 double-buffer",
        relp(&o_db),
        db_ms,
        sb_ms / db_ms
    );
    println!(
        "{:>22} {:>20} {:>7.2}x",
        "(recon f16→MPS L1)",
        format!("{l1_p:.3} ms"),
        sb_ms.min(db_ms) / l1_p
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS only");
}
