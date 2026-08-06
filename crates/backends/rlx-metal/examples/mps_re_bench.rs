// RLX — MPS vs our-kernel head-to-head at qwen3 shapes (RE / ceiling probe).
//
// Answers: at each decode (m=1) and prefill (m>1) projection shape, how does
// Apple's MPSMatrixMultiplication compare to our gemv_f16w_splitk /
// sgemm_simd_padded_f16w on latency, effective bandwidth, and numerical
// accuracy? A large MPS win localizes where to change our kernels (e.g. adopt
// simdgroup_matrix MMA for prefill); parity confirms our kernel is at the
// hardware ceiling (bandwidth-bound) and the gap is elsewhere (op count).
//
//   cargo run --release -p rlx-metal --example mps_re_bench
#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use half::f16;
    use metal::MTLResourceOptions;
    use rlx_metal::blas::metal_sgemm_f16w_bufs;
    use rlx_metal::device::metal_device;
    use rlx_metal::mps_blas::{encode_mps_hgemm, mps_supports_matmul};

    if !mps_supports_matmul() {
        eprintln!("MPS not available");
        return;
    }
    let dev = metal_device().expect("metal device");
    let _ = rlx_metal::kernels::kernels(); // warm the pipeline cache

    // GPU-window seconds from a completed command buffer (metal-rs doesn't wrap
    // GPUStartTime/EndTime).
    unsafe fn gpu_secs(cb: &metal::CommandBufferRef) -> f64 {
        use objc::{msg_send, runtime::Object, sel, sel_impl};
        let obj = cb as *const metal::CommandBufferRef as *mut Object;
        let start: f64 = msg_send![obj, GPUStartTime];
        let end: f64 = msg_send![obj, GPUEndTime];
        (end - start).max(0.0)
    }

    // splitmix64 → deterministic pseudo-random unit.
    let rnd = |mut z: u64| -> f32 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };

    // qwen3-0.6B projection shapes: (label, m, k, n).
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("decode o_proj   ", 1, 1024, 1024),
        ("decode q_proj   ", 1, 1024, 2048),
        ("decode k/v_proj ", 1, 1024, 1024),
        ("decode gate/up  ", 1, 1024, 3072),
        ("decode down_proj", 1, 3072, 1024),
        ("decode lm_head  ", 1, 1024, 151936),
        ("prefill140 qkv ", 140, 1024, 4096),
        ("prefill140 gateup", 140, 1024, 3072),
        ("prefill16 gateup", 16, 1024, 3072),
        ("prefill64 gateup", 64, 1024, 3072),
        ("prefill256 gateup", 256, 1024, 3072),
    ];

    // One shared arena big enough for the largest shape. Regions (per shape):
    // A_f32 | B_f16 | C_f32 | A_f16 | C_f16.
    let cap_bytes = 512usize * 1024 * 1024;
    let arena = dev
        .device
        .new_buffer(cap_bytes as u64, MTLResourceOptions::StorageModeShared);
    let base = arena.contents() as *mut u8;

    // Ramp the GPU clock out of its low-power idle state before timing, else the
    // first shape reads slow (DVFS). Run ~sustained work for a bit.
    {
        let al = |x: usize| (x + 255) & !255;
        let (m, k, n) = (256usize, 1024usize, 3072usize);
        let (ao, bo, co) = (0usize, al(m * k * 4), al(al(m * k * 4) + k * n * 2));
        unsafe {
            let a = base.add(ao) as *mut f32;
            for i in 0..m * k {
                *a.add(i) = 0.01;
            }
            let b = base.add(bo) as *mut f16;
            for i in 0..k * n {
                *b.add(i) = f16::from_f32(0.01);
            }
        }
        for _ in 0..80 {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            metal_sgemm_f16w_bufs(enc, &arena, ao, &arena, bo, &arena, co, m, k, n);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
    }

    println!(
        "{:<18} {:>10} {:>10} {:>7}   {:>9} {:>9}   {:>7}",
        "shape (m×k×n)", "ours µs", "MPS µs", "ratio", "ours GB/s", "MPS GB/s", "maxdiff"
    );
    println!("{}", "-".repeat(88));

    const R: usize = 100;
    for &(label, m, k, n) in shapes {
        // Region offsets (bytes), 256-aligned.
        let al = |x: usize| (x + 255) & !255;
        let a_f32_off = 0usize;
        let b_f16_off = al(a_f32_off + m * k * 4);
        let c_f32_off = al(b_f16_off + k * n * 2);
        let a_f16_off = al(c_f32_off + m * n * 4);
        let c_f16_off = al(a_f16_off + m * k * 2);
        let total = c_f16_off + m * n * 2;
        if total > cap_bytes {
            println!("{label}: skip (needs {} MB)", total / (1024 * 1024));
            continue;
        }
        // Fill A (f32 + f16) and B (f16 shared).
        unsafe {
            let a32 = base.add(a_f32_off) as *mut f32;
            let a16 = base.add(a_f16_off) as *mut f16;
            for i in 0..m * k {
                let v = rnd(i as u64 * 2654435761);
                *a32.add(i) = v;
                *a16.add(i) = f16::from_f32(v);
            }
            let b16 = base.add(b_f16_off) as *mut f16;
            for i in 0..k * n {
                *b16.add(i) = f16::from_f32(rnd(0x1234 ^ i as u64 * 40503));
            }
        }

        // Time OURS (metal_sgemm_f16w_bufs: gemv_f16w_splitk @ m=1, else padded).
        let mut ours_us = f64::INFINITY;
        for _ in 0..R {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            metal_sgemm_f16w_bufs(
                enc, &arena, a_f32_off, &arena, b_f16_off, &arena, c_f32_off, m, k, n,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            ours_us = ours_us.min(unsafe { gpu_secs(cb) } * 1e6);
        }
        // Time MPS (encode_mps_hgemm: F16 A×B→C).
        let mut mps_us = f64::INFINITY;
        for _ in 0..R {
            let cb = dev.queue.new_command_buffer();
            encode_mps_hgemm(cb, &arena, a_f16_off, b_f16_off, c_f16_off, m, k, n);
            cb.commit();
            cb.wait_until_completed();
            mps_us = mps_us.min(unsafe { gpu_secs(cb) } * 1e6);
        }

        // Accuracy: max abs diff of the two C's (ours f32 vs MPS f16→f32).
        let mut maxdiff = 0f32;
        unsafe {
            let c32 = base.add(c_f32_off) as *const f32;
            let c16 = base.add(c_f16_off) as *const f16;
            for i in 0..m * n {
                let d = (*c32.add(i) - (*c16.add(i)).to_f32()).abs();
                if d > maxdiff {
                    maxdiff = d;
                }
            }
        }

        // Weight-read bytes (F16) dominate decode; report as effective BW.
        let wbytes = (k * n * 2) as f64;
        let ours_gbs = wbytes / (ours_us * 1e3);
        let mps_gbs = wbytes / (mps_us * 1e3);
        println!(
            "{label} {m:>4}×{k}×{n:<6} {ours_us:>10.2} {mps_us:>10.2} {:>6.2}x   {ours_gbs:>9.1} {mps_gbs:>9.1}   {maxdiff:>7.3}",
            mps_us / ours_us,
        );
    }
    println!(
        "\nnote: ratio>1 = ours faster; GB/s vs M4 Pro ~273 peak; maxdiff = ours(f32acc) − MPS(f16)."
    );

    // ── Quantized GEMV sweep (decode m=1) ───────────────────────────────────
    // Time each quant format's GEMV at decode shapes, reporting effective
    // weight-read bandwidth. A format is bandwidth-bound (good) if GB/s ≈ the
    // F16 GEMV's; ALU/dequant-bound (bad) if far below despite reading fewer
    // bytes. Only Q4_K has a SIMD-cooperative kernel (`_sg`, 32 threads/2 rows);
    // the rest are one-thread-per-row (`gid`), so we expect them slow.
    let kk = rlx_metal::kernels::kernels();
    // (label, block_elems, block_bytes, sg_mode)
    // sg_mode: 0 = one-thread-per-row (gid); 1 = q4k SIMD (4 sg/tg × 2 rows);
    //          2 = q8_0 SIMD (2 sg/tg × 4 rows); 3 = q6k SIMD (4 sg/tg × 1 row).
    let quants: &[(&str, usize, usize, u8)] = &[
        ("Q4_K(_sg)", 256, 144, 1),
        ("Q4_0    ", 32, 18, 0),
        ("Q4_0(_sg)", 32, 18, 2),
        ("Q8_0    ", 32, 34, 0),
        ("Q8_0(_sg)", 32, 34, 2),
        ("Q6_K    ", 256, 210, 0),
        ("Q6_K(_sg)", 256, 210, 3),
        ("Q4_1    ", 32, 20, 0),
        ("Q4_1(_sg)", 32, 20, 2),
        ("Q3_K    ", 256, 110, 0),
        ("Q3_K(_sg)", 256, 110, 1),
        // Other 3-bit formats (codebook/LUT-based; one-thread-per-row only — no
        // _sg variant. Binds the IQ grid at buffer 6.)
        ("IQ3XXS  ", 256, 98, 0),
        ("IQ3XXS(_sg)", 256, 98, 3),
        ("IQ3S    ", 256, 110, 0),
        ("IQ3S(_sg)", 256, 110, 3),
    ];
    let qshapes: &[(usize, usize)] = &[(1024, 1024), (1024, 3072), (3072, 1024)];
    println!(
        "\n{:<10} {:>10} {:>8} {:>8} {:>10}   (decode m=1, F16 GEMV baseline in parens)",
        "quant", "k×n", "µs", "GB/s", "vs F16"
    );
    println!("{}", "-".repeat(72));
    for &(k, n) in qshapes {
        // F16 GEMV baseline for this shape (from a fresh arena slice).
        let al = |x: usize| (x + 255) & !255;
        let x_off = 0usize;
        let w16_off = al(x_off + k * 4);
        let dst_off = al(w16_off + k * n * 2);
        let wq_off = al(dst_off + n * 4);
        unsafe {
            let x = base.add(x_off) as *mut f32;
            for i in 0..k {
                *x.add(i) = rnd(i as u64 * 40503) * 0.1;
            }
        }
        // F16 baseline.
        let mut f16_us = f64::INFINITY;
        for _ in 0..R {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            metal_sgemm_f16w_bufs(
                enc, &arena, x_off, &arena, w16_off, &arena, dst_off, 1, k, n,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            f16_us = f16_us.min(unsafe { gpu_secs(cb) } * 1e6);
        }
        for &(label, be, bb, sg_mode) in quants {
            let wbytes = n * (k / be) * bb;
            if wq_off + wbytes > cap_bytes {
                continue;
            }
            let pipe = match label.trim() {
                "Q4_K(_sg)" => &kk.q4k_mv_f32_sg,
                "Q4_0" => &kk.q4_0_mv_f32,
                "Q4_0(_sg)" => &kk.q4_0_mv_f32_sg,
                "Q8_0" => &kk.q8_0_mv_f32,
                "Q8_0(_sg)" => &kk.q8_0_mv_f32_sg,
                "Q6_K" => &kk.q6k_mv_f32,
                "Q6_K(_sg)" => &kk.q6k_mv_f32_sg,
                "Q4_1" => &kk.q4_1_mv_f32,
                "Q4_1(_sg)" => &kk.q4_1_mv_f32_sg,
                "Q3_K" => &kk.q3k_mv_f32,
                "Q3_K(_sg)" => &kk.q3k_mv_f32_sg,
                "IQ3XXS" => &kk.iq3_xxs_mv_f32,
                "IQ3XXS(_sg)" => &kk.iq3_xxs_mv_f32_sg,
                "IQ3S" => &kk.iq3_s_mv_f32,
                "IQ3S(_sg)" => &kk.iq3_s_mv_f32_sg,
                _ => continue,
            };
            let k_u = k as u32;
            let n_u = n as u32;
            let xo = x_off as u64;
            let wo = wq_off as u64;
            let do_ = dst_off as u64;
            let mut us = f64::INFINITY;
            for _ in 0..R {
                let cb = dev.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(&arena), 0);
                enc.set_bytes(1, 8, &xo as *const u64 as *const _);
                enc.set_bytes(2, 8, &wo as *const u64 as *const _);
                enc.set_bytes(3, 8, &do_ as *const u64 as *const _);
                enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
                enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
                // IQ kernels read the codebook grid at buffer 6 (harmless for others).
                enc.set_buffer(6, Some(kk.iq_grid_buffer()), 0);
                match sg_mode {
                    1 => {
                        // q4k: 4 simdgroups/tg × 2 rows (encode_q4k_mv_f32_sg).
                        let tgs = (n as u64).div_ceil(2).div_ceil(4);
                        enc.dispatch_threads(
                            metal::MTLSize {
                                width: tgs * 4 * 32,
                                height: 1,
                                depth: 1,
                            },
                            metal::MTLSize {
                                width: 4 * 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    2 => {
                        // q8_0: 2 simdgroups/tg × 4 rows (encode_q8_0_mv_f32_sg).
                        let tgs = (n as u64).div_ceil(4).div_ceil(2);
                        enc.dispatch_threads(
                            metal::MTLSize {
                                width: tgs * 2 * 32,
                                height: 1,
                                depth: 1,
                            },
                            metal::MTLSize {
                                width: 2 * 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    3 => {
                        // q6k: 4 simdgroups/tg × 1 row (encode_q6k_mv_f32_sg).
                        let tgs = (n as u64).div_ceil(4);
                        enc.dispatch_threads(
                            metal::MTLSize {
                                width: tgs * 4 * 32,
                                height: 1,
                                depth: 1,
                            },
                            metal::MTLSize {
                                width: 4 * 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    _ => {
                        // one thread per output row.
                        enc.dispatch_threads(
                            metal::MTLSize {
                                width: n as u64,
                                height: 1,
                                depth: 1,
                            },
                            metal::MTLSize {
                                width: 64.min(n as u64),
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                }
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                us = us.min(unsafe { gpu_secs(cb) } * 1e6);
            }
            let gbs = wbytes as f64 / (us * 1e3);
            println!(
                "{label} {k}×{n:<5} {us:>8.2} {gbs:>8.1} {:>9.2}x",
                f16_us / us,
            );
        }
    }
    println!(
        "note: vs F16 >1 = quant GEMV faster than the F16 GEMV; GB/s = weight bytes read / time."
    );
}
