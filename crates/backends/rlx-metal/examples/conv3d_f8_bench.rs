// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Does f8 storage make the 3-D convolution faster?
//!
//! Apple GPUs have no f8 arithmetic, so f8 can only pay as memory traffic:
//! four values a 32-bit word, a quarter of f32's bytes and half of f16's,
//! decoded to half in registers as the tile is gathered. This times
//! `conv3d_gemm_c_h` (half in, half out) against the same kernel reading
//! its input as f8 e4m3, and its input and filter as f8, on SegResNet's
//! encoder shapes (3^3, stride 1, C -> C), by the GPU's own timestamps. The
//! f8 kernels must equal the half kernel fed the f8 values decoded to half:
//! the decode is exact, the arithmetic identical.
//!
//! `cargo run -p rlx-metal --release --example conv3d_f8_bench`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("conv3d_f8_bench: Metal only");
}

#[cfg(target_os = "macos")]
fn main() {
    use half::f16;
    use rlx_metal::device::metal_device;
    use rlx_metal::mtl::{CompileOptions, MTLSize};

    // The f8 variants, instantiated from the same template as the half kernel.
    const F8_KERNELS: &str = r#"
#define CONV3D_F8_ARGS(TI, TW) \
    device const TI* src [[buffer(0)]], device const TW* wt [[buffer(1)]], \
    device half* dst [[buffer(2)]], constant uint4& a [[buffer(3)]], \
    constant uint4& b [[buffer(4)]], constant uint4& c [[buffer(5)]], \
    constant uint4& d [[buffer(6)]], constant uint4& e [[buffer(7)]], \
    constant uint4& f [[buffer(8)]], device const half* bias [[buffer(9)]], \
    constant float& leaky_alpha [[buffer(10)]], device const TI* src2 [[buffer(11)]], \
    constant uint& c_split [[buffer(12)]], constant uint4& up [[buffer(13)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint sgid [[simdgroup_index_in_threadgroup]], uint slid [[thread_index_in_simdgroup]]
kernel void conv3d_gemm_c_f8in(CONV3D_F8_ARGS(uchar, half)) {
    threadgroup float Cs[64u * 33u];
    threadgroup half* As = reinterpret_cast<threadgroup half*>(Cs);
    conv3d_gemm_c_impl<half, 64u, 32u, 1u, uchar, half>(src, wt, dst, a, b, c, d, e, f, bias,
        leaky_alpha, src2, c_split, up, As, As + 64u * 33u, Cs, tgid, sgid, slid);
}
kernel void conv3d_gemm_c_f8inw(CONV3D_F8_ARGS(uchar, uchar)) {
    threadgroup float Cs[64u * 33u];
    threadgroup half* As = reinterpret_cast<threadgroup half*>(Cs);
    conv3d_gemm_c_impl<half, 64u, 32u, 1u, uchar, uchar>(src, wt, dst, a, b, c, d, e, f, bias,
        leaky_alpha, src2, c_split, up, As, As + 64u * 33u, Cs, tgid, sgid, slid);
}
"#;

    // e4m3: the value of each positive code, 0x00..=0x7e (0x7f is NaN).
    let decode = |b: u8| -> f32 {
        let (e, m) = ((b >> 3) & 0xf, b & 7);
        let v = if e == 0 {
            m as f32 / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
        };
        if b & 0x80 != 0 { -v } else { v }
    };
    let positive: Vec<f32> = (0..=0x7eu8).map(decode).collect();
    // Nearest code, saturating at 448.
    let encode = |v: f32| -> u8 {
        let a = v.abs();
        let i = positive.partition_point(|&p| p < a).min(positive.len() - 1);
        let i = if i > 0 && (a - positive[i - 1]) <= (positive[i] - a) {
            i - 1
        } else {
            i
        };
        i as u8 | if v < 0.0 { 0x80 } else { 0 }
    };

    let dev = metal_device().expect("no Metal device");
    let src = format!("{}\n{}", rlx_metal::kernels::msl_source(), F8_KERNELS);
    let lib = dev
        .device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile");
    let pso = |name: &str| {
        dev.device
            .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
            .unwrap()
    };
    let (half_k, f8in_k, f8inw_k) = (
        pso("conv3d_gemm_c_h"),
        pso("conv3d_gemm_c_f8in"),
        pso("conv3d_gemm_c_f8inw"),
    );
    println!("device: {}\n", dev.name);
    println!(
        "{:>16}  {:>10}  {:>16}  {:>16}",
        "C x D^3", "half in", "f8 in", "f8 in + filter"
    );

    let align = |x: usize| (x + 255) & !255;
    for (ch, dd) in [(16usize, 128usize), (32, 64), (64, 32), (128, 16)] {
        let vox = dd * dd * dd;
        let (nx, nw) = (ch * vox, ch * ch * 27);
        let mut seed = 1u32;
        let mut rnd = |scale: f32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 9) as f32 / 4_194_304.0 - 1.0) * scale
        };
        let x: Vec<f32> = (0..nx).map(|_| rnd(2.0)).collect();
        let w: Vec<f32> = (0..nw)
            .map(|_| rnd((3.0 / (ch * 27) as f32).sqrt()))
            .collect();
        let (x8, w8): (Vec<u8>, Vec<u8>) = (
            x.iter().map(|&v| encode(v)).collect(),
            w.iter().map(|&v| encode(v)).collect(),
        );
        // The half kernel's inputs: the f8 values, decoded, so outputs compare exactly.
        let xh: Vec<f16> = x8.iter().map(|&b| f16::from_f32(decode(b))).collect();
        let wh: Vec<f16> = w.iter().map(|&v| f16::from_f32(v)).collect();
        let w8h: Vec<f16> = w8.iter().map(|&b| f16::from_f32(decode(b))).collect();

        let o_xh = 0;
        let o_x8 = align(o_xh + nx * 2);
        let o_wh = align(o_x8 + nx);
        let o_w8h = align(o_wh + nw * 2);
        let o_w8 = align(o_w8h + nw * 2);
        let o_y = align(o_w8 + nw);
        let o_bias = align(o_y + nx * 2);
        let buf = dev.alloc_shared(o_bias + ch * 2 + 256);
        unsafe {
            let base = buf.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(xh.as_ptr() as *const u8, base.add(o_xh), nx * 2);
            std::ptr::copy_nonoverlapping(x8.as_ptr(), base.add(o_x8), nx);
            std::ptr::copy_nonoverlapping(wh.as_ptr() as *const u8, base.add(o_wh), nw * 2);
            std::ptr::copy_nonoverlapping(w8h.as_ptr() as *const u8, base.add(o_w8h), nw * 2);
            std::ptr::copy_nonoverlapping(w8.as_ptr(), base.add(o_w8), nw);
        }
        let d3 = dd as u32;
        let c = ch as u32;
        let params: [[u32; 4]; 6] = [
            [1, c, d3, d3],
            [d3, c, d3, d3],
            [d3, 3, 3, 3],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ];
        let up: [u32; 4] = [1, 1, 1, 0];
        let alpha = 0f32;
        let tiles_m = (vox as u64).div_ceil(64);
        let tiles_n = (ch as u64).div_ceil(32);
        let encode_conv = |cb: &rlx_metal::mtl::CommandBufferRef,
                           k: &rlx_metal::mtl::ComputePipelineState,
                           xs: usize,
                           ws: usize,
                           reps: usize| {
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(k);
            e.set_buffer(0, Some(&buf), xs as u64);
            e.set_buffer(1, Some(&buf), ws as u64);
            e.set_buffer(2, Some(&buf), o_y as u64);
            for (i, p) in params.iter().enumerate() {
                e.set_bytes(3 + i as u64, 16, p.as_ptr() as *const _);
            }
            e.set_buffer(9, Some(&buf), o_bias as u64);
            e.set_bytes(10, 4, &alpha as *const f32 as *const _);
            e.set_buffer(11, Some(&buf), xs as u64);
            e.set_bytes(12, 4, &c as *const u32 as *const _);
            e.set_bytes(13, 16, up.as_ptr() as *const _);
            for _ in 0..reps {
                e.dispatch_threads(
                    MTLSize {
                        width: tiles_m * 128,
                        height: tiles_n,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            e.end_encoding();
        };
        // Median of 5 buffers of 10 dispatches, ms a dispatch, by GPU timestamps.
        let time =
            |k: &rlx_metal::mtl::ComputePipelineState, xs: usize, ws: usize| -> (f64, Vec<u16>) {
                let mut ms: Vec<f64> = (0..6)
                    .map(|_| {
                        let cb = dev.queue.new_command_buffer();
                        encode_conv(cb, k, xs, ws, 10);
                        cb.commit();
                        cb.wait_until_completed();
                        (cb.gpu_end_time() - cb.gpu_start_time()) * 1e3 / 10.0
                    })
                    .skip(1)
                    .collect();
                ms.sort_by(f64::total_cmp);
                let cb = dev.queue.new_command_buffer();
                encode_conv(cb, k, xs, ws, 1);
                cb.commit();
                cb.wait_until_completed();
                let y = unsafe {
                    std::slice::from_raw_parts(buf.contents().add(o_y) as *const u16, nx).to_vec()
                };
                (ms[2], y)
            };
        let (t_h, y_h) = time(&half_k, o_xh, o_wh);
        let (t_8, y_8) = time(&f8in_k, o_x8, o_wh);
        let (t_hw, y_hw) = time(&half_k, o_xh, o_w8h);
        let (t_8w, y_8w) = time(&f8inw_k, o_x8, o_w8);
        assert!(
            y_h == y_8,
            "{ch}x{dd}^3: f8 input differs from half fed the same values"
        );
        assert!(
            y_hw == y_8w,
            "{ch}x{dd}^3: f8 input + filter differs from half fed the same values"
        );
        let _ = t_hw;
        println!(
            "{:>16}  {:>7.2} ms  {:>7.2} ms {:>5.2}x  {:>7.2} ms {:>5.2}x",
            format!("{ch} x {dd}^3"),
            t_h,
            t_8,
            t_h / t_8,
            t_8w,
            t_h / t_8w
        );
    }
}
