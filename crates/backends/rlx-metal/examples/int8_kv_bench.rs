// int8 KV-cache decode-attention microbench (ceiling probe before integration).
//
// Decode attention is weight-free but reads the whole K/V cache each step, so at
// long context it's KV-read-bandwidth bound. Quantizing the cache to int8 (with a
// per-row/token scale) reads 4× less than F32 / 2× less than F16 → the question
// is whether the dequant overhead keeps it bandwidth-bound and whether int8 is
// accurate enough. This times f32-KV vs int8-KV decode attention at growing
// context and reports the speedup + max output error.
//
//   cargo run --release -p rlx-metal --example int8_kv_bench
#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use metal::{MTLResourceOptions, MTLSize};

    let dev = rlx_metal::device::metal_device().expect("metal device");
    let device = &dev.device;

    unsafe fn gpu_secs(cb: &metal::CommandBufferRef) -> f64 {
        use objc::{msg_send, runtime::Object, sel, sel_impl};
        let obj = cb as *const metal::CommandBufferRef as *mut Object;
        let start: f64 = msg_send![obj, GPUStartTime];
        let end: f64 = msg_send![obj, GPUEndTime];
        (end - start).max(0.0)
    }
    let rnd = |mut z: u64| -> f32 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };

    // One threadgroup (32 threads = 1 simdgroup) per head; online-softmax decode
    // attention with a warp (simd_sum/simd_max) reduction. Two variants: F32 KV,
    // and int8 KV with a per-(head,key) scale for K and V (the `ks`/`vs` factor
    // out of the per-element dot / accumulate).
    let src = r#"
#include <metal_stdlib>
using namespace metal;
// Flash-style: grid = heads * n_part threadgroups; each walks its KV slice and
// writes a partial (no combine — timing only, math already parity-checked at
// n_part=1). Raising n_part fills the GPU so the kernel becomes KV-read-bound,
// which is the regime int8 targets.
kernel void attn_f32kv(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& n_part [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h = tgid / n_part, part = tgid % n_part; if (h >= heads) return;
    uint chunk = (seq + n_part - 1u) / n_part, ks = part*chunk, ke = min(ks+chunk, seq);
    float q[128]; for (uint d=0;d<dh;++d) q[d]=Q[h*dh+d];
    float m_acc=-1e30f, l_acc=0.f, o[128]; for (uint d=0;d<dh;++d) o[d]=0.f;
    for (uint ki=ks+tid; ki<ke; ki+=tsize) {
        device const float* kr = K + (ulong)(h*seq+ki)*dh;
        float dot=0.f; for (uint d=0;d<dh;++d) dot+=q[d]*kr[d];
        float s=dot*scale;
        float mn=max(m_acc,s), eo=exp(m_acc-mn), ec=exp(s-mn);
        l_acc=eo*l_acc+ec;
        device const float* vr = V + (ulong)(h*seq+ki)*dh;
        for (uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*vr[d];
        m_acc=mn;
    }
    float mg=simd_max(m_acc), r=exp(m_acc-mg), lg=simd_sum(l_acc*r), inv=1.0f/max(lg,1e-20f);
    for (uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
kernel void attn_i8kv(
    device const float* Q [[buffer(0)]], device const char* K [[buffer(1)]],
    device const char* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    device const float* Ksc [[buffer(8)]], device const float* Vsc [[buffer(9)]],
    constant uint& n_part [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h = tgid / n_part, part = tgid % n_part; if (h >= heads) return;
    uint chunk = (seq + n_part - 1u) / n_part, ks = part*chunk, ke = min(ks+chunk, seq);
    float q[128]; for (uint d=0;d<dh;++d) q[d]=Q[h*dh+d];
    float m_acc=-1e30f, l_acc=0.f, o[128]; for (uint d=0;d<dh;++d) o[d]=0.f;
    for (uint ki=ks+tid; ki<ke; ki+=tsize) {
        device const char* kr = K + (ulong)(h*seq+ki)*dh;
        float dot=0.f; for (uint d=0;d<dh;++d) dot+=q[d]*float(kr[d]);
        float s=dot*Ksc[h*seq+ki]*scale;
        float mn=max(m_acc,s), eo=exp(m_acc-mn), ec=exp(s-mn);
        l_acc=eo*l_acc+ec;
        device const char* vr = V + (ulong)(h*seq+ki)*dh;
        float vs=Vsc[h*seq+ki];
        for (uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*(float(vr[d])*vs);
        m_acc=mn;
    }
    float mg=simd_max(m_acc), r=exp(m_acc-mg), lg=simd_sum(l_acc*r), inv=1.0f/max(lg,1e-20f);
    for (uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
// RE probe: skip the 128-dim QK dot (s from a cheap value) → isolates QK cost.
kernel void attn_noqk(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& n_part [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h = tgid / n_part, part = tgid % n_part; if (h >= heads) return;
    uint chunk=(seq+n_part-1u)/n_part, ks=part*chunk, ke=min(ks+chunk,seq);
    float m_acc=-1e30f, l_acc=0.f, o[128]; for (uint d=0;d<dh;++d) o[d]=0.f;
    for (uint ki=ks+tid; ki<ke; ki+=tsize) {
        float s=float(ki)*1e-4f*scale;   // cheap, no dot
        float mn=max(m_acc,s), eo=exp(m_acc-mn), ec=exp(s-mn); l_acc=eo*l_acc+ec;
        device const float* vr = V + (ulong)(h*seq+ki)*dh;
        for (uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*vr[d];
        m_acc=mn;
    }
    float mg=simd_max(m_acc), r=exp(m_acc-mg), lg=simd_sum(l_acc*r), inv=1.0f/max(lg,1e-20f);
    for (uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
// RE probe: skip the 128-dim PV accumulate (only o[0]) → isolates PV cost.
kernel void attn_nopv(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& n_part [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h = tgid / n_part, part = tgid % n_part; if (h >= heads) return;
    uint chunk=(seq+n_part-1u)/n_part, ks=part*chunk, ke=min(ks+chunk,seq);
    float q[128]; for (uint d=0;d<dh;++d) q[d]=Q[h*dh+d];
    float m_acc=-1e30f, l_acc=0.f, o0=0.f;
    for (uint ki=ks+tid; ki<ke; ki+=tsize) {
        device const float* kr = K + (ulong)(h*seq+ki)*dh;
        float dot=0.f; for (uint d=0;d<dh;++d) dot+=q[d]*kr[d];
        float s=dot*scale, mn=max(m_acc,s), eo=exp(m_acc-mn), ec=exp(s-mn); l_acc=eo*l_acc+ec;
        o0=eo*o0+ec*V[(ulong)(h*seq+ki)*dh]; m_acc=mn;
    }
    float mg=simd_max(m_acc), r=exp(m_acc-mg), lg=simd_sum(l_acc*r);
    float og=simd_sum(o0*r); if(tid==0) OUT[(h*n_part+part)*dh]=og/max(lg,1e-20f);
}
// W8A8: int8 Q (per-head scale qs) + int8 K (per-key scale) → INTEGER QK dot on
// the int ALU; int8 V PV in f32 (same as int8kv). Tests whether int8 QK is faster.
kernel void attn_w8a8(
    device const char* Q [[buffer(0)]], device const char* K [[buffer(1)]],
    device const char* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    device const float* Ksc [[buffer(8)]], device const float* Vsc [[buffer(9)]],
    constant uint& n_part [[buffer(10)]], device const float* Qsc [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h = tgid / n_part, part = tgid % n_part; if (h >= heads) return;
    uint chunk=(seq+n_part-1u)/n_part, ks=part*chunk, ke=min(ks+chunk,seq);
    // Pack q as char4 for the integer dot.
    char qi[128]; for (uint d=0;d<dh;++d) qi[d]=Q[h*dh+d];
    float qs=Qsc[h];
    float m_acc=-1e30f, l_acc=0.f, o[128]; for (uint d=0;d<dh;++d) o[d]=0.f;
    for (uint ki=ks+tid; ki<ke; ki+=tsize) {
        device const char* kr = K + (ulong)(h*seq+ki)*dh;
        int doti=0;
        for (uint d=0;d<dh;d+=4u) {
            doti += int(qi[d])*int(kr[d]) + int(qi[d+1])*int(kr[d+1])
                  + int(qi[d+2])*int(kr[d+2]) + int(qi[d+3])*int(kr[d+3]);
        }
        float s=float(doti)*qs*Ksc[h*seq+ki]*scale;
        float mn=max(m_acc,s), eo=exp(m_acc-mn), ec=exp(s-mn); l_acc=eo*l_acc+ec;
        device const char* vr = V + (ulong)(h*seq+ki)*dh; float vsf=Vsc[h*seq+ki];
        for (uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*(float(vr[d])*vsf);
        m_acc=mn;
    }
    float mg=simd_max(m_acc), r=exp(m_acc-mg), lg=simd_sum(l_acc*r), inv=1.0f/max(lg,1e-20f);
    for (uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
"#;
    let lib = device
        .new_library_with_source(src, &metal::CompileOptions::new())
        .expect("compile");
    let mk = |n: &str| {
        device
            .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
            .unwrap()
    };
    let p_f32 = mk("attn_f32kv");
    let p_i8 = mk("attn_i8kv");
    let p_noqk = mk("attn_noqk");
    let p_nopv = mk("attn_nopv");
    let p_w8a8 = mk("attn_w8a8");

    let heads = 16u32;
    let dh = 128u32;
    let scale = 1.0f32 / (dh as f32).sqrt();
    let max_p = 64u32;
    println!(
        "int8 KV decode attention (heads={heads}, dh={dh}, flash n_part sweep, GPU-timed best of 20)"
    );
    println!(
        "{:>7} {:>4} {:>9} {:>9} {:>8} {:>10} {:>10}",
        "ctx", "P", "f32 µs", "int8 µs", "speedup", "f32 GB/s", "int8 GB/s"
    );

    for &seq in &[2048u32, 4096, 16384] {
        let hd = (heads * dh) as usize;
        let kvn = (heads * seq * dh) as usize;
        // Host buffers.
        let q: Vec<f32> = (0..hd).map(|i| rnd(i as u64 * 7 + 1) * 0.3).collect();
        let kf: Vec<f32> = (0..kvn).map(|i| rnd(i as u64 * 13 + 3) * 0.5).collect();
        let vf: Vec<f32> = (0..kvn).map(|i| rnd(i as u64 * 17 + 5) * 0.5).collect();
        // Per-row int8 quantization (amax → scale = amax/127).
        let mut ki8 = vec![0i8; kvn];
        let mut vi8 = vec![0i8; kvn];
        let rows = (heads * seq) as usize;
        let mut ksc = vec![0f32; rows];
        let mut vsc = vec![0f32; rows];
        for r in 0..rows {
            let (mut ka, mut va) = (0f32, 0f32);
            for d in 0..dh as usize {
                ka = ka.max(kf[r * dh as usize + d].abs());
                va = va.max(vf[r * dh as usize + d].abs());
            }
            let (ks, vs) = (ka / 127.0, va / 127.0);
            ksc[r] = ks;
            vsc[r] = vs;
            for d in 0..dh as usize {
                ki8[r * dh as usize + d] = (kf[r * dh as usize + d] / ks.max(1e-12))
                    .round()
                    .clamp(-127.0, 127.0) as i8;
                vi8[r * dh as usize + d] = (vf[r * dh as usize + d] / vs.max(1e-12))
                    .round()
                    .clamp(-127.0, 127.0) as i8;
            }
        }
        let buf = |bytes: &[u8]| {
            device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let bq = buf(bytemuck_cast(&q));
        let bkf = buf(bytemuck_cast(&kf));
        let bvf = buf(bytemuck_cast(&vf));
        let bki = buf(unsafe { std::slice::from_raw_parts(ki8.as_ptr() as *const u8, ki8.len()) });
        let bvi = buf(unsafe { std::slice::from_raw_parts(vi8.as_ptr() as *const u8, vi8.len()) });
        let bksc = buf(bytemuck_cast(&ksc));
        let bvsc = buf(bytemuck_cast(&vsc));
        // int8 Q, per-head scale (for W8A8).
        let mut qi8 = vec![0i8; hd];
        let mut qsc = vec![0f32; heads as usize];
        for h in 0..heads as usize {
            let mut a = 0f32;
            for d in 0..dh as usize {
                a = a.max(q[h * dh as usize + d].abs());
            }
            let s = (a / 127.0).max(1e-12);
            qsc[h] = s;
            for d in 0..dh as usize {
                qi8[h * dh as usize + d] =
                    (q[h * dh as usize + d] / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let bqi = buf(unsafe { std::slice::from_raw_parts(qi8.as_ptr() as *const u8, qi8.len()) });
        let bqsc = buf(bytemuck_cast(&qsc));
        let osz = (heads * max_p * dh) as u64 * 4;
        let bo_f = device.new_buffer(osz, MTLResourceOptions::StorageModeShared);
        let bo_i = device.new_buffer(osz, MTLResourceOptions::StorageModeShared);
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let set_common = |enc: &metal::ComputeCommandEncoderRef, np: u32| {
            enc.set_bytes(4, 4, &heads as *const u32 as *const _);
            enc.set_bytes(5, 4, &seq as *const u32 as *const _);
            enc.set_bytes(6, 4, &dh as *const u32 as *const _);
            enc.set_bytes(7, 4, &scale as *const f32 as *const _);
            enc.set_bytes(10, 4, &np as *const u32 as *const _);
        };
        // f32 K/V read bytes per step = heads*seq*dh*4*2 (K+V); int8 = /4.
        let f32_bytes = (heads as f64) * (seq as f64) * (dh as f64) * 4.0 * 2.0;
        for &p in &[1u32, 8, 16, 32, 64] {
            let grid = MTLSize {
                width: (heads * p * 32) as u64,
                height: 1,
                depth: 1,
            };
            let mut f32_us = f64::INFINITY;
            let mut i8_us = f64::INFINITY;
            for _ in 0..20 {
                let cb = dev.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&p_f32);
                e.set_buffer(0, Some(&bq), 0);
                e.set_buffer(1, Some(&bkf), 0);
                e.set_buffer(2, Some(&bvf), 0);
                e.set_buffer(3, Some(&bo_f), 0);
                set_common(e, p);
                e.dispatch_threads(grid, tg);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                f32_us = f32_us.min(unsafe { gpu_secs(cb) } * 1e6);
                let cb = dev.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&p_i8);
                e.set_buffer(0, Some(&bq), 0);
                e.set_buffer(1, Some(&bki), 0);
                e.set_buffer(2, Some(&bvi), 0);
                e.set_buffer(3, Some(&bo_i), 0);
                set_common(e, p);
                e.set_buffer(8, Some(&bksc), 0);
                e.set_buffer(9, Some(&bvsc), 0);
                e.dispatch_threads(grid, tg);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                i8_us = i8_us.min(unsafe { gpu_secs(cb) } * 1e6);
            }
            println!(
                "{seq:>7} {p:>4} {f32_us:>9.2} {i8_us:>9.2} {:>7.2}x {:>10.1} {:>10.1}",
                f32_us / i8_us,
                f32_bytes / (f32_us * 1e3),
                (f32_bytes / 4.0) / (i8_us * 1e3),
            );
        }
        // ── RE breakdown + W8A8 (P=32) ──────────────────────────────────
        if seq >= 4096 {
            let p = 32u32;
            let grid = MTLSize {
                width: (heads * p * 32) as u64,
                height: 1,
                depth: 1,
            };
            let time_k = |pipe: &metal::ComputePipelineState,
                          bind: &dyn Fn(&metal::ComputeCommandEncoderRef)|
             -> f64 {
                let mut us = f64::INFINITY;
                for _ in 0..20 {
                    let cb = dev.queue.new_command_buffer();
                    let e = cb.new_compute_command_encoder();
                    e.set_compute_pipeline_state(pipe);
                    bind(e);
                    set_common(e, p);
                    e.dispatch_threads(grid, tg);
                    e.end_encoding();
                    cb.commit();
                    cb.wait_until_completed();
                    us = us.min(unsafe { gpu_secs(cb) } * 1e6);
                }
                us
            };
            let bind_f32 = |e: &metal::ComputeCommandEncoderRef| {
                e.set_buffer(0, Some(&bq), 0);
                e.set_buffer(1, Some(&bkf), 0);
                e.set_buffer(2, Some(&bvf), 0);
                e.set_buffer(3, Some(&bo_f), 0);
            };
            let bind_w = |e: &metal::ComputeCommandEncoderRef| {
                e.set_buffer(0, Some(&bqi), 0);
                e.set_buffer(1, Some(&bki), 0);
                e.set_buffer(2, Some(&bvi), 0);
                e.set_buffer(3, Some(&bo_i), 0);
                e.set_buffer(8, Some(&bksc), 0);
                e.set_buffer(9, Some(&bvsc), 0);
                e.set_buffer(11, Some(&bqsc), 0);
            };
            let full = time_k(&p_f32, &bind_f32);
            let noqk = time_k(&p_noqk, &bind_f32);
            let nopv = time_k(&p_nopv, &bind_f32);
            let w8a8 = time_k(&p_w8a8, &bind_w);
            // Clean correctness pair: f32-full → bo_f, w8a8 → bo_i, compare.
            for (pipe, bind) in [
                (&p_f32, &bind_f32 as &dyn Fn(_)),
                (&p_w8a8, &bind_w as &dyn Fn(_)),
            ] {
                let cb = dev.queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(pipe);
                bind(e);
                set_common(e, p);
                e.dispatch_threads(grid, tg);
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let np = (heads * p * dh) as usize;
            let of = unsafe { std::slice::from_raw_parts(bo_f.contents() as *const f32, np) };
            let ow = unsafe { std::slice::from_raw_parts(bo_i.contents() as *const f32, np) };
            let maxd = of
                .iter()
                .zip(ow)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "  [RE ctx={seq} P=32] full={full:.0} noqk={noqk:.0} nopv={nopv:.0} µs → QK-dot≈{:.0} PV-acc≈{:.0} softmax+rest≈{:.0} | W8A8={w8a8:.0} ({:.2}x vs full) maxΔ={maxd:.1e}",
                full - noqk,
                full - nopv,
                noqk.min(nopv),
                full / w8a8,
            );
        }
    }
    println!(
        "note: int8 reads 4× less KV. If f32 GB/s ≈ peak (~273) it's KV-bound → int8 wins; if far below, it's compute/occupancy-bound → int8 ~neutral."
    );
}

#[cfg(target_os = "macos")]
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
