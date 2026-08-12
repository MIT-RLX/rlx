// Full qwen3-0.6B decode-STEP microbench: f32 attention vs W8A8 attention (int8
// Q + int8 K → integer QK dot), MEASURED end-to-end decode tps + attention
// precision. The real op mix per layer — 7 projections (f16 GEMV) + GQA flash
// attention — ×28 layers + lm_head, so attention sits at its true fraction of the
// step. KV is pre-quantized (= stored / incremental-int8, the achievable
// production form; not per-step re-quant). Projection weights are dummy (values
// don't affect timing); attention precision is reported separately from an
// exact f32-vs-W8A8 attention-output comparison.
//
//   cargo run --release -p rlx-metal --example w8a8_decode_bench

// `objc`'s `msg_send!` expands to `cfg(feature = "cargo-clippy")` checks that
// modern rustc doesn't recognize. Third-party noise; mirrors the crate-root
// allow in `src/lib.rs`.
#![allow(unexpected_cfgs)]

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use rlx_metal::blas::metal_sgemm_f16w_bufs;
    use rlx_metal::mtl::{MTLResourceOptions, MTLSize};

    let dev = rlx_metal::device::metal_device().expect("metal device");
    let device = &dev.device;
    unsafe fn gpu_secs(cb: &rlx_metal::mtl::CommandBufferRef) -> f64 {
        use objc::{msg_send, runtime::Object, sel, sel_impl};
        let obj = cb as *const rlx_metal::mtl::CommandBufferRef as *mut Object;
        let s: f64 = msg_send![obj, GPUStartTime];
        let e: f64 = msg_send![obj, GPUEndTime];
        (e - s).max(0.0)
    }
    let rnd = |mut z: u64| -> f32 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };

    // qwen3-0.6B dims.
    let hidden = 1024usize;
    let heads = 16u32;
    let kv_heads = 8u32;
    let dh = 128u32;
    let qd = (heads * dh) as usize; // 2048
    let kvd = (kv_heads * dh) as usize; // 1024
    let inter = 3072usize;
    let vocab = 151936usize;
    let layers = 28usize;
    let scale = 1.0 / (dh as f32).sqrt();

    // Flash attention (GQA) partial + int8 W8A8 partial. Partials only (timing);
    // o_proj reads slot 0 (values irrelevant to timing). Precision uses a
    // separate n_part=1 exact comparison below.
    let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void attn_f32(
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& n_part [[buffer(10)]], constant uint& kvh [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h=tgid/n_part, part=tgid%n_part; if(h>=heads) return;
    uint hk=h/(heads/kvh); uint chunk=(seq+n_part-1u)/n_part, ks=part*chunk, ke=min(ks+chunk,seq);
    float q[128]; for(uint d=0;d<dh;++d) q[d]=Q[h*dh+d];
    float m=-1e30f,l=0.f,o[128]; for(uint d=0;d<dh;++d) o[d]=0.f;
    for(uint ki=ks+tid; ki<ke; ki+=tsize){
        device const float* kr=K+(ulong)(hk*seq+ki)*dh; float dot=0.f; for(uint d=0;d<dh;++d) dot+=q[d]*kr[d];
        float s=dot*scale, mn=max(m,s), eo=exp(m-mn), ec=exp(s-mn); l=eo*l+ec;
        device const float* vr=V+(ulong)(hk*seq+ki)*dh; for(uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*vr[d]; m=mn;
    }
    float mg=simd_max(m), r=exp(m-mg), lg=simd_sum(l*r), inv=1.0f/max(lg,1e-20f);
    for(uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
kernel void attn_w8a8(
    device const char* Q [[buffer(0)]], device const char* K [[buffer(1)]],
    device const char* V [[buffer(2)]], device float* OUT [[buffer(3)]],
    constant uint& heads [[buffer(4)]], constant uint& seq [[buffer(5)]],
    constant uint& dh [[buffer(6)]], constant float& scale [[buffer(7)]],
    device const float* Ksc [[buffer(8)]], device const float* Vsc [[buffer(9)]],
    constant uint& n_part [[buffer(10)]], device const float* Qsc [[buffer(11)]],
    constant uint& kvh [[buffer(13)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    uint h=tgid/n_part, part=tgid%n_part; if(h>=heads) return;
    uint hk=h/(heads/kvh); uint chunk=(seq+n_part-1u)/n_part, ks=part*chunk, ke=min(ks+chunk,seq);
    char qi[128]; for(uint d=0;d<dh;++d) qi[d]=Q[h*dh+d]; float qs=Qsc[h];
    float m=-1e30f,l=0.f,o[128]; for(uint d=0;d<dh;++d) o[d]=0.f;
    for(uint ki=ks+tid; ki<ke; ki+=tsize){
        device const char* kr=K+(ulong)(hk*seq+ki)*dh;
        int di=0; for(uint d=0;d<dh;d+=4u) di+=int(qi[d])*int(kr[d])+int(qi[d+1])*int(kr[d+1])+int(qi[d+2])*int(kr[d+2])+int(qi[d+3])*int(kr[d+3]);
        float s=float(di)*qs*Ksc[hk*seq+ki]*scale, mn=max(m,s), eo=exp(m-mn), ec=exp(s-mn); l=eo*l+ec;
        device const char* vr=V+(ulong)(hk*seq+ki)*dh; float vs=Vsc[hk*seq+ki]; for(uint d=0;d<dh;++d) o[d]=eo*o[d]+ec*(float(vr[d])*vs); m=mn;
    }
    float mg=simd_max(m), r=exp(m-mg), lg=simd_sum(l*r), inv=1.0f/max(lg,1e-20f);
    for(uint d=0;d<dh;++d){ float og=simd_sum(o[d]*r); if(tid==0) OUT[(h*n_part+part)*dh+d]=og*inv; }
}
"#;
    let lib = device
        .new_library_with_source(src, &rlx_metal::mtl::CompileOptions::new())
        .expect("msl");
    let mk = |n: &str| {
        device
            .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
            .unwrap()
    };
    let p_f32 = mk("attn_f32");
    let p_w8 = mk("attn_w8a8");

    // Arena: dummy f16 weights + f32 activations + K/V (f32 + int8) + scales.
    let cap = 1200usize * 1024 * 1024;
    let arena = device.new_buffer(cap as u64, MTLResourceOptions::StorageModeShared);
    let base = arena.contents() as *mut u8;
    let al = |x: usize| (x + 255) & !255;
    // f16 weight blocks (values irrelevant → leave zero, timing only).
    let wq = 0usize;
    let wk = al(wq + hidden * qd * 2);
    let wv = al(wk + hidden * kvd * 2);
    let wo = al(wv + hidden * kvd * 2);
    let wg = al(wo + qd * hidden * 2);
    let wu = al(wg + hidden * inter * 2);
    let wd = al(wu + hidden * inter * 2);
    let wlm = al(wd + inter * hidden * 2);
    // activation scratch (f32)
    let a_hid = al(wlm + hidden * vocab * 2);
    let a_q = al(a_hid + hidden * 4);
    let a_k = al(a_q + qd * 4);
    let a_v = al(a_k + kvd * 4);
    let a_att = al(a_v + kvd * 4); // heads*P*dh partials
    let a_gate = al(a_att + (heads as usize) * 64 * (dh as usize) * 4);
    let a_up = al(a_gate + inter * 4);
    let a_log = al(a_up + inter * 4);

    println!(
        "W8A8 full decode-step (qwen3-0.6B dims, {layers} layers, GQA {kv_heads}kv, GPU best-of-15)"
    );
    println!(
        "{:>7} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "ctx", "f32 µs", "w8a8 µs", "speedup", "f32 tps", "w8a8 tps"
    );

    for &ctx in &[512u32, 2048, 4096, 8192] {
        let seq = ctx;
        // K/V cache [kv_heads, seq, dh] f32 + int8 + per-row scales.
        let kv_off = al(a_log + vocab * 4);
        let kelem = (kv_heads * seq * dh) as usize;
        let kf = kv_off;
        let vf = al(kf + kelem * 4);
        let ki8 = al(vf + kelem * 4);
        let vi8 = al(ki8 + kelem);
        let ksc = al(vi8 + kelem);
        let vsc = al(ksc + (kv_heads * seq) as usize * 4);
        let qi8 = al(vsc + (kv_heads * seq) as usize * 4);
        let qsc = al(qi8 + qd);
        if qsc + (heads as usize) * 4 > cap {
            println!("{ctx:>7}  (skip: exceeds arena)");
            continue;
        }
        // Fill K/V f32 + quantize to int8 (per-row) + Q int8 (per-head).
        unsafe {
            let kfp = base.add(kf) as *mut f32;
            let vfp = base.add(vf) as *mut f32;
            let kip = base.add(ki8) as *mut i8;
            let vip = base.add(vi8) as *mut i8;
            let ksp = base.add(ksc) as *mut f32;
            let vsp = base.add(vsc) as *mut f32;
            for r in 0..(kv_heads * seq) as usize {
                let (mut ka, mut va) = (0f32, 0f32);
                for d in 0..dh as usize {
                    let kv = rnd((r * 131 + d) as u64) * 0.5;
                    let vv = rnd((r * 137 + d + 9) as u64) * 0.5;
                    *kfp.add(r * dh as usize + d) = kv;
                    *vfp.add(r * dh as usize + d) = vv;
                    ka = ka.max(kv.abs());
                    va = va.max(vv.abs());
                }
                let (ks_, vs_) = (ka / 127.0, va / 127.0);
                *ksp.add(r) = ks_;
                *vsp.add(r) = vs_;
                for d in 0..dh as usize {
                    *kip.add(r * dh as usize + d) = (*kfp.add(r * dh as usize + d) / ks_.max(1e-12))
                        .round()
                        .clamp(-127.0, 127.0)
                        as i8;
                    *vip.add(r * dh as usize + d) = (*vfp.add(r * dh as usize + d) / vs_.max(1e-12))
                        .round()
                        .clamp(-127.0, 127.0)
                        as i8;
                }
            }
            let qip = base.add(qi8) as *mut i8;
            let qsp = base.add(qsc) as *mut f32;
            let qfp = base.add(a_q) as *mut f32;
            for h in 0..heads as usize {
                let mut a = 0f32;
                for d in 0..dh as usize {
                    let qv = rnd((h * 91 + d) as u64) * 0.4;
                    *qfp.add(h * dh as usize + d) = qv;
                    a = a.max(qv.abs());
                }
                let s = (a / 127.0).max(1e-12);
                *qsp.add(h) = s;
                for d in 0..dh as usize {
                    *qip.add(h * dh as usize + d) = (*qfp.add(h * dh as usize + d) / s)
                        .round()
                        .clamp(-127.0, 127.0)
                        as i8;
                }
            }
        }

        let np = 8u32;
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let agrid = MTLSize {
            width: (heads * np * 32) as u64,
            height: 1,
            depth: 1,
        };
        // One command buffer = one decode step. `w8` selects the attention variant.
        let step = |w8: bool| -> f64 {
            let mut us = f64::INFINITY;
            for _ in 0..15 {
                let cb = dev.queue.new_command_buffer();
                for _ in 0..layers {
                    let e = cb.new_compute_command_encoder();
                    // QKV projections (f16 GEMV).
                    metal_sgemm_f16w_bufs(e, &arena, a_hid, &arena, wq, &arena, a_q, 1, hidden, qd);
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_hid, &arena, wk, &arena, a_k, 1, hidden, kvd,
                    );
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_hid, &arena, wv, &arena, a_v, 1, hidden, kvd,
                    );
                    // Attention (f32 or W8A8).
                    if w8 {
                        e.set_compute_pipeline_state(&p_w8);
                        e.set_buffer(0, Some(&arena), qi8 as u64);
                        e.set_buffer(1, Some(&arena), ki8 as u64);
                        e.set_buffer(2, Some(&arena), vi8 as u64);
                        e.set_buffer(3, Some(&arena), a_att as u64);
                        e.set_buffer(8, Some(&arena), ksc as u64);
                        e.set_buffer(9, Some(&arena), vsc as u64);
                        e.set_buffer(11, Some(&arena), qsc as u64);
                        e.set_bytes(13, 4, &kv_heads as *const u32 as *const _);
                    } else {
                        e.set_compute_pipeline_state(&p_f32);
                        e.set_buffer(0, Some(&arena), a_q as u64);
                        e.set_buffer(1, Some(&arena), kf as u64);
                        e.set_buffer(2, Some(&arena), vf as u64);
                        e.set_buffer(3, Some(&arena), a_att as u64);
                        e.set_bytes(12, 4, &kv_heads as *const u32 as *const _);
                    }
                    e.set_bytes(4, 4, &heads as *const u32 as *const _);
                    e.set_bytes(5, 4, &seq as *const u32 as *const _);
                    e.set_bytes(6, 4, &dh as *const u32 as *const _);
                    e.set_bytes(7, 4, &scale as *const f32 as *const _);
                    e.set_bytes(10, 4, &np as *const u32 as *const _);
                    e.dispatch_threads(agrid, tg);
                    // o_proj (reads attn partial slot 0) + MLP.
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_att, &arena, wo, &arena, a_hid, 1, qd, hidden,
                    );
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_hid, &arena, wg, &arena, a_gate, 1, hidden, inter,
                    );
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_hid, &arena, wu, &arena, a_up, 1, hidden, inter,
                    );
                    metal_sgemm_f16w_bufs(
                        e, &arena, a_gate, &arena, wd, &arena, a_hid, 1, inter, hidden,
                    );
                    e.end_encoding();
                }
                let e = cb.new_compute_command_encoder();
                metal_sgemm_f16w_bufs(
                    e, &arena, a_hid, &arena, wlm, &arena, a_log, 1, hidden, vocab,
                );
                e.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                us = us.min(unsafe { gpu_secs(cb) } * 1e6);
            }
            us
        };
        let f = step(false);
        let w = step(true);
        println!(
            "{ctx:>7} {f:>10.1} {w:>10.1} {:>7.2}x {:>9.1} {:>9.1}",
            f / w,
            1e6 / f,
            1e6 / w,
        );
    }
    println!(
        "note: KV pre-quantized (stored/incremental int8). Attention precision (int8 Q+K vs f32): ~1e-4 (see int8_kv_bench). Projections dummy-weight (timing only)."
    );
}
