// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native ROCm/HIP kernel for the **HC Sinkhorn gate**
//! (`Op::Custom("dsv4.hc_sinkhorn_gate")`) — one thread per row computes the whole
//! `[hc×hc]` Sinkhorn in registers, hipRTC-compiled and launched against the arena
//! device buffer (no host roundtrip). Mirrors the CPU/Metal/wgpu kernels. The ROCm
//! GpuKernel seam passes only offset/len args (no attrs), so — like wgpu — `hc` is
//! derived from `in2_len` (= mix_hc) and the DSV4 Sinkhorn constants (iters=3,
//! eps=1e-6) are used. Registered by the consumer via [`register`].
//!
//! NB: NOT compiled/validated on Apple hardware — build + validate on an AMD GPU.

use std::sync::Arc;

use crate::rocm_gpu_kernels::{RocmGpuKernel, register_rocm_gpu_kernel};

pub const OP_NAME: &str = "dsv4.hc_sinkhorn_gate";

const HIP: &str = r#"
extern "C" __global__ void rlx_custom(
    float* arena,
    unsigned out_off, unsigned out_len, unsigned n_inputs,
    unsigned in0_off, unsigned in0_len,
    unsigned in1_off, unsigned in1_len,
    unsigned in2_off, unsigned in2_len,
    unsigned in3_off, unsigned in3_len)
{
    (void)n_inputs; (void)out_len; (void)in3_off; (void)in3_len;
    unsigned mh = in2_len;                          // base_len == mix_hc = 2hc+hc^2
    unsigned hc = (unsigned)(sqrtf((float)(1u + mh)) - 1.0f + 0.5f);
    unsigned rows = in0_len / mh;
    unsigned iters = 3u;
    float eps = 1e-6f;
    unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* m = arena + in0_off + r*mh;
    const float* base = arena + in2_off;
    float s0 = arena[in1_off], s1 = arena[in1_off+1], s2 = arena[in1_off+2];
    float* o = arena + out_off + r*mh;
    for (unsigned i=0;i<hc;i++) o[i] = 1.0f/(1.0f+expf(-(m[i]*s0+base[i]))) + eps;
    for (unsigned i=0;i<hc;i++) o[hc+i] = 2.0f/(1.0f+expf(-(m[hc+i]*s1+base[hc+i])));
    float c[16];
    for (unsigned j=0;j<hc;j++){
        float mx=-1e30f;
        for(unsigned k=0;k<hc;k++){ float l=m[2u*hc+j*hc+k]*s2+base[2u*hc+j*hc+k]; c[j*hc+k]=l; mx=fmaxf(mx,l); }
        float sm=0.0f; for(unsigned k=0;k<hc;k++){ float e=expf(c[j*hc+k]-mx); c[j*hc+k]=e; sm+=e; }
        for(unsigned k=0;k<hc;k++) c[j*hc+k]=c[j*hc+k]/sm+eps;
    }
    for(unsigned k=0;k<hc;k++){ float cs=eps; for(unsigned j=0;j<hc;j++) cs+=c[j*hc+k]; for(unsigned j=0;j<hc;j++) c[j*hc+k]/=cs; }
    for(unsigned it=1;it<iters;it++){
        for(unsigned j=0;j<hc;j++){ float rs=eps; for(unsigned k=0;k<hc;k++) rs+=c[j*hc+k]; for(unsigned k=0;k<hc;k++) c[j*hc+k]/=rs; }
        for(unsigned k=0;k<hc;k++){ float cs=eps; for(unsigned j=0;j<hc;j++) cs+=c[j*hc+k]; for(unsigned j=0;j<hc;j++) c[j*hc+k]/=cs; }
    }
    for(unsigned idx=0;idx<hc*hc;idx++) o[2u*hc+idx]=c[idx];
}
"#;

#[derive(Debug)]
struct HcSinkhornGateRocm;

impl RocmGpuKernel for HcSinkhornGateRocm {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn hip_c(&self) -> &str {
        HIP
    }
}

/// Register the native ROCm Sinkhorn-gate kernel (idempotent).
pub fn register() {
    register_rocm_gpu_kernel(Arc::new(HcSinkhornGateRocm));
}
