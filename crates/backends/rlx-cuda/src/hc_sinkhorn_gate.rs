// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native CUDA kernel for the **HC (hyper-connection) Sinkhorn gate**
//! (`Op::Custom("dsv4.hc_sinkhorn_gate")`) — one thread per row computes the whole
//! `[hc×hc]` Sinkhorn in registers (`hc ≤ 4` ⇒ ≤16 floats), NVRTC-compiled and
//! launched straight against the arena device buffer (no host roundtrip). Replaces
//! ~36 tiny Div/Reduce launches with one dispatch. Mirrors the CPU/Metal/wgpu
//! kernels (all bit-exact). `hc`/`iters`/`eps` arrive via [`CudaGpuKernel::extras`]
//! (`e0=hc, e1=iters, e2=eps bit-pattern`). Registered by the consumer via
//! [`register`].
//!
//! NB: NOT compiled/validated on Apple hardware — build + validate on a CUDA GPU.

use std::sync::Arc;

use rlx_ir::Shape;

use crate::cuda_gpu_kernels::{CudaGpuKernel, register_cuda_gpu_kernel};

pub const OP_NAME: &str = "dsv4.hc_sinkhorn_gate";

const CU: &str = r#"
extern "C" __global__ void rlx_custom(
    float* arena,
    unsigned out_off, unsigned out_len, unsigned n_inputs,
    unsigned in0_off, unsigned in0_len,
    unsigned in1_off, unsigned in1_len,
    unsigned in2_off, unsigned in2_len,
    unsigned in3_off, unsigned in3_len,
    unsigned e0, unsigned e1, unsigned e2, unsigned e3)
{
    (void)n_inputs; (void)in0_len; (void)in1_len; (void)in2_len;
    (void)in3_off; (void)in3_len; (void)e3;
    unsigned hc = e0;                 // e0=hc, e1=iters, e2=eps bits
    unsigned iters = e1;
    float eps = __uint_as_float(e2);
    unsigned mh = 2u*hc + hc*hc;
    unsigned rows = out_len / mh;
    unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* m = arena + in0_off + r*mh;      // mixes row
    const float* base = arena + in2_off;          // [mh]
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
struct HcSinkhornGateCuda;

impl CudaGpuKernel for HcSinkhornGateCuda {
    fn name(&self) -> &str {
        OP_NAME
    }
    fn cuda_c(&self) -> &str {
        CU
    }
    fn extras(&self, attrs: &[u8], _out: &Shape) -> [u32; 4] {
        // attrs = [hc: u32, iters: u32, eps: f32]; pass eps as its bit pattern.
        let rd =
            |i: usize| u32::from_le_bytes([attrs[i], attrs[i + 1], attrs[i + 2], attrs[i + 3]]);
        [rd(0), rd(4), rd(8), 0]
    }
}

/// Register the native CUDA Sinkhorn-gate kernel (idempotent).
pub fn register() {
    register_cuda_gpu_kernel(Arc::new(HcSinkhornGateCuda));
}
