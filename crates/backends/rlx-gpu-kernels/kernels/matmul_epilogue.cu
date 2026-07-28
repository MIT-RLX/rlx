// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// cuBLAS sgemm epilogue: bias + activation fused into one element-wise
// kernel. Used after a plain cublasSgemm to apply the bias+act fusion
// that cuBLAS doesn't natively do.
//
// Activation IDs match the unary kernel's table:
//   0=relu 1=sigmoid 2=tanh 5=sqrt 7=neg 8=abs 9=gelu 10=silu 11=gelu_approx
//   0xFFFF=identity (skip)

__device__ __forceinline__ float ep_apply_act(float v, unsigned int act_id) {
    if (act_id == 0xFFFFu) return v;
    switch (act_id) {
        case 0:  return fmaxf(v, 0.0f);
        case 1:  return 1.0f / (1.0f + expf(-fminf(fmaxf(v, -88.0f), 88.0f)));
        case 2:  return tanhf(fminf(fmaxf(v, -15.0f), 15.0f));
        case 5:  return sqrtf(v);
        case 7:  return -v;
        case 8:  return fabsf(v);
        case 9:  return gelu_erf(v);
        case 11: return gelu_approx(v);
        case 10: {
            float nx = fminf(fmaxf(-v, -88.0f), 88.0f);
            return v / (1.0f + expf(nx));
        }
        default: return v;
    }
}

extern "C" __global__ void matmul_epilogue(
    float* arena,
    unsigned int total,
    unsigned int cols,
    unsigned int c_off,
    unsigned int has_bias,
    unsigned int bias_off,
    unsigned int act_id
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = arena[c_off + i];
    if (has_bias) {
        unsigned int col = i % cols;
        v += arena[bias_off + col];
    }
    v = ep_apply_act(v, act_id);
    arena[c_off + i] = v;
}
