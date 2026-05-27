// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
        case 9:
        case 11: {
            const float c = 0.7978845608028654f;
            float x3 = v * v * v;
            float inner = c * (v + 0.044715f * x3);
            inner = fminf(fmaxf(inner, -15.0f), 15.0f);
            return 0.5f * v * (1.0f + tanhf(inner));
        }
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
