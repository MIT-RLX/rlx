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

// Conv-output epilogue: NCHW `y = act(y + bias[c])`, bias broadcast over
// N/H/W. Used by `Step::Conv2d` (has_bias=1) after the custom direct-conv
// kernel on shapes cuDNN's fused conv-bias-activation does not serve
// (1×1 / depthwise / grouped), and whenever libcudnn is absent. On the
// cuDNN-friendly forward path `cudnnConvolutionBiasActivationForward` does
// the same fold in one call, so this kernel is not launched.
//
// Activation IDs match the unary kernel's table (see `activation_op_id`):
//   0=relu 1=sigmoid 2=tanh 5=sqrt 7=neg 8=abs 9=gelu 10=silu 11=gelu_approx
//   0xFFFF=identity (skip). `FuseConvBiasAct` only folds relu/sigmoid/tanh,
// but the full table is kept so the epilogue matches `matmul_epilogue`.

__device__ __forceinline__ float cba_apply_act(float v, unsigned int act_id) {
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

extern "C" __global__ void conv_bias_act_epilogue(
    float* arena,
    unsigned int total,      // N * C_out * H_out * W_out
    unsigned int hw,         // H_out * W_out (per-channel spatial size)
    unsigned int channels,   // C_out
    unsigned int c_off,
    unsigned int has_bias,
    unsigned int bias_off,
    unsigned int act_id,
    unsigned int has_residual,  // ResNet residual (cuDNN `z`), added before act
    unsigned int residual_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = arena[c_off + i];
    if (has_bias) {
        unsigned int c = (i / hw) % channels;
        v += arena[bias_off + c];
    }
    // `y = act(conv + bias + residual)` — residual is a full NCHW tensor, so it
    // indexes by the flat element `i` (no broadcast).
    if (has_residual) {
        v += arena[residual_off + i];
    }
    v = cba_apply_act(v, act_id);
    arena[c_off + i] = v;
}
