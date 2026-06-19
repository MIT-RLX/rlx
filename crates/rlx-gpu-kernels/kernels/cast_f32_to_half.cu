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

// Cast a contiguous f32 buffer to a u16 half-precision buffer
// (f16 / IEEE-754 binary16, or bf16 / Brain float). Used by the
// mixed-precision matmul path to convert activations on-the-fly when
// the matching weight is stored in the half-arena side-buffer.
//
// `dtype`:
//   0 = F16
//   1 = BF16

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void cast_f32_to_half(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    unsigned int n,
    unsigned int dtype
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    unsigned short out;
    if (dtype == 0u) {
        __half h = __float2half_rn(v);
        out = *reinterpret_cast<const unsigned short*>(&h);
    } else {
        __nv_bfloat16 h = __float2bfloat16_rn(v);
        out = *reinterpret_cast<const unsigned short*>(&h);
    }
    dst[i] = out;
}
