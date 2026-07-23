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

// Single-rounded fused multiply-add: out[i] = fma(a[i], b[i], c[i]).
// Matches Metal `elem_fma` / wgpu `fma.wgsl` (one rounding).

extern "C" __global__ void fma_elem(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = arena[a_off + i];
    float b = arena[b_off + i];
    float c = arena[c_off + i];
    arena[out_off + i] = __fmaf_rn(a, b, c);
}
