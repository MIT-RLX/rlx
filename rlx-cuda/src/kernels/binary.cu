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

// Element-wise binary op. Selector lives in `op`:
//   0=add 1=sub 2=mul 3=div 4=max 5=min 6=pow

extern "C" __global__ void binary(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int op
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = arena[a_off + i];
    float b = arena[b_off + i];
    float c;
    switch (op) {
        case 0: c = a + b; break;
        case 1: c = a - b; break;
        case 2: c = a * b; break;
        case 3: c = a / b; break;
        case 4: c = fmaxf(a, b); break;
        case 5: c = fminf(a, b); break;
        case 6: c = powf(a, b); break;
        default: c = 0.0f;
    }
    arena[c_off + i] = c;
}
