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

// Element-wise comparison. Output is 1.0 / 0.0 in f32 (matches the
// rest of the workspace's "Bool stored as f32" convention).
//   0=eq 1=ne 2=lt 3=le 4=gt 5=ge

extern "C" __global__ void compare(
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
    bool r;
    switch (op) {
        case 0: r = a == b; break;
        case 1: r = a != b; break;
        case 2: r = a <  b; break;
        case 3: r = a <= b; break;
        case 4: r = a >  b; break;
        case 5: r = a >= b; break;
        default: r = false;
    }
    arena[c_off + i] = r ? 1.0f : 0.0f;
}
