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

// 1D pool NCL.
extern "C" __global__ void pool1d(
    float* arena,
    unsigned int n, unsigned int c, unsigned int l,
    unsigned int l_out,
    unsigned int kl, unsigned int sl, unsigned int pl,
    unsigned int op,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int total = n * c * l_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int lo = i % l_out;
    unsigned int q1 = i / l_out;
    unsigned int cc = q1 % c;
    unsigned int nn = q1 / c;
    float acc;
    switch (op) {
        case 0: acc = -3.4e38f; break;
        case 3: acc =  3.4e38f; break;
        case 4: acc = 1.0f; break;
        default: acc = 0.0f;
    }
    unsigned int count = 0;
    for (unsigned int ki = 0; ki < kl; ++ki) {
        int il = (int)(lo * sl + ki) - (int)pl;
        if (il < 0 || il >= (int)l) continue;
        float v = arena[in_off + (nn * c + cc) * l + il];
        switch (op) {
            case 0: acc = fmaxf(acc, v); break;
            case 1: case 2: acc += v; break;
            case 3: acc = fminf(acc, v); break;
            case 4: acc *= v; break;
        }
        count++;
    }
    if (op == 1 && count > 0) acc /= (float)count;
    arena[out_off + i] = acc;
}
