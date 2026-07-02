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

// Grouped (MoE) matmul. Each thread computes one C[m, n] output;
// per-token expert id picks which weight matrix to multiply against.

extern "C" __global__ void grouped_matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int idx_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    unsigned int e = (unsigned int)arena[idx_off + row];
    if (e >= num_experts) return;
    unsigned int wb = w_off + e * k * n;
    unsigned int ib = in_off + row * k;
    float acc = 0.0f;
    for (unsigned int kk = 0; kk < k; ++kk) {
        acc += arena[ib + kk] * arena[wb + kk * n + col];
    }
    arena[out_off + row * n + col] = acc;
}
