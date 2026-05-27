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
// Cumsum backward along last axis (one thread per row).

extern "C" __global__ void rlx_cumsum_bwd(
    float* arena,
    unsigned int outer,
    unsigned int inner,
    unsigned int dy_off,
    unsigned int dx_off,
    unsigned int exclusive
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    unsigned int dy_base = dy_off + row * inner;
    unsigned int dx_base = dx_off + row * inner;
    float suffix = 0.0f;
    for (int i = (int)inner - 1; i >= 0; --i) {
        if (exclusive != 0u) {
            arena[dx_base + (unsigned int)i] = suffix;
            suffix += arena[dy_base + (unsigned int)i];
        } else {
            suffix += arena[dy_base + (unsigned int)i];
            arena[dx_base + (unsigned int)i] = suffix;
        }
    }
}
