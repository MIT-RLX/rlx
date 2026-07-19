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

// Packed GatedResidual backward: out = [dx ∥ dy ∥ dgate] (1-D floats).
// Launch: grid=(mod_rows,1,1), block=(256,1,1).

extern "C" __global__ void gated_residual_backward(
    float* arena,
    unsigned int mod_rows,
    unsigned int seq_per_mod,
    unsigned int inner,
    unsigned int y_off,
    unsigned int gate_off,
    unsigned int dy_off,
    unsigned int out_off
) {
    unsigned int m = blockIdx.x;
    if (m >= mod_rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    unsigned int nx = mod_rows * seq_per_mod * inner;
    unsigned int gate_base = m * inner;
    float* dx = arena + out_off;
    float* dy_out = arena + out_off + nx;
    float* dgate = arena + out_off + 2 * nx;

    for (unsigned int i = tid; i < inner; i += bsz) {
        float acc = 0.0f;
        float g = arena[gate_off + gate_base + i];
        for (unsigned int seq = 0; seq < seq_per_mod; seq++) {
            unsigned int row = m * seq_per_mod + seq;
            unsigned int idx = row * inner + i;
            float d = arena[dy_off + idx];
            dx[idx] = d;
            dy_out[idx] = d * g;
            acc += d * arena[y_off + idx];
        }
        dgate[gate_base + i] = acc;
    }
}
