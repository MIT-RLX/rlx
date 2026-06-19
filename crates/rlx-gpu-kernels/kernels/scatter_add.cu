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

// Two-phase scatter-add. Phase 0 zeros the output; phase 1 atomic-adds
// updates by row. CUDA has native float atomicAdd, so unlike wgpu we
// don't need to serialize.

extern "C" __global__ void scatter_add_zero(
    float* arena,
    unsigned int out_off,
    unsigned int out_total
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_total) return;
    arena[out_off + i] = 0.0f;
}

extern "C" __global__ void scatter_add_acc(
    float* arena,
    unsigned int out_off,
    unsigned int upd_off,
    unsigned int idx_off,
    unsigned int num_updates,
    unsigned int trailing,
    unsigned int out_dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_updates * trailing;
    if (i >= total) return;
    unsigned int upd_i = i / trailing;
    unsigned int upd_j = i % trailing;
    unsigned int row = (unsigned int)arena[idx_off + upd_i];
    if (row >= out_dim) return;
    float v = arena[upd_off + upd_i * trailing + upd_j];
    atomicAdd(&arena[out_off + row * trailing + upd_j], v);
}
