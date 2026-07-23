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

// ONNX ScatterND (reduction=none): copy data→dst, then write update slices.
// Indices are f32-encoded integers (f32-uniform arena) or raw i32 bit patterns
// stored in f32 slots; i64 indices use a separate packed path (host for now).
//
// One thread per (update, slice_elem). Offset = sum_m index[m]*stride[m].

extern "C" __global__ void scatter_nd_f32(
    float* arena,
    unsigned data_off,
    unsigned idx_off,
    unsigned upd_off,
    unsigned dst_off,
    unsigned num_updates,
    unsigned slice,
    unsigned k,
    unsigned s0, unsigned s1, unsigned s2, unsigned s3,
    unsigned d0, unsigned d1, unsigned d2, unsigned d3,
    unsigned copy_data // 1 = memcpy data→dst first (handled on host before launch)
) {
    (void)copy_data;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = num_updates * slice;
    if (tid >= total) return;
    unsigned u = tid / slice;
    unsigned j = tid - u * slice;
    unsigned strides[4] = {s0, s1, s2, s3};
    unsigned dims[4] = {d0, d1, d2, d3};
    unsigned off = 0;
    for (unsigned m = 0; m < k && m < 4u; ++m) {
        float fv = arena[idx_off + u * k + m];
        int idx = (int)rintf(fv);
        int dim = (int)dims[m];
        if (idx < 0) idx += dim;
        if (idx < 0) idx = 0;
        if (dim > 0 && idx >= dim) idx = dim - 1;
        off += (unsigned)idx * strides[m];
    }
    arena[dst_off + off + j] = arena[upd_off + u * slice + j];
}
