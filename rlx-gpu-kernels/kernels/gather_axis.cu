// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Gather along an arbitrary axis. Output layout [outer, num_idx, trailing];
// source layout [outer, axis_dim, trailing]. Indices are f32-encoded.

extern "C" __global__ void gather_axis(
    float* arena,
    unsigned int total,
    unsigned int outer,
    unsigned int axis_dim,
    unsigned int num_idx,
    unsigned int trailing,
    unsigned int table_off,
    unsigned int idx_off,
    unsigned int out_off
) {
    unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total) return;
    unsigned int t = o % trailing;
    unsigned int tmp = o / trailing;
    unsigned int k = tmp % num_idx;
    unsigned int outer_o = tmp / num_idx;
    float idx_f = arena[idx_off + k];
    unsigned int row = (unsigned int)fmaxf(idx_f, 0.0f);
    if (row >= axis_dim) row = axis_dim - 1u;
    unsigned int src = (outer_o * axis_dim + row) * trailing + t;
    unsigned int dst = (outer_o * num_idx + k) * trailing + t;
    arena[out_off + dst] = arena[table_off + src];
}
