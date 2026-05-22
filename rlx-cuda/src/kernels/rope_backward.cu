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
// NeoX RoPE backward: dx = rope(dy, cos, -sin) on rotated pairs.

extern "C" __global__ void rlx_rope_bwd(
    float* arena,
    unsigned int batch,
    unsigned int seq,
    unsigned int hidden,
    unsigned int head_dim,
    unsigned int n_rot,
    unsigned int dy_off,
    unsigned int cos_off,
    unsigned int sin_off,
    unsigned int dx_off,
    unsigned int cos_len
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * seq * hidden;
    if (i >= total) return;

    unsigned int nh = hidden / head_dim;
    unsigned int d = i % head_dim;
    unsigned int q1 = i / head_dim;
    unsigned int hi = q1 % nh;
    unsigned int q2 = q1 / nh;
    unsigned int si = q2 % seq;
    unsigned int bi = q2 / seq;
    unsigned int half_dh = head_dim / 2u;
    unsigned int rot_half = n_rot / 2u;
    unsigned int tab_off = (si * half_dh) % cos_len;

    unsigned int dy_base = dy_off + bi * seq * hidden + si * hidden + hi * head_dim;
    unsigned int dx_base = dx_off + bi * seq * hidden + si * hidden + hi * head_dim;

    if (d < rot_half) {
        float y1 = arena[dy_base + d];
        float y2 = arena[dy_base + rot_half + d];
        float c = arena[cos_off + tab_off + d];
        float s = arena[sin_off + tab_off + d];
        arena[dx_base + d] = y1 * c + y2 * s;
        arena[dx_base + rot_half + d] = -y1 * s + y2 * c;
    } else if (d >= n_rot) {
        arena[dx_base + d] = arena[dy_base + d];
    }
}
