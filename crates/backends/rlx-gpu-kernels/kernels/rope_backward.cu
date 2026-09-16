// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RoPE backward: dx = rope(dy, cos, -sin) on rotated pairs.
//
// `interleaved` selects the pairing convention, exactly as in the forward
// `rope.cu` (rlx_ir::op::RopeStyle): 0 = NeoX, pairing lane i with lane
// i + n_rot/2; 1 = GptJ, pairing adjacent lanes 2i and 2i+1 (the llama.cpp /
// GGUF convention). The parameter used to not exist and the kernel was NeoX
// only, so every GptJ rotation got a NeoX adjoint — the forward was right, the
// gradient was not, and cross-backend parity stayed green because every backend
// shared the omission.

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
    unsigned int cos_len,
    unsigned int cos_row_stride,
    unsigned int interleaved
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
    unsigned int rot_half = n_rot / 2u;
    // `cos_row_stride` is the cos/sin table's ACTUAL row width, passed in from
    // the table's last dimension. It is neither head_dim/2 nor n_rot/2 in
    // general: the layout is a per-model choice (Qwen3.5 pads to head_dim/2 and
    // uses the leading n_rot/2 columns; DeepSeek-V4 MLA packs n_rot/2 exactly).
    // This used to be hardcoded to `rot_half`, matching every other backward
    // kernel — so cross-backend parity passed while all of them disagreed with
    // the forward under partial rotation. Finite differences caught it.
    unsigned int tab_off = (si * cos_row_stride) % (cos_len > 0u ? cos_len : 1u);

    unsigned int dy_base = dy_off + bi * seq * hidden + si * hidden + hi * head_dim;
    unsigned int dx_base = dx_off + bi * seq * hidden + si * hidden + hi * head_dim;

    if (interleaved != 0u) {
        // GptJ: one thread per PAIR — the even lane writes both halves.
        if (d < n_rot && (d & 1u) == 0u) {
            unsigned int j = d >> 1u;
            float y1 = arena[dy_base + 2u * j];
            float y2 = arena[dy_base + 2u * j + 1u];
            float c = arena[cos_off + tab_off + j];
            float s = arena[sin_off + tab_off + j];
            arena[dx_base + 2u * j] = y1 * c + y2 * s;
            arena[dx_base + 2u * j + 1u] = -y1 * s + y2 * c;
        } else if (d >= n_rot) {
            arena[dx_base + d] = arena[dy_base + d];
        }
    } else if (d < rot_half) {
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
