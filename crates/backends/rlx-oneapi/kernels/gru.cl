// Single-layer unidirectional GRU (gate order r, z, n; linear_before_reset=1;
// separate b_ih/b_hh; h0 = 0). One work-group per batch item; lane `k` owns
// hidden unit `k`, shared hidden in local memory. hidden ≤ 256 (larger /
// multi-layer / bidir / carry → CPU host-fallback). Barriers sit in uniform
// control flow. Mirrors wgpu `gru.wgsl` / CUDA `gru.cu`.

#define GRU_MAX_H 256u

__kernel void gru(__global float* arena,
                  uint batch, uint seq, uint input_size, uint hidden,
                  uint x_off, uint wih_off, uint whh_off,
                  uint bih_off, uint bhh_off, uint out_off,
                  uint seq_stride) {
    __local float h_sh[GRU_MAX_H];
    uint bi = get_group_id(0);
    uint k = get_local_id(0);
    uint lane_on = (bi < batch) && (k < hidden) && (hidden <= GRU_MAX_H);

    if (k < GRU_MAX_H) h_sh[k] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = 0u; t < seq; t++) {
        float h_k = 0.0f;
        if (lane_on) {
            uint x_base = x_off + (bi * seq_stride + t) * input_size;
            float xi[3], hipart[3];
            for (uint g = 0u; g < 3u; g++) {
                uint r = g * hidden + k;
                float ax = arena[bih_off + r];
                uint wih_row = wih_off + r * input_size;
                for (uint j = 0u; j < input_size; j++)
                    ax += arena[wih_row + j] * arena[x_base + j];
                float ah = arena[bhh_off + r];
                uint whh_row = whh_off + r * hidden;
                for (uint j = 0u; j < hidden; j++)
                    ah += arena[whh_row + j] * h_sh[j];
                xi[g] = ax;
                hipart[g] = ah;
            }
            float rg = 1.0f / (1.0f + exp(-(xi[0] + hipart[0])));
            float zg = 1.0f / (1.0f + exp(-(xi[1] + hipart[1])));
            float ng = tanh(xi[2] + rg * hipart[2]);
            h_k = (1.0f - zg) * ng + zg * h_sh[k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane_on) {
            h_sh[k] = h_k;
            arena[out_off + (bi * seq_stride + t) * hidden + k] = h_k;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
