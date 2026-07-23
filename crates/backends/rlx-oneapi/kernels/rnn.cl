// Single-layer unidirectional Elman RNN (`relu`!=0 ? relu : tanh; h0 = 0).
// One work-group per batch item; lane `k` owns hidden unit `k`. hidden ≤ 256
// (else host fallback). Mirrors wgpu `rnn.wgsl` / CUDA `rnn.cu`.

#define RNN_MAX_H 256u

__kernel void rnn(__global float* arena,
                  uint batch, uint seq, uint input_size, uint hidden,
                  uint x_off, uint wih_off, uint whh_off,
                  uint bias_off, uint out_off,
                  uint seq_stride, uint relu) {
    __local float h_sh[RNN_MAX_H];
    uint bi = get_group_id(0);
    uint k = get_local_id(0);
    uint lane_on = (bi < batch) && (k < hidden) && (hidden <= RNN_MAX_H);

    if (k < RNN_MAX_H) h_sh[k] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = 0u; t < seq; t++) {
        float h_k = 0.0f;
        if (lane_on) {
            uint x_base = x_off + (bi * seq_stride + t) * input_size;
            float acc = arena[bias_off + k];
            uint wih_row = wih_off + k * input_size;
            for (uint j = 0u; j < input_size; j++)
                acc += arena[wih_row + j] * arena[x_base + j];
            uint whh_row = whh_off + k * hidden;
            for (uint j = 0u; j < hidden; j++)
                acc += arena[whh_row + j] * h_sh[j];
            h_k = (relu != 0u) ? fmax(acc, 0.0f) : tanh(acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane_on) {
            h_sh[k] = h_k;
            arena[out_off + (bi * seq_stride + t) * hidden + k] = h_k;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
