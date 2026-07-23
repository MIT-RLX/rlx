// Single-layer unidirectional LSTM (gate order i, f, g, o; combined bias;
// h0=c0=0). One work-group per batch item; lane `k` owns hidden unit `k`,
// shared hidden/cell in local memory. hidden ≤ 256 (larger / multi-layer /
// bidir / carry → CPU host-fallback). Barriers sit in uniform control flow.
// Bit-exact with rlx_cpu::thunk::execute_lstm_f32 for the simple case.
// Mirrors Vulkan `lstm.comp` / OneAPI `gru.cl` launch geometry.

#define LSTM_MAX_H 256u

__kernel void lstm(__global float* arena,
                   uint batch, uint seq, uint input_size, uint hidden,
                   uint x_off, uint wih_off, uint whh_off,
                   uint bias_off, uint out_off,
                   uint seq_stride) {
    __local float h_sh[LSTM_MAX_H];
    __local float c_sh[LSTM_MAX_H];
    uint bi = get_group_id(0);
    uint k = get_local_id(0);
    uint lane_on = (bi < batch) && (k < hidden) && (hidden <= LSTM_MAX_H);

    if (k < LSTM_MAX_H) {
        h_sh[k] = 0.0f;
        c_sh[k] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint t = 0u; t < seq; t++) {
        float next_h = 0.0f;
        float next_c = 0.0f;
        if (lane_on) {
            uint x_base = x_off + (bi * seq_stride + t) * input_size;
            float zacc[4];
            for (uint g = 0u; g < 4u; g++) {
                uint r = g * hidden + k;
                float acc = arena[bias_off + r];
                uint wih_row = wih_off + r * input_size;
                for (uint j = 0u; j < input_size; j++)
                    acc += arena[wih_row + j] * arena[x_base + j];
                uint whh_row = whh_off + r * hidden;
                for (uint j = 0u; j < hidden; j++)
                    acc += arena[whh_row + j] * h_sh[j];
                zacc[g] = acc;
            }
            float ig = 1.0f / (1.0f + exp(-zacc[0]));
            float fg = 1.0f / (1.0f + exp(-zacc[1]));
            float gg = tanh(zacc[2]);
            float og = 1.0f / (1.0f + exp(-zacc[3]));
            next_c = fg * c_sh[k] + ig * gg;
            next_h = og * tanh(next_c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane_on) {
            h_sh[k] = next_h;
            c_sh[k] = next_c;
            arena[out_off + (bi * seq_stride + t) * hidden + k] = next_h;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
