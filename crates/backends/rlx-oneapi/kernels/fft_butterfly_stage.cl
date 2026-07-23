// Ternary-pruned radix-2 butterfly stage (interleaved C64 [batch, n_fft, 2]).
// Mirrors CUDA `fft_butterfly_stage.cu` / CPU `execute_fft_butterfly_stage_f32`.
// One work-item per (batch, butterfly); gate=0 copies the pair, else twiddle + optional rev.
__kernel void fft_butterfly_stage(__global float* arena,
                                  uint batch, uint n_fft, uint stage, uint half,
                                  uint state_off, uint out_off,
                                  uint gate_off, uint rev_off,
                                  uint tw_re_off, uint tw_im_off) {
    uint idx = get_global_id(0);
    uint n = batch * half;
    if (idx >= n) return;
    uint b = idx / half;
    uint bf = idx % half;

    uint stride = 1u << stage;
    uint row_elems = n_fft * 2u;
    uint inp_base = state_off + b * row_elems;
    uint out_base = out_off + b * row_elems;

    uint group = bf / stride;
    uint k = bf % stride;
    uint i0 = group * 2u * stride + k;
    uint i1 = i0 + stride;

    if (arena[gate_off + bf] == 0.0f) {
        arena[out_base + i0 * 2u]      = arena[inp_base + i0 * 2u];
        arena[out_base + i0 * 2u + 1u] = arena[inp_base + i0 * 2u + 1u];
        arena[out_base + i1 * 2u]      = arena[inp_base + i1 * 2u];
        arena[out_base + i1 * 2u + 1u] = arena[inp_base + i1 * 2u + 1u];
        return;
    }

    float w_re = arena[tw_re_off + bf];
    float w_im = arena[tw_im_off + bf];
    float in_a_re = arena[inp_base + i0 * 2u];
    float in_a_im = arena[inp_base + i0 * 2u + 1u];
    float in_b_re = arena[inp_base + i1 * 2u];
    float in_b_im = arena[inp_base + i1 * 2u + 1u];

    float b_re = in_b_re * w_re - in_b_im * w_im;
    float b_im = in_b_re * w_im + in_b_im * w_re;
    float top_re = in_a_re + b_re;
    float top_im = in_a_im + b_im;
    float bot_re = in_a_re - b_re;
    float bot_im = in_a_im - b_im;

    float oa_re, oa_im, ob_re, ob_im;
    if (arena[rev_off + bf] >= 0.5f) {
        oa_re = bot_re; oa_im = bot_im;
        ob_re = top_re; ob_im = top_im;
    } else {
        oa_re = top_re; oa_im = top_im;
        ob_re = bot_re; ob_im = bot_im;
    }
    arena[out_base + i0 * 2u]      = oa_re;
    arena[out_base + i0 * 2u + 1u] = oa_im;
    arena[out_base + i1 * 2u]      = ob_re;
    arena[out_base + i1 * 2u + 1u] = ob_im;
}
