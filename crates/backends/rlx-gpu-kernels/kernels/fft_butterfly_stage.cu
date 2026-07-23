// RLX — versatile ML compiler + runtime.
// Ternary-pruned radix-2 butterfly stage (interleaved C64 [batch, n_fft, 2]).
// Mirrors rlx-cpu `execute_fft_butterfly_stage_f32`: copy state → out, then for
// each bf in 0..half with gate[bf] != 0 apply the twiddle butterfly (optional
// rev swap). One block per batch row; threads stride over copy + butterflies.
// Reads butterfly inputs from the original state (not out), matching CPU.

extern "C" __global__ void fft_butterfly_stage(
    float* arena,
    unsigned state_off,
    unsigned out_off,
    unsigned gate_off,
    unsigned rev_off,
    unsigned tw_re_off,
    unsigned tw_im_off,
    unsigned batch,
    unsigned n_fft,
    unsigned stage
) {
    unsigned b = blockIdx.x;
    if (b >= batch) return;

    unsigned half = n_fft / 2u;
    unsigned stride = 1u << stage;
    unsigned row_elems = n_fft * 2u;
    float* inp = arena + state_off + b * row_elems;
    float* out = arena + out_off + b * row_elems;
    const float* gate = arena + gate_off;
    const float* rev = arena + rev_off;
    const float* tw_re = arena + tw_re_off;
    const float* tw_im = arena + tw_im_off;

    for (unsigned i = threadIdx.x; i < row_elems; i += blockDim.x) {
        out[i] = inp[i];
    }
    __syncthreads();

    for (unsigned bf = threadIdx.x; bf < half; bf += blockDim.x) {
        if (gate[bf] == 0.0f) continue;

        unsigned group = bf / stride;
        unsigned k = bf % stride;
        unsigned i0 = group * 2u * stride + k;
        unsigned i1 = i0 + stride;

        float w_re = tw_re[bf];
        float w_im = tw_im[bf];
        float in_a_re = inp[i0 * 2u];
        float in_a_im = inp[i0 * 2u + 1u];
        float in_b_re = inp[i1 * 2u];
        float in_b_im = inp[i1 * 2u + 1u];

        float b_re = in_b_re * w_re - in_b_im * w_im;
        float b_im = in_b_re * w_im + in_b_im * w_re;
        float top_re = in_a_re + b_re;
        float top_im = in_a_im + b_im;
        float bot_re = in_a_re - b_re;
        float bot_im = in_a_im - b_im;

        float oa_re, oa_im, ob_re, ob_im;
        if (rev[bf] >= 0.5f) {
            oa_re = bot_re;
            oa_im = bot_im;
            ob_re = top_re;
            ob_im = top_im;
        } else {
            oa_re = top_re;
            oa_im = top_im;
            ob_re = bot_re;
            ob_im = bot_im;
        }
        out[i0 * 2u] = oa_re;
        out[i0 * 2u + 1u] = oa_im;
        out[i1 * 2u] = ob_re;
        out[i1 * 2u + 1u] = ob_im;
    }
}
