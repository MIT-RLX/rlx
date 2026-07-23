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

// LSTM forward. The recurrent Whh·h step is sequential, but Wih·X across the
// whole sequence is a fat GEMV that we precompute in parallel (`lstm_pre_wih`)
// so the per-step kernel only pays for Whh·h + gates.
//
// Weight layout after `transpose_rc`: wih_t [in_l, 4h], whh_t [hidden, 4h]
// (coalesced: consecutive threads r read consecutive addresses at fixed j).
//
// Bit-exact intent of `rlx_cpu::thunk::execute_lstm_f32` (FP assoc may differ
// slightly because bias+Wih are folded before Whh):
//   z[r] = bias[r] + sum_j wih[r,j] x_t[j] + sum_j whh[r,j] h[j]
//   i=sig(z[0..h]) f=sig(z[h..2h]) g=tanh(z[2h..3h]) o=sig(z[3h..4h])
//   c = f*c + i*g ;  h = o*tanh(c)
// Output layout [batch, seq, out_width]; this direction owns the
// `dir*hidden .. dir*hidden+hidden` feature slice. All *_off are FLOAT offsets.

extern "C" __global__ void transpose_rc(
    const float* arena,
    float* scratch,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (idx >= total) return;
    unsigned int r = idx / cols;
    unsigned int c = idx - r * cols;
    scratch[dst_off + (size_t)c * rows + r] = arena[src_off + (size_t)r * cols + c];
}

extern "C" __global__ void lstm_pre_wih(
    float* arena,
    float* scratch,
    unsigned int in_off,
    unsigned int in_is_scratch,
    unsigned int wih_t_off,
    unsigned int bias_off,
    unsigned int pre_off,
    unsigned int batch,
    unsigned int seq,
    unsigned int in_l,
    unsigned int four_h
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * seq * four_h;
    if (idx >= total) return;
    unsigned int r = idx % four_h;
    unsigned int t = (idx / four_h) % seq;
    unsigned int b = idx / (four_h * seq);
    float* in_base = in_is_scratch ? scratch : arena;
    const float* x_t = in_base + in_off + (size_t)(b * seq + t) * in_l;
    const float* wih_t = scratch + wih_t_off;
    float acc = arena[bias_off + r];
    for (unsigned int j = 0; j < in_l; ++j) {
        acc += x_t[j] * wih_t[(size_t)j * four_h + r];
    }
    scratch[pre_off + (size_t)(b * seq + t) * four_h + r] = acc;
}

extern "C" __global__ void lstm_pre_add_bias(
    float* scratch,
    unsigned int pre_off,
    unsigned int bias_off,
    float* arena,
    unsigned int len,
    unsigned int four_h
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    scratch[pre_off + i] += arena[bias_off + (i % four_h)];
}

extern "C" __global__ void lstm_dir(
    float* arena,
    float* scratch,
    unsigned int in_off,
    unsigned int in_is_scratch,
    unsigned int out_off,
    unsigned int out_is_scratch,
    unsigned int wih_t_off,
    unsigned int whh_t_off,
    unsigned int bias_off,
    unsigned int h0_off,
    unsigned int c0_off,
    unsigned int carry,
    unsigned int batch,
    unsigned int seq,
    unsigned int in_l,
    unsigned int hidden,
    unsigned int out_width,
    unsigned int dir,
    unsigned int reverse,
    unsigned int pre_off
) {
    (void)in_off; (void)in_is_scratch; (void)wih_t_off; (void)bias_off; (void)in_l;
    extern __shared__ float sh[];
    float* h_sh = sh;
    float* c_sh = h_sh + hidden;
    float* z_sh = c_sh + hidden;

    unsigned int b = blockIdx.x;
    if (b >= batch) return;
    unsigned int tid = threadIdx.x;
    unsigned int nth = blockDim.x;
    unsigned int four_h = 4u * hidden;

    float* out_base = out_is_scratch ? scratch : arena;
    const float* whh_t = scratch + whh_t_off;
    const float* pre = scratch + pre_off;

    for (unsigned int k = tid; k < hidden; k += nth) {
        if (carry) {
            h_sh[k] = arena[h0_off + b * hidden + k];
            c_sh[k] = arena[c0_off + b * hidden + k];
        } else {
            h_sh[k] = 0.0f;
            c_sh[k] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int step = 0; step < seq; ++step) {
        unsigned int t = reverse ? (seq - 1u - step) : step;
        const float* pre_t = pre + (size_t)(b * seq + t) * four_h;

        for (unsigned int r = tid; r < four_h; r += nth) {
            float acc = pre_t[r];
#pragma unroll 8
            for (unsigned int j = 0; j < hidden; ++j) {
                acc += whh_t[(size_t)j * four_h + r] * h_sh[j];
            }
            z_sh[r] = acc;
        }
        __syncthreads();

        for (unsigned int k = tid; k < hidden; k += nth) {
            float ig = 1.0f / (1.0f + expf(-z_sh[k]));
            float fg = 1.0f / (1.0f + expf(-z_sh[hidden + k]));
            float gg = tanhf(z_sh[2u * hidden + k]);
            float og = 1.0f / (1.0f + expf(-z_sh[3u * hidden + k]));
            float c_new = fg * c_sh[k] + ig * gg;
            c_sh[k] = c_new;
            float h_new = og * tanhf(c_new);
            h_sh[k] = h_new;
            out_base[out_off + (size_t)(b * seq + t) * out_width + dir * hidden + k] = h_new;
        }
        __syncthreads();
    }

    if (carry) {
        for (unsigned int k = tid; k < hidden; k += nth) {
            arena[h0_off + b * hidden + k] = h_sh[k];
            arena[c0_off + b * hidden + k] = c_sh[k];
        }
    }
}
