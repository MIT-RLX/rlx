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

// LSTM forward. The timestep recurrence is inherently sequential, so one
// direction runs in a single threadblock (batch=1 in the common CRNN case);
// the win over the host path comes from staying on-device and, crucially, from
// COALESCED weight reads.
//
// Naive thread-per-output-row (thread r reads weight row r, stride = in_features)
// is fully uncoalesced and ~20x too slow. Instead the caller pre-transposes the
// gate weights to `[in_features, 4h]` with `transpose_rc`, so at a fixed input
// index j consecutive threads r read consecutive addresses `w_t[j*4h + r]`.
//
// Bit-exact mirror of `rlx_cpu::thunk::execute_lstm_f32`:
//   z[r] = bias[r] + sum_j wih[r,j] x_t[j] + sum_j whh[r,j] h[j]
//   i=sig(z[0..h]) f=sig(z[h..2h]) g=tanh(z[2h..3h]) o=sig(z[3h..4h])
//   c = f*c + i*g ;  h = o*tanh(c)
// (wih_t[j*4h+r] == wih[r,j], likewise whh_t — so the sums are identical.)
// Output layout [batch, seq, out_width]; this direction owns the
// `dir*hidden .. dir*hidden+hidden` feature slice. All *_off are FLOAT offsets.

// Transpose src[rows, cols] (in `arena`) into dst[cols, rows] (in `scratch`):
//   dst[c*rows + r] = src[r*cols + c].
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

extern "C" __global__ void lstm_dir(
    float* arena,
    float* scratch,
    unsigned int in_off,
    unsigned int in_is_scratch,
    unsigned int out_off,
    unsigned int out_is_scratch,
    unsigned int wih_t_off,   // transposed weights live in scratch: [in_l, 4h]
    unsigned int whh_t_off,   // [hidden, 4h]
    unsigned int bias_off,    // arena: [4h]
    unsigned int h0_off,
    unsigned int c0_off,
    unsigned int carry,
    unsigned int batch,
    unsigned int seq,
    unsigned int in_l,
    unsigned int hidden,
    unsigned int out_width,
    unsigned int dir,
    unsigned int reverse
) {
    extern __shared__ float sh[];
    float* x_sh = sh;              // in_l
    float* h_sh = x_sh + in_l;     // hidden
    float* c_sh = h_sh + hidden;   // hidden
    float* z_sh = c_sh + hidden;   // 4*hidden

    unsigned int b = blockIdx.x;
    if (b >= batch) return;
    unsigned int tid = threadIdx.x;
    unsigned int nth = blockDim.x;
    unsigned int four_h = 4u * hidden;

    float* in_base  = in_is_scratch  ? scratch : arena;
    float* out_base = out_is_scratch ? scratch : arena;
    const float* wih_t = scratch + wih_t_off;  // [in_l, 4h]
    const float* whh_t = scratch + whh_t_off;  // [hidden, 4h]
    const float* bias  = arena + bias_off;     // [4h]

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
        const float* x_t = in_base + in_off + (size_t)(b * seq + t) * in_l;
        for (unsigned int j = tid; j < in_l; j += nth) {
            x_sh[j] = x_t[j];
        }
        __syncthreads();

        // z[r] = bias[r] + sum_j wih_t[j,r] x[j] + sum_j whh_t[j,r] h[j]
        // Consecutive threads r read consecutive addresses (coalesced).
        for (unsigned int r = tid; r < four_h; r += nth) {
            float acc = bias[r];
            for (unsigned int j = 0; j < in_l; ++j) {
                acc += wih_t[(size_t)j * four_h + r] * x_sh[j];
            }
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
