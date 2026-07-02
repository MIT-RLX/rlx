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

// Dequant-on-the-fly matmul for Int8/Int8Asym/Int4 + FP8 (e4m3, e5m2).
// scheme_id selects the unpack path:
//   0 Int8Block        (signed i8, per-block scale)
//   1 Int8BlockAsym    (signed i8, per-block scale + zero-point)
//   2 Int4Block        (signed nibbles, per-block scale)
//   3 Fp8E4m3
//   4 Fp8E5m2
//   5 Nvfp4Block (E2M1 + FP8 block scales + f32 global_scale @ zp_off)

__device__ __forceinline__ float fp4_e2m1_lut(unsigned int nib) {
    static const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[nib & 0xf];
}

__device__ __forceinline__ unsigned int read_packed_byte(const float* arena, unsigned int off, unsigned int byte_idx) {
    unsigned int word = byte_idx / 4;
    unsigned int shift = (byte_idx % 4) * 8;
    unsigned int bits = __float_as_uint(arena[off + word]);
    return (bits >> shift) & 0xff;
}

__device__ __forceinline__ float decode_e4m3(unsigned int byte) {
    unsigned int sign = (byte >> 7) & 1;
    unsigned int exp  = (byte >> 3) & 0xf;
    unsigned int mant = byte & 0x7;
    float v;
    if (exp == 0) {
        v = ((float)mant / 8.0f) * exp2f(-6.0f);
    } else if (exp == 15 && mant == 7) {
        v = 0.0f;  // NaN coerced to 0
    } else {
        float m = 1.0f + (float)mant / 8.0f;
        v = m * exp2f((float)((int)exp - 7));
    }
    return sign ? -v : v;
}

__device__ __forceinline__ float decode_e5m2(unsigned int byte) {
    unsigned int sign = (byte >> 7) & 1;
    unsigned int exp  = (byte >> 2) & 0x1f;
    unsigned int mant = byte & 0x3;
    float v;
    if (exp == 0) {
        v = ((float)mant / 4.0f) * exp2f(-14.0f);
    } else if (exp == 31) {
        v = 0.0f;  // Inf/NaN coerced to 0
    } else {
        float m = 1.0f + (float)mant / 4.0f;
        v = m * exp2f((float)((int)exp - 15));
    }
    return sign ? -v : v;
}

extern "C" __global__ void dequant_matmul(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int block_size,
    unsigned int scheme_id,
    unsigned int x_off,
    unsigned int w_off,
    unsigned int scale_off,
    unsigned int zp_off,
    unsigned int out_off
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float acc = 0.0f;
    for (unsigned int kk = 0; kk < k; ++kk) {
        unsigned int elem_idx = kk * n + col;
        float w_dq;

        if (scheme_id == 0 || scheme_id == 1) {
            // Int8 byte stream packed 4 per f32 word.
            unsigned int word = elem_idx / 4;
            unsigned int shift = (elem_idx % 4) * 8;
            unsigned int bits = __float_as_uint(arena[w_off + word]);
            unsigned int byte = (bits >> shift) & 0xff;
            int q = (int)byte;
            if (q >= 128) q -= 256;
            unsigned int block = kk / block_size;
            float scale = arena[scale_off + block * n + col];
            float zp = (scheme_id == 1) ? arena[zp_off + block * n + col] : 0.0f;
            w_dq = ((float)q - zp) * scale;
        } else if (scheme_id == 2) {
            // Int4 nibble stream packed 8 per f32 word.
            unsigned int word = elem_idx / 8;
            unsigned int shift = (elem_idx % 8) * 4;
            unsigned int bits = __float_as_uint(arena[w_off + word]);
            unsigned int nib = (bits >> shift) & 0xf;
            int q = (int)nib;
            if (q >= 8) q -= 16;
            unsigned int block = kk / block_size;
            float scale = arena[scale_off + block * n + col];
            w_dq = (float)q * scale;
        } else if (scheme_id == 3 || scheme_id == 4) {
            // FP8 e4m3 / e5m2 — direct bit decode, no scale.
            unsigned int byte = read_packed_byte(arena, w_off, elem_idx);
            w_dq = (scheme_id == 3) ? decode_e4m3(byte) : decode_e5m2(byte);
        } else {
            // Nvfp4Block — E2M1 nibble × FP8 block scale × global f32 scale.
            unsigned int word = elem_idx / 8;
            unsigned int shift = (elem_idx % 8) * 4;
            unsigned int bits = __float_as_uint(arena[w_off + word]);
            unsigned int nib = (bits >> shift) & 0xf;
            unsigned int block = kk / block_size;
            unsigned int scale_byte = read_packed_byte(arena, scale_off, block * n + col);
            float scale = decode_e4m3(scale_byte);
            float gs = arena[zp_off];
            w_dq = fp4_e2m1_lut(nib) * scale * gs;
        }

        acc += arena[x_off + row * k + kk] * w_dq;
    }
    arena[out_off + row * n + col] = acc;
}
