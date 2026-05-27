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
// GGUF K-quant super-block dequant → f32 scratch (one 256-element block / thread).
// scheme_id: 0=Q4K, 1=Q5K, 2=Q6K, 3=Q8K, 4=Q2K, 5=Q3K.

static __device__ __forceinline__ float dq_read_f16(const unsigned char* b, unsigned int off) {
    unsigned short bits = (unsigned short)b[off] | ((unsigned short)b[off + 1u] << 8u);
    unsigned int sign = ((unsigned int)bits >> 15u) & 1u;
    unsigned int exp  = ((unsigned int)bits >> 10u) & 0x1Fu;
    unsigned int mant = (unsigned int)bits & 0x3FFu;
    float v;
    if (exp == 0u) {
        v = (float)mant / 1024.0f * exp2f(-14.0f);
    } else if (exp == 31u) {
        v = (mant == 0u) ? __int_as_float(0x7f800000) : 0.0f;
    } else {
        v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    }
    return (sign != 0u) ? -v : v;
}

static __device__ __forceinline__ void dq_get_scale_min_k4(
    const unsigned char* q, unsigned int j, unsigned int& sc, unsigned int& mn
) {
    if (j < 4u) {
        sc = (unsigned int)q[j] & 63u;
        mn = (unsigned int)q[j + 4u] & 63u;
    } else {
        sc = ((unsigned int)q[j + 4u] & 0x0Fu) | ((((unsigned int)q[j - 4u] >> 6u) & 3u) << 4u);
        mn = ((unsigned int)q[j + 4u] >> 4u) | ((((unsigned int)q[j] >> 6u) & 3u) << 4u);
    }
}

extern "C" __global__ void dequant_gguf(
    float* arena,
    unsigned int w_byte_off,
    unsigned int dst_f32_off,
    unsigned int scheme_id,
    unsigned int num_blocks
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_blocks) return;

    unsigned char* w_base = reinterpret_cast<unsigned char*>(arena) + w_byte_off;
    float* dst = arena + dst_f32_off + gid * 256u;

    if (scheme_id == 3u) {
        unsigned int off = gid * (4u + 256u + (256u / 16u) * 2u);
        float d = *reinterpret_cast<const float*>(w_base + off);
        const unsigned char* qs = w_base + off + 4u;
        for (unsigned int i = 0; i < 256u; ++i) {
            dst[i] = d * (float)(signed char)qs[i];
        }
        return;
    }

    if (scheme_id == 0u) {
        unsigned int blk = 2u + 2u + 12u + 256u / 2u;
        unsigned int off = gid * blk;
        float d = dq_read_f16(w_base, off);
        float dmin = dq_read_f16(w_base, off + 2u);
        const unsigned char* scales = w_base + off + 4u;
        const unsigned char* qs = w_base + off + 4u + 12u;
        unsigned int is = 0u;
        unsigned int out_i = 0u;
        for (unsigned int j = 0u; j < 8u; j += 2u) {
            unsigned int sc0, m0, sc1, m1;
            dq_get_scale_min_k4(scales, j, sc0, m0);
            dq_get_scale_min_k4(scales, j + 1u, sc1, m1);
            float d0 = d * (float)sc0;
            float m0f = dmin * (float)m0;
            float d1 = d * (float)sc1;
            float m1f = dmin * (float)m1;
            for (unsigned int l = 0u; l < 32u; ++l) {
                unsigned char q = qs[is + l];
                dst[out_i++] = d0 * (float)(q & 0x0Fu) - m0f;
            }
            for (unsigned int l = 0u; l < 32u; ++l) {
                unsigned char q = qs[is + l];
                dst[out_i++] = d1 * (float)(q >> 4) - m1f;
            }
            is += 32u;
        }
        return;
    }

    if (scheme_id == 1u) {
        unsigned int blk = 2u + 2u + 12u + 256u / 8u + 256u / 2u;
        unsigned int off = gid * blk;
        float d = dq_read_f16(w_base, off);
        float dmin = dq_read_f16(w_base, off + 2u);
        const unsigned char* scales = w_base + off + 4u;
        unsigned int qh_off = off + 4u + 12u;
        const unsigned char* qh = w_base + qh_off;
        const unsigned char* qs = w_base + qh_off + 256u / 8u;
        unsigned int is = 0u;
        unsigned int out_i = 0u;
        unsigned char u1 = 1u;
        unsigned char u2 = 2u;
        for (unsigned int j = 0u; j < 8u; j += 2u) {
            unsigned int sc0, m0, sc1, m1;
            dq_get_scale_min_k4(scales, j, sc0, m0);
            dq_get_scale_min_k4(scales, j + 1u, sc1, m1);
            float d0 = d * (float)sc0;
            float m0f = dmin * (float)m0;
            float d1 = d * (float)sc1;
            float m1f = dmin * (float)m1;
            for (unsigned int l = 0u; l < 32u; ++l) {
                unsigned int lo = (unsigned int)qs[is + l] & 0x0Fu;
                unsigned int hi = (qh[l] & u1) != 0u ? 16u : 0u;
                dst[out_i++] = d0 * (float)(lo + hi) - m0f;
            }
            for (unsigned int l = 0u; l < 32u; ++l) {
                unsigned int lo = (unsigned int)qs[is + l] >> 4u;
                unsigned int hi = (qh[l] & u2) != 0u ? 16u : 0u;
                dst[out_i++] = d1 * (float)(lo + hi) - m1f;
            }
            is += 32u;
            u1 <<= 2;
            u2 <<= 2;
        }
        return;
    }

    if (scheme_id == 2u) {
        unsigned int ql_len = 256u / 2u;
        unsigned int qh_len = 256u / 4u;
        unsigned int sc_len = 256u / 16u;
        unsigned int blk = ql_len + qh_len + sc_len + 2u;
        unsigned int off = gid * blk;
        const unsigned char* ql = w_base + off;
        const unsigned char* qh = w_base + off + ql_len;
        const unsigned char* sc = w_base + off + ql_len + qh_len;
        float d = dq_read_f16(w_base, off + ql_len + qh_len + sc_len);
        for (unsigned int h = 0u; h < 2u; ++h) {
            unsigned int dst_base = h * 128u;
            unsigned int ql_off = h * 64u;
            unsigned int qh_off_h = h * 32u;
            unsigned int sc_off = h * 8u;
            for (unsigned int l = 0u; l < 32u; ++l) {
                unsigned int is = l / 16u;
                unsigned char qh_b = qh[qh_off_h + l];
                float q1 = (float)(int)(((ql[ql_off + l] & 0x0Fu) | (((qh_b >> 0) & 3u) << 4u)) - 32);
                float q2 = (float)(int)(((ql[ql_off + l + 32u] & 0x0Fu) | (((qh_b >> 2) & 3u) << 4u)) - 32);
                float q3 = (float)(int)(((ql[ql_off + l] >> 4) | (((qh_b >> 4) & 3u) << 4u)) - 32);
                float q4 = (float)(int)(((ql[ql_off + l + 32u] >> 4) | (((qh_b >> 6) & 3u) << 4u)) - 32);
                dst[dst_base + l] = d * (float)(signed char)sc[sc_off + is] * q1;
                dst[dst_base + l + 32u] = d * (float)(signed char)sc[sc_off + is + 2u] * q2;
                dst[dst_base + l + 64u] = d * (float)(signed char)sc[sc_off + is + 4u] * q3;
                dst[dst_base + l + 96u] = d * (float)(signed char)sc[sc_off + is + 6u] * q4;
            }
        }
        return;
    }

    if (scheme_id == 4u) {
        unsigned int blk = 2u + 2u + 256u / 16u + 256u / 4u;
        unsigned int off = gid * blk;
        float d = dq_read_f16(w_base, off);
        float min = dq_read_f16(w_base, off + 2u);
        const unsigned char* q = w_base + off + 4u + 256u / 16u;
        unsigned int is = 0u;
        unsigned int out_i = 0u;
        for (unsigned int sb = 0u; sb < 2u; ++sb) {
            unsigned int shift = 0u;
            for (unsigned int t = 0u; t < 4u; ++t) {
                unsigned char sc = w_base[off + 4u + is];
                is += 1u;
                float dl = d * (float)(sc & 0xFu);
                float ml = min * (float)(sc >> 4);
                for (unsigned int l = 0u; l < 16u; ++l) {
                    dst[out_i++] = dl * (float)((q[l] >> shift) & 3u) - ml;
                }
                sc = w_base[off + 4u + is];
                is += 1u;
                dl = d * (float)(sc & 0xFu);
                ml = min * (float)(sc >> 4);
                for (unsigned int l = 0u; l < 16u; ++l) {
                    dst[out_i++] = dl * (float)((q[l + 16u] >> shift) & 3u) - ml;
                }
                shift += 2u;
            }
            q += 32u;
        }
        return;
    }

    if (scheme_id == 5u) {
        const unsigned int KMASK1 = 0x03030303u;
        const unsigned int KMASK2 = 0x0f0f0f0fu;
        unsigned int blk = 2u + 12u + 256u / 8u + 256u / 4u;
        unsigned int off = gid * blk;
        float d_all = dq_read_f16(w_base, off);
        const unsigned char* hm = w_base + off + 2u + 12u;
        const unsigned char* q = w_base + off + 2u + 12u + 256u / 8u;
        unsigned int aux0 = (unsigned int)w_base[off + 2u] | ((unsigned int)w_base[off + 3u] << 8u)
                          | ((unsigned int)w_base[off + 4u] << 16u) | ((unsigned int)w_base[off + 5u] << 24u);
        unsigned int aux1 = (unsigned int)w_base[off + 6u] | ((unsigned int)w_base[off + 7u] << 8u)
                          | ((unsigned int)w_base[off + 8u] << 16u) | ((unsigned int)w_base[off + 9u] << 24u);
        unsigned int aux2 = (unsigned int)w_base[off + 10u] | ((unsigned int)w_base[off + 11u] << 8u)
                          | ((unsigned int)w_base[off + 12u] << 16u) | ((unsigned int)w_base[off + 13u] << 24u);
        unsigned int tmp = aux2;
        aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        unsigned int aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        const signed char* scales = reinterpret_cast<const signed char*>(&aux0);
        unsigned int is = 0u;
        unsigned char m = 1u;
        unsigned int out_i = 0u;
        for (unsigned int sb = 0u; sb < 2u; ++sb) {
            unsigned int shift = 0u;
            for (unsigned int t = 0u; t < 4u; ++t) {
                float dl = d_all * (float)((int)scales[is] - 32);
                is += 1u;
                for (unsigned int l = 0u; l < 16u; ++l) {
                    int h = (hm[l] & m) != 0u ? 0 : 4;
                    dst[out_i++] = dl * (float)(int)(((q[l] >> shift) & 3u) - h);
                }
                dl = d_all * (float)((int)scales[is] - 32);
                is += 1u;
                for (unsigned int l = 0u; l < 16u; ++l) {
                    int h = (hm[l + 16u] & m) != 0u ? 0 : 4;
                    dst[out_i++] = dl * (float)(int)(((q[l + 16u] >> shift) & 3u) - h);
                }
                shift += 2u;
                m <<= 1;
            }
            q += 32u;
        }
    }
}
