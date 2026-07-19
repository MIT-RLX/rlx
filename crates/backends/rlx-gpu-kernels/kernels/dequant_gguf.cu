// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPL-3.0-only. See LICENSE.
//
// GGUF dequant — one thread per GGUF block. Block size varies per scheme;
// see rlx_metal/src/dequant_gguf.msl for the scheme_id table (ids 0–23,
// mirrored byte-for-byte here). Backend matrix: docs/gguf-backend-paths.md
//
// IQ-family schemes (12..=18) read grid tables from `iq_lut`. Same
// concatenated layout as the Metal kernel — see IQ_GRID_OFF_* constants.

static __constant__ int kvalues_iq4nl_lut_d[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

#define IQ_GRID_OFF_KMASK     0u
#define IQ_GRID_OFF_KSIGNS    8u
#define IQ_GRID_OFF_IQ2XXS    136u
#define IQ_GRID_OFF_IQ2XS     2184u
#define IQ_GRID_OFF_IQ2S      6280u
#define IQ_GRID_OFF_IQ3XXS    14472u
#define IQ_GRID_OFF_IQ3S      15496u
#define IQ_GRID_OFF_IQ1S      17544u

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

static __device__ __forceinline__ int lut_i8(const unsigned char* lut, unsigned int off) {
    return (int)(signed char)lut[off];
}
static __device__ __forceinline__ unsigned char lut_u8(const unsigned char* lut, unsigned int off) {
    return lut[off];
}
static __device__ __forceinline__ unsigned int lut_u16(const unsigned char* lut, unsigned int off) {
    return (unsigned int)lut[off] | ((unsigned int)lut[off + 1u] << 8u);
}

extern "C" __global__ void dequant_gguf(
    float* arena,
    // 64-bit arena offsets: a packed 27B (Bonsai Q1_0) arena exceeds 4 GB, so
    // a `unsigned int` byte offset overflows and the kernel reads/writes the
    // wrong slot (garbage logits). Matches the vulkan/wgpu >4 GiB u32-offset
    // fix. Host passes these as u64 (see gguf_gpu.rs).
    unsigned long long w_byte_off,
    unsigned long long dst_f32_off,
    unsigned int scheme_id,
    unsigned int num_blocks,
    const unsigned char* iq_lut
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_blocks) return;

    unsigned char* w_base = reinterpret_cast<unsigned char*>(arena) + w_byte_off;

    // ── 256-elem block schemes ──
    if (scheme_id != 6u && scheme_id != 10u && scheme_id != 11u
        && scheme_id != 19u && scheme_id != 20u
        && scheme_id != 21u && scheme_id != 22u && scheme_id != 23u
        && scheme_id != 24u) {
        float* dst = arena + dst_f32_off + gid * 256u;

        if (scheme_id == 3u) {
            unsigned int off = gid * (4u + 256u + (256u / 16u) * 2u);
            float d = *reinterpret_cast<const float*>(w_base + off);
            const unsigned char* qs = w_base + off + 4u;
            for (unsigned int i = 0; i < 256u; ++i) dst[i] = d * (float)(signed char)qs[i];
            return;
        }

        if (scheme_id == 0u) {
            unsigned int blk = 2u + 2u + 12u + 256u / 2u;
            unsigned int off = gid * blk;
            float d = dq_read_f16(w_base, off);
            float dmin = dq_read_f16(w_base, off + 2u);
            const unsigned char* scales = w_base + off + 4u;
            const unsigned char* qs = w_base + off + 4u + 12u;
            unsigned int is = 0u, out_i = 0u;
            for (unsigned int j = 0u; j < 8u; j += 2u) {
                unsigned int sc0, m0, sc1, m1;
                dq_get_scale_min_k4(scales, j, sc0, m0);
                dq_get_scale_min_k4(scales, j + 1u, sc1, m1);
                float d0 = d * (float)sc0, m0f = dmin * (float)m0;
                float d1 = d * (float)sc1, m1f = dmin * (float)m1;
                for (unsigned int l = 0u; l < 32u; ++l) dst[out_i++] = d0 * (float)(qs[is + l] & 0x0Fu) - m0f;
                for (unsigned int l = 0u; l < 32u; ++l) dst[out_i++] = d1 * (float)(qs[is + l] >> 4) - m1f;
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
            unsigned int is = 0u, out_i = 0u;
            unsigned char u1 = 1u, u2 = 2u;
            for (unsigned int j = 0u; j < 8u; j += 2u) {
                unsigned int sc0, m0, sc1, m1;
                dq_get_scale_min_k4(scales, j, sc0, m0);
                dq_get_scale_min_k4(scales, j + 1u, sc1, m1);
                float d0 = d * (float)sc0, m0f = dmin * (float)m0;
                float d1 = d * (float)sc1, m1f = dmin * (float)m1;
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
                is += 32u; u1 <<= 2; u2 <<= 2;
            }
            return;
        }

        if (scheme_id == 2u) {
            unsigned int ql_len = 256u / 2u, qh_len = 256u / 4u, sc_len = 256u / 16u;
            unsigned int blk = ql_len + qh_len + sc_len + 2u;
            unsigned int off = gid * blk;
            const unsigned char* ql = w_base + off;
            const unsigned char* qh = w_base + off + ql_len;
            const unsigned char* sc = w_base + off + ql_len + qh_len;
            float d = dq_read_f16(w_base, off + ql_len + qh_len + sc_len);
            for (unsigned int h = 0u; h < 2u; ++h) {
                unsigned int dst_base = h * 128u, ql_off = h * 64u, qh_off_h = h * 32u, sc_off = h * 8u;
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
            // Q2_K: scales[16] | qs[64] | d (f16) | dmin (f16).
            unsigned int blk = 2u + 2u + 256u / 16u + 256u / 4u;
            unsigned int off = gid * blk;
            unsigned int scales_off = off, qs_off = off + 256u / 16u, d_off = qs_off + 256u / 4u;
            float d = dq_read_f16(w_base, d_off);
            float min_v = dq_read_f16(w_base, d_off + 2u);
            const unsigned char* q = w_base + qs_off;
            unsigned int is = 0u, out_i = 0u;
            for (unsigned int sb = 0u; sb < 2u; ++sb) {
                unsigned int shift = 0u;
                for (unsigned int t = 0u; t < 4u; ++t) {
                    unsigned char sc = w_base[scales_off + is]; is += 1u;
                    float dl = d * (float)(sc & 0xFu); float ml = min_v * (float)(sc >> 4);
                    for (unsigned int l = 0u; l < 16u; ++l) dst[out_i++] = dl * (float)((q[l] >> shift) & 3u) - ml;
                    sc = w_base[scales_off + is]; is += 1u;
                    dl = d * (float)(sc & 0xFu); ml = min_v * (float)(sc >> 4);
                    for (unsigned int l = 0u; l < 16u; ++l) dst[out_i++] = dl * (float)((q[l + 16u] >> shift) & 3u) - ml;
                    shift += 2u;
                }
                q += 32u;
            }
            return;
        }

        if (scheme_id == 5u) {
            // Q3_K: hmask[32] | qs[64] | scales[12] | d (f16).
            const unsigned int KMASK1 = 0x03030303u;
            const unsigned int KMASK2 = 0x0f0f0f0fu;
            unsigned int blk = 2u + 12u + 256u / 8u + 256u / 4u;
            unsigned int off = gid * blk;
            unsigned int hm_off = off, qs_off = off + 256u / 8u, scales_off = qs_off + 256u / 4u, d_off = scales_off + 12u;
            float d_all = dq_read_f16(w_base, d_off);
            const unsigned char* hm = w_base + hm_off;
            const unsigned char* q = w_base + qs_off;
            unsigned int aux0 = (unsigned int)w_base[scales_off + 0u] | ((unsigned int)w_base[scales_off + 1u] << 8u)
                              | ((unsigned int)w_base[scales_off + 2u] << 16u) | ((unsigned int)w_base[scales_off + 3u] << 24u);
            unsigned int aux1 = (unsigned int)w_base[scales_off + 4u] | ((unsigned int)w_base[scales_off + 5u] << 8u)
                              | ((unsigned int)w_base[scales_off + 6u] << 16u) | ((unsigned int)w_base[scales_off + 7u] << 24u);
            unsigned int aux2 = (unsigned int)w_base[scales_off + 8u] | ((unsigned int)w_base[scales_off + 9u] << 8u)
                              | ((unsigned int)w_base[scales_off + 10u] << 16u) | ((unsigned int)w_base[scales_off + 11u] << 24u);
            unsigned int tmp = aux2;
            aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
            aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            const signed char* scales = reinterpret_cast<const signed char*>(&aux0);
            unsigned int is = 0u; unsigned char m = 1u; unsigned int out_i = 0u;
            for (unsigned int sb = 0u; sb < 2u; ++sb) {
                unsigned int shift = 0u;
                for (unsigned int t = 0u; t < 4u; ++t) {
                    float dl = d_all * (float)((int)scales[is] - 32); is += 1u;
                    for (unsigned int l = 0u; l < 16u; ++l) {
                        int h = (hm[l] & m) != 0u ? 0 : 4;
                        dst[out_i++] = dl * (float)(int)(((q[l] >> shift) & 3u) - h);
                    }
                    dl = d_all * (float)((int)scales[is] - 32); is += 1u;
                    for (unsigned int l = 0u; l < 16u; ++l) {
                        int h = (hm[l + 16u] & m) != 0u ? 0 : 4;
                        dst[out_i++] = dl * (float)(int)(((q[l + 16u] >> shift) & 3u) - h);
                    }
                    shift += 2u; m <<= 1;
                }
                q += 32u;
            }
            return;
        }

        if (scheme_id == 7u) {
            // IQ4_XS.
            unsigned int blk = 2u + 2u + 256u / 64u + 256u / 2u;
            unsigned int off = gid * blk;
            float d = dq_read_f16(w_base, off);
            unsigned int scales_h = (unsigned int)w_base[off + 2u] | ((unsigned int)w_base[off + 3u] << 8u);
            const unsigned char* scales_l = w_base + off + 4u;
            const unsigned char* qs = w_base + off + 4u + 256u / 64u;
            unsigned int out_i = 0u, qs_idx = 0u;
            for (unsigned int ib = 0u; ib < 256u / 32u; ++ib) {
                unsigned int lo = ((unsigned int)scales_l[ib / 2u] >> (4u * (ib % 2u))) & 0xFu;
                unsigned int hi = (scales_h >> (2u * ib)) & 0x3u;
                int ls = (int)(lo | (hi << 4u));
                float dl = d * (float)(ls - 32);
                for (unsigned int j = 0u; j < 16u; ++j) {
                    unsigned char bx = qs[qs_idx + j];
                    dst[out_i + j] = dl * (float)kvalues_iq4nl_lut_d[bx & 0x0Fu];
                    dst[out_i + j + 16u] = dl * (float)kvalues_iq4nl_lut_d[bx >> 4];
                }
                out_i += 32u; qs_idx += 16u;
            }
            return;
        }

        if (scheme_id == 8u) {
            // TQ1_0.
            const unsigned char POW3[5] = { 1, 3, 9, 27, 81 };
            unsigned int off = gid * 54u;
            const unsigned char* qs = w_base + off;
            const unsigned char* qh = w_base + off + 48u;
            float d = dq_read_f16(w_base, off + 52u);
            unsigned int y = 0u;
            for (unsigned int n = 0u; n < 5u; ++n) {
                for (unsigned int m = 0u; m < 32u; ++m) {
                    unsigned char q = (unsigned char)((unsigned int)qs[m] * (unsigned int)POW3[n]);
                    int xi = (int)(((unsigned int)q * 3u) >> 8u);
                    dst[y++] = (float)(xi - 1) * d;
                }
            }
            for (unsigned int n = 0u; n < 5u; ++n) {
                for (unsigned int m = 0u; m < 16u; ++m) {
                    unsigned char q = (unsigned char)((unsigned int)qs[32u + m] * (unsigned int)POW3[n]);
                    int xi = (int)(((unsigned int)q * 3u) >> 8u);
                    dst[y++] = (float)(xi - 1) * d;
                }
            }
            for (unsigned int n = 0u; n < 4u; ++n) {
                for (unsigned int j = 0u; j < 4u; ++j) {
                    unsigned char q = (unsigned char)((unsigned int)qh[j] * (unsigned int)POW3[n]);
                    int xi = (int)(((unsigned int)q * 3u) >> 8u);
                    dst[y++] = (float)(xi - 1) * d;
                }
            }
            return;
        }

        if (scheme_id == 9u) {
            // TQ2_0.
            unsigned int off = gid * 66u;
            const unsigned char* qs = w_base + off;
            float d = dq_read_f16(w_base, off + 64u);
            unsigned int y = 0u;
            for (unsigned int j = 0u; j < 64u; j += 32u) {
                for (unsigned int l = 0u; l < 4u; ++l) {
                    for (unsigned int m = 0u; m < 32u; ++m) {
                        int q = (int)((qs[j + m] >> (l * 2u)) & 3u);
                        dst[y++] = (float)(q - 1) * d;
                    }
                }
            }
            return;
        }

        if (scheme_id == 12u) {
            unsigned int off = gid * 66u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            unsigned int y = 0u;
            for (unsigned int ib32 = 0u; ib32 < 8u; ++ib32) {
                unsigned int base = 8u * ib32;
                unsigned int aux32_0 = (unsigned int)qs[base + 0u] | ((unsigned int)qs[base + 1u] << 8u)
                                     | ((unsigned int)qs[base + 2u] << 16u) | ((unsigned int)qs[base + 3u] << 24u);
                unsigned int aux32_1 = (unsigned int)qs[base + 4u] | ((unsigned int)qs[base + 5u] << 8u)
                                     | ((unsigned int)qs[base + 6u] << 16u) | ((unsigned int)qs[base + 7u] << 24u);
                float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;
                unsigned char grid_bytes[4] = {
                    (unsigned char)(aux32_0 & 0xFFu),
                    (unsigned char)((aux32_0 >> 8) & 0xFFu),
                    (unsigned char)((aux32_0 >> 16) & 0xFFu),
                    (unsigned char)((aux32_0 >> 24) & 0xFFu),
                };
                for (unsigned int l = 0u; l < 4u; ++l) {
                    unsigned int grid_off = IQ_GRID_OFF_IQ2XXS + (unsigned int)grid_bytes[l] * 8u;
                    unsigned char signs = lut_u8(iq_lut, IQ_GRID_OFF_KSIGNS + ((aux32_1 >> (7u * l)) & 127u));
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        int gv = lut_i8(iq_lut, grid_off + j);
                        unsigned char mask = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j);
                        float s = (signs & mask) != 0u ? -1.0f : 1.0f;
                        dst[y + j] = db * (float)gv * s;
                    }
                    y += 8u;
                }
            }
            return;
        }

        if (scheme_id == 13u) {
            unsigned int off = gid * 74u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            const unsigned char* scales = w_base + off + 2u + 64u;
            unsigned int y = 0u;
            for (unsigned int ib32 = 0u; ib32 < 8u; ++ib32) {
                float db0 = d * (0.5f + (float)(scales[ib32] & 0xFu)) * 0.25f;
                float db1 = d * (0.5f + (float)(scales[ib32] >> 4u)) * 0.25f;
                for (unsigned int l = 0u; l < 4u; ++l) {
                    unsigned int pos = (4u * ib32 + l) * 2u;
                    unsigned int q = (unsigned int)qs[pos] | ((unsigned int)qs[pos + 1u] << 8u);
                    unsigned int grid_off = IQ_GRID_OFF_IQ2XS + (q & 511u) * 8u;
                    unsigned char signs = lut_u8(iq_lut, IQ_GRID_OFF_KSIGNS + (q >> 9));
                    float dl = (l / 2u == 0u) ? db0 : db1;
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        int gv = lut_i8(iq_lut, grid_off + j);
                        unsigned char mask = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j);
                        float s = (signs & mask) != 0u ? -1.0f : 1.0f;
                        dst[y + j] = dl * (float)gv * s;
                    }
                    y += 8u;
                }
            }
            return;
        }

        if (scheme_id == 14u) {
            unsigned int off = gid * 82u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            const unsigned char* qh = w_base + off + 2u + 64u;
            const unsigned char* scales = w_base + off + 2u + 64u + 8u;
            unsigned int y = 0u, qs_idx = 0u, signs_idx = 32u;
            for (unsigned int ib32 = 0u; ib32 < 8u; ++ib32) {
                float db0 = d * (0.5f + (float)(scales[ib32] & 0xFu)) * 0.25f;
                float db1 = d * (0.5f + (float)(scales[ib32] >> 4u)) * 0.25f;
                for (unsigned int l = 0u; l < 4u; ++l) {
                    float dl = (l / 2u == 0u) ? db0 : db1;
                    unsigned int q = (unsigned int)qs[qs_idx + l];
                    unsigned int qh_b = (unsigned int)qh[ib32];
                    unsigned int idx = q | ((qh_b << (8u - 2u * l)) & 0x300u);
                    unsigned int grid_off = IQ_GRID_OFF_IQ2S + idx * 8u;
                    unsigned char sign = qs[signs_idx + l];
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        int gv = lut_i8(iq_lut, grid_off + j);
                        unsigned char mask = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j);
                        float s = (sign & mask) != 0u ? -1.0f : 1.0f;
                        dst[y + j] = dl * (float)gv * s;
                    }
                    y += 8u;
                }
                qs_idx += 4u; signs_idx += 4u;
            }
            return;
        }

        if (scheme_id == 15u) {
            unsigned int off = gid * 98u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            const unsigned char* sas = w_base + off + 2u + 64u;
            unsigned int y = 0u, qs_idx = 0u;
            for (unsigned int ib32 = 0u; ib32 < 8u; ++ib32) {
                unsigned int aux32 = (unsigned int)sas[4u*ib32] | ((unsigned int)sas[4u*ib32+1u] << 8u)
                                   | ((unsigned int)sas[4u*ib32+2u] << 16u) | ((unsigned int)sas[4u*ib32+3u] << 24u);
                float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
                for (unsigned int l = 0u; l < 4u; ++l) {
                    unsigned char signs = lut_u8(iq_lut, IQ_GRID_OFF_KSIGNS + ((aux32 >> (7u * l)) & 127u));
                    unsigned int g1_off = IQ_GRID_OFF_IQ3XXS + (unsigned int)qs[qs_idx + 2u * l] * 4u;
                    unsigned int g2_off = IQ_GRID_OFF_IQ3XXS + (unsigned int)qs[qs_idx + 2u * l + 1u] * 4u;
                    for (unsigned int j = 0u; j < 4u; ++j) {
                        int g1 = lut_i8(iq_lut, g1_off + j);
                        int g2 = lut_i8(iq_lut, g2_off + j);
                        unsigned char m0 = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j);
                        unsigned char m1 = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j + 4u);
                        float s0 = (signs & m0) != 0u ? -1.0f : 1.0f;
                        float s1 = (signs & m1) != 0u ? -1.0f : 1.0f;
                        dst[y + j] = db * (float)g1 * s0;
                        dst[y + j + 4u] = db * (float)g2 * s1;
                    }
                    y += 8u;
                }
                qs_idx += 8u;
            }
            return;
        }

        if (scheme_id == 16u) {
            unsigned int off = gid * 110u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            const unsigned char* qh = w_base + off + 2u + 64u;
            const unsigned char* signs = w_base + off + 2u + 64u + 8u;
            const unsigned char* scales = w_base + off + 2u + 64u + 8u + 32u;
            unsigned int y = 0u, qs_walk = 0u, signs_walk = 0u, qh_walk = 0u;
            for (unsigned int ib32 = 0u; ib32 < 8u; ib32 += 2u) {
                float db1 = d * (1.0f + 2.0f * (float)(scales[ib32 / 2u] & 0xFu));
                float db2 = d * (1.0f + 2.0f * (float)(scales[ib32 / 2u] >> 4u));
                for (unsigned int half_iter = 0u; half_iter < 2u; ++half_iter) {
                    float dl = (half_iter == 0u) ? db1 : db2;
                    unsigned int qh_byte = (unsigned int)qh[qh_walk + half_iter];
                    for (unsigned int l = 0u; l < 4u; ++l) {
                        unsigned int idx1 = (unsigned int)qs[qs_walk + 2u * l] | (((qh_byte << (8u - 2u * l)) & 256u));
                        unsigned int idx2 = (unsigned int)qs[qs_walk + 2u * l + 1u] | (((qh_byte << (7u - 2u * l)) & 256u));
                        unsigned int g1_off = IQ_GRID_OFF_IQ3S + idx1 * 4u;
                        unsigned int g2_off = IQ_GRID_OFF_IQ3S + idx2 * 4u;
                        unsigned char sign = signs[signs_walk + l];
                        for (unsigned int j = 0u; j < 4u; ++j) {
                            int g1 = lut_i8(iq_lut, g1_off + j);
                            int g2 = lut_i8(iq_lut, g2_off + j);
                            unsigned char m0 = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j);
                            unsigned char m1 = lut_u8(iq_lut, IQ_GRID_OFF_KMASK + j + 4u);
                            float s0 = (sign & m0) != 0u ? -1.0f : 1.0f;
                            float s1 = (sign & m1) != 0u ? -1.0f : 1.0f;
                            dst[y + j] = dl * (float)g1 * s0;
                            dst[y + j + 4u] = dl * (float)g2 * s1;
                        }
                        y += 8u;
                    }
                    qs_walk += 8u; signs_walk += 4u;
                }
                qh_walk += 2u;
            }
            return;
        }

        if (scheme_id == 17u) {
            unsigned int off = gid * 50u;
            float d = dq_read_f16(w_base, off);
            const unsigned char* qs = w_base + off + 2u;
            const unsigned char* qh_bytes = w_base + off + 2u + 32u;
            unsigned int y = 0u, qs_idx = 0u;
            for (unsigned int ib = 0u; ib < 8u; ++ib) {
                unsigned int qh = (unsigned int)qh_bytes[2u * ib] | ((unsigned int)qh_bytes[2u * ib + 1u] << 8u);
                float dl = d * (2.0f * (float)((qh >> 12) & 7u) + 1.0f);
                float delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
                for (unsigned int l = 0u; l < 4u; ++l) {
                    unsigned int idx = (unsigned int)qs[qs_idx + l] | (((qh >> (3u * l)) & 7u) << 8u);
                    unsigned int grid_off = IQ_GRID_OFF_IQ1S + idx * 8u;
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        int gv = lut_i8(iq_lut, grid_off + j);
                        dst[y + j] = dl * ((float)gv + delta);
                    }
                    y += 8u;
                }
                qs_idx += 4u;
            }
            return;
        }

        if (scheme_id == 18u) {
            unsigned int off = gid * 56u;
            const unsigned char* qs = w_base + off;
            const unsigned char* qh = w_base + off + 32u;
            const unsigned char* scales_b = w_base + off + 48u;
            unsigned int sc0 = lut_u16(scales_b, 0u);
            unsigned int sc1 = lut_u16(scales_b, 2u);
            unsigned int sc2 = lut_u16(scales_b, 4u);
            unsigned int sc3 = lut_u16(scales_b, 6u);
            unsigned int sc_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u)
                                | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
            unsigned int sign = (sc_u16 >> 15) & 1u;
            unsigned int exp_v = (sc_u16 >> 10) & 0x1Fu;
            unsigned int mant = sc_u16 & 0x3FFu;
            float d;
            if (exp_v == 0u) d = (float)mant / 1024.0f * exp2f(-14.0f);
            else if (exp_v == 31u) d = (mant == 0u) ? __int_as_float(0x7f800000) : 0.0f;
            else d = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp_v - 15));
            if (sign != 0u) d = -d;
            unsigned int sc_arr[4] = { sc0, sc1, sc2, sc3 };
            unsigned int y = 0u, qs_walk = 0u, qh_walk = 0u;
            for (unsigned int ib = 0u; ib < 8u; ++ib) {
                float dl1 = d * (2.0f * (float)((sc_arr[ib / 2u] >> (6u * (ib % 2u))) & 0x7u) + 1.0f);
                float dl2 = d * (2.0f * (float)((sc_arr[ib / 2u] >> (6u * (ib % 2u) + 3u)) & 0x7u) + 1.0f);
                unsigned int idx[4];
                idx[0] = (unsigned int)qs[qs_walk] | (((unsigned int)qh[qh_walk] << 8) & 0x700u);
                idx[1] = (unsigned int)qs[qs_walk + 1u] | (((unsigned int)qh[qh_walk] << 4) & 0x700u);
                idx[2] = (unsigned int)qs[qs_walk + 2u] | (((unsigned int)qh[qh_walk + 1u] << 8) & 0x700u);
                idx[3] = (unsigned int)qs[qs_walk + 3u] | (((unsigned int)qh[qh_walk + 1u] << 4) & 0x700u);
                float delta[4];
                delta[0] = (qh[qh_walk] & 0x08u) != 0u ? -0.125f : 0.125f;
                delta[1] = (qh[qh_walk] & 0x80u) != 0u ? -0.125f : 0.125f;
                delta[2] = (qh[qh_walk + 1u] & 0x08u) != 0u ? -0.125f : 0.125f;
                delta[3] = (qh[qh_walk + 1u] & 0x80u) != 0u ? -0.125f : 0.125f;
                float dls[4] = { dl1, dl1, dl2, dl2 };
                for (unsigned int l = 0u; l < 4u; ++l) {
                    unsigned int grid_off = IQ_GRID_OFF_IQ1S + idx[l] * 8u;
                    for (unsigned int j = 0u; j < 8u; ++j) {
                        int gv = lut_i8(iq_lut, grid_off + j);
                        dst[y + j] = dls[l] * ((float)gv + delta[l]);
                    }
                    y += 8u;
                }
                qs_walk += 4u; qh_walk += 2u;
            }
            return;
        }
        return;
    }

    // ── 32-elem schemes: IQ4_NL, Q4_0, Q8_0, MXFP4 ──
    if (scheme_id == 19u) {
        unsigned int off = gid * 18u;
        float d = dq_read_f16(w_base, off);
        const unsigned char* qs = w_base + off + 2u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            dst[j] = d * (float)((int)(bx & 0x0Fu) - 8);
            dst[j + 16u] = d * (float)((int)(bx >> 4) - 8);
        }
        return;
    }

    if (scheme_id == 20u) {
        unsigned int off = gid * 34u;
        float d = dq_read_f16(w_base, off);
        const unsigned char* qs = w_base + off + 2u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 32u; ++j) {
            dst[j] = d * (float)(signed char)qs[j];
        }
        return;
    }

    if (scheme_id == 24u) {
        // Q1_0 (prism-ml Bonsai-27B): f16 d | 16 sign bytes (18 bytes / 128 elems).
        // Bit LSB-first within each byte; 1 -> +d, 0 -> -d.
        unsigned int off = gid * 18u;
        float d = dq_read_f16(w_base, off);
        float neg_d = -d;
        const unsigned char* qs = w_base + off + 2u;
        float* dst = arena + dst_f32_off + gid * 128u;
        for (unsigned int j = 0u; j < 128u; ++j) {
            unsigned int bit = (qs[j >> 3u] >> (j & 7u)) & 1u;
            dst[j] = bit ? d : neg_d;
        }
        return;
    }

    if (scheme_id == 21u) {
        unsigned int off = gid * 20u;
        float d = dq_read_f16(w_base, off);
        float m = dq_read_f16(w_base, off + 2u);
        const unsigned char* qs = w_base + off + 4u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            dst[j] = d * (float)(bx & 0x0Fu) + m;
            dst[j + 16u] = d * (float)(bx >> 4) + m;
        }
        return;
    }

    if (scheme_id == 22u) {
        unsigned int off = gid * 22u;
        float d = dq_read_f16(w_base, off);
        unsigned int qh = (unsigned int)w_base[off + 2u]
            | ((unsigned int)w_base[off + 3u] << 8u)
            | ((unsigned int)w_base[off + 4u] << 16u)
            | ((unsigned int)w_base[off + 5u] << 24u);
        const unsigned char* qs = w_base + off + 6u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            unsigned int xh0 = ((qh >> j) & 0x01u) << 4u;
            unsigned int xh1 = ((qh >> (j + 16u)) & 0x01u) << 4u;
            dst[j] = d * (float)((int)((bx & 0x0Fu) | xh0) - 16);
            dst[j + 16u] = d * (float)((int)((bx >> 4) | xh1) - 16);
        }
        return;
    }

    if (scheme_id == 23u) {
        unsigned int off = gid * 24u;
        float d = dq_read_f16(w_base, off);
        float m = dq_read_f16(w_base, off + 2u);
        unsigned int qh = (unsigned int)w_base[off + 4u]
            | ((unsigned int)w_base[off + 5u] << 8u)
            | ((unsigned int)w_base[off + 6u] << 16u)
            | ((unsigned int)w_base[off + 7u] << 24u);
        const unsigned char* qs = w_base + off + 8u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            unsigned int xh0 = ((qh >> j) & 0x01u) << 4u;
            unsigned int xh1 = ((qh >> (j + 16u)) & 0x01u) << 4u;
            dst[j] = d * (float)((bx & 0x0Fu) | xh0) + m;
            dst[j + 16u] = d * (float)((bx >> 4) | xh1) + m;
        }
        return;
    }

    if (scheme_id == 6u) {
        unsigned int off = gid * 18u;
        float d = dq_read_f16(w_base, off);
        const unsigned char* qs = w_base + off + 2u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            dst[j] = d * (float)kvalues_iq4nl_lut_d[bx & 0x0Fu];
            dst[j + 16u] = d * (float)kvalues_iq4nl_lut_d[bx >> 4];
        }
        return;
    }

    if (scheme_id == 10u) {
        static const float FP4[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        unsigned int off = gid * 17u;
        unsigned char e = w_base[off];
        float s = (e == 0xFFu) ? 0.0f : exp2f((float)((int)(unsigned int)e - 127));
        const unsigned char* qs = w_base + off + 1u;
        float* dst = arena + dst_f32_off + gid * 32u;
        for (unsigned int j = 0u; j < 16u; ++j) {
            unsigned char bx = qs[j];
            dst[2u * j] = s * FP4[bx & 0x0Fu];
            dst[2u * j + 1u] = s * FP4[bx >> 4];
        }
        return;
    }

    // ── 16-elem schemes: NVFP4 ──
    if (scheme_id == 11u) {
        static const float FP4[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        unsigned int off = gid * 9u;
        unsigned char e = w_base[off];
        unsigned int sign_b = ((unsigned int)e >> 7) & 1u;
        unsigned int exp_v = ((unsigned int)e >> 3) & 0x0Fu;
        unsigned int mant = (unsigned int)e & 0x7u;
        float s;
        if (exp_v == 0u) {
            s = (mant == 0u) ? 0.0f : ((float)mant / 8.0f) * exp2f(-6.0f);
        } else if (exp_v == 0x0Fu && mant == 0x7u) {
            s = 0.0f;
        } else {
            s = (1.0f + (float)mant / 8.0f) * exp2f((float)((int)exp_v - 7));
        }
        if (sign_b != 0u) s = -s;
        const unsigned char* qs = w_base + off + 1u;
        float* dst = arena + dst_f32_off + gid * 16u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            unsigned char bx = qs[j];
            dst[2u * j] = s * FP4[bx & 0x0Fu];
            dst[2u * j + 1u] = s * FP4[bx >> 4];
        }
        return;
    }
}
