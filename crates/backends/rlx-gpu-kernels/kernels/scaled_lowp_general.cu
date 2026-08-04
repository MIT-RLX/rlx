// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// General (all-format, all-scale-layout) low-precision quantize + GEMM for
// Op::ScaledMatMul on CUDA / ROCm. This is the **decode-and-accumulate
// reference on GPU cores** — NOT tensor-core native. It's the path for formats
// the FP8 tensor-core GEMM can't do on the current toolkit (block-scaled MX,
// FP4 NVFP4/MXFP4, FP6), so those graphs still run on-device instead of
// erroring. Per-tensor FP8 keeps using the native cublasLt/hipBLASLt path.
//
// All decode/encode logic mirrors rlx-ir/src/lowp_codec.rs bit-for-bit (the
// CPU oracle every backend is checked against). Arena convention matches
// dequant_matmul.cu: f32 arena base + f32-element offsets for f32 tensors,
// byte offsets (via reinterpret_cast<unsigned char*>) for U8 code/scale tensors.

// Format word (`fmt`) — matches `ScaledFormat::kernel_id()` in rlx-ir/quant.rs:
//   Named ids (top bit clear): 0 e4m3, 1 e5m2, 2 e4m3fnuz, 3 e5m2fnuz,
//                              4 e2m3, 5 e3m2, 6 e2m1.
//   Custom `fNeXmY` (top bit set): 0x8000_0000 | exp_bits | mant_bits<<4 |
//     (bias&0xFF)<<8 — an all-finite parameterized minifloat, decoded
//     generically from the unpacked fields so a new format needs no kernel edit.
// Scale modes: 0 per-tensor (f32), 1 block E8M0 (u8), 2 NVFP4 E4M3 (u8).
#define RLX_LOWP_CUSTOM_BIT 0x80000000u

// NVRTC compiles this source without <math.h>, so the `INFINITY` macro is
// undefined there (host nvcc/hipcc provide it). Supply it via a bit intrinsic.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// Forward decl: rlx_encode_lowp saturates ±inf to ±max_finite (defined below).
__device__ __forceinline__ float rlx_max_finite(unsigned int fmt);

__device__ __forceinline__ float rlx_decode_lowp(unsigned int fmt, unsigned int code) {
    unsigned int e_bits, m_bits;
    int bias;
    unsigned int fnuz = 0u, has_inf = 0u, e4m3ocp = 0u;
    if (fmt & RLX_LOWP_CUSTOM_BIT) {
        // Parameterized fNeXmY: all-finite, fields packed in `fmt`.
        e_bits = fmt & 0xFu;
        m_bits = (fmt >> 4) & 0xFu;
        bias   = (int)(signed char)((fmt >> 8) & 0xFFu);
    } else {
        if (fmt == 6u) { // FP4 E2M1 LUT
            const float lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            return lut[code & 0xFu];
        }
        switch (fmt) {
            case 0u: e_bits = 4u; m_bits = 3u; bias = 7;  e4m3ocp = 1u; break;
            case 1u: e_bits = 5u; m_bits = 2u; bias = 15; has_inf = 1u; break;
            case 2u: e_bits = 4u; m_bits = 3u; bias = 8;  fnuz = 1u; break;
            case 3u: e_bits = 5u; m_bits = 2u; bias = 16; fnuz = 1u; break;
            case 4u: e_bits = 2u; m_bits = 3u; bias = 1;  break; // e2m3 (finite)
            case 5u: e_bits = 3u; m_bits = 2u; bias = 3;  break; // e3m2 (finite)
            default: return 0.0f;
        }
    }
    unsigned int width = e_bits + m_bits;
    unsigned int sign_bit = (code >> width) & 1u;
    unsigned int exp = (code >> m_bits) & ((1u << e_bits) - 1u);
    unsigned int mant = code & ((1u << m_bits) - 1u);
    float sign = sign_bit ? -1.0f : 1.0f;
    unsigned int max_exp = (1u << e_bits) - 1u;
    if (fnuz) {
        if (sign_bit && exp == 0u && mant == 0u) return nanf("");
    } else if (has_inf) {
        if (exp == max_exp) return mant == 0u ? sign * INFINITY : nanf("");
    } else if (e4m3ocp) {
        if (exp == max_exp && mant == ((1u << m_bits) - 1u)) return nanf("");
    }
    float m_div = (float)(1u << m_bits);
    float val;
    if (exp == 0u) {
        val = ((float)mant / m_div) * exp2f((float)(1 - bias));
    } else {
        val = (1.0f + (float)mant / m_div) * exp2f((float)((int)exp - bias));
    }
    return sign * val;
}

// Nearest-representable encode by exhaustive search of the code space (≤256) —
// simple and exact, round-half-to-even, saturating, NaN→0 (matches the oracle).
__device__ __forceinline__ unsigned char rlx_encode_lowp(unsigned int fmt, float x) {
    if (isnan(x)) return 0u;
    // ±inf saturates to ±max_finite (mirrors lowp_codec.rs — snapping to a
    // generic huge value would be equidistant from every code in the search).
    if (isinf(x)) { float mf = rlx_max_finite(fmt); x = (x > 0.0f ? mf : -mf); }
    unsigned int width = (fmt & RLX_LOWP_CUSTOM_BIT)
        ? (1u + (fmt & 0xFu) + ((fmt >> 4) & 0xFu))
        : ((fmt == 6u) ? 4u : ((fmt == 4u || fmt == 5u) ? 6u : 8u));
    unsigned int n_codes = 1u << width;
    unsigned char best = 0u;
    double best_err = 1.0e300;
    unsigned char best_lsb = 1u;
    for (unsigned int c = 0u; c < n_codes; ++c) {
        float v = rlx_decode_lowp(fmt, (unsigned char)c);
        if (!isfinite(v)) continue;
        double err = fabs((double)v - (double)x);
        unsigned char lsb = (unsigned char)(c & 1u);
        if (err < best_err || (err == best_err && lsb < best_lsb)) {
            best_err = err;
            best = (unsigned char)c;
            best_lsb = lsb;
        }
    }
    return best;
}

__device__ __forceinline__ float rlx_e8m0(unsigned char b) {
    return b == 0xFFu ? nanf("") : exp2f((float)((int)b - 127));
}

__device__ __forceinline__ unsigned char rlx_f32_to_e8m0(float s) {
    if (!(s > 0.0f) || !isfinite(s)) return 0u;
    int e = (int)ceilf(log2f(s)) + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

// Largest finite magnitude of a format (for amax→scale).
__device__ __forceinline__ float rlx_max_finite(unsigned int fmt) {
    if (fmt & RLX_LOWP_CUSTOM_BIT) {
        // Scan the (≤256-code) space like the CPU oracle — matches exactly.
        unsigned int width = 1u + (fmt & 0xFu) + ((fmt >> 4) & 0xFu);
        unsigned int n = 1u << width;
        float mx = 0.0f;
        for (unsigned int c = 0u; c < n; ++c) {
            float v = fabsf(rlx_decode_lowp(fmt, c));
            if (isfinite(v)) mx = fmaxf(mx, v);
        }
        return mx;
    }
    switch (fmt) {
        case 0u: return 448.0f;
        case 1u: return 57344.0f;
        case 2u: return 240.0f;
        case 3u: return 57344.0f;
        case 4u: return 7.5f;
        case 5u: return 28.0f;
        default: return 6.0f; // e2m1
    }
}

// Per-row block (or per-tensor) amax → scale; stores f32 (per-tensor) or u8
// (E8M0 / NVFP4-E4M3) snapped scale. One thread per scale element.
extern "C" __global__ void scaled_quant_scale_general(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_byte_off,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    unsigned int total = (scale_mode == 0u) ? 1u : rows * nblk;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float maxf = rlx_max_finite(fmt);

    float amax = 0.0f;
    if (scale_mode == 0u) {
        for (unsigned int i = 0u; i < rows * cols; ++i) {
            amax = fmaxf(amax, fabsf(arena[x_off_f32 + i]));
        }
    } else {
        unsigned int r = idx / nblk, b = idx % nblk;
        unsigned int lo = b * block, hi = min(lo + block, cols);
        for (unsigned int c = lo; c < hi; ++c) {
            amax = fmaxf(amax, fabsf(arena[x_off_f32 + r * cols + c]));
        }
    }
    float s = amax > 0.0f ? amax / maxf : 1.0f;
    if (scale_mode == 0u) {
        arena[scale_byte_off / 4u] = s; // per-tensor f32
    } else {
        unsigned char* out = reinterpret_cast<unsigned char*>(arena) + scale_byte_off;
        out[idx] = (scale_mode == 1u) ? rlx_f32_to_e8m0(s)
                                      : rlx_encode_lowp(0u, s); // NVFP4 E4M3 scale
    }
}

// Quantize x / scale(block) → codes for any format / scale layout.
extern "C" __global__ void scaled_quantize_general(
    float* __restrict__ arena,
    unsigned int x_off_f32,
    unsigned int scale_byte_off,
    unsigned int out_byte_off,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols, c = i % cols;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    float s;
    if (scale_mode == 0u) {
        s = arena[scale_byte_off / 4u];
    } else {
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(arena) + scale_byte_off;
        unsigned int si = r * nblk + c / block;
        s = (scale_mode == 1u) ? rlx_e8m0(sb[si]) : rlx_decode_lowp(0u, sb[si]);
    }
    float v = (s != 0.0f) ? (arena[x_off_f32 + i] / s) : 0.0f;
    unsigned char* out = reinterpret_cast<unsigned char*>(arena) + out_byte_off;
    out[i] = rlx_encode_lowp(fmt, v);
}

// Dequantize: codes → f32 via decode(code) * scale(block). The exact inverse of
// scaled_quantize_general; one thread per element. Used by the ScaledMatMul
// backward (straight-through QAT) to rebuild f32 operands, and as a standalone
// dequantizer.
extern "C" __global__ void scaled_dequantize_general(
    float* __restrict__ arena,
    unsigned int codes_byte_off,
    unsigned int scale_byte_off,
    unsigned int out_off_f32,
    unsigned int rows,
    unsigned int cols,
    unsigned int fmt,
    unsigned int scale_mode,
    unsigned int block)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int r = i / cols, c = i % cols;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((cols + block - 1u) / block);
    float s;
    if (scale_mode == 0u) {
        s = arena[scale_byte_off / 4u];
    } else {
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(arena) + scale_byte_off;
        unsigned int si = r * nblk + c / block;
        s = (scale_mode == 1u) ? rlx_e8m0(sb[si]) : rlx_decode_lowp(0u, sb[si]);
    }
    const unsigned char* codes = reinterpret_cast<const unsigned char*>(arena) + codes_byte_off;
    arena[out_off_f32 + i] = rlx_decode_lowp(fmt, codes[i]) * s;
}

// MxFp4x2 two-level residual FP4 decode: out = s0·E2M1[q0] + s1·E2M1[q1].
// The low-precision analog of double-word (see rlx_ir::residual). Codes are one
// byte each (0..15); scales are per-group f32. LUT matches
// rlx_ir::nvfp4::FP4_E2M1_LUT so it decodes rlx's F4E2M1 nibbles bit-for-bit.
extern "C" __global__ void mxfp4x2_decode(
    const unsigned char* __restrict__ q0,
    const unsigned char* __restrict__ q1,
    const float* __restrict__ s0,
    const float* __restrict__ s1,
    float* __restrict__ out,
    unsigned int n,
    unsigned int group)
{
    const float E2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int blk = i / group;
    out[i] = s0[blk] * E2M1[q0[i] & 0xF] + s1[blk] * E2M1[q1[i] & 0xF];
}

// MxFp4x2 DequantMatMul decode-to-scratch: dequant a packed two-level residual
// E2M1 weight into an f32 [k,n] scratch (row-major, dst[p*n+j]) — the on-GPU
// twin of rlx_cpu's `dequant_matmul_mxfp4x2`, byte-for-byte the same layout so
// the same hipBLAS/cuBLAS sgemm consumes it (col-major n×k, A[j + p*n]). w_q =
// [plane0|plane1] E2M1 nibbles packed 2/byte over the [k,n] grid; scales =
// [s0|s1] f32 per (block=k/group, n). Arena is the f32-uniform buffer; offsets
// are byte (weight) / f32-index (scales, dst). Run it, then sgemm x·scratch.
extern "C" __global__ void mxfp4x2_dequant(
    float* arena,
    unsigned long long w_byte_off,   // [plane0|plane1] start (arena bytes)
    unsigned long long s_f32_off,    // [s0|s1] start (arena f32 index)
    unsigned long long dst_f32_off,  // scratch start (arena f32 index)
    unsigned int k,
    unsigned int n,
    unsigned int group)
{
    const float E2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = k * n;
    if (idx >= total) return;

    unsigned int p = idx / n;           // k index
    unsigned int j = idx - p * n;       // n index
    unsigned int g = group ? group : 1u;
    unsigned int nblk = (k + g - 1u) / g;
    unsigned int blk = p / g;

    const unsigned char* w = reinterpret_cast<const unsigned char*>(arena) + w_byte_off;
    unsigned int plane = (total + 1u) >> 1;      // bytes per nibble plane
    unsigned int byte = idx >> 1;
    unsigned int shift = (idx & 1u) ? 4u : 0u;
    unsigned int q0 = (w[byte] >> shift) & 0xFu;
    unsigned int q1 = (w[plane + byte] >> shift) & 0xFu;

    const float* s = arena + s_f32_off;
    float s0 = s[blk * n + j];
    float s1 = s[nblk * n + blk * n + j];
    arena[dst_f32_off + idx] = s0 * E2M1[q0] + s1 * E2M1[q1];
}

// MxFp4x2 decode-to-scratch, TRANSPOSED output layout: writes the decoded
// weight as row-major [n,k] (dst[j*k + p]) instead of [k,n] — for backends
// whose GEMM wants the GGUF `matmul_bt` convention (C[m,n] = X[m,k]·W[n,k]ᵀ,
// e.g. rlx-cuda). Input packing / scales are read identically to
// `mxfp4x2_dequant`; only the store index differs.
extern "C" __global__ void mxfp4x2_dequant_nk(
    float* arena,
    unsigned long long w_byte_off,
    unsigned long long s_f32_off,
    unsigned long long dst_f32_off,
    unsigned int k,
    unsigned int n,
    unsigned int group)
{
    const float E2M1[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = k * n;
    if (idx >= total) return;

    unsigned int p = idx / n;           // k index
    unsigned int j = idx - p * n;       // n index
    unsigned int g = group ? group : 1u;
    unsigned int nblk = (k + g - 1u) / g;
    unsigned int blk = p / g;

    const unsigned char* w = reinterpret_cast<const unsigned char*>(arena) + w_byte_off;
    unsigned int plane = (total + 1u) >> 1;
    unsigned int byte = idx >> 1;
    unsigned int shift = (idx & 1u) ? 4u : 0u;
    unsigned int q0 = (w[byte] >> shift) & 0xFu;
    unsigned int q1 = (w[plane + byte] >> shift) & 0xFu;

    const float* s = arena + s_f32_off;
    float s0 = s[blk * n + j];
    float s1 = s[nblk * n + blk * n + j];
    arena[dst_f32_off + (unsigned long long)j * k + p] = s0 * E2M1[q0] + s1 * E2M1[q1];
}

// Decode-and-accumulate GEMM (TN: lhs[m,k]·rhs[n,k]ᵀ → out[m,n]) — the
// non-tensor-core fallback for formats cublasLt can't do. Shared-memory tiled:
// each 16×16 output tile cooperatively stages a 16-wide strip of decoded,
// scale-applied lhs / rhs into shared memory and reuses it, so each code is
// decoded once per tile instead of once per output element (≈16× fewer decodes,
// plus f32 reuse from shared mem). Launched with a fixed 16×16 block (see the
// LaunchConfig in backend.rs). Accumulation is tile-blocked in f32, so it tracks
// — but is not bit-identical to — the sequential CPU oracle.
#define RLX_LOWP_TILE 16u
extern "C" __global__ void scaled_matmul_decode(
    float* __restrict__ arena,
    unsigned int lhs_byte_off,
    unsigned int rhs_byte_off,
    unsigned int lhs_scale_byte_off,
    unsigned int rhs_scale_byte_off,
    unsigned int out_off_f32,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int lhs_fmt,
    unsigned int rhs_fmt,
    unsigned int scale_mode,
    unsigned int block,
    unsigned int has_bias,
    unsigned int bias_off_f32)
{
    __shared__ float As[RLX_LOWP_TILE][RLX_LOWP_TILE];
    __shared__ float Bs[RLX_LOWP_TILE][RLX_LOWP_TILE];
    const unsigned char* lhs = reinterpret_cast<const unsigned char*>(arena) + lhs_byte_off;
    const unsigned char* rhs = reinterpret_cast<const unsigned char*>(arena) + rhs_byte_off;
    const unsigned char* lsb = reinterpret_cast<const unsigned char*>(arena) + lhs_scale_byte_off;
    const unsigned char* rsb = reinterpret_cast<const unsigned char*>(arena) + rhs_scale_byte_off;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((k + block - 1u) / block);
    float ls0 = arena[lhs_scale_byte_off / 4u];
    float rs0 = arena[rhs_scale_byte_off / 4u];

    unsigned int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int i = blockIdx.y * RLX_LOWP_TILE + ty; // output row (m)
    unsigned int j = blockIdx.x * RLX_LOWP_TILE + tx; // output col (n)

    float acc = 0.0f;
    unsigned int ntiles = (k + RLX_LOWP_TILE - 1u) / RLX_LOWP_TILE;
    for (unsigned int t = 0u; t < ntiles; ++t) {
        // Stage A[i, t*TILE+tx] (decoded × its scale) into shared As[ty][tx].
        unsigned int pa = t * RLX_LOWP_TILE + tx;
        if (i < m && pa < k) {
            float ls;
            if (scale_mode == 0u) {
                ls = ls0;
            } else {
                unsigned int li = i * nblk + pa / block;
                ls = (scale_mode == 1u) ? rlx_e8m0(lsb[li]) : rlx_decode_lowp(0u, lsb[li]);
            }
            As[ty][tx] = rlx_decode_lowp(lhs_fmt, lhs[i * k + pa]) * ls;
        } else {
            As[ty][tx] = 0.0f;
        }
        // Stage B[j, t*TILE+ty] into shared Bs[ty][tx] (Bs[p][tx] ↦ rhs[j, ·]).
        unsigned int pb = t * RLX_LOWP_TILE + ty;
        if (j < n && pb < k) {
            float rs;
            if (scale_mode == 0u) {
                rs = rs0;
            } else {
                unsigned int ri = j * nblk + pb / block;
                rs = (scale_mode == 1u) ? rlx_e8m0(rsb[ri]) : rlx_decode_lowp(0u, rsb[ri]);
            }
            Bs[ty][tx] = rlx_decode_lowp(rhs_fmt, rhs[j * k + pb]) * rs;
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int p = 0u; p < RLX_LOWP_TILE; ++p) {
            acc += As[ty][p] * Bs[p][tx];
        }
        __syncthreads();
    }
    if (i < m && j < n) {
        if (has_bias) acc += arena[bias_off_f32 + j];
        arena[out_off_f32 + i * n + j] = acc;
    }
}

// Native low-precision *grouped* (MoE) decode-GEMM for Op::ScaledGroupedMatMul —
// the expert-indexed analogue of scaled_matmul_decode. One thread per output
// C[row=token, col=out]; the token's expert picks the weight slab, and only that
// routed expert's FP4 codes are decoded on the fly (no f32 weight
// materialization — memory-sane for large MoE). TN per expert:
//   out[i,j] = Σ_p decode(input[i,p])·s_in · decode(weight[e,j,p])·s_w  (+ bias[e,j])
// input codes [M,K], weight codes [E,N,K], input scale [M,nblk],
// weight scale [E·N,nblk], expert_idx [M] f32, bias [E·N] f32 (per-expert).
extern "C" __global__ void scaled_grouped_matmul_decode(
    float* __restrict__ arena,
    unsigned int input_byte_off,
    unsigned int weight_byte_off,
    unsigned int input_scale_byte_off,
    unsigned int weight_scale_byte_off,
    unsigned int idx_off_f32,
    unsigned int out_off_f32,
    unsigned int bias_off_f32,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int num_experts,
    unsigned int lhs_fmt,
    unsigned int rhs_fmt,
    unsigned int scale_mode,
    unsigned int block,
    unsigned int has_bias)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y; // token i
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x; // output j
    if (row >= m || col >= n) return;
    unsigned int e = (unsigned int)arena[idx_off_f32 + row];
    if (e >= num_experts) return;

    const unsigned char* inp = reinterpret_cast<const unsigned char*>(arena) + input_byte_off;
    const unsigned char* wt = reinterpret_cast<const unsigned char*>(arena) + weight_byte_off;
    const unsigned char* isb = reinterpret_cast<const unsigned char*>(arena) + input_scale_byte_off;
    const unsigned char* wsb = reinterpret_cast<const unsigned char*>(arena) + weight_scale_byte_off;
    unsigned int nblk = (scale_mode == 0u) ? 1u : ((k + block - 1u) / block);
    unsigned int wrow = e * n + col; // weight code/scale row for this expert+output

    // Per-tensor scales live at the very start of their f32 tensor.
    float ls0 = arena[input_scale_byte_off / 4u];
    float rs0 = arena[weight_scale_byte_off / 4u];

    float acc = 0.0f;
    for (unsigned int p = 0u; p < k; ++p) {
        float ls, rs;
        if (scale_mode == 0u) {
            ls = ls0;
            rs = rs0;
        } else {
            unsigned int li = row * nblk + p / block;
            unsigned int ri = wrow * nblk + p / block;
            ls = (scale_mode == 1u) ? rlx_e8m0(isb[li]) : rlx_decode_lowp(0u, isb[li]);
            rs = (scale_mode == 1u) ? rlx_e8m0(wsb[ri]) : rlx_decode_lowp(0u, wsb[ri]);
        }
        float a = rlx_decode_lowp(lhs_fmt, inp[row * k + p]) * ls;
        float b = rlx_decode_lowp(rhs_fmt, wt[wrow * k + p]) * rs;
        acc += a * b;
    }
    if (has_bias) acc += arena[bias_off_f32 + wrow];
    arena[out_off_f32 + row * n + col] = acc;
}
