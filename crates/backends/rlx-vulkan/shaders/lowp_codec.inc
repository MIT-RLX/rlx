// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared low-precision codec for the `scaled_lowp_*.comp` kernels (injected by
// build.rs alongside `arena_u32.inc`). Mirrors `rlx-ir/src/lowp_codec.rs` — the
// CPU oracle every backend is checked against — and its CUDA twin in
// `rlx-gpu-kernels/kernels/scaled_lowp_general.cu`.
//
// Two rules hold throughout, both learned from the wgpu port measuring against
// that oracle rather than from theory:
//
//   * **No transcendentals.** Drivers may compile with fast math, and an
//     approximate `exp2` is 1 ULP off for some integer arguments. Powers of two
//     are built from the exponent field instead.
//   * **No bare `/`.** Fast math lowers f32 division to a reciprocal-multiply,
//     also up to 1 ULP off. That is not cosmetic here: the scale it produces
//     multiplies an entire block, so `amax / max_finite` landed one ULP high and
//     shifted every element with it. `div_rn` corrects it with one Newton step.

const uint RLX_LOWP_CUSTOM_BIT = 0x80000000u;

// Exact 2^e for the exponent ranges these formats use (|e| <= 16), well inside
// f32's normal range.
float rlx_pow2i(int e) {
    return uintBitsToFloat(uint(e + 127) << 23);
}

// Correctly-rounded a/b. `fma` forms the residual `a - b*q` exactly; correcting
// by `r/b` lands on the correctly-rounded quotient (the second divide's error is
// second order).
float rlx_div_rn(float a, float b) {
    float q = a / b;
    float r = fma(-b, q, a);
    return q + r / b;
}

// naga's GLSL front end accepts isnan/isinf, but they are cheap to express and
// this keeps the file independent of that.
bool rlx_is_nan(float v) { return v != v; }
bool rlx_is_inf(float v) { return abs(v) > 3.4028235e38; }
bool rlx_is_finite(float v) { return !rlx_is_nan(v) && !rlx_is_inf(v); }
float rlx_qnan() { return uintBitsToFloat(0x7fc00000u); }
// naga's GLSL front end constant-folds and then rejects any expression that
// evaluates to infinity, so ±inf has to be built from its bit pattern.
float rlx_inf(float sgn) { return uintBitsToFloat(sgn < 0.0 ? 0xff800000u : 0x7f800000u); }

float rlx_decode_lowp(uint fmt, uint code) {
    uint e_bits;
    uint m_bits;
    int bias;
    bool fnuz = false;
    bool has_inf = false;
    bool e4m3ocp = false;

    if ((fmt & RLX_LOWP_CUSTOM_BIT) != 0u) {
        // Parameterized fNeXmY: all-finite, fields packed in `fmt`.
        e_bits = fmt & 0xFu;
        m_bits = (fmt >> 4) & 0xFu;
        uint b = (fmt >> 8) & 0xFFu;
        bias = (b >= 128u) ? (int(b) - 256) : int(b);
    } else if (fmt == 6u) {
        // FP4 E2M1 — matches rlx_ir FP4_E2M1_LUT bit-for-bit.
        uint c = code & 0xFu;
        float mag[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
        float v = mag[c & 0x7u];
        return ((c & 0x8u) != 0u) ? -v : v;
    } else if (fmt == 0u) {
        e_bits = 4u; m_bits = 3u; bias = 7;  e4m3ocp = true;
    } else if (fmt == 1u) {
        e_bits = 5u; m_bits = 2u; bias = 15; has_inf = true;
    } else if (fmt == 2u) {
        e_bits = 4u; m_bits = 3u; bias = 8;  fnuz = true;
    } else if (fmt == 3u) {
        e_bits = 5u; m_bits = 2u; bias = 16; fnuz = true;
    } else if (fmt == 4u) {
        e_bits = 2u; m_bits = 3u; bias = 1;
    } else if (fmt == 5u) {
        e_bits = 3u; m_bits = 2u; bias = 3;
    } else {
        return 0.0;
    }

    uint width = e_bits + m_bits;
    uint sign_bit = (code >> width) & 1u;
    uint exp = (code >> m_bits) & ((1u << e_bits) - 1u);
    uint mant = code & ((1u << m_bits) - 1u);
    float sgn = (sign_bit != 0u) ? -1.0 : 1.0;
    uint max_exp = (1u << e_bits) - 1u;

    if (fnuz) {
        if (sign_bit != 0u && exp == 0u && mant == 0u) { return rlx_qnan(); }
    } else if (has_inf) {
        if (exp == max_exp) {
            if (mant == 0u) { return rlx_inf(sgn); }
            return rlx_qnan();
        }
    } else if (e4m3ocp) {
        if (exp == max_exp && mant == ((1u << m_bits) - 1u)) { return rlx_qnan(); }
    }

    float m_div = float(1u << m_bits);
    float val;
    if (exp == 0u) {
        val = (float(mant) / m_div) * rlx_pow2i(1 - bias);
    } else {
        val = (1.0 + float(mant) / m_div) * rlx_pow2i(int(exp) - bias);
    }
    return sgn * val;
}

uint rlx_code_width(uint fmt) {
    if ((fmt & RLX_LOWP_CUSTOM_BIT) != 0u) {
        return 1u + (fmt & 0xFu) + ((fmt >> 4) & 0xFu);
    }
    if (fmt == 6u) { return 4u; }
    if (fmt == 4u || fmt == 5u) { return 6u; }
    return 8u;
}

float rlx_max_finite(uint fmt) {
    if ((fmt & RLX_LOWP_CUSTOM_BIT) != 0u) {
        // Scan the (<=256-code) space like the CPU oracle — matches exactly.
        uint n = 1u << rlx_code_width(fmt);
        float mx = 0.0;
        for (uint c = 0u; c < n; ++c) {
            float v = abs(rlx_decode_lowp(fmt, c));
            if (rlx_is_finite(v)) { mx = max(mx, v); }
        }
        return mx;
    }
    if (fmt == 0u) { return 448.0; }
    if (fmt == 1u) { return 57344.0; }
    if (fmt == 2u) { return 240.0; }
    if (fmt == 3u) { return 57344.0; }
    if (fmt == 4u) { return 7.5; }
    if (fmt == 5u) { return 28.0; }
    return 6.0; // e2m1
}

// Nearest-representable encode by exhaustive search — round-half-to-even,
// saturating, NaN -> 0.
//
// The oracle computes the error term in f64, which GLSL does not have here.
// `rlx-ir/tests/lowp_encode_f32_feasibility.rs` measures that an f32-only search
// picks the identical code for every grid point, every midpoint, ±1 ULP either
// side, and 20k pseudo-random values per format. That test is this function's
// licence.
uint rlx_encode_lowp(uint fmt, float x_in) {
    if (rlx_is_nan(x_in)) { return 0u; }
    float x = x_in;
    // ±inf saturates to ±max_finite: a value far outside the grid is equidistant
    // from every code, so the tie-break would otherwise pick code 0.
    if (rlx_is_inf(x)) {
        float mf = rlx_max_finite(fmt);
        x = (x > 0.0) ? mf : -mf;
    }
    uint n_codes = 1u << rlx_code_width(fmt);
    uint best = 0u;
    float best_err = 3.4028235e38;
    uint best_lsb = 1u;
    for (uint c = 0u; c < n_codes; ++c) {
        float v = rlx_decode_lowp(fmt, c);
        if (!rlx_is_finite(v)) { continue; }
        float err = abs(v - x);
        uint lsb = c & 1u;
        if (err < best_err || (err == best_err && lsb < best_lsb)) {
            best_err = err;
            best = c;
            best_lsb = lsb;
        }
    }
    return best;
}

float rlx_e8m0(uint b) {
    if (b == 0xFFu) { return rlx_qnan(); }
    return rlx_pow2i(int(b) - 127);
}

// `ceil(log2(s))` from the bits rather than from `log2`. Higher stakes than
// `rlx_pow2i`: this picks a whole E8M0 scale byte, so a 1-ULP `log2` near a
// power of two would shift a block by a factor of two.
uint rlx_f32_to_e8m0(float s) {
    if (!(s > 0.0) || !rlx_is_finite(s)) { return 0u; }
    uint bits = floatBitsToUint(s);
    int exp_field = int((bits >> 23) & 0xFFu);
    uint mant = bits & 0x7FFFFFu;
    int e;
    if (exp_field == 0) {
        // Subnormal: s = m * 2^-149, so ceil(log2 s) = ceil(log2 m) - 149.
        uint m = mant;
        int lz = 0;
        while (m > 1u) { m >>= 1u; lz += 1; }
        bool is_pow2 = (mant & (mant - 1u)) == 0u;
        e = lz - 149 + (is_pow2 ? 0 : 1);
    } else {
        e = exp_field - 127 + ((mant == 0u) ? 0 : 1);
    }
    int b = e + 127;
    if (b < 0) { b = 0; }
    if (b > 254) { b = 254; }
    return uint(b);
}

// Unsigned byte `byte_idx` relative to a tagged word-aligned tensor base —
// the same convention `scaled_grouped_matmul_decode.comp` uses, so every offset
// in a push block is a WORD offset and byte indexing stays tensor-relative.
uint rlx_rd_byte(uint word_off, uint byte_idx) {
    uint w = arena_load_u(word_off + (byte_idx >> 2u));
    return (w >> ((byte_idx & 3u) * 8u)) & 0xFFu;
}

// Assemble and store four packed code bytes as one word.
//
// A per-byte read-modify-write would race with the three neighbouring
// invocations sharing the word, and naga's GLSL front end has no
// `atomicCompSwap` (see `arena_store_i8`). Every writer below therefore owns a
// whole word, exactly as `quantize_i8.comp` does.
void rlx_wr_word(uint word_off, uint widx, uvec4 bytes) {
    uint packed = (bytes.x & 0xFFu)
        | ((bytes.y & 0xFFu) << 8)
        | ((bytes.z & 0xFFu) << 16)
        | ((bytes.w & 0xFFu) << 24);
    arena_store_u(word_off + widx, packed);
}
