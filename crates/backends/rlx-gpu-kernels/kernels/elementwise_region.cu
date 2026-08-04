// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// PLAN L2 — interpreted N-ary element-wise chain kernel.
//
// One thread per output element. Each thread walks a runtime chain
// encoding (length `num_steps`, packed inside `meta`) into a private
// scratch register array; the final step's result is written to dst.
//
// `meta` layout (150 u32 words, packed by the caller):
//   meta[0..16]   = input_offs[0..16]  (only first num_inputs used)
//   meta[16..144] = chain[0..128]      (32 steps * 4 u32s)
//   meta[144]     = prologue (0=none, 1=resize_nearest_2x NCHW)
//   meta[145..148]= out_n, out_c, out_h, out_w (for prologue)
//   meta[149]     = prologue_input (external input index for prologue)
//
// Chain encoding (4 u32s per step, indices into chain[]):
//   chain[k*4 + 0] = op_kind   (0=Activation, 1=Cast, 2=Binary,
//                               3=Compare, 4=Where)
//   chain[k*4 + 1] = op_sub    (discriminant of the inner op variant;
//                               for op_kind=4, carries cond_enc instead)
//   chain[k*4 + 2] = lhs_enc   (bit 31 = src kind: 0=Input, 1=Step;
//                               bits 0..30 = index. For op_kind=4 this
//                               is on_true)
//   chain[k*4 + 3] = rhs_enc   (same encoding; ignored for unary ops;
//                               for op_kind=4 this is on_false)
//
// op_sub mappings match the Metal MSL / wgpu WGSL chain kernels so
// the same encoder in rlx-opt produces correct results across all
// region-capable backends.
//
// Caps: 32 chain steps, 16 inputs (matches the schedule encoder).

struct InputModulus { unsigned int v[16]; };

// FKL-style closed region: map output gid to input row for resize-nearest 2x on NCHW.
__device__ unsigned int region_input_row_resize2x_nchw(
    unsigned int gid,
    unsigned int out_n,
    unsigned int out_c,
    unsigned int out_h,
    unsigned int out_w
) {
    unsigned int plane = out_c * out_h * out_w;
    unsigned int local = gid % plane;
    unsigned int batch = gid / plane;
    unsigned int w_pos = local % out_w;
    unsigned int tmp = local / out_w;
    unsigned int h_pos = tmp % out_h;
    unsigned int c_pos = tmp / out_h;
    unsigned int in_w = out_w / 2u;
    unsigned int in_h = out_h / 2u;
    unsigned int in_plane = out_c * in_h * in_w;
    return batch * in_plane + c_pos * in_h * in_w + (h_pos / 2u) * in_w + (w_pos / 2u);
}

__device__ unsigned int region_resolve_row(
    unsigned int gid,
    unsigned int kind,
    unsigned int idx,
    unsigned int prologue_row0,
    unsigned int has_prologue_row0,
    unsigned int prologue_input,
    unsigned int scalar_input_mask,
    InputModulus input_modulus
) {
    if (kind != 0u) { return 0u; }
    if (has_prologue_row0 != 0u && idx == prologue_input) {
        return prologue_row0;
    }
    if ((scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (input_modulus.v[idx] != 0u) { return gid % input_modulus.v[idx]; }
    return gid;
}

extern "C" __global__ void elementwise_region(
    float* arena,
    unsigned int len,
    unsigned int /*num_inputs*/,
    unsigned int num_steps,
    unsigned int dst_off,
    const unsigned int* meta,
    unsigned int scalar_input_mask,
    InputModulus input_modulus
) {
    const unsigned int* input_offs = meta;
    const unsigned int* chain      = meta + 16;
    const unsigned int prologue    = meta[144];
    const unsigned int out_n       = meta[145];
    const unsigned int out_c       = meta[146];
    const unsigned int out_h       = meta[147];
    const unsigned int out_w       = meta[148];
    const unsigned int prologue_input = meta[149];

    unsigned int i;
    if (prologue == 1u) {
        unsigned int wo = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ho = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int nc = blockIdx.z * blockDim.z + threadIdx.z;
        if (nc >= out_n * out_c || ho >= out_h || wo >= out_w) {
            return;
        }
        i = nc * out_h * out_w + ho * out_w + wo;
    } else {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= len) {
            return;
        }
    }

    unsigned int prologue_row0 = 0u;
    unsigned int has_prologue_row0 = 0u;
    if (prologue == 1u) {
        prologue_row0 = region_input_row_resize2x_nchw(i, out_n, out_c, out_h, out_w);
        has_prologue_row0 = 1u;
    }

    float scratch[32];
    unsigned int last_idx = 0;

    for (unsigned int k = 0; k < num_steps; ++k) {
        unsigned int base    = k * 4u;
        unsigned int op_kind = chain[base + 0u];
        unsigned int op_sub  = chain[base + 1u];
        unsigned int lhs_enc = chain[base + 2u];
        unsigned int rhs_enc = chain[base + 3u];

        // Resolve LHS operand. Scalar-broadcast inputs read element 0
        // (fast path); trailing-shape broadcast tiles by
        // `i % input_modulus.v[idx]`; modulus 0 ⇒ read by gid.
        float lhs;
        {
            unsigned int kind = lhs_enc >> 31;
            unsigned int idx  = lhs_enc & 0x7FFFFFFFu;
            unsigned int row = region_resolve_row(
                i, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                scalar_input_mask, input_modulus);
            lhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
        }

        float result;
        if (op_kind == 4u) {
            // Where (3-operand select). op_sub carries cond_enc; lhs
            // already resolved is on_true; rhs_enc is on_false.
            float cond;
            {
                unsigned int kind = op_sub >> 31;
                unsigned int idx  = op_sub & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(
                    i, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                unsigned int kind = rhs_enc >> 31;
                unsigned int idx  = rhs_enc & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(
                    i, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
            // Activation. op_sub (gelu-first): 0=Gelu, 1=GeluApprox, 2=Silu,
            // 3=Relu, 4=Sigmoid, 5=Tanh, 6=Exp, 7=Log, 8=Sqrt, 9=Rsqrt,
            // 10=Neg, 11=Abs, 12=Sin, 13=Cos, 14=Tan, 15=Atan, 16=Round,
            // 17=Recip, 18=Floor, 19=Ceil, 20=Sign, 21=Softplus, 22=Elu,
            // 23=Erf, 24=HardSwish, 25=HardSigmoid, 26=Mish, 27=Softsign,
            // 28=LogSigmoid. (12..28 were missing → identity → broke fused
            // Sin/Cos/Round, e.g. the StyleTTS2/Kokoro harmonic source.)
            if      (op_sub == 3u) result = fmaxf(lhs, 0.0f);
            else if (op_sub == 0u) { result = gelu_erf(lhs); }
            else if (op_sub == 1u) { result = gelu_approx(lhs); }
            else if (op_sub == 2u) result = lhs / (1.0f + expf(-lhs));
            else if (op_sub == 4u) result = 1.0f / (1.0f + expf(-lhs));
            else if (op_sub == 5u) result = tanhf(lhs);
            else if (op_sub == 6u) result = expf(lhs);
            else if (op_sub == 7u) result = logf(lhs);
            else if (op_sub == 8u) result = sqrtf(lhs);
            else if (op_sub == 9u) result = rsqrtf(lhs);
            else if (op_sub == 10u) result = -lhs;
            else if (op_sub == 11u) result = fabsf(lhs);
            else if (op_sub == 12u) result = sinf(lhs);
            else if (op_sub == 13u) result = cosf(lhs);
            else if (op_sub == 14u) result = tanf(lhs);
            else if (op_sub == 15u) result = atanf(lhs);
            else if (op_sub == 16u) result = rintf(lhs);
            else if (op_sub == 17u) result = 1.0f / lhs;
            else if (op_sub == 18u) result = floorf(lhs);
            else if (op_sub == 19u) result = ceilf(lhs);
            else if (op_sub == 20u) result = (float)(lhs > 0.0f) - (float)(lhs < 0.0f);
            else if (op_sub == 21u) result = fmaxf(lhs, 0.0f) + logf(1.0f + expf(-fabsf(lhs)));
            else if (op_sub == 22u) result = (lhs > 0.0f) ? lhs : (expf(lhs) - 1.0f);
            else if (op_sub == 23u) result = erff(lhs);
            else if (op_sub == 24u) result = (lhs * fminf(fmaxf(lhs + 3.0f, 0.0f), 6.0f)) / 6.0f;
            else if (op_sub == 25u) result = fminf(fmaxf(lhs / 6.0f + 0.5f, 0.0f), 1.0f);
            else if (op_sub == 26u) { float sp = fmaxf(lhs, 0.0f) + logf(1.0f + expf(-fabsf(lhs))); result = lhs * tanhf(sp); }
            else if (op_sub == 27u) result = lhs / (1.0f + fabsf(lhs));
            else if (op_sub == 28u) result = fminf(lhs, 0.0f) - logf(1.0f + expf(-fabsf(lhs)));
            else                    result = lhs;
        } else if (op_kind == 1u) {
            // Cast — at the f32-arena layer this is identity. The
            // Cast step is preserved in the chain so the IR shape
            // information stays intact for downstream passes.
            result = lhs;
        } else {
            float rhs;
            {
                unsigned int kind = rhs_enc >> 31;
                unsigned int idx  = rhs_enc & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(
                    i, kind, idx, prologue_row0, has_prologue_row0, prologue_input,
                    scalar_input_mask, input_modulus);
                rhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            if (op_kind == 2u) {
                // Binary. op_sub: 0=Add, 1=Sub, 2=Mul, 3=Div,
                // 4=Max, 5=Min, 6=Pow.
                if      (op_sub == 0u) result = lhs + rhs;
                else if (op_sub == 1u) result = lhs - rhs;
                else if (op_sub == 2u) result = lhs * rhs;
                else if (op_sub == 3u) result = lhs / rhs;
                else if (op_sub == 4u) result = fmaxf(lhs, rhs);
                else if (op_sub == 5u) result = fminf(lhs, rhs);
                else                   result = powf(lhs, rhs);
            } else {
                // Compare. op_sub: 0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge.
                bool b;
                if      (op_sub == 0u) b = (lhs == rhs);
                else if (op_sub == 1u) b = (lhs != rhs);
                else if (op_sub == 2u) b = (lhs <  rhs);
                else if (op_sub == 3u) b = (lhs <= rhs);
                else if (op_sub == 4u) b = (lhs >  rhs);
                else                   b = (lhs >= rhs);
                result = b ? 1.0f : 0.0f;
            }
        }

        scratch[k] = result;
        last_idx = k;
    }

    arena[dst_off + i] = scratch[last_idx];
}
