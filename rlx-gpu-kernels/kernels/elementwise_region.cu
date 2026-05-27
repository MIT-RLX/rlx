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

// PLAN L2 — interpreted N-ary element-wise chain kernel.
//
// One thread per output element. Each thread walks a runtime chain
// encoding (length `num_steps`, packed inside `meta`) into a private
// scratch register array; the final step's result is written to dst.
//
// `meta` layout (144 u32 words, packed by the caller):
//   meta[0..16]   = input_offs[0..16]  (only first num_inputs used)
//   meta[16..144] = chain[0..128]      (32 steps * 4 u32s)
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    const unsigned int* input_offs = meta;
    const unsigned int* chain      = meta + 16;

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
            unsigned int row;
            if (kind != 0u) { row = 0u; }
            else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
            else if (input_modulus.v[idx] != 0u) { row = i % input_modulus.v[idx]; }
            else { row = i; }
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
                unsigned int row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus.v[idx] != 0u) { row = i % input_modulus.v[idx]; }
                else { row = i; }
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                unsigned int kind = rhs_enc >> 31;
                unsigned int idx  = rhs_enc & 0x7FFFFFFFu;
                unsigned int row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus.v[idx] != 0u) { row = i % input_modulus.v[idx]; }
                else { row = i; }
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
            // Activation. op_sub: 0=Gelu, 1=GeluApprox, 2=Silu, 3=Relu,
            // 4=Sigmoid, 5=Tanh, 6=Exp, 7=Log, 8=Sqrt, 9=Rsqrt,
            // 10=Neg, 11=Abs.
            if      (op_sub == 3u) result = fmaxf(lhs, 0.0f);
            else if (op_sub == 0u || op_sub == 1u) {
                float c = 0.7978845608f;
                float inner = c * (lhs + 0.044715f * lhs * lhs * lhs);
                result = 0.5f * lhs * (1.0f + tanhf(inner));
            }
            else if (op_sub == 2u) result = lhs / (1.0f + expf(-lhs));
            else if (op_sub == 4u) result = 1.0f / (1.0f + expf(-lhs));
            else if (op_sub == 5u) result = tanhf(lhs);
            else if (op_sub == 6u) result = expf(lhs);
            else if (op_sub == 7u) result = logf(lhs);
            else if (op_sub == 8u) result = sqrtf(lhs);
            else if (op_sub == 9u) result = rsqrtf(lhs);
            else if (op_sub == 10u) result = -lhs;
            else if (op_sub == 11u) result = fabsf(lhs);
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
                unsigned int row;
                if (kind != 0u) { row = 0u; }
                else if ((scalar_input_mask & (1u << idx)) != 0u) { row = 0u; }
                else if (input_modulus.v[idx] != 0u) { row = i % input_modulus.v[idx]; }
                else { row = i; }
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
