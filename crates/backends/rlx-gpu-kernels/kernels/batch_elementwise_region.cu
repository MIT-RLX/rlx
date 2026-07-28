// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// FKL batch horizontal fusion: one launch over N identical slice chains.
// Grid: (ceil(slice_len/block), 1, num_batch). blockIdx.z selects the slice.
// Requires prologue == 0 (no resize prologue on batch slices).

struct InputModulus { unsigned int v[16]; };

__device__ unsigned int region_resolve_row(
    unsigned int gid,
    unsigned int kind,
    unsigned int idx,
    unsigned int scalar_input_mask,
    InputModulus input_modulus
) {
    if (kind != 0u) { return 0u; }
    if ((scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (input_modulus.v[idx] != 0u) { return gid % input_modulus.v[idx]; }
    return gid;
}

extern "C" __global__ void batch_elementwise_region(
    float* arena,
    unsigned int slice_len,
    unsigned int num_batch,
    unsigned int num_steps,
    unsigned int base_dst_off,
    unsigned int slice_elems,
    const unsigned int* batch_input_offs,
    const unsigned int* meta,
    unsigned int scalar_input_mask,
    InputModulus input_modulus
) {
    const unsigned int batch_idx = blockIdx.z;
    if (batch_idx >= num_batch) {
        return;
    }

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slice_len) {
        return;
    }

    const unsigned int* chain = meta + 16;
    unsigned int input_offs[16];
    for (unsigned int k = 0; k < 16u; ++k) {
        input_offs[k] = 0u;
    }
    input_offs[0] = batch_input_offs[batch_idx];
    const unsigned int dst_off = base_dst_off + batch_idx * slice_elems;

    float scratch[32];
    unsigned int last_idx = 0;

    for (unsigned int k = 0; k < num_steps; ++k) {
        unsigned int base    = k * 4u;
        unsigned int op_kind = chain[base + 0u];
        unsigned int op_sub  = chain[base + 1u];
        unsigned int lhs_enc = chain[base + 2u];
        unsigned int rhs_enc = chain[base + 3u];

        float lhs;
        {
            unsigned int kind = lhs_enc >> 31;
            unsigned int idx  = lhs_enc & 0x7FFFFFFFu;
            unsigned int row = region_resolve_row(i, kind, idx, scalar_input_mask, input_modulus);
            lhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
        }

        float result;
        if (op_kind == 4u) {
            float cond;
            {
                unsigned int kind = op_sub >> 31;
                unsigned int idx  = op_sub & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(i, kind, idx, scalar_input_mask, input_modulus);
                cond = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            float on_false;
            {
                unsigned int kind = rhs_enc >> 31;
                unsigned int idx  = rhs_enc & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(i, kind, idx, scalar_input_mask, input_modulus);
                on_false = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            result = (cond != 0.0f) ? lhs : on_false;
        } else if (op_kind == 0u) {
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
            else                    result = lhs;
        } else if (op_kind == 1u) {
            result = lhs;
        } else {
            float rhs;
            {
                unsigned int kind = rhs_enc >> 31;
                unsigned int idx  = rhs_enc & 0x7FFFFFFFu;
                unsigned int row = region_resolve_row(i, kind, idx, scalar_input_mask, input_modulus);
                rhs = (kind == 0u) ? arena[input_offs[idx] + row] : scratch[idx];
            }
            if (op_kind == 2u) {
                if      (op_sub == 0u) result = lhs + rhs;
                else if (op_sub == 1u) result = lhs - rhs;
                else if (op_sub == 2u) result = lhs * rhs;
                else if (op_sub == 3u) result = lhs / rhs;
                else if (op_sub == 4u) result = fmaxf(lhs, rhs);
                else if (op_sub == 5u) result = fminf(lhs, rhs);
                else                   result = powf(lhs, rhs);
            } else {
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
