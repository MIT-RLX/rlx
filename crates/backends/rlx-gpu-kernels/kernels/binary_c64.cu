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

//
// Element-wise C64 binary op, dispatched over the complex-element index
// `k in [0, n)`. C64 = 2 f32 lanes [re, im] (8 B/elem), so each thread reads
// BOTH lanes of its operands — the reason this cannot ride the fused
// scalar-per-thread `elementwise_region` path (that model can't reach the
// partner `im` lane). Formulas mirror rlx-cpu `exec_binary_full_c64`:
//   Add: (ar+br, ai+bi)   Sub: (ar-br, ai-bi)
//   Mul: (ar*br - ai*bi, ar*bi + ai*br)
//   Div: d = br*br + bi*bi; ((ar*br + ai*bi)/d, (ai*br - ar*bi)/d)
// Max/Min/Pow are rejected at lowering (undefined for complex).
//
// Broadcast: `n_a` / `n_b` are the operands' complex-element counts. Indexing
// uses `k % n_a` / `k % n_b` (complex-element units), matching the CPU modulo
// fallback — a scalar operand (count 1) reads element 0 for every k. Offsets
// are f32-ELEMENT offsets (lane j of complex element m is `off + 2*m + j`),
// declared `unsigned long long` so the host MUST pass u64 to match.

extern "C" __global__ void binary_c64(
    float* arena,
    unsigned int n,
    unsigned long long a_off,
    unsigned long long b_off,
    unsigned long long c_off,
    unsigned int op,
    unsigned int n_a,
    unsigned int n_b)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long k = (unsigned long long)gid;
    unsigned long long ka = (unsigned long long)(gid % n_a);
    unsigned long long kb = (unsigned long long)(gid % n_b);
    float ar = arena[a_off + 2ull * ka];
    float ai = arena[a_off + 2ull * ka + 1u];
    float br = arena[b_off + 2ull * kb];
    float bi = arena[b_off + 2ull * kb + 1u];
    float cr = 0.0f;
    float ci = 0.0f;
    switch (op) {
        case 0u: // add
            cr = ar + br;
            ci = ai + bi;
            break;
        case 1u: // sub
            cr = ar - br;
            ci = ai - bi;
            break;
        case 2u: // mul
            cr = ar * br - ai * bi;
            ci = ar * bi + ai * br;
            break;
        case 3u: { // div
            float d = br * br + bi * bi;
            cr = (ar * br + ai * bi) / d;
            ci = (ai * br - ar * bi) / d;
            break;
        }
        default:
            break;
    }
    arena[c_off + 2ull * k]      = cr;
    arena[c_off + 2ull * k + 1u] = ci;
}
