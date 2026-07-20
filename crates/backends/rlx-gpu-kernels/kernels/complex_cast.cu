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
// Standalone complex `Op::Cast` on the f32-uniform arena, dispatched over the
// complex-element index `k in [0, n)`. Representation:
//   C64  = 2 f32 lanes [re, im]                       (8 B/elem)
//   C128 = 4 f32 lanes [re_hi, re_lo, im_hi, im_lo] df64  (16 B/elem)
//
// Every source of a real->complex cast comes from an f32 real (lo=0), so all
// six directions are pure lane MOVES — no compensated df64 arithmetic. The
// C128->C64 narrow drops the `lo` lanes (keeps `hi`); the widen sets them 0.
//
//   mode 0 real->C64 : out[2k]=in[k];   out[2k+1]=0
//   mode 1 C64->real : out[k]=in[2k]
//   mode 2 real->C128: out[4k]=in[k];   out[4k+1..3]=0
//   mode 3 C128->real: out[k]=in[4k]
//   mode 4 C64->C128 : out[4k]=in[2k]; out[4k+1]=0; out[4k+2]=in[2k+1]; out[4k+3]=0
//   mode 5 C128->C64 : out[2k]=in[4k]; out[2k+1]=in[4k+2]
//
// `in_off` / `out_off` are f32-ELEMENT offsets (the start lane of each tensor),
// declared `unsigned long long` so arenas > 4 GiB index correctly — the host
// MUST pass u64 to match (a u32 leaves the high word as stack garbage).

extern "C" __global__ void complex_cast(
    float* arena,
    unsigned int n,
    unsigned long long in_off,
    unsigned long long out_off,
    unsigned int mode)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long k = (unsigned long long)gid;
    unsigned long long i = in_off;
    unsigned long long o = out_off;
    switch (mode) {
        case 0u: // real -> C64
            arena[o + 2ull * k]      = arena[i + k];
            arena[o + 2ull * k + 1u] = 0.0f;
            break;
        case 1u: // C64 -> real
            arena[o + k] = arena[i + 2ull * k];
            break;
        case 2u: // real -> C128
            arena[o + 4ull * k]      = arena[i + k];
            arena[o + 4ull * k + 1u] = 0.0f;
            arena[o + 4ull * k + 2u] = 0.0f;
            arena[o + 4ull * k + 3u] = 0.0f;
            break;
        case 3u: // C128 -> real
            arena[o + k] = arena[i + 4ull * k];
            break;
        case 4u: // C64 -> C128
            arena[o + 4ull * k]      = arena[i + 2ull * k];
            arena[o + 4ull * k + 1u] = 0.0f;
            arena[o + 4ull * k + 2u] = arena[i + 2ull * k + 1u];
            arena[o + 4ull * k + 3u] = 0.0f;
            break;
        case 5u: // C128 -> C64
            arena[o + 2ull * k]      = arena[i + 4ull * k];
            arena[o + 2ull * k + 1u] = arena[i + 4ull * k + 2u];
            break;
        default:
            break;
    }
}
