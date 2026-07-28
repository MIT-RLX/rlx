// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// C64 Wirtinger surface on the f32-uniform arena (interleaved [re, im] pairs).
// Formulas mirror rlx-cpu `exec_complex_norm_sq{,_backward}_f32` /
// `exec_conjugate_c64`. Dispatched over the complex-element index
// `k in [0, n)`. Offsets are f32-ELEMENT offsets (lane j of complex element m
// is `off + 2*m + j`), declared `unsigned long long` so the host MUST pass u64.
//
//   ComplexNormSq:          out[k] = re² + im²           (C64 → F32)
//   ComplexNormSqBackward: dz = g · z  (Wirtinger)       (C64, F32 → C64)
//   Conjugate:              out = (re, -im)              (C64 → C64)

extern "C" __global__ void complex_norm_sq(
    float* arena,
    unsigned int n,
    unsigned long long src_off,
    unsigned long long dst_off)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long k = (unsigned long long)gid;
    float re = arena[src_off + 2ull * k];
    float im = arena[src_off + 2ull * k + 1ull];
    arena[dst_off + k] = re * re + im * im;
}

extern "C" __global__ void complex_norm_sq_backward(
    float* arena,
    unsigned int n,
    unsigned long long z_off,
    unsigned long long g_off,
    unsigned long long dz_off)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long k = (unsigned long long)gid;
    float re = arena[z_off + 2ull * k];
    float im = arena[z_off + 2ull * k + 1ull];
    float gv = arena[g_off + k];
    arena[dz_off + 2ull * k]      = gv * re;
    arena[dz_off + 2ull * k + 1ull] = gv * im;
}

extern "C" __global__ void conjugate_c64(
    float* arena,
    unsigned int n,
    unsigned long long src_off,
    unsigned long long dst_off)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long k = (unsigned long long)gid;
    arena[dst_off + 2ull * k]      =  arena[src_off + 2ull * k];
    arena[dst_off + 2ull * k + 1ull] = -arena[src_off + 2ull * k + 1ull];
}
