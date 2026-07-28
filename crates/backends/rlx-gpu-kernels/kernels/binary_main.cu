// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Plumbing half of the standalone `binary` kernel. The per-op scalar math
// (`rlx_binary_apply`) is @generated once from the shared rlxsl manifest and
// prepended to this file by build.rs. Shared by CUDA (NVRTC) and HIP (hiprtc).

extern "C" __global__ void binary(
    float* arena,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int op
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = arena[a_off + i];
    float b = arena[b_off + i];
    arena[c_off + i] = rlx_binary_apply(op, a, b);
}
