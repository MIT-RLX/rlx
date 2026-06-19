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

// AMD matrix-core matmul. Uses rocWMMA — AMD's wrapper that
// abstracts over MFMA (CDNA: MI100/MI200/MI300) and WMMA (RDNA3+:
// RX 7900) hardware intrinsics with the same API surface as
// nvcuda::wmma. So this kernel is structurally identical to
// matmul_wmma.cu (in rlx-cuda) — only the namespace differs.
//
// Block: 4 wavefronts × 32 lanes (RDNA3+) or 64 lanes (CDNA). Each
// wavefront computes one 16×16 fragment of C. Block tile = 32×64.
// Inner-K tile = 16. fp16 inputs / fp32 accumulator with on-the-fly
// f32→f16 cast on tile loads (arena stays f32).
//
// hipRTC will compile this with `-x hip` plus the user's target
// architecture (gfx940 for MI300X, gfx1100 for RX 7900). rocWMMA's
// fragments dispatch to the right intrinsics per arch.
//
// ROCm 5.7+ ships rocWMMA in the standard include path; older
// installs may need to symlink. Loading this kernel is opt-in via
// `RLX_ROCM_MFMA=1` so older / unsupported configs gracefully fall
// back to the scalar matmul kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WAVES_PER_BLOCK 4
#define WAVE_M 2
#define WAVE_N 2
#define BM (WAVE_M * WMMA_M)            // 32
#define BN (WAVE_N * WMMA_N)            // 32
#define BK WMMA_K                       // 16
// rocWMMA's wavefront width is 32 on RDNA3+, 64 on CDNA. We use the
// max here for thread allocation; lanes beyond `wavefrontSize`
// participate as no-ops in the fragment ops.
#define LANES_PER_WAVE 64
#define THREADS (LANES_PER_WAVE * WAVES_PER_BLOCK)

extern "C" __global__ void matmul_mfma(
    float* arena,
    unsigned int m,
    unsigned int k,
    unsigned int n,
    unsigned int a_off,
    unsigned int b_off,
    unsigned int c_off,
    unsigned int batch,
    unsigned int a_batch_stride,
    unsigned int b_batch_stride,
    unsigned int c_batch_stride
) {
    __shared__ __half tile_a[BM][BK];
    __shared__ __half tile_b[BK][BN];

    unsigned int bz = blockIdx.z;
    if (bz >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int wave_id = tid / LANES_PER_WAVE;
    if (wave_id >= WAVES_PER_BLOCK) return;
    unsigned int wave_row = wave_id / WAVE_N;
    unsigned int wave_col = wave_id % WAVE_N;

    unsigned int a_base = a_off + bz * a_batch_stride;
    unsigned int b_base = b_off + bz * b_batch_stride;
    unsigned int c_base = c_off + bz * c_batch_stride;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    unsigned int n_tiles = (k + BK - 1) / BK;
    const unsigned int A_PER_THREAD = (BM * BK) / THREADS;
    const unsigned int B_PER_THREAD = (BK * BN) / THREADS;

    for (unsigned int t = 0; t < n_tiles; ++t) {
        // Cooperative tile loads: cast f32 → f16 on the way in.
        for (unsigned int li = 0; li < (A_PER_THREAD == 0 ? 1 : A_PER_THREAD); ++li) {
            unsigned int idx = tid + li * THREADS;
            if (idx >= BM * BK) break;
            unsigned int r = idx / BK;
            unsigned int c = idx % BK;
            unsigned int gr = blockIdx.y * BM + r;
            unsigned int gc = t * BK + c;
            float v = (gr < m && gc < k) ? arena[a_base + gr * k + gc] : 0.0f;
            tile_a[r][c] = __float2half(v);
        }
        for (unsigned int li = 0; li < (B_PER_THREAD == 0 ? 1 : B_PER_THREAD); ++li) {
            unsigned int idx = tid + li * THREADS;
            if (idx >= BK * BN) break;
            unsigned int r = idx / BN;
            unsigned int c = idx % BN;
            unsigned int gr = t * BK + r;
            unsigned int gc = blockIdx.x * BN + c;
            float v = (gr < k && gc < n) ? arena[b_base + gr * n + gc] : 0.0f;
            tile_b[r][c] = __float2half(v);
        }
        __syncthreads();

        load_matrix_sync(a_frag,
            (const __half*)&tile_a[wave_row * WMMA_M][0], BK);
        load_matrix_sync(b_frag,
            (const __half*)&tile_b[0][wave_col * WMMA_N], BN);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    __shared__ float tile_c[BM][BN];
    store_matrix_sync(
        (float*)&tile_c[wave_row * WMMA_M][wave_col * WMMA_N],
        acc_frag, BN, mem_row_major);
    __syncthreads();

    // Cooperative store-back to f32 arena.
    const unsigned int C_PER_THREAD = (BM * BN) / THREADS;
    for (unsigned int li = 0; li < (C_PER_THREAD == 0 ? 1 : C_PER_THREAD); ++li) {
        unsigned int idx = tid + li * THREADS;
        if (idx >= BM * BN) break;
        unsigned int r = idx / BN;
        unsigned int c = idx % BN;
        unsigned int gr = blockIdx.y * BM + r;
        unsigned int gc = blockIdx.x * BN + c;
        if (gr < m && gc < n) {
            arena[c_base + gr * n + gc] = tile_c[r][c];
        }
    }
}
