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

// Tensor Core SGEMM via WMMA. Same call shape as `matmul.cu` but each
// warp computes a 16×16 fragment using fp16 inputs / fp32 accumulators.
// We keep the arena f32 and cast to half on shared-memory load — cheaper
// than the saved FMAs once Tensor Cores kick in (4-8× over the scalar
// kernel on Ampere+).
//
// Block: 4 warps × 32 threads = 128 threads, computing a 32×64 block
// tile of C (2 warp-rows × 4 warp-cols). Inner-K tile = 16.
//
// Requires SM 70+. NVRTC compiles against CUDA's `-arch=compute_*` —
// the load_module call will fail at runtime on pre-Volta cards; we
// fall back to the scalar kernel via runtime arch detection in
// backend.rs.

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_PER_BLOCK 4
#define WARP_M 2
#define WARP_N 4
#define BM (WARP_M * WMMA_M)   // 32
#define BN (WARP_N * WMMA_N)   // 64
#define BK WMMA_K              // 16
#define THREADS (32 * WARPS_PER_BLOCK)  // 128

extern "C" __global__ void matmul_wmma(
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
    unsigned int warp_id = tid >> 5;       // 0..3
    unsigned int lane    = tid & 31u;      // 0..31

    // Each warp owns one (warp_row, warp_col) tile of the block tile.
    unsigned int warp_row = warp_id / WARP_N;  // 0..1
    unsigned int warp_col = warp_id % WARP_N;  // 0..3

    unsigned int a_base = a_off + bz * a_batch_stride;
    unsigned int b_base = b_off + bz * b_batch_stride;
    unsigned int c_base = c_off + bz * c_batch_stride;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    unsigned int n_tiles = (k + BK - 1) / BK;

    // Cooperative loads: 128 threads load BM*BK = 32*16 = 512 A elems
    // and BK*BN = 16*64 = 1024 B elems per K-tile. 4 A elems/thread,
    // 8 B elems/thread.
    const unsigned int A_PER_THREAD = (BM * BK) / THREADS;  // 4
    const unsigned int B_PER_THREAD = (BK * BN) / THREADS;  // 8

    for (unsigned int t = 0; t < n_tiles; ++t) {
        #pragma unroll
        for (unsigned int li = 0; li < A_PER_THREAD; ++li) {
            unsigned int idx = tid + li * THREADS;
            unsigned int r = idx / BK;
            unsigned int c = idx % BK;
            unsigned int gr = blockIdx.y * BM + r;
            unsigned int gc = t * BK + c;
            float v = (gr < m && gc < k) ? arena[a_base + gr * k + gc] : 0.0f;
            tile_a[r][c] = __float2half(v);
        }
        #pragma unroll
        for (unsigned int li = 0; li < B_PER_THREAD; ++li) {
            unsigned int idx = tid + li * THREADS;
            unsigned int r = idx / BN;
            unsigned int c = idx % BN;
            unsigned int gr = t * BK + r;
            unsigned int gc = blockIdx.x * BN + c;
            float v = (gr < k && gc < n) ? arena[b_base + gr * n + gc] : 0.0f;
            tile_b[r][c] = __float2half(v);
        }

        __syncthreads();

        // Each warp computes its 16×16 fragment.
        wmma::load_matrix_sync(a_frag,
            (const __half*)&tile_a[warp_row * WMMA_M][0], BK);
        wmma::load_matrix_sync(b_frag,
            (const __half*)&tile_b[0][warp_col * WMMA_N], BN);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // Store fragment to shared, then masked write to global.
    __shared__ float tile_c[BM][BN];
    wmma::store_matrix_sync(
        (float*)&tile_c[warp_row * WMMA_M][warp_col * WMMA_N],
        acc_frag, BN, wmma::mem_row_major);
    __syncthreads();

    // Cooperative store-back: BM*BN = 2048 elems, 128 threads → 16 each.
    const unsigned int C_PER_THREAD = (BM * BN) / THREADS;  // 16
    #pragma unroll
    for (unsigned int li = 0; li < C_PER_THREAD; ++li) {
        unsigned int idx = tid + li * THREADS;
        unsigned int r = idx / BN;
        unsigned int c = idx % BN;
        unsigned int gr = blockIdx.y * BM + r;
        unsigned int gc = blockIdx.x * BN + c;
        if (gr < m && gc < n) {
            arena[c_base + gr * n + gc] = tile_c[r][c];
        }
    }
}
