// Batched row-major matmul C[b,M,N] = A[b,M,K] · B[b,K,N] over the arena.
// `a_bs`/`b_bs` are the per-batch element strides (0 when that operand is not
// batched — broadcast across the batch); `mn` = M*N. One work-item per output
// element. Naive (no tiling/SLM) — the correctness baseline; oneMKL `gemm`
// is the peak-perf follow-up (see README).
__kernel void matmul(__global float* arena,
                     uint M, uint K, uint N,
                     uint off_a, uint off_b, uint off_out,
                     uint batch, uint a_bs, uint b_bs, uint mn) {
    uint gid = get_global_id(0);
    uint total = batch * mn;
    if (gid >= total) return;
    uint bz = gid / mn;
    uint rem = gid - bz * mn;
    uint row = rem / N;
    uint col = rem % N;
    uint a_base = off_a + bz * a_bs + row * K;
    uint b_base = off_b + bz * b_bs + col;
    float acc = 0.0f;
    for (uint kk = 0; kk < K; kk++) {
        acc += arena[a_base + kk] * arena[b_base + kk * N];
    }
    arena[off_out + bz * mn + row * N + col] = acc;
}
