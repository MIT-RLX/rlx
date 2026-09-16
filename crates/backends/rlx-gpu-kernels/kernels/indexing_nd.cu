// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// On-device ONNX ND indexing: GatherND, GatherElements, ScatterElements, and
// a reduction-capable ScatterND. Companions to `scatter_nd.cu`, which covers
// only ScatterND/reduction=none.
//
// These ops used to go through `Step::CpuIndexing` on every f32-uniform arena
// backend (CUDA, ROCm, wgpu) — a D2H of data+indices+updates, a CPU pass, and
// an H2D of the result, mid-graph. The reason was never the arithmetic: it is
// that an I64 index tensor is stored as f32-VALUED slots in these arenas (see
// `IndexingThunk::force_indices_f32`), so a kernel reading the slot as i64
// reinterprets float bits. Every index read below therefore goes through
// `rintf` on the f32 slot, exactly as `gather.cu` already does, and the host
// declines the launch when the thunk carries genuine packed i64 indices.
//
// Shapes and strides arrive in a `meta` u32 buffer uploaded once at compile
// time (the `slice.cu` convention), so there is no per-launch allocation and
// no rank cap.
//
// Semantics track `rlx_cpu::onnx_indexing` element for element, including its
// negative-index wrap, its clamp-to-last-valid on out-of-range, and (for
// ScatterElements) its non-finite sanitising prologue.
//
// **Duplicate destinations diverge from CPU, by ONNX's own admission.** With
// `reduction=none`, ONNX leaves ScatterND/ScatterElements undefined when two
// index tuples name the same slot. rlx-cpu resolves it sequentially — last
// update in flat order wins — while these kernels write concurrently and the
// winner is whichever thread the scheduler runs last. Both are legal; they are
// not equal. Measured, not assumed: a `[-1, 9, -4]` ScatterND into a `[4,3]`
// buffer (where -1 and 9 both resolve to row 3) returns the second update on
// CPU and the first on GPU. Use `reduction=add` when the outcome must be
// order-independent.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Read an f32-encoded index slot, wrap negatives, clamp into [0, dim).
// Mirrors the `idx.clamp(0, dim-1)` the CPU kernels apply after wrapping.
static __device__ __forceinline__ unsigned rlx_idx_f32(float raw, int dim) {
    int idx = (int)rintf(raw);
    if (idx < 0) idx += dim;
    if (idx < 0) idx = 0;
    if (dim > 0 && idx >= dim) idx = dim - 1;
    return (unsigned)idx;
}

// ---------------------------------------------------------------------------
// GatherND
// ---------------------------------------------------------------------------

// out[t*slice + j] = data[batch_base(t) + sum_m idx[t,m]*stride[m] + j]
//
// meta = [ data_strides[b .. b+k] , data_dims[b .. b+k] ] — already shifted by
// `batch_dims` on the host, so the kernel never needs `b` itself.
// One thread per OUTPUT element.
extern "C" __global__ void gather_nd_f32(
    float* arena,
    unsigned n,                 // batch_count * tuples_per_batch * slice
    unsigned data_off,
    unsigned idx_off,
    unsigned dst_off,
    unsigned k,
    unsigned slice,
    unsigned tuples_per_batch,
    unsigned batch_stride,
    const unsigned* meta
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned* strides = meta;
    const unsigned* dims = meta + k;

    unsigned t = i / slice;
    unsigned j = i - t * slice;
    unsigned bi = tuples_per_batch ? (t / tuples_per_batch) : 0u;

    unsigned off = bi * batch_stride;
    unsigned tuple = t * k;
    for (unsigned m = 0; m < k; ++m) {
        off += rlx_idx_f32(arena[idx_off + tuple + m], (int)dims[m]) * strides[m];
    }
    arena[dst_off + i] = arena[data_off + off + j];
}

// ---------------------------------------------------------------------------
// GatherElements (a.k.a. take_along_axis)
// ---------------------------------------------------------------------------

// out[lin] = data[ sum_d coord_d * dstride_d ], where coord_d decomposes `lin`
// by the INDICES' strides (ONNX lets indices be smaller than data off-axis) and
// coord_axis comes from the index value itself.
//
// meta = [ ishape[rank], dstride[rank], ostride[rank], dshape[rank] ]
// One thread per OUTPUT element. f32 data only — the host declines the launch
// when `data_elem_bytes != 4` (e.g. gathering i64 token ids).
extern "C" __global__ void gather_elements_f32(
    float* arena,
    unsigned n,                 // prod(indices_shape), clamped to out_len
    unsigned data_off,
    unsigned idx_off,
    unsigned dst_off,
    unsigned data_len,
    unsigned rank,
    unsigned axis,
    int axis_dim,
    const unsigned* meta
) {
    unsigned lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;

    const unsigned* ishape = meta;
    const unsigned* dstride = meta + rank;
    const unsigned* ostride = meta + 2u * rank;
    const unsigned* dshape = meta + 3u * rank;

    unsigned off = 0;
    for (unsigned d = 0; d < rank; ++d) {
        if (d == axis) {
            off += rlx_idx_f32(arena[idx_off + lin], axis_dim) * dstride[d];
        } else {
            unsigned dim = ishape[d] ? ishape[d] : 1u;
            unsigned coord = (lin / ostride[d]) % dim;
            unsigned last = dshape[d] ? dshape[d] - 1u : 0u;
            off += (coord < last ? coord : last) * dstride[d];
        }
    }
    // `data.get(off)` on the CPU side: an out-of-range read leaves the output
    // slot untouched rather than faulting.
    if (off < data_len) {
        arena[dst_off + lin] = arena[data_off + off];
    }
}

// ---------------------------------------------------------------------------
// ScatterElements
// ---------------------------------------------------------------------------

// Implements only the CPU kernel's *exact* path — indices rank == data rank and
// prod(indices_shape) == n — which is the one that decomposes a flat index by
// the indices' own strides. The host declines the launch for the rank-1 and
// "dense" shape-guessing fallbacks, which stay on CPU.
//
// meta = [ istride[rank], dstride[rank], dshape[rank] ]
// One thread per UPDATE element. `dst` must already hold the sanitised copy of
// `data` (see `copy_sanitize_f32`).
//
// reduction: 0 = overwrite, 1 = atomic add. Mul/Max/Min stay on the host — a
// float CAS loop would be both slower and harder to prove equal to the CPU
// result under duplicate indices.
extern "C" __global__ void scatter_elements_f32(
    float* arena,
    unsigned n,                 // min(indices_len, updates_len)
    unsigned upd_off,
    unsigned idx_off,
    unsigned dst_off,
    unsigned dst_len,
    unsigned rank,
    unsigned axis,
    unsigned reduction,
    const unsigned* meta
) {
    unsigned f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;

    const unsigned* istride = meta;
    const unsigned* dstride = meta + rank;
    const unsigned* dshape = meta + 2u * rank;

    // CPU: `indices[flat_i].max(0)` — a bare floor at 0, NOT the wrap-and-clamp
    // `rlx_idx_f32` applies elsewhere. Kept different on purpose.
    int row = (int)rintf(arena[idx_off + f]);
    if (row < 0) row = 0;

    unsigned rem = f;
    unsigned dst = 0;
    for (unsigned d = 0; d < rank; ++d) {
        unsigned c = istride[d] ? (rem / istride[d]) : 0u;
        if (istride[d]) rem = rem % istride[d];
        if (d == axis) c = (unsigned)row;
        unsigned last = dshape[d] ? dshape[d] - 1u : 0u;
        dst += (c < last ? c : last) * dstride[d];
    }
    if (dst >= dst_len) return;

    float v = arena[upd_off + f];
    if (reduction == 1u) {
        atomicAdd(&arena[dst_off + dst], v);
    } else {
        arena[dst_off + dst] = v;
    }
}

// ---------------------------------------------------------------------------
// ScatterND with reduction
// ---------------------------------------------------------------------------

// Same addressing as `scatter_nd.cu`'s `scatter_nd_f32`, but rank-general via
// `meta` and able to accumulate. reduction: 0 = overwrite, 1 = atomic add.
//
// meta = [ data_strides[k], data_dims[k] ]
// One thread per (update, slice element).
extern "C" __global__ void scatter_nd_reduce_f32(
    float* arena,
    unsigned n,                 // num_updates * slice
    unsigned idx_off,
    unsigned upd_off,
    unsigned dst_off,
    unsigned dst_len,
    unsigned k,
    unsigned slice,
    unsigned reduction,
    const unsigned* meta
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned* strides = meta;
    const unsigned* dims = meta + k;

    unsigned u = i / slice;
    unsigned j = i - u * slice;

    unsigned off = 0;
    for (unsigned m = 0; m < k; ++m) {
        off += rlx_idx_f32(arena[idx_off + u * k + m], (int)dims[m]) * strides[m];
    }
    if (off + j >= dst_len) return;

    float v = arena[upd_off + u * slice + j];
    if (reduction == 1u) {
        atomicAdd(&arena[dst_off + off + j], v);
    } else {
        arena[dst_off + off + j] = v;
    }
}

// ---------------------------------------------------------------------------
// Copy + sanitise prologue
// ---------------------------------------------------------------------------

// Seeds `dst` from `data` before a scatter, and optionally zeroes non-finite
// slots: `dst[i] = src[i]` for i < src_len, then `dst[i] = finite ? dst[i] : 0`.
//
// The zeroing reproduces `scatter_elements_f32`'s CPU prologue, which copies
// `data` into `out` and *then* zeroes every non-finite slot of `out`. Skipping
// it would diverge from CPU on any graph that scatters into a buffer holding a
// NaN — a difference that shows up as a slow drift rather than a test failure.
//
// `scatter_nd_into_f32` has no such zeroing, so ScatterND passes
// `do_sanitize = 0` and gets a plain copy. `do_copy` is 0 when `data` and `dst`
// already alias, matching the CPU's `ptr::eq` check.
extern "C" __global__ void copy_sanitize_f32(
    float* arena,
    unsigned n,                 // dst_len
    unsigned src_off,
    unsigned dst_off,
    unsigned src_len,
    unsigned do_copy,
    unsigned do_sanitize
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (do_copy && i < src_len) ? arena[src_off + i] : arena[dst_off + i];
    arena[dst_off + i] = (do_sanitize && !isfinite(v)) ? 0.0f : v;
}
