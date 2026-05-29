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

// rlx_mlx_shim.h — C ABI over MLX's C++ API.
//
// We expose just enough of MLX to (a) build a graph of `array` handles
// from rlx-ir's Op vocabulary, (b) eval that graph (lazy mode) or eval
// after each op (eager mode), and (c) read results back as host-side
// f32. Errors come back as int return codes; the message is in
// thread-local storage, fetched via rlx_mlx_last_error().
//
// All handles are opaque pointers. Lifetime is caller-owned: every
// constructor / op returns a fresh handle that the caller must
// rlx_mlx_array_free().

#ifndef RLX_MLX_SHIM_H
#define RLX_MLX_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rlx_mlx_array_s rlx_mlx_array_t;

// Element dtype, mirrors rlx_ir::DType. MLX has native dtypes for
// every variant rlx-ir declares, so the mapping is total.
typedef enum {
    RLX_MLX_DTYPE_F32  = 0,
    RLX_MLX_DTYPE_F16  = 1,
    RLX_MLX_DTYPE_BF16 = 2,
    RLX_MLX_DTYPE_I32  = 3,
    RLX_MLX_DTYPE_F64  = 4,
    RLX_MLX_DTYPE_I8   = 5,
    RLX_MLX_DTYPE_I16  = 6,
    RLX_MLX_DTYPE_I64  = 7,
    RLX_MLX_DTYPE_U8   = 8,
    RLX_MLX_DTYPE_U32  = 9,
    RLX_MLX_DTYPE_BOOL = 10,
} rlx_mlx_dtype_t;

// Return codes. Anything non-zero means "check rlx_mlx_last_error()".
#define RLX_MLX_OK 0
#define RLX_MLX_ERR_GENERIC 1
#define RLX_MLX_ERR_BAD_DTYPE 2
#define RLX_MLX_ERR_BAD_SHAPE 3

// Last-error retrieval — thread-local C string, valid until the next
// shim call on this thread. Never returns NULL (returns "" if empty).
const char* rlx_mlx_last_error(void);

// Set the thread-local last-error message. Used by Rust callbacks
// (mlx::compile lowering, etc) to propagate diagnostics across the
// FFI boundary so downstream callers see the actual cause instead
// of a generic "rust callback failed".
void rlx_mlx_set_last_error(const char* msg);

// Build a leaf array from host-side f32 data.
//   shape:  pointer to ndim ints
//   ndim:   rank
//   data:   ndim-product f32s
//   dtype:  the dtype to cast `data` into on the MLX side
// Returns a fresh handle in *out, or non-zero on error.
int rlx_mlx_array_from_data(
    const int* shape, size_t ndim,
    const float* data, size_t nelems,
    rlx_mlx_dtype_t dtype,
    rlx_mlx_array_t** out);

// Build a leaf array directly from raw bytes in the target dtype —
// no f32 widen/narrow round-trip. Useful when callers already hold
// half-precision (F16/BF16) buffers; otherwise from_data is fine.
//   nbytes must equal nelems * dtype_size_bytes(dtype).
int rlx_mlx_array_from_bytes(
    const int* shape, size_t ndim,
    const void* data, size_t nbytes,
    rlx_mlx_dtype_t dtype,
    rlx_mlx_array_t** out);

// Read the array's contents back as raw bytes in its native dtype.
// Forces eval; falls back to mc::contiguous when the array is a
// strided view (same logic as array_to_f32).
//   dst_cap: capacity of dst in bytes
//   *out_nbytes: bytes written
int rlx_mlx_array_to_bytes(
    rlx_mlx_array_t* h,
    void* dst, size_t dst_cap, size_t* out_nbytes);

// Element-size for a given dtype, in bytes. Convenience for callers
// sizing buffers around from_bytes / to_bytes.
size_t rlx_mlx_dtype_size(rlx_mlx_dtype_t dtype);

// Free a handle. Safe on NULL.
void rlx_mlx_array_free(rlx_mlx_array_t* h);

// Clone an array handle — produces a fresh wrapper around the same
// underlying mc::array (shared_ptr-counted, so cheap). Used by the
// sub-graph compose path to bind parent-env captures into a new env
// without giving up the parent's ownership.
int rlx_mlx_array_clone(rlx_mlx_array_t* h, rlx_mlx_array_t** out);

// Read the array's shape into `out_shape` (capacity `cap`); writes
// the actual rank to *out_ndim. If cap < rank, returns RLX_MLX_ERR_BAD_SHAPE.
int rlx_mlx_array_shape(
    rlx_mlx_array_t* h,
    int* out_shape, size_t cap, size_t* out_ndim);

// Force eval (if not already evaluated) and copy the array's contents
// into `dst` as f32. `nelems` is the destination capacity. Returns
// RLX_MLX_ERR_BAD_SHAPE if the array has more elements than `nelems`.
int rlx_mlx_array_to_f32(
    rlx_mlx_array_t* h,
    float* dst, size_t nelems);

// Eval a batch of arrays (forces materialization on the MLX side).
// Used by lazy-mode execution to fire all outputs at once.
int rlx_mlx_eval(rlx_mlx_array_t* const* handles, size_t n);

// Schedule the same arrays for eval but return immediately. Pair with
// rlx_mlx_synchronize() (or rlx_mlx_eval, which also drains) when you
// need the results to actually be ready. Used by commit_no_wait.
int rlx_mlx_async_eval(rlx_mlx_array_t* const* handles, size_t n);

// Wait for all in-flight async work on every MLX stream.
int rlx_mlx_synchronize(void);

// ── Ops ──────────────────────────────────────────────────────────
// Each op produces a new handle. In eager mode the Rust side calls
// rlx_mlx_eval() on the result immediately; in lazy mode it defers
// until the whole graph is built.

int rlx_mlx_op_matmul(rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_add   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_mul   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_sub   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_div   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);

int rlx_mlx_op_softmax (rlx_mlx_array_t* a, int axis, rlx_mlx_array_t** out);
int rlx_mlx_op_gelu    (rlx_mlx_array_t* a, rlx_mlx_array_t** out);
int rlx_mlx_op_silu    (rlx_mlx_array_t* a, rlx_mlx_array_t** out);
int rlx_mlx_op_cast    (rlx_mlx_array_t* a, rlx_mlx_dtype_t dtype, rlx_mlx_array_t** out);

// LayerNorm with explicit gamma/beta (beta may be NULL for no-bias).
int rlx_mlx_op_layernorm(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* gamma,
    rlx_mlx_array_t* beta_or_null,
    float eps,
    rlx_mlx_array_t** out);

// ── Binary (max/min/pow round out the basic set) ──────────────────
int rlx_mlx_op_max  (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_min  (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_pow  (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);

// ── Linalg ────────────────────────────────────────────────────────
// Dense linear solve A·x = b via mlx::core::linalg::solve.
// Accepts both rank-2 A (single system) and rank-3+ A (batched, with
// the leading dims as the batch axis) — same C entry point covers
// rlx-ir's `Op::DenseSolve` and `Op::BatchedDenseSolve`. Dtype must
// promote to float32 or float64 (MLX linalg constraint).
int rlx_mlx_op_solve(rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);

// ── Custom Metal kernel dispatch ─────────────────────────────────
// Wraps mlx::core::fast::metal_kernel for caller-supplied MSL.
// v1: single output — sufficient for batched LU+solve and similar
// per-system reductions. Multi-output is a follow-up if needed.
//
// `source` is the kernel BODY (not a full Metal function). MLX wraps
// it in `[[kernel]] void custom_kernel_<name>(...)` with parameters
// auto-generated from `input_names` + each input array's dtype.
// `header` is plain text inserted before the function signature
// (good for `#define`s, helper functions, includes).
//
// MLX scans `source` for built-in attribute names like
// `thread_position_in_threadgroup` and adds them with the right
// `[[…]]` Metal annotations. It also scans for `<name>_shape`,
// `<name>_strides`, `<name>_ndim` per input and injects those as
// extra `const constant` buffers when present.
//
// MLX requires the dispatch to land on the GPU stream — this entry
// point intentionally lets MLX pick its default GPU stream. Returns
// non-zero with rlx_mlx_last_error() set on any MLX-side validation
// or compile failure.
int rlx_mlx_op_metal_kernel_dispatch(
    const char*           name,
    const char*           source,
    const char*           header,
    const char* const*    input_names,
    size_t                n_inputs,
    const char*           output_name,
    rlx_mlx_array_t* const* inputs,
    const int*            output_shape,
    size_t                output_rank,
    rlx_mlx_dtype_t       output_dtype,
    int                   grid_x, int grid_y, int grid_z,
    int                   tg_x,   int tg_y,   int tg_z,
    rlx_mlx_array_t**     out);

// ── Comparisons (return bool tensor) ──────────────────────────────
int rlx_mlx_op_eq   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_ne   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_lt   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_le   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_gt   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);
int rlx_mlx_op_ge   (rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out);

// where(cond, x, y) — cond should be a bool tensor.
int rlx_mlx_op_where(
    rlx_mlx_array_t* cond,
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* y,
    rlx_mlx_array_t** out);

// ── Unary (everything that's a single named MLX function) ────────
typedef enum {
    RLX_MLX_UN_RELU = 0,
    RLX_MLX_UN_SIGMOID = 1,
    RLX_MLX_UN_TANH = 2,
    RLX_MLX_UN_EXP = 3,
    RLX_MLX_UN_LOG = 4,
    RLX_MLX_UN_SQRT = 5,
    RLX_MLX_UN_RSQRT = 6,
    RLX_MLX_UN_NEG = 7,
    RLX_MLX_UN_ABS = 8,
    RLX_MLX_UN_ERF = 9,
    RLX_MLX_UN_ROUND = 10,
    RLX_MLX_UN_SIN = 11,
    RLX_MLX_UN_COS = 12,
    RLX_MLX_UN_TAN = 13,
    RLX_MLX_UN_ATAN = 14,
} rlx_mlx_unary_t;
int rlx_mlx_op_unary(rlx_mlx_array_t* a, rlx_mlx_unary_t kind, rlx_mlx_array_t** out);

// ── Shape ops ─────────────────────────────────────────────────────
int rlx_mlx_op_reshape(
    rlx_mlx_array_t* a,
    const int* new_shape, size_t ndim,
    rlx_mlx_array_t** out);

int rlx_mlx_op_transpose(
    rlx_mlx_array_t* a,
    const int* perm, size_t ndim,
    rlx_mlx_array_t** out);

// Slice with stride 1 in every dim.
int rlx_mlx_op_slice(
    rlx_mlx_array_t* a,
    const int* start, const int* stop, size_t ndim,
    rlx_mlx_array_t** out);

int rlx_mlx_op_concat(
    rlx_mlx_array_t* const* arrays, size_t n,
    int axis,
    rlx_mlx_array_t** out);

int rlx_mlx_op_broadcast_to(
    rlx_mlx_array_t* a,
    const int* shape, size_t ndim,
    rlx_mlx_array_t** out);

// take_along_axis-style: indices is an integer array picking elements
// along `axis`. Maps rlx's Op::Gather { axis }.
int rlx_mlx_op_take(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    int axis,
    rlx_mlx_array_t** out);

// ── Reductions ────────────────────────────────────────────────────
typedef enum {
    RLX_MLX_RED_SUM = 0,
    RLX_MLX_RED_MEAN = 1,
    RLX_MLX_RED_MAX = 2,
    RLX_MLX_RED_MIN = 3,
    RLX_MLX_RED_PROD = 4,
} rlx_mlx_reduce_t;

int rlx_mlx_op_reduce(
    rlx_mlx_array_t* a,
    rlx_mlx_reduce_t kind,
    const int* axes, size_t n_axes,
    int keep_dim,
    rlx_mlx_array_t** out);

int rlx_mlx_op_cumsum(
    rlx_mlx_array_t* a,
    int axis,
    int exclusive,
    rlx_mlx_array_t** out);

// 1D FFT along the last axis (2N real-block f32/f64 or complex64).
// `inverse` non-zero selects ifft. `norm_tag` matches rlx_ir::FftNorm
// (0=Backward, 1=Forward, 2=Ortho).
int rlx_mlx_op_fft(
    rlx_mlx_array_t* a,
    int inverse,
    int norm_tag,
    rlx_mlx_array_t** out);

// ── PR2: norms + attention ───────────────────────────────────────

// RMSNorm with explicit weight (gamma) and eps. Maps Op::RmsNorm.
int rlx_mlx_op_rmsnorm(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* gamma,
    float eps,
    rlx_mlx_array_t** out);

// Mask kind for attention. Mirrors rlx_ir::op::MaskKind.
typedef enum {
    RLX_MLX_MASK_NONE = 0,
    RLX_MLX_MASK_CAUSAL = 1,
    // The two below pass an explicit mask through `mask` parameter:
    //   SLIDING — caller pre-computes sliding-window mask (MLX has no
    //             native sliding-window mode).
    //   CUSTOM  — caller supplies the mask tensor directly.
    RLX_MLX_MASK_SLIDING = 2,
    RLX_MLX_MASK_CUSTOM = 3,
} rlx_mlx_mask_t;

// Scaled dot-product attention. `q`, `k`, `v` are shape-broadcastable
// per MLX semantics (typically [B, H, S, D]). `scale` is the 1/sqrt(D)
// factor. `mask` may be NULL when mask_kind is NONE or CAUSAL.
int rlx_mlx_op_attention(
    rlx_mlx_array_t* q,
    rlx_mlx_array_t* k,
    rlx_mlx_array_t* v,
    float scale,
    rlx_mlx_mask_t mask_kind,
    rlx_mlx_array_t* mask_or_null,
    rlx_mlx_array_t** out);

// ── PR3: heavy ops ───────────────────────────────────────────────

// 2D convolution. Input/weight expected in MLX's NHWC convention
// (caller transposes from NCHW upstream).
int rlx_mlx_op_conv2d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int dil_h,    int dil_w,
    int groups,
    rlx_mlx_array_t** out);

int rlx_mlx_op_conv1d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride, int padding, int dilation, int groups,
    rlx_mlx_array_t** out);

int rlx_mlx_op_conv3d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d,    int pad_h,    int pad_w,
    int dil_d,    int dil_h,    int dil_w,
    int groups,
    rlx_mlx_array_t** out);

// Force `a` into row-major contiguous storage. Maps `mc::contiguous`.
// Needed after a transpose that MLX's compile pass would otherwise
// elide as a strided view: the elision corrupts the readback layout.
int rlx_mlx_op_contiguous(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t** out);

// Custom Metal kernel for 2D max-pool backward. NCHW-layout `x`/`dy`,
// produces dx [N, C, H, W] f32. Atomic-accumulating scatter (handles
// stride < kernel via per-thread argmax + atomic_fetch_add). Generally
// 5–10× faster than the primitive-composition lowering on shapes where
// MLX's `scatter_add_axis` is the bottleneck. Caller frees `*out`.
int rlx_mlx_op_maxpool2d_backward_metal(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* dy,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw,
    int sh, int sw,
    int ph, int pw,
    rlx_mlx_array_t** out);

// Element-wise gather along a single axis. `a` and `indices` must
// broadcast on non-axis dims; output has `indices`'s shape (broadcast
// up). Maps `mc::take_along_axis`.
int rlx_mlx_op_take_along_axis(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    int axis,
    rlx_mlx_array_t** out);

// Element-wise scatter-add along a single axis. `a`, `indices`, and
// `updates` must all have the same rank; non-axis dims must match
// exactly. For each multi-index i, a[i_axis_replaced_by_indices[i]]
// += updates[i]. Maps `mc::scatter_add_axis`.
int rlx_mlx_op_scatter_add_axis(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    rlx_mlx_array_t* updates,
    int axis,
    rlx_mlx_array_t** out);

// General N-D convolution with asymmetric padding, kernel dilation,
// input dilation (zero-stuffing), and an optional kernel-flip flag —
// the full surface MLX exposes for representing transposed/backward
// convs. Input/weight in MLX's channels-last layout.
int rlx_mlx_op_conv_general(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    const int* stride, size_t stride_n,
    const int* padding_lo, size_t padding_lo_n,
    const int* padding_hi, size_t padding_hi_n,
    const int* kernel_dilation, size_t kernel_dilation_n,
    const int* input_dilation, size_t input_dilation_n,
    int groups,
    int flip,
    rlx_mlx_array_t** out);

// argpartition along an axis — used by Top-K to get the indices of
// the k largest values without sorting the rest.
int rlx_mlx_op_argpartition(
    rlx_mlx_array_t* a,
    int kth, int axis,
    rlx_mlx_array_t** out);

// Scatter-add: `a` updated at `indices` along `axis` with `updates`,
// summed wherever multiple updates target the same row.
int rlx_mlx_op_scatter_add(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    rlx_mlx_array_t* updates,
    int axis,
    rlx_mlx_array_t** out);

// gather_mm: indexed batched matmul. `a` is [M, K], `b` is
// [num_experts, K, N], `idx` is [M] (int) — output[i] = a[i] @ b[idx[i]].
int rlx_mlx_op_gather_mm(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* b,
    rlx_mlx_array_t* idx,
    rlx_mlx_array_t** out);

// quantized_matmul: w is the packed quantized weight matrix; scales
// (and optional biases) come from the rlx QuantScheme metadata. bits
// = 4 or 8 depending on scheme. Returns x @ dequant(w).
int rlx_mlx_op_quantized_matmul(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* w,
    rlx_mlx_array_t* scales,
    rlx_mlx_array_t* biases_or_null,
    int transpose,
    int group_size,
    int bits,
    rlx_mlx_array_t** out);

// Multinomial categorical sample. `logits` shape [..., vocab]; output
// shape [..., 1] (single sample per row). `seed` controls the PRNG;
// pass 0 for "use a thread-local counter."
int rlx_mlx_op_categorical(
    rlx_mlx_array_t* logits,
    int axis,
    uint64_t seed,
    rlx_mlx_array_t** out);

// Argmax along an axis. Returns int32 indices.
int rlx_mlx_op_argmax(
    rlx_mlx_array_t* a,
    int axis, int keep_dim,
    rlx_mlx_array_t** out);

// Strided slice — same as op_slice but with explicit per-axis strides.
// Used by Pool to extract sliding windows.
int rlx_mlx_op_slice_strided(
    rlx_mlx_array_t* a,
    const int* start, const int* stop, const int* strides, size_t ndim,
    rlx_mlx_array_t** out);

// Constant-pad an array. `low` and `high` are per-axis pad widths
// (each `ndim` ints). `pad_value` is host-side f32; the shim wraps
// it as a 0-d array internally.
int rlx_mlx_op_pad(
    rlx_mlx_array_t* a,
    const int* low, const int* high, size_t ndim,
    float pad_value,
    rlx_mlx_array_t** out);

// Top-k VALUES along an axis (sorted descending). Used by Sample
// top_k filtering — the k-th largest value becomes the threshold.
// argpartition (we already expose it) is the indices counterpart.
int rlx_mlx_op_topk_values(
    rlx_mlx_array_t* a,
    int k, int axis,
    rlx_mlx_array_t** out);

// Sort along an axis (ascending). Used by Sample top_p to compute
// the nucleus threshold via cumulative sum over sorted probabilities.
int rlx_mlx_op_sort(
    rlx_mlx_array_t* a,
    int axis,
    rlx_mlx_array_t** out);

// ── PR7: persistent compiled graphs (mlx::compile) ───────────────
//
// Rust supplies a lowering callback; we wrap it as a std::function and
// hand it to mc::compile. The returned compiled-fn handle replays the
// optimized trace on subsequent calls (until input shapes change).
//
// The callback contract:
//   - `inputs` is an array of mlx_array_t* the callback may use as
//     leaves (it does NOT own them — does not free).
//   - The callback writes its output handles into `out_outputs`
//     (capacity `cap`) and stores the produced count in `*out_n_outputs`.
//     Ownership of the output handles transfers to C++ — the callback
//     must NOT free them.
//   - Returns RLX_MLX_OK on success; non-zero is treated as failure
//     and surfaces via rlx_mlx_last_error() (which the callback should
//     have set).

typedef int (*rlx_mlx_lower_fn)(
    void* ud,
    rlx_mlx_array_t* const* inputs, size_t n_inputs,
    rlx_mlx_array_t** out_outputs, size_t cap, size_t* out_n_outputs);

typedef struct rlx_mlx_compiled_s rlx_mlx_compiled_t;

int rlx_mlx_compile(
    rlx_mlx_lower_fn fn, void* ud,
    int shapeless,
    rlx_mlx_compiled_t** out);

int rlx_mlx_compiled_call(
    rlx_mlx_compiled_t* compiled,
    rlx_mlx_array_t* const* inputs, size_t n_inputs,
    rlx_mlx_array_t** out_outputs, size_t cap, size_t* out_n_outputs);

void rlx_mlx_compiled_free(rlx_mlx_compiled_t* compiled);

// MLX runtime version (string is statically allocated, do not free).
const char* rlx_mlx_version(void);

// Default-device name (e.g. "Apple M2 Pro") via mc::device_info().
// Returns a pointer to thread-local storage; copy if you need to keep
// it. Returns "" if the info isn't available.
const char* rlx_mlx_device_name(void);

#ifdef __cplusplus
}
#endif

#endif // RLX_MLX_SHIM_H
