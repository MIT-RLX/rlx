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

// rlx_mlx_shim.cpp — implementation of the C ABI declared in shim.h.
//
// Every entry point follows the same pattern: catch every exception,
// stash a description in thread-local last_error storage, return a
// non-zero error code. Rust never sees a C++ exception cross the FFI
// boundary.

#include "rlx_mlx_shim.h"

#include "mlx/array.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/fast.h"
#include "mlx/fft.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

#include <cstring>
#include <exception>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace mc = mlx::core;

namespace {

// Per-thread last-error string. Cleared on every successful call so a
// stale message from a prior failure can't be read by accident.
thread_local std::string g_last_error;

void clear_error() { g_last_error.clear(); }
void set_error(const char* what) {
    g_last_error.assign(what ? what : "(null)");
}

mc::Dtype to_mlx_dtype(rlx_mlx_dtype_t d) {
    switch (d) {
        case RLX_MLX_DTYPE_F32:  return mc::float32;
        case RLX_MLX_DTYPE_F16:  return mc::float16;
        case RLX_MLX_DTYPE_BF16: return mc::bfloat16;
        case RLX_MLX_DTYPE_I32:  return mc::int32;
        case RLX_MLX_DTYPE_F64:  return mc::float64;
        case RLX_MLX_DTYPE_I8:   return mc::int8;
        case RLX_MLX_DTYPE_I16:  return mc::int16;
        case RLX_MLX_DTYPE_I64:  return mc::int64;
        case RLX_MLX_DTYPE_U8:   return mc::uint8;
        case RLX_MLX_DTYPE_U32:  return mc::uint32;
        case RLX_MLX_DTYPE_BOOL: return mc::bool_;
    }
    throw std::runtime_error("invalid dtype");
}

// MLX's `array` is a value type wrapping a shared_ptr to ArrayDesc, so
// we keep ownership trivial: each handle is a heap-allocated array.
struct Handle { mc::array a; };

inline mc::array& unwrap(rlx_mlx_array_t* h) {
    return reinterpret_cast<Handle*>(h)->a;
}

inline rlx_mlx_array_t* wrap(mc::array a) {
    return reinterpret_cast<rlx_mlx_array_t*>(new Handle{std::move(a)});
}

template <typename Fn>
int guarded(Fn&& fn) {
    clear_error();
    try {
        fn();
        return RLX_MLX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return RLX_MLX_ERR_GENERIC;
    } catch (...) {
        set_error("unknown C++ exception");
        return RLX_MLX_ERR_GENERIC;
    }
}

} // namespace

extern "C" {

const char* rlx_mlx_last_error(void) {
    return g_last_error.c_str();
}

void rlx_mlx_set_last_error(const char* msg) {
    g_last_error.assign(msg ? msg : "");
}

const char* rlx_mlx_version(void) {
    return mc::version();
}

const char* rlx_mlx_device_name(void) {
    static thread_local std::string s_name;
    s_name.clear();
    try {
        const auto& info = mc::device_info();
        auto it = info.find("device_name");
        if (it != info.end()) {
            if (auto* str = std::get_if<std::string>(&it->second)) {
                s_name = *str;
                return s_name.c_str();
            }
        }
    } catch (...) {
        // Fall through to empty.
    }
    return "";
}

int rlx_mlx_array_from_data(
    const int* shape, size_t ndim,
    const float* data, size_t nelems,
    rlx_mlx_dtype_t dtype,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s;
        s.reserve(ndim);
        size_t expected = 1;
        for (size_t i = 0; i < ndim; ++i) {
            s.push_back(shape[i]);
            expected *= static_cast<size_t>(shape[i]);
        }
        if (expected != nelems) {
            throw std::runtime_error("nelems doesn't match shape product");
        }
        // Always build as float32, then cast if requested. The float-iterator
        // constructor copies, which is what we want — caller's `data` may be
        // a transient buffer.
        mc::array f32 = mc::array(data, std::move(s), mc::float32);
        mc::array result = (dtype == RLX_MLX_DTYPE_F32)
            ? std::move(f32)
            : mc::astype(f32, to_mlx_dtype(dtype));
        *out = wrap(std::move(result));
    });
}

void rlx_mlx_array_free(rlx_mlx_array_t* h) {
    if (!h) return;
    delete reinterpret_cast<Handle*>(h);
}

int rlx_mlx_array_clone(rlx_mlx_array_t* h, rlx_mlx_array_t** out) {
    return guarded([&] {
        // mc::array is shared_ptr-backed; copying is a refcount bump.
        // Wrap in a fresh Handle so the caller has independent
        // ownership of the wrapper.
        mc::array a = unwrap(h);
        *out = wrap(std::move(a));
    });
}

size_t rlx_mlx_dtype_size(rlx_mlx_dtype_t d) {
    switch (d) {
        case RLX_MLX_DTYPE_F32:  return 4;
        case RLX_MLX_DTYPE_F16:  return 2;
        case RLX_MLX_DTYPE_BF16: return 2;
        case RLX_MLX_DTYPE_I32:  return 4;
        case RLX_MLX_DTYPE_F64:  return 8;
        case RLX_MLX_DTYPE_I8:   return 1;
        case RLX_MLX_DTYPE_I16:  return 2;
        case RLX_MLX_DTYPE_I64:  return 8;
        case RLX_MLX_DTYPE_U8:   return 1;
        case RLX_MLX_DTYPE_U32:  return 4;
        case RLX_MLX_DTYPE_BOOL: return 1;
    }
    return 0;
}

int rlx_mlx_array_from_bytes(
    const int* shape, size_t ndim,
    const void* data, size_t nbytes,
    rlx_mlx_dtype_t dtype,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s;
        s.reserve(ndim);
        size_t expected_elems = 1;
        for (size_t i = 0; i < ndim; ++i) {
            s.push_back(shape[i]);
            expected_elems *= static_cast<size_t>(shape[i]);
        }
        size_t elem_size = rlx_mlx_dtype_size(dtype);
        if (elem_size == 0) {
            throw std::runtime_error("from_bytes: invalid dtype");
        }
        if (expected_elems * elem_size != nbytes) {
            throw std::runtime_error("from_bytes: nbytes mismatch with shape*dtype_size");
        }
        // mc::array's void* + Deleter constructor would be zero-copy,
        // but we don't have a stable lifetime guarantee on `data`.
        // Allocate-and-copy via the byte-pointer iterator constructor.
        // Pick the right typed iterator based on dtype so MLX gets
        // the type info it needs.
        mc::array result = [&]() -> mc::array {
            switch (dtype) {
                case RLX_MLX_DTYPE_F32: {
                    const float* p = static_cast<const float*>(data);
                    return mc::array(p, std::move(s), mc::float32);
                }
                case RLX_MLX_DTYPE_F16: {
                    // half-storage uses uint16; widen to float32 so MLX
                    // doesn't need a typed half iterator we can't easily
                    // synthesize from raw bytes. The widen happens once
                    // here; the resulting f32 array is then astype'd to
                    // float16 lazily, which is itself cheap.
                    const uint16_t* p = static_cast<const uint16_t*>(data);
                    std::vector<float> tmp(expected_elems);
                    for (size_t i = 0; i < expected_elems; ++i) {
                        // IEEE-754 binary16 → float32 expansion.
                        uint16_t h = p[i];
                        uint32_t sign = (h & 0x8000) << 16;
                        uint32_t exp  = (h & 0x7c00) >> 10;
                        uint32_t mant = (h & 0x03ff);
                        uint32_t f;
                        if (exp == 0) {
                            if (mant == 0) f = sign;
                            else {
                                // subnormal — normalize
                                while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
                                exp++; mant &= ~0x400u;
                                f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                            }
                        } else if (exp == 31) {
                            f = sign | 0x7f800000 | (mant << 13);
                        } else {
                            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                        }
                        std::memcpy(&tmp[i], &f, 4);
                    }
                    mc::array f32_arr(tmp.data(), s, mc::float32);
                    return mc::astype(f32_arr, mc::float16);
                }
                case RLX_MLX_DTYPE_BF16: {
                    const uint16_t* p = static_cast<const uint16_t*>(data);
                    std::vector<float> tmp(expected_elems);
                    for (size_t i = 0; i < expected_elems; ++i) {
                        // bf16 → f32: pad with zeros in low 16 bits.
                        uint32_t f = static_cast<uint32_t>(p[i]) << 16;
                        std::memcpy(&tmp[i], &f, 4);
                    }
                    mc::array f32_arr(tmp.data(), s, mc::float32);
                    return mc::astype(f32_arr, mc::bfloat16);
                }
                case RLX_MLX_DTYPE_I32: {
                    const int32_t* p = static_cast<const int32_t*>(data);
                    return mc::array(p, std::move(s), mc::int32);
                }
                case RLX_MLX_DTYPE_F64: {
                    const double* p = static_cast<const double*>(data);
                    return mc::array(p, std::move(s), mc::float64);
                }
                case RLX_MLX_DTYPE_I8: {
                    const int8_t* p = static_cast<const int8_t*>(data);
                    return mc::array(p, std::move(s), mc::int8);
                }
                case RLX_MLX_DTYPE_I16: {
                    const int16_t* p = static_cast<const int16_t*>(data);
                    return mc::array(p, std::move(s), mc::int16);
                }
                case RLX_MLX_DTYPE_I64: {
                    const int64_t* p = static_cast<const int64_t*>(data);
                    return mc::array(p, std::move(s), mc::int64);
                }
                case RLX_MLX_DTYPE_U8: {
                    const uint8_t* p = static_cast<const uint8_t*>(data);
                    return mc::array(p, std::move(s), mc::uint8);
                }
                case RLX_MLX_DTYPE_U32: {
                    const uint32_t* p = static_cast<const uint32_t*>(data);
                    return mc::array(p, std::move(s), mc::uint32);
                }
                case RLX_MLX_DTYPE_BOOL: {
                    // Treat each byte as a bool — non-zero is true.
                    const bool* p = static_cast<const bool*>(data);
                    return mc::array(p, std::move(s), mc::bool_);
                }
            }
            throw std::runtime_error("from_bytes: unhandled dtype");
        }();
        *out = wrap(std::move(result));
    });
}

int rlx_mlx_array_to_bytes(
    rlx_mlx_array_t* h,
    void* dst, size_t dst_cap, size_t* out_nbytes)
{
    return guarded([&] {
        mc::array& a = unwrap(h);
        mc::array out_arr = a;
        if (!out_arr.flags().row_contiguous) {
            out_arr = mc::contiguous(out_arr);
        }
        out_arr.eval();
        size_t n = out_arr.nbytes();
        if (n > dst_cap) {
            throw std::runtime_error("to_bytes: dst buffer too small");
        }
        // out_arr.data<void>() isn't available; use the dtype's
        // typed accessor and treat the bytes uniformly.
        const void* src;
        switch (out_arr.dtype().val()) {
            case mc::Dtype::Val::float32:  src = out_arr.data<float>(); break;
            case mc::Dtype::Val::float16:  src = out_arr.data<uint16_t>(); break;
            case mc::Dtype::Val::bfloat16: src = out_arr.data<uint16_t>(); break;
            case mc::Dtype::Val::float64:  src = out_arr.data<double>(); break;
            case mc::Dtype::Val::int8:     src = out_arr.data<int8_t>(); break;
            case mc::Dtype::Val::int16:    src = out_arr.data<int16_t>(); break;
            case mc::Dtype::Val::int32:    src = out_arr.data<int32_t>(); break;
            case mc::Dtype::Val::int64:    src = out_arr.data<int64_t>(); break;
            case mc::Dtype::Val::uint8:    src = out_arr.data<uint8_t>(); break;
            case mc::Dtype::Val::uint32:   src = out_arr.data<uint32_t>(); break;
            case mc::Dtype::Val::bool_:    src = out_arr.data<bool>(); break;
            default:
                throw std::runtime_error("to_bytes: unsupported dtype for raw readback");
        }
        std::memcpy(dst, src, n);
        *out_nbytes = n;
    });
}

int rlx_mlx_array_shape(
    rlx_mlx_array_t* h,
    int* out_shape, size_t cap, size_t* out_ndim)
{
    return guarded([&] {
        const auto& s = unwrap(h).shape();
        *out_ndim = s.size();
        if (s.size() > cap) {
            throw std::runtime_error("shape buffer too small");
        }
        for (size_t i = 0; i < s.size(); ++i) out_shape[i] = s[i];
    });
}

int rlx_mlx_array_to_f32(
    rlx_mlx_array_t* h,
    float* dst, size_t nelems)
{
    return guarded([&] {
        mc::array& a = unwrap(h);
        mc::array f32 = (a.dtype() == mc::float32) ? a : mc::astype(a, mc::float32);
        // Force a row-contiguous materialization. Ops like transpose
        // can leave the result as a strided view, so data<float>() on
        // the original would give the pre-transpose buffer order.
        // mc::copy is misleadingly named (it just shares the buffer);
        // mc::contiguous is the primitive that actually rewrites the
        // bytes into row-major order.
        if (!f32.flags().row_contiguous) {
            f32 = mc::contiguous(f32);
        }
        f32.eval();
        if (f32.size() > nelems) {
            throw std::runtime_error("output buffer too small");
        }
        std::memcpy(dst, f32.data<float>(), f32.size() * sizeof(float));
    });
}

int rlx_mlx_eval(rlx_mlx_array_t* const* handles, size_t n) {
    return guarded([&] {
        std::vector<mc::array> outs;
        outs.reserve(n);
        for (size_t i = 0; i < n; ++i) outs.push_back(unwrap(handles[i]));
        mc::eval(std::move(outs));
    });
}

int rlx_mlx_async_eval(rlx_mlx_array_t* const* handles, size_t n) {
    return guarded([&] {
        std::vector<mc::array> outs;
        outs.reserve(n);
        for (size_t i = 0; i < n; ++i) outs.push_back(unwrap(handles[i]));
        mc::async_eval(std::move(outs));
    });
}

int rlx_mlx_synchronize(void) {
    return guarded([&] {
        mc::synchronize();
    });
}

// ── Binary ops ────────────────────────────────────────────────────

#define BINARY_OP(name, mlx_fn)                                                 \
    int rlx_mlx_op_##name(                                                      \
        rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out)          \
    {                                                                            \
        return guarded([&] {                                                    \
            *out = wrap(mc::mlx_fn(unwrap(a), unwrap(b)));                      \
        });                                                                      \
    }

BINARY_OP(matmul, matmul)
BINARY_OP(add,    add)
BINARY_OP(mul,    multiply)
BINARY_OP(sub,    subtract)
BINARY_OP(div,    divide)

#undef BINARY_OP

// ── Linalg: dense solve ───────────────────────────────────────────
// Wraps mc::linalg::solve, which accepts:
//   • rank-2 A [n, n] · rank-1 b [n]      → rank-1 x [n]      (DenseSolve)
//   • rank-2 A [n, n] · rank-2 b [n, k]   → rank-2 x [n, k]   (multi-RHS)
//   • rank-3 A [B, n, n] · rank-2 b [B, n] → rank-2 x [B, n]  (BatchedDenseSolve)
// Same C entry point covers all three because MLX's solve infers rank
// from the inputs. Dtype must be float32 or float64 (validated upstream
// in mc::linalg::validate_solve — we let exceptions propagate to the
// guarded() handler so Rust gets a non-zero rc with the message).
//
// Stream selection: MLX's GPU backend doesn't yet implement linalg::solve
// (as of MLX vendor pinned in this tree — error: "[linalg::solve] This
// op is not yet supported on the GPU"). We pass an explicit CPU stream
// so the call always succeeds. The op still lives in MLX's lazy graph
// — surrounding ops fuse, no host roundtrip — but the LU factorization
// itself runs on MLX's CPU LAPACK path.
//
// When upstream MLX adds a Metal solve, dropping the explicit stream
// (or branching on dtype/shape) is a one-line change here.
int rlx_mlx_op_solve(
    rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out)
{
    return guarded([&] {
        auto cpu_stream = mc::default_stream(mc::Device::cpu);
        *out = wrap(mc::linalg::solve(unwrap(a), unwrap(b), cpu_stream));
    });
}

int rlx_mlx_op_metal_kernel_dispatch(
    const char*             name,
    const char*             source,
    const char*             header,
    const char* const*      input_names,
    size_t                  n_inputs,
    const char*             output_name,
    rlx_mlx_array_t* const* inputs,
    const int*              output_shape,
    size_t                  output_rank,
    rlx_mlx_dtype_t         output_dtype,
    int                     grid_x, int grid_y, int grid_z,
    int                     tg_x,   int tg_y,   int tg_z,
    rlx_mlx_array_t**       out)
{
    return guarded([&] {
        // Marshal C arrays to std::vectors. MLX's metal_kernel
        // factory and dispatch lambda both consume std::vector by
        // value, so these are cheap (small string lists, single
        // output slot in v1).
        std::vector<std::string> in_names;
        in_names.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_names.emplace_back(input_names[i] ? input_names[i] : "");
        }
        std::vector<std::string> out_names;
        out_names.emplace_back(output_name ? output_name : "out");

        std::vector<mc::array> in_arrs;
        in_arrs.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_arrs.push_back(unwrap(inputs[i]));
        }

        // mc::Shape is SmallVector<int32_t>. Construct from the
        // caller-supplied row of int dims.
        mc::Shape out_shape_v;
        out_shape_v.reserve(output_rank);
        for (size_t i = 0; i < output_rank; ++i) {
            out_shape_v.push_back(output_shape[i]);
        }
        std::vector<mc::Shape> out_shapes = { std::move(out_shape_v) };
        std::vector<mc::Dtype> out_dtypes = { to_mlx_dtype(output_dtype) };

        // Build the kernel factory. MLX caches the compiled MTL
        // function internally by source hash on first dispatch, so
        // calling this on every invocation with stable source is
        // cheap (a few µs of std::function build cost, not the full
        // kernel compile).
        auto kernel_fn = mc::fast::metal_kernel(
            name ? std::string(name) : std::string("anon"),
            in_names,
            out_names,
            source ? std::string(source) : std::string(""),
            header ? std::string(header) : std::string(""),
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false);

        auto grid = std::make_tuple(grid_x, grid_y, grid_z);
        auto tg   = std::make_tuple(tg_x,   tg_y,   tg_z);

        // No template args / init_value / verbose for v1.
        std::vector<std::pair<std::string, mc::fast::TemplateArg>> templates;
        std::optional<float> init_value;
        bool verbose = false;

        auto outs = kernel_fn(
            in_arrs,
            out_shapes,
            out_dtypes,
            grid,
            tg,
            templates,
            init_value,
            verbose,
            /*stream=*/{});

        if (outs.empty()) {
            throw std::runtime_error("metal_kernel returned no outputs");
        }
        *out = wrap(std::move(outs.front()));
    });
}

// ── Unary / activations ───────────────────────────────────────────

int rlx_mlx_op_softmax(rlx_mlx_array_t* a, int axis, rlx_mlx_array_t** out) {
    return guarded([&] {
        *out = wrap(mc::softmax(unwrap(a), axis));
    });
}

int rlx_mlx_op_gelu(rlx_mlx_array_t* a, rlx_mlx_array_t** out) {
    return guarded([&] {
        // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        const mc::array& x = unwrap(a);
        mc::array half = mc::array(0.5f, x.dtype());
        mc::array one  = mc::array(1.0f, x.dtype());
        mc::array inv_sqrt2 = mc::array(0.70710678118654752f, x.dtype());
        mc::array y = mc::multiply(
            mc::multiply(half, x),
            mc::add(one, mc::erf(mc::multiply(x, inv_sqrt2))));
        *out = wrap(std::move(y));
    });
}

int rlx_mlx_op_silu(rlx_mlx_array_t* a, rlx_mlx_array_t** out) {
    return guarded([&] {
        // silu(x) = x * sigmoid(x)
        const mc::array& x = unwrap(a);
        *out = wrap(mc::multiply(x, mc::sigmoid(x)));
    });
}

int rlx_mlx_op_cast(rlx_mlx_array_t* a, rlx_mlx_dtype_t dtype, rlx_mlx_array_t** out) {
    return guarded([&] {
        *out = wrap(mc::astype(unwrap(a), to_mlx_dtype(dtype)));
    });
}

int rlx_mlx_op_layernorm(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* gamma,
    rlx_mlx_array_t* beta_or_null,
    float eps,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::optional<mc::array> w = std::optional<mc::array>(unwrap(gamma));
        std::optional<mc::array> b = beta_or_null
            ? std::optional<mc::array>(unwrap(beta_or_null))
            : std::nullopt;
        *out = wrap(mc::fast::layer_norm(unwrap(x), w, b, eps));
    });
}

// ── Binary (rest of the set) ──────────────────────────────────────

#define BINARY_OP(name, mlx_fn)                                                 \
    int rlx_mlx_op_##name(                                                      \
        rlx_mlx_array_t* a, rlx_mlx_array_t* b, rlx_mlx_array_t** out)          \
    {                                                                            \
        return guarded([&] {                                                    \
            *out = wrap(mc::mlx_fn(unwrap(a), unwrap(b)));                      \
        });                                                                      \
    }

BINARY_OP(max, maximum)
BINARY_OP(min, minimum)
BINARY_OP(pow, power)

BINARY_OP(eq, equal)
BINARY_OP(ne, not_equal)
BINARY_OP(lt, less)
BINARY_OP(le, less_equal)
BINARY_OP(gt, greater)
BINARY_OP(ge, greater_equal)

#undef BINARY_OP

int rlx_mlx_op_where(
    rlx_mlx_array_t* cond,
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* y,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::where(unwrap(cond), unwrap(x), unwrap(y)));
    });
}

// ── Unary dispatch ────────────────────────────────────────────────

int rlx_mlx_op_unary(
    rlx_mlx_array_t* a, rlx_mlx_unary_t kind, rlx_mlx_array_t** out)
{
    return guarded([&] {
        const mc::array& x = unwrap(a);
        mc::array y = [&]() {
            switch (kind) {
                case RLX_MLX_UN_RELU: {
                    // relu(x) = maximum(x, 0)
                    return mc::maximum(x, mc::array(0.0f, x.dtype()));
                }
                case RLX_MLX_UN_SIGMOID: return mc::sigmoid(x);
                case RLX_MLX_UN_TANH:    return mc::tanh(x);
                case RLX_MLX_UN_EXP:     return mc::exp(x);
                case RLX_MLX_UN_LOG:     return mc::log(x);
                case RLX_MLX_UN_SQRT:    return mc::sqrt(x);
                case RLX_MLX_UN_RSQRT:   return mc::rsqrt(x);
                case RLX_MLX_UN_NEG:     return mc::negative(x);
                case RLX_MLX_UN_ABS:     return mc::abs(x);
                case RLX_MLX_UN_ERF:     return mc::erf(x);
                case RLX_MLX_UN_ROUND:   return mc::round(x);
                case RLX_MLX_UN_SIN:     return mc::sin(x);
                case RLX_MLX_UN_COS:     return mc::cos(x);
                case RLX_MLX_UN_TAN:     return mc::tan(x);
                case RLX_MLX_UN_ATAN:    return mc::arctan(x);
            }
            throw std::runtime_error("invalid unary kind");
        }();
        *out = wrap(std::move(y));
    });
}

// ── Shape ops ────────────────────────────────────────────────────

int rlx_mlx_op_reshape(
    rlx_mlx_array_t* a,
    const int* new_shape, size_t ndim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s;
        s.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) s.push_back(new_shape[i]);
        *out = wrap(mc::reshape(unwrap(a), std::move(s)));
    });
}

int rlx_mlx_op_transpose(
    rlx_mlx_array_t* a,
    const int* perm, size_t ndim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<int> axes;
        axes.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) axes.push_back(perm[i]);
        *out = wrap(mc::transpose(unwrap(a), std::move(axes)));
    });
}

int rlx_mlx_op_slice(
    rlx_mlx_array_t* a,
    const int* start, const int* stop, size_t ndim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s_start, s_stop;
        s_start.reserve(ndim);
        s_stop.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            s_start.push_back(start[i]);
            s_stop.push_back(stop[i]);
        }
        *out = wrap(mc::slice(unwrap(a), std::move(s_start), std::move(s_stop)));
    });
}

int rlx_mlx_op_concat(
    rlx_mlx_array_t* const* arrays, size_t n,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<mc::array> xs;
        xs.reserve(n);
        for (size_t i = 0; i < n; ++i) xs.push_back(unwrap(arrays[i]));
        *out = wrap(mc::concatenate(std::move(xs), axis));
    });
}

int rlx_mlx_op_broadcast_to(
    rlx_mlx_array_t* a,
    const int* shape, size_t ndim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s;
        s.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) s.push_back(shape[i]);
        *out = wrap(mc::broadcast_to(unwrap(a), s));
    });
}

int rlx_mlx_op_take(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        // Cast indices to int32 (rlx encodes them as f32 at the I/O
        // boundary, but Op::Gather semantics treat them as integer
        // positions; the lowering converts before calling us).
        mc::array idx = unwrap(indices);
        if (idx.dtype() != mc::int32 && idx.dtype() != mc::uint32) {
            idx = mc::astype(idx, mc::int32);
        }
        *out = wrap(mc::take(unwrap(a), idx, axis));
    });
}

// ── Reductions ───────────────────────────────────────────────────

int rlx_mlx_op_reduce(
    rlx_mlx_array_t* a,
    rlx_mlx_reduce_t kind,
    const int* axes, size_t n_axes,
    int keep_dim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<int> ax;
        ax.reserve(n_axes);
        for (size_t i = 0; i < n_axes; ++i) ax.push_back(axes[i]);
        bool kd = keep_dim != 0;
        const mc::array& x = unwrap(a);
        mc::array y = [&]() {
            switch (kind) {
                case RLX_MLX_RED_SUM:  return mc::sum(x, ax, kd);
                case RLX_MLX_RED_MEAN: return mc::mean(x, ax, kd);
                case RLX_MLX_RED_MAX:  return mc::max(x, ax, kd);
                case RLX_MLX_RED_MIN:  return mc::min(x, ax, kd);
                case RLX_MLX_RED_PROD: return mc::prod(x, ax, kd);
            }
            throw std::runtime_error("invalid reduce kind");
        }();
        *out = wrap(std::move(y));
    });
}

int rlx_mlx_op_cumsum(
    rlx_mlx_array_t* a,
    int axis,
    int exclusive,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        // mlx's cumsum has `inclusive` (the inverse of rlx's `exclusive`).
        bool inclusive = (exclusive == 0);
        *out = wrap(mc::cumsum(unwrap(a), axis, /*reverse=*/false, inclusive));
    });
}

int rlx_mlx_op_fft(
    rlx_mlx_array_t* a,
    int inverse,
    int norm_tag,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        auto x = unwrap(a);
        const auto sh = x.shape();
        if (sh.empty()) {
            throw std::invalid_argument("[rlx_mlx_op_fft] input must have rank >= 1");
        }
        const int nd = static_cast<int>(sh.size());
        const int axis = nd - 1;
        const bool inv = inverse != 0;

        auto rlx_output_scale = [&](int64_t n) -> double {
            const double nd = static_cast<double>(n);
            switch (norm_tag) {
                case 0:
                    return 1.0;
                case 1:
                    return inv ? 1.0 / nd : 1.0;
                case 2:
                    return 1.0 / std::sqrt(nd);
                default:
                    throw std::invalid_argument(
                        "[rlx_mlx_op_fft] invalid norm_tag (expected 0, 1, or 2)");
            }
        };

        auto mlx_effective_scale = [&](int64_t n) -> double {
            // MLX FFTNorm::Backward applies 1/N on ifft only.
            return inv ? 1.0 / static_cast<double>(n) : 1.0;
        };

        auto apply_norm = [&](mc::array y, int64_t n) {
            const double corr =
                rlx_output_scale(n) / mlx_effective_scale(n);
            if (std::abs(corr - 1.0) > 1e-12) {
                y = mc::multiply(y, mc::array(static_cast<float>(corr)));
            }
            return y;
        };

        const bool real_block = x.dtype() != mc::complex64;
        if (real_block) {
            const int64_t last = sh.back();
            if (last % 2 != 0) {
                throw std::invalid_argument(
                    "[rlx_mlx_op_fft] last axis must be even (2N real-block layout)");
            }
            const int64_t n = last / 2;
            mc::Shape starts(sh.size(), 0);
            mc::Shape stops = sh;
            mc::Shape re_st = starts;
            mc::Shape re_sp = stops;
            re_sp.back() = n;
            mc::Shape im_st = starts;
            mc::Shape im_sp = stops;
            im_st.back() = n;
            auto re = mc::slice(x, re_st, re_sp);
            auto im = mc::slice(x, im_st, im_sp);
            mc::array cx = mc::add(
                mc::astype(re, mc::complex64),
                mc::multiply(
                    mc::astype(im, mc::complex64),
                    mc::array(mc::complex64_t{0.0f, 1.0f})));
            mc::array y = inv
                ? mc::fft::ifft(cx, axis, mc::fft::FFTNorm::Backward)
                : mc::fft::fft(cx, axis, mc::fft::FFTNorm::Backward);
            y = apply_norm(y, n);
            auto y_re = mc::real(y);
            auto y_im = mc::imag(y);
            *out = wrap(mc::concatenate({y_re, y_im}, axis));
        } else {
            mc::array y = inv
                ? mc::fft::ifft(x, axis, mc::fft::FFTNorm::Backward)
                : mc::fft::fft(x, axis, mc::fft::FFTNorm::Backward);
            const int64_t n = sh[axis];
            y = apply_norm(y, n);
            *out = wrap(y);
        }
    });
}

int rlx_mlx_op_rmsnorm(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* gamma,
    float eps,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::optional<mc::array> w(unwrap(gamma));
        *out = wrap(mc::fast::rms_norm(unwrap(x), w, eps));
    });
}

int rlx_mlx_op_attention(
    rlx_mlx_array_t* q,
    rlx_mlx_array_t* k,
    rlx_mlx_array_t* v,
    float scale,
    rlx_mlx_mask_t mask_kind,
    rlx_mlx_array_t* mask_or_null,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::string mode;
        std::optional<mc::array> mask;
        switch (mask_kind) {
            case RLX_MLX_MASK_NONE:    mode = ""; break;
            case RLX_MLX_MASK_CAUSAL:  mode = "causal"; break;
            case RLX_MLX_MASK_SLIDING:
            case RLX_MLX_MASK_CUSTOM:
                if (!mask_or_null) {
                    throw std::runtime_error(
                        "attention: mask required for SLIDING/CUSTOM mask kinds");
                }
                mode = "array";
                mask = std::optional<mc::array>(unwrap(mask_or_null));
                break;
        }
        *out = wrap(mc::fast::scaled_dot_product_attention(
            unwrap(q), unwrap(k), unwrap(v), scale, mode, mask));
    });
}

// ── PR3 heavy ops ────────────────────────────────────────────────

int rlx_mlx_op_conv2d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int dil_h,    int dil_w,
    int groups,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::conv2d(
            unwrap(input), unwrap(weight),
            std::pair<int, int>{stride_h, stride_w},
            std::pair<int, int>{pad_h, pad_w},
            std::pair<int, int>{dil_h, dil_w},
            groups));
    });
}

int rlx_mlx_op_conv1d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride, int padding, int dilation, int groups,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::conv1d(
            unwrap(input), unwrap(weight),
            stride, padding, dilation, groups));
    });
}

int rlx_mlx_op_conv3d(
    rlx_mlx_array_t* input,
    rlx_mlx_array_t* weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d,    int pad_h,    int pad_w,
    int dil_d,    int dil_h,    int dil_w,
    int groups,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::conv3d(
            unwrap(input), unwrap(weight),
            std::tuple<int,int,int>{stride_d, stride_h, stride_w},
            std::tuple<int,int,int>{pad_d, pad_h, pad_w},
            std::tuple<int,int,int>{dil_d, dil_h, dil_w},
            groups));
    });
}

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
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<int> stride_v   (stride,           stride           + stride_n);
        std::vector<int> pad_lo_v   (padding_lo,       padding_lo       + padding_lo_n);
        std::vector<int> pad_hi_v   (padding_hi,       padding_hi       + padding_hi_n);
        std::vector<int> kd_v       (kernel_dilation,  kernel_dilation  + kernel_dilation_n);
        std::vector<int> id_v       (input_dilation,   input_dilation   + input_dilation_n);
        *out = wrap(mc::conv_general(
            unwrap(input), unwrap(weight),
            stride_v, pad_lo_v, pad_hi_v, kd_v, id_v,
            groups, flip != 0));
    });
}

int rlx_mlx_op_argpartition(
    rlx_mlx_array_t* a,
    int kth, int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::argpartition(unwrap(a), kth, axis));
    });
}

int rlx_mlx_op_scatter_add(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    rlx_mlx_array_t* updates,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::array idx = unwrap(indices);
        if (idx.dtype() != mc::int32 && idx.dtype() != mc::uint32) {
            idx = mc::astype(idx, mc::int32);
        }
        *out = wrap(mc::scatter_add(unwrap(a), idx, unwrap(updates), axis));
    });
}

int rlx_mlx_op_contiguous(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::contiguous(unwrap(a)));
    });
}

// ── Custom Metal kernel: max-pool 2D backward ────────────────────
//
// Per output position (n, c, ho, wo): scan the kh·kw window of `x`,
// pick the argmax (first hit on ties — strict `>`), atomically add
// `dy[n, c, ho, wo]` into `dx[n, c, hi_argmax, wi_argmax]`. Output
// is initialized to 0 via `init_value=0.0f`.
//
// Compilation result is cached in a static map keyed on a string of
// the unchanging source so repeat calls skip re-creation cost (MLX's
// own pipeline cache also hits, but std::function construction has
// nontrivial overhead we avoid this way).
namespace {
namespace mfast = mc::fast;

const char* kMaxPool2dBackwardKernelSrc = R"(
    uint wo = thread_position_in_grid.x;
    uint ho = thread_position_in_grid.y;
    uint nc = thread_position_in_grid.z;

    if (wo >= W_OUT_T || ho >= H_OUT_T || nc >= N_T * C_T) return;

    uint n = nc / C_T;
    uint cc = nc % C_T;
    uint x_base = ((n * C_T) + cc) * H_T * W_T;

    float best_v = -INFINITY;
    int best_hi = -1;
    int best_wi = -1;
    for (int ki = 0; ki < KH_T; ki++) {
        int hi = int(ho) * SH_T + ki - PH_T;
        if (hi < 0 || hi >= H_T) continue;
        for (int kj = 0; kj < KW_T; kj++) {
            int wi = int(wo) * SW_T + kj - PW_T;
            if (wi < 0 || wi >= W_T) continue;
            float v = x[x_base + uint(hi) * W_T + uint(wi)];
            if (v > best_v) {
                best_v = v;
                best_hi = hi;
                best_wi = wi;
            }
        }
    }

    if (best_hi < 0) return;

    uint dy_idx = ((n * C_T) + cc) * H_OUT_T * W_OUT_T + ho * W_OUT_T + wo;
    uint dx_idx = x_base + uint(best_hi) * W_T + uint(best_wi);

    atomic_fetch_add_explicit(&dx[dx_idx], dy[dy_idx], memory_order_relaxed);
)";

mfast::CustomKernelFunction& maxpool2d_backward_kernel() {
    static mfast::CustomKernelFunction k = mfast::metal_kernel(
        /*name=*/             "rlx_maxpool2d_backward",
        /*input_names=*/      {"x", "dy"},
        /*output_names=*/     {"dx"},
        /*source=*/           kMaxPool2dBackwardKernelSrc,
        /*header=*/           "",
        /*ensure_row_contiguous=*/ true,
        /*atomic_outputs=*/   true);
    return k;
}

} // namespace

int rlx_mlx_op_maxpool2d_backward_metal(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* dy,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw,
    int sh, int sw,
    int ph, int pw,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<mc::array> inputs = {unwrap(x), unwrap(dy)};
        std::vector<mc::Shape> output_shapes = {{n, c, h, w}};
        std::vector<mc::Dtype> output_dtypes = {mc::float32};

        std::tuple<int, int, int> grid{w_out, h_out, n * c};
        // Threadgroup: small enough to be valid for all reasonable
        // shapes, large enough to amortize launch cost. The Metal
        // backend will clamp if grid is smaller.
        int tg_x = std::min(w_out, 16);
        int tg_y = std::min(h_out, 16);
        if (tg_x < 1) tg_x = 1;
        if (tg_y < 1) tg_y = 1;
        std::tuple<int, int, int> threadgroup{tg_x, tg_y, 1};

        std::vector<std::pair<std::string, mfast::TemplateArg>> tpl = {
            {"N_T",     mfast::TemplateArg(int(n))},
            {"C_T",     mfast::TemplateArg(int(c))},
            {"H_T",     mfast::TemplateArg(int(h))},
            {"W_T",     mfast::TemplateArg(int(w))},
            {"H_OUT_T", mfast::TemplateArg(int(h_out))},
            {"W_OUT_T", mfast::TemplateArg(int(w_out))},
            {"KH_T",    mfast::TemplateArg(int(kh))},
            {"KW_T",    mfast::TemplateArg(int(kw))},
            {"SH_T",    mfast::TemplateArg(int(sh))},
            {"SW_T",    mfast::TemplateArg(int(sw))},
            {"PH_T",    mfast::TemplateArg(int(ph))},
            {"PW_T",    mfast::TemplateArg(int(pw))},
        };

        auto outs = maxpool2d_backward_kernel()(
            inputs,
            output_shapes,
            output_dtypes,
            grid,
            threadgroup,
            tpl,
            /*init_value=*/ std::optional<float>(0.0f),
            /*verbose=*/    false,
            mc::StreamOrDevice{}); // monostate → default stream
        if (outs.size() != 1) {
            throw std::runtime_error(
                "maxpool2d_backward_metal: kernel returned wrong number of outputs");
        }
        *out = wrap(std::move(outs[0]));
    });
}

int rlx_mlx_op_take_along_axis(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::array idx = unwrap(indices);
        if (idx.dtype() != mc::int32 && idx.dtype() != mc::uint32) {
            idx = mc::astype(idx, mc::int32);
        }
        *out = wrap(mc::take_along_axis(unwrap(a), idx, axis));
    });
}

int rlx_mlx_op_scatter_add_axis(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* indices,
    rlx_mlx_array_t* updates,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::array idx = unwrap(indices);
        if (idx.dtype() != mc::int32 && idx.dtype() != mc::uint32) {
            idx = mc::astype(idx, mc::int32);
        }
        *out = wrap(mc::scatter_add_axis(unwrap(a), idx, unwrap(updates), axis));
    });
}

int rlx_mlx_op_gather_mm(
    rlx_mlx_array_t* a,
    rlx_mlx_array_t* b,
    rlx_mlx_array_t* idx,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::array i = unwrap(idx);
        if (i.dtype() != mc::int32 && i.dtype() != mc::uint32) {
            i = mc::astype(i, mc::int32);
        }
        // gather_mm in MLX: gather_mm(a, b, lhs_indices, rhs_indices, sorted_indices)
        // For our use case (one expert per token), we want b indexed
        // by `i` along its leading dim — pass i as rhs_indices, no
        // lhs_indices.
        *out = wrap(mc::gather_mm(unwrap(a), unwrap(b),
                                  /*lhs_indices=*/std::nullopt,
                                  /*rhs_indices=*/std::optional<mc::array>(i)));
    });
}

int rlx_mlx_op_quantized_matmul(
    rlx_mlx_array_t* x,
    rlx_mlx_array_t* w,
    rlx_mlx_array_t* scales,
    rlx_mlx_array_t* biases_or_null,
    int transpose,
    int group_size,
    int bits,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::optional<mc::array> bias = biases_or_null
            ? std::optional<mc::array>(unwrap(biases_or_null))
            : std::nullopt;
        *out = wrap(mc::quantized_matmul(
            unwrap(x), unwrap(w), unwrap(scales), bias,
            transpose != 0,
            std::optional<int>(group_size),
            std::optional<int>(bits),
            /*mode=*/"affine"));
    });
}

int rlx_mlx_op_categorical(
    rlx_mlx_array_t* logits,
    int axis,
    uint64_t seed,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::optional<mc::array> key;
        if (seed != 0) {
            key = std::optional<mc::array>(mc::random::key(seed));
        }
        *out = wrap(mc::random::categorical(unwrap(logits), axis, key));
    });
}

int rlx_mlx_op_argmax(
    rlx_mlx_array_t* a,
    int axis, int keep_dim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::argmax(unwrap(a), axis, keep_dim != 0));
    });
}

int rlx_mlx_op_slice_strided(
    rlx_mlx_array_t* a,
    const int* start, const int* stop, const int* strides, size_t ndim,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        mc::Shape s_start, s_stop, s_strides;
        s_start.reserve(ndim); s_stop.reserve(ndim); s_strides.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            s_start.push_back(start[i]);
            s_stop.push_back(stop[i]);
            s_strides.push_back(strides[i]);
        }
        *out = wrap(mc::slice(unwrap(a),
            std::move(s_start), std::move(s_stop), std::move(s_strides)));
    });
}

int rlx_mlx_op_pad(
    rlx_mlx_array_t* a,
    const int* low, const int* high, size_t ndim,
    float pad_value,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        std::vector<std::pair<int, int>> widths;
        widths.reserve(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            widths.emplace_back(low[i], high[i]);
        }
        mc::array pv(pad_value);
        *out = wrap(mc::pad(unwrap(a), widths, pv));
    });
}

int rlx_mlx_op_topk_values(
    rlx_mlx_array_t* a,
    int k, int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::topk(unwrap(a), k, axis));
    });
}

int rlx_mlx_op_sort(
    rlx_mlx_array_t* a,
    int axis,
    rlx_mlx_array_t** out)
{
    return guarded([&] {
        *out = wrap(mc::sort(unwrap(a), axis));
    });
}

// ── PR7: persistent compiled graphs ──────────────────────────────

struct rlx_mlx_compiled_s {
    std::function<std::vector<mc::array>(const std::vector<mc::array>&)> fn;
};

int rlx_mlx_compile(
    rlx_mlx_lower_fn fn, void* ud,
    int shapeless,
    rlx_mlx_compiled_t** out)
{
    return guarded([&] {
        auto wrapped = [fn, ud](const std::vector<mc::array>& inputs)
            -> std::vector<mc::array>
        {
            // Wrap each input in a fresh Handle so the Rust callback
            // can treat the pointer as "owned" — its Array::Drop will
            // free the Handle, but the underlying mc::array (a
            // shared_ptr-backed value type) is reference-counted so
            // the original C++-side array stays alive.
            std::vector<rlx_mlx_array_t*> in_handles;
            in_handles.reserve(inputs.size());
            for (const auto& a : inputs) {
                in_handles.push_back(wrap(a));
            }

            // Reasonable upper bound — graph outputs typically ≤ a
            // few; bump if a workload trips this.
            constexpr size_t cap = 64;
            std::vector<rlx_mlx_array_t*> out_handles(cap, nullptr);
            size_t n_out = 0;
            int rc = fn(ud, in_handles.data(), in_handles.size(),
                        out_handles.data(), cap, &n_out);
            if (rc != RLX_MLX_OK) {
                throw std::runtime_error(
                    g_last_error.empty()
                        ? std::string{"rust lowering callback failed"}
                        : g_last_error);
            }

            // Take ownership of output handles back into mc::array
            // values; free the Handles. The Rust callback released
            // ownership of these by writing them into out_handles
            // and using std::mem::forget on the Array wrappers.
            std::vector<mc::array> outputs;
            outputs.reserve(n_out);
            for (size_t i = 0; i < n_out; ++i) {
                Handle* h = reinterpret_cast<Handle*>(out_handles[i]);
                outputs.push_back(h->a);
                delete h;
            }
            return outputs;
        };

        auto compiled = std::make_unique<rlx_mlx_compiled_s>();
        compiled->fn = mc::compile(std::move(wrapped), shapeless != 0);
        *out = compiled.release();
    });
}

int rlx_mlx_compiled_call(
    rlx_mlx_compiled_t* compiled,
    rlx_mlx_array_t* const* inputs, size_t n_inputs,
    rlx_mlx_array_t** out_outputs, size_t cap, size_t* out_n_outputs)
{
    return guarded([&] {
        std::vector<mc::array> in_arrays;
        in_arrays.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_arrays.push_back(unwrap(inputs[i]));
        }
        auto outs = compiled->fn(in_arrays);
        if (outs.size() > cap) {
            throw std::runtime_error("compiled_call: output buffer too small");
        }
        for (size_t i = 0; i < outs.size(); ++i) {
            out_outputs[i] = wrap(outs[i]);
        }
        *out_n_outputs = outs.size();
    });
}

void rlx_mlx_compiled_free(rlx_mlx_compiled_t* compiled) {
    if (compiled) delete compiled;
}

} // extern "C"
