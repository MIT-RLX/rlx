// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// C++ view of rlx_micro.h for firmware: header-only, freestanding — no
// exceptions, no heap, no STL. Buffers are `rlx::micro::Span<T>` (pointer +
// length; build one from a C array), so sizes travel with the data and the
// library checks them. Calls return `rlx::micro::Status`.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../rlx_micro.h"

namespace rlx::micro {

enum class Status : int32_t {
    Ok = RLX_MICRO_OK,
    Null = RLX_MICRO_ERR_NULL,
    Size = RLX_MICRO_ERR_SIZE,
    Param = RLX_MICRO_ERR_PARAM,
};

constexpr bool ok(Status s) { return s == Status::Ok; }

template <typename T>
struct Span {
    T* data = nullptr;
    size_t len = 0;
    constexpr Span() = default;
    constexpr Span(T* d, size_t n) : data(d), len(n) {}
    template <size_t N>
    constexpr Span(T (&a)[N]) : data(a), len(N) {}
};

using Conv2d = RlxMicroConv2d;
using Pool2d = RlxMicroPool2d;

inline const char* version() { return rlx_micro_version(); }

inline size_t out_len(const Conv2d& p) { return rlx_micro_conv2d_out_len(&p); }
inline size_t out_len(const Pool2d& p) { return rlx_micro_pool2d_out_len(&p); }

inline Status conv2d(const Conv2d& p, Span<const int8_t> x, Span<const int8_t> w, const int32_t* bias,
                     Span<int8_t> out) {
    return Status(rlx_micro_conv2d_i8(&p, x.data, x.len, w.data, w.len, bias, out.data, out.len));
}

inline Status dense(Span<const int8_t> x, Span<const int8_t> w, const int32_t* bias, int32_t x_zp, int32_t w_zp,
                    int32_t out_zp, const float* mult, uint8_t weight_bits, Span<int8_t> out) {
    return Status(rlx_micro_dense_i8(x.data, x.len, w.data, w.len, bias, x_zp, w_zp, out_zp, mult, weight_bits,
                                     out.data, out.len));
}

inline Status maxpool(const Pool2d& p, Span<const int8_t> x, Span<int8_t> out) {
    return Status(rlx_micro_maxpool2d_i8(&p, x.data, x.len, out.data, out.len));
}

inline Status relu(Span<int8_t> x, int32_t zero_point = 0) {
    return Status(rlx_micro_relu_i8(x.data, x.len, zero_point));
}

inline int32_t argmax(Span<const int8_t> x) { return rlx_micro_argmax_i8(x.data, x.len); }

inline Status quantize(Span<const float> x, float scale, int32_t zero_point, Span<int8_t> out) {
    if (out.len != x.len) return Status::Size;
    return Status(rlx_micro_quantize(x.data, x.len, scale, zero_point, out.data));
}

inline Status dequantize(Span<const int8_t> x, float scale, int32_t zero_point, Span<float> out) {
    if (out.len != x.len) return Status::Size;
    return Status(rlx_micro_dequantize(x.data, x.len, scale, zero_point, out.data));
}

// The bundled TinyConv MNIST model, with its two scratch buffers owned
// inline (place the object in .bss or on a large enough stack).
class Mnist {
  public:
    static constexpr size_t kInputLen = RLX_MICRO_MNIST_INPUT_LEN;
    static constexpr size_t kScratchLen = RLX_MICRO_MNIST_SCRATCH_LEN;

    // The digit 0-9, or a negative status.
    int32_t classify(Span<const int8_t> image) {
        return rlx_micro_mnist_infer(image.data, image.len, a_, b_, kScratchLen);
    }

    static float input_scale() { return rlx_micro_mnist_input_scale(); }
    static Span<const int8_t> test_image() { return {rlx_micro_mnist_test_image(), kInputLen}; }
    static uint8_t test_label() { return rlx_micro_mnist_test_label(); }

  private:
    int8_t a_[kScratchLen];
    int8_t b_[kScratchLen];
};

}  // namespace rlx::micro
