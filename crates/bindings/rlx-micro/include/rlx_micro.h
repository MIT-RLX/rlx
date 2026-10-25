/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * rlx-micro: rlx's microcontroller kernels (rlx-cortexm) for C and C++
 * firmware. INT8 conv2d / dense / max-pool / ReLU / argmax, quantization, and
 * a bundled TinyConv MNIST model — no_std, no heap, no OS, ~25 KB of code.
 * The same library builds for the host (to check firmware logic) and for
 * bare-metal ARMv7E-M (thumbv7em-none-eabihf: Cortex-M4F / M7).
 *
 * Every function checks its buffer sizes and returns a status (RLX_MICRO_OK
 * or a negative RLX_MICRO_ERR_*) instead of trapping; nothing allocates.
 * Tensors are NHWC i8; weights row-major, packed at 8, 4 or 2 bits.
 * Requantization: out = sat_i8(round(acc · mult) + out_zp), where acc is the
 * i32 accumulator (bias in x_scale · w_scale) and mult = x_scale · w_scale /
 * out_scale, one per output channel.
 */
#ifndef RLX_MICRO_H
#define RLX_MICRO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLX_MICRO_OK 0
/* A NULL pointer where one is required. */
#define RLX_MICRO_ERR_NULL (-1)
/* A buffer length that does not match the shape. */
#define RLX_MICRO_ERR_SIZE (-2)
/* An unsupported parameter (weight bits, stride 0, kernel larger than input). */
#define RLX_MICRO_ERR_PARAM (-3)

/* The bundled TinyConv MNIST model: [28, 28, 1] i8 image → digit. */
#define RLX_MICRO_MNIST_INPUT_LEN 784
/* Bytes each of the two scratch buffers needs (the largest activation). */
#define RLX_MICRO_MNIST_SCRATCH_LEN 5408

/* conv2d over NHWC `x` [h_in, w_in, c_in] with weights [c_out, kh, kw, c_in]. */
typedef struct {
    size_t h_in, w_in, c_in, c_out;
    size_t kh, kw, pad_h, pad_w, stride_h, stride_w;
    int32_t x_zp, w_zp, out_zp;
    const float* mult; /* [c_out] */
    uint8_t weight_bits; /* 8, 4 or 2 */
} RlxMicroConv2d;

/* Max-pool over NHWC [h_in, w_in, c], no padding. */
typedef struct {
    size_t h_in, w_in, c, kh, kw, stride_h, stride_w;
} RlxMicroPool2d;

/* The library version, e.g. "0.2.18". */
const char* rlx_micro_version(void);

/* Output dims of a conv2d / pool (0 when the parameters are invalid). */
size_t rlx_micro_conv2d_out_len(const RlxMicroConv2d* p);
size_t rlx_micro_pool2d_out_len(const RlxMicroPool2d* p);
/* Bytes `n` weights take at `bits` (8, 4, 2) per weight. */
size_t rlx_micro_packed_len(size_t n, uint8_t bits);

/* INT8 conv2d; `bias` [c_out] i32 may be NULL. */
int32_t rlx_micro_conv2d_i8(const RlxMicroConv2d* p, const int8_t* x, size_t x_len, const int8_t* w,
                            size_t w_len, const int32_t* bias, int8_t* out, size_t out_len);

/* INT8 dense: out[out_features] from x[in_features] and w[out_features, in_features];
 * `bias` [out_features] may be NULL; `mult` [out_features]. */
int32_t rlx_micro_dense_i8(const int8_t* x, size_t in_features, const int8_t* w, size_t w_len,
                           const int32_t* bias, int32_t x_zp, int32_t w_zp, int32_t out_zp, const float* mult,
                           uint8_t weight_bits, int8_t* out, size_t out_features);

/* INT8 max-pool. */
int32_t rlx_micro_maxpool2d_i8(const RlxMicroPool2d* p, const int8_t* x, size_t x_len, int8_t* out,
                               size_t out_len);

/* ReLU in place: values below `zero_point` clamp to it. */
int32_t rlx_micro_relu_i8(int8_t* x, size_t n, int32_t zero_point);

/* Index of the largest value (the first on ties); -1 when n is 0. */
int32_t rlx_micro_argmax_i8(const int8_t* x, size_t n);

/* f32 ↔ i8 at (scale, zero_point). */
int32_t rlx_micro_quantize(const float* x, size_t n, float scale, int32_t zero_point, int8_t* out);
int32_t rlx_micro_dequantize(const int8_t* x, size_t n, float scale, int32_t zero_point, float* out);

/* Classify a [28, 28, 1] image (symmetric i8 at the model's input scale) with
 * the bundled MNIST model; returns the digit 0-9, or a negative status. */
int32_t rlx_micro_mnist_infer(const int8_t* image, size_t image_len, int8_t* scratch_a, int8_t* scratch_b,
                              size_t scratch_len);
/* The model's input scale (image = round(pixel / scale), pixel in [-1, 1]). */
float rlx_micro_mnist_input_scale(void);
/* A test image bundled with the model, and its label. */
const int8_t* rlx_micro_mnist_test_image(void);
uint8_t rlx_micro_mnist_test_label(void);

#ifdef __cplusplus
}
#endif

#endif /* RLX_MICRO_H */
