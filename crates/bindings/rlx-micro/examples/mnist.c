/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * rlx-micro from C: classify a digit with the bundled INT8 MNIST model, then
 * run a hand-built quantized dense layer. The same source runs on a desktop
 * (port_host.c) and bare-metal on a Cortex-M (mcu/: startup + semihosting) —
 * no heap, no libc, every buffer static. */
#include <stdint.h>

#include "port.h"
#include "rlx_micro.h"

static int8_t scratch_a[RLX_MICRO_MNIST_SCRATCH_LEN];
static int8_t scratch_b[RLX_MICRO_MNIST_SCRATCH_LEN];

/* y = W·x with W = diag-ish [[1, 0, 0, 0], [0, 0, 0, 1]]: picks x[0] and x[3]. */
static int dense_layer(void) {
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float x_scale = 0.05f, w_scale = 0.1f, y_scale = 0.05f;
    const int8_t w[8] = {10, 0, 0, 0, 0, 0, 0, 10}; /* 1.0 at w_scale */
    const float mult[2] = {x_scale * w_scale / y_scale, x_scale * w_scale / y_scale};
    int8_t xq[4], yq[2];
    float y[2];
    if (rlx_micro_quantize(x, 4, x_scale, 0, xq) != RLX_MICRO_OK) return 1;
    if (rlx_micro_dense_i8(xq, 4, w, sizeof w, NULL, 0, 0, 0, mult, 8, yq, 2) != RLX_MICRO_OK) return 1;
    rlx_micro_dequantize(yq, 2, y_scale, 0, y);
    port_write("dense: ");
    port_fixed2(y[0]);
    port_write(" ");
    port_fixed2(y[1]);
    port_write("\n");
    /* A wrong size is a status, not a crash. */
    return rlx_micro_dense_i8(xq, 4, w, 7, NULL, 0, 0, 0, mult, 8, yq, 2) == RLX_MICRO_ERR_SIZE ? 0 : 1;
}

int main(void) {
    port_write("rlx-micro ");
    port_write(rlx_micro_version());
    port_write("\n");
    int32_t digit = rlx_micro_mnist_infer(rlx_micro_mnist_test_image(), RLX_MICRO_MNIST_INPUT_LEN, scratch_a,
                                          scratch_b, sizeof scratch_a);
    port_write("mnist: digit ");
    port_int(digit);
    port_write(" (label ");
    port_int(rlx_micro_mnist_test_label());
    port_write(")\n");
    return (digit == rlx_micro_mnist_test_label() ? 0 : 1) | dense_layer();
}
