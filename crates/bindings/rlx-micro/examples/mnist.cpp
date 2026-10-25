// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// rlx-micro from C++ (rlx/micro.hpp): the same model and layer as mnist.c,
// through spans and a model object that owns its scratch. Freestanding — no
// exceptions, RTTI, heap or STL — so it builds with -fno-exceptions -nostdlib
// for a Cortex-M as well as for a desktop.
#include <stdint.h>

#include "port.h"
#include "rlx/micro.hpp"

namespace micro = rlx::micro;

static micro::Mnist model;  // 2 × 5408 B of scratch, in .bss

static bool dense_layer() {
    const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float x_scale = 0.05f, w_scale = 0.1f, y_scale = 0.05f;
    const int8_t w[8] = {10, 0, 0, 0, 0, 0, 0, 10};
    const float m = x_scale * w_scale / y_scale;
    const float mult[2] = {m, m};
    int8_t xq[4], yq[2];
    float y[2];
    if (!micro::ok(micro::quantize(x, x_scale, 0, xq))) return false;
    if (!micro::ok(micro::dense(xq, w, nullptr, 0, 0, 0, mult, 8, yq))) return false;
    micro::dequantize(yq, y_scale, 0, y);
    port_write("dense: ");
    port_fixed2(y[0]);
    port_write(" ");
    port_fixed2(y[1]);
    port_write("\n");
    // Too small an output buffer is refused.
    return micro::dense(xq, w, nullptr, 0, 0, 0, mult, 8, micro::Span<int8_t>(yq, 1)) == micro::Status::Size;
}

int main() {
    port_write("rlx-micro ");
    port_write(micro::version());
    port_write("\n");
    const int32_t digit = model.classify(micro::Mnist::test_image());
    port_write("mnist: digit ");
    port_int(digit);
    port_write(" (label ");
    port_int(micro::Mnist::test_label());
    port_write(")\n");
    return (digit == micro::Mnist::test_label() && dense_layer()) ? 0 : 1;
}
