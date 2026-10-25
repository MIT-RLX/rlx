/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * The one thing the examples need from their platform: a way to print text.
 * port_host.c writes to stdout; mcu/semihost.c writes through the debugger
 * (ARM semihosting — QEMU, or a probe with OpenOCD / pyOCD). On a board,
 * point `port_write` at a UART instead. No printf, so no libc is needed. */
#ifndef RLX_MICRO_PORT_H
#define RLX_MICRO_PORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void port_write(const char* s);

static inline void port_int(int32_t v) {
    char buf[12];
    char* p = buf + sizeof buf;
    uint32_t u = v < 0 ? (uint32_t)(-(int64_t)v) : (uint32_t)v;
    *--p = '\0';
    do {
        *--p = (char)('0' + u % 10);
        u /= 10;
    } while (u);
    if (v < 0) *--p = '-';
    port_write(p);
}

/* `v` with two decimals, e.g. 4.00. */
static inline void port_fixed2(float v) {
    int32_t c = (int32_t)(v * 100.0f + (v < 0 ? -0.5f : 0.5f));
    if (c < 0) {
        port_write("-");
        c = -c;
    }
    port_int(c / 100);
    port_write(".");
    port_write(c % 100 < 10 ? "0" : "");
    port_int(c % 100);
}

#ifdef __cplusplus
}
#endif

#endif
