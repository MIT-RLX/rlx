/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * ARM semihosting: text and the exit status go to the debugger (QEMU with
 * -semihosting, or OpenOCD / pyOCD on a real board). Without a debugger
 * attached the BKPT faults — on a deployed board, print to a UART instead. */
#include <stdint.h>

#include "../port.h"

static int32_t semihost(int32_t op, const void* arg) {
    register int32_t r0 __asm__("r0") = op;
    register const void* r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void port_write(const char* s) { semihost(0x04 /* SYS_WRITE0 */, s); }

void port_exit(int status) {
    /* SYS_EXIT; on M-profile the argument is the reason code itself. */
    semihost(0x18, (const void*)(uintptr_t)(status == 0 ? 0x20026 /* ApplicationExit */
                                                        : 0x20024 /* RunTimeErrorUnknown */));
    for (;;) {
    }
}
