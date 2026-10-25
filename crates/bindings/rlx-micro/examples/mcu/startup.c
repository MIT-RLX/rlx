/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Minimal Cortex-M4F/M7 start-up for the examples: vector table, .data copy,
 * .bss clear, FPU on, then main(). Works on any ARMv7E-M part — only link.ld
 * (memory map) is board-specific. A vendor SDK's start-up file does the same
 * job and can replace this one. */
#include <stdint.h>

extern int main(void);
extern void port_exit(int status);

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

void Reset_Handler(void) {
    uint32_t* src = &_sidata;
    for (uint32_t* dst = &_sdata; dst < &_edata;) *dst++ = *src++;
    for (uint32_t* dst = &_sbss; dst < &_ebss;) *dst++ = 0;
    /* CPACR: full access to CP10/CP11 — the kernels use the FPU for requantization. */
    *(volatile uint32_t*)0xE000ED88 |= 0xFu << 20;
    __asm__ volatile("dsb\n isb");
    port_exit(main());
}

static void Fault_Handler(void) {
    port_exit(1);
}

__attribute__((section(".isr_vector"), used)) static void (*const vectors[])(void) = {
    (void (*)(void))(&_estack), /* initial stack pointer */
    Reset_Handler,
    Fault_Handler, /* NMI */
    Fault_Handler, /* HardFault */
    Fault_Handler, /* MemManage */
    Fault_Handler, /* BusFault */
    Fault_Handler, /* UsageFault */
};
