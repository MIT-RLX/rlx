# rlx-micro — rlx on microcontrollers, from C and C++

The C ABI of [`rlx-cortexm`](../../backends/rlx-cortexm): INT8 conv2d, dense,
max-pool, ReLU, argmax and (de)quantization, plus a bundled TinyConv MNIST
model. It is `no_std`, never allocates, and needs no OS or libc. The whole
MNIST firmware takes **24.9 KB flash and 10.8 KB RAM**. The same library also
builds for a desktop, so you can check firmware logic there first.

```c
#include "rlx_micro.h"

static int8_t a[RLX_MICRO_MNIST_SCRATCH_LEN], b[RLX_MICRO_MNIST_SCRATCH_LEN];
int32_t digit = rlx_micro_mnist_infer(image, RLX_MICRO_MNIST_INPUT_LEN, a, b, sizeof a);
```

```cpp
#include "rlx/micro.hpp"   // freestanding: no exceptions, heap or STL

static rlx::micro::Mnist model;          // owns its scratch, in .bss
int32_t digit = model.classify(image);   // Span<const int8_t> from an array
```

## Which library, where

| Hardware | Library | What runs |
|---|---|---|
| Servers, desktops | [`rlx-capi`](../rlx-capi) (`rlx.h`), [`rlx-cpp`](../rlx-cpp/README.md) | The full runtime: any graph, any backend (CPU, Metal, CUDA, ROCm, wgpu, …), training, models |
| Embedded Linux (Raspberry Pi, Jetson, i.MX, …) | The same `rlx-capi`, cross-built: `cargo build -p rlx-capi --release --target aarch64-unknown-linux-gnu` | The same as on a server. Its test suite runs in `linux/arm64` containers on Apple Silicon |
| Microcontrollers (Cortex-M4F / M7, no OS, KBs of RAM) | **`rlx-micro`** (`rlx_micro.h`, `rlx/micro.hpp`) | INT8 kernels and fixed models, with static buffers |

## Build and run

On the host:

```bash
cargo build --release -p rlx-micro
cd crates/bindings/rlx-micro
cc  -std=c11   -Iinclude -Iexamples examples/mnist.c   examples/port_host.c ../../../target/release/librlx_micro.a -o mnist   && ./mnist
cc  -std=c11   -Iexamples -c examples/port_host.c -o port_host.o
c++ -std=c++17 -Iinclude -Iexamples examples/mnist.cpp port_host.o          ../../../target/release/librlx_micro.a -o mnist++ && ./mnist++
```

On Linux, add `-lpthread -ldl -lm`, because the host build uses std.

On a Cortex-M4F, as bare-metal firmware in QEMU's Arm MPS2-AN386 board. This
needs `rustup target add thumbv7em-none-eabihf`, `arm-none-eabi-gcc` and
`qemu-system-arm`. It does not need newlib.

```bash
cargo build --release -p rlx-micro --target thumbv7em-none-eabihf
cd crates/bindings/rlx-micro
CPU="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16"
for f in examples/mnist.c examples/mcu/startup.c examples/mcu/semihost.c; do
  arm-none-eabi-gcc $CPU -ffreestanding -O2 -ffunction-sections -fdata-sections -Iinclude -Iexamples -c $f -o $(basename $f .c).o
done
arm-none-eabi-gcc $CPU -nostdlib -Wl,--gc-sections -Texamples/mcu/link.ld \
  mnist.o startup.o semihost.o ../../../target/thumbv7em-none-eabihf/release/librlx_micro.a -lgcc -o mnist.elf
qemu-system-arm -M mps2-an386 -nographic -monitor none -serial none \
  -semihosting-config enable=on,target=native -kernel mnist.elf
```

For C++, compile `mnist.cpp` with `arm-none-eabi-g++ -std=c++17
-fno-exceptions -fno-rtti -fno-threadsafe-statics`. Link it the same way.

Every build prints:

```
rlx-micro 0.2.19
mnist: digit 7 (label 7)
dense: 1.00 4.00
```

`cargo test -p rlx-micro` runs all four builds: C and C++, on the host and in
QEMU. The QEMU half is skipped when its tools are missing.

## On a real board

Only two files are board-specific:

- **`examples/mcu/link.ld`**: the memory map. For example, an STM32F4 or
  nRF52840 has its flash at `0x08000000` or `0x0`, with RAM at `0x20000000`.
- **`port_write`**: semihosting prints through a debug probe (OpenOCD, pyOCD,
  probe-rs). On a board with no probe attached, the semihosting breakpoint
  faults, so point `port_write` at a UART instead.

A vendor SDK's start-up file can replace `examples/mcu/startup.c`. Link
`librlx_micro.a` into the SDK project like any static library, built for the
same `--target` as the core:

| Core | Target |
|---|---|
| M4F / M7 | `thumbv7em-none-eabihf` |
| M4 / M7 with no FPU | `thumbv7em-none-eabi` |
| M33 | `thumbv8m.main-none-eabihf` |

Turn off the `panic-handler` feature when the firmware is written in Rust and
brings its own handler. For a Rust firmware, depending on `rlx-cortexm`
directly is simpler. See
[`rlx-cortexm/firmware`](../../backends/rlx-cortexm/firmware/README.md), which
targets the nRF52840.

## Your own model

The kernels are generic. Chain them the way
[`rlx-cortexm/src/model.rs`](../../backends/rlx-cortexm/src/model.rs) does:
conv2d → ReLU → max-pool → dense → argmax, ping-ponging between two scratch
buffers.

- **Weights**: per-channel symmetric INT8, packed at 8, 4 or 2 bits, with
  `mult[c] = x_scale · w_scale[c] / out_scale`. The rlx-cortexm trainer
  (`rlx-cortexm/trainer`) emits them, or any quantizer that follows the same
  scheme can.
- **Errors**: every call checks its buffer lengths against the shape. A
  mismatch returns `RLX_MICRO_ERR_SIZE` and nothing is written.

License: MIT OR Apache-2.0.
