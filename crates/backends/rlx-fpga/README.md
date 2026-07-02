# rlx-fpga

FPGA backend for RLX — per-graph datapath synthesis. Takes an `rlx-ir`
graph and emits a single self-contained SystemVerilog file plus weight
`.mem` files, ready for `yosys → nextpnr → <board>pack`.

Like [`rlx-cortexm`](https://crates.io/crates/rlx-cortexm), this crate
sits **one layer below** the rest of the workspace: it does not
implement the `Backend` trait. The runtime can't schedule onto an FPGA
the way it schedules onto a GPU — synth + P&R takes minutes, not
microseconds — so the entry point is a `cargo run` that emits hardware,
not a `Session::compile`.

## What's here

- **`src/quant.rs`** — integer-only Q0.31 requantize (gemmlowp /
  TFLite-Micro / CMSIS-NN style):
  - `quantize_multiplier(M_real: f32) -> (M0: i32, shift: i32)` — split
    a positive real multiplier into a Q0.31 fixed-point significand
    `M0 ∈ [2³⁰, 2³¹)` and a non-negative right shift.
  - `srdhm(a, b)` — saturating rounding doubling high multiply
    (`(2·a·b + 2³¹) >> 32`, saturating at `i32::MIN/MAX`).
  - `rdpot(x, shift)` — rounding divide by power of two.
  - `requantize_q31(acc, m0, shift, out_zp)` — full epilogue.
- **`src/verilog.rs`** — pure-Rust Verilog writer (`V::module`,
  `V::line`, `V::block`, `$readmemh`, `always_ff`, etc.). Synthesizable
  subset that Yosys's default frontend handles cleanly.
- **`src/codegen/`** — one Rust function per primitive op
  (`mac`, `bram`, `requant`, `relu`, `maxpool`, `conv2d`, `dense`,
  `argmax`, `top`). Each emits a parameterized SystemVerilog module.
- **`src/model.rs`** — TinyConv-MNIST graph description, mirroring
  `rlx-cortexm::model`. Shapes / layer sequence in one place so codegen
  and reference share their source of truth.
- **`src/reference.rs`** — Rust forward pass using the same Q0.31
  epilogue. Bit-exact parity oracle against emitted Verilog.
- **`src/weights.rs`** — pulls the cortexm INT8 blob (already trained)
  and converts every `*_MULT: &[f32]` into `(m0, shift)` tables.
- **`src/bin/emit.rs`** — `cargo run -p rlx-fpga --bin rlx-fpga-emit`.
  Writes `hw/tinyconv_mnist/{top.sv, weights/*.mem}`.

## Quantization

`rlx-cortexm` uses a single f32 multiplier in each requant. FPGA fabric
doesn't have an FPU — a soft-FPU is hundreds of LUTs per requant, which
defeats the point of the backend. So this crate ports the epilogue to
**integer-only Q0.31** (same shape as TFLite Micro / CMSIS-NN /
gemmlowp).

This is **not** bit-exact with the cortexm f32 path. It *is* bit-exact
across `reference` (Rust) ↔ emitted Verilog ↔ silicon — that's the
parity loop the test suite enforces.

Per-tensor symmetric for activations and weights, same as cortexm.
Per-channel weight scales come along for free because every
`(m0, shift)` table is per-output-channel.

## Weight bit-widths

`Layer::Conv2d` and `Layer::Dense` both carry `weight_bits ∈ {2, 4, 8}`;
the emitted Verilog handles all three:

- **8-bit (i8)** — one weight per byte; weight ROM depth = logical count.
- **4-bit (i4)** — two nibbles per byte, low first. Range `[-7, 7]`
  emitted (`-8` codepoint reserved). ROM depth = `ceil(N/2)`.
- **2-bit (ternary)** — four crumbs per byte, LSB up. Range `{-1, 0, 1}`
  emitted (`-2` codepoint reserved). ROM depth = `ceil(N/4)`.

The shared primitive `weight_unpack #(.BITS(W_BITS))` is a
combinational byte → 32-bit-signed extractor that mirrors
`rlx_cortexm::quant::read_weight` exactly. The kernel FSM splits a
logical weight index into `(byte_addr, lane)` at the same cycle it
issues the MAC, with no extra latency.

`pack(weights, bits)` / `packed_byte_len(n, bits)` in `src/pack.rs`
build the byte stream test layers expect (production weights come from
`rlx-cortexm-trainer` already in this layout). For ternary Conv2d /
Dense parity, see `tests/ternary.rs`.

A future optimization: ternary multiplies `(x · w)` with `w ∈ {-1, 0, 1}`
collapse to `add / sub / skip`, which would let the kernel drop the
DSP slice entirely. Not done in this commit; the generic i32 multiply
path handles ternary correctly today, just less efficiently.

## Install

```toml
[dependencies]
rlx-fpga = "0.1"
```

## Build / test

```sh
cargo test  -p rlx-fpga --release
cargo run   -p rlx-fpga --release --bin rlx-fpga-emit
```

The binary writes to `rlx-fpga/hw/tinyconv_mnist/`. From there:

```sh
# (Future, not in this commit:)
yosys -p 'synth_ecp5 -top top -json out.json' hw/tinyconv_mnist/top.sv
nextpnr-ecp5 --json out.json --textcfg out.config --25k --package CABGA381
ecppack out.config out.bit
```

## Tooling

Open-source toolchains only — `yosys` / `nextpnr` / `icepack` /
`ecppack` / `apicula` run identically on macOS / Linux / Windows.
Vivado / Quartus are not supported; Xilinx 7-series and Intel Cyclone
targets are out of scope until there's a strong reason.

## Gotchas

- **No `Backend` trait impl.** Same posture as `rlx-cortexm`: the
  optimizer / memory plan / runtime don't make sense for hardware that
  takes minutes to recompile. The entry point is the emit binary.

- **Q31 ≠ f32.** The MNIST predictions match the cortexm INT8 path on
  the embedded `TEST_IMAGE` and on bulk validation, but per-pixel
  intermediate logits will differ by ≤1 ulp at each requantize. Don't
  diff against `rlx-cortexm` byte-for-byte; diff against
  `rlx_fpga::reference`.

- **`$readmemh` paths are relative.** The emitter writes weight files
  under `hw/<model>/weights/*.mem` and references them from `top.sv`
  via `$readmemh("weights/...")`. Run the simulator / synth from the
  `hw/<model>/` directory, or pass an absolute `+define+`.

## License

GPL-3.0-only.
