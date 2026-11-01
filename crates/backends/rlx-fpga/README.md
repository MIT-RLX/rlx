# rlx-fpga

FPGA backend for RLX — per-graph datapath synthesis. Takes an `rlx-ir`
graph (or a [`Model`](src/model.rs)) and emits **target-agnostic**
SystemVerilog plus weight `.mem` files, with optional Yosys/nextpnr
scripts for a concrete part.

Like [`rlx-cortexm`](https://crates.io/crates/rlx-cortexm), this crate
sits **one layer below** the runtime `Backend` trait. The entry points
are [`export_graph`](src/export.rs) / `rlx_runtime::export` (feature
`fpga`), not `Session::compile`.

See **[docs/fpga-export.md](../../../docs/fpga-export.md)** for the
full export configuration and `HwTarget` matrix.

## Getting started

```sh
cargo test  -p rlx-fpga --release
cargo run   -p rlx-fpga --release --bin rlx-fpga-emit -- --hw generic
cargo run   -p rlx-fpga --release --example export_mnist
just fpga-emit
just fpga-mnist-demo
```

Checked-in demo: [`examples/mnist_sv/`](examples/mnist_sv/) (`top.sv` + layers + weights).

```rust,ignore
use rlx::prelude::*; // feature = "fpga"

ExportSession::fpga("hw/out")
    .hw_target(HwTarget::Generic)
    .sideband(SidebandSpec::input("temp", 8))
    .sideband(SidebandSpec::input("batch_id", 16))
    .export_model(&tinyconv_mnist_from_cortexm())?;
```

```rust,ignore
use rlx_fpga::{export_graph, FpgaExportConfig, HwTarget};
// ... build or load an INT8 Graph ...
export_graph(&graph, &FpgaExportConfig::default(), "hw/out")?;
```

## Hardware targets

| `HwTarget` | RTL | Sidecars |
|------------|-----|----------|
| `Generic` (default) | soft ports | `synth.sh` (Yosys generic), `EXPORT.md` |
| `Ecp5` (part only) | same RTL | + `constraints.lpf` stub, ecp5 P&R script |
| `Ice40` (part only) | same RTL | + `constraints.pcf` stub |
| `Lattice` board (`lattice:icebreaker`, `lattice:ulx3s-85f`, …) | + UART host-link `board_top` (+ ECP5 PLL) | real PCF/LPF, IceStorm / Trellis `synth.sh` → `out.bin` / `out.bit`, `program.sh`, `run_uart.py` |
| `Xilinx7` (part only) | same RTL | + `constraints.xdc` stub; openXC7 synth |
| `Xilinx7` + board (`xc7:arty-a7-35`, `basys3`, …) | + UART host-link `board_top` | real XDC, openXC7 `synth.sh` → `out.bit`, `program.sh`, `run_uart.py` |

Lattice boards, measured with the generated scripts in the `hdlc/impl`
containers (`tests/lattice_docker.rs`), on the MNIST TinyConv demo:

| board | device | logic | block RAM | DSP | fmax / design clock |
|---|---|---|---|---|---|
| iCEBreaker | iCE40UP5K | 41% | 26 / 30 | 7 / 8 | 16 MHz / 12 MHz |
| ULX3S | LFE5U-85F | 2% | 7 DP16KD | 14 MULT18X18D | 22.6 MHz / 12.5 MHz (PLL) |

ULX3S designs run at 12.5 MHz from the ECP5 PLL. At the 25 MHz oscillator
the shared Q0.31 requantize misses timing by about 2 ns: four chained
`MULT18X18D`s and a carry chain in one cycle.

Ternary (BitNet b1.58) weights + QAT → FPGA: [docs/bitnet-fpga.md](../../../docs/bitnet-fpga.md).

## What's here

- **`export` / `export_config`** — Graph → Model → emit with tune + `HwTarget`
- **`from_graph`** — `Model::from_graph` / `FromGraphOptions`
- **`quant`** — integer-only Q0.31 / Q0.15 requantize
- **`verilog`** — pure-Rust SystemVerilog writer
- **`codegen`** — per-op emitters + `emit_with_config`
- **`reference`** — Rust forward pass (parity vs Verilog)
- **`bin/rlx-fpga-emit`** — CLI

## `no_std`

The default `std` feature carries graph import/export, writing the RTL tree to
disk, and the sequential target. With `--no-default-features` the crate is
`no_std` + `alloc`: `Model` → `optimize` (still verified through rlx-ir's
`no_std` graph/verify core) → in-memory artifacts (`collect_artifacts`), plus
the reference model and weight packing.
`just check-no-std` builds it for `thumbv7em-none-eabihf`.

## License

MIT OR Apache-2.0.
