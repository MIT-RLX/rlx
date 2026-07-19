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

## Quick start

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
| `Ecp5` | same RTL | + `constraints.lpf` stub, ecp5 P&R script |
| `Ice40` | same RTL | + `constraints.pcf` stub |
| `Xilinx7` | same RTL | + `constraints.xdc` stub (experimental) |

## What's here

- **`export` / `export_config`** — Graph → Model → emit with tune + `HwTarget`
- **`from_graph`** — `Model::from_graph` / `FromGraphOptions`
- **`quant`** — integer-only Q0.31 / Q0.15 requantize
- **`verilog`** — pure-Rust SystemVerilog writer
- **`codegen`** — per-op emitters + `emit_with_config`
- **`reference`** — Rust forward pass (parity vs Verilog)
- **`bin/rlx-fpga-emit`** — CLI

## License

GPL-3.0-only.
