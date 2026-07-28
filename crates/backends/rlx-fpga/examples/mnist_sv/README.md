# TinyConv-MNIST SystemVerilog demo

Checked-in RTL from `rlx-fpga` for the TinyConv-MNIST INT8 classifier
(28×28 → 10 classes). **Target-agnostic soft ports** — suitable for ASIC
integration or FPGA board shells.

## Regenerate

```sh
cargo run -p rlx-fpga --example export_mnist --release
# or: just fpga-mnist-demo
```

## Layout

| Path | Role |
|------|------|
| [`top.sv`](top.sv) | Soft-port controller + arena BRAMs + layer instances |
| [`tb.sv`](tb.sv) | Verilator-style image TB (`tb_image.mem`) |
| [`EXPORT.md`](EXPORT.md) | Port table + I/O config |
| `layers/*.sv` | Conv / pool / dense / argmax kernels |
| `primitives/*.sv` | BRAM, requant, weight_unpack |
| `weights/*.mem` | Packed INT8 weights / bias / requant tables |
| `synth.sh` | Generic Yosys sketch |

## Soft ports (`top`)

```text
clk, rst, start, done
in_addr / in_we / in_din     — load 784 INT8 pixels while idle
temp / temp_q                — 8-bit sideband (sampled at start)
batch_id / batch_id_q        — 16-bit sideband (sampled at start)
pred                         — class index (Argmax)
out_addr / out_re / out_dout — memory readout of the result buffer
```

## Rust (prelude)

```rust,ignore
use rlx::prelude::*; // feature = "fpga"

let arts = ExportSession::fpga("hw/out")
    .hw_target(HwTarget::Generic)
    .io(IoConfig::default()
        .with_output(OutputIface::ScalarAndMemory)
        .sideband(SidebandSpec::input("temp", 8))
        .sideband(SidebandSpec::input("batch_id", 16)))
    .export_model(&tinyconv_mnist_from_cortexm())?;
```

See [docs/fpga-export.md](../../../../docs/fpga-export.md).
## License

MIT OR Apache-2.0.
