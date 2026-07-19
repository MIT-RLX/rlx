# FPGA / SystemVerilog export

First-class **offline** export of an inference graph to SystemVerilog.
This is an [`ExportTarget`](../crates/core/rlx-runtime/src/export.rs) /
[`ExportSession`](../crates/core/rlx-runtime/src/export.rs), **not** a
runtime [`Backend`](../crates/core/rlx-runtime/src/backend/mod.rs). There is
no `Device::Fpga` and no `Session::compile` → `run` path onto an FPGA.

## Pipeline

```text
rlx-ir Graph
  → prepare_model / legalize (f32 → INT8 | INT4 | FP4, or pass-through Q*)
  → Model::from_graph  (+ GraphIoBind)
  → optimize(tune)
  → target-agnostic SystemVerilog + .mem  (IoConfig soft ports)
  → synth.sh (+ optional board_top.sv / constraints)
```

Training stays on CPU/GPU. Quantize (or let the legalizer PTQ), then export
the **forward** graph.

## Quick start

```sh
just fpga-emit
just fpga-emit HW=ecp5 TARGET=energy
just fpga-mnist-demo   # refresh examples/mnist_sv/
just test-fpga
```

**Checked-in demo RTL:** [`crates/backends/rlx-fpga/examples/mnist_sv/`](../crates/backends/rlx-fpga/examples/mnist_sv/)
(`top.sv`, `tb.sv`, layers, weights). Regenerate with `just fpga-mnist-demo`.

### Rust — `ExportSession` (preferred)

```rust,ignore
use rlx::prelude::*; // enable crate feature `fpga`

let arts = ExportSession::fpga("hw/out")
    .quant_mode(ExportQuantMode::Int4)   // or Int8, Fp4
    .hw_target(HwTarget::Generic)        // ASIC / SoC: soft ports only
    .output_kind(OutputKind::Logits)     // upgrades I/O to ScalarAndMemory
    .io(IoConfig::default()
        .with_input(InputIface::Memory)
        .with_output(OutputIface::ScalarAndMemory)
        .sideband(SidebandSpec::input("temp", 8))
        .sideband(SidebandSpec::input("batch_id", 16)))
    .bind_input("x")
    .bind_outputs(["logits"])
    .export(&graph)?;

// Or the built-in TinyConv-MNIST weights:
let arts = ExportSession::fpga("hw/mnist")
    .hw_target(HwTarget::Generic)
    .sideband(SidebandSpec::input("temp", 8))
    .export_model(&tinyconv_mnist_from_cortexm())?;
```

### Soft I/O (`IoConfig`)

Default matches the historical TinyConv poke interface. Configure for ASIC/SoC:

| Piece | Options |
|-------|---------|
| `PortNames` | Rename `clk`/`rst`/`start`/`done`/`in_*`/`out_*`/`pred`/stream signals |
| `InputIface` | `Memory` (default), `Stream { beat_elems }`, `MemoryAndStream` |
| `OutputIface` | `ScalarPred` (default), `MemoryReadout`, `Stream`, `ScalarAndMemory`, `MemoryAndStream` |
| `GraphIoBind` | `input`: which `Op::Input`; `outputs`: primary + optional extra readout taps; `sideband_inputs`: scalar Inputs → soft ports |
| `SidebandSpec` | Soft scalar ports (temperature, batch id, …) sampled at `start`; optional `{name}_q` echo |

**Memory input:** `in_addr` / `in_we` / `in_din` — write activation bytes while idle.  
**Stream input:** `in_valid` / `in_ready` / `in_data` — fill the buffer before `start`.  
**Memory readout:** `out_addr` / `out_re` / `out_dout` — read the last-layer buffer after `done` (full logits).  
**Stream output:** `out_valid` / `out_ready` / `out_data` — drain after `done`.  
**Scalar `pred`:** class index (Argmax) or addr-0 peek.  
**Sidebands:** host-driven scalars outside the activation BRAM datapath — e.g. `.sideband(SidebandSpec::input("temp", 8))` or bind Graph Inputs via `bind_sideband_inputs`.

`OutputKind::Logits` auto-selects `ScalarAndMemory` when the output iface is still the default scalar.

CLI: `--in-iface`, `--out-iface`, `--bind-in`, `--bind-out`, `--sideband name[:bits[:signed]]`, `--bind-sideband`.

### Quant modes

| Mode | Weights | Notes |
|------|---------|--------|
| `Int8` | signed INT8 | default |
| `Int4` | nibble-packed signed INT4 | |
| `Fp4` | F4E2M1 codes, nibble-packed | MAC via FP4→fixed LUT (`ENCODING=1`) |

f32 `Conv`/`MatMul` graphs need `LegalizeOptions.weights_f32` (or baked
`Op::Constant` weights). `Op::ScaledMatMul` is rejected with guidance to
use `ExportQuantMode::Fp4` on an f32 MatMul instead.

### Board shell

`HwTarget::Generic` (default): soft-port `top.sv` only — preferred for ASIC.  
`Ecp5` / `Ice40` / `Xilinx7`: same RTL + `board_top.sv` pin wrapper + stub constraints.

### Output kinds

- `OutputKind::Argmax` (default) — class index on `pred`
- `OutputKind::Logits` — drop trailing Argmax; use memory readout for the full vector

### Per-channel requant side tables

`to_graph` bakes `{layer}_requant_m0` + `{layer}_requant_shift` Constants
(plus interleaved `{layer}_requant` for compatibility). Prefer these over
the scalar `mult` on `QConv2d`/`QMatMul`.

### Python (`pyrlx`, feature `fpga`)

```python
import pyrlx
files = pyrlx.export_fpga(
    graph, "hw/out",
    quant="fp4", hw="generic",
    in_iface="memory", out_iface="scalar+memory",
    bind_in="x", bind_out="logits",
    sideband="temp:8,batch_id:16",
)
```

## See also

- [`crates/backends/rlx-fpga/README.md`](../crates/backends/rlx-fpga/README.md)
