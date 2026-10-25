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
| `Ternary` | BitNet b1.58 `{-1,0,1}`, 2-bit packed | absmean (`rlx_ir::ternary`); multiplier-free add/sub/skip MAC on every tune. Train with QAT first — see [bitnet-fpga.md](bitnet-fpga.md) |

f32 `Conv`/`MatMul` graphs need `LegalizeOptions.weights_f32` (or baked
`Op::Constant` weights). `Op::ScaledMatMul` is rejected with guidance to
use `ExportQuantMode::Fp4` on an f32 MatMul instead.

### Board shell

`HwTarget::Generic` (default): soft-port `top.sv` only — preferred for ASIC.  
`Ecp5` / `Ice40` / `Xilinx7 { board: None }`: same RTL + `board_top.sv` pin wrapper + stub constraints.

### openXC7 boards (`HwTarget::xc7_board`, `--hw xc7:<board>`)

Xilinx 7-series through the open [openXC7](https://github.com/openXC7) flow.
Presets: `arty-a7-35`, `arty-a7-100`, `basys3`, `nexys-a7-100`. The export adds:

| File | Role |
|------|------|
| `board_top.sv` | power-on reset + **UART host link** (115200 8N1) around soft `top` |
| `constraints.xdc` | real clock / reset / UART / LED pins |
| `synth.sh` | `yosys synth_xilinx -family xc7` → `nextpnr-xilinx` → `fasm2frames` → `xc7frames2bit` → `out.bit` (chipdb built once, cached in `~/.cache/rlx-fpga/chipdb`) |
| `program.sh` | `openFPGALoader -b <board> out.bit` (`FLASH=1` writes SPI flash) |
| `run_uart.py` | host side (pyserial): sends `.mem` stimulus, prints `pred` / logits |

Protocol: host sends `0xA5` + `input_len` i8 bytes; the board replies with
`pred` (pred port) and then every output byte (memory readout,
`OutputIface::ScalarAndMemory`). Toolchain paths: `OPENXC7_PREFIX` (default
the snap, `/snap/openxc7/current/opt`), `PRJXRAY_DB_DIR`,
`NEXTPNR_XILINX_PYTHON_DIR`, `CHIPDB_DIR`; extra flags via `YOSYS_SYNTH_OPTS`
/ `NEXTPNR_OPTS`.

No local install? Run the generated script in the openXC7 container:

```sh
docker run --rm --platform linux/amd64 -v "$PWD":/work -v ~/.cache/rlx-fpga/chipdb:/chipdb \
    -e CHIPDB_DIR=/chipdb -w /work regymm/openxc7 bash ./synth.sh
```

`tests/openxc7_docker.rs` (`--ignored`) does exactly this and requires
`out.bit` plus post-route fmax ≥ the board clock. Synthesis uses
`-nolutram`: nextpnr-xilinx cannot place distributed RAM.

`tests/xc7_uart.rs` drives this link in simulation — RTL, and the
`synth_xilinx` netlist against Yosys's Xilinx cell models — and compares
the replies with `rlx_fpga::reference`.

### RTL simulation parity

`tests/sim_parity.rs` simulates the emitted RTL under Icarus Verilog and
checks every logit and the argmax against `rlx_fpga::reference`: dense and
conv models × 2/4/8-bit weights × every tune preset, plus TinyConv-MNIST.
Skipped when `iverilog` is not installed.

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

## `no_std` builds

`rlx-fpga` with `default-features = false` is `no_std` + `alloc`, for building
RTL on a target with no OS: `Model` (e.g. `tinyconv_mnist_from_cortexm`) →
`passes::optimize` (the model graph is still checked by `rlx_ir::verify`, from
rlx-ir's `no_std` core) → `collect_artifacts_opt`, which returns the files in
memory. Graph import (`Model::from_graph`, `prepare_model`), `export_graph`,
writing the tree to disk (`emit_*`), the sequential target and the CLI need the
default `std` feature. `just check-no-std` builds it for `thumbv7em-none-eabihf`.

## See also

- [`crates/backends/rlx-fpga/README.md`](../crates/backends/rlx-fpga/README.md)
## License

MIT OR Apache-2.0.
