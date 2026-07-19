# RLX FPGA export — `tinyconv_mnist`

| Field | Value |
|-------|-------|
| Model | `tinyconv_mnist` |
| Input length | 784 |
| Layers | 8 |
| Tune | `Tune { fold_zp=true ternary_fast=false shared_requant=false bram_en=false requant=Q0_31 P=1 P_ic=1 }` |
| Hardware target | generic (target-agnostic) |
| RTL | **target-agnostic** SystemVerilog (`top.sv` soft ports) |

## Ports (`top`)

| Port | Dir | Role |
|------|-----|------|
| `clk` | in | clock |
| `rst` | in | synchronous reset |
| `start` | in | pulse to begin inference |
| `done` | out | inference complete |
| `in_addr` / `in_we` / `in_din` | in | write input activation bytes |
| `temp` | in | sideband 8-bit (sampled at start) |
| `temp_q` | out | registered sideband echo |
| `batch_id` | in | sideband 16-bit (sampled at start) |
| `batch_id_q` | out | registered sideband echo |
| `pred` | out | argmax / output byte (addr 0 peek) |
| `out_addr` / `out_re` / `out_dout` | in/out | read last-layer buffer (len=1) |

## Soft I/O

| Field | Value |
|-------|-------|
| Input iface | `Memory` |
| Output iface | `ScalarAndMemory` |
| Bind input | `(first Op::Input)` |
| Bind outputs | `(graph.outputs[0])` |

## Build

```sh
bash synth.sh
```

Generic target produces `out.json` only. Board targets (`ecp5`, `ice40`, …)
run place-and-route when the open toolchain is on `PATH`.

## Parity

Bit-exact reference: `rlx_fpga::reference` (Rust) ↔ emitted Verilog.
Do not diff against f32 Cortex-M requant paths.
