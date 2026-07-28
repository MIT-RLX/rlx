# Compiling XDNA NPU overlays (INT8 GEMM) — Python-free

An "overlay" is the AIE program rlx runs on the NPU: a `.xclbin` + a paired
`insts_*.bin` instruction stream. `rlx-xdna` **loads** these at runtime with zero
Python (Rust → `csrc/xrt_gemm_shim.cpp` → libxrt → NPU; see `src/npu_gemm.rs`),
and **compiles** them with zero Python too — see below.

## The compiler is a native binary; `aiecc.py` is just a shim

The real MLIR-AIE compiler is a native ELF binary — `mlir_aie/bin/aiecc`
(~212 MB, **links no `libpython`**). `aiecc.py` is a 114-line Python shim that
just `exec`s it. So Python-free overlay generation = **invoke the native binary
directly**. Verified: it compiles `aie.mlir → xclbin + insts` with `python3`
symlinked to `/bin/false` and no Python on `PATH`, and the resulting overlay
runs on the NPU through rlx (`Device::Xdna` matmul, bit-exact vs CPU).

Internally the native `aiecc` runs `aie-opt` / `aie-translate` + Peano
(`opt`/`llc`/`ld.lld` from the `llvm-aie` package) + `bootgen` + `xclbinutil` —
all native, `--no-xchesscc` so **no Vitis/Chess** is needed.

## Compile one

Shell (`tools/compile_overlay.sh`):

```bash
MLIR_AIE_INSTALL_DIR=<.../mlir_aie> PEANO_INSTALL_DIR=<.../llvm-aie> \
  ./compile_overlay.sh aie.mlir overlay.xclbin overlay_insts.bin
```

From Rust (`rlx_xdna::compile::compile_overlay`):

```rust
rlx_xdna::compile::compile_overlay(&rlx_xdna::compile::OverlaySpec {
    aiecc: "…/mlir_aie/bin/aiecc",     // the NATIVE binary
    peano: "…/llvm-aie",
    mlir:  "aie.mlir",
    tmpdir:"./ovbuild",
    out_xclbin: "overlay.xclbin",
    out_insts:  "overlay_insts.bin",
})?;
```

Then point rlx at it: `RLX_XDNA_SHIM`, `RLX_XDNA_XCLBIN`, `RLX_XDNA_INSTS`,
`RLX_XDNA_GEMM=M,K,N` (see `XdnaBackend`). The `.xclbin` is a shippable data
artifact — compile once, ship, load with zero Python.

## Where the `aie.mlir` comes from

Today it's emitted by an IRON design (the one-time step that still uses the
mlir-aie Python API), or reused from a prior build's `<name>.prj/aie.mlir`. The
clean long-term shape mirrors rlx's other codegen backends
(`rlx-cerebras`→CSL, `rlx-fpga`→SystemVerilog): **rlx emits the AIE MLIR itself**,
then this native `aiecc` compiles it — no IRON Python design files at all.
## License

MIT OR Apache-2.0.
