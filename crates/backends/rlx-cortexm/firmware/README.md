# rlx-cortexm-firmware

MNIST-on-nRF52840 firmware demo for [`rlx-cortexm`](../) — a bare-metal
`no_std` binary that runs an INT8 MNIST classifier on an ARM Cortex-M4
(Nordic nRF52840) using the kernels from `rlx-cortexm`.

This is a **standalone crate** (its own `[workspace]`, its own version) and is
**not** part of the main RLX workspace or published to crates.io — it is flashed
to the microcontroller. The INT8 `model_weights.rs` it embeds is emitted by
`rlx-cortexm`'s native fp32 trainer (`../trainer`).

## Build & flash

```bash
# from this directory (needs the thumbv7em-none-eabihf target + probe-rs)
rustup target add thumbv7em-none-eabihf
cargo build --release --target thumbv7em-none-eabihf
probe-rs run --chip nRF52840_xxAA target/thumbv7em-none-eabihf/release/rlx-cortexm-firmware
```

See [`rlx-cortexm`](../README.md) for the kernel/quantization details and the
training workflow that produces the embedded weights.

## License

MIT OR Apache-2.0.
