# rlx-mcu

`no_std` building blocks for running ML models (speech so far) on
microcontrollers with no allocator. The model crates built on it live in
`rlx-models`: `rlx-conformer-ctc-core`, `rlx-citrinet-core`,
`rlx-moonshine-core` and `rlx-kroko-core`.

[`rlx-cortexm`](../rlx-cortexm) is the other MCU crate here: hand-written
INT8 conv / dense / pool kernels with zero points for small CNNs. This crate
targets transformer and conformer speech models: symmetric weights,
per-row dynamic activations, wide matrices read in place from flash.

* **`blob`**: the in-place model file. One flat, word-aligned, little-endian
  blob read straight from memory-mapped flash, or from PSRAM it was copied
  into from external NAND or an SD card at boot. Records: `u32s`, `f32s`,
  `q8` (int8, one scale per row), `q4` (int4, one scale per group), `q8g`
  (int8, one scale per group), and a tagged `mat` so the exporter picks the
  encoding per matrix.
* **`kernels`**: int8 × int8 → i32 tiles (2 rows × 4 outputs), int4 and
  group-scaled int8 dot products, a streaming argmax for wide LM heads,
  LayerNorm, SiLU, GELU and softmax through `libm`. Host and MCU builds
  produce the same bits.
* **`mel`**: the NeMo log-mel frontend (radix-2 FFT over fixed tables);
  models supply their window and banded filterbank through `MelSource`.
* **`Pcm`**: `[i16]` or `[f32]` audio input.
* **`Pager`**: reads a layer's weights from external NAND, eMMC or an SD card
  into a caller-owned buffer, for models whose file does not fit in RAM
  (`SlicePager` when the file is memory-mapped).
* **`next_cut`**: cuts long audio at its quietest point, for models whose
  scratch grows with the utterance. The same cuts on every target.
* **`split`**: one model across several chips or cores, each owning a
  contiguous range of layers. `balance` picks cut points with the smallest
  largest stage. `send_rows` / `recv_rows` frame f32 activations for any
  byte `Link` (SPI, UART, shared RAM) with a magic word, sequence number,
  shape and CRC-32. The payload is the f32 bits unchanged, so a pipeline gives
  the same bits as one chip.

Feature **`dsp`** (Arm Cortex-M4/M7/M33/M55, built with a matching
`-C target-cpu`): int8 dot products with `SXTB16`/`SMLAD`, and `linear_q8`
as a 2×2 tile in inline assembly over activations widened once per row
pair. That loop uses 19 instructions per 16 MACs, within the 12 registers Thumb
inline assembly gets. The same bits as the portable loops (integer sums are
exact). Stable Rust does not expose Arm target features to `cfg`, hence a
crate feature.

Measured with the `rlx-models` crates:

| model | crate | file | scratch, 5.2 s | rv32imafc, 5.2 s | WER (dev-clean list) |
|---|---|---|---|---|---|
| Citrinet-256 (CTC) | `rlx-citrinet-core` | 13.3 MB | 0.86 MB | 6.1 G instr (Cortex-M33 + `dsp`: 2.59 G) | 3.58% (500 utts) |
| Conformer-CTC small | `rlx-conformer-ctc-core` | 13.9 MB | 1.5 MB | 13.9 G instr | 3.60% (500 utts) |
| Moonshine streaming tiny | `rlx-moonshine-core` | 37.4 MB | 2.7 MB | 20.7 G instr | 5.71% (200 utts) |
| Kroko-L (Zipformer2 RNN-T) | `rlx-kroko-core` | 99 MB int4, paged | 6.9 MB + 10 MB page | 124 G instr | 2.96% int8, 3.01% int4 (100 utts) |

All four are bit-identical between the host build and the bare-metal
rv32imafc build (QEMU). None is real time on an ESP32-P4 without its SIMD:
at 400 MHz and one instruction per cycle, the 5.2 s clip takes 15 s
(Citrinet) to 310 s (Kroko-L).

On a Cortex-M33 with `dsp` (the nRF54H20's application core, QEMU MPS2
AN505), Citrinet takes 2.59 G instructions for the clip in 768 KB of RAM.
Split over two such chips with `split` it takes about 1.3 G each, under the
1.66 G cycles a 320 MHz core has in 5.2 s (`rlx-citrinet-nrf54h` in
`rlx-models`).

Group scales exist because one scale per row is sometimes not enough. In
Moonshine's frontend convs, per-row int8 lost 1.3% (cosine), which the
encoder amplified into a 23% miss one layer later. One scale per 16 values
brought the stage back to 0.9997.
## License

MIT OR Apache-2.0.
