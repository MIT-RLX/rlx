# rlx-cerebras

Cerebras Wafer-Scale Engine (WSE) backend for RLX — lowers an `rlx-ir` graph to
**CSL** (Cerebras Software Language) and the host harness that drives the
Cerebras SDK **fabric simulator**.

## Why CSL

Cerebras exposes three ingestion surfaces:

| Surface | What it is | Usable as an RLX backend? |
| --- | --- | --- |
| Inference API | OpenAI-compatible REST, hosted models only | No — serves their models, not your graph |
| PyTorch `cstorch` / PJRT-StableHLO | Lazy ATen → CIRH (MLIR) → CGC | Needs a CS-2/CS-3 appliance |
| **SDK / CSL** | C-like dataflow language + **cycle-accurate simulator** | **Yes — runs without wafer hardware** |

Only the CSL path ships a simulator that runs on commodity machines, so it is
the one RLX can target and validate end-to-end. This crate mirrors `rlx-fpga`'s
ethos (emit source, validate against a Rust oracle) — but because the simulator
runs locally, we can close the loop and actually execute the emitted program.

## Pipeline

```
rlx-ir Graph
  → rlx-cerebras::model    (recognize the supported subgraph, read shapes)
  → rlx-cerebras::codegen  (emit layout.csl + pe_program.csl + run.py + commands.sh)
  → cslc --memcpy + cs_python run.py   (Cerebras SDK container, Linux host)
```

## Status — milestone 1

- **Single rank-2 `MatMul` on one PE.** Host data streamed in/out via the
  `memcpy` library (the `gemv-03-memcpy` tutorial shape), parameterized by M/K/N.
- **Scalar kernel** — correctness first. The DSD/`@fmacs` vectorization and
  **multi-PE tiling** (where wafer-scale perf lives) are the next milestones;
  single-PE is only a correctness stepping stone.
- **Validated:** the Rust oracle + artifact structure are unit-tested here
  (`cargo test -p rlx-cerebras`).
- **Not yet validated:** compiling the CSL with `cslc` and running on the
  simulator needs the SDK Singularity container on a **Linux host** (e.g. ALCF)
  — not macOS/Windows. That step closes the loop and is tracked as next work.

## Emit

```sh
cargo run -p rlx-cerebras --bin rlx-cerebras-emit -- 32 64 32 ./cerebras-out
# then, on a Linux host with the Cerebras SDK container:
cd cerebras-out && bash commands_wse2.sh   # cslc + cs_python run.py → "SUCCESS!"
```
