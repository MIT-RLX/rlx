# Reference parity (#5) & kernel occupancy (#8)

Two rig/reference-gated workstreams. Unlike the other hardening items, these
cannot be *completed* from a dev box — #5 needs a reference implementation
(llama.cpp / NeMo) + weights to diff against, and #8 needs a real GPU to profile.
This document is the actionable harness/candidate plan so the runs are turnkey
once on the right machine.

---

## #5 — Reference parity (vs llama.cpp / NeMo), not just cross-backend

**Status today.** RLX has strong *cross-backend* parity (~177 `*_parity.rs`,
`rlx-opscope/parity.rs`), but for several shipped models — Qwen3.5 (`qwen35`) and
the NeMo Conformer — the note in the memory is explicit: *kernel parity is
self-consistency only, not verified against a reference decoder.* "Fast but
unverified-correct" is the worst quadrant for a perf framework; closing this is
higher-value than most new features.

**Where it lives.** rlx core stays model-agnostic (no model-shaped tests in
core). The reference-parity tests for `qwen35`/`nemo` therefore belong in the
sibling **rlx-models** workspace, next to the model builders. Core provides only
the *model-agnostic comparison metric* (below).

### Harness shape (per model)
1. **Produce the reference.** Run the model in the reference engine on a fixed
   prompt + greedy decode, dumping per-token logits (or final hidden states):
   - Qwen3.5 → `llama.cpp`: `llama-cli -m <gguf> -p <prompt> --logits-all` (or a
     small `llama_get_logits_ith` dump), save to `fixtures/qwen35/<case>.logits.f32`.
   - NeMo Conformer → NVIDIA NeMo: run the encoder on a fixed feature tensor, dump
     encoder outputs.
   Record: model revision/hash, quantization, prompt, tokenizer, seed, engine
   version. (This is exactly the provenance an FDA "validated configuration"
   would pin — the parity fixture doubles as validation evidence.)
2. **Run RLX** on the same input via the runner (`Qwen35Runner` / `rlx-nemo`),
   `RLX_CPU_MATMUL_F64_ACCUM=1` for the tightest accumulation, deterministic RNG.
3. **Compare** with the metric below; assert thresholds; the test **skips**
   (not fails) when the fixture env var is unset, so CI without the large
   reference artifacts stays green.

### Comparison metric (model-agnostic — the reusable piece)
Report all four, gate on the first three:
- **Cosine similarity** of the logit/hidden vectors (≥ `0.999` for f32, ≥ `0.99`
  for a 4-bit quant is a reasonable first bar; tighten per model).
- **Max abs / rel error** (catch a single blown element a cosine hides).
- **Top-k token agreement** (k=1 and k=5) — the property that actually matters
  for decoding.
- **Per-token KL** of the softmax (distribution-level drift).

A tiny generic implementation should land in a tooling crate (e.g.
`rlx-opscope`) as `reference_compare(a: &[f32], b: &[f32]) -> ParityReport` so
both core cross-backend tests and rlx-models reference tests share one metric.
Keep it dependency-free (pure Rust). *Not added here to avoid guessing the host
crate's layout — it is a ~40-line addition; see the metric spec above.*

### Definition of done
Green reference-parity test for `qwen35` and `nemo` in rlx-models, fixtures
checked in (or fetched), thresholds documented per model, wired into the
rlx-models CI once #1 (enforced CI) exists.

---

## #8 — Kernel occupancy tuning (CUDA, profile on `msi`)

**Status today.** The `RLX_DUMP_KERNELS` hook + `tools/kernel-inspect/kinspect.py`
(runs on the msi RTX 3080 Ti rig) already report SASS / regs / occupancy /
opcode mix. Prior runs surfaced concrete candidates; tuning them is iterative
profiling that needs the GPU in the loop, so this is a candidate list + method,
not a landed change.

### Candidates (highest leverage first)
1. **`matmul_bt`** — measured **88 registers, ~33% occupancy**. Register pressure
   is capping resident warps. Hypotheses to test on rig:
   - Trim live registers via smaller register-tiling (e.g. 8×8 → 8×4 microtiles)
     or `__launch_bounds__(BLOCK, MIN_BLOCKS)` to force the compiler to spill
     less-hot values and lift occupancy.
   - Compare against a cuBLAS/CUTLASS baseline for the same shapes to bound the
     achievable ceiling before hand-tuning.
   - Measure: occupancy, achieved DRAM/L2 BW, and end-to-end tokens/s — occupancy
     is a means, not the goal; only accept a change that moves wall-clock.
2. **Attention kernel** — reported **shared-memory bound**. Hypotheses:
   - Reduce smem footprint per block (smaller KV tile, or recompute vs store) to
     raise occupancy; or the opposite — larger tiles to amortize if latency-bound.
   - Check bank conflicts in the smem layout (`kinspect` opcode/stall report).
   - Consider a flash-attention-style online-softmax tiling if not already used.

### Method (per candidate, on `msi`)
```
RLX_DUMP_KERNELS=1 <run the target graph>          # dump SASS + cubin
python tools/kernel-inspect/kinspect.py run <dump> # regs / occupancy / stalls
# form a hypothesis → edit the kernel → re-dump → A/B tokens/s, not just occupancy
```

### Definition of done
For each candidate: a before/after `kinspect` report + a wall-clock delta on a
real graph, with any regression on other shapes ruled out. No silent wins —
record shapes tested and any that regressed.

---

## Why these are separated from the rest of the hardening batch
Items #2, #3, #4, #6, #7, #9, #10 are code changes verifiable on a dev box (or a
Metal Mac). #5 and #8 are **evidence-gathering** against an external oracle
(a reference decoder; a physical GPU). Shipping their harness/plan here means the
runs are one command away on the right machine, without conflating "harness
exists" with "parity proven / kernel tuned" — which would be exactly the kind of
silent over-claim a validated pipeline must not make.
