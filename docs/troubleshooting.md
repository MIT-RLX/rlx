# Troubleshooting

Start from the symptom. Each section names the tool that answers it.

## A device is missing, or you are not sure what this host can run

```sh
cargo rlx doctor                 # every device, a check run, RLX_* settings
cargo rlx doctor --device cuda,vulkan --json
cargo rlx doctor --report        # paste-ready Markdown for a GitHub issue
cargo rlx doctor --known         # every known issue, its symptom and fix
```

Full guide: [DOCTOR.md](../DOCTOR.md).

`doctor` also matches the host against the [known issues](known-issues.md)
table (missing XRT runtime, ROCm not seeing a 780M, NVRTC not found, …) and
prints each match with its fix. Build-time failures are listed there too, by
the error text they produce.

For each device the binary was built with, `doctor` reports whether it is
ready, and if not, the backend's own reason (driver, runtime library,
toolchain, env var). Each available device runs a small `relu(a·b + c)` graph,
scored against an f64 reference. The output also shows the adapter name and
external toolchain versions (CUDA driver and NVRTC, mlir-aie), next to the
versions rlx has actually been run against. It lists every `RLX_*` variable
that is set, and flags unknown ones with the nearest registered name. It
reports only backends compiled into `cargo-rlx`, so install it with the ones
you use: `cargo install … --features cuda,vulkan` (or `metal,gpu`). The exit
code is non-zero when an available device fails its check, or when an `RLX_*`
variable is unknown. A typo is also caught at run time: the first `RLX_*` read
prints `<name> is not a variable rlx reads; did you mean …?` (silence it with
`RLX_ENV_WARN_UNKNOWN=0`).

## Too many `RLX_*` switches

Three ways to set several at once. Anything set explicitly still wins.

- **Lists:** `RLX_DISABLE=cse,sccp` sets `RLX_DISABLE_CSE` and
  `RLX_DISABLE_SCCP`. `RLX_TRACE=metal,gguf` sets the `RLX_*_TRACE` switches,
  and `RLX_DEBUG` the `RLX_*_DEBUG` ones.
- **Profiles:** `RLX_PROFILE=debug` (logging, dispatch report, NaN lint),
  `nans`, `parity` (tightest GPU numerics), `deterministic`, or `ci`. List
  several to combine them; later ones win.
- **A settings file:** `RLX_CONFIG=rlx.toml`, with `KEY = value` lines (the
  `RLX_` prefix is optional) and `[profile.NAME]` sections that `RLX_PROFILE`
  can select.

`cargo rlx doctor` shows which variables each of these set, and from where.
The full list is in [rlx-env-vars.md](rlx-env-vars.md).

## `run` panics, or `try_run` returns an `IoError`

The compiled graph checks what you feed it against what it declares:

| Message | Meaning |
|---------|---------|
| `the graph has no input `X` (did you mean `x`?)` | misspelled or stale input name |
| `input `x` got 3 values; its shape [2, 4] holds 8` | wrong buffer length for a static input |
| `param `w` got 5 values; its shape [4, 3] holds 12` | wrong length in `set_param` (or a param passed to `run`) |
| `the graph has no output named `logit` (did you mean `logits`?)` | `output_index` with a name no output has |
| `unbound inputs …, params … (never set) — they read zeros` | printed once: something the graph declares was never given |

See what the graph expects with `compiled.inputs()` / `compiled.params()`
(name, dtype, static shape), and what it returns with `compiled.outputs()`, in
`run` order. Name outputs when building the graph (`g.set_named_outputs([("logits",
y)])` or `g.name_output(0, "logits")`; ONNX imports keep the model's names)
and pick a result with `compiled.output_index("logits")` instead of a
position. Turn the unbound warning into an error with
`compiled.check_bindings(&["x", …])`. Inputs meant to read zeros — an empty
KV-cache state on the first step — are declared with
`compiled.allow_unbound(&["past_k_0"])`, which also silences the warning.

A KV cache that an `Op::KvAppend` writes into may be fed only its filled
rows: decode graphs declare a spare row for the in-place write. Any other
input is fed whole.

A param name passed to `run` is uploaded with `set_param`; unknown *param*
names in `set_param` are accepted (loading a whole checkpoint into part of it).

## It won't compile

Use `Session::try_new(device)` and `session.try_compile(graph)`; the
`CompileError` says which of these it is:

| Variant | What to do |
|---------|------------|
| `Invalid` | the graph fails IR verification; the errors name nodes |
| `DeviceUnavailable` | build with the backend's feature (`--features metal`, …), or the device is absent — the reason is included |
| `Unsupported { node, op }` | the device cannot lower that op even after rewriting; try another device, or see the dispatch report below |
| `Backend` | the backend's compile failed; the message is its panic |

For the static picture — which op kinds run natively, lower to common IR, or
are missing on each target — run `cargo rlx check graph.json --all-backends`,
or set `RLX_DISPATCH_REPORT=1` before compiling.

A debug-build panic that reads `IR verifier failed at the input graph
(already invalid before any pass …)` means the graph was invalid before any
optimization ran; `after pass `X`` means pass X broke it.

## The numbers are wrong on one device

```sh
cargo run -p rlx-check --features metal --bin cargo-rlx -- \
    debug graph.json --device metal --probe --bisect
```

runs the graph on the device and on CPU with the same inputs and reports:

- a class: `mismatch`, `non-finite`, `run-panic`, `unsupported-op`, …
- `--probe`: which single compile option (fusion off, F32, common-IR kernels,
  no constant folding, no MPSGraph) makes it pass — that is the subsystem;
- `--bisect`: the first op whose output differs, and whether it reads only
  graph inputs (then suspect upload / dtype / layout).

In Rust: `rlx_runtime::debug::debug_graph`.

## NaN or Inf appears

- `RLX_DEBUG_NANS=1` names the op that produced the first non-finite value
  and separates it from ops that merely propagate it.
- `RLX_LINT_NUMERICS=1` reports constant expressions that provably blow up,
  at compile time.
- `RLX_POISON_ARENA=1` starts CPU scratch as NaN instead of zero, so a value
  nothing wrote — an unfed input, an op that ran as a no-op — reads NaN
  instead of a plausible zero.

Details: [nan-debugging.md](nan-debugging.md).

## It is slower than expected

- `RLX_FUSION_REPORT=1` prints what fused, before and after.
- `RLX_DISPATCH_REPORT=1` shows ops falling back to common IR or host.
- Fusions that must fire on Llama/BERT-shaped blocks are pinned by
  `rlx-runtime/tests/fusions_fire_on_model_blocks.rs`; if a model's block
  looks different, compare it against those.
- Before timing anything, `just throttle`.

## A test fails only on one backend or OS

Record it in `tests/known_failures/<test>.txt` (`case device[@os] issue`)
instead of skipping it: it keeps running, and the suite tells you when it
starts passing. See [ADR 0005](adr/0005-known-failures-as-data.md) and
`rlx_ir::known_failures`.

## Running outside rlx

- IREE / MLIR toolchains: [`rlx-stablehlo`](../crates/io/rlx-stablehlo/README.md)
  exports StableHLO, with weights as arguments or as a parameter file, and
  several sequence lengths in one module.
- FPGA: [fpga-export.md](fpga-export.md).
## License

MIT OR Apache-2.0.
