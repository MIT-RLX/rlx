# rlx-tpu

Google TPU backend for RLX. Drives `libtpu.so` (the same library JAX
and PyTorch-XLA dlopen) directly from Rust — no Python.

## Status: full inference op parity with rlx-cuda / rlx-rocm; off-TPU numerical + parse validation in Docker

Compiles on any host. `is_available()` returns true iff a libtpu /
libpjrt-compatible plugin is on the loader path **and**
`Plugin_Initialize` + `Client_Create` succeed. `TpuExecutable::compile`
emits HLO and calls `PJRT_Client_Compile`; `run` executes through
`PJRT_LoadedExecutable_Execute`.

## How it works

Modern `libtpu` (≥ 0.0.7, the version JAX ≥ 0.4 ships with) is a
**PJRT plugin**. The .so exposes a single C entry point:

```c
const PJRT_Api* GetPjrtApi();
```

The returned struct is a fat function-pointer table over the full
PJRT C API (`xla/pjrt/c/pjrt_c_api.h` upstream, ~150 entries).
"libtpu directly" today means "load libtpu.so, call GetPjrtApi,
drive the returned vtable."

The compile target is **HLO**: a protobuf module
(`xla/service/hlo.proto`). PJRT exposes `Client_Compile` which takes
the serialized module + a compile-options proto and returns a
`PJRT_LoadedExecutable`. `LoadedExecutable_Execute` then runs it
with `PJRT_Buffer` inputs/outputs.

## Architecture

```
build.rs   — prost-build: generates xla.* protobuf bindings from
             vendored proto/ tree at compile time
proto/     — vendored xla_data.proto + service/hlo.proto +
             service/metrics.proto from openxla/xla main
libtpu.rs  — dlopen + PJRT_Api struct (60+ fn pointer slots) + per-call args
device.rs  — process-global PJRT client (Plugin_Initialize + Client_Create)
hlo.rs     — HloModuleProto / HloComputationProto / HloInstructionProto
             builder over prost-generated types; Shape, Literal,
             DotDimNumbers, Window, ConvDimNumbers,
             Gather/ScatterDimNumbers
unfuse.rs  — composite ops (FusedSwiGLU, FusedAttentionBlock,
             FusedTransformerLayer, LoraMatMul, If, While, rank-3
             Attention) decomposed before lowering
lower.rs   — Graph → HLO walker covering ~40 ops
backend.rs — TpuExecutable: compile, set_param, run; param-cache,
             buffer drain, executable destroy on Drop
```

## IR optimization for HLO emission

`TpuExecutable::compile` runs a small rlx-opt pipeline before
lowering. XLA does its own aggressive fusion + layout selection
post-compile, so we keep it short — only passes that strictly
shrink the emitted module or pre-compute a tier-2 fusion that we
own the lowering for:

1. `DeadCodeElimination` + `ConstantFolding` — strictly remove work.
2. `FuseResidualLN` — collapses `Add → LayerNorm` into
   `Op::FusedResidualLN`. Lowered as one HLO subgraph that we own,
   instead of asking XLA's pattern matcher to recognize the
   residual + norm sequence.
3. `FuseMatMulBiasAct` — collapses `MatMul → bias-add → activation`
   into `Op::FusedMatMulBiasAct`. XLA's bias-add fusion is
   excellent; pre-fusing here saves a redundant pass on its end and
   produces a single tier-2 HLO subgraph.
4. `LegalizeBroadcast` — HLO requires explicit `broadcast_in_dim`
   shapes (no implicit numpy-style broadcasting), so canonicalize.
5. `MarkElementwiseRegions` — collapse maximal element-wise chains
   into one `Op::ElementwiseRegion`. The lowering walks the chain
   inline (one HLO primitive sequence per region) instead of
   emitting intermediate materializations between every primitive.

We deliberately do **not** run `UnfuseElementwiseRegions` (that
would undo step 5) and we don't run `AutoMixedPrecision` by default
(XLA's bf16 pass handles dtype selection on its own; the policy
knob is still wired through `CompileOptions.policy` for callers
who want explicit control).

## Op coverage

Lowered to HLO directly:
- Input, Param, Constant
- All 12 Activation kinds (Gelu via erf, GeluApprox via tanh form,
  Silu via logistic + multiply, etc.)
- All 7 BinaryOp + 6 CmpOp + Where (HLO `select`)
- ElementwiseRegion (chain inlined as primitive HLO ops)
- MatMul (HLO `dot` with batch + contracting dim numbers),
  DotGeneral
- LayerNorm / RmsNorm / FusedResidualLN (decomposed via reduce +
  rsqrt + multiply chain)
- Attention with MaskKind::None / Causal / SlidingWindow / Custom
  (causal + sliding-window masks synthesized via `iota` + `compare`
  + `select`, no host-side mask tensor)
- Rope (split-multiply-concat)
- Reshape, Transpose, Narrow (HLO `slice`), Concat (HLO `concatenate`),
  Expand (HLO `broadcast`), Gather, Cast (HLO `convert`)
- Reduce (sum/mean/max/min/prod, with mean post-divide)
- Softmax (max + sub + exp + reduce + div)
- Cumsum (HLO `reduce-window` with full-axis prefix span)
- Conv (1D / 2D / 3D, HLO `convolution`)
- Pool (max / mean / etc., HLO `reduce-window`)
- ScatterAdd (HLO `scatter`)
- FusedMatMulBiasAct

Tier-3 ops (all lowered, full parity with rlx-cuda / rlx-rocm):

- `TopK` — HLO `sort` (paired keys + iota indices) + `slice`,
  indices reported as f32 per the rlx-ir convention.
- `GroupedMatMul` — `gather` per-token expert weights along axis 0,
  then a batched `dot_general` with M as batch axis.
- `DequantMatMul` — `convert(w_q → f32)` + per-block scale/zp tile
  (`reshape → broadcast → reshape`) + `dot_general`.
- `QMatMul` / `QConv2d` — int8 inputs promoted to s32, `subtract`
  zero-points, `dot`/`convolution`, `add` bias, `multiply` by mult,
  `round-nearest-even`, `add` out_zp, clamp to [-128, 127], `convert`
  back to s8. Real INT8 path, not fake-quant.
- `Sample` — greedy fast-path is argmax via topk-1; non-zero
  temperature uses HLO `rng` (UNIFORM) + softmax + `reduce-window`
  cumsum + `compare`/`select`/`reduce` for the inverse-CDF lookup.
  `top_k` filter via sort+threshold; `top_p < 1.0` not yet supported.
- `SelectiveScan` — Mamba SSM scan compiled to an HLO `while` loop
  carrying `(i, state[B,D,N], outputs[B,L,D], x, delta, a, b, c)`,
  with `dynamic-slice` on the inputs and `dynamic-update-slice` on
  the outputs each step.

Backward / training ops (`ReluBackward`, `LayerNormBackward*`,
`Conv2dBackward*`, `MaxPool2dBackward`, `SoftmaxCrossEntropy*`)
are explicitly out of scope — rlx-tpu is inference-only, like the
TPU-class targets in rlx-cuda / rlx-rocm.

## Install

```toml
[dependencies]
rlx = { version = "0.1", features = ["tpu"] }
```

`is_available()` returns true iff `libtpu.so` (or
`libpjrt_c_cpu.so`) is on `LIBTPU_PATH` *and* `Plugin_Initialize` +
`Client_Create` succeed.

## Testing

Four test harnesses, in increasing order of weight:

```sh
# 1. Host-agnostic — prost decode + HLO builder + lowering walker.
cargo test -p rlx-tpu                        # 2 unit + 12 hlo_decode
                                              # + 32 hlo_match + 4 smoke
                                              # = 50 tests

# 2. Off-TPU parse — runs the full cargo test inside a Docker
# container with jaxlib installed, then parses each emitted HLO
# module through xla_extension to validate proto field numbers,
# dimension layout, and opcode strings.
./rlx-tpu/docker/validate.sh                 # 50 cargo + 29 HLO
                                              # parse checks (tier-3
                                              # ops included)

# 3. Off-TPU numerical — also builds XLA's libpjrt_c_cpu.so from
# source (Bazel, ~50 min, ~50 GB transient disk, cached after first
# run) and runs tests/pjrt_roundtrip.rs against it. This proves
# numerical correctness of every HLO lowering through real
# PJRT_Client_Compile + LoadedExecutable_Execute, not just proto
# parse.
./rlx-tpu/docker/validate.sh --numerical --build-plugin
                                              # 42 cargo + 4 PJRT
                                              # roundtrip exec

# 4. On a real GCP TPU VM — set LIBTPU_PATH; the same round-trip
# suite from (3) executes against silicon.
LIBTPU_PATH=/path/to/libtpu.so cargo test -p rlx-tpu
```

## Off-TPU numerical validation: what's in `--build-plugin`

`docker/Dockerfile.xla-cpu` is a multi-stage Bazel build of
`//xla/pjrt/c:pjrt_c_api_cpu_plugin.so` from openxla/xla main.
Stage 1 (~50 min on Apple Silicon, ~50 GB transient disk) compiles
LLVM + XLA + the CPU PJRT plugin; stage 2 keeps just the .so. The
result is the same shape as `libtpu.so` and dlopen-able through
`LIBTPU_PATH`. Once built, the layer is cached — re-runs of the
numerical harness skip straight to cargo test.

`docker/Dockerfile.numerical` extends this with Rust + cargo and
runs `cargo test -p rlx-tpu --release` so the whole flow exercises
real PJRT compile + execute. The 4 roundtrip tests
(`add_two_vectors`, `matmul_2d`, `activations_relu_sigmoid`,
`layernorm_minus1`) compare against in-test references with strict
tolerances.

### PJRT C-API ABI gotchas (lessons from the first end-to-end run)

The PJRT C API is forward-compatible by struct-size prefix: the
plugin checks `args->struct_size` against the offset of the last
known field (`PJRT_DEFINE_STRUCT_TRAITS`). When wiring up Rust FFI
mirrors of `xla/pjrt/c/pjrt_c_api.h`, getting any of these wrong
manifests as an opaque SIGSEGV during `Client_Create` or first
real call, sometimes with a `wrapper_impl.cc:NNN` log line just
before the crash. Things that bit us:

- **`PJRT_Api_Version` is itself a struct-size-prefixed PJRT struct,
  not a bare `(int major, int minor)` pair.** It has 4 fields:
  `struct_size`, `extension_start`, `major_version`, `minor_version`
  → 24 bytes. Declaring it as 8 bytes shifts every fn-pointer slot
  in `PJRT_Api` by 16 bytes, so calling `plugin_initialize` actually
  invokes `PJRT_Error_Message` and segfaults reading garbage.
- **`PJRT_Client_Create_Args` grew `kv_try_get_callback` +
  `kv_try_get_user_arg`** in PJRT API ≥ 0.59 (88 bytes total).
- **`PJRT_ExecuteOptions` grew `call_location`, `num_tasks`,
  `task_ids`, `incarnation_ids`, `multi_slice_config`** (120 bytes
  total). The plugin's struct-size check fails if you stop at the
  older 80-byte layout.
- **Empty `compile_options` is rejected.** XLA's default leaves
  `replica_count = 0`, which trips a CHECK in
  `xla::DeviceAssignment::DeviceAssignment()`. We pass a 6-byte
  hand-encoded `CompileOptionsProto` carrying
  `executable_build_options { num_replicas: 1, num_partitions: 1 }`.

If you see `Unexpected PJRT_<X>_Args size: expected N, got M`, the
fix is to compare your Rust struct against the upstream definition
in `xla/pjrt/c/pjrt_c_api.h` and add the missing trailing fields.

## What we are deliberately not doing

- No JAX / PyTorch-XLA shim. The whole point of "libtpu-based" is to
  skip the Python tax.
- No StableHLO MLIR text round-tripping. PJRT accepts HLO protobuf
  directly, and avoiding the MLIR linkage keeps the build slim and
  cuts a 100+ MB compile dep.
- No multi-host SPMD. TPU pods need GSPMD + collective permutations;
  single-chip first, pods later if anyone asks.

## License

GPL-3.0-only.
