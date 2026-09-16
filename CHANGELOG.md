# Changelog

All notable changes to RLX are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and the project
tracks [Semantic Versioning](https://semver.org/) — while pre-1.0, any
`0.x → 0.(x+1)` bump may carry breaking changes, per the `0.x` SemVer convention.

**Adding an entry.** Put user-visible changes under `## [Unreleased]`, newest
first, in the category that fits. Use these categories, in this order (omit any
that are empty):

- **Added** — new features, ops, or backends.
- **Changed** — changes in existing behavior.
- **Deprecated** — features headed for removal.
- **Removed** — features dropped in this release.
- **Fixed** — bug fixes.
- **Security** — vulnerability fixes (coordinate via [`SECURITY.md`](SECURITY.md) first).
- **Performance** — measured speedups on a backend's hot path; cite the before/after.

Version headings are `## [x.y.z] — YYYY-MM-DD` (ISO date, em-dash separator). On
release, rename `[Unreleased]` to the new version and add a fresh empty
`[Unreleased]` above it, then update the compare links at the bottom of the file.

## [Unreleased]

## [0.2.16] — 2026-09-15

### Added

- **`rlx-peft` — parameter-efficient adaptation as graph operations.** LoRA,
  IA3, AdaLoRA, DoRA and OFT reproducing HuggingFace PEFT's definitions, which
  is what published results are measured under. Both halves are there: host
  arithmetic (`lora_delta`, `ia3_apply`, `adalora_delta`, `dora_weight`,
  `oft_weight`, plus `cayley_orthogonal` for OFT's block rotation) and the same
  updates as `rlx_ir::Graph` builders (`lora_delta_graph`, `ia3_graph`,
  `adalora_delta_graph`, `dora_graph`), so they run on any compiled-in backend.
  Two initialisation conventions are reproduced rather than chosen — `B` starts
  at zero and IA3's `l` starts at one — because a nonzero `B` perturbs a
  pretrained network before training begins, which is the usual source of "PEFT
  hurt my model". `ParamBudget` / `lora_param_count` / `lora_is_economical`
  supply the denominator every PEFT claim needs.
- **Both new crates are reachable from the umbrella** as `rlx::peft` and
  `rlx::rng`, behind the `peft` / `rng` features. `rlx-peft` is pulled with
  `default-features = false` so enabling `peft` does not quietly select a
  backend — its own `cpu` default would have forwarded `rlx-runtime/cpu` to a
  consumer who never asked for one; `rlx/cpu` supplies the same cpu-enabled
  runtime when you do ask.
- **`rlx-rng` — numpy and PyTorch random streams, bit-exact.** Porting a Python
  reference usually stops at the arithmetic and then fails on anything seeded: a
  random projection, a subsample, a shuffled split need the *same stream*, not
  merely the same distribution. `numpy::RandomState` (MT19937 + legacy polar
  Gaussian), `numpy::SeedSequence` + `numpy::Generator` (PCG64 + Lemire
  `integers`), `torch::RandomState` / `torch::randperm`, and the keyed
  `deterministic_hash` / `hashing_projection` pair. Each needed a detail that is
  invisible until you diff the streams — numpy's legacy Gaussian *caches* `f·x1`,
  PCG64's `set_seed` reads word 0 as the **high** half, `Generator.integers`
  dispatches to a 32-bit Lemire path that consumes the stream at a different
  rate than the 64-bit one, and torch's `randperm` walks Fisher–Yates *forward*.
  Every one of those produces a perfectly reasonable random sequence when wrong.
  No dependencies, not even on `rlx-ir`.

- **Non-desktop nodes: iOS, Android and pre-synthesized accelerators can join a
  mesh, for training as well as inference.** The transport layer was already
  portable — `rlx-driver` has no platform gating and no default C deps — but the
  node driver lived in an *example*, and an example cannot be linked into an
  Android `.so` or an iOS framework. `rlx_runtime::dist::node` is that driver as
  a library: `NodeConfig` is fully programmatic (a phone has no meaningful
  `RANK`/`PEERS` to read), `NodeControl` bounds the serving loop, and
  `serve_worker` / `serve_trainer` cover both halves of what a rank is for. New
  `rlx-ffi` crate exposes it over a C ABI; `ios/` adds an xcframework build,
  Swift wrapper and demo app, and the Android JNI bridge gains node entry points
  plus the `MulticastLock` that UDP discovery silently needs. Validated on an
  iOS simulator and an Android emulator, including heterogeneous data-parallel
  training (Android on wgpu, host on CPU). `just check-nodes` cross-compiles the
  stack so this cannot regress unnoticed.
- **FPGA joins as a pre-synthesized fixed-function rank.** An FPGA cannot host a
  runtime — synthesis takes minutes — so `rlx-fpga` is a code generator with no
  `Backend` impl. A board instead advertises the one datapath its bitstream
  implements and the coordinator only ever assigns it that stage. Two guards,
  because a wrong feed clocked into fabric produces numbers rather than an
  error: `NodeCaps::accepts` refuses a mismatched *or untagged* `StageSpec` at
  placement, and the serving loop checks every activation's length before
  touching the board. `NodeCaps::can_train` refuses a training job outright — a
  bitstream has no backward pass, and a rank that never reduces would stall
  every rank that does. `LoopbackFixedFunction` makes the whole path testable
  with no hardware.
- `Op::arity` (`Arity::{Exact, Range, AtLeast}`) replaces a bare `usize` where
  `0` had to mean a leaf, an optional operand, *and* "variadic, don't check" —
  which exempted every variadic op from arity checking entirely. `rlx-ir`'s
  `test-support` feature adds `sample_ops`: one constructible `Op` per `OpKind`
  behind a compiler-enforced exhaustive match, so gates keyed on the op set can
  cover all of it. `just check-ops` runs them, including a static check of the
  declared arity against what backends actually index.

- **The quantized corpus family is validated against an independent authority
  for the first time** — 18 → 19 cases, on cpu, metal, mlx, wgpu, vulkan and
  (on the CUDA rig) cuda. Two things were in the way, and neither was the one the
  skip claimed:

  * **A byte channel.** `oracle::validate_with_packed` feeds `U8` params as raw
    bytes, mirroring `set_param_typed`. A quantized weight is not expressible as
    `&[f32]`, and `set_param` would write four bytes per element into a slot
    sized for one — a silent arena overflow, not a type error.
  * **A decoder that is not `rlx-gguf`.** `oracle::gguf_q8_0_decode` reads
    `block_q8_0` from the format description (f16 scale + 32 int8), with its own
    `f16_to_f64`. rlx-cpu's dequant calls into `rlx-gguf`, so scoring against
    `rlx-gguf` would have been a self-comparison wearing an oracle's clothes.

  Q8_0 first because *any* byte string is a valid block, so a fixture needs no
  encoder. The K-quants pack scales into shared nibble planes where arbitrary
  bytes are not necessarily meaningful; they remain unvalidated and say so.
  Passing also independently confirms the `[n, k]` weight orientation — the one
  rlx-rocm got backwards behind an `n = 1` test.

- **`rlx-corpus`'s device arm is precision-aware.** CUDA's default GEMM is TF32
  on sm_80+ (10 mantissa bits, not 23), so an f32-derived bound fails it by
  construction: measured 2.52e-3 and 4.50e-3 against 2.0e-4 / 4.0e-4 bounds on
  an RTX 3080 Ti, with `RLX_CUDA_NO_TF32=1` restoring 19/19 at the full bound.
  The gate now widens the bound 16x for TF32 GEMM graphs only, **prints that it
  did**, and leaves every other case at f32 strictness. 16x is empirical and
  labelled as such — the raw epsilon ratio is ~8000x, which would accept
  anything.

- **`Op::FftQ` — fixed-point FFT/IFFT, with an explicit scaling policy.** `I32`
  data in the same 2N-block layout `Op::Fft` uses for `F32`, radix-2, Q30
  twiddles, `i64` products. `Graph::fft_q` builds it; rlx-cpu executes it; the
  kernel is `rlx_ir::fft::fft1d_q32_block`, shared rather than backend-local.

  It is a separate op rather than a dtype of `Op::Fft` because it needs a
  parameter the floating-point transform does not — how to keep the datapath in
  range — and that choice costs precision, so it is in the type:

  | [`FftQScale`] | bits lost | headroom needed | overflow |
  |---|---|---|---|
  | `None` | 0 | `log2(n)` | wraps |
  | `Saturating` | 0 | `log2(n)` | clips |
  | `EveryOther` | `log2(n)/2` | `log2(n)/2` | — |
  | `PerStage` | `log2(n)` | 0 | cannot |

  `bits_lost` and `headroom_needed` are methods, so a caller can check the
  trade against its own signal rather than infer it. The textbook
  halve-every-stage default costs ten bits on a 1024-point transform, which is
  often more than the caller has to spend: full-scale i16 audio needs none of
  it and lands within 3e-5 of a direct DFT.

  Accuracy is **absolute** — every butterfly rounds to an integer — so relative
  accuracy is whatever the input's magnitude makes of it. That is documented on
  the kernel, and the tests use full-scale inputs because that is the intended
  case.

  `fft_meta` now accepts `I32`, which it previously rejected; the 2N-block
  layout it measures is dtype-agnostic.

  **Metal** carries the op through the same host fallback it already uses for
  the f64/C64 `Op::Fft` variants — flush, run the shared kernel against the
  unified-memory arena, restart the command buffer — and is **bit-identical to
  CPU** across all four scaling policies, forward and inverse
  (`metal_fft_q_parity`). Metal's integer-widening pass now exempts `Op::FftQ`
  operands, alongside the host `Op::Custom` kernels that already needed true
  integer widths.


  **CUDA and ROCm carry it** through the same host fallback, and are
  bit-identical to CPU (`cuda_fft_q_ingraph`, `rocm_fft_q_ingraph`, verified on
  an RTX 3080 Ti and a gfx1103). These arenas are f32-*valued* — an integer
  tensor is stored as the float with the same value — so the adapter
  ([`rlx_gpu_host::run_fft1d_q_valued`]) **converts** at the boundary rather
  than reinterpreting. Reading those slots as raw i32 instead is what made a
  first attempt return `2147483647` / `-2147483648`; being byte-addressable for
  staging is not the same as having typed i32 storage.

  The conversion is exact while every value stays inside f32's exact-integer
  range (2^24). Past that it **panics** rather than rounding, on both input and
  result: a fixed-point transform that quietly drops low bits is worse than one
  that stops. The bound is easy to respect — with `FftQScale::PerStage` the
  result never exceeds the input, and unscaled a length-`n` transform of
  `|x| < 2^24 / n` is always safe. A 1024-point frame of i16 audio needs
  `PerStage` or `EveryOther`.

  **wgpu carries it too**, through the same converting adapter
  (`wgpu_fft_q_ingraph`). It additionally needed registering in
  `step_runs_on_host` — without that the runtime never flushed pending GPU work
  before the host step, so the transform read an unsynced arena and returned all
  zeros. Silent zeros, not garbage: a step that is invisible to the host-sync
  classifier fails quietly, which is worth knowing when adding any host fallback
  to this backend.

  **The fixed-point batch now parallelizes**, which it previously did not — it
  was the only FFT variant still running single-threaded, and that alone (not
  its arithmetic) was what made it look ~4x slower than f32 at 256x1024. Forcing
  the f32 path serial with `RLX_FFT_CPU_PARALLEL=0` closed the gap to 0.95x,
  which is how the cause was identified.

  Rows of a block transform touch disjoint slices and share only read-only
  twiddles, so the fan-out is **bit-identical** to the serial loop, not an
  accuracy trade — `fft_q_parallel_is_exact` pins that across 288 combinations
  of shape, scaling policy, direction and normalisation, plus the error paths.
  The seam is [`rlx_ir::fft::FftQ32Plan`]: `rlx-ir` takes no rayon dependency,
  so it exposes the shared twiddles and a `run_row`, and
  [`rlx_cpu::thunk::fft1d_q32_block_parallel`] does the fan-out behind the same
  measured gate the f32 path uses (`outer >= 8 && outer*n >= 2^15`). Splitting a
  block into per-row `fft1d_q32_block` calls would have rebuilt the twiddle
  tables once per row.

  The f32<->i32 conversion in the host fallback moved to
  [`rlx_cpu::thunk::fft1d_q32_f32_valued`] and parallelizes with it — at
  256x1024 that is over a million elements each way, not a rounding error next
  to the transform. It sits in `rlx-cpu` because `rlx-gpu-host` owns staging,
  not compute.

  Every backend shares the one kernel, so all of them gained (best-of-three,
  net of the `Cast` nodes, microseconds):

  | 256x1024 | before | after | |
  |---|---|---|---|
  | CPU (M-series) | 1963 | **458** | 4.3x |
  | Metal | 3285 | **1754** | 1.9x |
  | wgpu (Apple) | 3809 | **2750** | 1.4x |
  | CPU (Ryzen) | 3815 | **573** | 6.7x |
  | ROCm gfx1103 | 6012 | **1624** | 3.7x |
  | CUDA RTX 3080 Ti | 6172 | **924** | 6.7x |

  On CPU the fixed-point transform is now *faster* than the f32 one at batch
  (0.55-0.81x); single-frame is unchanged at ~1.05-1.27x, correctly below the
  parallel threshold. Reproduce with `rlx-cpu --example fft_q_kernel` (kernels
  alone) and `rlx --example fft_q_speed` (graph + devices).

  Two measurement notes, since both nearly produced wrong conclusions: medians
  on a loaded rig yielded a *negative* net cost, so the benchmarks report
  min-of-N; and the CUDA rig shows a sporadic ~2.2 ms per-session floor that
  makes its small-size rows unreadable, so only its large-batch rows are quoted.

  So the op is CPU, Metal, CUDA, ROCm and wgpu. Metal reaches exactness differently —
  its compile keeps host-kernel operands at their true integer width
  (`custom_operands`, which `Op::FftQ` now joins), so it carries raw i32 and has
  no 2^24 bound at all.


- **`rlx_ir::audio::MelBands` — banded mel filterbank, and `Op::LogMel` now uses
  it.** A mel filter is a triangle, so a dense `[n_mels, n_bins]` bank is mostly
  zeros: Whisper's 80x201 carries about 1,120 useful weights of 16,080, and the
  dense product multiplied through all of them. Banding the rows and hoisting
  the per-frame `Vec` allocations makes `log_mel_block_f32` **10.4x faster**
  (14.05 ms -> 1.35 ms for 30 s of audio at 80 mels).

  It is an arithmetic identity, not an approximation: only exact `+0.0` weights
  at the two *ends* of a row are dropped, so the surviving terms are summed in
  the same order. A leading run of zeros leaves the accumulator at `+0.0`
  whatever the sign of the input, and a trailing run adds `±0.0` to a value that
  is already final. Interior zeros are kept, so the saving does not depend on
  the bank being triangular or its weights non-negative —
  `banded_mel_is_bit_identical_to_dense` covers negative weights, interior
  zeros and an all-zero row.

  `MelBands` is public so model crates can share the one implementation rather
  than each carrying its own dense loop.


- **`Op::Pool` reaches the `Tensor` and `Graph` APIs.** `Graph::pool2d` /
  `max_pool2d` / `avg_pool2d`, `GraphExt::pool2d` and `Tensor::pool2d` /
  `max_pool2d` / `avg_pool2d`, with `shape::pool2d_output_shape`. The op and its
  VJP already existed; there was simply no builder, so any graph with a pooling
  layer had to hand-assemble the node and its output shape. Forward and backward
  are covered by `rlx-tensor/tests/pool.rs`, including that max-pool routes
  gradient only to each window's argmax and accumulates across overlapping
  windows.

- **`Func::train_step_all_at_on_qat_with`** — the quantization-aware step, with
  the parameter's name and shape passed to the quantizer. Block formats need
  this: a per-block scale is only usable if its blocks run along the axis a dot
  product contracts over, and a flat `&mut [f32]` does not say which axis that
  is. `train_step_all_at_on_qat` is now a thin wrapper over it.

- **`rlx-fpga`: sequential-engine export target (`rlx_fpga::seq`).** Lowers an
  rlx-ir graph to a single microcoded datapath — one MAC, one activation LUT,
  one descriptor ROM — instead of a module per layer. Where the existing
  `Model::from_graph` path covers feed-forward INT8 classifiers, this covers any
  graph whose stages reduce to "bias, then a dot product over an address
  pattern, then requantise and activate", **including recurrent ones**: LSTM
  state is a tensor in the activation RAM and the engine updates it in place via
  `SeqConfig::with_carry`.

  The lowering does three things worth calling out:

  - **`Reshape` and `Transpose` never move data.** They are pushed back into the
    producing stage's destination strides, so the flatten in front of an LSTM
    costs zero cycles.
  - **`Concat` drives placement.** Its operands are made adjacent, which is what
    lets a concatenated tap run (`x‖h` for a gate projection) be one dot
    product. When two concats want the same operands in opposite orders, the
    contraction rows of the consuming matmul are permuted rather than copying
    activations.
  - **Padding is free.** A `Concat` of zero-valued params becomes a reserved
    range in a zero-initialised RAM, so the address generator needs no bounds
    check.

  Addresses come from running accumulators, so the address path is adders only.
  Emitted output is `tv_core.sv` and `tv_lut.sv` (fixed) plus generated defs,
  descriptor ROM and `.mem` images, so re-exporting a retrained network touches
  no hand-written Verilog.

  Validated end to end on TEN-VAD: 250 frames, **0 mismatches** against the Rust
  fixed-point reference under Icarus Verilog; `yosys synth_ecp5` gives 4,067
  LUT4, 891 FF, 74 × 18 kbit BRAM and 13 DSP.

  `rlx_fpga::seq::quantise_pow2` is public so callers can generate matching
  integer weights for a software reference — a power-of-two scale must round
  *up* in fractional bits or the largest magnitudes clip, which costs three
  orders of magnitude of accuracy.

- **`rlx-runtime/tests/lstm_carry_writeback.rs`** — pins `Op::Lstm { carry }`'s
  in-place `hn`/`cn` write-back on every backend. Stepping a `seq = 1` graph `N`
  times must reproduce one `seq = N` run; a backend that only *seeds* the state
  diverges from step 1. The existing `gru_carry_native` test could not catch
  this class of bug — it compares a single run's output, and the write-back is
  only observable across two runs.

- **`rlx_ir::env::skip_unless_device`** — one implementation of the
  "skip, or fail under `RLX_REQUIRE_DEVICE`" decision, taking `compiled` and
  `available` as plain booleans so it carries no `Device` type and no backend
  dependency. It lives in `rlx-ir` because the backend crates need the same
  discipline and cannot depend on `rlx-runtime` — it depends on them.
  `rlx-runtime/tests/common/mod.rs` now delegates to it rather than keeping a
  second copy; `mask_strides_for_shape` and `is_static_weight_tensor` both
  drifted that way already.

- **`tests/device_present.rs`, in `rlx-runtime` and in each of rlx-metal /
  wgpu / vulkan / mlx / cuda / rocm** — a floor under the whole problem: under
  `RLX_REQUIRE_DEVICE=1` the **binary** fails if a backend it was built with
  cannot be instantiated, no matter how its individual tests skip. Verified to
  bite: `RLX_REQUIRE_DEVICE=1 cargo test -p rlx-rocm --test device_present`
  fails on a host with no AMD GPU and skips cleanly without the flag.

- **`tests/require_device_coverage.rs`** — counts the test files that still gate
  on availability without the shared helper and **pins the number so it can only
  go down**. A new test written with a raw `if !is_available(..) { return }` now
  fails this gate rather than quietly reporting `ok` on a rig with no device.

  186 → 45, migrating ~640 call sites across 141 files in rlx-metal, wgpu,
  vulkan, mlx, cuda, rocm, coreml, tpu and rlx-runtime's own tests. Every
  rewrite was compile-checked; metal, wgpu, vulkan, mlx and rlx-runtime were
  also run.

  Two things the gate caught about itself, which is the argument for having it:

  * it **scanned a hardcoded list of backend directories** and missed seven,
    including rlx-tpu's three unmigrated files. Directories are now discovered
    under `crates/backends/*/tests`, so a new backend is covered the day it
    appears. Fixing that pushed the count 47 → 50; the ratchet refused the
    raise, and the files were migrated instead.
  * it treated `assert!(is_available())` — a positive assertion, the opposite of
    a skip — as unmigrated, putting two rlx-oneapi files on a backlog with no
    work in it. It now requires a *negated* call.

  The count reached **0**, so the pin is no longer a backlog — it is an absolute
  rule, and a raw `if !is_available(..) { return }` anywhere in the test tree
  now fails outright. Getting the last 45 there needed three things a pattern
  could not do on its own:

  * `common::skip_unless(dev)`, which derives the label from `{dev:?}`. Having
    no literal to name was the *only* thing keeping `if !is_available(dev)` —
    the single most common shape in the tree — on the raw form.
  * two compound conditions rewritten by hand, where the short-circuit ordering
    carries meaning: in `nontrailing_reduce_parity.rs`, `dev == Device::Cpu`
    means no GPU backend was compiled at all, so it must be tested *before* the
    assert — an absent backend is not a broken one.
  * three uses that are not skips excluded from the gate entirely:
    `egpu_seam.rs` and `rlx-tpu/basic.rs` assert a device is *absent*, and
    `graph_devices_parity.rs` searches *for* an unavailable device to exercise
    the fallback path on every host. Migrating any of them would have deleted
    what the test checks. `gates_on_availability` now requires a negated call
    **in an `if` condition**, so assertions and closure predicates do not count.

- **`just lint-features`** — clippy over the feature combinations
  `--all-targets` cannot reach, host-appropriate and probing CUDA/ROCm before
  linting them. `just lint` depends on it. It found three real defects on its
  first run, all invisible to the default-feature gate (see *Fixed*).

- **Oracle arms for `Transpose`, `Expand`, `Narrow` and `Concat`** in
  `rlx_corpus::oracle`. The evaluator covered nine op kinds, so the entire
  `structural` family scored **0/3 on every device** — the ceiling was the
  oracle, not the backends, and an unvalidated case is indistinguishable from an
  absent one in a coverage number. Independently validated cases went 15 → 18 of
  23 on every device at once. Reindexing is also where this tree's real defects
  live (a broadcast axis given a non-zero stride, a window walking off an
  extent), so the arms state the stride rules explicitly.

- **`rlx-corpus` device arm teeth test** (`a_wrong_device_result_is_caught`).
  The gate passing proves nothing on its own — a harness that compiled every
  case, discarded the output and tallied `validated` would look identical. This
  runs a real case on every device, perturbs one element of what the device
  returned, and requires the same `tolerance_for` + `validate` pair to reject
  it. Confirms 5 devices here.

- **`rlx-corpus` gained a device arm** (`tests/device_gate.rs`, `just
  check-corpus-device`, wired into `just test-gpu`). The crate's gate stack is
  device-free by design — structure, representation, memory plan — but every
  defect in this release was numerical and per-backend: the batch-broadcast
  attention mask, the `KvAppend` row stride, wide-hidden LSTM on Apple GPUs, the
  cross-stripe read of zeros. None is visible to a static gate.

  Two things distinguish it from the GPU parity tests already in tree, which is
  the reason to add it rather than point at them. It scores against
  `rlx_corpus::oracle`'s **independent f64 evaluator**, not against rlx's own
  CPU path — a defect both share is invisible to backend-vs-backend parity, and
  `rms_norm_backward` carried an extra `1/r` in all seven implementations at
  once. And it rolls up **per family**, which is the question a compiler change
  actually asks and which a suite organized by owning crate cannot answer.

  Backends are opt-in per host (`--features apple` / `gpu` / `cuda` / …) rather
  than default-on, because cargo unifies features across a workspace build and
  an Apple-only backend enabled here would be enabled for every crate on Linux.
  Verified on this host across cpu, metal, mlx, wgpu and vulkan × 9 families.

- **`rlx_cpu::NO_THUNK_ARM`** — the OpKinds rlx-cpu claims but has no thunk arm
  for, where the catch-all is `Thunk::Nop`. It is not documentation:
  `expand_cpu_nop_fused` dispatches on it, so it cannot drift from what actually
  gets expanded, and it is the input to the new gate below.

- **`rlx_vulkan::backend::routes_to_cpu_host`** — Vulkan's host-fallback routing
  decision, readable from outside the crate without a device.

- **`Op::KvAppend` is now native on CPU and wgpu too.** It was native on Metal,
  CUDA and ROCm only; everything else fell back to `lower_kv_append`, i.e. back
  to the O(context) `concat` the op exists to replace.
  - CPU: a raw `memcpy` sized in BYTES rather than an element loop, so it needs
    no per-dtype variant (an append moves data without touching its values) and
    reduces to a single contiguous copy for the usual `[1, seq, heads*dim]`
    cache.
  - wgpu and Vulkan: one `Step::BufferCopy` / `Step::ActCopy`, no new shader on
    either. That is only sufficient because the shape guard below guarantees the
    prefix is contiguous — otherwise both would need a strided kernel. Fixing
    the semantics first is what made the extra backends nearly free.

  Six of seven backends are now native (CPU, Metal, CUDA, ROCm, wgpu, Vulkan).
  **MLX still falls back** deliberately: it is a functional array API where an
  in-place aliased write does not express naturally, and Metal already covers
  the same hardware natively.

- **Device-resident KV row feed on wgpu** (`register_kv_row_feed` /
  `feed_kv_row`, `ExecutableCapabilities::kv_resident`). It existed on Metal,
  CUDA, ROCm and Vulkan only.

  This is the half that actually removes the O(context) decode cost.
  `Op::KvAppend` alone does not: the `concat` it replaces copies the cache
  on-GPU, but the runtime still re-uploads the past KV as a graph input every
  step, so the cost merely moves. Residency keeps the cache on the device and
  folds each new row in without a host round trip.

  Backed by a new shard-aware `Arena::copy_within_device`. **WebGPU forbids
  `copy_buffer_to_buffer` where source and destination are the same buffer**,
  and the arena is one buffer, so the copy bounces through a cached staging
  buffer — two copies on a few-KiB row rather than one, and grown on demand
  instead of allocated per feed.

- **`RLX_STATIC_WEIGHT_PACK`** — one switch, honoured by Metal, wgpu, CUDA, ROCm
  and Vulkan, for the step-invariant weight-pack skip described under
  *Performance*. Default on; `=0` restores the recompute **and** the memory it
  costs. Replaces the Metal-only `RLX_QWEN3_BAKE_WEIGHTS`, whose name never
  matched its scope (it applies to any graph the weight-concat fusions touch).
- **`RLX_REQUIRE_DEVICE`** — turns "no GPU present" in a backend test from a
  silent skip into a failure. A skipped test still reports `ok`, which is
  indistinguishable from a real pass in the summary line; on a rig whose card is
  unavailable a whole suite can go green while executing nothing.
- **`CompiledGraph::capabilities()`** — what the compiled executable itself
  reports, as opposed to `advisory_capabilities(Device)`, which answers from a
  device enum before anything is compiled. The table was the only public way to
  ask, and being hand-maintained it had drifted from four of the seven backends
  (see *Fixed*).
- **`rlx_compile::memory::is_static_weight_tensor`** is now public. Backends need
  the *same* predicate the planner used to decide pinning; `rlx-wgpu` carried a
  byte-identical private copy, which is a silent-divergence hazard.

- **Every crate now tells docs.rs which features to build.** 63 of the 72
  workspace members had no `[package.metadata.docs.rs]`. Because this workspace
  gates essentially all public surface behind features — the prelude is empty by
  design — docs.rs was building most crates with no features and rendering
  near-empty pages. Each crate now declares a host-buildable feature set (no
  vendor SDK, no network fetch, no device) plus `rustdoc-args = ["--cfg",
  "docsrs"]`; all 63 sets were verified to build.
- **`just check-doc-links`** — a Markdown link gate, wired into `just ci`. Every
  relative link in a tracked `*.md` must resolve, and every same-file `#anchor`
  must match a real heading under GitHub's slug rules. Relative links rot
  silently: nothing renders an error, GitHub 404s only on a click, and
  `cargo doc` never reads a README (see *Fixed*).
- **`rlx-hub`, `rlx-lbm`, `rlx-corpus`, `rlx-megakernel`, `rlxsl` and
  `rlx-onnx-proto` gained READMEs**, so the "every crate carries its own
  `README.md`" claim in the workspace README is now true of all 72.
- **`lint-features` now covers `rlx-cpu`'s AMX/SME/BNNS combination.** Those
  features are opt-in and default-OFF, so nothing else in the gate compiled
  them — which is how a non-compiling `amx-bnns` shipped (see *Fixed*).

### Deprecated

- **`RLX_QWEN3_BAKE_WEIGHTS`** — use `RLX_STATIC_WEIGHT_PACK`. Still honoured on
  Metal.

### Removed

- **Committed build artifacts and stray blobs are no longer tracked** (37 files,
  4.8 MiB):
  `tinystories.rlxts` at the repo root, the Verilator output under
  `rlx-fpga/hw/tinyconv_mnist/obj_dir/` (`.o`/`.a`/`.d` plus a `sim_bench`
  binary), the orphaned half-downloaded MNIST copy in `data/` (its
  `train-images-idx3-ubyte` was committed as a 0-byte file), and the stale
  `rlx-cortexm/trainer/rlx-cortexm/` shadow tree. `.gitignore` now covers each,
  and the trainer no longer recreates the shadow tree (see *Fixed*).
- **`docs/pr-split-wip.md`** — an ephemeral working note planning a PR split
  whose three parts (`Interpolate3d`, `QuantScheme::Mlx*`, `rlx-dduf`) all
  landed before 0.2.14.

### Fixed

- **Metal picked the padded GEMM on shapes that did not need padding.** At
  `m < 32` the sgemm cascade tested `k >= 256 && n >= 256 -> SimdPadded` *before*
  `aligned_8 -> Simd`, so every fully 8-aligned large projection with a small
  batch paid for padding it did not need. The `m >= 32` arm, for identical
  eligibility, already preferred `Simd` — the small-`m` arm's own comment
  claimed the two matched. The rule dated from when that arm's `else` was
  `Naive`, so its job was escaping `Naive`, not beating `Simd`; once the arm
  gained a general fallthrough it was both wrong and redundant
  (`k >= 256 && n >= 256 && k % 8 == 0` implies the `k % 8 == 0 && n >= 8` rule
  below it). Measured on M4 Pro with the arms alternated inside each timing
  round: `Simd` wins all six affected shapes in both of two runs — 1.38-1.51x at
  `m = 8`, 1.03-1.12x at `m = 16`, 1.22x at `m = 24`. Affects `m` in
  `{8, 16, 24}` with 8-aligned `k, n >= 256`; non-8-aligned small-`m` still
  routes to `SimdPadded` unchanged. Guard:
  `cost::tests::the_cascade_never_picks_a_variant_its_own_cost_model_beats`.
- **`rlx --features splat` did not compile, and the seam it should have exposed
  was unreachable.** `rlx-splat` now lives in a separate repo that depends on
  rlx, so the dependency points inward — but the umbrella still had three
  `#[cfg(feature = "splat")]` blocks naming the `rlx_splat` crate, which made
  the feature fail with `E0432`/`E0433`. `splat` is now a real feature over the
  in-tree half: `rlx::splat` gathers rlx-ir's opcodes and packed-buffer layout
  helpers together with rlx-cpu's executor registry, so a downstream renderer
  has one import path. `rlx-cpu`'s six executor type aliases
  (`RenderExec`, `HostBackwardExec`, …) are public for the same reason —
  `register_splat_executors` was `pub` while none of its parameter types could
  be named, so a renderer could call it but could not hold an executor in a
  struct or return one from a builder.

- **A zero-element tensor made the wgpu backend panic, then hang.** Two bugs, both
  reached by any graph carrying a tensor with a zero-length dimension — LuxTTS's
  flow decoder builds an `Expand [0,1,512]`, so the model was unrunnable on wgpu.

  `rlx_compile::memory` only records a buffer when its slot size is non-zero, so
  such a tensor gets no assignment at all and every `Arena::offset` lookup fired
  `no offset for node NodeId(n) (not in arena or weight buffer)`.
  `compile_static_inner` now gives each one an explicit empty slot — zero bytes to
  read or write, so any offset is as correct as any other — and `arena_span_bytes`
  skips zero-length ids so they never anchor a bind window.

  With the node compiling, `WgpuExecutable::run_inner` then span forever. Roughly 79
  match arms guard a degenerate dispatch with `if <scaled extent> == 0 { continue; }`,
  but the only `step_i += 1` for a dispatched step is at the bottom of that loop, so
  each one re-entered on the same step indefinitely. A skip must also consume the
  step's bind group, or every later step binds the wrong one — the `static_once` skip
  a few lines above already does exactly this, and the guards now follow it. New test
  `tests/empty_tensor_arena_slot.rs`.


- **Nearest `Resize` only lowered along the width axis, so a height upsample
  imported as zeros.** `lower_resize`'s nearest path handled a 2×2 upsample and a
  width-only resize gated on `h_in == h_out == 1`; anything else fell through to
  the zero-filled `__resize__/…` stub. KittenTTS's vocoder `f0_upsamp` is
  `[1,1,1,F] → [1,1,300,F]` — nearest ×300 on **height** — so its NSF f0 source
  was zeros on every backend, and the model ran and returned wrong audio. Added
  the rank-4 case of the identity the NCDHW path already uses: for integral
  scales, `[N,C,H,1,W,1]` broadcast to `[N,C,H,kh,W,kw]` has exactly the
  row-major order of `[N,C,H·kh,W·kw]`, which is ONNX's asymmetric + floor rule,
  and both reshapes are free. Identity resizes still take the passthrough path.
  New test `tests/resize_nearest_height.rs`.

- **`compile_compare` asserted on rank without naming the node.** An operand
  whose rank exceeds the output's tripped a bare `broadcast: input rank 4 >
  output rank 3` inside `broadcast_strides`, with no indication of which node in
  a 24 k-node graph was inconsistent. The assert now names the comparison and
  prints all three shapes.

- `Op::If` was declared as taking exactly one operand, with a note that captures
  were "handled separately". They are not: `sccp` builds `If` with a capture, and
  both `rlx-unfuse` and the MLX lowering read `inputs[1..]` as the captures — so
  `verify` rejected every `If` that captured anything.
- `rf::const_f32` emitted a 4-byte literal for any shape. `Op::Constant` carries
  the whole tensor, and backends copy `min(data, buffer)` floats, so a scalar
  paired with a wider shape filled element 0 and left the rest at zero — wrong
  numbers, not an error. It now fills the shape, and rejects a dynamic one.
- wgpu's rank-3 `Op::Conv` lowering computed `bias_off` / `has_bias` from the
  operand list and then hard-coded both to `0`, so `conv3d.wgsl` skipped its bias
  store. Unreachable today (`verify` rejects a 3-input `Conv`), but it would have
  dropped the bias silently.

- **KNOWN ISSUE — a workspace `cargo test` crashes a GPU test binary roughly one
  run in five.** `rlx-vulkan`'s `static_weight_pack_rebind` died with **SIGTRAP
  (signal 5)** and produced no output at all — no `test result` line, no panic —
  during `just test`. It passes 3/3 in isolation, and four of five `just ci` runs
  cleared the same stage.
  
  The cause is GPU contention **across test binaries**, which is why the
  `GpuTestGuard` work does not fix it: that guard is a process-local `Mutex`, so
  it serializes threads within one binary and cannot reach another process
  opening the same device. The `Justfile`'s existing `-j 4` cap on Darwin exists
  for this exact reason and is simply not tight enough.
  
  Two fixes, both with a real cost, left for a deliberate decision:
  * a **cross-process file lock** in the guard — correct in general, but needs
    `libc`/`fs2` in a core crate that has no such dependency today, plus
    stale-lock handling for a crashed holder;
  * **serializing the GPU backend crates** at the cargo level in `just test` —
    no new dependencies, at the cost of wall time on the canonical gate.
  
  Recorded rather than papered over: cargo *does* report the crash (exit 101,
  verified against a deliberately aborting test binary), so this fails loudly
  rather than passing vacuously — it is a flake in the gate, not a hole in it.


- **`just ci` had never been run end to end, and did not pass.** Its
  constituents had each been run in isolation; the chain had not. Two failures
  fell out on the first attempt, both invisible to any single recipe:

  * `lstm_three_way::other_gpu_backends_are_checked_for_the_wide_hidden_lstm_defect`
    **panicked unconditionally when CUDA was absent** — the intent ("this run
    would have proved nothing") is exactly right and exactly what
    `RLX_REQUIRE_DEVICE` expresses, but as a hard panic it failed on every host
    without CUDA. A workspace `cargo test` feature-unifies `cuda` in, so `just
    test` could never go green on a Mac. Now skips, and still fails loudly under
    the flag, which `rig.sh` sets. A second copy of the same pattern in that file
    was unreachable dead code after an earlier skip and has been removed.
  * **rlx-mlx had two `i32 -> i32` casts** that `-D warnings` rejects. `just
    lint` is default-features, so it never compiles rlx-mlx; only
    `just lint-features` reaches it.

  * **rlx-fpga's new `seq/lower.rs` had two clippy errors** — an unneeded
    struct pattern on the unit variant `Op::MatMul`, and a
    `Descriptor::default()` followed by thirteen field assignments. These are in
    the *default*-feature lint, so `just lint` itself had been failing.

  * **an `std::env::var("RLX_FORCE_DEVICE")` bypassing the `rlx_ir::env` shim**
    in `rlx-autodiff`'s `indexing_grad.rs`. The shim is what lets a test or an
    in-process A/B set a knob at all, which is why `check-rlx-env-vars` gates on
    it.
  * **`test-pyrlx` set up its venv `if [[ ! -d .venv ]]`** — guarding on the
    directory existing rather than on the dependencies being present. A venv left
    half-built by an interrupted run, or one predating a dependency being added
    to that list, was never repaired, and the recipe then failed forever on that
    machine with `No module named pytest`. Now each piece is installed on what is
    actually missing, so the recipe is idempotent and self-repairing (155 passed,
    17 skipped once it could run).

  Between them, `lint-features` has surfaced four defects the default-feature
  gate cannot see, and running the chain end to end surfaced four more that no
  individual recipe could.


- **Every GPU test now serializes device access; the guard is re-entrant and
  gated.** `GpuTestGuard` existed but was taken by almost nothing: 94 of the
  files in `rlx-runtime/tests` built a `Session` on a GPU device without it. Two
  were actively flaking (`elementwise_backend_parity`, `vulkan_parity`); the
  rest were latent.

  Three changes made the sweep safe rather than a deadlock generator:

  * the guard is **re-entrant** (thread-local depth count). A plain `Mutex`
    deadlocks the moment a guarded helper is called from a guarded test, which
    is exactly the shape these files have — a `run_on` chokepoint plus per-test
    `Session::new` sites. Only the outermost guard drains Metal's queues; an
    inner one releasing them mid-scope would defeat the point.
  * `common::serialize_gpu()` takes the lock without naming a device, which is
    the useful granularity for a test that loops over several.
  * `require_device_coverage::gpu_tests_serialize_device_access` pins the
    unguarded count at **0**, so a new GPU test that skips the guard fails
    rather than joining a backlog nobody is watching.

  517 insertions across 103 files. Suite: 229 binaries, 1337 passed, 0 failing,
  307s wall — the guard is a thread-local check plus an uncontended mutex, and
  only bites when two GPU tests actually overlap.


- **`GpuTestGuard` did not list `Device::Vulkan`**, so the one mechanism this
  tree has for serializing GPU use across a test binary's threads did not cover
  Vulkan at all — and on macOS Vulkan *is* Metal via MoltenVK, the same device
  the rest of the list exists to protect. `elementwise_backend_parity` returned
  `worst_rel=1.0` on Vulkan about one run in six and was clean at
  `--test-threads=1`; it also never took the guard. Both fixed. Measured 0/8
  failing runs after, against 1/6 before.


- **rlx-vulkan emitted a `Step` with no matching `StepDep` on the `KvAppend`
  path.** `record_segments` indexes the two arrays in lockstep, and the builder
  skipped *every* `Step::ActCopy` when attaching deps — correct only while the
  activation binder was the sole producer of one, since `drain_copies` pushes
  step and dep together. `Op::KvAppend` pushes an `ActCopy` directly, so it
  emitted 1 step and 0 deps. The loop now fills exactly the tail that has no
  dep, which states the invariant instead of approximating it.

- **Vulkan RNN `carry` never threaded state.** `Op::Lstm`/`Gru`/`Rnn` with
  `carry: true` take Vulkan's host fallback, and `host::eval` builds a throwaway
  graph on its **own** `rlx_cpu` arena and reads back only the output slot. The
  carry contract writes the final `hn`/`cn` over the `h0`/`c0` *inputs*, so
  those writes landed in the scratch arena and were dropped: four single-step
  runs diverged from one four-step run by 5.7e-2, from step 1 onward — state
  simply never advanced.

  `host::inplace_inputs` now names the operands an op rewrites in place and
  `eval_full` returns them, which both `Segment::Host` call sites copy back into
  the Vulkan arena. Vulkan measures 2.235e-8 against CPU now. Only Vulkan was
  affected; CUDA, ROCm, Metal and wgpu run carry natively and never take this
  path.

- **`just test-debug-verifier`** — the gate that would have caught both of the
  above, and `FusedTransformerLayer`'s arity, on the day they landed.
  `rlx_ir::debug_assert_valid!` re-verifies the graph after every pass that
  changed it, and it is a `debug_assert`: **every recipe in the `Justfile` is
  `--release`, so the verifier was compiled out of the entire gate.** Now wired
  into `just test`, scoped to the crates whose passes and schedulers carry the
  asserts.


- **`Op::FusedTransformerLayer` declared 10 inputs but its lowering reads 8.**
  `num_inputs` counted `ln1_b`/`ln2_b` unconditionally; `rlx_unfuse::expand_ftl`
  treats the LayerNorm betas as governed by `has_bias` like every other bias and
  **synthesizes zero betas** in the no-bias case. So a correctly-built no-bias
  node failed the IR verifier. Now `if has_bias { 14 } else { 8 }`, matching the
  code that actually reads the operands.

  It only ever failed in a **debug** build, because `debug_assert_valid!` is
  where the verifier lives and **no recipe runs a debug test build** — `just
  test` is `--release`. A whole class of IR-invariant violations is invisible to
  the canonical gate for that reason.

- **The quantized corpus cases sized every packed weight with a hardcoded
  256-element block.** Block size is per-scheme: the K-quants pack 256, but
  **Q8_0 packs 32**, so `q8_0_prefill`'s `w_packed` was allocated an eighth of
  the bytes its weight needs. Harmless while nothing fed the param real data,
  and a guaranteed out-of-bounds read the moment anything did — which is exactly
  what validating the family does. Now uses `scheme.gguf_block_size()`.

- **`OPKIND_TOTAL` had drifted** — `OpKind` grew to 187 variants (`Roll`,
  `FftQ`, …) while the corpus coverage denominator still said 186, so coverage
  was being reported against a stale total. Caught by `opkind_total_is_current`,
  which re-derives the count from `op.rs` rather than trusting the constant.


- **RoPE zeroed every position past the first when the cos/sin table was
  rank-1.** The row stride was read off the table's last dimension, which is
  right for a `[positions, width]` table and wrong for a flat
  `positions * n_rot/2` run — there the last dimension *is* the whole table, so
  position 1 indexed past the end, read as zeros, and silently blanked the rest
  of the sequence. `rope_identity_when_cos1_sin0` was failing on CPU. Rank-1
  rows are `n_rot/2` wide by construction.

  The rule now lives once, in `rlx_ir::shape::rope_table_stride`, and rlx-cpu,
  rlx-cuda, rlx-metal, rlx-rocm and rlx-wgpu all call it. Each had its own copy,
  which is how the forward and backward paths — and the CPU and GPU paths —
  drifted apart twice already; the surrounding comments were each warning about
  a different half of the same split. The four GPU backward paths were reading
  out of bounds on rank-1 tables for the same reason.

  Pinned by `rope_reads_a_row_per_position_from_a_flat_table`, which gives the
  two positions *different* angles — the pre-existing identity test passes even
  if every position reads row 0, since all its rows are equal.

- **`tests/ui/*.stderr` expectations regenerated.** A license header had been
  added to each UI test source without updating the expected diagnostics, so
  every line number was off by three and `--test ui` failed. Text is unchanged.


- **`Tensor::pad` silently moved the graph.** It built its constant edges with
  `Tensor::full`, which starts a fresh graph, and `cat` adopts into its *first*
  operand — so padding a traced value migrated the whole downstream chain off
  the graph being traced. The next `GraphScope` call, which does not adopt,
  then indexed that graph with a stale id and panicked out of bounds. The pad
  constants are now built on the padded tensor's own graph.

- **`Op::Lstm { carry: true }` silently never advanced state on Metal and wgpu.**
  Both kernels read `h0`/`c0` and then dropped the final state on the floor, so
  every decode step restarted from the same seed — plausible-looking outputs,
  wrong sequence, nothing failing. CPU, CUDA and ROCm already honoured the
  contract (the CUDA kernel's explicit `carry` flag and trailing write-back loop
  is the design the two GPU kernels now match). Found from `rlx-ten-vad`, whose
  streaming VAD scored `max|Δ| 0.52` with 64 voice-decision flips off-CPU while
  CPU was exact; both backends are now bit-identical to a single multi-step run.
  The `carry` flag is passed explicitly rather than inferred from `h0_off != 0`,
  since arena offset 0 is a legal placement for the state tensor.
- **CoreML/ANE dropped the same write-back**, with the identical signature
  (`max|Δ| 5.7e-2`, diverging from step 1). `Op::Lstm { carry: true }` is a host
  op there; `execute_lstm_f32` wrote `hn`/`cn` into the local staging arena and
  only `dst` was returned. The state now rides out through
  `host_exec::StateWriteback` and is written straight into `params` — not via
  `set_param`, which calls `invalidate_models()`: these params feed only the
  host-side LSTM and never enter the MIL graph, so recompiling the CoreML model
  per decode step would be pure cost. Found only because the new test's device
  loop was extended to `Device::Ane`.
- **`--features coreml` reported `feature_compiled(Device::Ane) == false`.** The
  feature is named `coreml` but the gate checks `cfg!(feature = "ane")`, and
  `coreml` did not imply it — so a build with the CoreML backend compiled in and
  `is_available() == true` still claimed the feature was absent. `skip_unless_device`
  picks skip-vs-fail from that pair, so ANE tests silently skipped on builds that
  could run them. `coreml` now enables `ane`.
- **`--features amx-sme` did not compile at all.** Three match arms in
  `rlx-cpu/src/intrinsics/apple_amx/sme.rs` still tested `Ok("true")` against
  `rlx_ir::env::var(..).as_deref()`, which returns `Option<&str>` — left behind
  when that helper moved from `Result` to `Option`. Feature-gated code CI does
  not lint. (For reference: with it fixed, the direct SME2 microkernel measures
  *slower* than the default Accelerate path on small GEMMs — 365× vs 398× RT on
  `rlx-ten-vad` — which is what `amx-dense`'s doc comment already claimed.)
- **`Device::Xdna` reported available and then panicked.** `is_available()` was
  `runtime_present() && (overlay || op_compile)`, but *both* kernel paths dlopen
  the rlx shim, so `AIECC`+`PEANO` alone passed the gate and the backend panicked
  at first compile with `no shim path`. `Backend::compile` returns a graph rather
  than a `Result`, so the gate is the only place to fail cleanly; it now also
  requires a resolvable `RLX_XDNA_SHIM`, and `diagnostic()` names exactly what is
  missing. The README also documented `RLX_XDNA_SHIM=…/libxrt_driver_xdna.so`,
  which dlopens fine and then fails with `undefined symbol: rlx_xdna_io_open` —
  it must be rlx's own `librlx_xdna_shim.so` from `csrc/xrt_gemm_shim.cpp`, and
  the build command is now in the README.
- **MLX dropped the same write-back.** Its `carry` path host-evalled the shared
  CPU kernel and returned only the output. MLX has no arena and holds `params`
  by shared reference for the whole lowering walk, so the state now rides back
  out through `lower_with_env_writeback` /
  `lower_and_run_typed_with_extent_writeback` and the backend applies it to the
  param buffers once the borrow ends. The old entry points remain, and
  `debug_assert` that nothing was dropped. `carry = false` is unaffected.

- **Three hand-rolled copies of the skip-or-fail decision, one of them broken.**
  `activation_batch_parity.rs` defined its own `skip_unless_available` — same
  name, same signature as the shared helper, **without** the
  `RLX_REQUIRE_DEVICE` assert — so the file was exempt from the flag while
  reading exactly like one that honoured it, and it fooled the new coverage
  gate, which matched on the name. rlx-cuda and rlx-rocm's
  `static_weight_pack_rebind.rs` each carried a third and fourth copy as
  `device_missing()`. All now call `rlx_ir::env::skip_unless_device`, and a
  local definition disqualifies a file from counting as migrated.

- **`lstm_three_way.rs::reference_f64` was dead code under `cpu,cuda`.** The
  file compiles under any of metal / gpu / cuda / rocm but consults the arbiter
  from the metal arm alone. Gated to match its caller rather than blanket
  `allow(dead_code)`, so adding a caller in another arm is a compile error
  pointing at it.

- **The capability guard skipped unavailable devices with a bare `continue`.**
  A rig where CUDA is *compiled in and broken* — the case that most needs
  reporting — looked identical to a Mac that has no CUDA at all, and CUDA's
  entry stayed unverified either way. It now goes through
  `skip_unless_available`, so `RLX_REQUIRE_DEVICE=1` separates the two.

- **The `quantized` corpus skip cited a reason that had stopped being true.**
  It said "set_param is an f32 API", but `CompiledGraph::set_param_typed` takes
  raw bytes and predates the test. The actual blocker is an *authority*: those
  cases need real Q4_K/Q6_K/Q8_0 fixtures and a block-decode reference in the
  oracle before a score against them means anything. Reworded so it reads as
  missing work rather than a permanent API limitation.

- **A per-query mask under `MaskKind::Custom` was undefined and the backends
  disagreed about it.** `Custom` is key padding — one bit per `(batch, key)`.
  Handed a real query axis, MLX and wgpu broadcast it per query while CPU, Metal
  and Vulkan read the same tensor as key padding. That is not a wrong number you
  can catch by comparing backends: there was no contract saying which was right,
  so `attention_mask_shapes.rs` documented the divergence as out of scope rather
  than freeze whichever backend happened to be asked.

  `rlx_ir::repr_check` rule R8 now rejects it, naming `MaskKind::Bias` as the op
  that *does* define a per-query mask. Undefined-and-refused beats
  undefined-and-divergent. Every broadcast spelling of a legal key-padding mask
  — `[B, S_k]`, `[1, S_k]`, `[1, 1, S_k]`, `[B, 1, 1, S_k]` — stays accepted,
  pinned by the same test that the parity suite enumerates.

- **`RLX_REQUIRE_DEVICE` was set by nothing.** It shipped in nine Rust files
  with the note that "rig runs should set it", and no Justfile recipe, script or
  CI config did — so the mechanism built to stop a card-less rig reporting green
  was off everywhere it mattered, which is the state the MI100 falling off the
  PCIe bus exploited.

  `rig.sh` now exports it on all three remote runtimes (`RIG_REQUIRE_DEVICE`,
  default 1, overridable in `scripts/rig/local.env` while a rig is known
  degraded), and the GPU recipes in the `Justfile` set it from a `require_device`
  variable — asking for `just test-gpu` is asserting a GPU is present, and a host
  without one wants `just require_device=0 test-gpu`.

  `just test-rocm` deliberately does **not** set it: that recipe is in `just ci`,
  which runs on developer machines where rlx-rocm compiles fine and no AMD GPU
  exists — exactly the case the flag is designed to fail.

  It was also read as `env::var(..).is_none()` at all six sites, so
  `RLX_REQUIRE_DEVICE=0` *armed* it. The registry declares it `EnvKind::Bool`;
  it now reads through `env::flag`, so `=0` means off.

- **`rlx-runtime/tests/host_fallback_never_nops.rs`** — the claim-then-Nop
  composition is now caught statically, with no device and no numerical
  comparison.

  `cpu_nop_fused_ops_parity.rs` catches this class by running the three ops on
  every available device and requiring a non-zero answer. That works, but it
  needs a device, and it needs someone to have written a case for the op — and
  coverage-by-enumeration is what drifted in the first place. The new gate
  derives it from the two static facts that compose into the bug: rlx-cpu
  declares what it cannot thunk (`NO_THUNK_ARM`), and Vulkan declares what it
  routes to the host (`routes_to_cpu_host`). Their intersection must be empty.

  Verified to bite: re-adding `Op::PartitionedConv` to Vulkan's
  `is_host_fallback` turns the gate red with the original diagnosis. It also
  requires rlx-cpu to actually expand every kind it lists, so the list cannot
  claim something the expander misses.

  Coverage is reported rather than implied: metal, wgpu, cuda, rocm, oneapi, mlx
  and coreml reach rlx-cpu through op-specific scheduler arms with no single
  predicate to ask, so they are **not** checked, and a test says so out loud
  instead of letting a green run read as full coverage.

- **A batch-broadcast attention mask was wrong on every backend**, two
  different ways, and read out of bounds on three.

  `MaskKind::Custom` is a binary `[batch, key_len]` key-padding mask. `[1, S_k]`
  — one padding row shared by the whole batch — is the same mask, as are its
  rank-3/rank-4 spellings. None of them agreed.

  * **CUDA, ROCm, wgpu** derive mask strides from the tensor's shape, but built
    each one from a product of dims, so an axis of extent 1 got a non-zero
    stride instead of 0. `rlx_ir::mask_strides_for_shape` now zeroes broadcast
    axes. **wgpu carried a byte-identical private copy of that function**, which
    is how it drifted; it now calls the shared one.
  * **CPU, Metal, Vulkan** index the mask at a hard-coded `mask[b * S_k + k]`
    with no strides at all, so `[1, S_k]` read *past the end of the tensor* for
    every batch above 0, into whatever the arena placed next to it. New pass
    `rlx_opt::legalize_custom_attention_mask` materializes the broadcast in the
    IR ahead of them — the same trade `LegalizeBroadcast` already makes for
    `Op::Binary`, and split out of it so Metal (whose binary kernels are
    stride-aware and skip that pass) can run just this rewrite.

  Out-of-bounds reads of a neighbouring tensor produce *plausible* numbers, so
  `rlx-runtime/tests/attention_mask_shapes.rs` scores against a reference
  computed in the test rather than against another backend, and separately
  requires every spelling of one mask to agree with itself on each backend.

  A per-query mask (`[S_q, S_k]`, `[B, H, S_q, S_k]`) is still undefined for
  `Custom` — `MaskKind::Bias` is the per-head, per-query tensor. MLX and wgpu
  broadcast one correctly; CPU, Metal and Vulkan read it as key padding. That
  disagreement is documented in the test rather than pinned.

- **Vulkan rejected `Op::Qr` and `Op::Svd` at legalize although its scheduler
  already ran them.** Both were listed in the `Step::HostOp` arm next to
  `Cholesky` / `TriangularSolve` / `Det` / `LogDet`; only the two
  `SUPPORTED_OPS` entries were missing, so compilation failed before the
  scheduler was ever reached. `qr_vulkan` and `svd_vulkan` now match CPU
  bit-exactly, as the other four backends already did.

- **39 parity tests asserted instead of skipping when a GPU was present but out
  of memory.** They called `Session::new(Device::Gpu)` with no availability
  check — the CUDA and ROCm cases beside them had one, wgpu was assumed always
  present. On a box already running a training job, wgpu's `request_device`
  fails with "Not enough memory left", `is_available` goes false, and the suite
  turned 37 tests red at once across 25 files, indistinguishable at a glance
  from a regression in the kernels.

  They now share `tests/common/mod.rs::skip_unless_available`, which skips
  loudly — **and asserts under `RLX_REQUIRE_DEVICE=1`**, so a rig where the
  device *should* work cannot report `ok` for tests that never ran. Rig runs
  should set it. The assert fires only for a backend that was actually compiled
  in; one the build does not contain is not what the flag is about.

- **Vulkan `Op::PartitionedConv` silently returned a buffer of zeros.**
  Reproduced identically on MoltenVK, NVIDIA's driver and RADV, so not a driver
  quirk — the op never ran at all.

  Three correct-looking decisions composed into the bug. Vulkan *claims*
  `PartitionedConv` in `SUPPORTED_OPS`, so it never enters the unsupported set
  that makes `legalize_or_rewrite_for_backend` expand it. `rlx-unfuse` (shared
  by CUDA / ROCm / wgpu / Vulkan / oneapi) has no arm for it — only
  `rlx-fusion`'s unfuse does. So the node survived to the scheduler, which
  routed it to the CPU host fallback, where it is a **Nop**: rlx-cpu expands
  `PartitionedConv` before building thunks and therefore has no kernel for it
  either. A claimed op, a fallback that accepts it, and a kernel that does
  nothing — output stays zero and nothing reports an error.

  `rlx_vulkan::unfuse::expand_partitioned_conv` now expands it in Vulkan's own
  compile entry, mirroring the `expand_cpu_nop_fused` oneapi already carried.
  It is also removed from `is_host_fallback`: if one ever reaches the scheduler
  again, failing loudly beats zeroes.

  `gpu_filters_parity` gained named `partitioned_conv_op` cases for wgpu (it had
  only the anonymous `all()` batch) and oneapi (which had none, despite carrying
  the same class of fix untested).

  The class as a whole is now guarded by
  `rlx-runtime/tests/cpu_nop_fused_ops_parity.rs`, which runs all three ops
  rlx-cpu Nops — `FusedConvBiasAct`, `PartitionedConv`, `FusedTransformerLayer`
  — on every available device and requires both agreement with CPU **and** a
  non-zero result. The second check is the load-bearing one: a tolerance
  comparison alone can pass against a reference that happens to be near zero.

- **`advisory_capabilities` had drifted from four of the seven backends it
  describes**, in both directions:

  * ROCm implemented `register_kv_row_feed` / `feed_kv_row` and the table said
    `kv_resident`, but its own `capabilities()` did not — so a caller gating on
    the executable took the slow path on a backend that supported the fast one.
  * ROCm and CUDA forward `set_moe_resident_experts` and never claimed `moe`.
  * MLX claimed `moe` it does not implement, and hid the `async_pipeline` it
    does.
  * ROCm and Vulkan omitted `typed_io` / `active_extent` they honour.

  Neither direction shows up as a test failure anywhere else — a capability the
  table omits is a fast path planners never take, and one it invents is a fast
  path callers take and lose. Both are missed optimisations, not wrong numbers.

  `CompiledGraph::capabilities()` now exposes the executable's own answer (the
  table was the only public way to ask), and
  `rlx-runtime/tests/capability_table_matches_backends.rs` pins the two
  together for every device the host can instantiate. Verified against cpu,
  metal, mlx, wgpu, rocm, vulkan — and, on the CUDA rig (RTX 3080 Ti, driver
  595.84), **cuda**, which closes the last of the seven: `kv_resident` is backed
  by a real row feed and every probeable flag matches what its method returns.

  A third case the table could not catch, because both copies were wrong the
  same way: **CPU implements `bind_handle` / `read_handle` and reported
  `persistent_handles: false`**. `probeable_flags_agree_with_what_the_methods_return`
  now calls each probeable hook and requires the flag to match what it returned,
  in both directions.

  The `moe` flag is also documented more narrowly than it was: it means expert
  residency, whose setters return `()` and therefore need a flag. The
  instrumentation hooks return `Option`/`bool`, report for themselves, and are
  CPU-only — they are no longer implied by it.

- **wgpu GGUF dequant tests reported 24 failures when the GPU was merely busy.**
  They called `.expect("no wgpu adapter")`, so a machine whose VRAM was full
  produced two dozen simultaneous FAILUREs that read exactly like a numerical
  regression in the dequant kernels. They now skip loudly, and
  `RLX_REQUIRE_DEVICE=1` turns the skip back into a failure for rig runs where a
  missing device means the run proved nothing.

- **`Op::KvAppend` used the wrong row stride on Metal, CUDA and ROCm.** All three
  derived the stride between outer slices from the OUTPUT shape, whose axis dim
  is `pos + 1` (the op returns the `[..pos+1]` prefix), rather than the cache's
  capacity. Their shared comment — *"Output shape == cache shape"* —
  contradicted `infer_shape`. The stride is only read when `outer > 1`, i.e.
  batch > 1, so batch-1 decode never exercised it.

- **`Op::KvAppend`'s output cannot alias the cache for every shape.** The
  aliasing contract needs the `[..pos+1]` prefix to be CONTIGUOUS from the start
  of the buffer, which holds only when nothing precedes `axis`. For the natural
  attention layout `[batch, heads, seq, dim]` with `axis = 2` the prefix is
  `batch*heads` strided slices, so a row write into the aliased slot returns a
  different tensor. `rewrite_for_backend` now lowers those shapes back to
  narrow + concat on **every** backend, native support or not — a shape guard,
  not a capability one, so it runs before the "backend supports everything"
  early return. Batch-1 `[1, seq, heads*dim]` decode keeps the O(1) path.

- **Metal: the dual-output residual fusion could alias its own operands.**
  `fused_residual_rms_norm` writes the residual sum in its first pass and then
  re-reads `x` and `res` in the second to recompute `x + res`; the overlap check
  covered the norm output but not the sum's destination, so a plan that aliased
  the add's output onto an operand would have had the second pass read what the
  first overwrote. Guarded before enabling the path by default.

- **Metal `set_param_bytes` silently wrote a quarter of any f32 param.**
  `Arena::write_bytes` clamped a **byte** length against an **element** count, so
  a wider-than-byte param was truncated to `num_elements` bytes and the copy then
  succeeded with no error. It survived because that path is overwhelmingly used
  for quantised U8 weights, where the two units coincide.
- **A re-bound param did not reach a fused weight pack.** With the pack skip
  armed, `run(); set_param(w, ..); run()` kept the first run's pack and the
  consuming GEMM silently used the old weights — the shape of every weight-swap
  workload (training step, LoRA merge, quantisation re-bind, sweep harness
  reusing one executable). All param setters now invalidate, on every backend.
- **wgpu: the static-weight skip never armed on three of five readback paths**,
  so whether the optimisation engaged depended on which path a graph happened to
  take; small-output graphs rebuilt their constant packs every run.
- **wgpu: complex ops were routed to a lane-blind host fallback** on discrete
  NVIDIA Vulkan, returning the lane-wise product `(ar·br, ai·bi)` instead of the
  complex one. Complex add *is* lane-wise, so only mul/div failed. `complex_parity`
  13/14 → **14/14** on that backend.
- **wgpu: an op whose operands span two arena stripes read zeros.** A kernel binds
  one stripe, and slot placement guarantees no single tensor straddles a boundary
  — not that an op's *operands* share one. Measured: 786432 of 786432 outputs
  returned 0.0 instead of 18.0 under `RLX_WGPU_SHARD_GPU=1`. Such ops now host.
- **Wide-hidden LSTM was wrong on Apple GPUs** (`hidden > 32`), on both the Metal
  MSL and wgpu WGSL kernels. Metal's default `exp`/`tanh` are the fast variants
  and are not accurate enough for the recurrence: the error compounds through the
  cell state until units collapse to exactly 0.0. Metal now uses
  `metal::precise::`; WGSL has no such namespace, so `lstm.wgsl` uses a
  range-reduced `exp` of its own. CPU, CUDA and wgpu-on-NVIDIA-Vulkan were always
  correct. Native paths restored at full width on both.
- **The weight-pack skip was silently dead on graphs deeper than one layer.**
  Packs were given arena slots that an earlier activation also owns — legal
  within a single run, but the skip means that activation is re-executed next run
  and clobbers the pack. Liveness is now whole-graph for packs, and `rlx-cuda`
  and `rlx-rocm` (which have their own planners) pin them too. A one-layer graph
  cannot detect this, which is why it passed the original guards.

- **`rlx-cpu`'s `amx-bnns` feature did not compile.** Two `matches!` arms in
  `intrinsics/apple_amx/bnns.rs` still read `Ok("true")` after the env lookup
  moved from `std::env::var` (`Result`) to `rlx_ir::env::var` (`Option`), so
  `RLX_CPU_BNNS_BF16` / `RLX_CPU_BNNS_F16` failed to typecheck and the whole
  feature was unbuildable. `just lint-features` now compiles that combination.
- **`rlx-wgpu`'s `Step` enum lost its documentation and its `dead_code` allow.**
  The per-shader kernel enums (`BatchNormKernel`, `QuantI8Kernel`,
  `ScaledLowpKernel`, …) were inserted *between* `Step`'s doc comment plus
  `#[allow(dead_code)]` and `Step` itself, so both attached to
  `BatchNormKernel` — which then carried two stacked doc comments — while
  `Step` warned on six fields it deliberately keeps (`mask_buf` extends a
  buffer's lifetime; `meta_idx` is consulted during bind-group construction).
- **`scripts/publish.sh` refused to run.** Its `validate_tier_coverage` requires
  every workspace member to appear in a publish tier or in `SKIPPED`, and
  `rlx-fem` was in neither, so the release driver aborted before its first
  upload. It has no path dependencies, so it joins tier 0.
- **`rlx-runtime` declared `build = "build.rs"` inside its docs.rs metadata
  table.** A blank line does not close a TOML table, so the key landed in
  `[package.metadata.docs.rs]` where cargo ignores it; the build script only ran
  because cargo auto-detects `build.rs` at the package root. It is now declared
  in `[package]`.
- **The Cortex-M trainer wrote its weights to a CWD-relative path.**
  `--out` defaulted to the literal `rlx-cortexm/src/model_weights.rs`, so
  running the trainer from its own directory created a `trainer/rlx-cortexm/`
  shadow tree — which is how a duplicate `model_weights.bin` and
  `test_set.bin` came to be committed. The default now resolves against
  `CARGO_MANIFEST_DIR`.
- **59 relative Markdown links pointed at nothing**, nearly all of them left
  over from regrouping the crates under
  `crates/{core,backends,io,numerics,tooling,bindings}/`: a crate README moved
  two levels deeper, and every `../docs/foo.md` in it went stale at once. All
  496 links across the 133 tracked Markdown files now resolve, and
  `just check-doc-links` keeps them that way.
- **~230 broken rustdoc intra-doc links.** Where the target resolved to a
  dependency it is now fully qualified (`rlx_ir::Op::TopK`); where it did not
  resolve from that module, the link is gone and the text stays as inline code.
  De-linking rather than guessing is deliberate — a link that silently points at
  the wrong item reads as authoritative.

- **The op-coverage claim was three releases stale, and overstated.** The README
  and per-backend docs advertised "full **153/`OpKind`**" on ten backends.
  `OpKind` has since grown to **187**, and no backend claims all of them —
  actual claims run 161 (oneAPI) to 179 (Metal). `docs/op-coverage.md` is
  regenerated from each backend's `SUPPORTED_OPS`, and the README, docs index
  and the CUDA / TPU / Vulkan / oneAPI crate READMEs now quote the generated
  numbers instead of a hand-copied total. `just check-op-coverage` already
  guarded the matrix; nothing guarded the prose around it.
- **Private note-taking slugs shipped in public doc comments.** Fifteen
  `[[feedback_perf_is_north_star]]`-style wiki-links across `rlx-cpu`'s Apple
  AMX/SME modules, an `rlx-vulkan` test and a design doc referenced a
  machine-local notes system that no reader of this repository can resolve —
  and rustdoc tried to resolve two of them as intra-doc links. Each is replaced
  by the convention it stood for, spelled out.
- **Two doc comments had drifted off the item they document**, leaving the
  function undocumented and the comment attached to an unrelated one:
  `compile_overlay` in `rlx-xdna` and `needs_broadcast_prologue` in
  `rlx-unfuse`. Both reattached.
- **`cargo clippy --all-targets -- -D warnings` is clean again** across the
  workspace: eleven needless borrows and an `eprint!("{t}\n")` in `rlx-wgpu`, a
  manual slice-size product in `rlx-metal`, a no-op `1 *` in an `rlx-mlx` test,
  a manual checked division in `rlx-cpu`, and a `vec_init_then_push` in an
  `rlx-runtime` test whose pushes are all `#[cfg]`-gated (the lint cannot model
  that, so it is allowed with the reason stated).

- **`just lint` failed outright on the `cpu,cuda` feature combination.**
  `tests/wgpu_conv3d_bias_parity.rs` gated only its `#[test]` on `gpu`, leaving
  four helper functions and their imports unconditional — dead code in any build
  without `gpu`, and under `-D warnings` that is 15 hard errors rather than a
  warning. The module is gated as a whole now. Nothing caught it because
  `lint-features` reaches that combination only on a host where `rlx-cuda`
  compiles.
- **Rustdoc is warning-free across the workspace** (353 → 0 under each crate's
  docs.rs feature set).

- **The shape verifier read a GGUF `DequantMatMul` weight's axes backwards.**
  `infer_shape` sent every rank-2 weight through `matmul_shape`, which assumes
  `[k, n]` — but GGUF stores a linear as `[out_dim, in_dim]`, i.e. `[n, k]`
  (the same order that lets `dequant_grouped_matmul_packed` take an expert bank
  from a `ffn_*_exps.weight` blob with no transpose). Any graph declaring the
  logical rank-2 weight instead of a packed byte blob was rejected with
  `matmul K mismatch: 64 vs 3`. New `shape::dequant_matmul_shape` states the
  `[n, k]` rule and `infer_shape` picks it for `scheme.is_gguf()`; the Int8 /
  NVFP4 schemes, `LoraMatMul` and `QMatMul` keep `[k, n]` per their contracts.
  The wrong rule was always there — it only became visible now that a rejected
  inference is reported instead of silently discarded.

- **A flaky `rlx-compile` test could fail on an assertion unrelated to what it
  tests.** `RLX_NO_NATIVE_FK_REGIONS` is a process-global env var that
  `apply_native_fk_defaults` reads for *every* target, and
  `tpu_native_fk_region_pass_policy` sets it mid-test — but that writer held
  `ENV_FK_TEST_LOCK` alone, so a mutex with exactly one user serialized nothing.
  Any of the eight sibling tests that build a pass list could observe
  `native_fk_regions = false` and fail. The eight now take the lock, which is
  documented at its declaration as the contract for new tests here, and taking
  it tolerates poisoning so one real failure cannot cascade into eight
  `PoisonError` panics.

- **A second flaky test of the same shape, in `rlx-ffi`.** The node C ABI keeps
  one process-global last-error string — correct for a C consumer, but a test
  that calls `rlx_node_start` and *then* reads `rlx_node_last_error` can read a
  sibling's message. `rejects_bad_rank_world` failed asserting on `2/2` and got
  `unknown mode [trian]`, the string `rejects_an_unknown_mode` had just written.
  The seven tests that drive the ABI now share `FFI_TEST_LOCK`. Measured: 5
  failures in 60 runs without it, 0 in 60 with.

- **A third flaky test of the same shape, and the highest-rate one:
  `rlx-metal`'s `mpsgraph_sync_compile`.** `the_env_override_still_selects_the_control_arm`
  sets `RLX_MPSGRAPH_NO_SYNC_COMPILE=1` to check the A/B control arm, while
  `sync_compile_mitigation_is_in_force` reads the same variable — and cargo runs
  both on threads of one process, so the reader intermittently reported the
  mitigation as *disabled* when it was not. The `// SAFETY: single-threaded
  within this test` note was the tell: `set_var` is process-global, so what
  matters is the sibling thread, not this test's own body. Both now take
  `ENV_ARM_LOCK` and the safety comment states the real justification.
  Measured: 22 failures in 60 runs without it, 0 in 60 with — it had passed
  three consecutive full-suite runs by luck.

- **A fourth flaky test, and the one that was *not* an env leak:
  `rlx-runtime`'s `custom_ops` registry tests.** All three call
  `clear_for_tests`, which wipes the process-global custom-op registry, so a
  sibling's clear could land between `re_register_replaces`'s `register` and its
  `execute` — making `execute` return `None` and the test unwrap a value the
  code under test never failed to produce. The three now share
  `REGISTRY_TEST_LOCK`. Measured: 4 failures in 300 runs without it, 0 in 300
  with.

  Found by measuring rather than by pattern-matching: a survey of the 22 test
  files that mutate process-global state ran each 25 times and cleared 21 of
  them, and the two obvious hypotheses for this one (`RLX_DEVICE` and
  `RLX_TRACE_PERFETTO` leaking) were both disproved by forcing each variable
  globally and seeing the suite still pass.

### Performance

- **wgpu: the residual→norm tee now covers RmsNorm, not just LayerNorm.**
  `detect_residual_ln_tee_pattern` recovers the case both residual fusions
  decline — a residual sum with two consumers, which is every transformer layer
  — by emitting one step that writes the sum *and* the normalised result. It was
  written for a LayerNorm vision transformer and matched only `Op::LayerNorm`,
  so it missed every Llama-class model. Both norms already shared the same wgpu
  lowering arm, so this is a mode flag (`is_rms`) on the existing kernel rather
  than a second one. On a 6-block probe: **24 → 17 steps** (8 `binary` + 8 norm
  → 1 + 1 + 7 tee). Numerics unchanged against the CPU reference for both norms.
  Discrete Vulkan/DX12 route these ops to the host entirely, so the tee is
  correctly inapplicable there.

- **Metal: the residual + RMS-norm fusion now fires once per LAYER instead of
  once per graph.** In a transformer the residual sum feeds both the norm and
  the next residual (`h += attn; n = rms(h); h += ffn(n)`), so it has two
  consumers and the fusion declined — on Carbon-500M it fired **1** time across
  28 layers. The fused Metal kernel already had a dual output for exactly this
  (it can emit the sum alongside the normalised result); it was gated off. On a
  28-layer decode step **55 `binary` + 56 `rms_norm` dispatches collapse to 56
  fused ones — 55 fewer kernel launches per token**, with byte traffic unchanged
  (these are `[1, hidden]` tensors, so the win is launches, not bandwidth) and
  output bit-identical to CPU.
  `RLX_METAL_FUSE_RESIDUAL_DUAL=0` opts out.

  Not quoted as a wall-clock figure: this Mac carried a load average of 50-122
  throughout, where the same arm varied 5x run to run. The dispatch-count
  reduction is exact and machine-independent; the time it buys is not measured.

- **Fused QKV / gate-up weight packs are materialised once instead of per step.**
  The matmul-fusion passes build them by concatenating weight tensors, and that
  concat was re-run every decode step. On Carbon-500M (28 layers) it was **1879 MB
  of 3941 MB of DRAM traffic per token (47.7%)**.
  - CUDA (RTX 3080 Ti, 28-layer decode graph): **7.16 → 3.07 ms/iter (2.33×)**,
    at **+864 MiB** arena residency (1480 → 2344 MiB) — inherent, since skipping
    the recompute means keeping the result.
  - ROCm (Radeon 780M, gfx1103, 12-layer decode graph): **24.39 → 12.32 ms/iter
    (~1.98×)**.
  - Metal (M4 Pro): **−15.2%** total GPU device time; 3941 → 2062 MB/token.
  - Vulkan: mechanism validated; no timing (no meaningful Vulkan perf target).

  `RLX_NO_WEIGHT_CONCAT_FUSION=1` is **not** a substitute: it recovers the same
  bytes but costs 84 extra GEMM launches/token and measured −0.9%, i.e. nothing.

## [0.2.15] — 2026-08-16

### Added

- **`rlx_ir::verify_unique_leaf_names`** — reports `Op::Param` / `Op::Input`
  leaves that share a name. Such a leaf is never bound and silently reads zeros.
  Deliberately *not* part of `verify` (which is debug-asserted after every
  fusion pass): graphs in the wild can still trip it, and each needs its own look
  first. Every instance it found was a real bug. Two are fixed below (the
  `Rewriter::copy_node` duplication it was written for, and `jvp`'s
  `tangent_<name>` collision); one is open in `rlx-models`, where Qwen3.5 prefill
  graphs declare `last_token_idx` twice — once at flow level as F32, once inside
  the `GatherLastToken` block as I32. That one is currently harmless only by
  accident: the unbound node feeds a redundant second gather along an axis the
  first already collapsed to extent 1, where index 0 is the only legal index.
  Nothing guarantees which of two same-named nodes gets bound, and the other way
  round every `last_logits_only` prefill would return the *first* token's logits.
  Diagnosed in full in `rlx-qwen35/tests/last_token_idx_gather.rs` (`#[ignore]`d:
  the fix changes the logits output rank, which the runner, speculative-decode
  and serving paths all consume).

- **External GPU over USB4 / Thunderbolt on Apple Silicon (`Device::Egpu`,
  `rlx-egpu`).** PCIe tunnelling works on Apple Silicon — a device behind the
  tunnel enumerates as a normal `IOPCIDevice` with `IOPCITunnelled = true`. What
  macOS does not ship on arm64 is a driver for PCI base class 0x03, so a discrete
  GPU enumerates and is then left unclaimed. The gap is a driver, not a bus.

  What is complete: PCI discovery over the tunnel (IOKit on macOS, `/sys/bus/pci`
  on Linux), config-space / BAR / DMA transport through a PCIDriverKit extension
  (`dext/`, with its build, entitlement and signing procedure in `docs/egpu.md`),
  ahead-of-time kernel packs that compile on a ROCm/CUDA host and are readable
  with no toolchain, and SHA-256 verification of pinned vendor firmware.

  What is **not**: device bring-up. The AMD RDNA3/4 sequence (PSP firmware load,
  SMU, GFX/SDMA rings, GPUVM page tables) is written but **has never been
  executed** — no card has been attached to run it — and NVIDIA GSP bring-up is
  reserved, not written. So there is no execution path: `SUPPORTED_OPS` is empty
  and `compile` surfaces a diagnostic rather than falling back to the CPU.
  `is_available()` is gated on `am::VALIDATED_ON_HARDWARE`, *not* on
  `cfg!(feature = "am")` — compiling an unexecuted driver must not make the
  device dispatchable. `just test-egpu`, `just egpu-probe`, `just egpu-inspect`,
  `just egpu-bake`, `just egpu-firmware`.

- **`Op::Roll`** — cyclic shift along one or more axes (`jnp.roll` /
  `torch.roll`). Shifts may be negative or exceed the axis length; both reduce
  modulo `n`. Output shape always equals input shape, which is what separates it
  from `Op::Slice` and from a circular `Op::Pad`. No backend has a native kernel:
  `rlx_fusion::LowerRoll` legalizes each axis to
  `concat(narrow(tail), narrow(head))`, so that pass is the only implementation
  and therefore also the definition. The two narrows cost exactly one copy
  regardless of shift, so a fused kernel would save one pass, not an order of
  magnitude.

- **`Op::ScatterAdd` gained an `axis`** — the "unpermute" half of MoE routing,
  the transpose of `Op::Gather`, and the shape every immersed-boundary or
  segment-reduction kernel wants. Backends implement `axis == 0` only and say so
  by panicking; `rlx_fusion::LowerScatterAddAxis` normalizes every other axis
  ahead of them in the compile pipeline.

- **Lane masks (`rlx_ir::lanes`) — per-lane reset as *data*, not control flow.**
  A batched workload running `n_lanes` independent problems (parallel RL
  environments, a batch of decode sequences) finishes lanes at different times.
  Resetting on the host throws away the captured graph every time, because the
  graph is what encodes which buffers are read — on CUDA the capture is dropped
  and re-taken, and the cost lands exactly on the episode boundary, which for
  short episodes is most of them. Expressing it as
  `state' = where(mask[lane] != 0, reset_value, state)` with `mask` a plain
  `[n_lanes]` graph input means the *contents* change every step and the *graph*
  never does, so a captured schedule stays valid across any sequence of resets.

- **Pestle GGUF schemes `G8_0` and `Q2_0`.** `G8_0` (ggml type 143) is exact
  ternary — every weight a 2-bit code `{0,1,2}` mapped to `(q−1)·d` — but unlike
  `Q2_0` the scale is per *group of 8*, so a 32-element block carries four bf16
  scales for 4 bits/weight. That finer granularity is what lets an embedding
  table survive at ternary code width. `Doses-AI/Pestle-27B-Ternary-GGUF` uses
  `G8_0` for the untied `token_embd` / `output` tables and `Q2_0` pairs for the
  transformer linears.

- **`rlx_gguf::invariants` — a quantization oracle that references no other
  implementation.** Every other check on the quant path is *differential*:
  fused-vs-unfused, backend-vs-backend, or against a `w_ref` from the same
  decoder under test. A differential check cannot fail when the shared formula is
  itself wrong — not hypothetical here, given the RmsNorm `1/r` bug (wrong in all
  seven backends), the RoPE table stride (wrong and agreeing in three) and the
  GELU constant (wrong in `rlxsl`, therefore in every generated backend). This is
  the forward-path equivalent of finite differences: properties that follow from
  what a quantizer *is* — projection closure, value idempotence, exact constant
  and zero blocks, error bounded by the scheme's own step size.

- **`rlx-lbm` — moment-encoded lattice Boltzmann (HOME-LBM).** Stores the first
  three velocity moments per node and rebuilds the populations in-kernel rather
  than storing 9 (D2Q9) or 27 (D3Q27) distribution values. In tree for two
  reasons, neither of them fluid dynamics: it is the same DRAM-for-ALU trade as
  `DequantMatMul` / `ScaledMatMul` / `SynthMatMul`, applied to a stencil; and a
  27-point periodic stencil plus a large-grid-to-small-array reduction stresses
  the fusion and region machinery where transformer graphs never reach. The `ir`
  feature builds one step as an rlx `Graph` (streaming via `Op::Roll`) so it runs
  on every backend through `Session`; `just test-lbm` runs that path.

- **`rlx-opscope::bytes` — a static memory-traffic ledger read off the IR.** Most
  of the perf work in this repo has converged on the same finding from different
  directions: the bottleneck was bytes, not math. Each of those was found with a
  profiler, after the fact. A graph already contains enough to predict them —
  this computes per node the bytes that *must* move and the FLOPs that *must*
  happen, and reports arithmetic intensity, with no device present. It is a lower
  bound (one read per operand, one write per output — perfect reuse within an op,
  none across ops) and diagnostic only: "measured ≫ this" means headroom,
  "measured ≈ this" means the win has to come from somewhere else.

- **Metal concurrent command encoders (`RLX_METAL_CONCURRENT`, opt-in).** Opens
  compute encoders in `Concurrent` dispatch mode so independent decode dispatches
  (q/k/v, gate/up) overlap on the GPU, inserting a `memoryBarrier` only before a
  dispatch that data-depends on the current wave. Off by default; a classic
  Serial encoder is the unchanged path. Ships with its own bisection harness
  (`_SPLIT_ENC`, `_IDX_LO`/`_IDX_HI`, `_OPAQUE`, `_FENCE_ALL`,
  `_BARRIER_FRESH`, `_STATS`) because a Concurrent encoder that opts out of
  hazard tracking is not implicitly ordered after the previous one, so the first
  dispatch of every encoder can go unsynchronised.

- **`rlx_bbo::powell`** — Powell's conjugate-direction method, filling the gap
  between `adam_opt_nd` (needs gradients) and `cmaes` (stochastic, needs a
  population): deterministic, function-values only, superlinear on smooth
  objectives. The direction update uses Brent's rule rather than replacing
  unconditionally, which would collapse the direction set toward linear
  dependence and degrade the search onto a subspace.

- **`rlx_autodiff::grad_with_loss_wrt` — gradients w.r.t. an intermediate
  activation.** `wrt` entries are now `Wrt` designators rather than bare
  `NodeId`s: `Wrt::Leaf(name)` for a named `Op::Input`/`Op::Param`,
  `Wrt::Output(i)` for an index into `forward.outputs`, `Wrt::Node(id)` for the
  previous raw-id behavior. Only leaf *names* and output *positions* survive the
  renumbering done by `prepare_graph_for_ad`, so `Wrt::Output` is what makes an
  intermediate — a residual-stream activation, say — addressable: publish it as
  an auxiliary forward output and refer to it by index. Combined with the
  existing `d_output` input (which takes the shape of `outputs[0]`, scalar or
  not), this gives a VJP at an arbitrary cut point: seed any cotangent, read
  `∂outputs[0]/∂tap`. `grad_with_loss` / `grad_with_loss_opts` are unchanged and
  now delegate to it.
- **`GradWithLossOptions::emit_aux`** (default `true`, set via `with_aux`) —
  suppresses mirroring `forward.outputs[1..]` into the backward graph's outputs.
  For taps that exist only to designate `Wrt::Output` targets, the values are
  dead weight; dropping them lets the compiler DCE the readback.
- **Finite-difference VJP tests for the ops a decoder lens depends on** —
  `attention_causal_vjp_fd` (causal and unmasked, including cross-position),
  `rope_vjp_fd` (full and partial rotation), `rms_norm_rank_vjp_fd` (ranks 2/3/4;
  Qwen3 normalizes *per attention head*, so rank 4 with `axis = -1` is a real
  shape and not just a hypothetical), `gated_delta_net_vjp_fd` (gradient through
  the recurrent state across positions, plus causality), and
  `depthwise_conv1d_vjp_fd` (the length-in-H causal conv1d, checking every input
  position receives gradient). All pass; they were written to *localize* a
  gradient discrepancy and ended up exonerating each op individually.
- **Output-position stability is now asserted.** `grad_with_loss_wrt` checks that
  `prepare_graph_for_ad` preserved the output count, and
  `tests/prepare_output_stability.rs` checks that each `outputs[i]` still holds
  the same *value* across prepare. The invariant held before but was emergent —
  a pass that reordered or dropped an output would have made `Wrt::Output(i)`
  differentiate the wrong tensor with matching shapes and no error.

### Fixed

- **Metal MPS matmul crashed when two threads encoded the same shape.** The
  `(m, k, n, transposeA, transposeB)` kernel cache handed the *same*
  `MPSMatrixMultiplication` to every caller, and that object mutates itself
  while encoding (`setIndexingArithmaticTypeMask:sourceArrays:…`). Two threads
  encoding concurrently raced inside MPS and took `EXC_BAD_ACCESS` in
  `MPSNDArrayMultiaryBase` — an Apple frame, with nothing in rlx's own stack
  looking wrong, which is why the existing `CACHE_GUARD` comment reads as if the
  case were covered. It was not: that `RwLock` keeps a cached pointer *alive*
  across an encode (against `invalidate_caches` freeing it), which is a lifetime
  guarantee, not an exclusivity one.

  The cache is now keyed by thread as well as shape, so concurrent encodes still
  overlap — the property the `RwLock` design was chosen for, where a mutex around
  the encode would have serialized them — and the hit rate for the case that
  motivated the cache (recurring shapes on one encode thread) is unchanged. The
  map stays global rather than becoming `thread_local!` so `invalidate_caches`
  can still release every live kernel; entries for finished threads linger until
  the next invalidate, which runs on every compile and on
  `MetalExecutable::drop`.

  Found because `metal_q2_0_fused_decode_parity`'s three tests run in parallel:
  5/5 SIGSEGV before, 0/5 after. It is deterministic with ≥2 threads on one
  shape, and is *not* the intermittent MPSGraph compile crash — it reproduces
  with `RLX_DISABLE_MPSGRAPH_EXECUTABLE=1`, and the faulting frame is
  `MPSNDArrayMultiaryKernel`, not `MPSGraphExecutable`.

  Regression test: `rlx-metal/tests/mps_matmul_concurrency.rs` — half the threads
  hammer one shared shape (one cache key, maximal contention), half insert
  distinct keys concurrently, and every result is checked against a CPU
  reference rather than only checked for survival, since a corrupted kernel can
  return a plausible wrong answer instead of a signal. It SIGSEGVs with the
  `ThreadId` removed from the key, and passed 15/15 with it.

- **webgl read the wrong row of the RoPE cos/sin tables under partial
  rotation.** It derived the table row stride as `n_rot/2` while the rest of the
  stack moved to taking it from the table's own last dimension. The two differ
  exactly when `n_rot < head_dim`, and the layout is a per-model choice — Qwen3.5
  allocates `[max_pos, head_dim/2]` and uses the leading `n_rot/2` columns of
  each row, DeepSeek-V4 MLA packs `[.., n_rot/2]` tight — so *any* derived stride
  is wrong for one of them. Every position past the first picked up another
  token's angles. Caught by `rlx-webgl/tests/ops2_parity.rs::rope_full_and_partial`.

- **`Op::GatedDeltaNetBackward` was unreachable on five of the eight backends
  that run the forward.** CPU, Metal and MLX have the fused kernel; CUDA, ROCm,
  wgpu, TPU and CoreML run `Op::GatedDeltaNet` but not its backward — and the
  gradient walk emits the backward op by default. The only escape was setting
  `RLX_GDN_UNFUSE_FOR_AD=1`, a global env flag answering a per-backend question,
  and it has to be set *upstream of autodiff*, so a caller who reaches the error
  has already built the wrong graph.

  It now decomposes like every other fused backward op, through
  `decompose_backward_ops_except` in `rewrite_for_backend` — automatically, and
  only for a backend that does not claim the kind. The decomposition does not
  re-derive the reverse scan by hand: it rebuilds the forward from the backward
  node's own inputs, unrolls it and differentiates *that*, so it is the
  `RLX_GDN_UNFUSE_FOR_AD` path reconstructed at the backend boundary rather than
  chosen globally beforehand — and it agrees with the fused kernel by
  construction, since the unrolled path is what that kernel is already pinned
  against. Measured agreement **≤ 6e-8** across both gate modes and both state
  modes (`rlx-autodiff/tests/gated_delta_net_backward_decompose.rs`), and
  **2.2e-8** through the real `rewrite_for_backend` path
  (`rlx-runtime/tests/gdn_backward_backend_fallback.rs`). A backend listing the
  kind in `supported_ops` still keeps the fused kernel; the decomposition is
  ~32× slower and is the fallback, not the default.

  `gpu_family_supports` also stopped answering `true` for everything on the
  CUDA/ROCm/wgpu family and now consults the per-backend `SUPPORTED_OPS`, so a
  caller gets the chance to pick a fallback instead of a compile-time
  unsupported-op error.

- **`jvp` re-declared a tangent input the graph already had, so `jvp(hvp(f))`
  returned zero instead of the third derivative.** `jvp` mirrors the forward
  graph verbatim — including its leaves — then adds an `Op::Input` named
  `tangent_<name>`. `hvp` *is* a `jvp`, so its result always already has that
  name, and forward-over-reverse-over-reverse collided every time. Nothing
  errored: binding is by name and reaches a single node, so the second
  declaration was never bound, read zeros, and the derivative came back exactly
  zero. This had been read as the composition being unsupported and was
  documented as such ("the outer `jvp` graph is still not AD-ready for another
  pass"); the cause was narrower than that.

  The outer tangent is now `tangent_<name>_2` (first free suffix), and
  `jvp_with_tangent_names` returns the names actually used rather than making
  callers guess either spelling. `jvp(hvp(f))` now yields the correct third
  derivative — verified against `∂(H·v)/∂x·w = 24xvw` on `Σxᵢ⁴`, which also
  pins the Hessian-vector product at the same time
  (`rlx-runtime/tests/jvp_over_hvp_third_order.rs`).

- **`Q2_0` was registered on wgpu and CUDA with no kernel branch — weights read
  as zeros.** `dequant_gguf.{msl,cu,wgsl}` are `if (scheme_id == N) { … return; }`
  chains with no default branch, and schemes reach them through the shared
  `define_gguf_gpu_dequant_ids!` table. An id in that table with no matching
  branch does not raise: the kernel writes nothing. `tests/pestle_schemes_all_backends.rs`
  now requires every backend to match the CPU reference for both schemes the
  Pestle-27B model needs.

- **Host steps that rewrite the whole arena left `HostTensorCache` stale.**
  Host fallbacks come in two families: *cache-aware* steps (`HostOp`,
  `Conv2dHost`, `ExpandHost`, `NarrowHost`, `TransposeHost`, `ConcatHost`,
  `BufferCopy`) that read and write through the host mirror, and *whole-arena*
  steps (everything routed via `rlx_gpu_host::with_whole_arena` — `GroupNormHost`,
  `LayerNorm2dHost`, `ReverseHost`, `GruHost`, `RnnHost`, `MsDeformAttnHost`, …)
  that read the device and rewrite the arena directly. The second kind got the
  mirror flushed *before* it ran but nothing invalidated it *after*, so a
  following cache-aware step could serve a pre-step copy of a region the
  whole-arena step had just overwritten. `host_cache.clear()` is only reached
  after real GPU work (`pass_dispatched`), and a run of consecutive host steps
  never gets there. It needed a dead tensor whose arena slot the whole-arena
  step reused, plus both families adjacent in one schedule — which in practice
  meant discrete NVIDIA, where elementwise, conv and norms are all hosted.
  Regression test: `tests/host_stage_cache_parity.rs` (uses
  `RLX_WGPU_FORCE_HOST=1`, so it reproduces on any adapter).

- **wgpu `Op::GroupNorm` disagreed with every other backend once the norm had a
  consumer.** The op lowered to `Step::GroupNormHost`, which mirrors the *whole
  arena* to the CPU and back. Read in isolation the result looked right — an
  isolated `GroupNorm` matched CPU to 1e-7 at every group count and shape — but
  a truncation sweep over a 225-layer MobileNet put the first divergence
  immediately after a norm (34 of 96 hash bits, max|Δ| 0.15), i.e. the staged
  writeback was not what the next kernel read. Now lowered to a native WGSL
  kernel, so the default route no longer stages at all. The staging fault
  itself is fixed separately (below), so `RLX_WGPU_HOST_NORM=1` and
  virtually-sharded arenas are correct too.
- **Fusion passes could delete a node that is a graph output.** `has_single_use`
  — documented as "the precondition almost every fusion pattern checks before
  absorbing a producer into its consumer" — counted only consuming *nodes*. A
  value that is consumed once *and* exported as a graph output looked absorbable,
  so the pass skipped it during the rewrite without recording a replacement and
  `Rewriter::finish` panicked mapping the outputs (`no entry found for key`).

  `has_single_use` now also requires that the node is not a graph output, and
  `UseCounts::is_graph_output` exposes the test. `FuseSwiGLU` and
  `FuseSwiGLUDualMatmul` were switched onto it; the latter also checked only one
  of the three nodes it absorbs, so the two matmuls could be dropped even with
  another consumer. Several passes (`ada_layer_norm`, `gated_residual`,
  `rmsnorm_reshape`, `attention_block`, `residual_ln`, `residual_rmsnorm`)
  already guarded this explicitly — the hazard was known, just applied
  inconsistently.

  Ordinary model graphs rarely export an interior value, which is why this went
  unnoticed. Any graph that publishes activations as outputs hits it at once:
  `split_vjp`'s save half, or an instrumentation tap.

- **A fusion pass could silently zero gradient terms.** `Rewriter::copy_node`
  re-copied nodes that `ensure_mapped` had already hoisted to a fusion site,
  emitting a second node for one original and repointing `id_map` at it. For
  `Op::Param` that is silent corruption rather than wasted work: binding is by
  name and reaches a single node, so the duplicate kept the arena's zeros and
  every value flowing through it vanished — no shape error, no missing-input
  error, just a wrong answer.

  A **forward** graph never showed it, because the fused-away consumer was the
  weight's only reader and the duplicate was dead code. A **backward** graph
  did: `grad_with_loss` mirrors the forward alongside the gradient ops, so a
  weight is read twice — once by the mirrored matmul, once by `dX = dY · Wᵀ` —
  and the duplicate was live. Any backward graph whose forward had several
  matmuls sharing an input (a fused QKV or in-projection — i.e. most decoders)
  lost the gradient terms through the hoisted weights.

  Found in a Qwen3.5 gated-delta-net block, where it erased *all* cross-position
  gradient while leaving the same-position gradient exactly right. `copy_node`
  is now idempotent. `Rewriter` backs ten fusion passes, so the fix is not
  specific to `FuseSharedInputMatMul`. Regression tests:
  `rlx-fusion/tests/rewriter_no_duplicate_leaves.rs` (structural) and
  `rlx-autodiff/tests/fused_shared_input_matmul_grad.rs` (gradients with fusion
  on vs off); both fail without the fix.

### Performance

- **Metal simdgroup decode GEMVs for the remaining hot K-quants.** Decode is
  weight-streaming — every token reads the whole model once, so a GEMV kernel's
  achieved GB/s *is* the token rate. Q5_K was the last hot K-quant still on the
  one-thread-per-row kernel, which is occupancy-starved: 90 GB/s at n=17408 and
  19 GB/s at n=1024, against ~200 / ~120 for Q4_K and Q6_K. `q5k_mv_f32_sg` has
  32 threads cooperate per output row via `simd_sum`; Q3_K and Q4_1 get the same
  treatment. Both kernels dequantize identically and differ only in summation
  order, so the parity test uses a relative tolerance rather than equality.
  Per-kernel off-switches (`RLX_METAL_Q5K_SG_DISABLE`, `_Q3K_`, `_Q41_`) keep the
  scalar path reachable for A/B.

  All four Q2_0 GEMVs now share one 16-bit `q2_0_dot16` inner loop, which is why
  `metal_q2_0_fused_decode_parity.rs` exists: `q2_0_dual_mv_f32_sg`,
  `q2_0_swiglu_mv_f32_sg` and `q2_0_mv_residual_f32_sg` are reached only by
  pattern-fusion of a decode MLP and had no test at all, so a mistake in the
  shared loop would have shown up only in `q2_0_mv_f32_sg`. The reference is
  computed on the host in exact f32.

  `tests/gemv_bandwidth.rs` measures achieved GB/s per kernel directly at a 27B
  FFN shape — a whole-model benchmark cannot, because at model scale the number
  is confounded by paging and on a small model everything is launch-bound.

- **Metal fused ggml `L2_NORM`.** Collapses the
  `mul → sum(last) → sqrt → max(·, eps) → div` chain `rlx_qwen35`'s `l2_norm`
  emits into a single `L2NormLastDim` dispatch. Gated-DeltaNet runs it twice per
  linear layer, so it is 36×/token on a Qwen3.5 block. Off-switch
  `RLX_METAL_FUSE_L2NORM=0`, which the parity test uses to run fused, unfused and
  CPU against each other.

- **CUDA warp-per-row Q4_K decode GEMV (`RLX_CUDA_Q4K_GEMV_WARP=1`, opt-in).**
  One warp per output row with lanes splitting each super-block's 256 elements,
  8 warps per block. Full occupancy at small `k`, unlike the block-per-row coop
  kernel which idles all but `k/256` lanes.

- **wgpu `Op::GroupNorm` — 88× on a 35-norm MobileNet (1096 ms → 12.4 ms per
  forward).** The host fallback moved ~83 MB each way per norm; the new
  `group_norm.wgsl` runs one workgroup per `(batch, group)` with a
  shared-memory tree reduction and the same stable two-pass variance as
  `layernorm.wgsl`. Covers `num_groups == C` (instance norm), `num_groups == 1`
  and the general case.

- **Metal `Op::GatedDeltaNetBackward` — 5.7× over CPU on a real block.** The op
  had a CPU kernel only, so every other backend fell back to unrolling the time
  loop. Metal now has its own.

  The kernel is **threadgroup-cooperative**: one threadgroup per `(batch, head)`
  with `state_size` threads inside it. A first version used one *thread* per
  `(batch, head)` — the same shape as the forward's default kernel — and was
  **3.7× slower than CPU**, because that is only `batch·heads` threads (256 at
  the lens's batch) and leaves the GPU almost entirely idle. That is the lesson
  the forward already learned with `gated_delta_net_sg`. Making it cooperative
  was **19.6×** on its own (replay 31.2 s → 1.59 s over 64 passes).

  Phases alternate between row-parallel and column-parallel over the state, so
  every reduction stays thread-local — a row phase walks `ds[tid·n + j]`
  contiguously, a column phase walks `ds[i·n + tid]` so neighbouring threads
  touch neighbouring addresses. Only dβ and the per-head dg need a cross-thread
  sum.

  On a Qwen3.5-0.8B gated-delta-net block (1024×1024 Jacobian, 64 cotangents):
  **8.5 s on CPU → 1.5 s on Metal**; an attention block goes 1.3 s → 0.3 s.
  Verified against the CPU kernel — itself finite-difference-verified — for both
  gate modes and with a carried state, agreeing to **6e-8**
  (`rlx-metal/tests/gated_delta_net_backward_parity.rs`).

  The state history costs `(seq + 2) · n²` floats per `(batch, head)` of
  ephemeral scratch, sized by `gdn_ephemeral_state_bytes` alongside the
  forward's. Reconstructing states backwards instead, by dividing out
  `exp(g) < 1`, would remove that but amplifies rounding without bound.

- **`rlx_autodiff::split_vjp` — run the forward once, replay the gradient per
  cotangent.** `grad_with_loss` mirrors the whole forward into the backward
  graph so gradient kernels can recompute activations. That is right for
  training and wrong for anything sweeping many cotangents over one forward —
  a Jacobian taken a block of output dimensions at a time, per-sample
  gradients, influence functions — where the forward is recomputed once per
  cotangent. PyTorch's `retain_graph` covers that case; a graph has no tape to
  retain.

  `split_vjp` cuts the backward graph in two. The cut is structural, needing no
  cooperation from the gradient walk: a node belongs to the gradient half
  exactly when it is reachable from `d_output`. Saved activations cross as
  `Op::Param`, not `Op::Input` — parameters are bound once and persist, so `N`
  replays cost one bind rather than `N` feeds of the same bytes. Forward leaves
  the gradient half reads are copied rather than round-tripped through the host.

  Measured on a Qwen3.5-0.8B attention block (1024×1024 Jacobian, 64 cotangents
  over one forward): **2.7 s → 1.3 s**. Roughly neutral on a gated-delta-net
  block, where the fused backward had already made the forward cheap relative to
  the gradient work — 151 ms of forward against 8.3 s of replay.

- **`Op::GatedDeltaNet` now has a dedicated backward — ~5.6× on a real
  gated-delta-net block, and the op alone was ~32× off.** Autodiff previously
  unfused the op, unrolling its time loop into per-timestep primitives so the
  gradient walk could reach their existing VJPs. At Qwen3.5-0.8B shapes
  (`B=16 S=24 H=16 N=128`) that is 585 nodes running **254.84 ms** against
  **8.04 ms** for the fused kernel, same checksum. The cost is structural rather
  than dispatch overhead: in SSA form every timestep materializes a fresh
  `[B·H, N, N]` state — 16.8 MB here — while the kernel updates one working set
  in place.

  New `Op::GatedDeltaNetBackward` computes every input gradient from one reverse
  scan, with a CPU kernel and a VJP rule that slices its packed output back into
  per-input gradients (`rlx_ir::GdnBackwardLayout` owns the packing, shared by
  the rule and the kernel so they cannot drift). `unfuse_fused_for_autodiff`
  leaves the op fused by default; `RLX_GDN_UNFUSE_FOR_AD=1` restores the unrolled
  decomposition for a backend that runs the forward but not yet the backward.

  Measured on a Qwen3.5-0.8B gated-delta-net block: a 1024×1024 per-block
  Jacobian went **47.0 s → 8.4 s**, closing the gap to an attention block from
  19× to 3.1×, with the fitted Jacobian unchanged to the last reported digit.

  Verified three ways: the kernel against finite differences for both gate modes
  (worst 8.7e-6, `rlx-cpu` `gdn::backward_tests`); the fused path against the
  unrolled one it replaces, which is itself finite-difference-verified (worst
  **4.5e-8**, `rlx-autodiff/tests/gated_delta_net_fused_backward.rs`); and
  end-to-end through `gated_delta_net_vjp_fd.rs`, including the cross-position
  recurrence and causality.

- **Smaller `GatedDeltaNet` unfusing.** The unrolled fallback held its state as
  `[BH, N, N]` throughout instead of round-tripping to `[B, H, N, N]` six times
  per timestep, and broadcasts `exp(g)` / `beta` / the readout scale rather than
  expanding them to state size first — which also moves `exp` off a state-sized
  tensor. Backward-graph tensor output drops 12.87 GB → 6.82 GB at the shapes
  above (`Reshape` 4976 → 141 MB), numerics unchanged. Note this did **not**
  change wall-clock on CPU, where those reshapes were free views; it is a
  smaller graph, not a faster one.

## [0.2.14] — 2026-08-11

### Changed

- **`rlx-metal` no longer depends on `metal-rs`** — it binds the Metal compute
  API directly, in `rlx_metal::mtl`. The `metal` crate pins `block 0.1.6`, whose
  `static` of an uninhabited type trips the `static of uninhabited type`
  future-incompatibility lint (an error in a future rustc); `metal-rs` still pins
  it at 0.33 and `block` is unmaintained, so no bump cleared it. This backend
  installs no Metal completion handlers, so nothing in `block` was ever used.
  Dropping `metal-rs` also drops `foreign-types` and `core-graphics-types`;
  `objc` stays. Scope is our surface only — compute, no render pipeline, heaps,
  events, or fences.

  **Downstream `MetalGpuKernel` authors**: the custom-op seam still hands you a
  live encoder and buffer, but the types now come from `rlx_metal::mtl` rather
  than `metal`. Names, signatures, and the `Foo`/`FooRef` ownership split are
  unchanged, so `use rlx_metal::mtl as metal;` is normally the whole migration.
  The `Ref` types still `impl objc::Message`, so raw `msg_send!` against them
  keeps working.

### Added

- **GPU kernel argument-count validation (CUDA + ROCm).** `cuLaunchKernel` and
  `hipModuleLaunchKernel` read exactly as many pointers as the compiled kernel
  declares and never see how many the caller supplied: too few reads past the
  end of the argument array, too many silently drops the tail. Neither is a
  compile error and neither reliably faults. This repo has already shipped that
  bug — `gguf_gpu::launch_dequant_gguf` passed 5 arguments to a 6-parameter
  kernel, misreading the 64-bit arena offsets, so a >4 GB arena overflowed u32
  into SIGSEGV or garbage.
  - `rlx_gpu_kernels::declared_param_count` parses arity from the shared `.cu`
    sources. It strips comments first: real signatures document each parameter
    inline, and those comments carry commas and unbalanced parens (`// [batch,n,n]`,
    `[λ(n) ∥ U(n²)]`) that a naive scan counts, which reported 9 for the
    5-parameter `eigh_assemble` on the ROCm rig.
  - **ROCm** checks at launch: `HipKernel::launch_checked` takes a *slice*, so
    the count exists at all — the old `*mut *mut c_void` could not express it —
    and compares it against the kernel's signature under
    `RLX_GPU_VALIDATE_PARAMS=1` (now on in `just test-rocm`). Verified on an
    MI100: 48/48 with the check enabled, no mismatches.
  - **CUDA** goes through cudarc's typed builder across ~157 sites with no
    single place to count, so it is checked statically by
    `tests/launch_arity.rs`, which resolves each launcher back to its kernel and
    compares `.arg()` count against the `__global__` signature: **151 of 156
    sites, 5 unresolved and reported rather than silently passed.** Verified by
    mutation — deleting one `.arg()` is caught. Green on an RTX 3080 Ti.
- **Metal buffer-binding validation (`RLX_METAL_VALIDATE_BINDINGS=1`,
  `just validate-metal-bindings`).** The backend binds buffers by integer index
  against MSL signatures in `kernels.rs`, across ~680 call sites, with nothing
  connecting the two — the defect class that made the ICB path write nothing for
  months. Each dispatch is now cross-checked against the indices its kernel
  declares. Gated by env rather than `debug_assertions` deliberately: the
  workspace gate runs `--release`, so a `cfg(debug_assertions)` check would be
  compiled out of the one place it needs to run. Off by default (a relaxed
  atomic load). Declared indices are parsed from the MSL we compile rather than
  taken from Metal's pipeline reflection — requesting reflection
  (`MTLPipelineOptionArgumentInfo`) aborts unrelated encoders. Violations are
  raised at `endEncoding`, not at the dispatch: unwinding out of an open encoder
  trips Metal's own dealloc assertion, which aborts the process and destroys the
  message explaining the bug.
- **`mtl::autoreleasepool`, one pool per forward pass.** Command buffers and
  encoders are autoreleased, so with no pool boundary a decode loop accumulated
  one of each per token for the process lifetime. Correct refcounting does not
  help; only a pool does.
- **`just leak-check`** — runs Metal test binaries under `leaks --atExit`. The
  suite cannot see a refcount bug, since an over-retained object still computes
  the right answer; this is how the `MPSGraphTensorData` leak below was found.
- **MSL gate tests** — the assembled source must compile and every kernel it
  declares must resolve by name (previously only checked when a pipeline was
  first built, so a bad kernel shipped and failed on first use), plus an
  assertion that the kernels `icb.rs` encodes keep their buffer-0 ABI.
- **`decode_overhead_bench`** — steady-state cost of a decode-shaped pass. Every
  test runs its graph once or twice, so the only `RLX_METAL_TRACE` samples
  available carried one-time MSL compilation (~790 ms) and said nothing about
  steady state.

### Performance

- **Measured: a decode-shaped Metal pass is 96% wait.** For `m=1, k=n=768`,
  200 warm iterations: **encode 6.9 µs, commit 2.0 µs, wait 178.2 µs** (total
  ~185 µs/iter, p50 159 µs). This confirms the non-monotonic `sgemm_check`
  curve — `m=6` at 0.374 ms versus `m=60` at 0.278 ms — as a fixed per-pass
  overhead rather than compute, and bounds what encode-side work can buy:
  eliminating encoding entirely would save under 4%. Submission batching, not
  kernel tuning, is where the decode regime improves.
- **MPSGraph compile-crash mitigation verified.** The
  `waitForCompilationCompletion` descriptor added earlier was never measured,
  and at the reported ~2% rate a short run proves nothing (P(0 crashes in 20)
  ≈ 0.67). A 200-process soak (`scripts/mpsgraph-soak.sh`) recorded **0
  crashes**, putting a 95% upper bound of ~1.5% on the rate — below the 2%
  previously observed.

### Fixed

- **wgpu-on-Vulkan wrong results: deferred host writes vs. arena slot reuse.**
  On a discrete Vulkan/DX12 adapter `rlx-wgpu` lowers Expand / Concat /
  Transpose / Narrow to *host* steps (`wgpu_prefer_structure_host`), whose
  outputs "live only in the mirror until a device-reading host step or GPU pass
  needs them" — the arena write happens at a later flush. The memory planner
  treated them as ordinary single-step consumers, so liveness-aware reuse handed
  the slot to another tensor and the deferred write landed on top of it. Those
  slots are now kept reserved; `rlx-wgpu` declares that it hosts them via
  `rlx_compile::memory::set_pin_host_structure`.

  This is the same hazard `dequant_host_fallback` already guarded for
  `Op::DequantMatMul` (task #50, exact-zero downstream values) — it had simply
  never been generalised to the other host-deferred ops.

  It presented as a family of unrelated Vulkan-only bugs (`pad`, PartitionedConv
  `conv_reverb`, layer/group-norm second derivatives, `wgpu::all`), because the
  affected graphs are the structural-op-heavy ones; it was nondeterministic,
  since it depends on when the flush falls; and it was Vulkan-only because Metal
  keeps these ops on the GPU. No effect on Metal/CUDA/ROCm, which do not host
  these ops. Ruled out along the way, for whoever revisits: reuse slack (1 and 2
  steps), region-fusion decomposition, elementwise-region fusion, and the host
  concat path — none change the outcome; only reservation does. Regression test:
  `rlx-runtime/tests/expand_vulkan_repro.rs`.

- **wgpu host-tensor cache truncated a parent buffer when a view wrote over it.**
  `HostTensorCache` is keyed by byte offset, but *views alias their parent's
  slot* — `Narrow(start=0)` of a concat carries the parent's exact offset. A
  shorter view write replaced the entry outright, discarding the parent's mirror
  beyond the view's length. A sibling view at an interior offset
  (`Narrow(start=6)`, parent + 144 B) then missed `get_arc_covering` (exact-key),
  found nothing dirty for `flush_offset` to flush (also exact-key), and fell
  through to a D2H read of a device region the deferred parent had never
  written — returning **zeros**, which propagated until the graph output was
  entirely zero.

  Diagnosed by tracing host writes: the concat computed correctly
  (`off=4896 len=72 nonzero=42`) and was immediately followed by
  `off=4896 len=36`, the view clobbering it. `insert` now merges a shorter write
  into a longer entry instead of replacing it — the arena bytes past the write
  are unchanged, so the mirror must keep showing them.

  Fixes the remaining four: `logeig_lowering_wgpu`, `reeig_lowering_wgpu`,
  `wgpu::biquad`, `wgpu::iirfilt`. Together with the pinning fix above, the
  wgpu-on-Vulkan suite goes from **10 failures to 0** on an RTX 3080 Ti
  (`rlx-runtime --features cpu,cuda,gpu`: 945/10 → 958/0). Unit guard:
  `rlx_gpu_host::scan::host_cache_tests`.

- **Metal ICB path produced silently wrong results; now correct and covered by
  a test.** `rlx-metal`'s indirect-command-buffer encoder (opt-in,
  `RLX_USE_ICB=1`) binds kernel buffers *by index* against MSL signatures that
  live in `kernels.rs` — and those signatures had moved. Several kernels went
  from taking a `device float*` already offset to the data, to taking the arena
  base plus explicit `ulong` byte offsets, which shifts every later index. The
  failure is invisible: the unbound index reads 0, `len` comes back 0, every
  thread hits `if (gid >= len) return`, and the command buffer completes with no
  error having written nothing. `icb_check` reported all-zero output.
  - `elem_add`/`elem_mul` and `copy_f32`: rebound to the arena-base form
    (`arena, ulong a_off, ulong b_off, ulong c_off, uint len`).
  - `gelu_inplace`: is *generated* (arena-base ABI) while `silu_inplace` is
    hand-written (offset-pointer ABI) — this arm now binds per activation
    instead of assuming one layout for both.
  - `narrow_lastax`: its `src_byte_off`/`dst_byte_off` were left unbound; with
    `inherit_buffers=false` that reads undefined data.
  - `rope` binds 13 buffers but the descriptor declared
    `maxKernelBufferBindCount = 8`, silently dropping the rest.
  - `layer_norm`/`fused_residual_ln` dispatched rows along **y**, but both
    kernels read a *scalar* `threadgroup_position_in_grid` (the x component),
    so every row recomputed row 0; threadgroup width is now a power of two, as
    the reduction requires.
  New `icb_parity` test asserts ICB output against a CPU reference — and
  specifically rejects an all-zero result, which is the signature of an index
  mismatch and would otherwise slip past a relative-error check.
- **MPSGraph tensor-data leak.** `mps_tensor_data_from_buffer` returns a `+1`
  `MPSGraphTensorData`; the eight call sites handed it to an `NSMutableArray` /
  `NSMutableDictionary` (which retains) and then dropped their own reference on
  the floor, so every one bottomed out at refcount 1 and was never freed. Found
  with `leaks` on `metal_swiglu_full_parity` (10 leaks / 4160 bytes → 0). It
  leaked per executable bind rather than per inference, so it did not grow
  during decode.
- **A contradictory output shape is now rejected in release builds too.**
  `sync_graph_shapes` recomputes inferrable output shapes, but it *replaced* the
  declared shape unconditionally — including when the declaration was fully
  static and disagreed with what the operands give. Since every backend's
  lowering validates the declared shape against the operands, rewriting the
  declaration first made that check compare a value with itself, so the only
  thing catching a malformed graph was the debug-only IR verifier: release
  builds — the ones that ship — had no guard at all. This is how a
  `GroupedMatMul` expert bank left in `[E, N, K]` order slipped through (right
  rank, right element count, wrong axes), with the kernel writing `M·K` floats
  into the `M·N` slot the planner had sized. It now refines freely where the
  declaration was dynamic, and rejects a static-vs-static contradiction,
  comparing only non-unit extents so the interchangeable `[]` / `[1]` scalar
  spellings still pass. Regression test:
  `square_bank_with_a_wrong_output_is_rejected`.
- **Compiler panic-isolation (no more process aborts on an invalid graph).**
  `CompilePipeline::lower_hir` previously `panic!`-ed via the debug graph verifier
  (`debug_assert_graph!`); on the parallel compile threads that model builds use,
  that panic can't unwind (`__rust_start_panic` fails → hard `SIGABRT`), taking
  the whole host process down. It now returns the verifier failure as
  `LowerError::Panicked` instead, and `compile_hir`/`lower_hir` wrap the pipeline
  in `catch_unwind` to contain any other (unwindable) compile panic.
- **IR verifier accepts collapsible affine norm params.** `LayerNorm` / `RmsNorm`
  `gamma`/`beta` with rank > 1 whose element count collapses to the normalised
  width (e.g. `[1,1,C]`, as some whisper / TTS graphs build) are now accepted —
  it's the same flat `[C]` buffer the norm kernels read — rather than rejected as
  "must be rank-1". A genuinely wrong element count is still flagged.
- **`rlx-cpu` linalg no longer panics without a linked BLAS.** On a no-BLAS build
  (`--no-default-features`, or a target with no OpenBLAS/Accelerate/MKL — a bare
  aarch64 / Raspberry Pi, wasm, …) the LAPACK ops used to be panic-stubs, so
  Cholesky / eigh / QR / SVD / least-squares (and the `dtrsm` triangular solve)
  aborted at runtime. They now have dependency-free pure-Rust fallbacks —
  Cholesky (Banachiewicz), LU (partial-pivot), symmetric eig (cyclic Jacobi), QR
  (Householder + `dorg2r`), thin SVD (one-sided Jacobi), and lstsq built on them —
  matching the column-major LAPACK ABI the wrappers expect, so the row-major
  linalg wrappers are backend-agnostic. Validated against the linked-BLAS path:
  the full `rlx-cpu` lib suite passes identically with and without BLAS (new
  reconstruction tests in `blas::linalg_fallback_tests`). The linked-BLAS path is
  untouched and remains the fast default where present.

### Performance

- **aarch64 `SDOT` fast path for the fused int8 Q4_K decode GEMV.** The NEON
  Q4_K×Q8 group dot used a baseline `vmull_s8` + `vpadalq_s16` widen-multiply
  chain; on ARMv8.2-A (Cortex-A76 / Raspberry Pi 5, Apple Silicon, Graviton2+,
  most ARM servers) it now emits a single `SDOT` per 16 lanes — runtime-detected
  (`dotprod`), Pi-4-safe (falls back to the baseline), opt-out via
  `RLX_Q4K_NO_DOTPROD=1`. The intrinsic `vdotq_s32` is still unstable on the
  pinned stable toolchain, so `SDOT` is emitted with inline `asm!`. It's a pure
  integer dot, so the output is byte-identical to the baseline (asserted by a new
  test). Measured on the fused decode path (Qwen3.5-0.8B Q4_K_M, aarch64):
  **~+12% decode tok/s and faster prefill/TTFT** (prefill gains more — it runs
  the dot over the whole prompt).
- **`rlx-cpu` no-BLAS `sgemm` fallback ~8–10× faster.** The BLAS-less f32 GEMM
  (used on a bare aarch64 / Raspberry Pi, wasm, or `--no-default-features` — where
  no OpenBLAS/Accelerate/MKL is linked) was a naive `i,j,p` triple loop whose
  inner reduction strides through B by `ldb`, defeating both vectorization and the
  cache. The common NoTrans×NoTrans case now uses an `i,p,j` order with a
  unit-stride inner loop, which the compiler auto-vectorizes (NEON/AVX) and keeps
  cache-resident. Measured on a 256³ GEMM: **2.8 → 23 GFLOP/s (8.2×) on aarch64**,
  3.2 → 31 GFLOP/s (9.8×) on Apple Silicon. Transposed cases keep the scalar path;
  results match the naive reference within tolerance (the summation reorder is the
  same class of reordering the vendor BLAS path already applies).

### Added

- **`RLX_BLAS_LINK` — link any cblas+LAPACK provider (BLIS / ATLAS / vendored /
  cross).** `rlx-cpu`'s BLAS auto-detection is OpenBLAS-specific (it's the one
  distro library that ships both a CBLAS interface and LAPACK in a single `.so`).
  For anything else — or to link a BLAS into a cross build — set
  `RLX_BLAS_LINK="<libs>"` (space-separated names, e.g. `"blis lapack"`, searched
  in `RLX_BLAS_SEARCH` / `OPENBLAS_LIB_DIR`). It sets only the umbrella
  `rlx_cpu_blas` cfg (not the OpenBLAS thread-control sub-cfg), so no
  vendor-specific symbol is referenced and the alternative BLAS self-manages
  threads. The OpenBLAS auto-probe also now searches the Debian `openblas-*`
  variant subdirs. With none of these, the crate still runs on the pure-Rust
  fallback. Verified on aarch64: `RLX_BLAS_LINK=openblas` links + passes the
  `rlx-cpu` linalg suite (10/10) through the generic path.
- **CUDA opt-in warp-per-row Q4_K decode GEMV** (`RLX_CUDA_Q4K_GEMV_WARP=1`):
  one warp per output row, lanes split each super-block for full occupancy at
  small `k`; byte-identical output to the scalar path.

- **MoE hot-on-GPU / cold-on-CPU expert offload (`rlx_runtime::hot_expert_cache`).**
  Runs a MoE layer whose expert stack does not fit in VRAM by keeping only the
  *hot* experts device-resident. `HotExpertCache` models one layer as `num_slots`
  device slots drawn from `num_experts`, tracks which expert occupies which slot,
  and emits the minimal host→device copies (`SlotLoad`) to match the pool's
  resident set; `SlotRoute` splits per-token routing into device slots plus a
  `(token, expert)` cold list. `reconcile_layers` drives a whole model. The cold
  half runs on the host via `rlx_cpu::moe_split::cold_grouped_matmul`, shared by
  CUDA and ROCm through `rlx_gpu_host::moe`. Because MoE residency is *per
  expert* — never per token — every output row is still produced by exactly one
  sgemm over one expert's weights, so the split is **byte-identical** to the
  full-stack grouped matmul; `grouped_matmul_split_reference` makes that testable
  on CPU, and `moe_hot_cold_parity` covers CUDA + ROCm.

- **Expert-placement hysteresis (`rlx_runtime::HysteresisConfig`).** `ExpertPool`
  gained two gates that damp slot thrash when expert popularity is near-tied:
  `margin` (a challenger must be a given *fraction* hotter than the incumbent
  before it may evict it) and `min_dwell` (an incumbent is held for a minimum
  number of refreshes). Both default to **off**, so existing callers keep the
  plain top-S paired-swap behavior. Adds `refresh_from_counts`, `resident_since`,
  and `with_hysteresis`.

- **`Q2_0` int8 dot path + x86-64 VNNI kernels (`rlx_cpu::intrinsics::vnni`).**
  The 2-bit `Q2_0` format (ggml type 42) gained a packed int8 activation block
  (`Q8_0_G128_BYTES` = f32 scale · 128×i8 · i32 group sum) plus
  `quantize_q8_0_g128_row`, so low-bit GEMV keeps weights in their packed byte
  form instead of materializing an f32 slab. The x86 kernels run `VPDPBUSD`
  directly on the 2-bit codes, selected at runtime between AVX-512-VNNI+VL
  (`_mm256_dpbusd_epi32`) and AVX-VNNI (`_mm256_dpbusd_avx_epi32`); the stored
  group sum folds the `−1` weight offset into one subtract so unsigned codes feed
  the instruction directly. Neither encoding saturates, so results are
  bit-identical to the scalar reference.

- **Cooperative Q4_K GEMV kernels (CUDA/HIP + Metal).**
  `dequant_matmul_gguf_q4k_gemv` assigns one block per output row with threads
  splitting the k/256 super-blocks, so loads are coalesced (unlike the
  one-thread-per-row kernel whose lanes each stride a whole row) and dequant is
  fused into the dot with no local `float[256]` slab to spill. Accumulation is
  Neumaier-compensated and tree-reduced. Metal gains the simdgroup-cooperative
  equivalent (`q4k_mv_f32_sg`) plus fused-epilogue variants that reuse the
  identical block math for down-proj+residual and SwiGLU.

- **Fused QKV projection (`FlowCtx::resolve_linear_fused`).** Concatenates
  several packed weights that share an input dim and quant scheme into one
  `DequantMatMul`, then `narrow_`-splits the result — one GEMV dispatch instead
  of N. GGUF blobs are `[out, in]` row-major, so the fused weight is just their
  bytes concatenated. Returns `None` on non-packed weights *without* consuming
  them, so callers fall back to per-key `resolve_linear` safely. Opt-in for
  qwen3 decode via `RLX_QWEN3_FUSED_QKV` (2 fewer dispatches per layer).

- **Opt-in fused Q4_K int8 decode GEMV on CPU (`RLX_Q4K_FUSED_MIN_N`).** Default
  **disabled**: the fused path quantizes the activation to Q8_K, which is not
  bit-identical to the f32 cached-BLAS path and flips occasional near-tie greedy
  tokens, so rlx keeps decode on f32 for fidelity and decode↔prefill parity.

- **GPU thermal/power monitoring + control (`rlx_runtime::hwinfo`, `Device::Cuda`
  & `Device::Rocm`).** Cross-backend, read-only telemetry —
  `device_thermal(device, index) -> Option<GpuThermal>`, `device_thermal_count`,
  and `all_gpu_thermal()` — reporting temperature (die/edge + junction/hotspot +
  VRAM), board power, power cap, fan %, SM clock, and utilization. Every field is
  `Option`: a sensor the board doesn't expose stays `None` rather than a fabricated
  zero (a laptop GPU has no junction sensor / settable cap; an APU/iGPU reports
  socket power only). The concrete readers are self-contained `libloading` shims —
  `rlx_cuda::nvml` (NVML `libnvidia-ml.so`) and `rlx_rocm::rsmi` (ROCm-SMI
  `librocm_smi64.so`) — that mirror `roctx.rs`: they dlopen at runtime and return a
  clean `None` on hosts without the vendor library, so the crates still compile and
  test on macOS/CI. **Control** (root-only): `set_power_cap` /
  `set_locked_clocks` / `set_fan_percent` and their resets, plus `power_cap_range`,
  returning a typed `ThermalError { Unavailable, Unsupported, PermissionDenied,
  OutOfRange, Driver }` with range pre-validation so a caller can't drive a GPU
  outside its safe envelope. Power-cap + fan work on both vendors; clock-lock is
  NVIDIA-only (the effective lever on laptop parts that reject a power cap) — ROCm
  clock-lock reports `Unsupported` (it needs perf-level=MANUAL + a frequency
  bitmask; the MI100's lever is its power cap). Surfaced by a new **`rlx-gpu` CLI**
  (a `rlx-bench` bin, zero-dep): `rlx-gpu --watch` to monitor and
  `sudo rlx-gpu --device rocm --index 0 --power-cap 200` to control; plus a
  **bench watchdog** that attaches `gpu_peak` (peak temp/power around the timed
  loop) to `BenchResult`, so a thermally-throttled run is visible instead of
  silently absorbed by wall-clock timing. Validated on real hardware: an AMD
  **Instinct MI100** (edge/junction temps, 0–290 W cap range) alongside a Radeon
  **780M** iGPU (socket-power fallback, no cap), and an **RTX 3080 Ti Laptop**
  (1455 MHz SM clock, fan `NOT_SUPPORTED`, 1–150 W cap range); the read,
  range-query, `PermissionDenied`, and `Unsupported` paths are all exercised
  on-device (a successful privileged *set* still needs root on the box).

- **AMD XDNA / Ryzen AI NPU backend (`rlx-xdna`, `Device::Xdna`).** Runs graphs on
  the AI Engine (`aie2`) tile array via the in-kernel `amdxdna` driver — validated
  bit-exact (cosine for the quantized matmul) against the CPU backend on a Ryzen
  **Phoenix `npu1`** APU. `XdnaBackend::supported_ops()` covers ~39 kinds: **INT8
  GEMM** (~638 GOP/s peak, via the vendor `aie::mmul` overlay), multi-head causal
  attention, RoPE (NeoX/GptJ), RMS/Layer/GroupNorm, softmax, 26 activations,
  elementwise / reduce / scan, data-movement, Quantize / Dequantize / FakeQuantize,
  and 2-D pool + im2col (a conv is `im2col → INT8 GEMM` on the NPU). **Backward +
  training:** backward graphs decompose to these primitives and run on the NPU
  (including a dynamic-weight GEMM for `xᵀ @ dy`); a host-optimizer SGD loop trains
  with the gradient computed on-device. Pure-Rust **AIE-MLIR emitter** + Python-free
  overlay compilation (native `aiecc`). `RLX_XDNA_TURBO=1` clocks the array to max
  DPM (+11% on the GEMM; needs root). The AIE array is INT8/BF16, so f32 matmuls run
  quantized (cosine ≈ 0.99); everything else is bit-exact. A `direct` amdxdna-ioctl
  path (no XRT / no C++ shim) is code-complete but parked (firmware exec-hang on
  Phoenix under Secure Boot lockdown). No CPU fallback — a missing runtime is a clear
  `XdnaError`, never a masquerade.

- **`rlx! { … }` declarative graph DSL (`rlx-tensor`, feature `dsl`).** A compact
  little language for declaring an `rlx_ir::Graph`: `graph`/`input`/`param`/
  `const`/`let`/`out` statements with `@` matmul (NumPy precedence), `+ - * /`
  elementwise (broadcasting + scalar promotion), `f(x)` activation sugar, and a
  `x.method(args)` escape hatch to the full `Tensor` API (bare-ident args
  validated + auto-borrowed; `(value)` opts an external value out). Shapes,
  wiring, and outputs are inferred; inputs/params auto-name from the binding.
  A semantic pass reports unknown bindings, matmul-on-scalar, and non-tensor
  `let`s as spanned compile errors. Implemented as a `macro_rules!` wrapper (for
  `$crate` hygiene across `rlx_tensor::rlx!` / umbrella `rlx::rlx!`) over a
  Pratt-parser proc macro in `rlx-macros`; lowers to the existing shape-inferring
  builders at zero runtime cost. Opt-in on `rlx-tensor` (`dsl`); enabled by
  default for umbrella `rlx` users via the `tensor` feature.

- **`.rlxp` package format (`rlx-pkg`).** Default **flat mmap** (`RLXPFLAT` +
  JSON TOC + 64-byte-aligned data) with **hybrid tiers**: hot (raw mmap), warm
  (`zstd_blocks` / `ZBLK`), cold (whole-blob zstd sidecars). No weight duplex;
  optional ZIP/dir containers (hybrid codecs in shards). Load path: O(1) name
  index, hot-only graph materialize, `madvise(WILLNEED)`, parallel warm decode,
  xxh3 verify, optional bincode TOC / string table, weight-only packs,
  auto-tier, GGUF→RLXP import, `rlx-pkg` CLI, pyrlx `load_rlxp` /
  `convert_gguf_to_rlxp`, feature-gated `encrypt` (`RLXSEAL1`) and `remote`
  HTTP Range. Optional executable MIR graph (ONNX-like) via
  `rlx-bake --features onnx -- import-onnx` (`--no-graph` for weights-only).
  Bake: `--format rlxp`; runtime: `rlx_runtime::pkg` + `rlxp://`.
  Spec: [`docs/rlxp.md`](docs/rlxp.md). `RLXBAKE1` remains supported.

- **CPU activation vmath.** Added `Activation::Recip` and host
  `vvexpf`/`vvtanhf`/`vvrecf`/`vvlogf`/`vvsqrtf`/`vvrsqrtf`; CPU activation hot
  paths use SIMD fast exp/tanh by default (`RLX_VMATH_ACCURATE=1` →
  Accelerate/libm). Forward + `ActivationBackward` parity for Recip/Exp/Tanh on
  Metal, wgpu, and CUDA.

- **Full `OpKind` claim parity (153/153) across backends.** CPU, Metal, MLX,
  wgpu, CoreML/ANE, CUDA, ROCm, Vulkan, OneAPI, and TPU all claim every
  `OpKind`. Coverage matrix regenerated via `just gen-op-coverage`
  ([`docs/op-coverage.md`](docs/op-coverage.md)). Specialty paths that stay
  host-by-design (`CustomFn`, SPD/Eigh LAPACK, splat prepare/rasterize, some
  Scaled* encode) still legalize; they are no longer claim gaps.

- **Native depth — CUDA / ROCm (shared `rlx-gpu-kernels`).** Training and
  inference kernels wired end-to-end (Step/compile/run): LayerNorm/GroupNorm
  backward, FakeQuantize Fixed/PerBatch/EMA + LSQ/STE, SoftmaxCrossEntropy*,
  Relu/ActivationBackward, BatchNormInference (+bwd), ComplexNormSq*/Conjugate,
  Gru/Rnn/Mamba2 (size-capped + host fallback), FftButterflyStage, packed-I8
  `QMatMul`/`QConv2d`, PartitionedConv unfuse, DenseSolve via cuSOLVER /
  hipSOLVER (+ batched LU). ROCm brought to CUDA parity for Conv2d/MaxPool2d
  backward and FusedConvBiasAct.

- **Native depth — Metal / wgpu.** Conv3d/ConvTranspose3d, LayerNorm/GroupNorm
  backward, FakeQuantize, ActivationBackward, ComplexNormSq*/Conjugate (plus
  Metal C64 host I/O fix), FftButterflyStage, fused Gru/Rnn/Mamba2 (existing
  path kept; parity extended).

- **Native depth — CoreML / MLX.** MIL: Fma, FusedMatMulBiasAct, FusedSwiGLU,
  FusedResidual LN/RMS, FakeQuantize, Conv3d (Param weights), ComplexNormSq*/
  Conjugate, FftButterflyStage. MLX: FakeQuantize rounding (half-away-from-zero),
  BatchNormInference (+bwd), Complex*/Quantize/QMatMul/QConv2d, FakeQuantizeLSQ*,
  FftButterflyStage, Mamba2 time-unroll, PerTensor ScaledMatMul/Dequant/QuantScale.

- **Native depth — Vulkan / OneAPI.** Claim parity to 153; SPIR-V / OpenCL for
  GroupNorm (+bwd), fused residual/SwiGLU, SoftmaxCE*, Complex*, Fma,
  FakeQuantize, BatchNormInference, AxialRope2d, act bwd, Conv3d, LN/RMS bwd,
  FusedConvBiasAct, FftButterflyStage, Gru/Rnn/Mamba2/Lstm (capped), Conv2d
  bwd, MaxPool2d bwd, Rope/Attention bwd, packed-I8 Quantize/QMatMul/QConv2d,
  Cumsum/Gather bwd, WelchPeaks. Unfuse for LoraMatMul / FusedTransformerLayer /
  If / While / DotGeneral / GatedDeltaNet / regions; DenseSolve via HostOpDesc
  + CPU LAPACK.

- **Native depth — TPU.** Claim parity to 153. HLO compose for Fma, norms/QAT,
  Complex*, ReluBackward, Conv3d, Conv2d bwd (ConvGeneralDilated VJP),
  MaxPool2dBackward (`SelectAndScatter`), AttentionBackward (autodiff expand),
  AxialRope2d, Im2Col, ConvTranspose*, PerTensor Scaled*, Reverse,
  ResizeNearest2x. Unfuse for recurrent/fused/control-flow. Host segments for
  DenseSolve, FftButterflyStage, SPD/Eigh, splat prepare/rasterize.

- **CPU fused/control expand.** Claims + `prepare_graph_for_thunks` expand for
  `If`/`While`/`FusedTransformerLayer`/`PartitionedConv`/`FusedConvBiasAct`/
  `TransformRegion`/`BatchElementwiseRegion` (was 146 → 153).

### Changed

- **`Session::new` says which half of "unavailable" is missing.** The panic
  blamed the Cargo feature unconditionally, which is wrong on a host where the
  feature *is* compiled in and no such device is present — "enable the `gpu`
  Cargo feature" on a machine with no GPU adapter. New
  `rlx_runtime::feature_compiled(device)` separates the two, and the message now
  distinguishes a missing feature from a missing device/driver.

- **MSRV raised to Rust 1.89** (was 1.87). The `Q2_0` VNNI kernels call
  `_mm256_dpbusd_epi32` / `_mm256_dpbusd_avx_epi32`, stabilized in 1.89; on
  x86-64 the crate did not build on the previously declared minimum.

- **Layout-preserving `Transpose` → `Reshape` (`rlx-unfuse`).** A permutation is
  a pure relabel — row-major bytes unchanged — when every axis of size > 1 keeps
  its relative order and only size-1 axes move. That is the common **decode**
  case (`seq == 1`), where the attention rank-4 promotion's `[0,2,1,3]` between
  `[B,S,H,D]` and `[B,H,S,D]` is a no-op. `Reshape` is a free arena alias on
  every backend while `Transpose` always emits a copy kernel, so this is
  bit-identical and removes real dispatches.

- **im2col/GEMM degenerate-shape guard (CPU conv).** im2col pays for itself
  through M-dimension reuse (`c_out` per group). When `c_out` is tiny the GEMM
  degenerates toward a matrix-*vector* product — no blocking, no register
  tiling, AMX idle — while im2col still materializes the full patch matrix. Such
  shapes now bypass im2col; wide-channel ML convolutions keep it, which remains
  their measured optimum.

- **Memory planner: custom-op liveness extension is now gated.**
  `extend_custom_op_input_liveness` only runs when a dequant matmul can reach the
  deferred-host flush. A backend that runs *all* dequant matmuls on-GPU passes
  `dequant_host_fallback = false`, restoring slot reuse that the chain-walk
  otherwise defeats by pinning nearly every activation in a packed prefill to the
  end of the graph.

- **`RLX_*` unification (phased).** Single registry in
  [`crates/core/rlx-ir/src/env_registry.rs`](crates/core/rlx-ir/src/env_registry.rs)
  (`kind` / `stability` / `aliases` / `layer`); Public catalog via
  `just env-catalog`; full inventory `docs/rlx-env-vars.md`
  (`just gen-rlx-env-vars`, fails on unregistered `env::flag` reads).
  Typed loaders: `CompileOptions::from_env()`, `CudaRuntimeConfig`,
  `MetalRuntimeConfig`, `WgpuRuntimeConfig`, `MlxRuntimeConfig`. Compile-layer
  flags (`lint_numerics`, `fusion_report`, `no_io_peaks_output`,
  `disable_conv_bias_act_fusion`, …) live on `CompileOptions` /
  `FusionOptions`. Deprecated alias `RLX_DISABLE_METAL_DEQUANT_GPU` →
  `RLX_METAL_DEQUANT_GPU_DISABLE` (warn with `RLX_ENV_DEPRECATIONS=1` or
  `RLX_VERBOSE=1`). Prefer `CompileOptions` for compile semantics; env remains
  the CLI/bisect surface.

### Fixed

- **`Op::GroupedMatMul` silently miscomputed on a mis-laid-out expert bank.**
  Every lowering path read `N` off `weight.dim(2)` while the arena sized the
  output slot from the node's declared shape, and nothing checked the two
  agreed. An expert stack left in a checkpoint's `[E, N, K]` (`[out, in]`) order
  has the right rank *and* the right element count, so it passed every existing
  check: the kernel then wrote `M·K` floats into an `M·N` slot — under-writing
  (every output row keeps a tail of whatever the arena last held, which reads as
  a MoE model that is not causal) or, when `K > N`, running past the slot into
  the neighbouring tensor. `rlx_ir::shape::grouped_matmul_dims` now validates K
  against the input and the derived `[M, N]` against the declared output; CPU,
  Metal, MLX, CUDA, ROCm, wgpu, Vulkan, CoreML and TPU all route their dim
  extraction through it, and `Op::GroupedMatMul` gained shape inference so the
  IR verifier reports the mismatch first in debug builds. New
  `HirGraphExt::grouped_matmul(input, weight, expert_idx)` derives the output
  shape from the operands so a builder cannot declare it wrong;
  `rlx-deepseek`/`rlx-motif` use it. Covered by
  `rlx-runtime/tests/grouped_matmul_bank_layout.rs`.

- **Metal: ~2% crash per `MPSGraphExecutable` initialization.** Compiling with a
  nil `MPSGraphCompilationDescriptor` lets MPSGraph defer `GPURegionRuntime`
  construction and its optimizer passes onto its own `MPSGraphExecutable_queue`,
  and that deferred path faults inside MetalPerformanceShadersGraph — a null
  global read at `+0x10`/`+0x18` from `MPSGraphOSLog`, with no RLX frames on the
  faulting thread. Measured at ~2% **per process**, reproducing with nothing else
  running, so it is a hard crash on ordinary Metal model load, not a
  parallel-test artifact. `compile_executable` now requests
  `waitForCompilationCompletion = YES`, keeping that work on the calling thread:
  **0 crashes in ~260 executable initializations** (vs ≈5 expected), with compile
  time unchanged (~2 ms/layer, flat to 28 layers). Guarded by
  `respondsToSelector:`, so an OS without the setter keeps the previous
  behaviour; `RLX_DISABLE_MPSGRAPH_EXECUTABLE=1` remains as a fallback.
  Write-up for Apple in [`docs/apple-feedback-mpsgraph-crash.md`](docs/apple-feedback-mpsgraph-crash.md).

- **aarch64 / armv7 Linux: every linalg op panicked.** `rlx-cpu`'s `build.rs`
  skipped the OpenBLAS link on any non-x86_64 target unless `OPENBLAS_LIB_DIR`
  was pinned, leaving `rlx_cpu_blas` unset. Matmul fell back to the portable SIMD
  gemm, but Cholesky / eigh / QR / SVD / logdet / pinv / solve_triangular have no
  pure-Rust fallback and their wrappers are panic stubs — so on Raspberry
  Pi-class hardware they aborted at runtime (86 tests across 13 binaries). The
  script now probes the standard multiarch lib dirs, enumerating
  `/usr/lib/*-linux-*` rather than guessing a triple (32-bit Pi is arch `arm` but
  the directory is `arm-linux-gnueabihf`). Native builds only — cross-compiles
  must still pin explicitly, so a host OpenBLAS is never linked for the wrong
  architecture. aarch64 Linux goes from 13 failing binaries to **225 suites, 0
  failures**, unpinned.

- **Partial RoPE read the wrong cos/sin table row (`rlx-flow`, wgpu).** The
  tables store exactly the rotation angles — `n_rot/2` per row — but both the
  builder and the WGSL kernel strided by `head_dim/2`. Whenever `n_rot <
  head_dim` that stride overshoots, so every sequence position from 1 onward
  indexed into a *later* token's angles; only position 0 was correct, and the
  result was a plausible-looking rotation rather than a crash. Affects the
  partial-rotary shape used by Gemma 4 global layers, DeepSeek-V4 MLA and
  MiniMax-H3. Stride is now `n_rot`-derived throughout, which equals `head_dim/2`
  for full RoPE, so that case is unchanged. Covered by
  `metal_partial_rope_matches_cpu`.

- **Partial-RoPE *backward* read the wrong cos/sin row on CPU, CUDA/ROCm and
  wgpu.** The mirror of the forward bug above, and the same failure mode as the
  `rms_norm_backward` `1/r` bug: the forward kernels index the table by
  `n_rot/2` but these three backward kernels strided by `head_dim/2`, so under
  partial rotary every position ≥1 differentiated against a later token's
  angles. Metal's `rope_bwd` was already correct, which is why cross-backend
  parity never caught it — three backends agreed with each other and disagreed
  with the forward pass. Caught by `cpu_rope_backward_matches_finite_difference`
  (finite differences of the forward are the independent check). Full rope
  (`n_rot == head_dim`) is bit-identical to before.

- **`GatedResidual` fusion no longer fires on non-F32.** Every backend's
  `GatedResidual` kernel is f32 (CPU reads through the f32 slice helpers, Metal
  takes `device const float*`) and none checked the node dtype, so fusing an F64
  `Add(x, Mul(gate, y))` produced a kernel walking 4-byte lanes over 8-byte data
  and returned garbage — silently, since the shapes still agreed. The pass now
  declines, leaving the plain `Mul` + `Add`, which do have correct per-dtype
  kernels. Regression test: `does_not_fuse_non_f32`.

### Performance

- **Metal: MPS for tiny-`n` decode GEMVs.** At `m == 1, n < 64` no fast GEMV
  kernel applies (splitk/kpart need `n >= 64`) and the fallthrough is
  occupancy-starved — one threadgroup with a serial K-loop, ~0.33 ms each on
  qwen3.5. These now route to MPS: **+25% on qwen3.5-0.8B decode**.

- **Memory planner: packed-prefill arena reuse.** Gating the custom-op liveness
  extension cuts qwen3.5 8K packed prefill from **61.9 GB pinned to ~4 GB
  reused**.

- **CPU: opt-in fused Q4_K int8 decode GEMV.** Measured on Apple, BLAS leads at
  `n ≈ 1k`, ties at `n ≈ 3k`, and the fused path is ~3× ahead at `n ≈ 128k`;
  `n >= 4096` captures the wide matmuls (LM head, large FFN) for a **~15–25% CPU
  decode speedup**. Off by default (see `RLX_Q4K_FUSED_MIN_N` above).

## [0.2.13] - 2026-07-18

### Added

- **CUDA: fused conv + bias + activation (+ residual) via cuDNN.**
  `Op::FusedConvBiasAct` folds a convolution's bias + `relu` — and an
  optional ResNet residual (cuDNN `z` operand) — into one
  `cudnnConvolutionBiasActivationForward`. `FuseConvBiasAct` matches
  `conv→bias→relu`; `FuseConvAffineAct` folds a host-pre-folded BatchNorm
  block `conv→Mul(scale)→Add(shift)→[Add(residual)]→relu` (per-channel scale
  folds into the weights). Fires only for cuDNN-friendly shapes
  (`groups==1, k>1`) + identity/relu — **1.6–2.2×** vs unfused at batch 1 on an
  NVIDIA GPU. cuDNN stays optional: falls back to the direct-conv kernel +
  `conv_bias_act_epilogue.cu` (bit-exact) when libcudnn is absent. Diagnostics:
  `RLX_CUDA_LOG_CONV_PATH`, `RLX_CUDA_LOG_FALLBACK`,
  `RLX_DISABLE_CONV_BIAS_ACT_FUSION`. See
  [`crates/backends/rlx-cuda/README.md`](crates/backends/rlx-cuda/README.md).

- **Compute weight-derived tensors once, not every forward** — two mechanisms.
  Compile-time: `CompileOptions::param_bindings` bakes weights to constants and
  `ConstantFolding` (now with NumPy **broadcasting**) folds per-channel weight
  math away. Run time: `CompileOptions::cache_param_invariant` /
  `RLX_CACHE_PARAM_INVARIANT=1` splits the param-invariant closure into a
  *prepare* graph run once (`rlx_compile::split_param_invariant`), injected into
  the main graph via persistent `bind_handle` (CPU) or a feed fallback (CUDA).
  Complementary + transparent; validated CPU + CUDA. See
  [`docs/weight-compute-caching.md`](docs/weight-compute-caching.md).

- **QNN / Hexagon: x86 HTP functional simulator recipes** (`just qnn-htp-sim`).
  Points `RLX_QNN_BACKEND_LIB` at `libQnnHtp.so` (no Snapdragon silicon).
  Re-runs quantized MatMul probes plus LinearStatic `run_qnn.sh` /
  `run_qnn_context.sh`. Emitted run scripts honor `RLX_QNN_BACKEND_LIB`
  (default still `libQnnCpu.so`). I8×I8 MatMul IR lowers as Dequantize both →
  f32 MatMul (portable across CPU / HTP sim).

- **QNN / Hexagon: offline context-binary path** (`run_qnn_context.sh`).
  Emit → `qnn-model-lib-generator` → `qnn-context-binary-generator` →
  `qnn-net-run --retrieve_context model.bin`. Complements the FFI
  `export_context_binary` / `reload_from_context_binary` path.

- **QNN / Hexagon: `from_graph` → `LinearStatic` with Constant weights.**
  When `Add(MatMul(Input, Constant), Constant)` is recognized, weight/bias
  f32 payloads are baked into the emitted `qnn_model.cpp` (not seed-0).

- **QNN / Hexagon: codegen `LinearStatic`** (STATIC weight/bias packing).
  Activation-only `APP_WRITE`; `W`/`b` baked into `qnn_model.cpp`. CLI
  `--linear-static` / `just qnn-emit-linear-static`.

- **QNN / Hexagon: codegen `Mlp2`** (two-layer `LinearRelu → Linear`).
  Offline `qnn-net-run` path; CLI `--mlp2 M K H N` / `just qnn-emit-mlp2`.

- **QNN / Hexagon: per-channel int8 MatMul weights** (`AXIS_SCALE_OFFSET`).
  `Dequantize` with `axis=Some` + `scales.len() > 1` stages STATIC
  `SFIXED_POINT_8` with QNN `AXIS_SCALE_OFFSET` (e.g. Linear `[K,N]` axis=1).

- **QNN / Hexagon: codegen `MatMulSoftmax`** (`softmax(in0·in1, axis=1)`).
  Offline `qnn-net-run` path; CLI `--matmul-softmax` / `just qnn-emit-matmul-softmax`.

- **QNN / Hexagon: bidirectional `QMatMul` ↔ QNN bridges.**
  Host INT8 accumulate mixes with QNN in either direction:
  `QMatMul → Dequantize → Relu`, and
  `Quantize → QMatMul → Dequantize → Relu` (pre-session → host → post-session).

- **QNN / Hexagon: on-device int4 weights** (BW_SCALE_OFFSET bitwidth=4).
  IR may store tightly packed nibbles (`(K·N+1)/2` bytes); the runtime unpacks
  to 1 byte/elem because `libQnnCpu` rejects native `SFIXED_POINT_4`
  (`UNSUPPORTED_TENSOR_PARAM`). Dequantize → f32 MatMul on device.

- **QNN / Hexagon: on-device `MatMul(I8, I8)` lower.**
  `Quantize(x)` × STATIC I8 weight without Dequantize in IR. Runtime inserts
  Dequantize on both operands → f32 MatMul (libQnnCpu rejects direct
  sfixed8×sfixed8 with `0xc26`; HTP prepare accepts but execute fails on
  MatMul_bias). Host `Op::QMatMul` remains the fully-quantized path.

- **QNN / Hexagon: `Op::QMatMul` (fully quantized INT8, no f32 weight bake).**
  Integer accumulate + requantize on the host (same kernel as `rlx-cpu`);
  weights stay I8. Plus multi-op codegen `Linear` / `LinearRelu` for the
  offline `qnn-net-run` path.

- **QNN / Hexagon: persistent session + context-binary save/load (M3).**
  Shim `rlx_qnn_session_*` finalizes once and reuses across `run`s. Export via
  `contextGetBinary` / reload via `createFromBinary` + `libQnnSystem` metadata.
  `QnnExecutable::{export_context_binary,reload_from_context_binary}`.

- **QNN / Hexagon: on-device int8 MatMul weights** (`MatMul(x, Dequantize(I8_w))`).
  I8 `Param`/`Constant` weights stay STATIC `SFIXED_POINT_8`; QNN runs
  Dequantize → f32 MatMul (mixed f32×int8 MatMul is rejected by the CPU
  backend). Shim `clientBuf.dataSize` uses 1 byte/elem for sfixed8.
  `set_param_typed(I8)` fills deferred weights. FFI + Session parity vs CPU.

- **QNN / Hexagon: host-dequant `DequantMatMul` (GGUF → f32 MatMul).**
  Packed GGUF weights (`Q8_0`/`Q4_0`/K/IQ/…) decode on the host (same posture
  as CoreML's off-device path), transpose `[N,K]→[K,N]`, and run a plain QNN
  MatMul. `set_param_typed(U8)` fills deferred weights. Plus earlier FAB /
  Expand / Silu / Custom Attention work; `just qnn-ffi` validates on Linux.

- **QNN / Hexagon: `FusedAttentionBlock`, Expand, Silu, Custom rank-4 Attention.**
  `QnnBackend` now claims `SUPPORTED_OPS` (legalize) and runs
  `unfuse_attention_block` before FFI lower. New lowers: `Expand` (Reshape+Tile),
  `Silu` (Sigmoid·x), rank-4 `Attention` with additive `MaskKind::Custom` (FAB
  shape), and NeoX RoPE broadcast from compact `[S,D/2]` tables. Parity via
  `fused_attention_block_parity` (`Device::Hexagon`) and `just qnn-ffi`
  (native Linux, no Docker). Shim includes `<stdlib.h>` for newer gcc.

- **Static graph checker (`rlx_runtime::check`) + `cargo rlx check` + `#[rlx_model(check)]`.**
  Folds the analyses rlx already computes — shape/dtype verification (`verify_all`),
  backend dispatch (native / common-IR / unsupported via each backend's real op
  claim), missed fusions (`FusionReport.missed` with fix hints), and the
  provable-NaN/Inf lint (`lint_numerics`) — into one structured `CheckReport`
  (human or `--json`). Fusion/shape/numeric run with no GPU or driver; CPU
  execution legality is always available, other backends' legality is opt-in
  behind per-backend features (op claim read driver-free from the registry —
  factories build unit structs, `supported_ops()` is a const).
  - The checker lives in `rlx_runtime::check::check_graph` (single source of
    truth, no new deps for consumers).
  - **`#[rlx_model(check)]`** injects a `model_self_check(&graph)` call right
    after tracing, so building a model surfaces findings on stderr — tuned by
    `RLX_CHECK` (`off` / `all` / `strict`). The generated code already routes
    through `::rlx_runtime`, so no extra dependency is needed.
  - **`cargo rlx check`** (`crates/tooling/rlx-check`): a `cargo-rlx` subcommand
    over the same `check_graph`, with built-in `--demo` graphs, `--json`, backend
    filtering, and `just check-graph` / `just install-check`. Non-zero exit on any
    error-level finding for CI/pre-commit gating.

- **Vulkan / OneAPI packed DiT reverse SPIR-V.** Dedicated
  `ada_layer_norm_backward` / `gated_residual_backward` compute kernels
  (Vulkan GLSL via naga; OneAPI OpenCL-C via `ocloc` when
  `RLX_ONEAPI_BUILD_KERNELS=1`). `compile_rng` no longer decomposes these
  ops; OneAPI host-fallbacks through `rlx-cpu` when kernels are not embedded.

- **CoreML ANE native MIL for DiT modulation forward (`AdaLayerNorm` /
  `GatedResidual`).** Composed lowering (implicit broadcast, no `Expand`) in
  `rlx_coreml::mil`; `compile_with_options` no longer calls
  `unfuse_dit_modulation`. Host-portable `mil_lower` tests; op-coverage ANE ✅.

- **DiT modulation ops on all backends (`Op::AdaLayerNorm` /
  `Op::GatedResidual`).** Claimed in every `*_SUPPORTED_OPS` set and fused via
  `FuseAdaLayerNorm` / `FuseGatedResidual` on all fusion targets. Native
  kernels: CPU, Metal (MSL), CUDA/ROCm (shared `.cu` in `rlx-gpu-kernels`),
  wgpu (WGSL). Composed: MLX (`ops::layer_norm` / `rms_norm` + broadcast),
  TPU (HLO). Claim-then-`unfuse_dit_modulation` (broadcast Mul/Add, no Expand):
  Vulkan, OneAPI, WebGL (CoreML ANE uses native composed MIL for forward +
  packed reverse). Shared `rlx_ir::ada_modulation_lead_pack` for `[B,1,D]`
  broadcast metadata. Parity: CPU Session + Metal vs CPU.

- **DiT modulation autodiff (fused + unfuse).** Default AD keeps
  `AdaLayerNorm` / `GatedResidual` fused and emits packed
  `AdaLayerNormBackward` / `GatedResidualBackward` (`[dx ∥ dscale ∥ dshift]` /
  `[dx ∥ dy ∥ dgate]`), avoiding Expand of `[B,1,D]` modulation in the
  backward graph. Native reverse kernels on CPU, Metal, CUDA/ROCm (shared
  `.cu`), and wgpu; CoreML ANE uses native composed MIL for packed reverse
  (`COREML_NATIVE_BACKWARD_OPS`); MLX and TPU use native composed lowering in
  `lower.rs`; Vulkan / OneAPI ship dedicated SPIR-V kernels
  (`ada_layer_norm_backward` / `gated_residual_backward`).
  Import-shaped LN+Expand+Mul/Add graphs fuse
  via `FuseAdaLayerNorm` / `FuseGatedResidual` under Session (FLUX adaLN-Zero +
  F5 identity/RMS fixtures in `dit_import_fuse`). Exact JVP for
  LayerNorm / RmsNorm / Ada (full mean/var / RMS pushforward). Microbench:
  `rlx-bench` `bench_dit_modulation`. Flow: `dit_ada_gated_linear` train step.
  The unfuse-for-AD path remains available through
  `rlx_fusion::unfuse_dit_modulation` before `grad_with_loss` (primitive
  VJPs). JVP rules for both forward ops; `vmap` auto-unfuses then lifts
  primitives. FD coverage for fused/unfused reverse, decompose_backward, JVP,
  and vmap; Metal/CUDA/wgpu/ROCm/MLX↔CPU grad parity; tiny DiT-block train
  step on CPU.

- **DiT ONNX adaLN fixture + multi-block train.** Real ONNX `dit_adaln.onnx`
  (affine-free `LayerNormalization` + `Expand` modulation) imports strictly and
  fuses to `AdaLayerNorm` after param specialization; 3-block FLUX-style train
  graph asserts six `AdaLayerNormBackward` / six `GatedResidualBackward` ops.
  `bench_dit_modulation` median numbers recorded in `docs/op-coverage.md`.

- **Differentiable symmetric eigendecomposition (`Op::Eigh` / `Op::EighBatch`).**
  First-class `eigh` as a graph op — `A [n,n] → (λ [n] ascending, U [n,n]`
  columns = eigenvectors, `A = U diag(λ) Uᵀ)`, packed `[λ ∥ U]` internally —
  plus the batched `[B,n,n]` form. Full reverse mode via `Op::EighBackward` /
  `Op::EighBatchBackward` (the standard symmetric-eigendecomposition adjoint
  `Ā = sym(U (diag(λ̄) + F∘(Uᵀ Ū)) Uᵀ)`, `F_ij = 1/(λ_j−λ_i)`, degeneracy-
  guarded). Builders `Graph::eigh` / `Graph::eigh_batch`; VJP wired in autodiff.
  Runs on **all backends**: CPU-native (LAPACK `dsyevd`, f64) and F64 CPU
  host-fallback on Metal / CUDA / ROCm / wgpu / Vulkan / oneAPI (`is_spd_host`
  + every `*_SUPPORTED_OPS` list). This is the differentiable primitive the SPD
  spectral functions build on, and the seam a native batched eigensolver
  (cuSOLVER `syevjBatched` &c.) plugs into. Verified: forward `A = U diag(λ) Uᵀ`
  reconstruction, VJP finite-difference checked (incl. the eigenvector gradient,
  sign-aligned), the exact `Σλ = trace ⇒ ∂/∂A = I` end-to-end gradient through
  `Session`, and gradient parity on real CUDA hardware (bit-exact vs CPU).
- **Native CUDA batched eigensolver (`Step::EighNative`, cuSOLVER
  `SsyevjBatched`).** `Op::Eigh` / `Op::EighBatch` with `n ≤ 32` now run
  **on-device** on the CUDA backend via cuSOLVER's batched Jacobi eigensolver —
  no D2H→CPU→H2D round-trip. An `eigh_assemble` NVRTC kernel transposes
  cuSOLVER's column-major eigenvectors into the packed `[λ ∥ U]` layout and
  interleaves the eigenvalues; `n > 32` falls back to the CPU host path. f32
  (matches the widened SPD arena; f32 Jacobi ≈ f64 LAPACK to cos≈1.0 on these
  small SPD blocks). Measured on an NVIDIA GPU: ~0.5 µs/matrix at n=32,
  batch≥4096 (a single kernel launch for the whole batch) vs ~7.6 µs on the
  rayon CPU path. Requires cudarc's `cusolver` feature (bumped to 0.19.8 for the
  CUDA-13 cuSOLVER symbol set). Verified on CUDA hardware: native forward
  reconstruction (single + batched) and gradient parity vs CPU; full rlx-cuda
  suite green.
- **Native ROCm batched eigensolver (`Step::EighNative`, hipSOLVER
  `SsyevjBatched`).** Same on-device path for AMD: `Op::Eigh` / `Op::EighBatch`
  with `n ≤ 32` when `libhipsolver` is loadable. Hand-rolled libloading shim
  (`hipsolver.rs`) + hipRTC `eigh_assemble` kernel mirroring the CUDA layout.
  Missing hipSOLVER or `n > 32` keeps the existing CPU host-fallback.

- **SPD-manifold Riemannian primitives — host helpers + first-class ops on
  every backend.** `rlx_cpu::spd` gains `karcher_mean_weighted` (the true AIRM
  barycentre `argmin_M Σ wᵢ δ²(M, Cᵢ)` — what a barycentric-OT projection /
  soft-clustering / weighted-MDM needs, replacing the log-Euclidean
  `expm(Σ w̄ᵢ logm Cᵢ)` shortcut), the arbitrary-base `log_map` / `exp_map` and
  `parallel_transport` (AIRM `Log_P`/`Exp_P` and the isometric transport of
  Yair et al. 2019 — `geodesic_interp(A,B,t)` is now just
  `exp_map(A, t·log_map(A,B))`), and rayon-batched `logm_batch` / `expm_batch` /
  `sqrtm_batch` / `invsqrtm_batch`. These are also promoted to graph ops
  `Op::SpdKarcherMeanWeighted` / `SpdLogMap` / `SpdExpMap` /
  `SpdParallelTransport` / `SpdMatrixFnBatch { kind }` (builders
  `Graph::spd_karcher_mean_weighted` / `spd_log_map` / `spd_exp_map` /
  `spd_parallel_transport` / `spd_{logm,expm,sqrtm,invsqrtm}_batch`), so they run
  on **all backends**: CPU-native (F64), and F64 CPU host-fallback on Metal /
  CUDA / ROCm / wgpu / Vulkan / oneAPI (the same eigendecomposition-free
  host-delegation path as the existing SPDNet ops — claimed in every
  `*_SUPPORTED_OPS` list + `is_spd_host`). **Fully differentiable**: the maps
  carry analytic Riemannian VJPs (`SpdLogMapBackward` / `SpdExpMapBackward` /
  `SpdParallelTransportBackward` / `SpdMatrixFnBatchBackward`, packed per-input
  gradients) — the base point is differentiated too, so a learned base trains
  correctly, not just the moving argument (`SpdKarcherMean{,Weighted}` stay
  detached statistics). Verified: weighted-barycentre stationarity, `exp∘log`
  round-trip, transport isometry, batch-vs-scalar parity, and every VJP
  finite-difference checked (rlx-cpu unit tests); end-to-end gradients through
  `Session` on CPU; forward parity on Metal / wgpu / Vulkan / oneAPI /
  CUDA(hardware) / ROCm; and a gradient parity check on the GPU host path (wgpu).
- **First-class `ScatterElements` / `GatherNd` / `GatherElements`.** ONNX
  import emits `Op::ScatterElements { axis, reduction }`,
  `Op::GatherNd { batch_dims }`, and `Op::GatherElements { axis }` (no longer
  Custom / flatten decompositions). CPU thunks are the reference; Metal /
  wgpu / CUDA / ROCm / Vulkan / MLX host-delegate; CoreML hybrid host; TPU
  XLA gather/scatter. Autodiff VJP for all three (plus existing ScatterNd);
  finite-diff checked.
- **First-class `Op::ScatterNd`.** ONNX ScatterND lowers to
  `Op::ScatterNd { reduction }` (none/add/mul/max/min) instead of
  `Op::Custom("onnx.ScatterND")`. CPU thunk is the reference; Metal /
  wgpu / CUDA / ROCm / Vulkan / MLX host-delegate via `HostOpDesc`;
  CoreML MIL `scatter_nd`; TPU XLA scatter. Claimed in every runtime
  `*_SUPPORTED_OPS` list. Autodiff VJP covers all five reductions
  (finite-diff checked).
- **FPGA export legalizer + DX.** `ExportQuantMode::{Int8,Int4,Fp4}`,
  `prepare_model` / `LegalizeOptions`, split requant side tables
  (`*_requant_m0` / `*_requant_shift`), `board_top.sv` shells, bias-free
  Dense / logits-only output, `ExportSession`, and `pyrlx.export_fpga`
  (feature `fpga`). Soft scalar **sidebands** (`SidebandSpec`,
  `bind_sideband_inputs`, CLI `--sideband`). See
  [`docs/fpga-export.md`](docs/fpga-export.md).
- **First-class FPGA / SystemVerilog export.** `Model::from_graph`,
  `FpgaExportConfig` + `HwTarget` (default **target-agnostic** soft-port RTL;
  optional ECP5 / iCE40 / Xilinx7 synth scripts), `rlx_fpga::export_graph`,
  and `rlx_runtime::export` (`ExportTarget::Fpga`, feature `fpga`).
  Docs: [`docs/fpga-export.md`](docs/fpga-export.md). Recipes: `just fpga-emit`,
  `just test-fpga`. `rlx` feature `edge` now includes `fpga`.
- **Unified `Op::Scan` host contract across GPU backends.** Long scans that survive
  legalize run through shared `rlx-cpu` macros (`rlx_scan_host_desc!`,
  `rlx_execute_scan_on_bytes!`, `rlx_scan_stage_d2h!`, `rlx_maybe_unroll_scans!`)
  and `ScanHostDesc` / `run_scan_packed_f32` — Metal / CUDA / ROCm / wgpu /
  Vulkan / OneAPI / CoreML / MLX. Short scans prefer on-device IR via
  `CompileOptions::scan_unroll_max_length` (default **64**) plus
  `maybe_unroll_scans_budget(4096)` (`length × body_nodes`).
- **`Op::ScanBackward` / `ScanBackwardXs` GPU host path** via shared
  `HostOpDesc` (byte offsets) + macros (`rlx_host_op_desc!`,
  `rlx_execute_host_op_on_bytes!`, `rlx_host_op_stage_d2h!`) and value-map
  helpers `run_scan_node_f32` / `run_host_op_node_f32` — Metal / CUDA / ROCm /
  wgpu / Vulkan / CoreML / MLX / OneAPI. Discrete GPUs also share
  `rlx_arena_stage_d2h!` for Scan / HostOp / Spd; wgpu uses `HostOpSpan`.
  Parity: `scan_backward_parity`.
- **MLX `RLX_MLX_PROFILE`:** per-op-kind wall-time dump on executable drop
  (restored after lower.rs host-path work).
- **TPU `QuantScheme::GgufQ1_0`** host dequant at HLO emit
  (`rlx_gguf::q1_dequant::dequant_q1_0`).

### Changed

- **De-duplicated GPU-backend infrastructure into two shared crates.** Three
  copy-pasted-across-backends surfaces now live once:
  - **`rlx-gpu-host`** — host-fallback staging (`D2H → CPU → H2D`) for ops with
    no native kernel (RNN/SSM, im2col, deformable attention, UMAP kNN, log-mel,
    welch-peaks, splat, rms/rope/cumsum/gather backward, Scan/HostOp/indexing,
    RNG fill, SPD manifold eval, GGUF dequant-matmul CPU fallback, maxpool/conv
    training compact-scratch, …). Ops that were duplicated across GPU backends
    are now single-source generic fns over a `DeviceArena` staging trait; each
    backend keeps a ~15-line adapter (`CudaArena`/`RocmArena`/`WgpuArena`) and
    thin per-op wrappers, so call sites are unchanged. SPD `eval` /
    `is_spd_host` are re-exported from Metal/Vulkan/oneAPI/CUDA/ROCm/wgpu.
    Vision/RNN/collective host paths and compact conv/pool training also live
    here. `device_report` rows include advisory `capabilities`; `just new-op`
    prints the AGENTS checklist. ONNX custom-op f32-slot dtype bridging and
    SAM debug hosts also share `rlx-gpu-host`. Verified: CUDA
    `gated_delta_net` / `welch_peaks` parity on an NVIDIA GPU + CPU
    equivalence guard tests.
  - **`rlx-unfuse`** — the IR-level unfuse/decompose pass (`FusedAttentionBlock`,
    `FusedTransformerLayer`, `FusedSwiGLU`, `LoraMatMul`, `DotGeneral`, rank-3
    `Attention`, control flow) shared by `rlx-cuda`/`rlx-rocm`/`rlx-wgpu` behind
    a per-backend `DecomposePolicy` (capability flags). The ~80%-identical logic
    lives once; the CUDA-native-FAB gate, attention-backward promotion, and
    wgpu's fused-op/rank-3 variants are policy flags. Behavior-preserving
    (safety-critical gates confirmed byte-identical; wgpu decompose parity green
    on Metal, CUDA attention parity green on the NVIDIA GPU).
- **Split `rlx-cuda` `backend.rs` (12.5k lines) into a `backend/` module dir**
  (`mod`/`run`/`compile`/`set`/`fill`/`output`), mirroring `rlx-rocm`. Pure code
  move — all methods preserved, public API unchanged.
- **Peel mega backend / lower files into navigable modules** (pure moves; public
  APIs unchanged):
  - **Metal** — `backend/encode/{mod,ops}.rs` (`encode_and_run` + `encode_*`
    helpers); `backend/mod.rs` keeps `MetalExecutable`.
  - **CUDA / ROCm** — `backend/{step,helpers}.rs` (+ CUDA `bwd_launch.rs`);
    `Step` / op-id / schedule helpers off `mod.rs`.
  - **wgpu** — `backend/{step,helpers}.rs`; `compile/{mod,lower}.rs` (match body
    in `lower.rs`).
  - **MLX** — `lower/{mod,env,subgraph,helpers}.rs` (`lower_with_env` match in
    `env.rs`).
- **CPU `RLX_FAST_CONV` defaults on.** Forward Conv2D uses im2col+BLAS (and
  rayon fans for pool / elementwise / bwd) unless `RLX_FAST_CONV=0`. Fixes
  orders-of-magnitude-slow MNIST CNN training that previously fell through to
  the scalar nested-loop kernel. When the fast path is on, OpenBLAS/MKL/
  Accelerate inner threads are capped to 1 (unless the user already set
  `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / …) so Rayon outer parallelism
  does not nest into an N² thread storm. Cortex-M trainer labels stay correct
  (`rlx-fused` by default; set `=0` for the unfused `rlx` bar).
- **Vulkan / oneAPI claim `OpKind::Custom`.** In-graph collectives
  (`collective.all_reduce`, …) used by `rlx-vision-bench` data-parallel graphs
  no longer fail Session legalize on Vulkan; any Custom routes to the existing
  host-fallback. `rlx_oneapi::is_available()` is now always true (CPU reference
  when no Level Zero GPU / kernels) so `RLX_DEVICE=oneapi` works without
  `ze_intel_gpu`. OneAPI `host::eval` now returns dtype-aware `HostOut`
  (Bool/`U8`/`I8` as bytes) so SoftmaxCE→`Compare`→`Where` no longer panics
  reading a 1-byte mask as f32. `CompilePipeline::backend_label` keeps Vulkan /
  OneAPI legalize errors from mislabeling as `"wgpu"`.
- Clippy cleanup: drop unused `CnnGeom::final_hw` / `ident_mat`; split Vulkan
  instance-ext cfg so non-Apple builds don't trip `unused_mut`.
- Codegen backends (QNN / Cerebras) and TPU rewrite route nested Scan through
  `LowerScan` / `LowerControlFlow` so arbitrary-body Scan does not need a
  device body-ISA interpreter.

## [0.2.12] — 2026-07-06

### Added

- **FIR / RIR / IIR digital filters across every backend.** New `Graph` builders
  that compose the existing FFT + elementwise + `Op::Scan` primitives, so they
  lower on all backends with **no new kernels**:
  - **FIR** — `fir_conv1d(x, taps, FirMode)` with `Full`/`Same`/`Valid`/`Causal`
    output modes; `≤ 64` taps use an exact direct time-domain shift-and-add,
    longer filters the FFT convolution theorem (auto-selected).
  - **RIR / convolution reverb** — `partitioned_conv1d(x, ir, block)` and
    `conv_reverb(x, &ir, block)` implement **uniform-partitioned overlap-save
    (UPOLS)**: the impulse response is split into `⌈M/B⌉` partitions summed in a
    frequency-domain delay line, keeping every FFT a fixed `2B` points so long
    room impulse responses stay on the native Metal/WGPU FFT kernels instead of
    one giant `L+M−1` transform.
  - **IIR** — `iirfilt(x, b, a)` (arbitrary-order Direct-Form-II-Transposed via
    `Op::Scan`; native on CPU/MLX, host-fallback on Metal/wgpu) plus
    `iir_as_fir(..)` / `iir_impulse_response(..)`, which reduce a stable IIR to a
    truncated impulse response applied as an FIR so IIR runs natively on **every**
    backend.
  - **Fused `Op::PartitionedConv`** — a first-class op for the partitioned
    convolution. `partitioned_conv(x, ir, block)` builds it; it decomposes (via
    the shared `unfuse` rewrite, before any backend sees it — no per-backend
    kernel) into a **batched complex matmul over the partition axis**
    (`partitioned_conv1d_gemm`), routing the frequency-domain delay line through
    the native batched-GEMM kernels (cuBLAS / rocBLAS / MPS) rather than `P`
    elementwise multiply-accumulates.
  - Validated: CPU numeric (vs direct-convolution / DF2T references) and
    CPU-parity on **Metal, MLX, wgpu, CUDA, and Vulkan** — the last two on a
    discrete NVIDIA GPU (incl. `Op::PartitionedConv`); CoreML compile-checked.
    (Vulkan required the native-FFT kernel below; before it, the FFT filters
    segfaulted on discrete GPUs via host-fallback.)
  - Fixed a latent **batched `irfft`** bug uncovered by the framed convolution:
    the Hermitian mirror reversed the *flattened* last axis via `Op::Gather`,
    corrupting every batch row but the first (any rank ≥ 2 `rfft`/`irfft`,
    including multi-channel `fft_conv1d`); it now uses the batch-general
    `Op::Reverse`, which every backend lowers.
- **Native Vulkan FFT kernel (`Op::Fft` on-device).** Vulkan previously ran
  `Op::Fft` via CPU host-fallback, which **crashes on discrete GPUs** (it
  assumes a host-visible mapped arena the CPU can read the op's inputs from).
  A new `fft` kernel (radix-2 Cooley-Tukey, one workgroup of 256 threads per
  batch row, shared-memory butterflies) runs the forward/inverse f32 FFT
  on-device for power-of-two `n ≤ 1024` (larger `n` / non-f32 still fall back to
  host); dispatch mirrors the wgpu native-FFT guard. This fixes the discrete-GPU
  crash and lets the FIR/RIR/IIR filters run on discrete Vulkan (validated on an
  NVIDIA GPU). Like `matmul_tiled`/`matmul_coop`, it is **precompiled offline
  with glslang** to `shaders/precompiled/fft.spv` — naga's GLSL frontend has no
  `memoryBarrierShared`/`groupMemoryBarrier` and its bare `barrier()` doesn't
  enforce cross-subgroup shared visibility on NVIDIA, so the shared-memory
  stages would race. Also adds an opt-in `RLX_VULKAN_VALIDATION=1` env to enable
  the Khronos validation layer.
- **Parameterized `fNeXmY` minifloats (`ScaledFormat::Custom`).** Beyond the
  seven named tensor-core formats (FP8 e4m3/e5m2 + FNUZ, FP6 e2m3/e3m2, FP4
  e2m1), `ScaledFormat` now carries a `Custom { exp_bits, mant_bits, bias }`
  variant for an arbitrary all-finite minifloat whose whole code fits in a byte
  (`1 + exp + mant ≤ 8`). Build one with `ScaledFormat::custom(exp, mant)` (IEEE
  bias `2^(exp-1)-1`), `custom_with_bias(..)`, or `"f4e3m0".parse()`; `Display`
  round-trips the `fNeXmY` name. Example — **`f4e3m0`** (3 exp, 0 mant): a signed
  power-of-two 4-bit grid, `±0` and `±{0.25, 0.5, 1, 2, 4, 8, 16}`.
  - The f32↔code codec (`lowp_codec`) was already generic over
    `(exp, mant, bias)`, so `ScaledQuantScale` / `ScaledQuantize` / `ScaledMatMul`
    / `ScaledDequantize` accept a `Custom` format with **no new kernel** on **CPU**
    and **Metal** (host decode-and-accumulate reference). Bit-exact CPU round-trip
    test on the `f4e3m0` grid; codec parity tests (a `Custom{2,1}` decodes
    identically to the named `F4E2M1`). **Metal hardware-validated** (Apple GPU):
    `f4e3m0` is bit-for-bit == the CPU oracle (`max_abs = 0`), cosine-vs-f32 0.998.
  - **CUDA / ROCm**: the decode kernel (`scaled_lowp_general.cu`) gains a generic
    path — `kernel_id()` packs `(exp, mant, bias)` into the `fmt` word with a
    top-bit sentinel, which the kernel unpacks and decodes generically. The seven
    named ids (`0..=6`) keep the existing `switch`, so the hardware FP8 path is
    byte-for-byte unchanged. No hardware tensor core exists for these research
    formats — they always take the decode fallback.
    - **Hardware-validated on CUDA** (NVIDIA GPU, NVRTC,
      `crates/backends/rlx-cuda/tests/cuda_scaled_custom.rs`): `f4e3m0` grid GEMM
      **bit-exact vs f32** (`max_abs = 0`), mx-block cosine 0.998, multi-tile
      (37×80×45) cosine 0.998; and a **12-format sweep** (6 custom splits + native
      fp8 + fnuz/fp6/fp4) where on-device quantize→dequantize is **bit-for-bit ==
      the CPU oracle** for every format.
    - The decode GEMM (`scaled_matmul_decode`) is now **shared-memory tiled**
      (16×16 — each code decoded once per tile instead of once per output
      element): **~5.4× faster** on the NVIDIA GPU at 1024³ (74.9 → 405 GFLOP/s),
      same launch config, correctness unchanged.
    - **ROCm/HIP**: the shared kernels compile cleanly under `hipcc` for a real
      AMD target (gfx90a / CDNA2); on-device run pending AMD hardware (none
      reachable here — the rig is NVIDIA).
    - Fixed two latent NVRTC bugs uncovered by these first on-device runs (the
      scaled kernels had never actually NVRTC-compiled before): (1) the decode
      kernel used the `INFINITY` macro, undefined under NVRTC → now
      `__int_as_float` (`#ifndef`-guarded, so nvcc/hipcc unaffected); (2) the
      native per-tensor fp8 quantize kernel did `#include <cuda_fp8.h>` /
      `<hip/hip_fp8.h>` (no NVRTC/hipRTC include path) → the f32→fp8 conversion is
      now closed-form, matching the oracle bit-for-bit, removing the toolkit-header
      dependency entirely.
  - **Vulkan**: the four scaled ops are now wired into `rlx-vulkan` as CPU
    host-fallbacks (the same `rlx-cpu` oracle Metal uses) against the mapped
    host-visible arena — added to `SUPPORTED_OPS` + `is_host_fallback`, and the
    generic host path now writes **U8 outputs** (quant codes / block scales) as
    raw bytes instead of reinterpreting them as f32. **Validated on a native
    NVIDIA Vulkan driver** (NVIDIA GPU): `f4e3m0` grid GEMM bit-exact vs f32,
    mx-block cosine 0.998 (`tests/vulkan_scaled_custom.rs`). (First native-driver
    Vulkan validation of any scaled path.)

- **Pick a minifloat format at the high level** — no more hand-wiring the
  `ScaledQuantScale → ScaledQuantize → ScaledMatMul` chain. The same
  `ScaledFormat` (any named format or a parameterized `Custom` like `f4e3m0`)
  now flows through every composition/execution surface:
  - **Compose ops** — `Graph::scaled_matmul(lhs, rhs, fmt, layout)` (+
    `scaled_quantize` / `scaled_dequantize` / `scaled_matmul_bias`) on the
    low-level graph, and the mirror `HirGraphExt::scaled_matmul` on the HIR
    builder. One call emits the whole quantize→GEMM chain.
  - **Tensor DSL** — `Tensor::scaled_matmul(&rhs, fmt, layout)` on the lazy
    `rlx-tensor` tensors (e.g. `a.scaled_matmul(&w, ScaledFormat::custom(3,0),
    ScaleLayout::mx())`).
  - **Execute a flow** — `CompileOptions::scaled_quant(ScaledQuantConfig{..})`
    (re-exported as `rlx_runtime::ScaledQuantConfig`); `Session::compile_with`
    runs the existing `insert_scaled_matmul` pass so *every* 2-D matmul in a
    graph is rewritten to the chosen format at compile time.
  - **Python** — `pyrlx.Graph.scaled_matmul(lhs, rhs, format="f4e3m0",
    layout="mx")` parses the `fNeXmY` name (via `ScaledFormat: FromStr`) and
    rejects invalid splits with a clear `ValueError`.
  - Verified end-to-end on CPU at every layer (builder, Session policy, Tensor
    DSL, and a pyrlx→CPU run: `f4e3m0` cosine 0.997 vs numpy).

- **`ScaledFormat` Rust API / DX.** Working with the different float formats is
  now ergonomic without reaching for the free-function codec or tuple `fields()`:
  - `const` constructors — `ScaledFormat::custom(3, 0)` / `custom_with_bias(..)`
    are `const fn`, usable in `const` items and pattern-free construction.
  - Introspection accessors — `exp_bits()`, `mant_bits()`, `bias()`,
    `is_custom()`, `is_named()`, and a `ScaledFormat::NAMED: [_; 7]` array for
    format sweeps (all `const`).
  - Codec methods on the format itself — `fmt.decode(code)`, `fmt.encode(x)`,
    `fmt.quantize(x)` (round-trip a single f32 to its nearest representable
    value, e.g. `custom(3,0).quantize(1.4) == 1.0`), and
    `fmt.representable_values()` to inspect a format's whole grid.
  - `ScaleLayout: FromStr` (`"mx"`, `"nvfp4"`, `"per_tensor"`, `"mx/<block>"`)
    to match `ScaledFormat: FromStr` (`"f4e3m0"`); the pyrlx layout arg now
    delegates to it (one source of truth). Doc-tested.

- **Low-precision `encode(±inf)` now saturates to `±max_finite`** (`lowp_codec`
  + the GPU `rlx_encode_lowp`), matching the codec's documented contract — it
  previously returned code `0` because every finite candidate is equidistant from
  a huge value in f64. Applies to all formats, incl. E5M2 (whose inf code is never
  emitted). CPU + GPU kept in lockstep.

- **`rlx-torch-import` diffusion-model op coverage.** New aten→rlx lowerings so
  UNet / DiT image models import cleanly:
  - `group_norm` / `native_group_norm` → `Op::GroupNorm`.
  - `upsample_nearest2d` / `_upsample_nearest_exact2d` (`.default` + `.vec`, any
    2ⁿ× scale) → chained `Op::ResizeNearest2x`.
  - `constant_pad_nd` (constant mode) → `concat` with `Op::Constant` fills.
  - `baddbmm` → `beta·input + alpha·(b1@b2)` (the non-SDPA attention score path;
    `beta = 0` fast path drops the bias).
  - `split.Tensor` / `chunk` → narrow tuples (GEGLU / adaLN modulation).
  - `zeros` / `ones` / `empty` (+ `_like` / `new_`) → `Op::Constant`.
  - `pixel_shuffle` / `pixel_unshuffle` → reshape + permute.
  - `_scaled_dot_product_{flash,efficient,math}_attention` → same `Op::Attention`
    as the public op (routes each overload's own arg layout; `decomposition=core`).
  - `leaky_relu`, `hardswish`, `hardsigmoid`, `hardtanh` → clamp/mul decompositions.
  - `masked_fill` → `x + mask·(value − x)` (arithmetic; avoids bool-cond `Where`
    on the f32 arena).
  - `upsample_{bilinear,bicubic}2d` and their antialiased `_aa` overloads (any
    output size, `align_corners` either way) via new
    `HirMut::resize_{bilinear,bicubic}2d[_aa]` builders — every filter is
    separable, so they lower to two constant-matrix interpolation matmuls
    (`MatMul`/`reshape`/`transpose`). Bit-exact PyTorch parity on **every backend
    with no new kernel** (verified CPU + Metal + MLX on Apple).

  Unblocks Stable Diffusion / SDXL (UNet + VAE) and Sana (linear-attention
  ones-padding). 18 exact-parity CPU tests (+ Metal/MLX resize parity) + README.

- **`rlx-torch-import` dynamic shapes.** A model exported with
  `torch.export(dynamic_shapes=…)` now imports with symbolic input dims instead of
  raising. The front-end (`pyrlx.from_torch(..., dynamic_shapes=…)`) emits a
  per-axis `dynamic` marker (stable symbol id per `SymInt`); the importer builds
  `Dim::Dynamic(sym)` inputs (`InputDef.dyn_dims` / `hir_shape`), the compile pass
  re-infers the symbolic graph, and `DimBinding` specializes it per run — so a
  model is imported **once** and run at any batch/seq (`run_dynamic`; `verify`
  binds the reference shape). End-to-end verified (Linear+GELU with dynamic batch
  imports + parity), plus a Rust test running one dynamic-batch HIR at batch 2 & 4.
- **CPU interpreter `Op::Expand` now broadcasts.** The executor treated `Expand`
  like `Reshape` (a plain `input.len()` copy), leaving stride-0 axes unfilled — so
  any imported broadcast (e.g. a `torch.eye` built from `arange`/`eq`) produced
  garbage. Now does a proper strided broadcast walk.
- **Importer `sum`/`mean` with an empty dim list reduce all dims.** aten's
  `sum.dim_IntList(x, [])` / `mean.dim(x, [])` — how bare `Tensor.sum()`/`.mean()`
  decompose — reduce over *every* dim (→ scalar); the importer reduced *nothing*
  (returned the input), so e.g. `torch.eye(n) + x.sum()*0` left a rank-2 tensor and
  broke the downstream broadcast. Now an empty (or absent) dim list reduces all axes.
- **Importer int/bool intermediates on the f32 arena.** A byte-sized
  `arange`/compare *intermediate* was mis-read (the f32-uniform arena under-sizes a
  bool node — `9 bytes → 2 f32 slots` — so a comparison feeding a matmul went OOB).
  Fixes: integer `arange` is materialized F32 (values exact); comparisons emit
  `Compare → Cast(F32)` so the consumed value is a properly-sized f32 `{0,1}`
  tensor (same pattern as grid_sample); non-float node dtypes are tracked as F32.
  The `torch.eye` (`arange`/`eq`/`mm`) identity now imports with exact parity — the
  case that surfaced this.
- **`pyrlx.from_torch` auto-decompose fallback.** When the Rust importer reports
  ops the RLX registry doesn't cover, the front-end re-exports with those ops
  decomposed (via torch's full decomposition registry) and retries, up to
  `max_decompose_rounds` — so torch-decomposable ops are handled automatically
  (reported in `summary["auto_decomposed_ops"]`). Decompositions that emit
  `prims.*` (breaking functionalization) are bisected out one at a time, so a bad
  op degrades to a clean "unsupported op" report instead of a crash. On by default
  (`auto_decompose=False` to disable). Pure-logic tested in
  `tests/test_torch_import_autodecomp.py`.
- **`grid_sampler` / `grid_sampler_2d` — all variations.** New
  `HirMut::grid_sample2d` decomposes `grid_sample` into universal ops (transpose →
  axis-0 `Gather` + `Round`-based floor + arithmetic weights, batch unrolled):
  every interpolation mode (nearest / bilinear / bicubic) × padding
  (zeros / border / reflection) × `align_corners`. Exact PyTorch parity verified
  on **CPU, Metal, and MLX** for all mode/padding combos. Also fixed the CPU
  interpreter `Op::Gather` general-axis path (previously silently zeroed for
  `axis != 0`).
- **rlx-mlx fusion cap.** The deep grid_sample decomposition made `mlx::compile`
  fuse an elementwise region into one Metal kernel that exhausted the
  argument-buffer limit. Fixed with (a) eval barriers in the Lazy lowering
  (`lower_with_env`) that materialize a fusable chain every `RLX_MLX_FUSE_CAP`
  (default 12) ops — non-elementwise ops reset the counter, so ordinary models
  are untouched, and it's disabled inside the `mlx::compile` trace; and (b) a
  retry in `run_read_outputs` that, on an over-fused-kernel failure, disables
  compile and re-runs in barrier'd Lazy.

- **`rlx-ir::HirMut::resize_{bilinear,bicubic}2d[_aa]`.** Separable NCHW
  bilinear/bicubic resize built from universal ops (constant interpolation matrix
  + `MatMul`); no per-backend kernel. Bicubic uses PyTorch's Keys cubic kernel
  (`a = -0.75`) with edge clamping; the `_aa` path widens the filter by the
  downsampling ratio and renormalizes (and matches plain when upsampling, as
  PyTorch documents). Unit-tested for shape, partition of unity, sample
  interpolation, antialias downsample weights, and the aa==plain upsample identity.

## [0.2.11] — 2026-07-05

### Added

- **GGUF IQ / TQ / MX end-to-end.** Encoders in `rlx-gguf`; `rlx-gguf-convert` scheme
  enum; GPU dequant parity on Metal / WGPU / CUDA / ROCm; CoreML MIL on-device splits
  for IQ / TQ / MX and K-quants; TPU compile-time and runtime Param bake paths.
- **Metal fused IQ GEMV (`m = 1`).** `iq4_nl`, `iq2_xxs/xs/s`, `iq3_xxs/s`, `iq1_s/m`
  MV kernels with per-scheme disable env vars.
- **Grouped MoE GGUF tests.** Q4_0 / Q8_K / IQ2 / IQ3 / TQ2 / IQ1 expert stacks on
  CPU / Metal / WGPU / CUDA / ROCm (`dequant_grouped_matmul_gguf.rs`).
- **pyrlx GGUF.** `quantize` / `dequant`, `load_gguf` / `write_gguf`, `convert_to_gguf`
  (safetensors → GGUF via `rlx-gguf-convert`); tests in `crates/pyrlx/tests/test_gguf_*.py`.
- **Docs.** Canonical GGUF backend matrix in `docs/gguf-backend-paths.md`; op coverage
  updates in `docs/op-coverage.md`; `just test-gguf-grouped` recipe.
- **`FusedAttentionBlock` is first-class on every inference backend.** All backends now
  declare `OpKind::FusedAttentionBlock` and lower it — CPU/MLX natively, everyone else
  by decomposing to the primitive chain (matmul → narrow → reshape/transpose → \[rope\]
  → attention → matmul). New FAB-only `rlx_fusion::unfuse::unfuse_attention_block` pass
  (shared decomposition with the autodiff unfuse): CUDA/ROCm/TPU/WGPU decompose through
  their own crate `unfuse`, Metal/Vulkan/oneAPI/CoreML through an explicit pre-lowering
  pass (FAB-only, so each backend's native `FusedMatMulBiasAct` / `FusedResidualLN` /
  `LoraMatMul` survive). Cross-backend parity test
  `crates/rlx-runtime/tests/fused_attention_block_parity.rs`. WebGL stays
  excluded — it cannot lower the resulting `Op::Attention`. QNN claims FAB and
  decomposes via `unfuse_attention_block` before the FFI lower.
- **Native CUDA + Metal fused-attention kernels.** A `fused_attn_block` kernel
  (CUDA `fused_attn.cu`; Metal MSL in `kernels.rs`) fuses inline NeoX RoPE + softmax
  SDPA over the packed QKV projection — one block/threadgroup per batch·head, score
  matrix in shared/threadgroup memory — collapsing the decompose chain's narrow×3 +
  transpose×3 + rope×2 + attention into a single launch; the QKV / output projections
  stay as GEMMs into appended arena scratch. Each backend keeps the block native when
  the `[seq,seq]` scores fit (`seq ≤ 96` CUDA / `≤ 64` Metal; Metal additionally gates
  to f32 + no-bias) and otherwise falls back to the primitive decomposition. Validated
  vs the CPU reference: CUDA on an NVIDIA GPU (identity / bias / rope), Metal on Apple
  Silicon (identity / rope native; bias decomposes).

### Performance

- **`rlx-vulkan` dependency-aware barriers.** The scheduler emitted a global
  shader-memory barrier between *every* dispatch; it now emits one only on a real
  read/write hazard, tracked by arena slot offset (safe given the unique-slot bump
  allocator — aliasing shares an offset). On MoltenVK, where each barrier forces a
  Metal compute-encoder restart, this is ~25% faster on the resident-MLP MNIST
  step (135 dispatches, most independent) with bit-identical results. New env
  knobs: `RLX_VULKAN_DEBUG=1` (per-step dispatch histogram), `RLX_VULKAN_FULLBARRIER=1`
  (restore the old between-every-pair behaviour), `RLX_VULKAN_NOBARRIER=1` (drop all
  — unsafe, diagnostic only).
- **`rlx-vulkan` pre-recorded command buffers.** The static schedule (kernels,
  push constants, workgroup counts are fixed; inputs flow through the host-visible
  arena, not the command stream) is now recorded once into reusable command
  buffers and resubmitted with a persistent fence, instead of allocating a command
  buffer + recording + creating/destroying a fence every step. Neutral on MoltenVK
  (re-encodes per submit) but a real latency win on native Vulkan drivers
  (Linux/NVIDIA). `RLX_VULKAN_NOCACHE=1` restores the per-step record path.
- Combined with a larger batch (the `rlx-mnist-device` bench gained an `RLX_BATCH`
  knob), native Vulkan resident-MLP MNIST goes from ~55–87k to ~340–395k img/s on
  an M4 Pro at *higher* accuracy (0.944 → 0.963).

### Fixed

- **`dequant_gguf` routing for legacy 32-byte schemes.** Q4_1 / Q5_0 / Q5_1 no longer
  fall into the 256-element K-quant branch (MSL / CUDA / WGSL).
- **IQ3 fused GEMV element order.** MV kernels now match dequant layout (g1 block then
  g2 block per sub-quant).
- **MLX GGUF dequant cache.** Cache key includes scheme and packed bytes hash.
- **WGPU TQ1_0 / IQ1_M dequant.** Trit u8-wrap and scale-word fixes in WGSL.

## [0.2.10] — 2026-06-25

### Added

- **Distributed / multi-node transport (`rlx-driver`).** New `transport.rs` +
  `net.rs` add a two-sided point-to-point `Transport` trait and a `ProcessGroup`
  layering tensor-shaped `all_reduce` / `all_gather` / `broadcast` / `barrier` on
  top, plus a full-mesh TCP `NetTransport` (one connection per rank pair, a demux
  reader thread per connection routing `SEND` / `PUT` / `GETREQ` / `GETRESP`
  frames). `NetTransport` implements both the two-sided `Transport`
  (pipeline-parallel hidden-state handoff) and the one-sided `SymmetricTransport`
  (put/get/barrier) surfaces, so the existing symmetric collectives run over it
  unchanged. `TcpTransport` and a `ThunderboltTransport` (same wire protocol,
  intended for the macOS Thunderbolt Bridge link, with a `looks_like_thunderbolt`
  IP heuristic) are exposed; re-exported from `rlx-runtime`. Verified by
  multi-rank loopback tests (pipeline handoff, all-reduce / broadcast / barrier
  over TCP, remote put/get). Real multi-node hardware not exercised in this
  environment.
- **`rlx-driver::ring_all_reduce`** — bandwidth-optimal ring all-reduce over a
  symmetric heap (reduce-scatter ring + all-gather ring, Baidu / NCCL pattern)
  expressed with one-sided `put` + `barrier`; each rank moves ~`2(N-1)/N` of the
  vector vs the naïve gather-to-root `O(N²)`. Verified multi-threaded (Sum + Mean)
  against the serial reference.
- **New crate `rlx-collectives`** — an in-graph `collective.all_reduce` custom op
  for tensor-parallel layers: the op carries a `u64` group id in its attrs, each
  rank registers its `ProcessGroup` under an id, and the CPU kernel resolves it
  and sums across ranks at execution time (blocking rendezvous). Validated
  end-to-end by tensor-parallel matmul, Megatron-style SwiGLU MLP, and a Qwen3
  decoder-layer shard test against hand-computed references.
- **`rlx-mlx::MlxTransport` (`distributed.rs`)** — a `Transport` backed by MLX's
  distributed module (`jaccl` RDMA-over-Thunderbolt / `ring` TCP / `mpi`,
  auto-selected), with native `all_sum` / `all_gather` plus two-sided
  length-prefixed byte send/recv and barrier over the MLX C ABI. Also registers a
  **device-resident** `collective.all_reduce` MLX kernel that composes
  `mc::distributed::all_sum` on the lazy device array (no host round-trip), so a
  tensor-parallel layer's all-reduce stays on-GPU. New C-ABI shim entry points in
  `rlx-mlx-sys` (`rlx_mlx_dist_*`). Singleton (no-launcher) path tested; multi-rank
  jaccl / ring needs MLX's launcher.
- **`Op::Reverse { axes }`** — batch-general flip along listed axes (output shape
  unchanged; non-listed axes pass through). Native kernels on CPU, Metal, MLX, and
  WGPU; host-staged on CUDA / ROCm. Reverses a `[batch, seq, …]` sequence without
  a `batch == 1` assumption.
- **`Op::Gru`, `Op::Rnn`, `Op::Mamba2` native GPU kernels.** Single-layer /
  unidirectional / no-carry kernels: Metal MSL `gru` / `rnn` / `mamba2` (hidden
  ≤ 1024, Mamba2 state ≤ 128) and native WGSL `gru` / `rnn` / `mamba2` (hidden /
  state ≤ 256), each with a host-staged CPU fallback for the multi-layer /
  bidirectional / carry / oversized cases. The CPU references
  (`execute_gru_f32` / `execute_rnn_f32` / `execute_mamba2_f32`) back the
  fallbacks. HIR gains a `Gru` op + lowering. Validated by `metal_rnn_native` /
  `wgpu_rnn_native` parity tests against the CPU reference.
- **Native Metal `selective_scan` MSL kernel** (f32, state ≤ 128) replacing the
  host fallback for Mamba S6 on Metal, plus native Metal `argreduce`, `sample`,
  and `reverse` paths. Their CPU references (`execute_selective_scan_f32`,
  `execute_sample_f32`, `execute_argreduce_f32`) are now shared host-delegate
  functions.
- **Native Metal block-quantized matmul** (`dequant_matmul_int8` /
  `dequant_matmul_int4` MSL kernels) — dequant-on-the-fly over the unified-memory
  arena for int8 / int4 block schemes, matching the CPU reference (new
  `metal_dequant_matmul_int_parity` test).
- **WGPU vision + recurrent op coverage.** Host-staged `ConvTranspose2d`,
  `GroupNorm`, `LayerNorm2d`, `ResizeNearest2x`, `Reverse`, and `ArgMax` / `ArgMin`
  (readback → verified CPU kernel → writeback, mirroring the existing
  `im2col_host` pattern; new `vision_host.rs` / `conv_transpose2d_host.rs`),
  closing the correctness gap for SAM / U-Net decoders on cross-platform GPU. New
  `wgpu_vision_ops_parity` test.
- **CUDA / ROCm host-staged `Reverse`, `ArgMax`, `ArgMin`, `AxialRope2d`, and
  `StopGradient`** (`host_misc.rs` on both backends: sync → dtoh → verified
  rlx-cpu kernel → htod). TPU also gains `StopGradient` (forward-identity HLO
  alias). **Compile-verified only; not run on NVIDIA / AMD / TPU hardware** (no
  device in this environment).
- **MLX op coverage** — native `Reverse`, `ArgMax`, `ArgMin`, `Im2Col`
  (NCHW → rows), `GroupNorm` (NCHW), and `GroupNorm` backward (input / gamma /
  beta) lowerings.
- **CoreML / ANE fused-attention layouts** — `lower_attention` now disambiguates
  the `[B,S,H,D]` (heads at axis 2) vs `[B,H,S,D]` (canonical) operand layouts via
  `attention_geom`, transposing the former to canonical, attending, and
  transposing back — without it CoreML would attend over the heads axis and fail
  once `s_q != s_k` (KV-cache decode).
- **Dynamic-shape support on every backend (`rlx-runtime::deferred`).**
  `Session::compile` now detects `Dim::Dynamic` graphs and wraps them in a
  `DeferredExecutable` that infers the concrete shape from input lengths on each
  `run`, specializes to a static graph, and caches the most recent specialization
  — giving CPU / Metal / CUDA / ROCm the multi-shape support wgpu / MLX already
  had internally, with no recompile when a shape repeats.
- **`rlx-text` streaming detokenization + tool-call parsing.** New
  `StreamingDetokenizer` (`detokenize.rs`) re-decodes the full id sequence each
  step and emits only the newly-stable suffix, holding back trailing U+FFFD runs —
  fixes byte-level-BPE multi-byte splits and SentencePiece context-dependent
  spacing that per-token decode mangles. New `tool_parse.rs` parses model-emitted
  tool/function calls (Hermes / Qwen `<tool_call>` blocks, bare JSON, and Pythonic
  `[fn(a="x")]`) into `{name, arguments}`, with a `detect_and_parse` helper.
- **Min-p sampling + logit bias (`rlx-runtime`).** New `MinP` sampler (keep tokens
  with prob ≥ `p · p_max`, with a `min_keep` floor; Nguyen et al. 2024) wired
  through `SampleOpts::min_p`, and a host-side `apply_logit_bias` (OpenAI
  `logit_bias` semantics, bounds-checked, additive).
- **Host-driven logits decode + prompt-cache session reuse (`LmRunner`).** New
  default-implemented `prefill_logits` / `decode_logits` (caller owns sampling,
  logit bias, log-probs, stop detection — for the HTTP server), plus
  `export_session` / `restore_session` / `prefill_logits_reusing` over a new
  `SessionSnapshot` (KV cache + token history) for prefix reuse.
- **Streaming attention over a quantized KV cache (`rlx-runtime::quantized_kv`).**
  `attend_quantized` does single-query GQA-aware attention directly over the
  quantized layer, dequantizing one row at a time so peak extra memory is
  `O(kv_dim)` not `O(past_len · kv_dim)`; new `read_rows(start, count)`
  generalizes `read_window`. Validated against full-dequant attention.
- **Sliding-window decode masking.** `attn_mask::bucket_decode_mask_windowed`
  masks cached keys outside `[past_seq − window, past_seq]` for incremental decode
  (reduces to the causal mask when the window is wide), and the Qwen3 decoder block
  (`rlx-flow`) now takes a configurable `MaskKind` (`Causal` or
  `SlidingWindow(w)`) instead of hard-coded causal — enabling Mistral / Gemma-style
  sliding-window models.
- **`Device::as_arg`** — canonical lowercase CLI token that round-trips through
  `FromStr` (unlike the human-facing `name`, e.g. `"GPU (wgpu)"`).
- **New parity / coverage tests** across backends (argreduce, conv2d-groups,
  conv-bias, conv-transpose2d, GRU, group-norm-backward, LoRA-matmul-decompose,
  LSQ-quant, native RNN on Metal/WGPU, im2col on MLX, multi-shape / dynamic,
  reverse, sample, vision-ops, sliding-window-attn, dequant-matmul-int, expand,
  CPU selective-scan), a `bench_new_ops.rs` example, and a new
  [`docs/op-coverage.md`](docs/op-coverage.md) — the single-source-of-truth
  op × backend matrix (113 `OpKind`s).

### Performance

- **CPU executor: per-run arena reset is now O(scratch), not O(params).**
  `restore_arena_baseline` previously cloned and rewrote the entire (multi-GB)
  weight region on every `run()`, making large models swap-thrash. Params /
  constants now live solely in their dedicated never-aliased arena slots (no
  redundant CPU-side copy that doubled the weight footprint), and only the
  complement of the persistent byte ranges is zeroed each run. Constants are
  written once.
- **`Op::GroupedMatMul` on CPU is now a real segmented GEMM** — counting-sort
  tokens by expert, one GEMM per expert, then unpermute — replacing the naive
  per-token implementation. (GPU backends dispatch a dedicated grouped kernel.)
- **`OpKind::LoraMatMul` added to `FUSED_KINDS`** so its `unfuse` decomposition
  actually fires on backends without a native LoRA kernel (Metal / WGPU / CUDA /
  ROCm / TPU). Standalone LoRA previously failed legalization unless another fused
  op happened to be present (verified exact vs CPU).

### Fixed

- **Per-token (ragged) RoPE on CPU and Metal.** The RoPE kernels now index the
  cos/sin table per token (one row per batch·seq element) when the table has
  `total_tokens` rows, so ragged batched decode — each sequence in the batch at a
  different absolute position — gets its own RoPE row instead of collapsing to row
  0. New Metal `cos_per_token` kernel path; `device_ext::supports_ragged_rope`
  gates this to CPU + Metal (other GPU RoPE kernels still index by seq position, so
  callers fall back to per-length uniform grouping there). Validated by the new
  ragged `metal_rope_parity` test against CPU.
- **`ElementwiseRegion` output dtype mis-inference.** A fused elementwise chain now
  takes the dtype of its final chain step (walking `Compare → Bool`,
  `Cast → its dtype`, …) rather than input 0's — input 0 may be a bool `Where`
  condition (`where(cond, a, b) + …`), which previously mis-typed the whole region
  as bool.
- **`conv_transpose2d` dynamic-batch shape inference** now preserves a dynamic
  batch dim (mirroring `conv2d_output_shape`) instead of force-unwrapping it to a
  static value.
- **Metal uninitialized-buffer reads.** `new_buffer` (shared storage) is now zeroed
  on allocation — ops that read unwritten arena regions (e.g. conv halo padding)
  previously picked up per-process garbage, a nondeterminism / correctness bug.
- **`rlx-coreml` stale `.mlmodelc` cache** — a version-incompatible or corrupt
  compiled-model cache no longer permanently breaks loading; a load failure
  discards the cache entry and recompiles from the `.mlpackage`.
- **`moe_residency` GroupedMatMul ordinal made thread-local.** The process-global
  atomic let one thread's `reset` / `next` ordinal sequence clobber another
  in-flight forward's (corrupting layer/matrix decode → wrong TIDE host-expert
  weights + residency accounting); forwards run per-thread, so the counter is now
  thread-local.
- **`LocalTransport` barrier is now a real rendezvous** (`std::sync::Barrier`,
  auto-reset) instead of an arrival counter, so multi-step collectives (ring
  all-reduce) synchronize correctly across threads.
- **`memory_estimate::would_exceed_soft_budget`** boundary logic split into a pure,
  deterministically-testable `exceeds_budget` predicate (the prior test was flaky
  against live fluctuating RSS).

### Changed

- **CPU now declares `Reverse`, `Gru`, `Rnn`, `Mamba2`, `FakeQuantizeLSQ` (+ LSQ
  backward X/scale), and `GroupNorm` backward (input/gamma/beta)** in
  `CPU_SUPPORTED_OPS` — the kernels already existed in `thunk.rs` / training-bwd,
  but the legalization const omitted them, so compiled CPU graphs couldn't use
  them. CPU is once again the reference that lowers every `OpKind` any backend does.
- Backend `supported_ops()` consts updated to reflect the new native / host
  kernels above; the canonical per-backend op counts now live in
  [`docs/op-coverage.md`](docs/op-coverage.md) (CPU 104, MLX 84, MTL 76, WGPU 75,
  CUDA 71, ROCm 68, TPU 50 of 113 `OpKind`s).
- New env opt-out: `RLX_METAL_RNN_HOST_FALLBACK` forces the Metal GRU / RNN host
  path.

## [0.2.9] — 2026-06-22

### Performance

- **`rlx_cpu::ms_deform_attn`** (the shared host-delegate behind the fused
  `Op::Custom("gdino.ms_deform_attn")` on CPU/Metal/MLX/CUDA/WGPU) now routes its
  value/offset/attention/output projections through `blas::sgemm_bt` instead of a
  naive triple loop. The projections dominate at full token counts (~18k); the
  GPU host-delegates were ~7× slower than the CPU backend (which already used
  BLAS) purely from this. Grounding DINO MLX enhancer 8.8→1.0s, decoder 2.6→0.1s.
- **`rlx_cpu::conv_fwd::conv2d_forward_nchw_f32`** rewritten as im2col +
  `blas::sgemm` (groups/stride/pad/dilation preserved); replaces a naive 6-deep
  loop. Benefits CNN-backbone models on CPU and the GPU conv host-delegates.
  Validated by the existing conv fwd/bwd/1×1/q_conv2d tests.

### Added

- **`rlx-coreml` fused multi-head attention & RoPE layouts.** `lower_attention`
  and the RoPE lowering now dispatch on operand layout: the original split
  `[..,S,D]` / last-dim-==-`head_dim` path is byte-for-byte unchanged, and a new
  fused `[B,S,H·D]` path (heads packed in the last axis, as in Qwen3 / Qwen3-ASR
  fused-QKV) reshapes+transposes to canonical `[B,H,S,D]`, runs the shared
  `attention_core`, and folds the result back. RoPE gains the same per-head view
  (cos/sin broadcast over a singleton head axis). Expands the set of transformer
  models lowerable to CoreML/ANE.

### Fixed

- **`rlx-wgpu` attention additive-bias mask.** `MaskKind::Bias` now lowers to its
  own kernel path (mask kind `4`, `score += mask`) instead of being folded into
  the binary key-padding path (kind `2`, `mask < 0.5 → -inf`). The two are not
  interchangeable — an additive block-diagonal window bias (e.g. the encoder
  winmask) was silently corrupted by the binary path.
- **`rlx-wgpu` decode-step causal/sliding-window masking.** The attention kernel
  now compares against the absolute query position `qi + (seq_k − seq_q)` rather
  than the local `qi`, so causal and sliding-window masks are correct when
  `seq_q == 1` during incremental decode (past KV precedes the query). Prefill
  (`seq_q == seq_k`) is unaffected.
- **`rlx-wgpu` matmul arena-window assertion** no longer spuriously panics for
  models whose entire arena fits within `max_binding` (whole-arena bind reports
  `param_anchor = false`, but the large param B is trivially in-window); the
  assertion now keys on actual addressability.
- **`rlx-cuda` / `rlx-rocm` attention kernels** (`attention.cu`,
  `attention_row.cu` in `rlx-gpu-kernels`, shared by both backends) gained
  parity with the Metal/WGPU fixes above: (1) an additive-bias path for
  `MaskKind::Bias` (kernel mask kind `4`, `score += mask`) — the backends
  already bound the bias tensor for kind `4` but the kernels had no branch for
  it, so the bias was silently dropped (ALiBi / block-diagonal window bias);
  (2) decode-step causal/sliding-window masking now compares against the
  absolute query position `qi + (seq_k − seq_q)` instead of the local `qi`, so
  causality is correct when `seq_q < seq_k` during incremental decode. Masking
  logic validated by a host-C++ harness; **not yet verified on CUDA/ROCm
  hardware** (no device available in this environment).
- **`rlx-mlx` concurrent free/eval crash.** `Array::drop` freed its MLX handle
  (`rlx_mlx_array_free`) without the runtime lock, so a result array freed on
  one thread could race a guarded `eval()` on another and SIGSEGV (intermittent,
  release + multi-threaded). The runtime lock is now reentrant (thread-local
  depth over the existing mutex) and is held by `Array::drop`, `eval`,
  `async_eval`, `synchronize`, and `clone_handle` — so cross-thread frees
  serialize against in-flight eval, while intermediate drops inside a guarded
  `run_*` on the same thread don't deadlock. Single-threaded inference (the hot
  path) only pays a thread-id check.

### Changed

- **`rlx-metal` built-in custom-op kernels auto-register.** The bundled
  host-delegate kernels (e.g. `ms_deform_attn`) register themselves on first
  custom-op lookup — no explicit `register()` call or extra cargo feature
  required. (`llada2_gate` stays consumer-registered to avoid double-registration.)
- **`rlx-mlx` Lazy-fallback warning** now fires at most once per distinct reason
  per process (was once per executable — models with many graphs sharing a
  host-eval op flooded the log). `RLX_MLX_WARN_LAZY=all` restores per-executable
  warnings. Logging-only; execution is unchanged.

## [0.2.8] — 2026-06

### Added — PyTorch `.pt` weight loading

- **`rlx_nemo::PtModel`** — standalone loader for plain PyTorch `.pt` /
  `.pth` / `pytorch_model.bin` `torch.save` checkpoints, reusing the
  non-executing pickle VM + STORED-zip reader that already backs the
  `.nemo` loader. Tensors materialize on demand as contiguous f32
  regardless of on-disk dtype (fp32 / fp16 / bf16 / int). Modern (≥ PyTorch
  1.6) zip format only; legacy non-zip pickles are rejected with a clear
  error.
- **`rlx-gguf-convert` `pt` feature** — `Converter::from_pt(...)` + a
  `PtReader` `TensorReader`, so `.pt` checkpoints convert/quantize to GGUF
  like safetensors/ONNX. The `convert` example now dispatches by input
  extension (`.safetensors` / `.pt` / `.pth` / `.bin` / `.onnx`).

### Added — Recurrent / sequence ops & complex matmul

- **`Op::Lstm`** — multi-layer, optionally bidirectional, optional decode
  carry (`h0`/`c0` threaded in place). Packed weights; gate order i,f,g,o.
  Real CPU kernel (`execute_lstm_f32`) shared by the CUDA / ROCm / wgpu /
  Metal host fallbacks; **native Metal MSL kernel** for the single-layer
  unidirectional path (`hidden ≤ 1024`, opt out via `RLX_METAL_LSTM_CPU=1`).
  Decomposes via `unfuse` for MLX / CoreML / TPU and backprop-through-time
  (verified against central finite differences).
- **`Op::Gru`** (PyTorch r/z/n, separate `b_ih`/`b_hh`) and **`Op::Rnn`**
  (Elman, tanh/ReLU) — multi-layer / bidirectional / carry, via the same
  `unfuse`-for-autodiff decomposition.
- **`Op::Mamba2`** — Mamba-2 / SSD scalar-decay structured state-space scan
  (sibling of `SelectiveScan` / `GatedDeltaNet`); `unfuse` decomposition.
- **Complex (C64) matmul** — `Thunk::CgemmC64` CPU kernel + a `MatMul`
  lowering branch, completing the deferred piece of C64 support. `MatMul`
  VJP now inserts Wirtinger conjugates for C64
  (`dA = upstream·conj(B)ᵀ`, `dB = conj(A)ᵀ·upstream`).
- **ONNX coverage ops** — GatherND, ScatterND, OneHot, NonZero, CumProd,
  and Einsum (with a real equation parser) as `Op::Custom("onnx.*")` CPU
  reference kernels wired through the ONNX import path.

### Changed

- **`rlx-mlx-sys`**: bumped the vendored MLX submodule to **0.32.0** (from
  0.31.2). The C-ABI shim builds unchanged against the new upstream API;
  all `rlx-mlx` (128) and runtime-MLX integration (455) tests pass.

### Added — `rlx-tensor` symbolic Tensor DSL

- **New crate `rlx-tensor`**: a native `ndarray` alternative — NumPy-style,
  operator-overloaded `Tensor` handles that trace into `rlx-ir` instead of
  executing eagerly. The graph stays lazy (so fusion + memory planning see
  the whole expression) until forced. Re-exported through the prelude as
  `rlx::tensor` / `rlx::prelude::{Tensor, graph, s, shape, ...}` behind the
  umbrella `tensor` feature (**on by default**; `--no-default-features` drops
  it). The umbrella backend flags (`cpu`/`metal`/`cuda`/…) now also enable the
  matching `rlx-tensor` `eval` backend via weak features, so
  `rlx::tensor::Tensor::to_vec` / `.on(Device::…)` materialize out of the box.
- **Op surface**: arithmetic + scalar ops, activations (`relu`/`gelu`/`silu`/
  …), reductions (`sum`/`mean`/`var`/`logsumexp`/`cumsum`/`argmax`/…), shape &
  view ops (`reshape`/`narrow`/`slice` via `s![]`/`split`/`cat`/`stack`/…),
  indexing (`gather`/`where_`/`masked_fill`/comparisons), and NN/linalg
  (`matmul`/`softmax`/`layer_norm`/`rms_norm`/`conv2d`/`attention`/`rope`/
  `fft`/`inv`/`solve`). First-class `Dim::Dynamic` for variable batch/seq.
- **Opt-in features**: `eval` (materialize via `rlx_runtime::Session`, default
  CPU; `eval-metal`/`eval-mlx`/`eval-cuda`/`eval-rocm`/`eval-gpu`/`eval-coreml`/
  `eval-apple`/`eval-blas` for other backends), `grad`/`transforms` (reverse-mode
  AD + composable `Func` transforms `vmap`/`jvp`/`hvp`), `optim`
  (`Func::train_step` + `rlx_optim` optimizers), and `ndarray` interop
  (`Tensor::from(array)` / `to_ndarray`). Base crate stays a pure `rlx-ir`
  graph builder with no backend pulled in.

### Added — IQ / TQ / MX dequant family

- **`rlx-gguf`**: dequant kernels for every llama.cpp scheme — IQ4_NL,
  IQ4_XS, IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M, TQ1_0/TQ2_0, MXFP4, NVFP4.
  Grid LUTs auto-extracted from `ggml-common.h` and shipped in
  `src/iq_grids.rs`. Real-weight parity tests against `llama-quantize`
  output on Qwen3-0.6B (`tests/iq_tq_real_weights.rs`).
- **Q2_K / Q3_K layout fix** (`rlx-gguf`, `rlx-cpu/gguf_matmul`,
  `rlx-metal/dequant_gguf.msl`, `rlx-cuda/dequant_gguf.cu`): pre-existing
  encoder/decoder put f16 `d`/`dmin` at the front of the block, but
  llama.cpp's actual layout is `scales | qs | d` (Q2_K) and
  `hmask | qs | scales | d` (Q3_K). Constant-value round-trip tests
  passed because the encoder used the same flipped layout; real GGUFs
  produced NaN. Decoder and encoder now match `ggml-common.h` exactly.
- **GPU dequant**: native MSL (rlx-metal) + CUDA (rlx-cuda) kernels for
  all 19 schemes. IQ-family grids staged into a ~33 KB device buffer
  per session/context via `Kernels::iq_grid_buffer` (Metal) /
  `cuda_iq_grid_buffer` (CUDA), bound as the 6th kernel argument.
  `has_metal_dequant_kernel(QuantScheme)` reports coverage.
- **`rlx-ir`**: 13 new `QuantScheme` variants (`GgufIQ4NL`, `GgufIQ4XS`,
  `GgufIQ2XXS`, `GgufIQ2XS`, `GgufIQ2S`, `GgufIQ3XXS`, `GgufIQ3S`,
  `GgufIQ1S`, `GgufIQ1M`, `GgufTQ1_0`, `GgufTQ2_0`, `GgufMXFP4`,
  `GgufNVFP4`) with byte-counts wired into `gguf_block_size` /
  `gguf_block_bytes` / `is_gguf` / `bits_per_element_x10`.
- **`rlx-gguf` encoders**: IQ/TQ/MX quantize path (`iq_quantize`,
  `iq2_encode`, `iq3_encode`, `iq1_encode`, `tq_quantize`, `mx_quantize`).
  IQ2 uses llama.cpp kmap + sign-extraction; parallel block encoding via
  `rayon`. CoreML on-device dequant for NVFP4 + IQ2/3/1
  (`split_*_ondevice` in `rlx-coreml/src/mil/helpers.rs`).

### Added — Samplers

- **`rlx-runtime::samplers`** module: backend-agnostic `Sampler` trait +
  `SamplerChain`. Implements Temperature, DynamicTemperature, TopK,
  TopP, TopNSigma, TypicalP, Mirostat v1 / v2, XTC, DRY, RepetitionPenalty
  in the canonical llama.cpp order.
- **`SampleOpts` extended** (`rlx-runtime::lm`, `rlx_qwen3::sampling`):
  fields for every advanced sampler default to off; `into_chain()`
  builds the chain; `is_classic()` lets legacy callers stay on the
  fast inline path. `MirostatMode` enum exposed at the runtime top
  level. `sample_token_with_history()` and `sample_token_stateful()`
  added to rlx-qwen3 for chain-aware decoders.

### Added — Quantized KV cache

- **`rlx-runtime::quantized_kv`**: per-layer K/V history stored as
  `KvQuant::{F16, Q8_0, Q5_0, Q4_0}` GGUF blocks instead of f16/f32.
  ~2–4× memory cut on long decodes. `QuantizedKvLayer::{append_rows,
  read_all, read_window, drop_front}` + `QuantizedKvCache` aggregate.
- **`mmap-kv` feature**: optional `MmapKvLayer` / `MmapKvCache` backed by
  `memmap2` for anonymous or file-backed mappings. Pages cold history
  in/out via the OS page cache; `prefetch_window` issues madvise
  WILLNEED. Use case: 100k-token contexts that exceed RAM.

## [0.2.7] — 2026-06

### Added

- **In-graph RNG** (`rlx-ir`): `Op::RngNormal` / `Op::RngUniform` for ONNX
  `Random*` / `Random*Like`; `RngOptions` / `RngBackend` (Philox default,
  Ort CPU parity, Zero for deterministic tests). `CompileOptions::rng` and
  `CompiledGraph::set_rng` override policy without recompiling.
- **RNG backends**: CPU Philox + ORT reference (`rlx-cpu`); host-fill on Metal/MLX;
  D2H→fill→H2D on CUDA/ROCm/wgpu; XLA `rng` lowering on TPU (`rlx-tpu`).
- **ONNX `Random*` import** (`rlx-onnx-import`): native lowering to
  `Op::RngNormal` / `Op::RngUniform` (direct import + codegen); shared
  `random.rs` helpers; conformance harness coverage (`rlx-onnx-conformance`).
- **Autodiff / vmap**: RNG ops treated as stateless w.r.t. gradients (`rlx-autodiff`).

### Changed

- Patch bumps for all workspace crates in this release train to **0.2.7**
  (`rlx-ir`, `rlx-compile`, `rlx-gpu-kernels`, `rlx-cuda`, `rlx-wgpu`, `rlx-metal`,
  `rlx-mlx`, `rlx-runtime`, `rlx`, …); `rlx-mlx-sys` and `pyrlx` at **0.2.7**.

## [0.2.6] — 2026-06

### Added

- **Native GPU `Op::WelchPeaks`** (`rlx-cuda`, `rlx-wgpu`, `rlx-rocm`, `rlx-gpu-kernels`):
  in-arena Welch PSD top-K when eligible (`rlx-ir::welch_peaks_gpu_native_eligible`,
  f32 spectrum, ≤512 one-sided bins, K≤64); host CPU path unchanged for out-of-range
  shapes.
- **`rlx-compile` IO-gated fusion**: `SelectPeaksOnlyOutputs` drops FFT spectrum from
  graph outputs when peaks-only readback wins the per-target IO gate; compile-time
  `profile_graph_io` / `profile_graph_io_outputs`; thread-local `FusionTarget` for
  gated passes. Opt out with `RLX_NO_IO_PEAKS_OUTPUT=1`.

### Fixed

- **`rlx-mlx` `Activation::GeluApprox`**: lower through `ops::gelu_approx` (tanh
  form matching `rlx-cpu`) instead of exact `gelu`. Fixes ~3% Brain-JEPA predictor
  drift vs CPU while SDPA stayed within tolerance. Optional `RLX_MLX_SDPA_REFERENCE=1`
  composes unfused matmul+softmax for SDPA bisects.
- **`rlx-metal` MPSGraph `Activation::Gelu`**: use `erfWithTensor` + the CPU
  erf GELU formula (`0.5·x·(1+erf(x/√2))`) instead of the tanh approximation.
  Fixes large CPU/Metal drift on REVE-style GEGLU blocks (~0.08 max abs → ~5e-3
  on a single transformer layer; full-model parity restored with MPSGraph enabled).

### Changed

- Patch bumps for all workspace crates in this release train to **0.2.6**
  (`rlx-ir`, `rlx-compile`, `rlx-gpu-kernels`, `rlx-cuda`, `rlx-wgpu`, `rlx-metal`,
  `rlx-mlx`, `rlx-runtime`, `rlx`, …); `rlx-mlx-sys` remains **0.2.6**.

## [0.2.5] — 2026-06

### Fixed

- **`rlx-runtime`**: import `rlx_opt::pass::Pass` in CUDA and ROCm `Backend::compile`
  so `LegalizeBroadcast` / `AutoMixedPrecision` compile on Rust ≥1.87 (crates.io
  0.2.4 tarball missed this in `compile()` while `compile_lir()` had it).

### Added

- **`Op::WelchPeaks`**: Welch PSD top-K spikes from block-layout FFT segment spectra
  (`rlx-ir`, CPU + Metal + MLX lowering, CUDA/wgpu host sidecars, runtime supported-op lists).
- **`rlx-runtime::graph_io`**: static IO / sync profiling for compile-time fusion
  planning (`GraphIoProfile`, `profile_graph_io`, peaks-only output sizing).
- **`rlx-compile::fusion_benefit`**: IO-aware fusion benefit scoring and per-target
  gates (`io_fusion_gate_for_target`).

## [0.2.3] — 2026-06

### Added

- **Multi-backend runtime** (`rlx-runtime` 0.2.3): `DevicePolicy`,
  `GraphDevices`, `FlexibleSession`, `DeviceRouter`, env-driven resolve /
  fallback (`RLX_DEVICE`, `RLX_DEVICE_CHAIN`, `RLX_BENCHMARK_PICK`),
  `BackendsManifest`, `warm_all` / `benchmark_devices`, typed param sync
  across cached backends.
- **Prelude** (`rlx` 0.2.3): re-exports above + `register_backends!` macro.
- **Python** (`pyrlx` 0.2.3): `GraphDevices`, `DeviceRouter`, `DevicePolicy`,
  `FlexibleSession`, `parse_device`, `backends_manifest`, `fastest_device_for`,
  `device_report`, `set_param_typed` on multi-backend runners.
- **GPU calibrators**: on-disk matmul micro-bench caches for CUDA
  (`rlx-cuda` 0.2.3), ROCm (`rlx-rocm` 0.2.3), wgpu (`rlx-wgpu` 0.2.3);
  feed heterogeneous cost-model ranking.
- **ROCm full CUDA parity** (`rlx-rocm` 0.2.3): all 48 hipRTC kernels, Session-path
  `GroupNorm` / `ResizeNearest2x`, GPU backward ops, GGUF GPU dequant, splat prepare/rasterize,
  im2col, pinned host I/O (`host_staging.rs`, `RLX_ROCM_PINNED_IO`).
- **Runtime ROCm** (`rlx-runtime` 0.2.3): ROCm supported-op parity, `rocm_op_parity` tests,
  ROCm arms in higher-order / autodiff GPU parity suites.
- **FKL-style region fusion** ([`docs/fk-fusion.md`](docs/fk-fusion.md)): resize prologue
  (`FuseRegionPrologue`), batch preprocess (`FuseBatchPreprocess` /
  `BatchElementwiseRegion`), `MarkBatchSliceRegions`, `apply_native_fk_defaults` on
  GPU-class targets and TPU. CUDA/ROCm/Metal/wgpu single-launch batch kernel via
  `RLX_FK_BATCH_SINGLE_KERNEL=1`. TPU HLO lowering in `rlx-tpu` (`prepare_graph_for_hlo`,
  `fk_pipeline`). Parity: `rlx-runtime/tests/fk_prologue_parity.rs`, `pyrlx` FK tests,
  `rlx-bench` `bench_fk_fusion`.
- **HIP-CPU**: Docker-only fetch into `rlx-cuda/docker/vendor/HIP-CPU` via
  `just test-hip-cpu-validate` (linux-gnu; not a git submodule).
- **Autodiff**: `prepare_graph_for_ad` runs `DecomposeFusionRegions` so FKL batch/transform
  ops decompose before reverse-mode AD.
- **Docs**: [`docs/backend-selection.md`](docs/backend-selection.md),
  [`docs/development.md`](docs/development.md), [`docs/README.md`](docs/README.md).
- **Examples**: `rlx-runtime/examples/graph_devices_demo.rs`.
- **Tests**: full `hip_cpu_validate` suite (38 kernel families), `rlx-rocm/tests/basic.rs`
  GatedDeltaNet, `rlx-runtime/tests/rocm_op_parity.rs`,
  `rlx-runtime/tests/graph_devices_parity.rs`, `crates/pyrlx/tests/test_graph_devices.py`,
  ROCm suites in higher-order / autodiff GPU parity tests, `prologue_input` on region op
  literals in Metal/MLX/wgpu parity tests.
- **CI**: `just test-rocm`, `just test-hip-cpu-validate`, ROCm arm in `just ci` /
  `test-third-order-gpu`.

### Changed

- Patch bumps for all crates in this release train (`rlx-ir`, `rlx-opt`,
  `rlx-fusion`, `rlx-compile`, `rlx-autodiff`, `rlx-cpu`, `rlx-cuda`, `rlx-metal`,
  `rlx-mlx`, `rlx-mlx-sys`, `rlx-gpu-kernels`, `rlx-bench`, `rlx-wgpu`); workspace
  dependency pins leveled to 0.2.3.

### Fixed

- **`rlx-gguf`**: `dequant_q6_k_block` now casts per-sub-block scales as
  `i8` (matching `dequant_q6_k`). The old `as f32` path misread bytes
  ≥128 (e.g. `0xFF` → 255 instead of −1), breaking `Op::DequantMatMul`
  on Q6_K tensors such as MiniCPM5 `v_proj` / `down_proj`.

## [0.2.2] — 2026-05

### Added

- **`rlx-umap`** crate — UMAP / fast-umap custom ops (k-NN from pairwise distances).
- **`rlx-gpu-kernels`** crate — shared CUDA/HIP `.cu` sources for `rlx-cuda` + `rlx-rocm`.
- **`rlx-cpu`** kernel and executor improvements.

## [0.2.1] — 2026-05

### Changed

- Workspace/crate version bump only.

### Added

- **`rlx::run` runner API** (`model builders` crate, `run` module):
  builder-style entry points for the supported model families,
  re-exported in the prelude under the `models` cargo feature.
  - `Qwen3Runner::builder()` — `.weights(p)`, `.device(d)`,
    `.max_seq(n)`, `.precision(F32 | F16LmHead)`,
    `.max_memory_gb(g)`, `.stream(bool)`, `.use_mtp(bool)`,
    `.sample(opts)`, `.config(ConfigSource::…)`,
    `.format(WeightFormat::…)`, `.build()`.
  - `SamRunner::builder(SamArch::Sam1 | Sam2 | Sam3)` — uniform
    builder shape, `.predict_image(...)` method dispatches to the
    per-arch `Sam{,2,3}::from_safetensors_on` + forward call.
  - Helpers `open_loader(path)`, `list_mtp_keys(path)`,
    `debug_resolve_name(hf_name)`.
- **`rlx-run` CLI** (`model builders` crate, `rlx-run` binary): subcommands
  `qwen3`, `sam1`, `sam2`, `sam3`, `inspect`, `help`. Hand-rolled
  arg parser — no clap dep. Mirrors the builder API 1:1.
- **`Op::DequantMatMul` GGUF schemes** (`rlx-ir/src/quant.rs`):
  `QuantScheme::GgufQ4K`, `GgufQ5K`, `GgufQ6K`, `GgufQ8K`. CPU
  implementation in `rlx-cpu` dequants the packed bytes to f32
  scratch then sgemm — keeps the arena footprint small (Q4_K ≈
  4.5 bpe vs F32's 32 bpe) at the cost of per-call dequant. Metal
  lowering is on the roadmap (per-op thunk path still dequants at
  load time today).
- **GGUF K-quant decoders** (`rlx-gguf`): Q4_K, Q5_K, Q6_K, Q8_K
  block decoders, mirroring llama.cpp's `ggml-quants.c` reference.
  Made `pub` so `rlx-cpu`'s `DequantMatMul` GGUF arm can call them.
- **`GgufLoader`** (`model builders::weight_loader`): pluggable
  `WeightLoader` for `.gguf` files with transparent
  HF↔GGUF name resolution (`hf_to_gguf_name` /
  `gguf_to_hf_name`), MTP-head isolation (`is_mtp_weight`,
  `mtp_keys`), and shape normalization (innermost-first GGUF dims
  reversed to safetensors order without byte movement).
- **Qwen3 graph builder** (`model builders` crate, `qwen3`): GQA via
  graph-level KV head repetition, QK-norm, RoPE, SwiGLU,
  tied-embedding LM head with build-time weight pre-transpose
  (eliminates 600 MB per-call Transpose op), prefill + cached
  decode generators with bucketed compile cache.
- **MPSGraph Metal fast path** (`rlx-metal`):
  - `rms_norm` via `normalizationWithTensor:mean=0:variance=mean(x²):`
    (uses Apple's fused norm kernel).
  - `attention_causal` via `scaledDotProductAttention` builtin with
    in-graph constant causal mask — bypasses the slice-of-computed
    MPSGraph optimizer bug that hits the BERT QKV-split pattern.
  - `ElementwiseRegion` chain replay for fused SwiGLU.
  - Pre-compiled `MPSGraphExecutable` with feed/result permutation
    recovered from `executable.feedTensors`/`targetTensors`; per-call
    dispatch is one ObjC call with the input/output `NSArray`s built
    once at compile (`bind_arena` + `run_cached`).
  - Default-on whenever lowering succeeds; opt out via
    `RLX_DISABLE_MPSGRAPH` / `RLX_DISABLE_MPSGRAPH_EXECUTABLE`.
  - Opt-in `RLX_MPSGRAPH_PARAM_CONST=1` bakes weights as graph
    constants (production single-shape callers).
- **F16 LM-head path** (opt-in via `RLX_QWEN3_F16_LM_HEAD=1`): casts
  hidden + lm-head weight to F16 before the final matmul. Wins
  1.3-1.45× on B≥2, L≥64 `last` cells.
- **Examples per model family** (`model builders` repo `examples/`):
  `run_qwen3_safetensors.rs`, `run_qwen3_gguf.rs`, `run_sam1.rs`,
  `run_sam2.rs`, `run_sam3.rs`, plus `qwen3_gguf_inference.rs` and
  `gguf_qwen3_probe.rs` for deeper walk-throughs.
- **Publish script** (`scripts/publish.sh`): tier-ordered workspace
  publisher with active sparse-index polling, HTTP 429 backoff, and
  live countdown timers. See `--help`.

### Changed

- **`Op::DequantMatMul::num_inputs()` is now scheme-dependent**
  (was always 4). Returns 2 for GGUF schemes (`[x, packed_w]`),
  4 for legacy Int8 schemes (`[x, w_q, scale, zp]`). **Breaking**
  for any downstream code that hard-coded the input count — match
  on `scheme.is_gguf()` before reading inputs.
- **`GgufLoader::take_transposed` now actually transposes**
  (was a buggy no-op that returned GGUF native bytes with the GGUF
  shape label, silently producing wrong logits when the builder
  expected `[in, out]` row-major). The fix routes through
  `GgufLoader::take` which now normalizes GGUF's innermost-first
  shape convention to safetensors' outermost-first ordering (no
  byte movement — only the shape label flips). **Breaking** for
  any downstream code that compensated for the old buggy
  behavior; drop the workaround.
- **`Qwen3Generator::from_loader`** canonicalizes cache keys to the
  HF naming convention (via `gguf_to_hf_name`) so the same generator
  works against safetensors OR GGUF loaders without builder changes.
- **`set_param_typed`** on the f32-arena backends (CPU, Metal, wgpu)
  now accepts `DType::U8` and `DType::I8` via the existing
  `set_param_bytes` path. Needed by the GGUF `Op::DequantMatMul`
  path to hand raw packed bytes to the arena. Behavior for
  F32/F16/BF16 is unchanged.
- **Pre-transposed tied LM-head embedding** in the qwen3 builder:
  computed once at graph-build time as a distinct param of shape
  `[hidden, vocab]`. The earlier scheme emitted a runtime
  `Transpose(embed_w, [1,0])` op that materialized ~600 MB per
  forward. CPU `last`-mode prefill drops from ~970 ms → ~70 ms on
  this fix alone.
- **MPSGraph lowering is opt-out** (was opt-in). Env-var name
  changed from `RLX_USE_MPSGRAPH=1` to **`RLX_DISABLE_MPSGRAPH=1`**.
  The matrix harness `model builders` example `qwen3_matrix.rs` no longer needs to
  set anything to engage the fast path.
- `WeightFormat::from_path` / `ConfigSource` / `Precision` /
  `SamArch` enums + the runner builders are re-exported as
  `rlx::run::*` (under the `models` feature).
- `rlx::QuantScheme` flat re-export added to the prelude.
- Workspace version bumped from `0.1.0` → `0.2.0` (all 23
  crates).

### Fixed

- **`Q8K` block byte count off by 16** in `QuantScheme::gguf_block_bytes()`
  (was 276, should be 292 = 4 + 256 + 32). Caught by the new
  `dequant_matmul_q8k_matches_dequant_then_matmul` integration test.
- **MPSGraph attention `MaskKind::Causal`** is now lowered correctly
  (was returning `None` from `try_lower` and falling back to the
  per-op encoder path; now uses Apple's fused SDPA with an in-graph
  constant causal mask).
- **`Op::DequantMatMul` `scheme` field** is now used by the
  CPU lowerer to dispatch to the right kernel; previously the
  GGUF schemes panicked with "scheme not implemented".
- Three pre-existing `model builders` warnings (unused
  `multihead_attention` import in `sam3/detector_decoder.rs`,
  unused `data` arg in `sam3/detector_encoder_ir.rs:add_param`,
  dead `sigmoid` fn in `sam3/tensor.rs`) cleaned up so the
  publish script's `clippy -- -D warnings` gate passes.

### Performance

- **Qwen3-0.6B prefill on Apple Silicon (Metal):** RLX beats
  Python+PyTorch+MPS in 11/23 (B, L, mode) cells, ties in 6,
  with the win margin growing from ~5% at L=32 to 1.45× at L=128.
  Beats Candle CPU on every cell tested (2.6×–9×).
- **Qwen3-0.6B Q4_K_M GGUF on Metal end-to-end:** cosine 0.976 vs
  F32 safetensors — textbook Q4_K_M loss, no NaN, top-1 plausible.

### Docs

- New `CHANGELOG.md` (this file).
- `model builders` repo README: added a Qwen3 section, runner DX section,
  per-example table, env-var matrix for the MPSGraph fast path.
- `rlx-ir/README.md`: added a `QuantScheme` table covering legacy
  Int8 + new GGUF schemes, and a Gotchas note about
  `Op::DequantMatMul`'s variable input count.
- `rlx-gguf/README.md`: replaced overclaiming feature list with an
  honest per-format table; documents the shape-convention quirk
  callers need to know about.
- Root `README.md`: new runner section, Status-by-area entries for
  Qwen3 LM + Op::DequantMatMul GGUF schemes + rlx::run.

### Performance / memory

- **Packed-weights qwen3 builder** (`build_qwen3_graph_sized_packed`):
  K-quant matmul weights stay packed in the arena and the graph
  emits `Op::DequantMatMul { scheme }` per projection. On
  Qwen3-0.6B Q4_K_M: arena drops from 2.22 GB → 1.42 GB end-to-end
  with **bit-exact parity** against the F32-load path (cosine
  1.00000, max\|Δ\| 0.000, top-1 match). End-to-end example at
  `model builders` example `qwen3_packed_inference.rs`; set
  `RLX_QWEN3_PARITY=1` to also build the F32 reference for the
  same file and report cosine.
- **`Qwen3RunnerBuilder::packed_weights(true)`** + CLI `--packed`
  flag — high-level entry to the packed-weights path. Builds the
  packed prefill graph, uploads K-quant params as U8 byte tensors
  via `set_param_typed`, exposes `Qwen3Runner::predict_logits` for
  a single forward AND `Qwen3Runner::generate_packed` for
  streaming via repeated prefills. `generate(...)` auto-routes to
  the packed path in packed mode, so the same caller-side code
  works in both modes. Trade-off: each generated token costs one
  full prefill (no decode-graph KV cache in packed mode yet —
  bucketed decode-graph machinery is still F32-only); throughput
  is ~`max_seq` × slower than the F32 streaming path but memory
  stays packed — the only path that fits 14 B+ Q4_K_M GGUFs on
  commodity Macs today.
- **Layout bug fixed** in CPU `Op::DequantMatMulGguf`: the dequant
  output is `[n, k]` row-major (GGUF byte order), not `[k, n]` —
  the original arm called `sgemm` which silently produced wrong
  outputs for `n > 1` cells. Now uses `sgemm_bt` (B transposed).
  Pinned by a new `dequant_matmul_q8k_correct_layout_for_n_gt_1`
  regression test that's specifically picked to fail under the
  old layout.

### Known limitations

- **Qwen3.5 / Qwen3.6 (`qwen35`) hybrid gated-DeltaNet + attention**:
  the unsloth/froggeric `Qwen3.5-0.8B-MTP-GGUF` and
  `Qwen3.6-27B-MTP-GGUF` files both tag `general.architecture =
  "qwen35"` (Qwen3-Next style: gated DeltaNet "linear attention"
  trunk layers interspersed with standard attention every
  `full_attention_interval`, plus an MTP head). End-to-end forward
  pipeline shipped this release:
  - `Op::GatedDeltaNet { state_size }` — new IR op + CPU
    autoregressive scan kernel mirroring
    `delta-net-base.cpp::build_delta_net_autoregressive`. Parity-
    tested against a scalar reference + per-batch state-reset test
    (`rlx-runtime/tests/cpu_gated_delta_net_parity.rs`, 2/2 green).
  - `Qwen35Config::from_gguf` + `Qwen35Weights::from_loader{,_packed}`
    — full per-layer tensor bundle. Auto-detects linear-attn vs
    full-attn layers from `full_attention_interval`; loads the MTP
    layer's NextN `eh_proj` / `enorm` / `hnorm` / optional
    `embed_tokens` / `shared_head_*`. `MatWeight::{F32, Packed}`
    enum routes K-quant matmul weights through `Op::DequantMatMul`
    when `from_loader_packed` is used.
  - `build_qwen35_graph_sized` — full prefill IR: gated-DeltaNet
    trunk (norm → joint qkv+gate split → α/β/dt + softplus gate
    → unrolled k=4 depthwise causal conv → SiLU → q/k/v split →
    L2-norm → GQA-repeat → `Op::GatedDeltaNet` → silu(z)-gated
    norm → `ssm_out`) + every-`full_attention_interval` standard
    attention block (joint Q+gate, sigmoid-gated attn output) +
    optional MTP head. 2/2 basic tests green (graph builds,
    executes, produces finite logits on both trunk + MTP outputs).
  - `Qwen35Runner` / `Qwen35RunnerBuilder` — mirrors the
    `Qwen3Runner` API; `.packed_weights(true)` opts into the K-
    quant in-arena path. `.generate(prompt_ids, n_new, on_token)`
    runs autoregressive greedy generation via repeated prefills.
  - `rlx-run qwen35` CLI subcommand + `examples/run_qwen35.rs`
    end-to-end. Flags: `--packed`, `--mtp`, `--max-tokens N`,
    `--prompt-ids 1,2,3`, `--max-seq N`.
  - Deviations from the llama.cpp reference (flagged for the
    next-slice parity oracle): standard per-axis RoPE substituted
    for the rope-sections MRoPE; depthwise k=4 conv unrolled into
    narrow+mul+add (no `Op::Conv`); per-batch state reset (no
    decode-time state cache).
  Memory: F32 dequant path needs ~1.5 GB for 0.8B (fits) /
  ~65 GB for 27B (doesn't fit). Packed path drops 27B to ~16 GB
  (fits) by keeping K-quant bytes in the arena. Numerical parity
  vs llama.cpp on a real GGUF is the next milestone.

  **Packed-loader perf**: zero-copy upload path. `take_packed`
  used to `.to_vec()` each K-quant tensor's bytes (~16 GB of
  memcpy on 27 B Q4_K_M). New flow:
  `take_packed_metadata` records `(scheme, shape)` only,
  `MatWeight::Packed` holds the loader key, and the runner uploads
  via `loader.tensor_bytes_borrowed(key) → compiled.set_param_typed`
  — bytes flow straight from mmap into the arena, no intermediate
  Vec. Also: reuse the loader's already-parsed `GgufFile` for
  `Qwen35Config::from_gguf` (was re-parsing 800+ tensor headers,
  ~10 s saved on 27 B). Builder/runner now log per-phase timing
  via `eprintln!` so future regressions surface.
- **`Op::DequantMatMul` on Metal** still falls through to the
  per-op thunk path; the GGUF schemes only have CPU lowerings
  today. On Apple GPUs the F32-load path is the working option
  until the native Metal Dequant kernel lands.
- **Streaming decode tok/s** in `Qwen3Runner::generate` recompiles
  per token in `stream(true)` mode — the bucketed compile cache
  doesn't get hit until the second pass. Fix in 0.2.0: callback
  threaded through `Qwen3Generator::generate_cached` so a single
  compile covers the whole `n_new` decode loop.
- **Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q1_0**
  GGUF formats are not decoded. Files containing them raise a
  clean "dequant for {type} not implemented yet" error.
- **27 B-class GGUF on Mac**: requires the Metal `Op::DequantMatMul`
  kernel above (108 GB F32-dequant footprint doesn't fit anywhere
  affordable). Models up to ~8 B Q4_K_M load and run today on a
  32 GB unified-memory Mac.
- **MTP heads** are now loadable end-to-end on
  `unsloth/Qwen3.6-27B-MTP-GGUF`-style files: pass
  `--use-mtp` (CLI) or `.use_mtp(true)` (runner builder) to flip
  the `GgufLoader::include_mtp` visibility; MTP tensors are drained
  into the generator's weights cache and a diagnostic logs how many
  heads were captured. Direct access via `GgufLoader::take_mtp(name)`
  is also exposed. The base generation path still runs single-token
  decode (the speculative + verify loop that would *use* the heads
  is the follow-up); inference succeeds either way.

### Internal

- `Op::DequantMatMulGguf` thunk variant added in `rlx-cpu` to
  carry the GGUF scheme through scheduling + VJP recompute paths
  cleanly.
- Workspace member layout unchanged.

---

## [0.2.0] — 2026-05

The first release with end-to-end **Qwen3 LM inference** on Apple
Silicon (safetensors + GGUF, F32, parity-checked against the
HuggingFace reference), a high-level **`rlx::run`** runner API, a
**`rlx-run`** CLI, and **GGUF K-quant dequantization** baked into
`Op::DequantMatMul`.

## [0.1.0] — 2026-04

Initial release. Tracked at [git history root].

[Unreleased]: https://github.com/MIT-RLX/rlx/compare/v0.2.16...HEAD
[0.2.16]: https://github.com/MIT-RLX/rlx/compare/v0.2.15...v0.2.16
[0.2.15]: https://github.com/MIT-RLX/rlx/compare/v0.2.14...v0.2.15
[0.2.14]: https://github.com/MIT-RLX/rlx/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/MIT-RLX/rlx/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.12
[0.2.11]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.11
[0.2.10]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.10
[0.2.9]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.9
[0.2.8]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.8
[0.2.7]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.7
[0.2.6]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.6
[0.2.5]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.5
[0.2.3]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.3

## License

MIT OR Apache-2.0.
