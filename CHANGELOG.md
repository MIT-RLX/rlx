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

### Added

- `rlx-spectral`: spectral front-ends in pure Rust, generic over precision
  (`Float`: f32, f64): `FftPlan` (Stockham radix 8/4/2/3/5, Bluestein for other
  primes) and `RealFftPlan`; Hann / Hamming / Povey windows; framing
  (centered reflect, snip-edges) with Kaldi DC removal / preemphasis; Slaney /
  HTK / Kaldi mel filterbanks in Hz or mel space, applied sparsely; the
  `WhisperLogMel` and `KaldiFbank` (SeamlessM4T `input_features`) pipelines.
  An n = 400 FFT (whisper's) runs 2.8 µs (rustfft: 3.5); powers of two are
  still 1.75–2.1× behind rustfft's SIMD kernels.
- `rlx-fastmath`: accurate vectorized f32 `exp` / `log` / `tanh` (NEON;
  ≤ 1.07 / 0.84 / 1.33 ulp over all 2^32 inputs, NEON lanes = scalar). exp
  0.67 ns/element against Accelerate's `vvexpf` 0.77 and the platform libm
  2.5; rlx-cpu re-exports it as `rlx_cpu::fastmath`.
- rlx-cpu feature `parity`: rlx-libm (optional now) for the accurate vector
  math outside Accelerate builds and the FFT twiddles. Default: `fastmath`
  and the pure-Rust `libm` crate.
- rlx-libm feature `libm` (forwarded by rlx-special, rlx-signal,
  rlx-pocketfft, rlx-rng, rlx-linux-ref; rlx-cpu takes `parity` instead): Apple's libm for `reference` (and the
  merged `__sincos_stret`, now `reference::sincos`, that pocketfft's
  `AppleSinCos` twiddles use). **Default is now the pure-Rust ports on every
  target**: on macOS, results equal the Linux reference's algorithms instead of
  Apple's (they differ in 0.05–38 % of inputs by ≤ 7 ulp, `examples/
  ports_vs_platform`). The SciPy-on-macOS fixture suites of
  rlx-special / rlx-signal / rlx-pocketfft enable `libm` for their tests.
- `rlx-cpu` feature `pure-rust` (with `rlx-ir/pure-rust`): no native library in
  the build even when `blas` is on — no Accelerate / OpenBLAS / MKL link, vector
  exp/tanh/log in Rust instead of vForce, BNNS compiled to its stubs. GEMMs go
  to the SME2 kernel (`amx-sme`, now the default there; `RLX_CPU_SME=0` turns it
  off) or the `gemm` crate. The BLAS-less `cblas_sgemm` shim routes strided /
  transposed products to SME2 too. On an M4 Pro, SME2 runs whisper-sized GEMMs
  5–10× faster than the `gemm` crate and about 2–3× slower than Accelerate.
- `rlx-text` builds HF `tokenizers` with its pure-Rust
  `fancy-regex` backend on every target (Oniguruma, a C library, is gone):
  LLaMA 3.2's tokenizer gives identical ids to Python `tokenizers` 0.22.2 on
  46783 texts / 2.37M tokens, Unicode stress strings included.
- Distributed inference lays a step out by its IO as well as its compute.
  `split_function_cut` moves each cut up to 25% of its share to where the
  fewest bytes cross. `StepLayout::argmax` reduces logits to token ids before
  splitting. `exchange_speeds_links` / `probe_links` measure every pair's
  latency and bandwidth at the speed exchange. With `StepLayout::links`,
  `InferJob::start_with_layout` orders the workers by the estimated step time;
  stages carry `prev` / `next`. `dist_node ARGMAX=1 TOPOLOGY=1`. A Qwen3.5
  verify step over 5 nodes sent ~12 MB a pass across its hops and now sends
  ~40 KB.
- Distributed inference calibrates itself: stages report their compute time
  in the values they pass on, and `InferJob::calibrate` re-plans the layout
  (order and cuts) from measured per-rank speeds. It re-deploys the stages
  mid-job when the estimate improves (`StagePlan::index`, a `RESTAGE`
  marker, `deploy`). `dist_node TOPOLOGY=1` runs it before decoding.
- Mesh links over Bluetooth LE: `rlx_driver::ble` (feature `ble`, on the
  `webbluetooth` crate) bridges TCP to an L2CAP channel. `serve` is the node
  side, a node given `RLX_BLE_SERVE`; `bridge` is the launcher's
  (`via = "ble"`, `cargo-rlx --features ble`). The iOS app (`BLE=1`) carries
  `NSBluetoothAlwaysUsageDescription`. mm4 ↔ iPad: ~125 ms one-way, ~24 KB/s.
  The link probe now times 64 KiB first and 4 MiB only on a link fast enough,
  rather than taking minutes on this one.
- `scripts/publish.sh --dry-run` changes nothing in the tree (its format gate
  names the files a real run would format) and runs one `cargo publish
  --workspace --dry-run`. That verifies every crate, non-leaf ones included,
  against the workspace's unpublished versions; older cargo falls back to the
  per-crate walk. Dry and real runs check every package against crates.io's
  10 MB limit (a real run before uploading anything), and the dry run lists
  crates whose version is already on crates.io. Tiers now include rlx-libm,
  rlx-fastmath, rlx-linux-ref and rlx-spectral. rlx-linalg's package leaves
  out its two 34 MB NumPy fixtures, which put it at 14.3 MiB; rlx-libm no
  longer names a README it doesn't have.
- `ios/deploy-node.sh`: one command builds the node app and installs it on
  every attached iOS device, or the ones named. It builds the Rust node for
  devices only (new `build-xcframework.sh --device-only`, one build instead
  of three for iOS), builds the app once for any iOS device, and keeps the
  app build incremental. `build-xcframework.sh` no longer stops silently
  after packaging when a simulator target was not built.
- Persistent distributed workers: `cargo rlx dist --persist` (or
  `job.persist`, `MODE=model`) keeps every worker rank up between runs and
  launches only rank 0 again. A worker started with the same settings is
  found and reused, its log followed to the end of its part in the job;
  `--restart`, `--build` and `--sync` start them again. A worker (`dist_node
  PERSIST=1`, the iOS app with `PERSIST=1`) keeps its stage, compiled and
  with its weights. `deploy` waits for each worker's answer and sends weights
  only where the plan differs (`InferJob::kept`). The speed exchange plans by
  the last job's speeds while every rank is on the same device and no
  worker's new measurement is more than 3× off them, so probe noise (2× on
  an iPad) does not move the split. A worker drops a kept stage before
  receiving a different one (holding both got an iPad app killed), and a
  calibration restage reuses an unchanged stage instead of resending it.
  Calibrated layouts carry over to the next job: plans record the speeds and
  order they were laid out by (`StagePlan::layout`, `PlannedLayout`), workers
  remember them, and a steady exchange returns them (order in
  `Links::order`). Speeds are planned at f32 precision, the exchange's.
  New `InferJob::calibrate_for(…, steps)`: a re-layout is made only where
  its estimated saving over `steps` requests beats moving the changed
  stages' weights. `dist_node` uses it and reports the stage deploy time
  apart from calibration.
- A mesh dial now gives up in real time: each connection attempt is bounded
  (2 s), and the retry budget (`RLX_CONNECT_WAIT`, 10 s) is wall-clock time
  rather than a try count. A peer whose SYNs went unanswered blocked each
  plain `connect` for over a minute, so a 5-node job hung for its whole
  30-minute timeout instead of failing; the error now names the address.
- A mesh rank accepts until every expected rank has said hello, not until a
  count of connections arrived. A persistent worker from the run before,
  still dialing until it was replaced, announced its rank before the new
  process did; counted twice, it left rank 0's connection unaccepted, and
  rank 0 waited at the first barrier until the job's timeout. A rank heard
  again replaces its earlier connection (whose closing marks no one lost),
  and a connection that sends no valid hello in 10 s is dropped rather than
  failing the mesh.
- `cargo rlx dist --persist` stops every worker a run replaces, on every
  host and device, and waits until each is gone before it starts any. Started
  side by side, a new worker could dial the process it was about to replace
  on another node (an iPad's old app while `devicectl` was still relaunching
  it), hand that one its hello, and then wait forever for the rank it had
  lost. (`Launch::stop`; on iOS, `devicectl … terminate --kill` by the app's
  process.)
- `RLX_STAGE_KEEP_WARM=1`: a distributed worker keeps its device busy while
  its stage waits for the next request. An integrated GPU otherwise powers
  down between requests. amd's 780M stage went 21.1 → 11.2 ms a step (31.6 ms
  before `power_dpm_force_performance_level = high`); the iPads and a
  discrete GPU didn't gain, so it's opt-in, per host.
- ROCm: a carried-state `GatedDeltaNet` (one kernel) no longer rules out
  hipGraph capture (`RLX_ROCM_EXEC=graph`), which had excluded every Qwen3.5
  stage. It didn't speed up amd's stage, which wasn't launch-bound.
- GGUF decode GEMV (CUDA/ROCm): each row's 32 lanes sum with register
  shuffles instead of shared memory and five block barriers, and a
  16-byte-aligned `x` is read with vector loads. Bit-identical. New
  `rocm_gguf_gemv_bench` (µs and weight GB/s per Qwen3.5 shape, an output-bit
  hash, `--gap-ms` / `--warm` / `--spin` for idle-GPU effects) and a ROCm
  Q8_0 test against an f64 reference. New
  `RLX_CONNECT_WAIT` (seconds a mesh dial keeps trying, `0` = no limit); the
  BLE advertisement starts once per process.
- `cargo rlx dist`: a `via` value naming `local` (`msi = "local"`) routes that
  link through a relay the launcher runs, the host dialing it (an iPad over
  its cable, onward over the Mac's wired link).
- `cargo rlx dist`: a host's `via` table (host → address) says how its ranks
  reach others; `via = "usb"` joins `local` and a wired iPad over the cable,
  resolved at launch to the iPad's USB link-local address and the Mac's end.
  `RLX_MESH_REVERSE` (`0-1,0-4`, set by the launcher for those pairs) has the
  higher rank of a pair open its connection: an iPad accepts none over USB-C
  but can open one.
- Speculative decoding across the cluster: a `generate` entry's `verify`
  step (`generate.verify`: its function and token count) runs the last token
  and a draft head's guess at once. `InferJob::decode_verify` keeps both rows
  of a right guess and retries a wrong one with the right token, so the tokens
  are greedy one-token decoding's; `decode_local_verify` is the unsplit
  reference, `DecodeEntry::feeds_n` a multi-token step's inputs. A stage step
  run under `KEEP` holds its state update until the next step keeps or drops
  it, and a step may append several cache rows. `dist_node MODE=model
  SPECULATE=1`. Qwen3.5-0.8B Q8_0 over Metal + CUDA + ROCm, same tokens as one
  CPU: 1.2–1.5x on prose and on code (the range is load on the Metal node).
- CUDA and ROCm `dequant_matmul_gguf_gemv_rows`: Q8_0 / Q4_K / Q5_K / Q6_K
  matmuls of 2–4 rows (a verify step's) through the element-strided GEMV, each
  weight chunk decoded once for every row, each row bit for bit the `m = 1`
  kernel's (tested against the reference and against each row alone). They
  went through the f32 dequant slab and BLAS before.
- Metal runs dequant matmuls of 2–4 rows as the GEMV per row by default
  (`RLX_METAL_DEQUANT_FORCE_GEMV` now only adds larger `m`): a Qwen3.5 verify
  stage's matmuls ran 2.5x faster that way, and match one-token decoding.
- A `generate` entry may name a draft head's logits (`draft_logits`: the
  output holding, e.g., Qwen3.5's MTP guess at the token after next).
  `model::distributed::DecodeEntry::draft` reads it, and
  `decode_local_drafts` runs the unsplit reference with each step's draft.
- `rlx_libm::expf` and `reference::expf`: glibc's `expf` (the `-mfma` x86-64
  variant), bit-exact against glibc on all 2^32 inputs.
- `RLX_WGPU_DUMP_WGSL=<dir>` (native): write every shader module rlx-wgpu
  creates to `<dir>/<hash>.wgsl`, to put the exact sources through another
  compiler (e.g. the browser's).
- `rlx_libm::reference::sincosf` (Apple's `sinf`/`cosf` on Apple, glibc's port
  elsewhere).
- rlx-wgpu test `no_wgsl_declaration_uses_a_reserved_word`: WGSL reserves
  words such as `active` that naga accepts and Tint rejects.
- Nodes advertise themselves over Bonjour/Avahi as `_rlx-node._tcp` (TXT:
  platform, devices, mem_gb, version): `rlx_driver::mdns`,
  `dist_node MODE=advertise`, `cargo rlx nodes [advertise]`, and
  `addr = "mdns"` hosts in cluster files. Rendezvous servers advertise
  `_rlx-rdv._tcp`, which `RENDEZVOUS=auto` browses. The iOS node app
  publishes its own record through `NetService`.
- CUDA and ROCm run decode-shaped (`m = 1`) GGUF matmuls for Q8_0, Q4_K, Q5_K
  and Q6_K with one fused kernel (`dequant_matmul_gguf_gemv_elem`, shared
  `.cu`): 32 threads per row, 8 rows per block, 8 weights per thread per
  step. Q8_0 and Q5_K (and Q6_K on ROCm) used to unpack the whole weight to f32
  every step, and ROCm's Q4_K kernel kept 4 of 128 threads busy at k = 1024.
  On a Radeon 780M a 3584×1024 matmul went from 15.7 ms to 0.21 ms (Q6_K),
  21 ms to 0.28 ms (Q5_K), 2.0 ms to 0.28 ms (Q4_K). Qwen3.5-0.8B split over
  mm4 + msi + amd + iPad: Q8_0 4.8 → 13–15 steps/s, Q4_K_M 2.3 → 11.1.
  `RLX_GGUF_GEMV_ELEM=0` restores the older Q4_K/Q5_K/Q6_K paths. Parity:
  `dequant_matmul_gguf_gemv_{cuda,rocm}_matches_cpu`.
- `cargo rlx dist` starts an `ios_device` host's app with `-nodename` when
  its `addr = "mdns"` record is not on the LAN yet, then resolves it.
- `RLX_STAGE_PROFILE=1`: distributed inference stages report state in / run /
  state out per step. Stage weights ship as bf16 when that is exact (a BF16
  checkpoint), f32 otherwise.
- Distributed decoding of stateful steps: `InferJob::start_with_state` /
  `decode`, `decode_entry`, `decode_local`, and `StagePlan::state`. Each
  stage keeps its own layers' recurrent state and KV cache. `dist_node
  TASK=generate` uses it for a package whose `generate` entry has `state`.
  Qwen3.5-0.8B ran on mm4 + msi + amd + iPad, matching one CPU
  (logits 3.3e-6). Workers and rank 0 report per-step compute, send and
  wait times, with the first step (kernels compiling) reported apart.
- Distributed stages ship each param in the dtype its graph declares:
  bf16/f16 as stored, packed GGUF blocks (Q8_0, Q4_K_M behind
  `DequantMatMul`) as raw bytes, bound with `set_param_typed`. Qwen3.5-0.8B
  Q8_0 and Q4_K_M ran split over mm4 + msi + amd + iPad with tokens matching
  one CPU.
- `TrainOptions::lr_scaling` (`none` | `sqrt` | `linear`) and `warmup` scale
  the learning rate with each update's samples (`--lr-scaling` in
  rlx-vision-bench, `LR_SCALING` / `WARMUP` in dist_node). CIFAR-10 with
  CPU+GPU+ANE lanes went from ~54% to 69.75% in 6 epochs, against 71.77% for
  GPU only.
- A rank's training lanes work-steal micro-batches from one queue, so a lane
  slowed mid-run hands its share to the others. The sum stays in batch order
  up to 256 MB of buffered gradients.
- `dist_node VERIFY=1` adds a teacher-forced logit comparison (worst relative
  error, plus the reference's margin wherever the top token differs).
  rlx-vision-bench builds with Core ML on macOS and evaluates on the ANE.
- `TrainJob::eval_loss` and `dist_node MODE=model TASK=train
  HELDOUT=<text.txt>`: held-out loss and perplexity before and after a
  distributed run.
- `cargo rlx dist` passes an iOS rank's `RLX_*` variables to the app
  (`devicectl --environment-variables`). `RLX_TRAIN_LANES = "metal,cpu"` runs
  the iPad's CPU and GPU together, like every other node. MNIST on 4 machines
  with CPU+GPU lanes ran at 6.6× the throughput of GPU-only, and reached 95.87%
  in 6.5 s with the learning rate scaled.
- `dist::trainserve_worker`, and rlx-ffi `mode = "trainserve"`: a phone is a
  worker for cluster-options training jobs (`rlx-vision-bench --coordinate
  --cluster`). The iOS app stays open between jobs. The launcher reads its
  `-logfile` through `devicectl device copy from` instead of `--console`,
  which killed the app when the session ended.
- An iPhone/iPad attached to the launching Mac can be a cluster host
  (`ios_device` in a cluster file). Its rank runs the iOS node app through
  `devicectl`. rlx-ffi gained `mode = "model"` (feature `model`), so a phone
  serves `dist_node MODE=model` inference and training, plus
  `rlx_node_advert`. A real iPad ran a Qwen3-0.6B pipeline stage and a
  SmolLM2-135M LoRA training lane beside mm4, msi and amd.
- `model::distributed::{worker, exchange_speeds, gemm_gflops}`: one source
  for the collective sequence every `MODE=model` rank runs.

- Training lane fallback in `run_train_group`: a device lane that cannot
  compile or run the graph (GPU out of memory, busy with another job, a
  backend limit) is dropped and its micro-batches move to another lane, or to
  a CPU lane when none is left, with the cluster's plan unchanged.
  `RLX_TRAIN_LANE_FAULT` injects a failure for tests. A `model::distributed`
  inference stage whose GPU panics is likewise rebuilt on the CPU and the
  request is retried.
- `model::distributed::{train_job_text, text_samples, package_codec}`:
  `dist_node MODE=model` takes a text `PROMPT` and a text `DATA=*.txt`
  through the package's own tokenizer, and prints generated text.

- `cargo rlx dist <cluster.toml>`: one command launches a distributed job on
  every rank of a cluster file (several ranks per host, one per device), with
  `--sync`, `--build`, `--retries`, `--timeout`, prefixed output, and the whole
  job stopped when any rank fails. No processes are left behind on remote hosts.
- `rlx_runtime::dist::run_train_group` + `TrainOptions`: gradient accumulation,
  a bf16/f16 gradient wire, compute/communication overlap, per-lane
  throughput balancing, local SGD, ZeRO-1 momentum sharding, and per-epoch
  checkpoint/resume. Selected by `TrainSpec.options`; `dist_node
  MODE=trainserve` and `rlx-vision-bench --coordinate --cluster …` use it.
  `device: "all"` now means one lane per physical unit; a list pins lanes;
  `RLX_TRAIN_LANES` overrides per node. On mm4 + msi + amd, MNIST reached
  96.8% in 16.8 s against 95.35% in 24.6 s before.
- Streaming pipeline inference: `planner::split_layers`,
  `planner::recommend_inference`, `dist::pipeline_run` /
  `serve_stage_stream`, and 16-bit activations (`send_activation_as`).
  `dist_node MODE=pipestream` runs it planned from the live mesh: 2.65× over one
  batch across 7 devices.
- TCP and UDP rendezvous (`RENDEZVOUS=tcp://…` / `udp://…`, `RANK=auto`) and
  zero-config `RENDEZVOUS=auto` (rank 0 announces itself by broadcast +
  multicast). Collectives over QUIC/UDP with `TRANSPORT=quic`
  (`IrohTransport::connect_lan`, `iroh` feature). Every combination was
  verified on the 3-node mesh.
- Train and infer in any number format across a cluster:
  `TrainOptions::number_format` and `StageSpec::number_format` take any
  `NumberFormat` (`f64`, `f64x2`, `tf32`, `bf16`, `f16`, `f8e4m3`, … ,
  `weights/activations`), with f32 master weights and loss scaling for narrow
  formats that backs off on overflow. `rlx-vision-bench --number-format`,
  `dist_node PS_FORMAT=`. `tf32` is a new `FloatFormat` alias (e8m10).
- Gradient wire `WireDtype::Auto` (the new default) uses bf16 with error
  feedback when the measured link is bandwidth-bound, else f32. Activation
  `Auto` sends f16 per message when the values fit, else bf16. Metrics now
  report `wire`, `wire_bytes`, and `wire_rel_err`.
- Distributed training and inference of **any** `.rlxp` package:
  `rlx_runtime::model::distributed` (`describe`, `train_job` → `TrainJob::run`,
  `InferJob::{start, run, generate}`, `serve` for workers) and
  `dist_node MODE=model` (`TASK=info|infer|generate|train`). Inference splits
  the function by measured rank speed and streams requests. Training builds a
  `TrainSpec` from the package trainer's options (LoRA, globs, SAE). New:
  `TrainSpec::frozen`, an AdamW `TrainOptions::optimizer` (ZeRO and
  checkpoints included), integer safetensors datasets, `#i` output names, and
  `TrainMetrics::skipped`. Qwen3-0.6B generates across Metal + CUDA + ROCm
  identically to one CPU.
- `CompileOptions::half_casts` / `RLX_HALF_CASTS=widen|round`. Execution
  widens half graphs to f32, so a `Cast` to f16/bf16 kept the f32 value; with
  `round` it rounds to the format first, on every backend. The default
  (`widen`) is unchanged. Also `TrainOptions::half_casts` and
  `StageSpec::round_half_casts()` for distributed jobs.
- `dist_node MODE=dtypes`: a per-device table of GFLOP/s and error vs f64 for
  f32/f16/bf16 matmuls, plus an accumulation-order probe and a cast-rounding
  probe.
- f64 training runs end to end on CPU (native), Metal, CUDA, ROCm, and wgpu
  (double-single). New: CPU f64 kernels for `Mean`, softmax cross-entropy,
  its backward, and `relu_backward`. Double-single forms for ReLU, `Mean`,
  softmax cross-entropy (log via one Newton step on an f32 seed), its
  backward, and `relu_backward`.
- `ProcessGroup::reduce_scatter_owned` / `all_gather_owned` / `owned_range`.
- `RLX_COLLECTIVE_TIMEOUT`: bound any single collective wait.

- IPv6 for distributed nodes. `PEERS`, `dist_job` `peers`, and the
  `rlx-distributed` hostfile/RPC accept `[v6]:port` and link-local addresses
  scoped by interface name (`[fe80::1%en0]:port`), via the new
  `rlx_driver::resolve_addr`. Mesh discovery (`DISCOVER=1`) also announces by
  IPv6 multicast (`ff02::7266`) on a dual-stack socket. The star coordinator
  listens dual-stack, `DISCOVER_HOST` takes an IPv6 literal, and `local_ip()`
  falls back to IPv6. Windows keeps IPv4 wildcard binds. Verified on a
  Metal + CUDA + ROCm 3-node mesh over IPv4, global IPv6 and link-local, with
  identical pipeline (5.22e-8) and MNIST data-parallel (95.35%) results. See
  [docs/distributed.md](docs/distributed.md#ipv6).
- Opt-in fast GPU arithmetic for wgpu, including the browser's WebGPU:
  `rlx_runtime::set_gpu_fast_math(true)` (or `rlx_wgpu::set_fast_math`),
  called before the first GPU compile. It swaps the correctly-rounded `/`,
  `sqrt` and `1/sqrt` for the native WGSL operators; `fmod` stays exact.
  Exact remains the default. The switch costs bit-identity but not
  accuracy:

  | Device | `/` off | `sqrt` off | EEGNet score bits | Kitten corr vs CPU |
  |---|---|---|---|---|
  | M4 Pro | 27% of quotients, ≤1 ULP | 27% of roots, ≤1 ULP | `3f192f68` | 0.7470 (exact 0.7468) |
  | RTX 3080 Ti | 29% of quotients, ≤2 ULP | 17% of roots, ≤1 ULP | `3f192f68` | 0.9997 (exact 0.9997) |

  EEGNet's score bits are the same as with exact math and on the CPU. No
  speedup was measurable on EEGNet in either run order (3.67 s vs 3.68 s on
  the M4 Pro; 6.4 s both ways on the RTX). The tests are
  `fast_math_option` and `every_helper_shader_carries_the_exact_block_once`.
  The second pins the string swap the switch relies on.

### Changed

- rlx-wgpu on discrete Vulkan/DX12 (NVIDIA) now runs elementwise, norm and
  structural ops on the GPU, as on Metal. They used to run on the CPU there as
  a workaround for a KittenTTS failure whose real causes, the stripe-addressing
  bugs below, are now fixed. On an RTX 3080 Ti, EEGNet takes 6.4 s instead of
  295 s, with the same score bits (`3f192f68`, as on the CPU), and Kitten's
  waveform correlates 0.9997 with the CPU. `RLX_WGPU_DISCRETE_HOST=1` restores
  the old routing. Arenas split past the binding limit still host these ops
  unless `RLX_WGPU_SHARD_GPU=1`.
- `WireDtype::Auto` sends activation messages up to 1 MiB as f32 (was f16
  whenever in range). They are latency-bound, so f16 saved little, and its
  rounding flipped a near-tied greedy token in a split Qwen3 decode.

### Fixed

- CoreML: the compiled-model cache (`$TMPDIR/rlx-coreml-cache`) never hit.
  It is keyed by a hash of the serialised model, and prost encoded the
  protobuf map fields in `HashMap` order, which differs per process. So every
  process wrote the full `.mlpackage` again, recompiled it and added a new
  cache entry; 472 entries (24 GB) had built up. The protobuf maps are now
  `BTreeMap`s (`.btree_map(["."])`). V-JEPA2 ViT-g's first CoreML GPU run
  in a new process went from 106 s to 12 s. The shim now publishes a cache
  entry with an atomic rename. Without that, two threads compiling the same
  model could see a half-copied entry and delete it under the writer; that
  failed `coreml_quant` in 2 of 6 cold runs, and 0 of 5 after the fix.
- CoreML: a weight that is only read through a `cast` to f16, as mixed
  precision leaves weights, is baked as an f16 blob const and the cast becomes
  an alias. Before, CoreML folded each cast into an inline f16 tensor, and
  MPSGraph printed every one as text on each load
  (`MILToMLIRRewriter::getLocationForOp` → `IROperation::ToString`); that was
  about 40 s of V-JEPA2's load.
- CoreML f16 mode (`RLX_COREML_F16=1` / `Precision::F16`): scalar operands
  built as f32 immediates (LayerNorm's eps, rsqrt's epsilon, ...) are now
  emitted as f16. Before, CoreML refused to load the model: "In 'ios16.add'
  operations, tensors parameter x[0], parameter y[0] ... must have the same
  data type". f16 mode is still incomplete for attention graphs: about 100
  lowering sites declare f32 output shapes, so V-JEPA2 then fails at an
  attention `reshape`. Mixed precision (`PrecisionPolicy`) works.
- Metal: a `FusedMatMulBiasAct { Relu }` lowered to MPSGraph dropped the ReLU
  (`mps_graph_lower` matched it with the `None` arm), so any graph that took the
  full-MPSGraph path returned matmul + bias with its negatives intact — off by
  ~1.0 relative. Pinned by `mpsgraph_fused_linear_applies_relu`
  (`rlx-metal/tests/attention_block_kernels_parity.rs`).
- SME2 kernels (`rlx_sme_fmopa_panel`, `_smopa_32x32_i8`, `_bfmopa_32x32`)
  clobbered the caller's d8–d15: `smstart` / `smstop` zero every SIMD&FP
  register and those are callee-saved under AAPCS64, so floats live across
  the call came back as 0 (the SME tests failed with a tolerance of 0). They
  are now saved around streaming mode.
- `transposed_fallback_matches_the_scalar_loop_bit_for_bit` asserted bit
  equality for shapes `portable-gemm` sends to the blocked kernel; those get an
  error bound now, the loop-nest shapes stay bit-exact.
- A mesh rank with a static peer list listened only on its own entry's
  address, so a peer that reached it by another (an iPad over USB-C, a
  link-local IPv6) was refused. `TcpTransport::bind` now listens on every
  local address at that port.
- `cargo rlx dist` and iOS: a peer list holding an IPv6 address reached the
  app empty, since `UserDefaults` parses each launch argument as a property
  list and a leading `[` is not a string; such values now go as quoted plist
  strings. Its log polling also no longer prints `head: illegal line count`
  when no new lines arrived.
- rlx-unfuse dropped a `MaskKind::Bias` attention's mask when it promoted the
  attention to rank 4 (ROCm and every rank-4-only policy): the kernel then
  indexed a fourth input that was not there, so a multi-token verify step
  failed to compile on ROCm and ran on the CPU. The bias now goes through like
  a custom mask.
- rlx-flow `gather_last_token_dynamic` declared its own I32 `last_token_idx`
  input next to the one a flow had declared under the same name; only one is
  bound, so whatever read the flow's (Qwen3.5's MTP head) read zeros, i.e.
  position 0. The stage now reuses a declared input, cast to I32.
- Browser WebGPU: an activation arena of 2 GiB or more misread its upper
  offsets as weight-buffer offsets ("weight-tagged off without
  weight_buffer"): on 32-bit targets `WEIGHT_BUF_TAG` is bit 31.
  `arena_budget::device_arena_limit(Gpu)` reports 2 GiB there, so
  `fit_batch` keeps training graphs under it, and `WgpuArena::from_plan`
  refuses a plan with a tensor (or the scratch zone) starting at or past 2 GiB
  with a message instead.
- Browser WebGPU: every FFT ran as a no-op. `fft_inner` returned early on
  `local_invocation_id` before a `workgroupBarrier()`, which fails WGSL's
  uniformity analysis; Chrome's compiler (Tint) rejects it, and with it every
  entry point in `fft_gpu.wgsl`, while naga accepts it. The FFT output stayed
  zero, so FFT-fronted models (exgb's brainrvq, corticalssm, eegformer)
  embedded every trial to the same vector. Out-of-tile threads now skip the
  work but reach every barrier; native results are unchanged.
- rlx-cpu's FFT twiddles call `rlx_libm::reference::{sin, cos, sincosf}`
  instead of std. On wasm, std is Rust's own libm; the reference is glibc's
  (Linux) and Apple's (macOS) libm, so the browser computes what Linux does.
  Native results are unchanged: a new rlx-libm test checks, on Linux, that the
  ports equal glibc bit for bit on every twiddle angle up to n = 2^16.
- `CompiledGraph::run_async` / `run_read_outputs_async` (wasm) now do what
  `run` does: upload params passed as inputs, check the inputs, prepare and
  feed staged weights (awaiting the prepare graph), and strip a tap's outputs
  and hand them to it. They used to pass the inputs straight to the backend.
- A distributed split of a stateful step could cut a layer between its state
  update and the update's readers (attention over the appended KV cache). The
  stage kept the update as its cache and never sent it, so the next rank
  failed with "stage input … never arrived", or the job refused the split.
  The split now keeps each state's reads, update and the update's readers on
  one stage, and each stage's share is taken from the cost still left when it
  starts, so a stage held past its share no longer leaves the next one empty.

- rlx-rocm claimed `GatedDeltaNet` but ran it on the host (download, CPU
  recurrence, upload): 512 ms a launch on a 780M for Qwen3.5-0.8B's 16 heads
  × 128². With the state carried, it now launches the shared GPU kernel, as
  CUDA does: amd's half of the model went from 2,171 ms to 65 ms a step.

- Opening a referenced GGUF package spent minutes reparsing the GGUF header
  (with its tokenizer vocabulary) once per param. `PackageWeights` now keeps
  the opened files: Qwen3.5-0.8B's stages ship in 20 s instead of 197–427 s.
- rlx-qwen35's `.rlxp` export refused GGUFs with MTP heads ("refusing to take
  MTP weight"). The export's loader now includes them; the decode step does
  not use them.

- rlx-wgpu shaders failed to compile in Chrome, so every WebGPU graph in
  the browser ran nothing. WGSL makes a float constant outside `f32`'s finite
  range a shader-creation error. Chrome's compiler (Tint) enforces that;
  naga, which every native test uses, silently rounds such a literal and
  accepts it.
  - `3.4028235e38` and `3.40282347e38` are both slightly above `f32::MAX`. That
    hit the exact-division helper and the softmax, scan, scaled-low-precision,
    indexing and UMAP kernels. They now use `0x1.fffffep+127f`.
  - The exact `sqrt`/`fmod` helpers and the double-word `exp` built NaN and
    infinity as constant `bitcast<f32>(0x7fc00000u)`, which Tint rejects as
    non-finite. They now derive them from a runtime value; the bits are
    unchanged.
  - The GLSL division helper's `f32::MAX` is now spelled as its exact decimal.

  `no_wgsl_float_constant_is_outside_the_f32_range` scans every kernel and
  generated shader for both patterns. Found by running exgb in headless
  Chrome.
- A synchronous `run` of a WebGPU graph in the browser waited forever: the
  readback completes only once control returns to the event loop, which the
  synchronous call is blocking (an exgb sweep sat at 0% CPU in Chrome). It now
  panics with the way out: await `CompiledGraph::run_async`, or compile the
  graph for the CPU.
- A Core ML (`ane`) training lane slowed a cluster job about 3×. Core ML
  recompiles on every parameter upload, and the balancer's probe never
  uploaded any, so it planned that lane a full share. The probe now uploads
  the trained params before each timed step, and a balanced plan sits out a
  lane more than 8× slower than its rank's fastest (`SLOW_LANE`). The probe
  also zero-fills params before the first upload, which Core ML requires
  ("missing param"). Unavailable lanes in `RLX_TRAIN_LANES` are named and
  skipped. `ios/build-xcframework.sh --model` builds with unwinding panics, so
  iOS lane fallback works.

- The mDNS advertiser published its host record as `<name>.local`, so
  `NAME=msi` on the machine called msi claimed Avahi's own hostname. Avahi
  settled the conflict by renaming the machine (`msi` became `msi-2`, and
  macOS's `mm4` became `mm4-2`). The record is now `rlx-<name>-<pid>.local`.

- `ios/build-xcframework.sh` exported every Apple platform's deployment
  target at once, and clang refused to compile rlx-coreml's ObjC shim under
  `--apple`. Each target now gets only its own.

- CUDA and ROCm int8 `Quantize` returned -128 for huge positive inputs:
  1e30 / scale was cast to `int` before clamping, and that out-of-range
  conversion gives INT_MIN on NVIDIA. The shared quantize, `QMatMul` and
  `QConv2d` kernels now clamp in float first (NaN goes to 0, matching
  Rust's `as i32`). `wgpu_quant_i8_parity` now holds CUDA and ROCm to its
  exact gates too.
- rlx-cuda stored the int8 quantize and quantized-matmul byte offsets as
  `u32`, which wraps in an arena past 4 GiB. They now fail loudly instead.
- rlx-wgpu gave wrong results once a graph's arena outgrew the device's
  storage-binding limit. KittenTTS's waveform on an RTX 3080 Ti was all
  zeros: a 3.2 GB arena against 2 GiB bindings. Past the limit the arena is
  split into stripes and every parameter is parked in a separate weight
  buffer, and several lowerings did not handle that:
  - The fused `x·W + b` matmul kept operand offsets from before staging
    moved its window.
  - `ConvTranspose2d`/`3d`, the int8 `DequantMatMul`, and the host-delegated
    custom ops (`Step::CustomHost`, e.g. `onnx.QMatMulBaked`) stored a parked
    weight's offset as `u32`. That drops the weight-buffer tag, so they read
    unrelated activations. The first two now run on the host when an operand
    is parked or too large for that path, and `CustomHost` offsets are now
    `usize`.
  - `Compare`, `Where` and `Fma` staged operands into the arena-wide scratch
    area, which lies in another stripe. They also moved their window there
    even when nothing needed staging, so Kitten's attention mask (`Eq` of two
    zero tensors) came out all false. They now stage into the output
    stripe's own reserve and keep the window unless an operand is actually
    outside it.
  - Sixteen other host-lowered ops (`GroupNormHost`, `LayerNorm2dHost`, the FFT,
    LogMel, LSTM, `Reverse`, `ArgMax`/`ArgMin`, `AxialRope2d`, … steps) also
    kept absolute offsets as `u32`; `GroupNorm` with parked gamma/beta was
    off by 3.4. These ops now run as a generic `HostOp` when an operand is
    parked or past 4 GiB. Four custom-op host steps (LLaDA2 gate, UMAP kNN,
    `MsDeformAttn`, collectives) refuse such operands with a panic instead.
  - 31 kernels (`Cumsum`, `TopK`, `Pool`, `BatchNormInference`, the
    scaled-quant ops, the norm/rope/gather backward ops, …) bind only the
    output's window but address operands by absolute arena offset, which is
    right only when that window starts at 0 and holds every operand. Off
    stripe 0, or on a parked param, they read the wrong data (a cumsum of a
    parked param came back wrong in all 128 elements). They now run as a
    `HostOp` in that case.
  - The "bind window clamped on a sharded arena" warning fired for the last
    stripe's whole-stripe window, which only drops addresses past the end of
    the arena. It now fires only when the clamp cuts off live bytes.

  `RLX_WGPU_MAX_BIND_MB` now caps the binding limit the device is created
  with, so this regime reproduces on any adapter.
  `parked_weight_matmul_over_bind_cap` covers each case. It runs once with
  the default policy and once with `RLX_WGPU_SHARD_GPU=1`, which puts
  `Where`/`Fma` on the GPU, and each case fails when its fix is reverted.
- The AOT cache served LIR compiled from a different graph whenever a key was
  reused: `AotCache::compile_hir_cached`/`compile_graph_cached` trusted the
  caller's key alone. KittenTTS's probe keys (`probe_cpu`, `probe_dev`) name
  neither the device nor the graph. Two CPU compiles of the same model in
  one process therefore disagreed (waveform correlation 0.73): one loaded a
  stale entry. Each entry now records a fingerprint of its input (device,
  compile options, and the graph's ops, attributes, shapes, edges and
  constant bytes). A mismatch, or an entry from before this change, is
  recompiled and overwritten. Parameter values are not fingerprinted, only
  their names and lengths.

- Metal's attention backward returned zero dQ, dK and dV for `[B, H, S, D]`
  inputs whenever dY was computed earlier in the same graph (any autodiff
  loss). That layout takes the CPU reference, which read the arena before
  the GPU had written dY. It now waits for the GPU first, and so does the
  in-graph quantized-matmul host path, which had the same gap.
  `attention_gradients_through_a_computed_dy_match_the_cpu` checks every
  device.
- rlx-rocm refused any arena past 4 GiB. Now it refuses only graphs with a
  step that isn't audited for 32-bit element offsets, which reach 16 GiB,
  and names those steps. The audit covers the 22 step kinds a transformer's
  LoRA training uses, and it found that the RMSNorm backward steps
  truncated their byte offsets to 32 bits. amd's 780M now trains a Qwen3-0.6B
  LoRA replica (a 6.7 GB arena), matching the CPU-lane run, and
  `arena_past_4gib` (opt-in, `BIG_ARENA_TEST=1`) checks a 4.4 GiB training
  step against the CPU to ~1e-8.

- ROCm on an APU (Radeon 780M, gfx1103) of a multi-node host could fail
  mid-run with `HipError(719)` and a GPU reset. The kernel log showed `MES
  failed to respond to msg=REMOVE_QUEUE … Failed to evict process queues`.
  The kernel's NUMA balancing made amdgpu evict the process's GPU queues,
  and the firmware hung doing it. rlx-rocm now opts the process out
  (`MPOL_LOCAL` before `hipInit`; `RLX_ROCM_NUMA_GUARD=0` turns that off).
  A generate stage run right after training failed 5 times in 6 before
  the fix and 0 in 4 after it. `cargo rlx doctor` reports the host setup as
  RLX-ROCM-003.

- SiLU and Sigmoid backward could return NaN for inputs near −88 in the
  rlxsl-generated kernels (`exp(−x)` overflowed to inf in the auto-derived
  gradient, and inf·0 gave NaN). On Metal this made Qwen3 LoRA training on Metal produce all-NaN
  gradients. rlxsl now emits the closed forms `s·(1 + x·(1 − s))` and
  `s·(1 − s)` with the exponent clamped. `activation_backward_extreme_inputs`
  covers every device.
- A balanced training plan could ask a fast lane for more samples than the
  dataset holds and fail. The shares now shrink together to fit.

- A crashed rank hung the whole job. A receive waiting on a peer whose
  connection ended now fails and names the rank, on TCP and on QUIC. A receive
  from a rank with no link fails at once. A QUIC rank that sent its last
  message and exited could drop it; streams are now flushed and acknowledged
  on drop. A QUIC dial to a peer that is not up yet no longer waits out the
  handshake timeout.
- `RLX_PUSH_DATA` / `RLX_WORKER_DEVICE` no longer warn as unknown variables.
- `run_train_group` now fails, naming the lanes, when every sync's gradient is
  non-finite. It used to skip every update silently.
- `set_param` on a double-single (f64-on-GPU) parameter failed its length
  check: it counted the (hi, lo) pair axis.
- Mesh discovery (`DISCOVER=1`) could hang: a rank that found every peer sent
  5 back-to-back announcements and stopped, so a peer that missed them waited
  forever. Announcements now go out on a 100 ms timer with a `done` flag, and a
  finished rank keeps announcing until every peer reports done (at most 3 s).
- `MarkElementwiseRegions` admitted any broadcast operand whose element count
  divides the output's, but region kernels read it at `gid % len`: a
  per-channel `[1,C,1,1]` against `[N,C,H,W]` was read along the innermost
  axis. With `RLX_KEEP_ELEMENTWISE_REGIONS=1` a batch-norm affine came out
  wrong in 70 of 120 elements on CPU and wgpu alike. Only scalar, dense and
  trailing-slice operands join a region now.
- wgpu and native Vulkan `Div` were not correctly rounded: WGSL and Vulkan
  GLSL promise only 2.5 ULP for `/`, and on Metal (MoltenVK for Vulkan)
  18,681 of 65,556 quotients came out 1 ULP off rlx-cpu. The
  generated binary kernel now corrects the quotient with an exact `fma`
  residual (scaled for quotients below `2^-100`, whose correction Metal would
  flush), and returns `x/0`, `x/inf`, `inf/x`, NaN and overflow unchanged.
- wgpu and native Vulkan `Mod` were `a - b * trunc(a / b)`, wrong once the
  quotient outgrows 24 bits, often with the wrong sign (`-42166.227 %
  0.000107623026` gave `3.0e-3` for `-9.6e-5`). Both are now an exact
  shift-subtract `fmod` that keeps the sign of a zero dividend.
- The wgpu fused elementwise-region kernel had its own copy of the binary op
  math, still using the approximate `/`. Region steps now call the same
  generated `rlx_binary_apply` as the standalone kernel.
- wgpu `Sqrt` and `Rsqrt` were not correctly rounded on Metal (18,667 and
  17,131 of 65,544 off by 1 ULP), and batch norm's `1 / sqrt(var + eps)` was
  1 ULP off on NVIDIA: its Newton step repaired Metal's approximate root but
  moved NVIDIA's exact one (`backward_input_matches_cpu_bit_exactly` failed
  there). New `rlxsl::exact` holds correctly-rounded `rlx_div_rn`,
  `rlx_sqrt_rn` and `rlx_rsqrt_rn` (the hardware result, corrected from exact
  `fma` remainders), shared by the binary, activation and batch-norm shaders.
  All three are bit-exact against rlx-cpu on both Metal and NVIDIA.
- The wgpu fused elementwise-region kernel had its own activation switch:
  `Round` and `Recip` (both region-fusable) fell through to `return x`, so a
  kept region skipped them, and `Sqrt`/`Rsqrt` were uncorrected. Region steps
  now run the generated rlxsl manifest under a new `OpcodeScheme::Region`
  (the `activation_sub` numbering), and every WGSL division inside an
  activation expression (`Recip`, `Sigmoid`, `Silu`, ...) uses
  `rlx_div_rn`. The generated modules no longer embed the exact helpers;
  build.rs prepends them once per shader.
- rlx-onnx-import: a rank-3 `Resize` was viewed through `ensure_nchw_4d`'s
  BLC-vs-NCL size guess, which read KittenTTS's
  `/decoder/decode.3/upsample` (`[1, 1090, 512] → [1, 1090, 1024]`) as BLC,
  transposed it and upsampled the channel axis. It is now viewed as
  `[N, C, 1, L]` in ONNX order (Resize's scales name the axes).
- rlx-onnx-import: an `output_shape_fix` that changes the element count was
  lowered to a `Reshape`, which since reshape started checking element counts
  panicked the whole import (KittenTTS failed to load on every device), and
  before that read past its input. It is now a kernel-less
  `onnx_import.shape_relabel:<node>` custom op: free when dead, a compile
  error naming the node when live.
- The host-op replay of `Binary`, `Where` and `Fma` (rlx-cpu
  `eval_single_op_f32`) read a broadcast operand at `i % len`, along the wrong
  axis for a per-channel `[1,C,1]` against `[N,C,T]`. On discrete NVIDIA,
  where wgpu hosts every elementwise op, a batch-norm affine gave `0.21` for
  `2.28`. Operands are now read at `(i / rep) % len`; a broadcast no single
  `(rep, len)` describes takes the shape-aware thunk path.
- wgpu built `matmul_coop16` (8×8×8 f16 cooperative matrices) on any adapter
  with cooperative-matrix support. An RTX 3080 Ti offers only 16×16 and 16×8
  shapes, so the kernel failed its self-check under every load/store
  convention. It is now built only when the adapter reports 8×8×8 f16, and an
  unbuilt kernel is no longer reported as a failed self-check (nor trusted
  under `RLX_WGPU_NO_COOP_PROBE`).
- The Vulkan/DX12 f16 cooperative matmuls (`RLX_WGPU_COOP_F16_VK_ENABLE=1`)
  never produced `a·b`; on an RTX 3080 Ti the self-check refused all four and
  every matmul fell back. Two bugs:
  - Layout. naga lowers `coopLoadT`/`coopStoreT` to SPIR-V `RowMajorKHR`, and
    A, B and the output tile are row-major, but the kernels loaded A and stored
    the tile with the column-major forms. Of all eight load/store combinations
    only all-`T` passes. The `_widen` variants, which also loaded B
    column-major and were picked for `n > 768`, are now identical to their
    base kernels.
  - Offsets. A and B are both read from the f16 shadow, but the binding was a
    window around B alone while A's offset was relative to the f32 arena
    window, so A was read from the wrong place: 32×16×32 came out ~100× off.
    The binding now covers both and A's f16 offset travels in its own
    parameter. The f32→f16 cast step had the same mismatch (its f16 window is
    aligned separately) and now has its own destination offset.

  Output now matches a reference on f16-rounded inputs to 3e-7–1.5e-6 (K up
  to 3072). Against f32 CPU it differs by 3e-3–1e-2, the f16 input rounding the
  opt-in accepts.
- The CoopF16Vk wide-f32 fallback looked up its bind group by `gpu_bi` but had
  stored it by schedule index. Host steps advance only the latter, so on
  discrete NVIDIA (where elementwise ops are host steps) any earlier ReLU made
  the fallback bind another step's group and wgpu panicked with a validation
  error. It is now looked up by schedule index
  (`coop_f16_vk_wide_after_host_step`).

### Performance

- CPU attention over long sequences (score matrix past `RLX_SDPA_BLOCK_MIN`,
  4M) runs row-blocked: one task per (batch, head, ~1M-score block of query
  rows) with a per-worker buffer, instead of a `q_s × k_s` matrix per head —
  and the executor no longer allocates and zeroes `batch·heads·S²` floats per
  run (4.3 GB for V-JEPA2's 8192 tokens). Masks: none, per-key, additive bias.
- SME2 packing: A panels and transposed B in L1-sized blocks, B tiles packed in
  parallel (it was serial while the workers waited). Bit-identical.
- Metal lowers `Op::GroupNorm` to MPSGraph (reshape to `[N, G, C/G·spatial]`,
  two-pass mean / variance, per-channel γ, β — the `group_norm` thunk's math),
  so graphs containing one now take the full-MPSGraph path instead of falling
  back to per-op thunks. EEG-DINO (GroupNorm in its patch-embedding convs)
  went from 1.3–1.4× slower than MLX to 0.74–0.97× (round-robin, M4, shared
  GPU). Pinned by `rlx-metal/tests/mpsgraph_group_norm_parity.rs`.
- Metal multi-head attention and ragged projections (EEG-DINO: d_model 200,
  8 heads of 25, 191 tokens; M4). Output bits unchanged; each path has an
  opt-out. Pinned by `rlx-metal/tests/attention_block_kernels_parity.rs`.
  - f32 `BatchedSgemm` is one `bmm_simd_4x4` dispatch (batch on grid z) instead
    of one MPS matmul per matrix — B·H encoder switches per attention matmul.
    Interleaved minimum, EEG-DINO Small batch 8: 152 → 16.5 ms.
    `RLX_METAL_BMM_MPS=1` restores the MPS loop.
  - f32 `Softmax` runs out of place (`softmax_lastax_out`) instead of a full
    Copy + in-place pass, and keeps the row in registers when it fits.
  - `[B,S,H,Dh] <-> [B,H,S,Dh]` transposes copy in output order
    (`transpose_swap12_trail_rows_f32`); the 3-D kernel put the contiguous axis
    on grid z. `RLX_METAL_TRANSPOSE_TRAIL_3D=1`.
- wgpu on Metal takes CoopF32 for f32 matmuls with K % 8 != 0, N % 32 != 0 or
  an activation right operand (`matmul_coop_f32_edge.wgsl`); these fell to the
  scalar tiled kernel. Full B tiles load directly and only the ragged tile /
  K tail is zero-padded through workgroup memory (the branch goes through
  `workgroupUniformLoad` — naga rejects coop loads under a plain uniform `if`).
  Split-QKV keeps its aligned-Param contract. `RLX_WGPU_NO_COOP_F32_EDGE=1`.
  Pinned by `rlx-wgpu/tests/coop_f32_edge_parity.rs`.
- wgpu multi-head attention reads Q / Kᵀ / V straight from the qkv projection
  and writes the merged heads in place: `unfuse::fold_strided_bmm` rewrites a
  batched matmul whose operands are `narrow → reshape → transpose → reshape`
  copies (and whose output feeds `reshape → transpose → reshape`) into an
  `rlx_wgpu.strided_bmm` custom node over the base tensors, run by
  `matmul_coop_f32_strided.wgsl`. The base tensors become the node's inputs,
  so the planner keeps qkv alive across the softmax. EEG-DINO: 3 narrows +
  4 transposes per layer gone, bits unchanged, wgpu/MLX 1.9–3.0× → 1.4–2.5×.
  Metal backend + cooperative matrix only; `RLX_WGPU_NO_STRIDED_BMM=1`.
  Pinned by `rlx-wgpu/tests/strided_bmm_parity.rs`.
- wgpu `Softmax` keeps rows of ≤ 512 elements in registers: one read and one
  `exp` per element instead of three reads and two, same Kahan merges, bits
  unchanged.
- SME2 f32 GEMM packs A and B straight from their strides
  (`sme::sme_sgemm_strided`): transposed / strided products no longer take an
  extra full copy of both operands, and A panels are read along contiguous
  rows. Bit-identical (each output is a sequential fused multiply-add, pinned
  by `sme_is_sequential_fma_bit_exact` / `sme_strided_route_is_sequential_fma`).
  `RLX_CPU_SME_THREADS` sets its worker count (default: the pool).
- f32 LSTM: the input projection is one GEMM over all timesteps (BLAS / SME)
  and the recurrent mat-vec uses four accumulators — pyannote's VAD 3.8–4.9×
  faster, its frame scores within the same distance of whisperX's (4.8e-7 on a
  100 s file) and every segment unchanged.
- wgpu runs `ArgMax` / `ArgMin` and `Reverse` on the GPU (they were host
  steps). A host step makes a run wait on the GPU mid-graph, which the
  browser cannot do, so any WebGPU model with an argmax or a flip fell back to
  the CPU there. The new `rlx.argreduce` and `rlx.reverse` kernels
  (`builtin_gpu_kernels`) dispatch through the existing `WgpuGpuKernel` step,
  using offsets into a window that holds both tensors. Reverse needs 4-byte
  elements. Either op stays a host step if its tensors cannot share a window
  (a parked param, different stripes). Argreduce is rlx-cpu's sequential
  first-wins scan, so it matches bit for bit: ties keep the first index, a NaN
  never wins, a leading NaN keeps index 0.
  `wgpu_argreduce_reverse_parity` checks argmax and argmin on every axis, with
  tied maxima in every row group, and five reverse cases, and asserts no host
  steps remain. It fails on a last-wins kernel. Eight exgb encoders that had
  stayed on the CPU in the browser now run on WebGPU.
- An exact `f64` fused multiply-add for wasm, which has no FMA instruction:
  `rlx_libm::fma_emulated`. It builds the correctly rounded `a·b + c` from
  about 30 `f64` operations (Dekker's product, TwoSum and a round-to-odd add;
  Boldo & Melquiond 2008). Zeros take an exact shortcut, and inputs outside
  `2^±450` or non-finite fall back to `libm::fma`. With the `wasm-fma`
  feature it also defines the `fma` symbol, replacing the software `fma`
  behind every `f64::mul_add` in a wasm binary with no call-site changes.
  Under V8, a fused `f64` multiply-add drops from 11.6 to 5.8 ns (plain
  multiply-add: 2.6 ns). It matches the M4's hardware FMA bit for bit on
  8.4 M stress cases (cancellation, rounding ties, zeros, range edges, all
  special values), and matches `libm::fma` inside wasm on 3 M more.
- `rlx_linux_ref::openblas::dgemm_ab` packs A's rows and B's columns
  contiguously, advances four output columns together, and on wasm inlines
  the exact FMA instead of calling it. Every element keeps its arithmetic
  (the same k blocks, accumulator chain and fused operations); the Haswell
  OpenBLAS fixtures still pass and fail when the new path is altered. In the
  browser, exgb's BENDR linear probe went from 278 s to about 205 s with
  identical score bits. Most of the remaining time is `dsyevd`'s exact
  eigensolve on the 1536-wide embedding.
- wgpu: rlx's own kernels compile without naga's forced loop bounding (bounds
  checks stay on). The guard kept Metal from unrolling the tiled matmul's
  fixed loops: `matmul` 32×128×2 went from 440 µs to 37 µs per dispatch.
  User shaders keep the fully checked path.
- wgpu: a `Binary` reads a broadcast operand straight from the `Expand`'s
  source through its `(rep, len)` indexing, and `Expand`s left unread are
  dropped. EEGNet's training step: 216 → 185 dispatches, arena 83 → 75 MiB.
- wgpu: the conv weight-gradient kernel steps its output-position decode by
  carries instead of four integer divisions per multiply-add, about 3× faster
  on EEGNet's temporal conv, bit-identical output.
- wgpu CoopF16Vk (opt-in f16 tensor-core matmul): a pass split between an
  f32→f16 mirror and the coop matmul also waited on the CPU for the GPU to go
  idle — a full round trip per chained matmul, 14 of 20 ms per run of an
  8-layer 512×768 chain on an RTX 3080 Ti. The split stays; the wait is gone
  (the queue already orders the submissions).
- wgpu CoopF16Vk: new `matmul_coop_f16_vulkan_f32acc_32`, one subgroup per
  32×32 block with 2×2 register-blocked 16×16 tiles, used when M and N are
  multiples of 32 and the device self-check passes it. The 16×16 kernel ran
  64 invocations, i.e. two NVIDIA subgroups computing the same tile, and reused
  no fragment. On an RTX 3080 Ti (shared with another job), 8 chained layers:

  | shape | f32 | coop before | coop now |
  |---|---|---|---|
  | 512×768×768 | 5.9 ms | 20.6 ms | 3.4 ms |
  | 2048×768×768 | 14.9 ms | 29.7 ms | 12.0 ms |
  | 2048×768×3072 | 40.7 ms | 85.0 ms | 25.7 ms |
  | 4096×1024×1024 | 40.9 ms | 88.0 ms | 33.0 ms |

## [0.2.19] — 2026-10-24

### Added

- `rlx-numpy::torch`: torch CPU float32/float64 reductions bit-for-bit (`sum_f32`,
  `sum_f64`, `sum_rows_f32/_f64`, `mean_f32`). These follow ATen's `cascade_sum`: 4/2
  lanes on arm64, ILP 4, a 4-level cascade. Also `exp_f32`, which is SLEEF's `xexpf`,
  the exp torch calls on arm64 (libm `expf` differs in about 1 value in 10). Pinned
  against torch 2.11.
- `rlx-numpy::npy`: float32 `.npy` read/write.
- `rlx-linalg::lsap::linear_sum_assignment`: a port of SciPy's `rectangular_lsap`,
  including its tie rules.
- `rlx-linalg::algos::eigh_torch` / `eigh_torch_f32`: `torch.linalg.eigh`'s spelling,
  LAPACK `?syevd` on the lower triangle, with eigenvectors in columns. The f32 version is
  bit-identical to torch 2.11 on macOS. Backed by the new `rlx-cpu::blas::ssyevd` and
  `dsyevd_uplo`.
- `rlx-rng::torch`: `normal_tensor_f64` (the double `normal_fill`), plus `Init`,
  `random_int64`, `epoch_permutation` and `dropout_mask`. The last four moved here from
  an exg crate: torch default initialisers and the `DataLoader(shuffle=True)` stream.
- `rlx-optim::clip_grad_norm_torch`: `torch.nn.utils.clip_grad_norm_` exactly
  (`max/(total + 1e-6)`, clamped at 1, always applied).

- **`rlx_runtime::placement`: measured device placement for concurrent jobs**, moved from
  exg's moabb crate. `measured_device(key, candidate, time_step)` times a few real steps of
  the caller's workload on the cost model's GPU pick and on the CPU, keeps the GPU only at a
  1.5× margin, and decides once per key for the process (`RLX_PLACEMENT_CACHE` remembers
  decisions across runs; `RLX_PLACEMENT=static` / `RLX_FORCE_DEVICE` skip it). `gpu_slot`
  shares a GPU among concurrent jobs, optionally capped (`RLX_GPU_JOBS`); it nests on one
  thread, gives a placement probe the GPU to itself, and records the devices each thread
  compiled for (`take_devices_used`). Written after the cost model put EEGConformer's
  training graph on Metal at 20–99 s per 64 steps against 1.7 s on the CPU. Tests moved
  with it.
- **`rlx_runtime::job_slots::JobSlots`: a first-come-first-served pool bounding concurrent
  jobs by count and memory**, moved from exgb. With `gate_memory`, the next job waits while
  RSS plus the measured growth per running job would exceed the soft memory budget; a job
  always starts on an idle pool, so it never deadlocks. Tests: the memory gate, and the
  count cap under 12 threads.

- **Graph outputs have names.** `Graph::name_output(i, "logits")` /
  `set_named_outputs([("logits", y), …])`, and `CompiledGraph::outputs()`
  (name, dtype, static shape, in `run` order) / `output_index("logits")`,
  which names the closest outputs when there is no match. Optimizer passes
  keep output positions, so a name still finds its output after fusion has
  renumbered the nodes behind it. ONNX imports keep the model's output names
  (through HIR, `HirModule::output_names`). `.rlxmodel` writes them as
  `return %lm_head as logits`. `extract`, `parallel` and `sequential` carry
  them. In every binding: `name_output` / `output_names` on graphs,
  `outputs()` / `output_index()` on `Compiled`.

- **Bindings report boundary and compile errors by kind.** Every language's
  `Compiled` (Python, JS, C, Swift and the other generated bindings) now goes
  through `try_run` / `try_set_param` / `try_compile`. A misspelled input or a
  wrong-length buffer is a `Type` error (`ValueError` in Python, `TypeError` in
  JS, `RLX_ERR_TYPE` in C) that names it. Before, it ran on zeros or failed
  with an opaque panic. New `Compiled` rows: `inputs()`, `params()`,
  `check_bindings(names)`, `allow_unbound(names)`. See
  [docs/bindings.md](docs/bindings.md).

- **README examples are doctests** (`just check-readme`, part of `just ci`):
  every Rust block in the README is now a complete program that compiles and
  asserts its result, using only the `rlx` crate. Several were fragments with
  undefined variables, imported crates the prelude already covers, or called
  `std::env::set_var` (unsafe in edition 2024); the training example could not
  have trained (see **Fixed**). New [`docs/troubleshooting.md`](docs/troubleshooting.md)
  — errors by message, wrong numbers on one device, NaN, slowness. README crate
  list now covers all 85 workspace crates.

- **Caller mistakes at the `CompiledGraph` boundary are errors, by name.**
  A misspelled input (`the graph has no input `token` (did you mean
  `tokens`?)`), or a buffer whose length is not a static input's or param's
  shape, used to compute on the arena's zeros and return plausible numbers; it
  now panics in `run` / `run_typed` / `set_param`, and the new
  `CompiledGraph::try_run` returns it as an `IoError`. Inputs left unsupplied
  and params never set are named in a one-time warning (some callers rely on
  them reading zero); `check_bindings` returns them as an error and
  `allow_unbound` declares the ones meant to read zeros (an empty KV state). Unknown
  *param* names stay accepted (loading a full checkpoint into a subgraph).
  `inputs()` / `params()` list the declared boundary (name, dtype, shape).
  A cache that an `Op::KvAppend` writes into takes any whole number of its
  rows: decode graphs declare a spare row for the in-place write
  (`rlx_flow::blocks::kv_spare_rows`) and are fed only the filled ones. The
  whole `rlx-runtime` CPU suite runs unchanged with the checks on, as do the
  Gemma and Qwen3 decode tests in rlx-models. The unbound warning stays a
  warning: it fired in none of about 350 rlx-models tests across 12 model
  families.

- **`Session::try_new` / `try_compile` / `try_compile_with`** return a
  `CompileError` — `Invalid` (the full IR verifier, run up front, so errors
  name the caller's graph rather than the first pass), `DeviceUnavailable`
  (not built in, or absent with the backend's reason), `Unsupported` (node
  and op the device cannot lower), or `Backend` — instead of panicking.

- **`rlx_stablehlo::export_specializations`** — one symbolic graph exported
  at several `DimBinding`s as functions `@main_<suffix>` in one module (prefill
  buckets); with parameters, all functions share one weight file.
  `DimBinding::with` and `FromIterator<(u32, usize)>`.

- **`SimplifyTransposes`** (`rlx_compile`; opt-in
  `CompileOptions::simplify_transposes` / `RLX_SIMPLIFY_TRANSPOSES`, default
  off). Removes identity transposes, composes chains, folds transposes of
  constants, and sinks a transpose through single-use elementwise ops when that
  makes it cancel another. On imported Supertonic TTS models it removes 35–38%
  of transposes (duration predictor 49→31, text encoder 80→50, vocoder 34→22)
  with bit-identical outputs on CPU and Metal; transformer and CNN imports have
  none to remove. No reliable speedup measured yet (host load ~80 during the
  runs), hence off by default. `rlx-onnx-import --example transpose_census
  [--run] [--device D]` reproduces the census.

- **StableHLO weights as IREE parameters** — `StableHloOptions::with_parameters`
  emits weights as `util.global` + `#flow.parameter.named`, read from a file at
  load time (`write_safetensors` writes one). Int8 weights export as `i8`
  tensors dequantized in the module. Both are in the IREE round trip.

- **StableHLO / IREE export** — new crate `rlx-stablehlo` and
  `ExportTarget::StableHlo` (`rlx-runtime` feature `stablehlo`). Graphs go
  through the TPU backend's preparation and rlx → HLO lowering and are printed
  as StableHLO MLIR (`model.mlir` + `model.signature.json`) for `iree-compile
  --iree-input-type=stablehlo`. NCHW convolutions export without transposes.
  `tests/iree_roundtrip.rs` compiles and runs seven graphs — MLP, conv + pool,
  embedding/LayerNorm/shape ops, RmsNorm, batched matmul, and Llama and BERT
  layers built through `rlx-flow` — under IREE (llvm-cpu) and matches rlx CPU,
  with an op-by-op bisect on mismatch. It found the five bugs under **Fixed**.

- **Known failures as data** (`rlx_ir::known_failures`). A case a backend is
  known to get wrong is a line in `tests/known_failures/<test>.txt` —
  `case device[@os] issue [note]`, issue required — instead of a skip: it still
  runs, its failure is reported as known, and once it passes the test fails
  until the entry is removed. `RLX_RUN_KNOWN_FAILURES`,
  `RLX_KNOWN_FAILURES_ISSUE`, `RLX_KNOWN_FAILURES_REPORT` (JSON lines) and
  `RLX_KNOWN_FAILURES_DIR` (rig-local lists); `just known-failures
  list|update` applies a report to the files. Wired into `fd_backward_gate`
  and `gpu_filters_parity`. See `docs/adr/0005-known-failures-as-data.md`.

- **`cargo rlx debug`** (`rlx_runtime::debug::debug_graph`): runs a graph on a
  device and on CPU with the same deterministic feeds and classifies the
  result (`unsupported-op`, `compile-panic`, `run-panic`, `non-finite`,
  `mismatch`, …). `--probe` reruns with one compile option changed at a time
  (fusion off, regions unfused, F32, common-IR kernels, no constant folding, no
  MPSGraph); `--bisect` exposes every op and names the first that diverges,
  flagging one that reads only graph inputs. On the known Metal BF16-input
  defect it names the `Narrow` reading the input, with every probe failing.

- **Fusions-must-fire test** (`rlx-runtime/tests/fusions_fire_on_model_blocks.rs`):
  Llama and BERT blocks built through `rlx-flow`, compiled on every registered
  device, with exact fused-op counts plus two controls (the block still emits
  the pattern; with fusion off none of the fused ops appear), and checks that
  `fuse()` and `supported_for_target` agree with each backend's real claims.

- **`ExpertPager::with_prefetch_cap(fraction)`** and `PagerStats::{prefetched,
  prefetch_skipped, prefetch_used, prefetch_wasted}`: bound how much of the
  cache unconsumed prefetches may hold, and see whether prefetching pays.
  Default unchanged (no cap beyond the budget).

- **Architecture decision records** in `docs/adr/`.

- **`RLX_POISON_ARENA=1`**: CPU scratch starts as `0xFF` bytes (NaN, −1)
  instead of zeros — at allocation and at the per-run reset — so an output
  nothing wrote reads NaN instead of a plausible zero. The whole `rlx-runtime`
  CPU suite passes with it on, so CPU kernels do not depend on zeroed scratch;
  it found the two test defects under **Fixed**.

- **`Model::run` on a language model takes media beside `input_ids`** (a
  tensor named as the entry's media kind, a speaker vector): the prompt's
  logits with each medium spliced in at its placeholder token
  (`prefill_with_media`); the next run, the context grown, goes on from its
  cache. A package program drives a speaker-conditioned LM through
  `entry.run` this way (KittenTTS 2).
- **`tokenizer.encode` with `special: false`** for package programs: the
  text's ids alone (`TextCodec::encode_plain`), none of the special tokens
  the tokenizer's template adds around them — ChatterBox's wraps every text
  in `[EXAGGERATION] [START] … [STOP] [START_SPEECH] [START_SPEECH]`, which
  its program builds itself.
- **`rlx_program::run_body`**: a function run with the request body the
  host reads — `from`, `keep` and `dims` together, which `run_with` and
  `run_sized` each send one part of. A prompt graph whose caches have as
  many rows as the program asks (no input's shape says) keeps them in
  slots (MetaVoice).
- **Training a package** (`rlx_runtime::model::trainer`, features
  `model-io` + `training`): `open_trainer(path, options)` trains any
  function of an `.rlxp` with nothing about the model it is — its symbols
  sized (`dims`), a loss (`{output}`, `{cross_entropy: {logits, labels,
  mask?}}`, `{mse: {output, target}}`), the params (`"all"`, `{params:
  [globs]}`, `{lora: {targets, rank, alpha}}`, or `{sae: {tap, width, l1}}`
  on one activation, the base frozen) and AdamW, on `ResidentTrainer`. A
  package's model block may hold the same object as `train`, its defaults.
  `save_adapter` writes LoRA factors as an adapter package (each update in
  the layout its tensor is stored, transposed references included), which
  opens as the trained model; `save_sae` writes the SAE as a package of its
  own (`h` → `features`, `reconstruction`); `save` writes a full fine-tune
  as the package with the trained values held. Integer inputs (token ids)
  are fed as f32 and cast in the graph. `prepare` gives any trainer the
  graph, the trainable names and the starting values, and `Saver` writes
  the results back. Tests `package_training`.
- **An SAE spliced into a package as it opens** (open option `sae: "<sae
  package>"` or `{path, mode}`): `replace` (default) feeds each function's
  tap through the SAE so the model runs on its reconstruction; `features`
  leaves the model as it was and adds the output `sae.features`. Test
  `package_training`.
- **`Trainer.from_package` / `Trainer.save_package`** in every binding
  (`api.idl`; rlx-capi `rlx_trainer_from_package` /
  `rlx_trainer_save_package`, rlx-js `Trainer.fromPackage` /
  `savePackage`, pyrlx `Trainer.from_package` / `save_package`, Swift and
  the generated languages): the bindings' trainer (its optimizers,
  freezing, resident path, checkpoints) on a package's function. Test
  `rlx-bind/tests/package_trainer.rs`.
- **Every op kind round-trips through `.rlxmodel`**: test
  `every_op_kind_round_trips` prints and reads back a graph of each
  `sample_ops` kind, so a new op the text format cannot carry fails there.
- **Package program inputs read from the weights**: a run's body
  `weights: {input: [tensor, …]}` feeds each input the named tensors of the
  package's weights, read for that run and stacked as the input declares
  them (`WeightCache::bytes` now reads a safetensors tensor's bytes as
  stored). A program routes a mixture of experts this way: Drama 3's 32 GB
  of FP8 experts are read eight at a time, none bound to a function.
- **Symbolic fits of a symbol twice beside another** (`rlx_ir::dynamic`):
  a dim `2·s − 1` or `2·s − l` (a flow decoder's relative positions over a
  prompt's and a generation's tokens, a prompt's mel frames padded to them)
  is derived when no fit with each symbol once is; test
  `a_dim_of_a_symbol_twice_and_another_is_derived`.
- **The `sample` host service for package programs.** A program sends
  logits (`[n]` f32) and the sampling options (`seed`, `step`,
  `temperature`, `top_p`, `top_k`, …) and gets back `{id}`, drawn by the
  runtime's sampler chain from the stream `seed + step` (the argmax when
  greedy). Orpheus's program applies its own repetition penalty, then samples
  through it.

- **Batched, differentiable ReEig/LogEig on the CPU** (`rlx_cpu::spd_batched`).
  `spectral_batched(g, x, batch, n, eps, Re | Log)` emits two custom ops:
  `spd.matfn_batched` (`[B,n,n]` → packed `Y ‖ λ ‖ U`, one f64 LAPACK `dsyevd`
  per matrix, rayon over the batch, f32 or f64 graphs) and its Loewner
  (Daleckii–Krein) backward, which reuses the forward's eigendecomposition.
  GPU backends run both through their host-kernel fallback, bit-identical to
  the CPU on Metal, MLX and wgpu. It replaces the graph Jacobi scan
  (`spd_*_batched`) for training: TensorCSPNet and GraphCSPNet train ~6×
  faster. Tests `rlx-runtime/tests/spd_batched_native.rs`: forward vs host f64
  (4e-8), gradient vs finite differences (1e-7; the Jacobi path is 1e-5) and
  vs autodiff through Jacobi on symmetric parts (8e-6), and the GPU backends.
  The raw gradients of the two paths differ (0.85 here) in how they split
  `dX + dXᵀ` between the triangles, which a symmetric input cannot see.
  It also covers `√`, `1/√` and powers (`BatchedSpectral::{Sqrt, InvSqrt,
  Pow}`), each against finite differences (≤ 1.2e-7), and a
  `SpectralBatchedExt::spd_native` method as the drop-in for the Jacobi
  builders.
  Every function's gradient is the derivative of `f(max(λ, ε))`, zero where
  the floor clamps, as the Jacobi builders compute it. For `Log` that differs
  from the native single-matrix `Op::LogEig` (`1/max`), which matters when
  `Log` at ε = 1e-4 stands for ReEig∘LogEig; the test clamps part of the
  spectrum, and `1/max` fails it.
- **A program runs its package's other entries.** The host service
  `entry.run` runs an entry of the package (by name or kind) on tensors
  with options, as `Model::run` — a `generate` entry generates ids — so an
  LM-and-codec TTS is one package: the LM's entries, the codec's graphs, a
  program between. Generation's `seed_per_step` option draws each sampled
  token from its own stream (`seed + step`, history empty), as rlx-qwen3's
  `sample_token_at` does.
- **Generalizing from many sizes.** `rlx_ir::dynamic::generalize_many`
  fits through the first two builds and keeps, of the fits, the one
  through every other — dims and attributes both, including those the same
  at the first two (a quotient can be: `⌈37/16⌉ = ⌈46/16⌉`). Of quotients
  as small, a power-of-two divisor is taken.
- **Quotient derived dims.** A `SymDef` may divide, rounding down, and add
  a second symbol (`inner`, `div`, `plus`, `plus_scale`; `.rlxmodel`
  `(audio)/256+1`, `(t+1)/2`, `(audio*-1+255)/256+frames*1-1`): an STFT's
  frames, a strided conv's, a decoder's frames past a prompt's.
  `generalize_checked` fits them through its third size (a quotient alone
  first, then the smallest offset), and refuses sizes two symbols step in
  proportion across, which no fit can tell apart.
- **Programs size symbols by name.** `rlx_program::run_sized` (a request's
  `dims: {name: size}`) sizes a symbol no input carries.
- **Scaled symbolic attributes.** A `SymAttr` is the size times `scale`
  plus `offset` (`.rlxmodel` `start: seq*-1+1000`): a start that falls as
  the size grows, a relative position table read around its middle
  (ZipVoice's and LuxTTS' encoders).
- **Square derived dims; generalizing from three sizes.** A derived dim
  (`rlx_ir::dynamic::SymDef`) may be a square with a linear term
  (`SymDef::linear`; `.rlxmodel` `f32[seq*seq*2+seq-1]`), what an
  attention's relative positions flatten to (VITS, Piper's encoder).
  `rlx_ir::dynamic::generalize_checked` takes the graph at a third size
  and keeps, of the fits through the first two, the one through it — two
  sizes cannot tell `96·seq − 480` from a square.
- **Programs: a package's host logic as WebAssembly** (`rlx-program`, and
  rlx-runtime's feature `program`). An entry of kind `program` names a
  wasm module (built with `rlx-program`'s guest SDK) that answers the
  model's tasks — `run`, `transcribe`, `synthesize`, `embed`, `generate`,
  `chat`, `score` — and calls back for the package's functions (each
  specialized to its inputs' shapes and compiled once per shape, on the
  model's device; outputs kept in host-side slots so caches never cross
  into the module), its files and tokenizers. Run by `wasmi`, an
  interpreter, so it works where JIT does not (iOS, watchOS). For what a
  graph says badly: phonemizers, FST and beam searches, entropy decoders,
  a model's own decode loop.
- **Programs read the package's files.** A program built for
  `wasm32-wasip1` gets a read-only WASI filesystem of the package's
  sidecars at `/package` (in memory, nothing of the host's), so `std::fs`
  code — espeak-ng reading its dictionaries — runs as is. A package of a
  program alone keeps its model block in `model.json`.
- **`transcribe` for a front end symbolic in the audio's length.**
  `audio_multiple` pads the audio to whole frames, the stages read the real
  length as `audio_samples`, and `token_budget: {per_second, extra}` sets
  the decode length from the audio's. Its stages open from their own
  handle on the package (as `synthesize`'s do), so a symbolic one keeps
  its weights. A no-step decoder's `prefill.inputs` take `{context: …}`
  feeds, and a function symbolic in their length (an encoder's) is
  specialized to it when it runs, compiled once per length.
- **`Op::While` loops at run time, unbounded or not.** It was unrolled to
  `max_iterations` everywhere, so a loop with no bound could not run at all.
  The CPU now compiles its `cond` and `body` once and loops
  (`rlx_cpu::thunk::{Thunk::While, execute_while}`); Metal, CUDA, ROCm and
  wgpu run it through their one-op host fallback (`HostOp`). `LowerIf`
  (inline `Op::If` alone) replaces `LowerControlFlow` on those backends and
  in the fusion pipeline when the backend claims `While`
  (`DecomposePolicy::while_native` for the GPU decompose pass); the others
  still unroll. Checked on CPU, Metal and wgpu against a host loop
  (`rlx-runtime/tests/while_runtime_loop.rs`: a data-dependent count, a
  bound, a false start); CUDA and ROCm compile-checked.
- **Several entries in one package, as one model.** An entry's key is its
  name and its kind its `kind` field (or the name, as before), so a
  package holds several entries of any kinds. Opened, it is one model:
  each task goes to the entry of its kind (`embed`, `chat`, `transcribe`,
  …) and `run` to the entry whose inputs are the tensors given. The first
  entry opens with the package, the others when first asked for. The open
  option `entry` opens one by name. `rlxp::{entries_of, entry_of_kind}`;
  the feature `entry.kind` keeps an older rlx from missing a named entry.
- **`rlx-search`** (re-exported as `rlx_runtime::search`): host-side sequence
  search shared by every decoder, dependency-free. `argmax` / `top_k` /
  `log_softmax` with one tie rule (lowest index); beam search as a stepping
  state machine (`BeamSearch::advance`, callers own and reorder their KV by
  `Step::parents`) with the Hugging Face rule (`BeamConfig::new`) and
  `openai/whisper`'s (`BeamConfig::whisper`) as presets plus `Constraint`
  masks; CTC greedy, streaming prefix beam (`PrefixBeam`), and the CTC
  trellis `ctc::forward` over the `Log` / `Tropical` semirings; monotonic
  Viterbi alignment. `lm::beam_search`, `lm::argmax_u32` and
  `samplers::argmax_logits` now delegate to it, so greedy decode and the
  sampler chain get the lane-parallel `argmax`: 46 µs vs 164 µs on a
  151,936-entry row (Apple M-series, release). Verified token-identical to
  the previous `lm::beam_search` and to a transcription of `openai/whisper`'s
  `BeamSearchDecoder` over 600 scripted models each.

- **`dynamic::generalize_with`**: a derived dim's offset may exceed its
  scale (a convolution's padding: `4·frames + 6`) when the caller checks
  the result at a third size; and an attribute may be a derived symbol's
  size (a slice as long as a padded dim), not only a base symbol's.
- **Packages with no decode step.** A `generate` entry with a `prefill` and
  no `functions` runs its prompt function over the whole context for each
  token, as a runner with no cache does, so a language model whose only
  graph is a full-sequence forward (Llama 4, MiniMax M3) runs from a
  package. `prefill.functions` lists one prompt function per length, for a
  graph not symbolic in its length.
- **A held tensor of the wrong size fails the open**: `PackageWeights::bind`
  checks each held tensor against the shape its param declares. It was
  bound silently, and a package written wrong ran with the wrong values.
- **Prompt-cache snapshots for packages.** A package's language model
  (`GraphLm`) takes `LmRunner::export_session` / `restore_session` /
  `prefill_logits_reusing`: its caches (read back from the executable when
  they live there) and the tokens in them, restored by the next step, so a
  prompt sharing a snapshot's prefix runs only its new tokens.
- **`rlx-pkg inspect` shows the model**: who wrote it and who may read it,
  its symbols by name, each function's inputs, outputs, params and
  variables, its entries, and its weight sources.
- **Chat templates with `{% generation %}`** (Hugging Face's assistant-mask
  tag, LFM2's and SmolLM3's templates) compile: the tag renders its body, as
  transformers does. A package with such a template failed to open.
- **Combine and split graphs: `rlx_ir::compose`, in every binding.**
  `Graph::splice` copies one graph into another, binding named inputs *or
  params* to existing nodes (a weight tied under another name); same-named
  leaves are shared after a kind and shape check, or kept apart with
  `Splice::prefix`; `Splice::adapt_shapes` re-infers shapes from bound
  operands. `compose::parallel` / `compose::sequential` build on it.
  `Graph::partition` (any order-respecting node→part assignment, named
  `__boundary.<id>` tensors), `Graph::split_at` and `Graph::components` all
  return a `Partition` of self-contained graphs that `merge` reassembles;
  `Graph::extract` takes one subgraph. Variables and symbolic attributes are
  carried through. Works under `no_std`. Bound in Python (`pyrlx.splice`,
  `parallel`, `sequential`, `extract`, `Partition`), JS (`rlx.*`), C
  (`rlx_splice`, `rlx_partition_*`) and every generated language. See
  [`docs/graph-compose.md`](docs/graph-compose.md).
- **`no_std` graph verification: rlx-ir core and rlx-fpga.** rlx-ir has a
  default-on `std` feature; with `--no-default-features` it builds as `no_std` +
  `alloc` and keeps the graph / op / shape / verify core (`RLX_*` switches read
  as unset, the custom-op registry is empty, float math comes from `libm`).
  rlx-fpga's `no_std` build now works (it never compiled before, GitHub #7):
  Model → `optimize` (still checked by `rlx_ir::verify`) → in-memory
  `collect_artifacts`; graph import/export and file emission stay on `std`.
  `just check-no-std` (in `just ci`) builds rlx-ir, rlx-cortexm and rlx-fpga for
  `thumbv7em-none-eabihf` and runs their host tests without default features.
- **XDNA bf16 matmul (opt-in, not yet hardware-validated).**
  `Session::new_with_precision(Device::Xdna, Precision::BF16)` runs matmuls
  through mlir-aie's `bf16_f32` microkernel (bf16 operands, f32 accumulation)
  instead of per-channel int8. Host-side tile packing is tested against the
  kernel's tile contract; the NPU path has not run yet.
- **XDNA host checks and live geometry.** `rlx_xdna::host_blockers()` (XDNA 2
  part, `/dev/accel/accelN` not read/write for the user), `aie_geometry()`
  (columns/rows/AIE version from the driver), and a memlock warning in
  `diagnostic()`. `RLX_XDNA_MAX_CONTEXTS` (default 4) bounds the NPU hardware
  contexts a process holds; chains evict idle ones to stay under it.

- **rlx-distributed: addressed layer pipeline building blocks** (lifted from
  `rlx-kimi-k3`'s pipeline):
  - `rpc` — request/reply over TCP whose server sends heartbeat frames while a
    request runs, so a client bounds every read: a peer silent for
    `RpcOptions::timeout` is an error naming it, a slow request is not. `serve`
    runs an `RpcHandler` with one thread per connection (a health probe is never
    stuck behind an idle client), returns handler errors and panics as messages
    and keeps serving; `OP_SHUTDOWN` stops it and closes idle connections.
    `Enc` / `Dec` are the checked little-endian payload codec (incl. lists of
    `f32` vectors).
  - `stages` — `StageSpec` (an address and its layer range), `plan_stages` (even,
    explicit counts, or `w:` weights, on `LayerSplit`), `validate_stages` (refuses
    stages that do not tile the model), `parse_stages`.
  - `cluster::sizing` — roofline step timing: `NodeSpec` with presets, a model's
    `StepWork` (dense / expert FLOPs and bytes, disk-served expert share, pipeline
    boundary and all-to-all bytes), and `time_step` for pipeline (micro-batches in
    flight) and expert-parallel layouts, with the bottleneck resource.

- **`.rlxmodel`: rlx IR as text, and its stable encoding
  (`rlx_ir::rlxmodel`, feature `serialize`).** A graph prints as
  `%2 = MatMul %0, %1 : f32[1, 2]`, and every op and attribute is written as
  the `Op` variant itself, by name: `Reshape{new_shape: [1, -1]}`,
  `Activation(Gelu)`. Nested `If` / `While` / `Scan` bodies print as
  `graph @… { … }` blocks. Values keep `nan`, `±inf` and exact floats, and
  bytes print as `b64"…"`. The file starts with `rlxmodel 1` / `opset 1`,
  and a file from a newer version is refused rather than misread. Every op
  round-trips to identical bytes (`tests/rlxmodel_every_op.rs`, in
  `just check-ops`), and `tests/fixtures/opset1.rlxmodel` keeps the format
  readable across later `Op` changes. `.rlxp` packages now store their graph
  this way (`rlxmodel_v1`, readable as `graph/model.rlxmodel` in a directory
  pack), and bincode packages from earlier versions still open. A
  100k-node graph is 6.8 MB of text (bincode: 9.3 MB) and reads in 129 ms.
  - **Versions in the file.** `rlx 0.2.18` records the writer.
    `requires 0.2.19` makes an older rlx refuse the file, with a message
    saying so. `model {name: "…", version: "…", …}` is the model's own block:
    its name and version, plus whatever its runtime reads. All three are
    optional, and files without them read as before (`rlxmodel::Module`).
  - **A package holds a whole model.** It can store several graphs
    (`WriteOptions::extra_graphs`, `Package::module`) and the model block
    (`WriteOptions::model`, `requires_rlx`). The `rlxp` provider reports
    the model's `name`, `version`, writer version and functions in
    `info().metadata`.
  - **Weights by reference.** The model block's `weights: [{uri, prefix?,
    map?}]` names where each param the package doesn't hold comes from:
    `safetensors://`, `gguf://`, `rlxp://` and the other `resolve_weight_uri`
    schemes, with relative paths beside the package. A param found nowhere
    fails the open with its name, instead of running on zeros. Only f32
    params bind this way so far.
  - **Model cards.** The model block's `authors`, `license` (SPDX, one or
    a list), `repository`, `homepage` and `card` (the full Markdown model
    card) have accessors on `Module` and appear in `Model.info().metadata`.
    A multi-line text prints verbatim as a `"""` block, and the block prints
    one field per line, name first and card last.
  - **Entries: an `embed` method from the package alone.** A model block's
    `entries.embed` gives the encoder graphs, which are buckets at
    `[batch, seq]`, the tokenizer (`sidecar:` or a path), each input's role,
    the position offset and the pooling. The `rlxp` provider then answers
    `embed(texts)` with no provider code. Tokenizers load through
    `model::set_tokenizer_loader`, which the bindings set to rlx-text's
    (`rlx_text::install_model_tokenizer`). Referenced weights may be
    transposed (`map: {p: {tensor, transpose: true}}`). rlx-models'
    `rlx-embed` exports BERT / XLM-R checkpoints this way
    (`export::export_rlxp`): multilingual-e5-base becomes an 89 MB package
    that references 125 of its 149 params. Opened by rlx alone, from Rust
    or Python, it embeds within 2e-7 (cosine ≥ 0.9999995) of rlx-models' own
    embedder.
  - **Symbolic graphs: one graph for every size.** Named symbolic dims
    (`dims {seq: 0}` in the header; shapes read `f32[1, seq]`), and
    symbolic integer attributes: an op field that is a symbol plus a
    constant, written in place (`KvAppend{axis: 1, pos: upper}`,
    `start: seq-1`). The table lives on `Graph` (`sym_attrs`), skipped by
    serde so bincode packages still read. `rlx_ir::dynamic::specialize`
    binds the symbols, writes those attributes, re-derives `Reshape` /
    `Expand` targets from the bound shapes, and re-infers. The deferred
    dynamic-shape compile now uses it, where before it re-inferred from
    stale targets. `dynamic::generalize(a, b, sizes)` turns two static
    builds of a model into one symbolic graph, and an error names any node
    that isn't symbolic. Entries run a symbolic function at the `sizes`
    they list. multilingual-e5 (`encode`) and Qwen3-0.6B (`decode`) now
    export as one graph each instead of six and four. They give the same
    vectors, within 2e-7, and the same 24 greedy tokens as before.
  - **Readable `.rlxmodel`.** Values print by their node's name
    (`%q_proj`, `%model.layers.0.self_attn.k_proj.weight`) where it is a
    unique label, else by position. Inputs and params named like their
    value print as `%x = input` / `%w = param`, and optional fields that
    are `none` are left out. Labels are checked on read: a number must be
    its position, a name defined once, and a reference to a value above.
    Files written before still read, and every op still round-trips to
    identical bytes.
  - **Constants and variables in `.rlxmodel`.** A small constant reads as
    its values (`%iota = const [0, 1, 2, 3] : f32[4]`, exact for f32/f64
    and integers, NaN bits included), and a larger one stays base64.
    Variables (`Graph::vars`) are inputs whose next value is an output:
    `%past_k_0 = var : …` with `return …, %past_k_0 = %k_0`.
    `stateful_decode` declares a step's caches as variables, and a
    `generate` entry then needs no `state` pairs. Exporters drop the
    builder's provenance and shared node names (`tidy`), so a layer reads
    `%83 = MatMul %82, %model.layers.0.self_attn.q_proj.weight : f32[1, 1, 2048]`.
    A long `return` lists one value per line, and long objects in the
    model block break one field per line. `generalize` keeps a `Reshape` /
    `Expand` target symbolic (`new_shape: [1, cap, 8, 128]`) instead of
    showing the size it was probed at. A stateful step's written caches read
    `%past_k_0.next` and its first output `%logits`. A weight reference
    that is the param's own tensor, transposed, is just
    `{transpose: true}`.
  - **Decode state in the graph.** `rlx_ir::stateful::stateful_decode`
    gives a symbolic decode step a `position` input. RoPE rows become a
    `Gather` from held tables, the mask is computed in-graph
    (`iota[0..cap] <= position`, through a symbolic `Narrow` length), and
    each `KvAppend` becomes a `ScatterNd` at the position. The caches it
    returns are its next inputs. The `generate` entry gains
    `position` feeds and `state_update: "replace"`. `stateful::rebuild` is
    the general node-by-node rewriter it uses.
  - **A stateful step's caches stay in the executable.** A `generate`
    entry whose step keeps its caches as variables binds them once
    (`bind_gpu_handle`) and copies only the new row each step
    (`feed_kv_row`), reading back just the logits. The host used to pass
    every cache in and read it back whole each token: 56 × `cap` rows of
    4 KB each way for Qwen3-0.6B, ~0.9 GB a token at `cap` 4097. When the
    context outgrows a step, its caches move to the longer one, and a new
    generation keeps the longer step rather than recompiling the shorter.
    A backend that keeps no handles falls back to the host path.
  - **Derived symbols.** A dim that is a multiple of a symbol plus a
    constant (`seq*16`, `seq+1`) is a symbol of its own,
    `rlx_ir::dynamic::SymDef`, bound wherever its base is. `generalize`
    finds them (its offset no larger than its scale, so `seq²` flattened is
    refused), and `.rlxmodel` writes them in place: `f32[seq*16, 128]`.
    Qwen3's prompt graph, which flattens `[seq, heads, head_dim]`, now
    generalizes.
  - **`Op::DynamicUpdateSlice { axis }`.** `operand` with `update`
    written at a `start` read when the graph runs (JAX's
    `dynamic_update_slice_in_dim`; start clamped as JAX clamps), its output
    aliasing `operand`. Native on CPU and Metal (one kernel reads the start
    on the GPU); `rlx_fusion::lower_dynamic_update_slice` gives every other
    backend a `ScatterNd`. `stateful_decode` writes cache rows with it
    instead of `ScatterNd`, which Metal runs on the host: 56 GPU round trips
    a token for Qwen3-0.6B, each copying the cache. With it the exported
    Qwen3 step (f16 weights) decodes at ~14 ms a token on an M4 Pro, ahead
    of `Qwen3Runner`'s ~16 ms, and its first token (the prefill) takes
    ~34 ms against ~155 ms. `tests/dynamic_update_slice_backends.rs`
    checks CPU, Metal, MLX and wgpu against a computed reference.
  - **Package builds for a device.** `rlx-pkg build <in> -o <out> --target
    metal --target cpu` (`rlx_pkg::build::build_for_targets`) adds a
    rewritten version of each function, `step:metal` beside `step`, and a
    model opened on that device runs it (else the portable one). Recipes
    (`TargetRecipe::for_target`): Metal f16 matmul weights and attention on
    the shared K/V heads; CPU the attention only. The rewrites are
    `rlx_ir::retarget::{matmul_weights_as, gqa_native_attention, prune}`,
    which keep a function's variables and symbols. Weights and sidecars are
    copied byte for byte, and referenced weight paths rebased.
    `rlx_pkg::Package::sidecar_ids` lists a package's sidecars.
  - **Vulkan `barrier()` only synchronized a sub-group.** naga's GLSL front
    end parses `barrier()` with every barrier flag, sub-group included, and
    its SPIR-V writer takes the sub-group flag for a sub-group *execution*
    scope. Each warp then waited only for itself before reading shared tiles
    the others were still writing: on NVIDIA `scaled_lowp_matmul` was wrong
    from row 8 of every 16-row tile (e4m3 cosine 0.87 against the CPU; MoltenVK
    happened to keep warps in step). `build.rs` now drops the sub-group flag
    from workgroup barriers before emitting SPIR-V, for every shader. The
    Philox normal-stream gate is relative above 1 and absolute below, at
    1e-5: Box–Muller scales a last-ULP `cos` difference by up to ~4, which
    NVIDIA shows near zero crossings (Vulkan promises `cos` to 2^-11). The
    whole `rlx-vulkan` suite passes on an RTX 3080 Ti and on MoltenVK.
  - **Vulkan dropped a `KvAppend` whose row came from the GPU.** The
    executor's two run loops consumed a host copy that directly followed GPU
    work without running it (copies used to come only before a dispatch), so
    the cache kept its old row — and the build-segments loop also left every
    later step's barrier info one off. `Qwen3Runner` on Vulkan decoded
    garbage after its first token (and differently each run), while
    `kv_append_backends`, whose rows are graph inputs, passed. Both loops
    now leave such a copy to the loop head; `Qwen3Runner` on Vulkan matches
    the CPU's 40 tokens on an RTX 3080 Ti, and
    `kv_append_writes_a_row_computed_on_the_device` covers the case.
  - **Attention with fewer K/V heads than Q (grouped-query) was wrong on
    CUDA and Vulkan.** `Op::Attention` takes K / V with fewer heads; the
    CPU, Metal and wgpu kernels index a query head's K/V head, but the CUDA
    (and so ROCm, which shares them) and Vulkan kernels read K / V as
    `num_heads` heads and returned a wrong tensor, no error (0.62 / 0.72 off
    on an 8-over-2 case, RTX 3080 Ti). MLX and Metal's MPSGraph lowering
    aborted on it. CUDA, ROCm, Vulkan and MLX now repeat shared heads first
    (`rlx_fusion::expand_shared_kv::ExpandSharedKvHeads`, at each crate's
    own compile entry), and MPSGraph hands such attention to the thunks.
    `tests/gqa_native_attention_backends.rs` checks every backend against the
    broadcast graph on the CPU: all within 1.8e-7 (CUDA, Vulkan, wgpu on
    msi; Metal, MLX, wgpu here). The package builds' shared-head rewrite
    (`rlx_ir::retarget::gqa_native_attention`) is what found it.
  - **Referenced weights bind in their param's dtype.** An f16 / bf16
    param takes a checkpoint tensor converted once, two bytes an element,
    instead of failing (only f32 params bound before); held weights convert
    the same way. The host keeps them in that dtype too. The Qwen3 exporter's
    `weight_dtype` declares the matmul weights f16 or bf16
    (`rlx_core::package_export::matmul_weights_as`).
  - **A prefill function.** A `generate` entry's `prefill` runs a whole
    prompt at once, padded to the shortest of its sizes that holds it, and
    seeds the step's caches from its K/V outputs; `last` picks the row whose
    logits it returns. Longer prompts and continuations go through the step
    one token at a time. The Qwen3 exporter writes one (sizes 16, 64, 256),
    checked against a build at a third length. Each executable holds its own copy
    of the weights, so the prefill beside the step doubles them (2 × ~4 GB
    for Qwen3-0.6B in f32); the open option `prefill: "release"` frees it
    after each prompt, at a compile per prompt. On CUDA with
    `RLX_CUDA_SHARED_PARAMS=1` the two map one copy instead: the package
    compiles in its own sharing scope
    (`rlx_runtime::with_param_sharing_scope`, which restores the caller's),
    and Qwen3-0.6B's package takes 5.4 GB of GPU memory rather than 8.3 GB
    (RTX 3080 Ti; same tokens, chat "Paris").
  - **State kinds, packed references, speech recognition.** A `generate`
    entry's state is `rows`, `append` or `replace` (recurrent state, taken
    whole; a step with only that has no length limit), per state or by
    `state_kinds`. A U8 / I8 param referenced from a GGUF binds the
    tensor's bytes as stored (`WeightCache::bytes`), so K-quant weights stay
    packed. A `transcribe` entry runs `stages` (functions reading and naming
    values; a stage's `package` takes its function from another package, so
    an audio front end is exported once and shared), then a decoder that is
    a `generate` entry's loop with `{context: …}` inputs, a prompt,
    `stop_ids`, `suppress_ids` and `begin_suppress_ids`. In rlx-models:
    `rlx-qwen35` exports a Qwen3.5 GGUF (3.9 MB package, same tokens as
    `Qwen35Runner` on CPU and Metal) and `rlx-whisper` a Whisper checkpoint
    with its log-mel front end as a graph (2e-5 from `pcm_to_log_mel`; the
    same transcript as `WhisperRunner`). `Op::LogMel`'s doc now says it
    floors each frame at its own max, which is not Whisper's.
  - **Pipelines and image models.** A `pipeline` entry is stages alone:
    `Model.run` takes named tensors, runs the stages and returns named
    outputs. Stages, here and in `transcribe`, can be symbolic in their
    inputs' dims. They are specialized to the shapes they are given and
    compiled once per shape. A `"<value>#shape"` source feeds a value's
    dims as numbers, input tensors of any number type are converted to the
    graph's dtype, and a stage's `device` runs it elsewhere.
    `Opened::take_graph` returns a function as packaged. In rlx-models:
    `rlx-siglip2` exports a fixed-resolution SigLIP 2 image tower. Its
    preprocessing is a graph that is symbolic in the image's height and
    width: Pillow's antialiased bilinear resize in Pillow's fixed point,
    exact in f32. On the test images it is closer to Pillow than the
    hand-written preprocessing (1 vs 227 of 150,528 values off at 300×200),
    and it can be exported as its own package that runs alone. Embeddings
    match `Siglip2Runner::encode_image` on CPU and Metal (cosine ≥
    0.999996). Whisper's front-end package runs alone too.
  - **Context past the exported sizes.** A `generate` entry with
    `max_context` grows its symbolic step past `sizes`. When the context
    outgrows every listed size, the step is specialized at twice the
    longest cache, capped at `max_context`, with each symbol's offset from
    the cache length kept. `Opened::{specialize_at, specialize}` replace
    `take_function`'s inline specialization. In rlx-models,
    `rlx-qwen3`'s `ExportOptions::max_context` sizes the RoPE tables and
    sets the entry. Qwen3-0.6B exported with one 16-token size produces
    `Qwen3Runner`'s 40 tokens on Metal, and the same ids on CUDA (checked
    against those, as the runner did not fit beside another GPU tenant).
    Also on CUDA (RTX 3080 Ti): the SigLIP 2 package (cosine ≥ 0.99999 to
    the runner), the Qwen3.5 package (chat "Paris") and Whisper's package
    (the CPU transcript).
  - **Batch generation from a package.** `Model::generate_batch` and
    `LmRunner::generate_batch` (by default one after another). A
    `generate` entry naming its step's batch symbol (`batch`) decodes the
    prompts together. The sequences run in lockstep from position 0,
    prompts of any lengths, each ending at its own stop id; a stateful
    step keeps the caches resident and feeds each sequence's new row.
    `rlx_ir::stateful` accepts a batched step: table rows and the mask
    repeat over the batch, and the cache write is a `DynamicUpdateSlice`
    at any batch. `rlx_flow::blocks::with_kv_append_any_batch` builds such
    a step with `KvAppend` past batch 1, for that rewrite. `rlx-qwen3`
    exports its step symbolic in the batch too (`ExportOptions::batch`,
    default on). Three prompts give exactly the text each gives alone on
    CPU, Metal and CUDA, in about half the time (Metal 300 vs 664 ms; CPU
    1.30 vs 2.59 s; CUDA 498 vs 1030 ms). Batch 8 decodes at 254 tok/s on
    Metal (317 built for Metal), against 67 for one sequence.
    `generate_batch` is forwarded by every model and runner wrapper
    (`Stamped`, `Box<dyn LmRunner>`, `HostDriven`); a wrapper that
    forwarded only `generate_text` had made every batch run one by one,
    which equality checks alone did not catch. A runtime test now checks
    that the batched path ran. With `prefill: "release"` the batched step
    is freed after each batch. A batch runs its prompts' shared length less
    one through the prefill function for each sequence, seeding the
    batch's caches (open option `batch_prefill`, on by default). Four
    200-token prompts on Metal: 1.26 s, against 10.3 s a token a step.
  - **Beam search.** `num_beams` / `length_penalty` on any text model.
    One algorithm (`rlx_runtime::lm::beam_search`, ties broken the same
    way everywhere) under `LmRunner::generate_beam`. By default each
    hypothesis is extended by its own `predict_logits`. A package with a
    batch symbol keeps the hypotheses as the sequences of its batched
    step, prompt once, caches reordered after each step. Batched and
    one-by-one give the same text, on a toy package (2-4 beams) and on
    Qwen3-0.6B (CPU, Metal, CUDA). On Metal 3 beams take 328 ms vs 4.3 s.
    Open option `batched: false` takes the one-by-one paths. In
    `prefill: "release"` mode a batch prefills before its step compiles,
    and the single-sequence step is freed meanwhile: three executables
    had run CUDA out of memory beside another GPU tenant.
  - **Packages say which entry features they use.** Writing a package
    records `features` (each entry and each of its keys, `generate.check`, …)
    in its model block. An rlx opening one that uses a feature it lacks
    refuses it by name ("uses generate.teleport that this rlx (0.2.18) does
    not run; update rlx") rather than run without it
    (`rlxp::FEATURES`, `rlxp::features_of`). Format and opset versions
    (`requires`, `opset`) were already checked.
  - **Input / output contracts.** A `pipeline` entry's `inputs` and
    `outputs` may be contracts, `{name, kind, dtype, shape, …}` with
    anything else a caller needs (`color: "rgb"`, `sample_rate`). `info()`
    gives the typed specs and, under `metadata.inputs` / `outputs`, the
    contracts. `run` refuses a tensor of another rank or static size, and
    `embed`, `generate` and `transcribe` describe theirs (text; audio with
    its sample rate). The SigLIP 2 and Whisper exporters declare theirs
    (u8 RGB `[height, width, 3]`; 30 s of 16 kHz mono).
  - **Generation defaults, tools, JSON output.**
    - A `generate` entry's `generation` (from `generation_config.json`:
      `do_sample`, `temperature`, `top_p`, `top_k`, `min_p`,
      `repetition_penalty`, `max_new_tokens`) fills in what a caller leaves
      out of `generate_text`, `chat` and `generate_batch`. `do_sample: false`
      stays greedy. `run` with ids keeps its own defaults.
    - `TextCodec::apply_chat_template_json` renders JSON messages (an
      assistant turn's `tool_calls`, a tool turn's `tool_call_id`) with
      `tools`, `enable_thinking` and `add_generation_prompt`; `chat`
      passes them through. A template with no `tools` branch refuses tools
      rather than drop them.
    - `response_format: "json"` (or `{"type": "json_object"}`) keeps the
      reply to one JSON value: each step, of the 256 likeliest tokens, only
      those whose text continues a JSON prefix may be picked
      (`model::json_prefix::JsonPrefix`), and a stop token only once the
      value is whole.
    - Qwen3-0.6B: `{"city": "Paris"}` in JSON mode; told about a
      `get_weather` tool, it calls it with `{"city": "Paris"}`.
  - **Loops in a pipeline.** A `pipeline` entry may run `stages`, then a
    `loop: {stages, steps, schedule, until}`, then `finish` stages. Each
    step sees `step` and `steps`, and its value from each `schedule` list
    (a diffusion's timesteps, a sampler's sigmas). The loop's outputs
    replace earlier values, and it ends early once the `until` value is
    true. The `steps` option overrides the count, and the `seed` option
    is a value for stages that draw noise. All the stages share one
    weight set. Tested with Newton's iteration for √2, stopping when it
    converges.
  - **Images (and other media) in a prompt.** A `generate` entry's `media`
    gives an `embed` function (ids to embeddings), a prefill that takes
    embeddings (padded to listed `sizes`), and `kinds` (`image: {token,
    input, stages, output}`). `Model::generate_with_media(prompt, media,
    options)` (and `LmRunner::prefill_with_media`) works like this: each
    run of a kind's placeholder token takes the next medium, expanded to
    as many positions as its embedding has rows; the media rows replace
    those positions in the prompt's embedding; the prefill seeds the
    caches; decoding continues by tokens.
  - **Multimodal RoPE (Qwen-VL).** `media.mrope: {sections, layout, cos,
    sin}` gives each token a `(t, h, w)` position (an image's rows its grid,
    from `kinds.<kind>.grid.patch`) and the prefill each token's RoPE rows
    (`rope_cos` / `rope_sin`), frequencies by section (Qwen2/2.5-VL) or
    interleaved (Qwen3-VL). Decoding goes on from one past the image's
    largest position: `rlx_ir::stateful::StatefulDecode::rope_offset` reads
    the tables at `position + rope_offset`, the decode input feed
    `rope_offset` supplies it (and a host-fed table row is taken there).
  - **Graphs symbolic in a product of sizes.** A derived dim may be the
    product of two symbols times a scale plus an offset (`h*w*4`;
    `SymDef::times`, written `h*w*4` in `.rlxmodel` text). `generalize`
    tries pairs of symbols when no single one fits, and binding works both
    ways: a bound derived dim (an image's `h*32` rows) binds its base, so a
    stage is specialized from its inputs' shapes alone. The Qwen3-VL and
    Qwen2.5-VL vision towers are now one graph each for every image size,
    their position tables, patch order, RoPE rows and window masks
    computed in the graph from the grid.
  - **Images of any size.** `media.kinds.<kind>.resize: {multiple,
    min_pixels, max_pixels}` resizes a `u8 [height, width, 3]` image as
    Qwen-VL's processor does (`model::image_resize::smart_resize`, Python
    rounding) with an antialiased bicubic that matches torchvision's
    `uint8` resize value for value (`resize_rgb8_bicubic`; int16 weights at
    the widest precision that fits, rows first, each pass rounded).
    Qwen2.5-VL-3B and Qwen3-VL-2B on 448×448 and 500×375 images (the
    latter resized to 364×504 / 384×512): `transformers`' text, CPU and
    Metal.
  - **Weights stream, and fit.** Measured on Qwen2.5-VL-3B (CPU, exact
    `/usr/bin/time -l` footprints), the run was 74 GB at its peak on a
    64 GB machine — the OOM kills.
    - Binding reads each tensor's byte range from its safetensors file (the
      `WeightCache` held whole files, 7 GB here), and `PackageWeights` no
      longer keeps every resolved weight beside the backend's copy.
    - Open option `weight_dtype: "f16" | "bf16" | "auto"` narrows matmul
      weights as they bind; the CPU now keeps a half weight read only as a
      matmul's right operand packed (`promote_to_f32_keeping_half_weights`;
      its `SgemmF16` / `SgemmBf16` read it), where it used to widen it back.
    - A package opened narrower or short of memory compiles without the
      weight-concatenating fusions (`FusionOptions::no_weight_concat`, the
      `RLX_NO_WEIGHT_CONCAT_FUSION` env as a compile option): their packed
      QKV / gate-up weights were a resident second copy of ~65% of a
      transformer's weights.
    - `prefill: "release"` also frees media stages (and the step before a
      media prompt), and is the default when the functions' f32 copies
      would not fit in half the memory (`"keep"` opts out).
    - Metal keeps f16 weights and embedding tables half: its compile
      promotes a graph with any half node to f32, and the half rows of a
      half embedding table widened every weight of the step and the text
      prefill back (`prepare_f32_exec_graph_keeping_half_matmul_weights`;
      bf16 still widens, Metal has no bf16 storage). The table's gather
      stays a half row copy, read through a new `gather_axis0_h_fidx`
      kernel: `gather_axis0_h` read the ids as half, but the arena holds
      integer ids as f32 (`Thunk::Gather.idx_dt` picks). Qwen2.5-VL-3B's
      whole test on Metal with `weight_dtype: "f16"`: 36.9 → 21.1 GB peak.
      A video's tower compiles once for all its frame pairs under `prefill:
      "release"`.
    - **One copy of each weight across Metal executables.** A package's
      decode step, prompt prefill and media prefill each held their own
      weights. Under a sharing scope (`with_param_sharing_scope`; every
      package compile now runs under its own) the Metal weight buffer is a
      page-aligned VM region wrapped no-copy, and each weight in it is an
      owned Mach memory entry from a process-wide registry
      (`rlx_metal::shared_weights`, by scope, name, dtype and size): later
      executables map the same pages and skip binding them
      (`CompiledGraph::param_already_set`). Owned entries are charged to
      the footprint once however often mapped (plain shared anonymous pages
      are charged per mapping — what iOS's limit counts). Under a scope the
      weight buffer may pass 4 GiB; such an executable stays off MPSGraph,
      whose binds fail there. Whole tests on Metal (images, video, text),
      peak: Qwen3-VL-2B f32 39.9 → 16.4 GB, f16 22.5 → 8.9 GB; Qwen2.5-VL-3B
      f32 36.8 → 22.5 GB, f16 21.1 → 12.3 GB — the same logits and text, and
      faster (mapped weights are not read again).
      `RLX_METAL_SHARED_WEIGHTS=0` keeps a copy per executable.
    - A dropped Metal executable gives its memory back. Compiling to
      MPSGraph made autoreleased objects (the lowered graph, its
      executable, the arrays bound to the arena) that retain the arena;
      on a thread with no autorelease pool they lived until the thread
      exited, so each compile-and-drop kept its whole arena (6 cycles of a
      128 MiB weight: +775 MiB, now 0). Each compile now runs in its own
      pool, and the no-copy view MPSGraph binds at an offset is released.
      Qwen2.5-VL-3B f32 test on Metal: 22.5 → 16.4 GB shared, 36.8 →
      30.7 GB unshared; the released stages' memory now returns.
    - A package compiles lean (no weight-concat packs) also when its
      functions' f32 params times 1.7 — the packs measured — would pass
      half the memory: Qwen3-VL-2B's f32 test on Metal 43.1 → 32.1 GB, and
      faster (73 → 56 s).
    - One step of Qwen2.5-VL-3B: 24 GB → 11.3 GB (f16); a prompt with an
      image 24 → 14.6 GB. Its whole test (two images, then text): 74 GB →
      34 GB as opened (now lean and releasing by itself), 20 GB with
      `weight_dtype: "f16"`, the same text.
  - **Per-medium extra rows.** `media.kinds.<kind>.rows: [names]`: further
    outputs of the kind's stages, laid out like its embedding, reach the
    prefill as `[1, seq, width]` values, zeros but at each medium's
    positions (Qwen3-VL's deepstack features).
  - **Video.** `media.kinds.<kind>.frames: {patch, layout, t_step}` takes
    `u8 [frames, height, width, 3]`, runs the kind's stages `patch` frames
    at a time (the last repeated to fill a group), and lays the groups out
    as one placeholder run each (`"runs"`, Qwen3-VL) or as one run with
    group `g` at time position `g · t_step` (`"joined"`, Qwen2.5-VL).
    `resize.over_frames` bounds a video's pixels over all its frames
    (`image_resize::smart_resize_frames`, Qwen3-VL's video rule).
  - **Half embedding tables.** `weight_dtype` now narrows embedding tables
    too (`rlx_ir::retarget::embedding_tables_as`: the table f16/bf16, each
    gathered row widened); the CPU keeps a table read only by
    `Gather{axis: 0}` packed and widens just the rows it gathers
    (`Thunk::Gather.table_half`).
  - **Qwen3-family MoE MLP in rlx-flow.** `Qwen3Moe` on a decoder layer
    spec: router softmax, top-k, optional renormalization, then each slot's
    experts through two `GroupedMatMul`s. It reads the fused expert tensors
    in either orientation (the original `[E, H, 2I]` or transformers 5's
    `[E, 2I, H]`) or per-expert linears.
  - **Sliding-window masks in a stateful step.**
    `StatefulDecode::windows: [(input, window)]`: further mask inputs, each
    the step's mask admitting only `position - window < key`, computed from
    the same `iota` (rlx-flow's Gemma decode layer reads one by name,
    `GemmaDecodeLayerSpec::mask_input`). The rewrite now makes one param
    node per name: a second `iota` node was left unbound (a backend binds a
    name once), so the plain mask compared zeros and admitted every key.
  - **Batched prefill functions.** `generate.prefill.batches: [{function,
    batch}]`: the prompt function at fixed batch sizes. `generate_batch`
    runs its prompts' shared prefix through the smallest that holds them in
    one call instead of one after another (`batch × seq` products keep it
    from being one symbolic function). The Qwen3 exporter adds batch 2, 4
    and 8 (`ExportOptions::prefill_batches`); three prompts decode to the
    same text.
  - **Speech synthesis entry.** `synthesize: {tokenizer, template,
    sample_rate, stages, voices, decoder, token_offset, codebook,
    output}`. The text, tokenized into the template's `{text}`, is the
    value `ids`, and a `voice` vector is taken from `voices` (option
    `voice`). Then either stages make the audio straight from the ids, or
    a `decoder` (a `generate` entry, greedy or sampled) writes codec
    tokens first, and their ids, less `token_offset`, within `codebook`,
    are the value `codes` the stages decode. It is a `text-to-speech`
    model (`synthesize`).
  - **Weights: checksums, adapters, quantized storage.**
    - A weight source may carry the file's `size` (checked on every open:
      "the weights changed under it") and `sha256` (checked with the open
      option `verify_weights: true`). `rlxp::weight_source` writes both,
      and the Qwen3, Qwen3.5, Whisper, SigLIP 2 and BERT exporters use it.
    - Adapter packages: `rlxp::save_adapter(path, name, base,
      [LoraUpdate{tensor, rank, a, b, scale}])` writes `{adapter: {base,
      lora}}`. Opening one opens its base with `W + scale · B·A` merged
      into each named checkpoint tensor (or held param) as the weights
      bind, and adapters stack.
    - `rlxp::quantize_package(in, out, GgufQ8_0 | GgufQ4_0)` stores a
      package's f32 matmul weights quantized in it. Each `MatMul(x, w)`
      becomes a `DequantMatMul` over `w`'s GGUF blocks, read from
      wherever `w` was. Qwen3-0.6B, from 2.4 GB of f32 weights: Q8_0 is
      636 MB with the same 40 greedy tokens, Q4_0 is 338 MB (chat
      "Paris"; its tokens part from f32's after the first).
    - `rlx_pkg::build::rebase_weight_uris` is now public.
  - **Serving: a draft model, continuous batching.**
    - The open option `draft: "<package>"` gives speculative decoding a
      draft model, a package with the same vocabulary, instead of prompt
      lookup. Its greedy guesses come from its own cache, rolled back
      (`truncate`) to where it and the context agree. With the package as
      its own draft, Qwen3-0.6B gives the same 40 tokens.
    - Continuous batching: `LmRunner::serve_open(slots)`,
      `serve_add(prompt)`, `serve_step()` (each slot's next token once its
      prompt is in), `serve_remove(slot)`. Requests join and leave between
      steps.
      - Every slot runs at the batch's position, from its own start row:
        `rlx_ir::stateful` takes a `starts` input, and the mask admits
        `starts[b] ..= position`. RoPE depends only on the distance
        between positions, so a late start computes what a start at 0
        does.
      - When the batch reaches the end of its cache, each request's rows
        move back so they all end together. Its keys are rotated back by
        the rows they moved, `R(-δ)`, using the entry's
        `rope: {cos, sin, style, head_dim, keys}`, which the Qwen3
        exporter writes.
      - On Qwen3-0.6B: B joins A's batch ten steps in, A leaves, and the
        32-row cache moves back. Each request gets exactly its solo
        tokens (CPU and Metal).
  - **Rerankers: a `score` entry.** `Model::score(query, passages,
    options)` returns one relevance score per passage, for a
    `"text-ranking"` model. `TextRanking` + `text_ranking_model` adapt a
    provider's own, and `run` takes a `query` and `passage` text tensors
    and returns `scores`. A package's `score: {functions, sizes,
    tokenizer, inputs, position_offset, label, activation}` runs each pair
    as the tokenizer pairs it (`TextCodec::encode_pair`: ids and segment
    ids, the new input role `token_types`). The pair goes through an
    encoder whose output is `[batch, labels]` logits. `label` picks the
    column, and `activation: "sigmoid"` maps it into (0, 1). A pair too long for the
    longest function loses the end of its passage. `embed` and `score`
    share one encoder runner. rlx-embed's `export::export_reranker_rlxp`
    appends the classifier head (BERT's pooler + `classifier`, or RoBERTa's
    `classifier.dense` + `out_proj`) to the encoder graph:
    ms-marco-MiniLM-L6-v2 scores within 2e-6 of `transformers` (CPU,
    Metal).
  - **Llama, Mistral and Qwen2 checkpoints through the Qwen3 exporter.**
    The family comes from `config.json`'s `model_type`. A config that
    does not mention `qk_norm` defaults it off for those types. Scaled
    RoPE (`rope_scaling`) is refused rather than built wrong.
    SmolLM2-135M exports and greedy-decodes the same 20 tokens as
    `transformers`, alone and in a batch (CPU, Metal).
  - **Speculative decoding (prompt lookup).** A `generate` entry's `check`
    function runs several tokens after its cache (the last token, then a
    draft). With the open option `speculative: true`, greedy decoding
    drafts what followed the context's latest 3-, 2- or 1-gram, checks it
    in one run, and keeps the draft as far as the model agrees, plus the
    model's own pick. The tokens are the same as one at a time.
    `rlx_ir::stateful` takes such a step: `concat(cache, rows)` becomes a
    `DynamicUpdateSlice` of all the rows, the tables give
    `position..position + tokens`, and the mask is a bias letting token `i`
    see keys `0..=position + i`. rlx-flow's Qwen3 decode layer takes
    several tokens (per-head norm over them, a bias mask);
    `rlx_qwen3::flow::build_qwen3_check_built` builds the check, and the
    exporter adds it (`ExportOptions::speculate`, 3 by default). Qwen3-0.6B
    on Metal: the same 40 tokens in 680 ms against 891 ms.
  - **Metal: bias attention in the tiled prefill kernel.** An additive bias
    (`MaskKind::Bias`) fell to `sdpa_long`, one thread per head: a
    4-token check's attention took 70 ms a step. `sdpa_prefill_fa` (and
    its MMA variant) now adds the bias. Checked against the CPU with 4
    queries over 37 keys on Metal, MLX and wgpu.
  - **Metal: a few rows of f32 matmul read the weights once.**
    `gemv_f32_splitk_rows` runs m = 2-4 as the split-K GEMV a single row
    uses: each weight is read once for every row, with the same 32-way K
    split and order, so each row is bit-identical to that row alone. The
    tiled sgemm had reread the weights and cost 2× a single decode step at
    m = 2-4. Now a batch-2 step costs 1.18× and batch 4 1.51×. At m = 8,
    two passes lost to the sgemm (38 vs 31 ms), so the limit is 4.
    `RLX_METAL_GEMV_ROWS=0` opts out.
  - **`DynamicUpdateSlice` lowers at any batch.** The `ScatterNd`
    lowering (CUDA, ROCm, Vulkan, wgpu, MLX) indexes every position
    before the axis, so a batched cache `[batch, cap, width]` writes each
    sequence's rows; it covered batch 1 only, and CUDA rejected a batched
    stateful step. Checked against the reference on CPU, Metal, MLX, wgpu,
    CUDA and Vulkan.
  - **Open options for packages.** `stage_devices` sets a stage's device
    over its entry's; `max_context` lowers a package's; `context_growth`
    sets how much a grown step grows (2).
  - **`rlx_ir::env` reads are cached.** Every `RLX_*` read went to the
    override map and then `getenv`, allocating the key and the value: ~80-100
    ns each, and backends read their switches per dispatch (92 reads in
    Metal's encoder, 54 in CUDA's run loop; 844 call sites read 581 keys).
    Values are now resolved once and kept; `flag` / `var` / `parse_or` take
    17-24 ns, with no allocation for `flag`, `flag_or`, `parse_or` and
    `is_unset`. `set`, `remove`, `unset`, `clear_overrides` and
    `RuntimeOverrides` empty the cache; `rlx_ir::env::reload()` does it for
    code that changes `RLX_*` variables with `std::env::set_var` after
    reading them (the PS5 node, which seeds them from its config file).
    One read path: `rlx_ir::env` resolves a key through the registry
    (canonical spelling, aliases, a deprecated name's replacement) on its
    first read and caches the result, so every reader gets alias and
    deprecation handling; `env_registry::{var, flag, parse_or}` forward to
    it (`flag_or` keeps the registry's declared default) and its copy of the
    truthy parser is gone. rlx-wgpu's FFT switches and the Cortex-M trainer
    read flags with `env::flag` instead of private helpers that accepted
    fewer spellings (`RLX_FFT_WGPU_ONCHIP=yes` was off).
  - **`prefill_device`: the prompt on another device.** The open option
    `{"prefill_device": "cpu"}` runs a package's prefill on the CPU and
    hands its K/V rows to the step's caches on the model's device, keeping
    the prefill's weight copy out of GPU memory. It is a memory lever, not
    a speed one: the first token is slower at every prompt length, and the
    gap grows with the prompt (Qwen3-0.6B, best of five, same tokens):

    | prompt | Metal | CPU prefill | CUDA | CPU prefill |
    |-------:|------:|------------:|-----:|------------:|
    | 16     | 25 ms | 77 ms       | 21 ms | 155 ms     |
    | 64     | 128 ms | 323 ms     | 33 ms | 456 ms     |
    | 250    | 120 ms | 393 ms     | 116 ms | 2656 ms   |

    (M4 Pro; RTX 3080 Ti shared with another job.) On CUDA,
    `RLX_CUDA_SHARED_PARAMS=1` saves the same memory without the cost.
    `rlx-qwen3`'s `tests/prefill_device_bench.rs` (opt-in,
    `RLX_BENCH_PREFILL=1`) measures it; each run takes a different prompt,
    since the same prompt again is a continuation of the cache and returns
    its first token without running anything.
  - **Hugging Face model cards.** The model block's `card_data` holds a
    card's metadata under the Hub's own keys (`license`, `language`,
    `tags`, `datasets`, `base_model`, `pipeline_tag`, `model-index`, …), and
    `card` holds its Markdown body. Exporters read a checkpoint's
    `README.md` into them (`add_huggingface_card`, which parses the YAML
    front matter). `rlxp::huggingface_readme` writes a Hub `README.md` back
    from a package. Both are in `Model.info().metadata`.
  - **Entries: a language model from the package alone.**
    `entries.generate` declares the decode functions (one per cache
    capacity), what feeds each input (the token, a row of a held table, the
    bucket's mask, a cache), which output updates each cache, the stop ids,
    and the chat template. The `rlxp` provider then runs it as an
    `LmRunner`, so `chat`, `generate_text` and `generate` (greedy or
    sampled) work with no provider code. A continuation feeds only its new
    tokens. Tokenizer loaders now also take the chat setup
    (`TokenizerLoader = fn(&[u8], &Value)`), and rlx-text renders the
    template. rlx-models' `rlx-qwen3` exports Hugging Face checkpoints this
    way (`export::export_rlxp`): Qwen3-0.6B becomes a 4.9 MB package with a
    4096-token context that references 311 of its 313 params. Opened by rlx
    alone, it greedy-decodes the same 24 tokens as `Qwen3Runner` and
    answers `chat` from the model's own template, from Rust and Python.
    The shared matcher that splits params into referenced and held ones
    lives in `rlx-models-core`'s `package_export`.
  - **`dist::WeightCache` reads a safetensors file once.** It used to
    re-read the whole file for every tensor. For a 1.1 GB checkpoint with
    200 tensors, that was about 220 GB of reads.
  - **`NaN` and `±inf` in `rlx_ir::text`.** They used to print as JSON
    `null`, which didn't read back. An op holding one now prints in the
    `.rlxmodel` value syntax (`Clamp{min: -inf, max: nan}(%0)`), which the
    parser also accepts. Every other op prints as before. `.rlxmodel` keeps
    a non-canonical NaN's exact bits (`nan(0xffc00000)`).
- **Linked model plugins, for iOS, Android, wasm and consoles.** These
  platforms can't load a plugin at run time, so the same plugin crate links
  into the app's library instead. A plugin lists its providers once:
  `export_model_plugin!("name", [A, B])`, and an entry may carry a `#[cfg]`.
  The host names its plugins with `link_model_plugins!` (rlx-runtime feature
  `linked-plugins`), and the app calls `rlx_link_model_plugins()` once
  (Swift: `linkModelPlugins()`). That call keeps the providers in a static
  library whatever the build profile. Before this, a C linker silently
  dropped their start-up registrations unless the library was built with fat
  LTO. Linked builds also leave out each plugin's `rlx_model_plugin_v1`
  symbol, so several plugins can share one fat-LTO library, where before
  they failed with "symbol multiply defined". `ios/build-xcframework.sh
  --capi-crate` packages such a library as `RLX.xcframework`. On those
  targets rlx no longer scans `~/.rlx/plugins`, and the "no provider" error
  says how to link one. The built-in `rlxp` provider is now registered in
  code, not at start-up.
- **Models from any provider, in every language: `rlx_runtime::model`.** A
  `Model` opens a model by path or id and runs it by name, with named, typed
  tensors. Language models also get `generate`, `generate_text` and
  `predict_logits`, with sampling options. Providers register with
  `register_provider!`. The built-in `rlxp` provider runs a graph saved with
  `save_model` on any device. Plugins are `cdylib`s exporting a C table
  (`rlx_model_plugin_v1`, ABI 1), loaded from `RLX_MODEL_PLUGINS` or
  `load_model_plugin`, so a model collection downstream of rlx reaches every
  binding without rlx depending on it. `rlx-model-plugin` is the reference
  plugin and authoring guide. Every binding gets `Model`, `model_providers`,
  `load_model_plugin` and `save_model` from `api.idl`, plus a sixth `models`
  example. The example is checked in all 25 languages: natively, under
  `RLX_TEST_DOCKER=1`, and in the Python, JS, Swift and C suites.
- **Task methods: `chat`, `transcribe`, `synthesize`, `embed`.** `Model`
  answers each task with one call, in every binding. Making a model runnable
  takes one task trait (`SpeechToText`, `TextToSpeech`, `TextEmbedding`, or
  `LmRunner` for text) plus a declarative `Provider` with `Claim`s: by id,
  prefix, GGUF `general.architecture`, HF `model_type`, file name or
  extension. `chat` applies the tokenizer's chat template
  (`TextCodec::apply_chat_template`). Plugins in `~/.rlx/plugins` load with
  no setting needed, and a method called on the wrong kind of model names the
  one that fits. The tasks travel as named tensors over `run`, so plugin ABI 1
  is unchanged.
- **C and C++ on microcontrollers: `rlx-micro`.** The C ABI of `rlx-cortexm`'s
  INT8 kernels (conv2d, dense, max-pool, ReLU, argmax, quantize) and its MNIST
  model. It is `no_std`, never allocates, and needs no OS or libc. Calls
  check buffer sizes and return a status instead of trapping. Ships
  `rlx_micro.h`, a freestanding C++ `rlx/micro.hpp` (no exceptions, heap or
  STL), C/C++ examples, and Cortex-M start-up, semihosting and linker-script
  files. The firmware is 24.9 KB flash / 10.8 KB RAM. `cargo test -p rlx-micro`
  runs the examples on the host and bare-metal on a Cortex-M4 under QEMU
  (`mps2-an386`).
- **One op table generates every language binding: `rlx-bind`, `rlx-capi`,
  `rlx-bindgen`, and Swift.** `crates/bindings/rlx-bind` holds the graph
  builders as one table (`for_each_graph_op!`, ~200 rows) plus the runtime as
  plain Rust (`Session`, `Compiled`, `Trainer`, `Optimizer`, tokenizer,
  sampler, `check`, devices, GGUF, RLXP, weight loaders). rlx-js and pyrlx now
  expand that table at compile time instead of keeping two hand-written copies,
  and wrap the shared runtime. `crates/bindings/rlx-capi` is a C ABI over the
  same surface (`include/rlx.h`, 366 functions) for any FFI language;
  `crates/tooling/rlx-bindgen` generates its `rlx_graph_*` glue and its header
  (parsed from the crate, so it cannot drift), the Swift `Graph` methods, the
  `.pyi` `Graph` stubs and `docs/bindings-api.md`. `crates/bindings/rlx-swift`
  is a SwiftPM package on the C ABI whose runtime classes are generated too
  (see the next entry), with `RLXError.kind` and an XCTest suite (`swift test`);
  `ios/build-xcframework.sh --capi [--apple]` packages the static library as
  `RLX.xcframework` (iOS + macOS) and `RLX_XCFRAMEWORK=1` builds the package on
  it, for apps. Adding an op is one table row and
  `just gen-bindings`; `just check-bindings` (in `just ci`) fails on stale
  output, and an `ir_drift` test fails when rlx-ir gains a builder the table
  does not reach. C and Swift (example + XCTest, and a check that every C
  function has a Swift wrapper) run end to end in `cargo test -p rlx-capi`. See
  `docs/bindings.md`.
- **A runtime API table keeps every binding one-to-one.**
  `crates/bindings/rlx-bind/src/api.idl` lists every runtime class member and
  function — sessions, `FlexibleSession`, `Runner` / `Router`, autodiff,
  `check`, devices, GGUF, weights, RLXP, training, text, distributed — with
  argument and return kinds and its name in Python, JS, C and Swift.
  `rlx-bindgen` checks each row against the exported C function's parameter
  types (and each C function against a row) and generates
  `rlx-swift/Sources/RLX/API.swift` from the rows; only the callback loop and
  the in-place optimizer step are hand-written. rlx-js and pyrlx test the table
  both ways (every row is exported, every export is a row or a listed host
  helper). Filling the gaps it found added `parse_device`, `FlexibleSession`,
  `Runner.from_env` / `fastest` / `resolve` / `set_params` / `run_on` /
  `run_chain` / `run_with_inputs`, `Router.from_env` / `devices` / `run_on` /
  `set_params` / `set_param_typed`, `Gguf.architecture` / `metadata_value`,
  `convert_to_gguf` and `export_fpga` to the C ABI and Swift; `Graph.clone` /
  `is_consumed`, `load_weights` and `to_rlxp` to Python and JS where they were
  missing; and `rlx_trainer_freeze` / `unfreeze` in place of `set_frozen`.
  `load_weights` also takes `"safetensors"`. The name map is in
  `docs/bindings-api.md`.
- **Distributed nodes from every binding.** `rlx_bind::runtime::dist::Group`
  (and `Group` in Python, JS and Swift, `rlx_group_*` in C) joins a mesh of rlx
  nodes: `connect` / `from_env`, `send` / `recv`, `all_reduce`, `all_gather`,
  `broadcast`, `barrier`, and the ship-graph protocol — `collect_caps`,
  `ship_stage` / `serve_worker` / `send_activation` / `recv_activation` for
  pipelined inference, `ship_train` / `run_train` / `serve_trainer` for
  data-parallel training — plus `loopback_peers` for running every rank on one
  machine. Python releases the GIL in each blocking call, so ranks can be
  threads.
- **Fifteen more generated languages.** C++ (`rlx-cpp`, header-only RAII,
  designated-initializer options), Node.js / TypeScript (`rlx-node`, koffi, a
  `.d.ts`), Kotlin (`rlx-kotlin`, generated JNI glue — the JVM and Android),
  Dart / Flutter (`rlx-dart`, `dart:ffi`), Lua (`rlx-lua`, a C module), MATLAB /
  GNU Octave (`rlx-matlab`, one MEX gateway + a `+rlx` package), R (`rlx-r`,
  `.Call`), PHP (`rlx-php`, FFI), Nim (`rlx-nim`), D (`rlx-d`), V (`rlx-v`),
  Haskell (`rlx-haskell`, `foreign import ccall`), OCaml (`rlx-ocaml`, C
  stubs), Elixir (`rlx-elixir`, dirty NIFs) and Odin (`rlx-odin`). Each is
  printed by `rlx-bindgen` from the same lowered IR, has a README, a test
  suite and the five examples printing the same numbers, and runs in
  `cargo test -p rlx-capi --test ffi_languages` — natively, or with
  `RLX_TEST_DOCKER=1` in the language's container against a Linux build of
  the library (`just linux-capi-volume`).
- **Go, Java, C#, Ruby, Julia and Zig bindings, all generated.** `rlx-bindgen`
  lowers both tables into one typed view of the C ABI (`src/ir.rs`, every
  function checked against its exported signature) and prints each language
  from it: `crates/bindings/rlx-go` (cgo; `…Opts` structs, `(T, error)`),
  `rlx-java` (the FFM API, Java 22+; no JNI), `rlx-csharp` (P/Invoke, .NET 8+),
  `rlx-ruby` (stdlib Fiddle, no gems), `rlx-julia` (`@ccall gc_safe`, no
  dependencies) and `rlx-zig` (`@cImport`, Zig 0.16). Each has every C
  function, a README, a test suite and the same five examples printing the
  same numbers as Python, JS and Swift; `cargo test -p rlx-capi --test
  ffi_languages` runs them all (C# in the .NET SDK container with
  `RLX_TEST_DOTNET_DOCKER=1` when `dotnet` is absent). The C ABI gains
  `rlx_{u32s,f32s,bytes}_release` for FFIs that cannot pass a struct by value.
- **The same five examples in Python, JavaScript and Swift.** `inference`,
  `training`, `loading`, `distributed` and `backends` live in each package
  (`pyrlx/examples/`, `rlx-js/examples/`, `rlx-swift/Examples/` as
  `rlx-example-*` products) and print the same numbers; each binding's tests
  run them (the JS distributed example as a real three-process mesh, the Swift
  and Python ones as three threads). `crates/bindings/README.md` indexes every
  language's package and examples, and the top-level README links it. The C
  example moved to `rlx-capi/examples/train_linear.c`.
- **The JS (`rlx-js`) and Python (`pyrlx`) bindings now reach the whole
  `rlx_ir::Graph` builder surface.** JS grows from 101 to 205 `Graph` methods
  (all chainable through `rlx.dsl`), Python from ~60 to ~200: conv 1d/3d and
  transposed variants, pooling, interpolation, attention with a bias / v-dim /
  score scale / logit softcap, RoPE `style` (`neox` / `gptj`), batch / adaLN /
  fused residual norms, LSTM / GRU / RNN, Mamba-2, Gated DeltaNet, `scan` over
  a body graph, scatter / gather ND, per-channel and block-scaled quantization,
  `dequantMatmul` (int8, MLX, GGUF K- and IQ-quants), VQ, KAN splines,
  FIR / IIR / SOS / Hilbert DSP, spectral features and SPD manifold layers.
  `activation` / `binary` in both read `Activation::ALL` / `BinaryOp::ALL`, so
  every IR activation and binary op is reachable by name. Python also gains
  what only JS had — `check`, precision policies, `Trainer` / `Optimizer`, the
  `rlx-text` tokenizer / sampler / chat-template layer, `verify_rlxp` — and JS
  gains `directionalNthGrad` and `check`'s `budget` switch. Trainer checkpoints
  share one GGUF layout across both.

### Changed

- **`supported_for_target` now matches every backend's real `SUPPORTED_OPS`**
  on the kinds that gate fusion (new `FUSION_GATING_KINDS`), and
  `FusionOptions::for_target` unfuses regions for CUDA/ROCm as the runtime does.
  The hand-copied lists had drifted — CUDA/ROCm lacked `FusedSwiGLU`, Metal
  `FusedAttentionBlock`, all of them `While`/`Fft`/`WelchPeaks`/
  `FusedConvBiasAct` — so `fuse(g, target)`, benchmarks, and an `rlx-metal`
  compile without an explicit claim set fused differently from the backend.
  Runtime compiles already passed the real claims and are unaffected.

- **`ExpertPager::prefetch` is no longer a demand read**: it does not move
  `hits`/`misses` (it used to inflate `hit_rate`), only refreshes an
  already-resident expert, and tracks what it admitted as speculative.

- **XDNA tiled GEMM cuts each activation tile once per K-block** instead of
  once per (N-block, K-block) in the resident-weight path; same accumulation
  order, so bit-identical by construction. Compiles; not yet run on the NPU.

- **`Device::Xdna` comes after `Cpu` in automatic device selection.** A
  Phoenix NPU measured 1.9–2.7× slower than same-precision CPU bf16 on
  transformer models, so `fastest_device()` no longer picks it; select it
  explicitly.
- **A `Session` on an unavailable XDNA / eGPU device says why.** The panic now
  carries the backend's diagnostic (missing env var, path, library, toolchain
  error) instead of "no such device/driver was found".

- **Both bindings now share one implementation, so a few behaviours line up.**
  pyrlx `CompiledGraph.run` refuses non-f32 outputs (as rlx-js always did)
  instead of reinterpreting their bytes — use `run_typed` or `cast`. A Rust
  panic inside a pyrlx call is a `RuntimeError`, not pyo3's `PanicException`
  (a `BaseException` that `except Exception` misses). Unknown optimizer and
  sampling options are an error in rlx-js too. pyrlx `dequant_gguf` handles
  every GGUF type (F16 / BF16 were refused), and `GgufFile.metadata()` returns
  arrays as lists rather than a debug string. JS `Gguf.info().dtype` reads
  `"Q4_K"` (the GGML name) rather than `"Q4K"`. Graph builders gained the
  other binding's spellings: JS `eq` / `ne` / `lt` / `le` / `gt` / `ge`,
  Python `ifft`, `le_`, `node_count`, `name`, `fused_residual_layer_norm`; JS
  `Runner` gained `runOn` / `runTry` / `runChain`, JS `Gguf` `tensorBytes` /
  `architecture`.

- **rlx-cuda: `RLX_CUDA_PARITY=1` now also keeps cuDNN's forward conv on the
  GEMM algorithms.** cuDNN's heuristic picks WINOGRAD for most 3×3 convs. Even
  in strict FP32 math (`CUDNN_FMA_MATH`), its transform loses about 10× in
  accuracy. CLEF's VQGAN `conv_in` weight-gradient was 1.05e-4 off PyTorch with
  WINOGRAD and 1.06e-5 with IMPLICIT_PRECOMP_GEMM. The default (non-parity) path
  is unchanged.

- **Rust builds can share an sccache cache; two settings decide whether it ever
  hits.** `docs/development.md` gains a *Build cache* section. `incremental =
  false` is required — sccache forwards `-C incremental` compilations to rustc
  uncached — and `CARGO_TARGET_DIR` must not be exported, because the
  environment variable is part of sccache's Rust cache key: varying it misses
  every time, while the identical directory change passed as `--target-dir`
  hits. Measured 2026-10-16 on a box that had run sccache for months through
  `CARGO_TARGET_DIR`-exporting scripts and sat at a 0.00% Rust hit rate against
  83% for C/C++. `rlx-onnx-import`'s `tests/codegen_compile.rs` now passes
  `--target-dir` rather than setting the variable, keeping its isolated target
  dir without defeating the cache. The MLX C++/CUDA build was already cached
  (`RLX_MLX_CCACHE`); only Rust was uncovered.

### Fixed

- **`FuseSwiGLUDualMatmul` on batched weights**: a `silu(x·Wg)·(x·Wu)`
  with `[B, K, N]` weights (an expert per batch row) was concatenated as if
  2-D, declaring a `[B, 2N]` weight the operands give as `[B, K, 2N]`; the
  pass now fuses `[K, N]` weights only (test
  `only_two_dimensional_weights_fuse`).
- **Backend calibration could be re-measured mid-process.** rlx-metal, rlx-cuda and rlx-mlx
  saved their calibration with `fs::write` (truncate, then write) and re-read the file on every
  cost-model query. A concurrent reader that caught the file empty or half-written measured
  again, with timing-dependent numbers, so two device queries in one process could disagree.
  The calibration is now loaded once per process (`OnceLock`) and saved write-then-rename.

- **ONNX `ArgMax` / `ArgMin` / `MaxPool` indices imported as invalid IR.**
  The importer declared `Op::ArgMax` itself `i64` (ONNX's index type), but the
  IR op returns f32-encoded indices, so the verifier rejected any model with
  an argmax and it could not compile in a debug build (the conformance
  `gqa_layernorm` fixture). It now emits the f32 op, widening a
  half-precision input first, and casts the indices to i64
  (`tests/arg_reduce_indices.rs`).

- **A param passed to `run` was ignored**, so a training loop that fed its
  updated weights through `run` (as the README's example did) kept computing
  with the old ones. `run` / `try_run` now upload a param name with
  `set_param`.

- **An invalid input graph was blamed on the first pass that changed it.**
  Debug builds re-verify after each pass; an error the input already had read
  `IR verifier failed at after pass `select_peaks_only_outputs``. It now reads
  `the input graph (already invalid before any pass …)`; inputs a lowering
  pass legitimately repairs (`LowerIf`) still compile.

- **TPU `Op::Rope` rotated only the first head, and ignored `n_rot` and
  `style`.** The lowering treated the last axis as one head, so on a packed
  `[B, S, H·D]` tensor it emitted a shape-inconsistent module; partial rotary
  and GPT-J pairing were silently NeoX full-width. Rewritten to the CPU
  semantics (per head, first `n_rot` dims, NeoX or GPT-J pairing, per-position
  or per-token tables); `RopeBackward` shares it with `sin` negated, so GPT-J
  gradients now lower instead of asserting.

- **TPU `RmsNorm` dropped `beta`** — the defect fixed on the other backends
  earlier; TPU had been missed.

- **TPU `Attention` with `MaskKind::Custom` added the mask to the scores**
  instead of treating it as the `[B, S_k]` keep-mask the IR defines (`< 0.5`
  ignored). A 0/1 padding mask masked nothing.

- **Fusions absorbed producers that were also graph outputs.**
  `FuseMatMulBiasAct`, `FuseConvBiasAct`/`FuseConvAffineAct` and
  `FuseMatMulResidual` checked one consumer with `users().len() == 1` rather
  than `has_single_use`, so an exported matmul returned `x·w + b` on CPU.

- **CPU returned zeros for an exported `Narrow` feeding `Attention` or
  `Rope`.** The thunk rewrites that read the parent buffer with a row stride
  turn the Narrow into a Nop when it has one reader, and did not count the
  caller reading a graph output.

- **`jvp_hvp_third_regression` asserted the bug it was named after.** It
  checked that `jvp(hvp(f))` returns zero rather than the third derivative —
  the behaviour the `tangent_<name>_2` collision fix removed — and kept
  passing because it fed only `tangent_x`, so the outer tangent read the
  arena's zeros. Under `RLX_POISON_ARENA` it read NaN. It now feeds both
  tangents and requires `6.0`, matching `nth_order_grad`.

- **`lstm_autodiff` never wrote graph constants into its arena** (it drives
  thunks directly, bypassing the runtime that does). It passed only while
  every constant in the unfused LSTM is zero.

- **`ExpertPager::prefetch` counted as cache hits and misses**, inflating
  `PagerStats::hit_rate` — the number that decides whether paging is viable.

- **CPU conv weight gradients were not deterministic.** The parallel
  `Conv2dBackwardWeight` path added each chunk's partial gradient under a
  lock, in finishing order, so the f32 sum could change between two runs of
  the same graph. Chunks are now added in chunk order: bit-identical runs for a
  given thread count. Test `rlx-runtime/tests/conv_dw_deterministic.rs` (10
  runs, plus an f64 reference); the old reduction fails it on the second run.
  The conv batch loops also take a work-aware grain (`pool::grain_for`).
- **`RLX_WORKERS` could be silently ignored.** The global rayon pool is built
  at the first kernel; if the host program used rayon before that, rayon's
  default pool stood and the setting had no effect. rlx-cpu now says so when
  `RLX_WORKERS` is set and cannot apply.
- **`process_rss_bytes()` returned 0 on macOS.** It called the legacy
  `TASK_BASIC_INFO` with a `repr(C)` mirror of a struct the SDK declares under
  `#pragma pack(4)`, and read a resident size of 0. So `memory_headroom_bytes`
  was always the whole budget and `would_exceed_soft_budget` never fired on
  macOS. It now reads `MACH_TASK_BASIC_INFO` into a naturals buffer. Test
  `process_rss_is_the_real_resident_size` (touching 64 MiB must show up).
- **Memory plan: a folded transpose's input died too early.** A 2-D
  transpose read only by a matmul is folded into the GEMM's flag (never
  executed; the matmul reads its input), but liveness ended that input's
  life at the transpose. With two `transpose(x) @ w` branches whose `x` are
  computed, the second branch's tensor took the first's freed slot before
  the first matmul read it — off by hundreds, silently (an ISTFT head,
  `re @ cos + im @ sin`). The input now lives until the matmul. Test
  `rlx-runtime/tests/folded_transpose_liveness.rs`.
- **wgpu: a skipped static weight pack dropped deferred host writes, so
  every run after the first was wrong.** After a host step, the run loop
  flushes the host mirror only if the *next* step is a GPU step. It decided
  on the raw schedule, so when the next step was a static-once host `Concat`
  (a fused QKV/adaLN weight, skipped from run 2 on), it kept deferring. It
  then landed on a GPU matmul whose pass cleared the mirror with the writes
  still unflushed, and a later host op read its input slot's previous
  tenant from the device. Run 1 was exact, later runs were wrong (cosine
  −0.12 on `rlx-montageflow`'s final layer, discrete Vulkan with
  matmul-bias fusion off). The lookahead now skips the steps the loop is
  about to skip. The test `rlx-wgpu/tests/static_pack_skip_rerun.rs`
  reproduces it on Metal via `RLX_WGPU_FORCE_HOST=1` and fails with the fix
  reverted. Separately, two of the five static-once arming sites set the
  flag directly, so `RLX_STATIC_WEIGHT_PACK=0` did not disable the skip
  for a graph that ends on a host step. All five now go through
  `arm_static_once`.
- **wgpu: the wide-matmul GELU epilogue used the tanh form for exact GELU.**
  `matmul_wide`, `matmul_wide_vec4`, `matmul_wide_nv`, `matmul_f16w` and
  the four `matmul_coop_f16_vulkan*` kernels mapped activation code 9
  (`GeluErf`) and 11 (`GeluApprox`) both to the tanh approximation. That is
  about 6e-5 per transformer block wherever the fused matmul+bias+GELU
  kernel was picked: discrete Vulkan, rank-3 inputs. Code 9 now evaluates
  erf, the same way `matmul.wgsl` does. Test
  `rlx-wgpu/tests/fused_matmul_bias_parity.rs` drives pre-activations
  across [-4, 4]. It fails at 1.2e-4 with `matmul_wide_vec4` reverted.
- **ONNX import: a negative `Pad` crops.** ONNX defines a negative pad as
  a crop; the importer stubbed such a `Pad` with zeros. VITS' relative
  attention pads its position embeddings by `len − 5`, below zero for a
  sequence under five, so a short Piper input ran every attention layer on
  zero embeddings, silently. Test `rlx-onnx-import/tests/pad_negative.rs`.
- **`shared_weights` footprint test read a `u64` off its alignment** (an
  `i32` buffer from `task_info`): a debug build aborted the whole
  `rlx-metal` test binary. Read unaligned.
- **A CPU `Scan` body gathered row 0 at any integer index.** The CPU
  compiled scan bodies (forward, backward, and the recompute between
  checkpoints) without the preparation every other graph gets
  (`prepare_graph_for_thunks`): an I32 index was read as f32 bits, about
  3e-45, so a gather in a body took its first row, and a claimed fused op
  in a body would have run as a no-op. TEN-VAD's exported pitch track (its
  Viterbi backtrack gathers by state) found it; regression test
  `rlx-runtime/tests/scan_body_integer_index.rs`.
- **`.rlxmodel`: an attribute naming a derived symbol pointed at another.**
  The reader numbers each derived dim anew, past every symbol the header
  declares; an attribute (`Narrow.len: frames*8-7`) kept the number
  `generalize` gave it. With the header declaring a symbol the graph does
  not use (a pipeline's `codebooks`), the slice read its length from the
  next derived dim and the specialized graph failed its shape check. The
  writer now names each such attribute by the reader's number.
- **`inline_into` / autodiff `merge_subgraph` declared a param twice.** An
  unbound param or input re-emitted as a new leaf even when the target already
  had one of that name — a scan backward merging its body per step got one
  `Param` per step, and binding by name reaches only one of them. Both are now
  `Graph::splice`, which shares the existing leaf (and errors on a kind or
  shape clash instead).
- **XDNA detection failed silently (GitHub #5).** Each missing
  `RLX_XDNA_*` variable or nonexistent path is now named, the toolchain probe
  reports the compiler's error, and an absent NPU lists what `/sys/class/accel`
  held. `RLX_XDNA_XCLBIN` / `RLX_XDNA_INSTS` are registered (and documented)
  env vars. The device node is reported at `/dev/accel/accelN` (was
  `/dev/accelN`), and the XRT search covers `/usr/lib` (Arch) and `/usr/lib64`.
- **XDNA matmul kernel build broke on mlir-aie v1.4.4.** The vendor kernel moved
  to `aie_kernels/linalg/mm.cc` and dropped its `zero_*` functions; rlx now
  reads either layout and supplies `zero_<type>` when absent.

- **`ExpandSharedKvHeads` rebuilt every graph.** The pass copied each
  graph node by node even when no attention had shared K/V heads. It now
  returns the graph as is.
- **(rlx-models) GPU-resident K/V lost rows on CUDA; batch > 1 Whisper
  broken everywhere.** A decode step that returns its caches as
  `concat(past, new)` has its new row last; the handle feed copied only the
  rows before it, so every new row inside a bucket was dropped
  (`WhisperRunner` on CUDA looped "the capital of the capital of …").
  `rlx-models-core::gpu_kv` now keeps the handles by a per-model
  `KvLayout`. `InPlace` copies the leading rows, as before. `AppendLast`
  moves only each sequence's new row (`feed_kv_row`) and reads a bucket's
  last row from the output. `Detect`, the default, reads the layout off
  the step's shapes when it compiles, so callers that weren't changed
  (Gemma, Qwen3-TTS, LocateAnything) get the right one. Whisper declares
  `AppendLast`. Batch > 1 was also wrong on every device, host path
  included:
  - the host helpers padded and compacted `[batch, past, kv]` caches as
    one block (`pad_kv_sequences` pads each sequence; the compaction is
    per sequence);
  - the cross K/V install asserted at batch > 1;
  - Whisper broadcast one position row over a batch that gathers one per
    sequence;
  - Whisper fed one mask row for `[batch, upper + 1]`.

  Two regions decoded together now transcribe as each does alone, on CPU
  and on CUDA, with resident K/V on and off.

- **CPU: I32 / U32 / I64 tensors meant different numbers to different
  kernels.** Casts, scans, the fixed-point FFT and custom ops read integers
  as integers; arithmetic, compare, reduce and gather indices computed in
  f32 over the same bytes; and `run()` wrote I32 as f32 bits and I64 as half
  an 8-byte slot. So a cast after an integer multiply returned `1065353216`
  for `1`, token ids fed through `run_typed` all selected row 0, and any I64
  input through `run()` was garbage. Integers are now stored as integers
  throughout. A CPU pass (`rlx_cpu::int_storage`) casts around each
  f32-computing op that touches one (gather: only its indices; reduce over
  I64: through f64), and `run()` writes and reads I32 / U32 / I64 as
  integers. `tests/cpu_integer_storage.rs` checks 8 ops × 3 dtypes × both
  feeds.
- **Metal: a view of a parameter read from offset `usize::MAX` once the
  weights were split out.** With large weights packed into their own buffer
  (`RLX_METAL_WEIGHT_SPLIT_MIN`), the activation arena is planned without
  params, and a pure view rooted at one (a `Narrow` at 0, a `Reshape`, a
  same-dtype `Cast`) got no offset at all. A stateful decode step's mask
  (`iota[0..cap] <= position`) read back as all ones, so the exported Qwen3
  package attended to every cache row and diverged after the first token.
  Views now take their root's offset in the arena. `tests/metal_view_offsets.rs`
  fails without the fix; the Qwen3 package now decodes the same 24 tokens as
  `Qwen3Runner` on Metal.
- **rlx-cpu: `Concat` of 1- and 2-byte element types.** The CPU `Concat`
  thunk moved 4-byte words whatever the dtype, so a bf16 / f16 / u8 / i8 /
  bool concat came out with half its elements missing and rows interleaved
  (bf16 `[a | b]` read back as `[a1, a3, a5, b1, …, 0, 0]`), silently. It
  surfaced through `FuseSharedInputMatMul`, which concatenates the weights of
  matmuls sharing an input: a model holding bf16 weights (gate + up of an MLP)
  computed garbage on CPU while every single-matmul test passed. Half types now
  go through `Thunk::ConcatHalf` (packed params and widened activations are
  different layouts, so inputs are widened to f32, assembled, and a packed
  output rounded back); other 1- and 2-byte types through a raw byte copy
  (`Thunk::ConcatBytes`). Test: `rlx-runtime/tests/concat_narrow_dtypes.rs`.

- **rlx-coreml: training graphs with bias gradients segfaulted (Apple BNNS
  defect, now avoided).** CoreML's CPU engine mis-executes
  `transpose → reduce → transpose`. It reads out of bounds inside
  `BNNSGraphContextExecute_v2` (SIGSEGV/SIGBUS), or at some shapes silently
  returns wrong sums: `[3,32,32]` summed over two axes was off by 98. This
  reproduces with coremltools and no rlx code (macOS 26.6.2), on every
  compute-unit setting. rlx-autodiff's sum-to-shape emits exactly this chain for
  every bias gradient. The new `hybrid::hoist_transpose_out_of_reduce` pass
  rewrites `reduce(transpose(x, p), A)` as `reduce(x, p[A])` plus a reshape, or a
  transpose after the reduce, so the pattern can't form. Regression test:
  `rlx-coreml/tests/coreml_transpose_reduce.rs`. Its bias-gradient cases crash
  without the pass, and two breakages of the rewrite are each caught.

- **rlx-coreml: a MIL segment could declare an input it never used.** The hybrid
  plan can hand a segment an `Input` whose readers all live in other segments,
  and CoreML refuses to load that model ("Error in declaring input d_output with
  error -1"). `build_mil_subgraph` now leaves such an input out, unless it's the
  segment's only input: a model with no inputs makes CoreML's E5 runtime throw
  and silently fall back to another engine. With both fixes, CLEF's three
  training stages run on CoreML at torch-gradient parity (≤2e-5).

- **rlx-vulkan: `StopGradient` outputs were never written.** The lowering treated
  `StopGradient` as a no-op view, like `Reshape`. But the memory planner
  (`rlx_compile::memory::pure_view_offset`) aliases only `Reshape`, same-dtype
  `Cast` and axis-0 `Narrow`, so `StopGradient` owns a slot, and every reader got
  stale bytes. It now copies when the slots differ. Found porting CLEF: the VQ
  codebook loss `‖sg(z) − q‖²` read as exactly 0, and real-weight T5 overflowed
  to inf, because the softmax max-shift `x − sg(max x)` subtracted 0. rlx-cuda and
  rlx-rocm alias `StopGradient` in their own arenas and were not affected.
  Regression test: `rlx-vulkan/tests/stop_gradient_slot.rs`, which fails on the
  old lowering (reads 0.0 for 9.0).

- **rlx-wgpu: LayerNorm backward was wrong on most transformer shapes.** The
  kernel had two defects. First, the workgroup reduction is called back-to-back
  with no barrier after the read, so lane 0's next write raced the other lanes'
  read: a 64×64 input failed while 65×64 passed. Second, `dgamma` only covered
  the first 64 columns, so any hidden size above 64 got wrong gamma gradients
  for column 64 onward. `ada_layer_norm_backward` had the same reduction race.
  Found porting CLEF, whose Stage-I/II gradients were off by up to 8 % on wgpu
  only. Regression test: `tests/layer_norm_backward_parity.rs`.
- **rlx-wgpu: `StopGradient` is now lowered**, as the forward identity it is
  (Metal already did this). `supported_ops` advertised the op, so any graph
  containing one was accepted and then panicked in `lower.rs`.
- **`rlx-js`: a resident trainer loading a checkpoint aborted the host.**
  `Trainer.load` size-checked each parameter against the host path's weight
  copy, which the resident (on-device AdamW) path leaves empty, so the index
  panicked outside any guard; and the weights it would have restored were
  rebuilt from that same empty copy. A host checkpoint now resumes on the
  resident path (`a_checkpoint_resumes_on_the_resident_path`, Metal).
- **`pyrlx`: `rlx.load_rlxp` (as documented) and the other RLXP / weight
  loaders raised `AttributeError`** — compiled into the extension but never
  re-exported from the package.

### Performance

- **Graph compile no longer dominates small training runs** (rlx-compile, rlx-runtime, rlx-cpu).
  A CPU profile of exgb's 16-cell test sweep (small EEG decoders, a fresh compile per fold)
  had 24.6 % of busy time in graph compilation. Three costs, none of which touches
  arithmetic (the sweep's results are bit-identical before and after):
  - `CpuExecutable::restore_arena_baseline` rebuilt the persistent byte ranges (a scan of
    every node plus a sort) at the start of every run, so every training step. They follow
    from the compiled layout, so they are now computed once and recomputed only when
    `arena_handles` changes. Test `arena_keep_ranges` fails if that invalidation is dropped.
  - `is_elidable_matmul_transpose` scanned the whole graph for each 2-D transpose, which is
    quadratic in a training graph with one per weight. The planner and the CPU thunk compiler
    now ask a `ReaderIndex` built in one pass. `the_reader_index_agrees_with_the_scan`
    compares it with the scan on every node.
  - The planner's placement loop sorted the time-overlapping buffers for every buffer it
    placed. `placed` is now kept sorted by offset, giving the same sequence without the sort.
    `plan_sweep_matches_candidate_scan_with_long_lived_branches` and
    `a_gap_filled_below_an_earlier_buffer_is_respected` check every offset against the cubic
    candidate scan, planned with reuse on. The earlier test planned with every buffer pinned
    to the end, so nothing was reused and no ordering bug could show.

  - `FusionReport::scan`, built on every compile even with no report requested, called
    `Graph::users` (a whole-graph scan) for every matmul. It now uses a users index built in
    one pass. `the_users_index_matches_graph_users` checks it node for node.

  After: compile 8.2 % of busy time (the planner sorts, the transpose check and the
  per-step range rebuild are gone from the profile). Wall time on the shared, loaded
  machine was too noisy to state a speedup.
- **Faster conv im2col gathers, bit-identical** (rlx-cpu). The forward gather (`im2col`,
  `im2col_par`, now sharing one `fill_col_row`) zeroes each output row's padding and copies
  its in-bounds range, one slice at stride 1, instead of a bounds test per element: 2–20×
  faster on EEGNet/ShallowConvNet shapes, slower only on toy rows a few values wide. The
  weight-gradient gather (`im2col_rows_layout`) does the same for kernels 16 taps wide or
  more (2–4.5× faster on temporal convs) and keeps its original per-element loop for
  narrower ones. Routing every kernel through the copy was 2–3× slower on 1-wide kernels
  (EEGNet's spatial conv, 1×1), where each copy is a library call for one value. That was a
  first version, caught when the end-to-end profile showed the time moving into `memmove`.
  The buffers are identical, so every GEMM sees the same input. Tests compare both gathers
  with the original loops bit for bit over more than 500 and 600 stride, padding, dilation
  and kernel shapes; an off-by-one in either range fails them. `gather_timing` /
  `forward_gather_timing` (ignored) time them against the originals. 1-wide kernels (EEGNet's
  spatial conv, 1×1) then got their own branch in the weight-gradient gather: the
  kernel rows' input offsets are worked out once per output row and the single input
  column is checked once per position. About 2× faster on the spatial conv, 0.5–0.8× the old
  time on 1×1 in 5 of 6 timings. Same buffer: `rows_layout_matches_the_original_loop` covers
  it, and an off-by-one in its column fails it.
- **`col2im` (conv input gradient) 6–9× faster, bit-identical** (rlx-cpu). The scatter-add
  ran a bounds test per position; the in-bounds outputs of each row are one range that
  depends on the kernel column alone (`wo_range`, now shared with the forward gather). At
  stride 1 it adds them as one slice. Every input element receives the same additions in
  the same order. `col2im_matches_the_per_element_loop` compares it with the original over
  more than 600 shapes, accumulating into a non-zero gradient, and an off-by-one in the
  slice offset fails it. Timed on EEGNet/ShallowConvNet shapes in `forward_gather_timing`:
  0.11–0.18× the old time, and 0.5–0.66× on rows a few values wide.
- **Reductions over a non-last axis 12–14× faster, bit-identical** (rlx-cpu). `Thunk::Reduce`
  reduced each output on its own, reading its strip with stride `inner`, which is a cache
  miss per element when the reduced axis isn't last: batch means, gradients summed over the
  batch. The kernel is now `reduce_into`: it accumulates whole contiguous rows of `inner`
  outputs at a time, each output still seeing the same operations in the same order (f64
  for Sum/Mean). Last-axis reductions (`inner == 1`) keep a scalar loop and their speed
  (0.97–1.02×). `reduce_into_matches_the_strided_loop` compares it with the original kernel
  bit for bit for all five ops, serial and parallel, over shapes whose chunks cross outer
  boundaries, with NaN, ±0 and ±1e17 against 1.0 (so a sum in any other order lands on
  different bits, even in f64). Reversing the accumulation order fails it. Timed in
  `reduce_timing` (ignored).
- **Conv bias broadcasts take the per-channel fill** (rlx-cpu). `Thunk::Transpose`'s
  per-channel fast path (`[C]` to `[N,C,H,W]`) required a zero stride on every axis but the
  channel's. A bias reshaped `[1,C,1,1]` and expanded to `[N,C,1,W]` carries stride 1 on its
  size-1 height axis, so every EEG temporal conv's bias fell through to the per-element
  index walk. A per-thunk profile (`RLX_PROFILE_THUNKS`) of EEGNet + ShallowConvNet put
  Transpose at 22.9 % of thunk time, and a shape trace put 88 % of that on this one
  pattern. Size-1 axes now count as unused, whatever their stride. Same values (a copy).
  Test `expand_channel_broadcast` checks the broadcast against the definition, bit for bit,
  over the traced shapes; corrupting the fast path's channel index fails it.
- **Binary ops: parallel same-shape NEON, and a held-scalar path** (rlx-cpu,
  `Thunk::BinaryFull`), bit-identical. A shape trace of EEGNet + ShallowConvNet found the
  same-shape NEON Add/Mul loop running on one thread for 2.4 M-element tensors (538 µs a
  call), and `x op scalar` going through the per-element broadcast index walk at 0.6–0.9
  elements/ns. The NEON loop now splits a large tensor across threads, each element still
  getting one IEEE add or multiply. A 1-element operand on either side is held in a plain
  loop. The three f32 loops share one op table (`f32_op!`) instead of three copies of the
  14-op match. Test `binary_elementwise_paths` checks Add/Sub/Mul/Div/Max/Min, same-shape
  and scalar on either side, at 7 / 1000 / 2.4 M elements, against per-element arithmetic
  bit for bit. A wrong chunk offset or scalar operand fails it.

- **Metal: BF16-weight decode matmuls read the weight in-kernel.** A
  `Sgemm { b_bf16 }` with M ≤ 4 used to unpack the whole bf16 weight into an
  f32 scratch on every call and run the f32 GEMM — 5 bytes moved per weight,
  3× slower than storing f32. New `gemv_bf16w_splitk8` (8 columns per thread,
  16-byte loads, 16-way K split; `_n64` variant with a 64-column tile and
  64-way split for narrow N), `gemv_bf16w_splitk` (even N) and
  `sgemm_bf16w_small_m` (M 2–4). Fish S2-Pro decode step on an M4 Pro:
  f32 312 ms → unpacked bf16 961 ms → in-kernel bf16 ~110 ms, at 9 GB instead
  of 25. Bit-exact widening; `RLX_METAL_BF16W_DISABLE=1` restores the unpack
  path. Pinned by rlx-models' `half_weights_match_f32_math_at_decode_shapes`.
- **Metal: contiguous mid-axis concat.** With one outer row (a KV cache plus
  its new rows) each segment is a single block; it is now copied as `float4`s
  (`copy_f32x4`) instead of one element per thread with three div/mods. Fish
  S2-Pro at 4096 cached rows, where the 72 per-step concats took 76 ms of a
  265 ms step: 284 → 217 ms per step (best of 4 batches; shared machine).
- **`samplers::TopP`** sorts only a growing top prefix (64, 256, 1024, …) of
  the probabilities until it reaches `p`, instead of every finite one; the
  prefix is in exactly the full sort's order, so the kept set is identical
  (`top_p_prefix_search_matches_a_full_sort`). `TopP`, `TypicalP` and
  `sorted_desc` sort with `total_cmp` + index ties (same order as before, no
  panic on NaN). Through `rlx_qwen3::sample_token_at` at temperature 0.7 /
  top-p 0.95 on a 151,936-entry row: 2.7 ms → 1.0 ms per token (Apple
  M-series, release).

## [0.2.18] — 2026-10-15

### Added

- **Emulated number formats (`CompileOptions::number_format`).** A graph can
  run as `f64`, double-single `f64x2`, or any binary float format from `f16`
  down to 4-bit minifloats (`bf16`, `f8e4m3`, `f6e2m3`, `f4e2m1`, custom
  `fNeXmY`), parsed from text such as `"f4e2m1@b16"`. For the narrow formats,
  `rlx_compile::emulate_float_format` inserts the new `Op::RoundToFormat` after
  the weights and after every op that produces a float tensor, with no scale,
  one per tensor, or one per block (`FormatScaling`; MX 32-element blocks by
  default at 8 bits and below), so every value a consumer reads is
  representable in the format while arithmetic inside an op still accumulates
  in `f32`. `FloatFormat::round` (ties to even, saturating, power-of-two
  scales) is the one definition: the op is native on CPU, Metal, MLX, CUDA,
  ROCm, wgpu and Vulkan (Core ML runs it on the host), and
  `rlx-runtime/tests/number_format.rs` checks every backend bit-identical to
  the CPU kernel.

- **sccache for the MLX build (`rlx-mlx-sys`).** MLX's C++ and CUDA sources
  now compile through sccache when it is installed, else ccache, on macOS,
  iOS-family targets and Linux; `RLX_MLX_CCACHE=sccache|ccache|<path>|off`
  chooses. Linux builds used no compiler cache at all before, although the
  README said MLX picked up ccache by itself — so every new `OUT_DIR`
  recompiled the CUDA kernels from scratch (~1 h).

- **Checkpoint files and adapters for flow weights (`rlx-flow`).** With the new
  default-off `formats` feature, `formats::SafetensorsSource` (one file or a
  directory of shards, memory-mapped; F16/BF16/F64/integer read as f32) and
  `formats::GgufSource` (dequantized, or `.packed(true)` for `DequantMatMul`)
  are `WeightSource`s, and `formats::open(path)` picks one by path. Without the
  feature, `WeightSourceExt` (in the prelude) adds `CloningSource` (a
  non-consuming source over any `TensorLookup`), `with_prefix` / `renamed`,
  `alias` (tied weights), `map_tensors`, `or` (fallback source) and `tracked`
  (served / missing / unused keys after a build), all passing packed weights
  through. `WeightSource::keys` lists a source's tensors, and a missing key's
  error names the closest ones (`missing_weight`). `tests/weight_formats.rs`
  builds one projection from a GGUF file as F32 and packed and checks both
  against a reference.

- **`rlx_gguf::quant_scheme` (feature `ir`)** — the GGML type → `QuantScheme`
  table, now in one place; `rlx_cpu::quant_scheme_for_ggml` forwards to it.
  Without `ir`, rlx-gguf still has no rlx dependencies.

- **`rlx-cpu` feature `int8-gemv` (default off):** int8 decode GEMVs (`m == 1`)
  for Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 and Q4_1 on x86-64 AVX2, after llama.cpp's
  kernels; the activation is quantized to int8 (Q8_K or 32-blocks), so results
  differ from the f32 path by that rounding. `RLX_Q4K_EXACT_GEMV` keeps Q4_K on
  the exact f32-activation arm.

- **`rlx-cpu` feature `portable-gemm` (default off):** without a BLAS, `m ≥ 8`
  GEMMs go to the `gemm` crate's blocked kernels instead of the loop nest.

- **Resident KV cache on CPU.** `rlx-runtime`'s CPU backend implements the
  resident decode handles (`bind_gpu_handle`, `register_kv_row_feed`,
  `feed_kv_row`, `read_output_row`); a bound KV handle stays in place in its
  arena slot and only the new row is written per token. Guard:
  `rlx-runtime/tests/cpu_resident_kv.rs`.

- **Opt-in runtime flags:** `RLX_PACKED_LM_HEAD` (the flow's LM head reads the
  packed `output.weight`, or the packed embedding table when tied, instead of an
  F32 copy) and `RLX_CPU_EXEC_IN_POOL` (run each CPU schedule on a thread-pool
  worker, so parallel kernels join by spinning instead of parking the caller).

- **`rlx-ps5`** (standalone, not published): jailbroken-PS5 target bring-up —
  `just ps5 setup|build|send|run`, a probe payload, a distributed-node payload,
  and a platform seam for model payloads.

- **Ternary (BitNet b1.58) weights, including quantization-aware training.**
  `rlx_ir::ternary` is the one definition training and export share: absmean
  weights (`γ = mean|W|`, codes `clamp(round(W/γ), −1, 1)`, per tensor by
  default, `TernaryScale::PerChannel` optional) and int8 per-token absmax
  activations. `rlx_compile::inject_bitnet_qat` rewrites every `MatMul` / `Conv`
  whose weight traces to a leaf into `w + stop_gradient(ternary(w) − w)` (and
  the activation likewise) — the BitNet straight-through estimator, built from
  existing primitives, so it trains on any backend; `keep_full_precision`
  leaves embeddings / LM heads in f32. `rlx-runtime/tests/bitnet_qat.rs` checks
  the CPU forward loss and `dW` against the host definition. The
  `bitnet_ternary_fpga` example shows why QAT matters: ternary PTQ alone drops
  test accuracy from 81.8 % (f32) to 55.9 %, QAT recovers 82.2 %.

- **`ExportQuantMode::Ternary` for FPGA export** (`quant="ternary"` / `"bitnet"`).
  f32 `Conv` / `MatMul` weights are ternarized with the same `TernaryConfig`
  (`LegalizeOptions::ternary`) and packed 2 bits per weight, and the export
  always takes the multiplier-free add/sub/skip MAC — on every tune preset, since
  it is exact. Walkthrough: [`docs/bitnet-fpga.md`](docs/bitnet-fpga.md).

- **openXC7 boards: Xilinx 7-series bitstreams and a UART host link.**
  `HwTarget::xc7_board` / `--hw xc7:<board>` (Arty A7-35/100, Basys 3,
  Nexys A7-100) emits a `board_top` with power-on reset and a 115200-baud host
  link (`0xA5` + input bytes in, `pred` and optionally every output byte back),
  the board's real pin constraints, a `synth.sh` that runs the full open flow —
  `synth_xilinx` → `nextpnr-xilinx` → `fasm2frames` → `xc7frames2bit` → `out.bit`,
  with the part's chipdb built once and cached — plus `program.sh`
  (`openFPGALoader`) and `run_uart.py`. The script finds the toolchain in the
  snap, a source build, or the `regymm/openxc7` container. Built in that
  container for Arty A7-35: the example MLP routes at 164.9 MHz and a padded
  ternary CNN at 110.4 MHz against the 100 MHz board clock
  (`tests/openxc7_docker.rs`, `--ignored`). Not yet run on a board.

- **Zero padding in FPGA conv layers.** Both conv kernels (scalar and
  output-channel-parallel) accept any `pad_h` / `pad_w` and stride; out-of-bounds
  taps skip the accumulate, as the reference does. Previously any padding
  panicked in codegen, so most real CNNs could not export.

- **The emitted RTL is now simulated, not just inspected.**
  `rlx-fpga/tests/sim_parity.rs` runs the SystemVerilog under Icarus Verilog and
  compares every logit and the argmax with `rlx_fpga::reference`: dense and
  padded/strided conv models × 2/4/8-bit weights × every tune preset, plus
  TinyConv-MNIST. `tests/xc7_uart.rs` drives the UART link bit by bit, on the RTL
  and on the `synth_xilinx` netlist against Yosys's Xilinx cell models. Both
  skip when the tools are absent. Every RTL bug under **Fixed** was found by
  these on their first run.

- **`just check-rustdoc` — a rustdoc gate, wired into `just ci`.** rustdoc was in
  no repo gate, and `cargo doc --workspace` is the wrong instrument for one: it
  resolves the whole workspace under a single unified (default) feature set, so
  an intra-doc link into a feature-gated item reads as "unresolved" even though
  docs.rs resolves it perfectly. 26 of 28 workspace warnings were phantoms of
  exactly that kind, and the two real ones were buried among them.
  `scripts/check-rustdoc.py` builds one crate at a time with that crate's own
  `[package.metadata.docs.rs]` features — what docs.rs actually does — and skips
  `publish = false` crates, which docs.rs never builds.

### Changed

- **MLX submodule bumped to v0.32.4** (from v0.32.2): upstream Metal fixes
  that reach rlx-mlx — a fence deadlock, split-K quantized matmul (row count,
  float32 partials), conv bounds and integer overflow, SDPA threadgroup size on
  M1/M2, negative-stride scatter/SDPA-mask indexing. Both tracked patches in
  `rlx-mlx-sys/patches/` apply unchanged; the metallib still carries the
  target platform (iOS `82`, tvOS `83` at byte `0x0B`). rlx-mlx 196/196,
  rlx-runtime with `mlx` 988 passed.

- **Dependencies updated** to the latest versions compatible with the 1.89
  MSRV (`CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo update`; the
  workspace resolver is `2`, which is not MSRV-aware). `cargo +1.89.0 check
  --workspace --all-targets` passes.

- **Chat templates print booleans as `True` / `False`** (`rlx-text`, minijinja
  ≥ 2.24, now required), as Python's Jinja2 — and so transformers — renders
  them; a template that prints a bool used to emit `true` / `false`.

- **x86-64 one-query decode attention** runs a task per head, skips masked keys
  (dot, softmax chunk and V sum), and computes eight keys' dots per AVX register;
  bit-identical to the scalar loop it replaces
  (`decode_attention_lane_dots_are_bit_identical`,
  `rlx-cpu/tests/decode_attention_m1.rs`).

- **`MapWeights::has` reports the keys it holds** (it returned the trait
  default, `false`).

- **One decode KV-append decision for every model family.**
  `rlx_flow::blocks::kv_cache` holds it: `append_kv_row` emits the O(1)
  in-place `Op::KvAppend` when the flow bound its inputs with
  `bind_decode_inputs_with_capacity` and the cache layout can alias (nothing but
  size-1 dims before the sequence axis), and `concat` otherwise;
  `kv_spare_rows` shapes the cache from the same predicate so the declared shape
  and the emitted op cannot disagree. The Llama and Gemma decode layers can now
  take the in-place path too (previously only Qwen3) — once their flow binds
  with capacity; no in-tree flow does yet, so they still `concat`. The opt-out is
  `RLX_NO_INPLACE_KV` for all families; the old `RLX_QWEN3_NO_INPLACE_KV` is no
  longer read. A Qwen3 cache without a spare row now falls back to `concat`
  instead of returning an error.

- **FPGA export API additions that break exhaustive matches / struct
  literals:** `HwTarget::Xilinx7` gained `board: Option<Xc7Board>`,
  `ExportQuantMode` gained `Ternary`, and `LegalizeOptions` gained
  `ternary: TernaryConfig` (construct with `..Default::default()`). The
  `xilinx7:<part>` string form still parses, with no board.

- **`rlx-js` takes `quickrs-core` from crates.io instead of a sibling checkout.**
  The engine is published as of 0.16.2, so building this workspace no longer
  needs a `../quickjs-rs` directory beside it. That requirement reached further
  than one crate: a path dep makes the *whole* workspace unresolvable without the
  sibling, which is why `just test-wgpu-linux` and the emulated ISA runs both
  bind-mounted it into their containers. Both mounts are gone. To iterate on the
  engine and the binding together, add a `[patch.crates-io]` entry rather than
  reinstating the path dep.

- **`just test-js` covers the whole JS surface by default.** Its default features
  were `cpu,gguf`, but `tests/js_api.rs` carried no feature gates, so 26 tests
  failed with `undefined is not a constructor` for `rlx.Trainer`,
  `rlx.Optimizer` and `renderChat` — the crate's canonical command was red. Each
  test is now gated to the features it exercises, so a narrower set is green
  instead of broken, and the recipe's default is the crate's own default set, so
  the bare command runs all 91 rather than silently skipping 26. The sandbox test
  behind the README's "no I/O a script can reach" claim is the exception: its
  probe list follows the enabled features instead of being gated away wholesale,
  because a claim tested too narrowly is worse than no claim.

### Fixed

- **MLX's CUDA backend now builds and runs on Linux** (`rlx-mlx-sys`; checked
  on an RTX 3080 Ti, rlx-mlx 176/176 with `RLX_MLX_DEVICE=gpu`):
  - Asking for CUDA (`RLX_MLX_CUDA=1` / feature `cuda`) without cuDNN or the
    toolkit silently built the CPU-only MLX, whose tests then passed for a GPU
    run that never happened; the build now names what is missing.
  - Every `cargo:warning` of the build script went to stderr, where cargo
    does not read them, so none was ever shown.
  - `nvcc` is pinned to the probed toolkit (`CMAKE_CUDA_COMPILER`); a distro
    `/usr/bin/nvcc` beside `/usr/local/cuda` was picked first.
  - **Several CUDA toolkits on one machine:** the build picks one itself —
    `RLX_MLX_CUDA_ROOT`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda*` (newest
    first), `/opt/cuda`, Ubuntu's split `nvidia-cuda-toolkit` (via a shim
    root in `OUT_DIR`) — skipping versions MLX's `CMakeLists.txt` refuses
    (13.1), and names its choice and what it skipped. Host compiler: the
    toolkit's `bin/g++` wrapper, else the newest `g++-N` within the
    toolkit's GCC limit, unless `CUDAHOSTCXX` is set. cuDNN from
    `CUDNN_ROOT` / `CUDNN_PATH`, the toolkit, or the system dirs. MLX ≥
    0.32.3's `cusolver` is linked.
  - With NCCL installed (as in NVIDIA's CUDA images), MLX builds its NCCL
    distributed backend and `libmlx.a` failed to link (`undefined reference
    to ncclAllReduce`); the build now links the NCCL that MLX's CMake found.
  - Patch 0003: MLX's NVRTC JIT defaults to the build's toolkit headers and
    the CCCL copy MLX installs, instead of `/usr/local/cuda` — no
    `CUDA_HOME` needed at run time.
  - `rlx-mlx-sys/docker/cuda_matrix.py` checks toolkit selection and the full
    MLX CUDA compile + link on NVIDIA's CUDA 12.8 / 13.0 / 13.1 / 13.2 images
    (no GPU needed).

- **Grouped conv weight gradients with stride > 1 were wrong on MLX-CUDA**
  (max |Δ| ≈ 1 on `conv2d_backward_weight_groups_3_stride2`; MLX's CUDA
  backend mishandles a grouped conv with a dilated kernel). With the `cuda`
  feature, rlx-mlx computes that case per group with ungrouped convs; CPU and
  Metal builds are unchanged.

- **Packed weights were dropped through `&mut` weight sources.** `impl
  WeightSource for &mut T` forwarded only `take` and `has`, so through an
  adapter over a `&mut dyn WeightSource` `take_packed` answered `None` and a
  quantized model silently built as F32.

- **No-BLAS `sgemm`/`sgemv`/`dgemm` with `beta == 0` leaked NaN** from the
  output buffer (`0 · NaN`). Guard: `rlx-cpu/tests/sgemm_general_strided.rs`.

- **2- and 4-bit FPGA exports computed wrong results.** The weight-byte lane
  select was combinational while the weight ROM reads a cycle late, so each MAC
  took the crumb/nibble for the *next* index out of the *previous* byte. INT8
  ignores the lane, which is why nothing caught it; every INT4 and ternary export
  was affected.

- **Fused conv+ReLU never clamped in hardware.** The clamp compared
  `q_raw < OUT_ZP[7:0]`; a part-select is unsigned, so negative values passed
  through. The checked-in TinyConv-MNIST demo RTL (`examples/mnist_sv/`) had it
  and is regenerated.

- **Emitted SystemVerilog that strict tools reject.** Signals were used before
  their declaration (dense and conv `acc` / counters), the FP4 decode table wrote
  `32'sd-1`, `block_rom` declared `parameter string` (Yosys has no `string`
  parameter type, so no Yosys flow could read the RTL), and argmax read
  `best_idx[7:0]` past its MSB (X in simulation). State transitions written as
  enum `? :` are now `if`/`else`, which every simulator accepts.

- **nextpnr-xilinx could not place exported designs.** Yosys mapped small
  buffers to distributed RAM (`RAM64X1S`), which nextpnr-xilinx cannot pack; the
  openXC7 script synthesizes with `-nolutram`.

- **CPU `Round` rounded halves away from zero on two of its three paths.** The IR
  defines half-to-even (ONNX `Round`, `rlxsl`, the third CPU path); `x.round()`
  is now `round_ties_even()` everywhere, so `Round`-based quantizers give the
  same codes on every path.

- **A second in-flight WebGPU run panicked instead of erroring.**
  `WebModel::run_async` and `ClusterWorker::grads` hold the executor borrow
  across an await; calling either again before the first promise settled hit a
  `BorrowMutError` panic. Both take the borrow with `try_borrow_mut`, so JS gets a
  catchable "already in flight" error.

- **`-D warnings` failures on wasm and in `rlx-web` tests.** `q4k_exact_gemv` is
  only called from a `cfg(not(wasm32))` path and warned as dead code on the wasm
  build; it is gated the same way now. `webgpu_trainable_models.rs` source-includes
  the vision module and now allows the dead code it does not call. In
  `autodiff_training_backward_gpu.rs` the packed-table RoPE fixtures are gated to
  the GPU features of the tests that use them, so `cargo clippy --all-targets`
  without a GPU feature no longer fails on them.

- **The env-var registry had drifted by 15 reads.** `RLX_BLAS_ALIAS_CHECK`, seven
  Metal settings (small-m MPS routing and its trace, the share-driven weight split
  and its byte threshold, the three PTQ1_0 levers), five wgpu ones (the
  cooperative-matrix probe and its verbose mode, the feed trace, the two
  wide-GEMM pins) and `RLX_XDNA_AIECC_GET` were read by the code but absent from
  `env_registry_data.inc.rs` — so none of them appeared in
  `docs/rlx-env-vars.md` or `just env-catalog`, and a user had no way to learn
  they existed.

  `just check-rlx-env-vars` returns at its first failing stage, so this failure
  had been masking the next two: the shim bypasses below, and a generated doc one
  entry out of date.

- **Two `RLX_*` reads bypassed the `rlx_ir::env` shim, so nothing in-process
  could set them.** `RLX_MODELS_DIR` (rlx-js tests) and `RLX_SHOW_VALS`
  (rlx-metal tests) went through `std::env`, which `RlxEnv` and any harness that
  A/Bs two settings inside one process cannot reach. Both read through the shim
  now.

- **Two intra-doc links pointed at the wrong item or at no item.**
  `StepOptions::broadcast_token` linked `Self::generate`, but `generate` is on
  `PipelineCoordinator`; and `rlx-torch-ckpt`'s public `pickle` module docs
  linked the private `crate::torch`. Both rendered as dead links on docs.rs, and
  both are what the new `just check-rustdoc` found once the phantom warnings were
  out of the way.

### Performance

- **Qwen3-0.6B decode on three AVX2 cores, no BLAS** (the PS5 build: rlx-cpu
  `portable-gemm` + `int8-gemv` and the opt-in flags above), average tok/s
  before → after, with a CPU-only llama.cpp on the same cores for reference:
  Q4_K_M 11.0 → 80.5 (82.5), Q4_0 → 82.5 (88), Q5_K_M → 73.5 (75), Q6_K →
  70.6 (73), Q8_0 → 61.6 (63). The int8 kernels alone run 1.5–4.4× the
  previous ones per core; decode attention ~2×.

## [0.2.17] — 2026-10-03

### Fixed

- **A checkpointed scan's backward indexed its upstream by the wrong axis.** With
  partial checkpointing a scan emits one row per *checkpoint*, so `trajectory`
  and `upstream` carry `k_total` rows while `xs` carries `length` — but
  `process_t` narrowed `upstream` at the time step. With `length = 4` and 2
  checkpoints it asked for row 3 of a 2-row tensor. It now maps the time step to
  its checkpoint index and omits the upstream term at non-checkpoint steps, whose
  cotangent arrives through `dcarry`. Second-derivative CPU-parity tests
  (`higher_order_decompose_parity`) cover it.

  Previously silent: `narrow_` inferred the output shape without checking
  `start`, so the graph was built with an out-of-range window and no complaint.
  The bounds check that surfaced this is a genuine improvement, not the bug.
- **`CompiledGraph`'s narrow-boundary conversion fired on `F64`.** The f32 entry
  points convert an `F16`/`BF16` boundary rather than reinterpreting it, but the
  predicate matched *any* non-`F32` dtype. SPD / Riemannian-geometry graphs
  declare theirs `F64`, which already round-trips through the plain f32 path, so
  they were converted and returned garbage — every `spd_host_parity` test failed
  with max|Δ| in the single digits. Scoped to `F16 | BF16`.

### Added

- **`rms_norm_mul_silu` and `l2_norm_lastdim` can emit the intermediate they
  recompute** (`silu(z)` and the per-row denominator, both optional). This lets
  those fusions absorb a producer whose value a *later* thunk still reads —
  a backward graph re-reads the recomputed forward values — where before they had
  to decline. Sound because there is still exactly one writer, unlike the two
  earlier attempts at keeping the producer alongside the fusion, which put 24 of
  54 and then 1 of 54 Qwen3.5 gradients wrong.

  **No measured speedup on the graph that motivated it**: instrumenting the
  pattern shows *0 blocks fused of 34 mul anchors* in that backward, so the
  decline was never the cost. The earlier attribution of ~2× to it was wrong.
  Measured per-step cost is 10.2 s against a ~6.2 s f32 roofline, i.e. the step
  is compute-bound and there is no large dispatch overhead to recover. The verbose
  fusion log now reports anchors alongside fusions, which is how this was caught.

- **MLX could not run a low-precision training graph at all.** Two independent
  bugs, both of which only bite once a graph is differentiated:
  - `grad_with_loss` sized a zero cotangent `Constant` at a hardcoded 4 bytes per
    element while taking the shape — dtype included — from a graph node. On an
    autocast graph that node is bf16, so the constant declared 2 B/elem and held
    4. The consequence landed far away: `cpu_low_precision::promote_to_f32`
    widened those f32 bytes a *second* time as if they were bf16, and MLX refused
    the leaf. Both sites (`zero_missing_wrt` and the Min/Max VJP) now size by
    `Shape::size_bytes`. Pinned by `rlx-autodiff`'s
    `constant_payload_matches_dtype` — a length invariant, which is what catches
    this: zero is zero in every float format, so no value check would have.
  - MLX's `Op::RmsNorm` lowering returned `ops::rms_norm(x, …)`, i.e. **`x`'s**
    shape, ignoring the node's declared output shape. `fuse_rms_norm_reshape`
    folds a following `reshape` into the norm and so changes its RANK, and
    dropping that made a following `gather(axis=0)` index the batch axis; a
    matmul then produced rank 3 and the failure surfaced as "[reshape] Cannot
    reshape array of size 984 into shape (1,3,8)". Pinned by rlx-mlx's
    `gather_after_fused_rmsnorm_reshape` — whose parent node must be *computed*,
    since with a graph `Input` there the fusion declines and the test passes
    vacuously.

  Together these make bf16 training work on MLX (CPU, Metal and MLX now all
  reproduce the f32 run), and they are what let `rlx-kev` re-enable MLX in its
  backend-parity test.

- **Metal returned NaN or zero for most parameter gradients of a Qwen3.5 trunk
  while computing the forward bit-identically to CPU** (32 of 54 on a 4-token
  synthetic trunk). Nothing in a training run said so — the loss looked healthy,
  it just sat on `ln(n_options)`, the chance floor — and it cost a 1529-step
  `rlx-kev` reproduction. Two independent causes, both invisible to per-op tests
  because every op was correct in isolation:
  - **Thunk-level fusions dropped a producer that a later thunk still read.**
    `fuse_gdn_gated_norm` and `fuse_l2_norm` `Nop` out the thunks that produced
    the intermediates they collapse, guarding only against graph *outputs*. Their
    liveness scan stops at the fused window — deliberately, because a full scan
    hits an opaque thunk and conservatively rejects, which would disable the
    fusion on the decode graphs it exists for — so a reader *after* the window
    was invisible. A forward graph has none; a BACKWARD graph re-reads the
    recomputed forward values, which then read an unwritten slot (zeros for
    `silu(gate)`, and 0/0 NaN out of the L2-normalize chain). They now also
    consult a `protected_offsets` set built from IR **use counts**, which is
    complete whatever the thunk scan can analyze.
  - **The conv weight gradient was history-dependent at `N == 1`.** That case
    took a per-group im2col+GEMM path whose answer depended on what was already
    in the arena: a one-shot run agreed with finite differences, but after other
    compiles had dirtied memory the same shapes came back 30–50% off. `N == 1`
    now uses the same two-pass kernels as `N > 1`, which write every element they
    read back; the scratch was already sized for them.

  Pinned by `rlx-qwen35`'s `trunk_backward_device_parity` (whole-trunk, with the
  forward as a control, ~2 s), `rlx-metal`'s `thunk_fusion_second_reader` (the
  fusion hazard in isolation) and `conv_backward_weight_parity` (with a
  finite-difference oracle, and a multi-shape sweep in one process — the history
  is what made the old path fail, and a single-shape test passed right through
  it).

  The criterion is *use count ≤ readers inside the window*, not "use count > 1":
  two readers that are both inside the fused window — one combined matmul feeding
  two narrows, which is what the decode-MLP fusions match — are legitimate, and a
  coarser test declines them (`metal_fused_decode_mlp_parity` catches that).
  Declining when a reader IS outside costs ~2× on a Metal training step, where
  these chains are hot, and nothing on a forward/decode graph, where the values
  have a single reader and the fusion still fires. Fusing anyway while keeping
  the still-read producer was tried and is *wrong*: the fused output can share an
  arena slot with the intermediate being kept (24 of 54 gradients went bad).
- **`AutoMixedBf16` produced NaN on CPU and Metal.** One root cause, six
  symptoms: a low-precision tensor lives in the arena either PACKED at 2 B/elem
  or WIDENED to 4 B, and kernels on both backends read one layout out of the
  other. Every op kind, and the full policy, now match the f32 run on CPU, Metal
  and MLX. The NaN surfaced at an innocent downstream op whose own input was
  intact, which is what made it hard to place.
  - `ArenaWidthPolicy::NativeHalfWidened` (new, now CPU's policy): params and
    integer/bool tensors stay native, f16/bf16 **activations** widen to 4 B —
    what a thunk executor that computes in f32 and reads every activation slot
    as `&[f32]` actually needs. `Native` packed them, contradicting the comments
    that claimed otherwise.
  - `ArenaWidthPolicy::NativeBf16Widened` (new, now Metal's policy): widens bf16
    activations only. Metal has genuine `half` kernels, but `HalfFlag` has no
    BF16 lane, so every Metal kernel reads a bf16 tensor as f32.
  - rlx-cpu `Op::MatMul` picked the packed dequant GEMM from `Param`-ness and
    the declared dtype instead of the real slot width; a packed operand or
    destination now goes through a new `Thunk::SgemmHalf`, and a node with no
    buffer is no longer mistaken for a packed one (that was the bf16 backward's
    "range end index 16 out of range for slice of length 0").
  - rlx-metal `Op::Cast` to/from BF16 decoded 4 B f32 words as packed bf16
    pairs — every odd element right, every even one mantissa junk. It now rounds
    in place via a new `round_bf16_f32` MSL kernel or plain-copies.
  - rlx-metal refuses an MPSGraph plan for a graph with bf16 activations
    (`graph_has_bf16_activation`): it would be handed `MPSDataTypeBFloat16` for
    an f32-wide buffer.
  - rlx-metal's arena derived element counts by dividing the slot size by the
    dtype width, double-counting any widened slot; it now takes them from the
    shape, and exposes `Arena::bytes_per_elem` so a kernel can ask.
- **rlx-metal `encode_cast` wrote zeros** for a same-dtype cast: it bound
  `copy_f32` as if it took two pre-offset buffers when it takes
  `(arena, src_byte_off, dst_byte_off, len)`.

### Changed

- **MLX reports which boundary tensor was mis-sized.** `Array::from_f32_slice`
  checks the shape/length agreement itself, and `build_leaf_for` names the node,
  so a mismatch reads "shape [1,41,2,1] wants 82 elements, got 164 (F32) — while
  building leaf NodeId(903) Constant …" instead of a bare "nelems doesn't match
  shape product" from the C++ shim, which said nothing about which tensor, shape
  or dtype was involved.

### Added

- **tvOS, watchOS and visionOS join the Apple platform set.** `just check-apple`
  now cross-compiles all four Apple OSes, device *and* simulator (8 targets), and
  `just test-apple-sim` **executes** the backend smoke + parity tests on all four
  simulators rather than only cross-compiling them.
  `ios/build-xcframework.sh` emits one `RlxNode.xcframework` with 8 slices, and
  `ios/Demo` builds four apps from one source tree. Backend surface: CPU + Metal +
  CoreML/ANE + MLX on iOS/tvOS/visionOS; **watchOS is CPU/Accelerate only** — it
  has no public Metal API and no CoreML runtime model-compile.
  `--all-archs` adds `arm64_32-apple-watchos` (Apple Watch Series 4–8, which
  cannot load the arm64 slice at all) plus the x86_64 tvOS/watchOS simulators via
  `-Zbuild-std`.
- **MLX builds for every Metal-capable Apple platform**, device and simulator,
  via two tracked patches to the MLX submodule
  (`crates/backends/rlx-mlx-sys/patches/`). Submodule bumped to **v0.32.2**.
- **`rlx-runtime/model-io` feature** (in `default`) gates the `.rlxp` / MLX /
  DDUF readers. A distributed node receives its stage over the wire and never
  opens a model file, so dropping them keeps the Apple/embedded archive a compute
  artifact rather than a file-format library.
- **Per-architecture budget axis in `cargo rlx check`.** Reports what a graph
  *costs* on a backend — arena bytes, liveness reuse, width-policy overhead,
  per-run host-boundary traffic, parameter residency, dispatch count and bytes
  moved — as notes, never errors. Rendered in the CLI and in `--json`.
  Supporting: `arena_width_policy(FusionTarget)` as one source of truth for how
  each architecture stores an activation, and `plan_memory_with_policy` so the
  width policy can be priced with every other planner option held fixed.
- **`ios/build-xcframework.sh` emits MLX's kernel library** to
  `ios/build/metallib/<target>/mlx.metallib` when built with `--apple`. A static
  archive cannot carry it — MLX looks it up next to the executable at the first
  GPU kernel — so the xcframework alone shipped an MLX that linked, reported
  itself available, and failed at first use. Copy it into the `.app` beside the
  executable.
- **First low-precision coverage in `rlx-corpus`.** The corpus was entirely F32
  plus three U8 packed-weight cases; a `low_precision` family now carries
  bf16/f16 elementwise and MLP graphs plus a mixed `bf16 → Cast → f32` case.

### Fixed

- **An over-limit `ElementwiseRegion` emitted a dangling operand.** The fusion
  pass replaces every non-tail region member with a `NodeId(u32::MAX)` sentinel
  on the promise that the tail will emit the region and rewire them; the limit
  check ran *at the tail*, so bailing out left consumers pointing at a node that
  does not exist. Caught only by a debug assertion — a release build handed the
  dangling operand to the backend. The check now runs in the analysis phase,
  before anything is rewritten.
- **`RLX_KEEP_ELEMENTWISE_REGIONS=1` could overrun the CPU region interpreter.**
  It removes the unfuse pass for CPU, MLX, Metal and wgpu at once without
  touching their fusion limits, and CPU asks for `FusionLimits::UNBOUNDED`
  precisely *because* it normally unfuses — so an arbitrarily long chain reached
  an interpreter whose scratch is a fixed `[0f32; 32]`. Limit resolution now
  lives in one shared `resolve_fusion_limits`; it had been two copies of the same
  line, and a clamp added to one never ran on the path the runtime takes.
- **An `F16`/`BF16` boundary was reinterpreted rather than converted.**
  `run(&[(name, &[f32])])` handed f32 bytes to the backend to be read as pairs of
  halves — `[1,2,3,4] + 0` returned `[2.0038757, 4.007843, 0.0, 0.0]` on CPU, with
  no error. Root cause: `cpu_low_precision::promote_to_f32` promoted the *ops* to
  F32 while keeping every `Input`/`Param` at its declared dtype with no `Cast`
  between them, so a BF16 input got a 2 B/elem slot the promoted f32 thunk read
  4 B/elem from. A boundary now keeps low precision only while every consumer
  does too — conditional because `KvAppend` aliases its cache input's buffer and
  strides from the node's dtype.
- **The tvOS *simulator* SDK ships MPS without MPSGraph**, so the Metal backend
  failed to link there. Gated behind `rlx_mps_graph_host`; nothing in the
  MPSGraph module binds a framework symbol at link time, so only the `#[link]`
  directive had to go.
- **`objc` 0.2 does not know tvOS or visionOS exist.** It selects its dispatch
  backend with `cfg(any(target_os = "macos", target_os = "ios"))` and falls
  through to the **GNUstep** runtime otherwise, so every app link failed on
  `_objc_msg_lookup`. `rlx-metal` now supplies both lookup symbols from Apple's
  runtime.
- **MLX stamped every metallib as macOS.** Upstream hardcodes
  `xcrun -sdk macosx metal`, so an iOS/tvOS/visionOS cross-build produced a macOS
  metallib that Metal refuses to load — verified by the platform byte at offset
  `0x0B` (macOS `81`, iOS `82`, tvOS `83`, visionOS `8b`). Fixed by patch 0001.
- **MLX's CPU backend called `system()`/`popen()`**, both unavailable on tvOS and
  visionOS, failing the build at the capability probe itself. Patch 0002.
- **`tolerance_for` in `rlx-corpus` derived an f32 bound for every graph**, asking
  a BF16 case to be more accurate than the format can represent. Now scaled from
  the format's own unit roundoff.
- **`rlx-onnx-conformance` required ORT on platforms it has no binaries for**
  (tvOS/watchOS/visionOS), failing the build script rather than linking.
- **`is_available(Device::Mlx)` answered from a `cfg`, not from the device.** It
  returned `true` for any target linking the native MLX stack, while callers
  treat it as a probe — `fastest_device()` picks the best *available* backend and
  then runs on it. On an iPad with MLX's kernel library missing from the app
  bundle, selection returned MLX and execution then failed with
  `Failed to load the default metallib. library not found`. MLX loads that
  library lazily at the first GPU kernel, so it now probes with a one-element add
  plus `eval` (the add alone is lazy), cached and `catch_unwind`-guarded.
  Verified both ways on hardware: without the metallib MLX reports unavailable
  and selection falls to Metal; with it, MLX is selected and matches CPU.

### Changed

- **iOS and tvOS deployment floors raised to 17.0.** That is where Metal gained
  native `bfloat`, which MLX's kernels use — measured against the Metal compiler,
  `bfloat` is an unknown type at 16.0 on both. visionOS stays at 1.0 (already past
  that line); watchOS is 26.0, because rustup's prebuilt `std` for
  `aarch64-apple-watchos` is itself compiled at 26.0.
- **The Apple xcframework builds with a size-tuned `apple` cargo profile** (LTO,
  one codegen unit, `panic = "abort"` — safe because every `rlx-ffi` entry point
  is `extern "C"` and already aborts on unwind). `opt-level` stays at 3
  deliberately: this is numeric kernel code on a thermally-limited device.

### Validated

- **All four Apple backends executed on hardware** — iPad Pro 11-inch (3rd gen),
  iOS 26.5.2 — each matching the CPU reference at `max|Δ| = 1.49e-8`:

  | backend | vs CPU |
  |---|---|
  | Metal | 1.49e-8 |
  | CoreML / ANE | 1.49e-8 |
  | MLX | 1.49e-8 |
  | wgpu | 1.49e-8 |

  This is the first execution of Metal, MLX and wgpu on an Apple platform other
  than the macOS host. A headless `simctl spawn` exposes no Metal device, so on
  every simulator these three report `unavailable` and skip — meaning the parity
  assertions for them had never actually run. `skipped: []` on device is the line
  that says they finally did.

### Performance

- **Apple release artifact: 11.46 MB → 6.36 MB** for a linked watchOS app
  (−44.5%), measured. `__eh_frame` 709 KB → 38.6 KB, `__gcc_except_tab`
  329 KB → 6.0 KB, `__unwind_info` 124 KB → 14.4 KB. Numerics unchanged:
  ANE-vs-CPU max|Δ| is 1.49e-8 under both the stock release and the `apple`
  profile, and `just test-apple-sim apple` re-runs the simulator suite under the
  shipped settings as a standing precision gate.

### Added

- **Portable polynomial transcendentals in `rlx_cpu::vmath`** — `exp`, `log`,
  `ln_1p`, `softplus`, `tanh`, `sin`, `cos`, `tan`, `atan`, plus
  `vv{sin,cos,tan,atan}f_fast` and the `vvlogf_fast` / `vvlogf_hot` pair that
  `exp` and `tanh` already had. Written in plain Rust rather than intrinsics on
  purpose: with no libm call and no branches they auto-vectorize to **baseline
  SSE2** and to NEON, so they need no `#[target_feature]` and no CPUID gate and
  land on every CPU of the architecture — including the pre-AVX Atoms, plus
  wasm and riscv. Accuracy is pinned by guard tests with bounds set from
  measurement on both architectures: `exp` 2.53e-7 rel, `log` 1.19e-7 rel over
  2^±30 including denormals, `sin`/`cos` 1.19e-7 abs to |x| = 1e6, `tan`
  2.15e-7 rel (finite and accurate at the poles), `atan` 1.19e-7 abs.
- `kernels::swiglu_rows`, `kernels::softmax_rows_poly`, `kernels::layer_norm_rows`.
- **ISA portability gate** — `just check-isa` runs
  `crates/backends/rlx-cpu/tools/isa_portability.py`. `scan` attributes every
  above-baseline instruction to its enclosing symbol and fails on any that is
  not runtime-gated, judged against the baseline of the target the binary was
  built *for* (`sdot` is baseline on `aarch64-apple-darwin`, a finding on
  `aarch64-unknown-linux-gnu`); it also catches the inverse, a
  `-C target-cpu=native` artifact with above-baseline code smeared across
  ordinary symbols where no runtime dispatch can help. `atom` and `arm` build
  and run the suite under `qemu-user` on an emulated Goldmont Atom and
  Cortex-A53.

### Changed

- **`bias_gelu` and `layer_norm_row` dispatch on CPUID at runtime** instead of
  `cfg(target_feature = "avx2")`. One artifact now takes the AVX2 kernel on a
  capable host and the portable loop on a pre-AVX one. Previously a portable
  build silently *lost* those AVX2 kernels, and the only way to get them was a
  `-C target-cpu=native` binary that `SIGILL`s on any pre-Gracemont Atom.
  `x86_dispatch_arms_agree` pins the two arms together (measured divergence
  1.19e-7 and 4.77e-7).
- Nine `Activation` arms — Sigmoid, Softplus, Elu, Mish, LogSigmoid, Sin, Cos,
  Tan, Atan — and `Erf` now use the polynomial arms, matching Tanh/Exp (via
  `vmath::*_hot`) and Gelu/Silu (via `kernels`), which already did.
  `Activation::Log`, log-softmax and the cross-entropy `ln` deliberately stay
  on accurate libm: they feed losses, and that is worth more than throughput.
- `rlx-cpu/build.rs` probes for OpenBLAS on every native Linux target, searches
  `/usr/lib64` (Fedora/RHEL/SUSE), and asks the C compiler whether
  `-lopenblas` resolves before giving up.

### Deprecated

- `rlx_ir::target::has_avx2_likely` — it reads as a capability check and is
  really a build-flag check, which is how `bias_gelu` ended up existing only in
  non-portable artifacts. Dispatch on `is_x86_feature_detected!` instead.

### Fixed

- **x86-64 Linux could not build without OpenBLAS.** `build.rs` probed only
  when `target_arch != "x86_64"`, so x86-64 fell through to an unconditional
  `-lopenblas` and a hard `unable to find library -lopenblas` on the one target
  most likely to lack it — a container, a minimal distro, a cross build — while
  every other target degraded to the portable gemm.
- **A Q4_K accuracy test compared a kernel against itself.**
  `exact_q4k_decode_gemv_beats_the_q8k_arm_on_accuracy` took its reference from
  `gguf_matmul_bt_cached`, which under `cfg(not(rlx_cpu_blas))` forwards to the
  arm being measured, so the error was identically 0 and the assertion could
  never hold. Now an independent f64 oracle, which is stricter in every build.
- **`exec_dispatch`'s AVX2 elementwise path could not inline its intrinsics.**
  The runtime check and the `_mm256_*` calls sat in the same function, which
  carried no `#[target_feature]`, so each intrinsic stayed an out-of-lined
  `call` plus a `vzeroupper` — four per 8 lanes.
- Redundant `unsafe` block in `hsum_i32_avx2` (the crate's only warning).

### Performance

Measured; Apple figures are real hardware, x86 figures are from a
Rosetta-backed container and so are indicative rather than final.

- **SwiGLU 1.63–1.78× (M4) / 2.69–3.16× (x86).** Both `Thunk::FusedSwiGLU`
  paths had the loop inlined byte-identically with a libm `expf` per element on
  *every* architecture — no NEON arm, no AVX2 arm — so the FFN activation of
  every modern transformer was scalar on Apple Silicon too. Now one shared
  `kernels::swiglu_rows`.
- **Row softmax ~2.0× (M4) / ~2.5× (x86)**, of which 1.42–1.51× / 1.19–1.28×
  comes from giving the `max` and `sum` reductions four independent
  accumulators so they vectorize at all.
- **`exp` ~3× on x86** (746–797 vs 247 Melem/s): the pre-AVX2 fallback was a
  libm call per element, which is both slow and unvectorizable.
- **`atan` 3.93× (M4) / 6.95× (x86)**; `sin` 1.25× / 1.63× (lower because its
  argument reduction must run in f64, halving the lane count).
- **LayerNorm +13.2% at h = 32.** Every transformer-block path called
  `layer_norm_row` inside `for r in 0..m`, running the CPUID check once per
  *row*; `layer_norm_rows` checks once per op. 2.4% at h = 128, nil by h = 1024.

### Known issues

- `kernels::avx2_exp8` builds its clamp from `_mm256_max_ps` / `_mm256_min_ps`,
  which return the non-NaN operand, so the AVX2 arm turns a NaN input into
  ~1e-38 where `vmath::exp_poly` now propagates it. Fixing it costs a compare
  and a blend per call in the AVX2 kernel; recorded rather than copied.


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

[Unreleased]: https://github.com/MIT-RLX/rlx/compare/v0.2.19...HEAD
[0.2.19]: https://github.com/MIT-RLX/rlx/compare/v0.2.18...v0.2.19
[0.2.18]: https://github.com/MIT-RLX/rlx/compare/v0.2.17...v0.2.18
[0.2.17]: https://github.com/MIT-RLX/rlx/compare/v0.2.16...v0.2.17
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
