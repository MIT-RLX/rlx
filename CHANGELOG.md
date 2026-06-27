# Changelog

All notable changes to RLX. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
tracks SemVer with the understanding that any `0.x → 0.(x+1)`
bump may carry breaking changes per `0.x`-semver convention.

## [Unreleased]

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
  `crates/rlx-runtime/tests/fused_attention_block_parity.rs`. WebGL and QNN stay
  excluded — neither can lower the resulting `Op::Attention`.
- **Native CUDA + Metal fused-attention kernels.** A `fused_attn_block` kernel
  (CUDA `fused_attn.cu`; Metal MSL in `kernels.rs`) fuses inline NeoX RoPE + softmax
  SDPA over the packed QKV projection — one block/threadgroup per batch·head, score
  matrix in shared/threadgroup memory — collapsing the decompose chain's narrow×3 +
  transpose×3 + rope×2 + attention into a single launch; the QKV / output projections
  stay as GEMMs into appended arena scratch. Each backend keeps the block native when
  the `[seq,seq]` scores fit (`seq ≤ 96` CUDA / `≤ 64` Metal; Metal additionally gates
  to f32 + no-bias) and otherwise falls back to the primitive decomposition. Validated
  vs the CPU reference: CUDA on an RTX 3080 Ti (identity / bias / rope), Metal on Apple
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

[Unreleased]: https://github.com/MIT-RLX/rlx/compare/v0.2.10...HEAD
[0.2.10]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.10
[0.2.9]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.9
[0.2.8]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.8
[0.2.7]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.7
[0.2.6]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.6
[0.2.5]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.5
[0.2.3]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.3

## License

GPL-3.0-only.
