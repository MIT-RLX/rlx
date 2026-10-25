# rlx-cpu

CPU backend for RLX — SIMD kernels, BLAS dispatch, persistent thread
pool, arena executor.

A graph is lowered once into a flat `Vec<Thunk>` (`thunk::compile_thunks`),
each thunk holding its arena offsets and dimensions; `thunk::execute_thunks`
then walks the schedule and calls kernels directly — no per-call dispatch,
lookup or shape math. A Thunk variant has one compile arm
(`thunk/compile_dispatch.rs`) and one execute arm (`thunk/exec_dispatch.rs`).

## Features

- NEON / AVX2 + FMA SIMD kernels for softmax, layer norm, RMS norm,
  GELU / SiLU / RoPE, fused matmul-bias-act, and vForce-style
  `vmath` (`vvexpf` / `vvtanhf` / `vvrecf` / `vvlogf` / `vvsqrtf` /
  `vvrsqrtf`, plus `*_fast` SIMD). Exp/tanh activation paths use SIMD fast
  math by default; set `RLX_VMATH_ACCURATE=1` for Accelerate/libm. The same
  host API is re-exported from Metal / CUDA / ROCm / wgpu / Vulkan / oneAPI /
  MLX / `rlx-gpu-host` for staging; GPU unary kernels cover exp/tanh/recip
  on-device.
- BLAS dispatch via Apple Accelerate (default on macOS) or OpenBLAS /
  MKL via Cargo features.
- LAPACK bindings (`dgesv`/`sgesv`, `dpotrf`, `dgeqrf`, `dgesvd`/`dgesdd`,
  `dsyevd`/`ssyevd`, `dgelsd`, `dtrsm`, `geev`) for `Op::DenseSolve` and the
  downstream [`rlx-linalg`](https://crates.io/crates/rlx-linalg) crate.
  `numpy_blas` calls the exact Accelerate symbols NumPy / SciPy link on macOS,
  for bit-identical ports.
- Integer tensors (I32 / U32 / I64) are stored as integers; `int_storage`
  casts around the kernels that compute in f32.
- `Op::While` runs as a loop (`cond` and `body` compiled once), not unrolled.
- Work-stealing thread pool with `par_for(total, grain, &|off, cnt| …)`
  primitive.
- Reverse-mode AD support: thunks for every backward op
  `rlx_opt::autodiff` emits.
- **FFT** — `Op::Fft` for F32 / F64 / C64 (2N real-block or native C64).
  In-place Cooley-Tukey radix-2 for pow-2 (radix-4 for pure powers of four,
  `RLX_FFT_RADIX4=0` to disable); naive DFT for small composite N (≤16);
  Bluestein for other non-pow-2. Independent batch rows run rayon-parallel
  above a work threshold (`RLX_FFT_CPU_PARALLEL=0` to force serial). Host entry
  shared with GPU fallbacks.

## What's here

- `thunk/` (the bulk) — `types.rs` (the Thunk enum), `compile_dispatch.rs`
  (lowering from `Op`), `exec_dispatch.rs` (execution); both are compact
  dispatchers routing to per-op `compile_<op>` / `exec_<op>` fns in `ops/`
  (the latter `#[inline(always)]`, verified perf-neutral against the
  `tests/bench_execute_hotloop.rs` gate). `scan.rs` / `while_loop.rs` run
  control flow; `waves.rs` is inter-thunk parallelism (`wave-exec`).
- `executor.rs` — alternate non-thunk executor used by old paths and
  some unit tests.
- `kernels.rs` — NEON intrinsics: softmax, layer norm, RMSNorm, matmul
  inner loops.
- `blas.rs` — Accelerate / MKL dispatch. SGEMM variants for different
  alignment regimes.
- `naive.rs` — reference scalar implementations. Used by tests for
  parity and as a fallback.
- `pool.rs` — work-stealing thread pool.
- `arena.rs` — buffer planning interface (the actual byte buffer comes
  from rlx-runtime).
- `autotune.rs` — `Tick`-based search over `RuntimeConfig`. Use
  `rlx_ir::Tick` for sub-ms timing.
- `cost.rs` / `config.rs` — model selection + runtime options
  (par_threshold, sdpa_seq_threshold, attn_mask_neg_inf, ...).
- `spd.rs` / `spd_batched.rs` / `spd_kernels.rs` — SPD-manifold kernels
  (logm/expm/sqrtm, Karcher mean) and batched, differentiable ReEig/LogEig.
- `gguf_matmul.rs` / `dequant_cache.rs` / `int8_gemv.rs` — packed GGUF
  quantized matmul and decode GEMVs.

## Cargo features

| feature              | what it links                                |
|----------------------|----------------------------------------------|
| `blas` *(default)*   | platform CBLAS + LAPACK via FFI: Accelerate on Apple, OpenBLAS elsewhere |
| `blas-mkl`           | Intel MKL (`mkl_rt`, honours `MKL_ROOT`) instead of OpenBLAS |
| `portable-gemm`      | no-BLAS builds: `m ≥ 8` GEMMs via the `gemm` crate's blocked kernels |
| `parity-gemm`        | route `sgemm_auto` through the `gemm` crate, bit-exact with candle |
| `int8-gemv`          | x86-64 AVX2 int8 decode GEMVs for Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1 (int8-rounded activation, as llama.cpp) |
| `wave-exec`          | inter-thunk parallelism: independent thunks run concurrently (`RLX_CPU_WAVES=0` disables) |
| `splat`              | Gaussian-splat executors registered from `rlx-splat` |
| `amx-bnns` / `amx-dense` / `amx-sme` | Apple matrix-unit paths: BNNS W8A8 matmul, Accelerate-AMX measurement harness, SME2 GEMM microkernel (runtime-gated) |

With `--no-default-features`, a portable scalar gemm is linked instead
— slow, but useful on hosts without a system BLAS.

## Install

```toml
[dependencies]
rlx-cpu = "0.2"
```

Or via [`rlx`](https://crates.io/crates/rlx)'s `cpu` feature.

## Build / test

```sh
cargo build -p rlx-cpu --release
cargo test  -p rlx-cpu --release   # mostly parity vs. naive
```

### ISA portability gate

One binary has to run on the AVX-512 server or M4 that built it *and* on
an Atom box or a Raspberry Pi, so every above-baseline instruction must
sit inside a function reached through a runtime CPU-feature check.
`cargo build` enforces none of that and the failure mode is a bare
`Illegal instruction` on hardware you don't own, so
`tools/isa_portability.py` checks both halves:

```sh
just check-isa                                 # emulated Atom (x86-64)
just check-isa arm                             # emulated ARMv8.0 (Pi 3/4)
just check-isa scan target/release/<binary>    # static scan, no Docker needed
```

`scan` attributes every above-baseline instruction to its enclosing symbol
and fails on any that isn't runtime-gated. "Above baseline" is per-target,
not per-arch: `sdot` is baseline on `aarch64-apple-darwin` (apple-m1, v8.5)
and a finding on `aarch64-unknown-linux-gnu` (ARMv8.0-A), so the check is
judged against the baseline of the target the binary was built for.
`--baseline` overrides that to ask the cross-target question ("would this
survive a Cortex-A53?"), and the report says so when you do. It also catches
the inverse problem: a `-C target-cpu=native` artifact smears above-baseline
code across ordinary symbols (`Op::clone` included) where no runtime
dispatch can save it.

`atom` and `arm` run the suite for real. Docker `--platform linux/amd64`
supplies an x86-64 Linux toolchain and `qemu-user-static -cpu Denverton`
emulates a Goldmont Atom that traps AVX (`Snowridge` = Tremont, `Haswell` =
control); `arm` uses `linux/arm64` — fully native on Apple Silicon — with
`-cpu cortex-a53`, which has neither DotProd nor FP16 arithmetic.

## Gotchas

- `Thunk::Attention` carries `mask_kind: MaskKind`. Custom reads the
  `mask` slice; the others synthesize it via `apply_synthetic_mask`.
- `RuntimeConfig::global()` is read once per thunk closure. If you need
  per-call config, pass it through the thunk fields, not via global.
- `cfg.sdpa_seq_threshold` controls the NEON-vs-BLAS attention crossover.
  The NEON path skips dispatch for batch=1 / short seq.
- Thunk-level fusion inside `compile_thunks_with_rng` rewrites
  Q/K/V → Narrow×3 → [Rope×2] → Attention → out_proj sequences into a
  single `FusedAttnBlock`. Fragile pattern matching.

## License

MIT OR Apache-2.0.
