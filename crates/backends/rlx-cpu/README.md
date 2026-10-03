# rlx-cpu

CPU backend for RLX — SIMD kernels, BLAS dispatch, persistent thread
pool, arena executor.

Two execution paths share the same Thunk types:

1. **Direct execution** (single `match` over `&Thunk` at line ~1280
   onward in `thunk.rs`) — hot path, zero closure overhead.
   Performance-critical ops (Attention, FusedAttnBlock, matmul) live here.
2. **Closure-based** (`Box<dyn Fn(*mut u8)>` per thunk; line ~780
   onward in `thunk.rs`) — older, used for ops where dispatch overhead
   doesn't matter, and for some unit tests.

Keep both paths in sync when changing a Thunk variant.

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
- LAPACK bindings (`dgesv`, `dpotrf`, `dgeqrf`, `dgesvd`, `dsyevd`,
  `dtrsm`) for `Op::DenseSolve` and the downstream
  [`rlx-linalg`](https://crates.io/crates/rlx-linalg) crate.
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

- `thunk.rs` (the bulk) — Thunk enum + lowering from `Op` + both execution
  paths. The two giant match functions (`compile_thunks_with_rng`,
  `execute_thunks`) are now compact dispatchers routing to ~200 per-op
  `compile_<op>` / `exec_<op>` fns (the latter `#[inline(always)]`, verified
  perf-neutral against the `tests/bench_execute_hotloop.rs` gate).
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
- `cost.rs` / `config.rs` — model selection + runtime knobs
  (par_threshold, sdpa_seq_threshold, attn_mask_neg_inf, ...).

## Cargo features

| feature              | what it links                                |
|----------------------|----------------------------------------------|
| `blas` *(default)*   | platform CBLAS via FFI                       |
| `blas-accelerate`    | Apple Accelerate                             |
| `blas-mkl`           | Intel MKL                                    |
| `blas-openblas`      | OpenBLAS                                     |

With `--no-default-features`, a portable scalar gemm is linked instead
— slow, but useful on hosts without a system BLAS.

## Install

```toml
[dependencies]
rlx-cpu = "0.1"
```

Or via [`rlx`](https://crates.io/crates/rlx)'s `cpu` feature.

## Build / test

```sh
cargo build -p rlx-cpu --release
cargo test  -p rlx-cpu --release   # 26 tests — mostly parity vs. naive
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

- `Thunk::Attention` carries `mask_kind: MaskKind` (plan #20). Custom
  reads `mask` slice, others synthesize via `apply_synthetic_mask`. Both
  execution paths handle this — keep them in sync.
- `RuntimeConfig::global()` is read once per thunk closure. If you need
  per-call config, pass it through the thunk fields, not via global.
- `cfg.sdpa_seq_threshold` controls the NEON-vs-BLAS attention crossover.
  The NEON path skips dispatch for batch=1 / short seq.
- Thunk-level fusion runs *after* compile_thunks (line ~990) — it
  rewrites Q/K/V → Narrow×3 → [Rope×2] → Attention → out_proj sequences
  into a single FusedAttnBlock. Fragile pattern matching.

## License

MIT OR Apache-2.0.
