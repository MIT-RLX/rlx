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
  GELU / SiLU / RoPE, fused matmul-bias-act.
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
  Stockham radix-2 for pow-2; naive DFT for small composite N (≤16);
  Bluestein for other non-pow-2. Host entry shared with GPU fallbacks.

## What's here

- `thunk.rs` (2.5k LOC, the bulk) — Thunk enum + lowering from `Op` +
  both execution paths.
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

GPL-3.0-only.
