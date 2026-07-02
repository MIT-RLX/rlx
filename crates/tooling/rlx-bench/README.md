# rlx-bench

Uniform benchmark harness for RLX backends + workload patterns. The
canonical answer for "how fast does my graph run on Device X with
PrecisionPolicy Y?"

## What's here

- **`BenchmarkPattern`** — common workload shapes (matmul-only,
  matmul + GELU, full FFN block, attention-only, end-to-end BERT layer).
- **Per-backend timing harness** — measures each pattern across
  Device::Cpu / Metal / Mlx / Cuda / Rocm / Wgpu / Tpu / Fpga
  (whichever are enabled), reports p50 / p95 / GFLOP/s.
- **Examples**:
  - `bench_all` — sweep every (pattern × device × policy) cell.
  - `bench_autodiff` — measure reverse-mode AD overhead per op.
  - `bench_nth_order` — 3rd-order `sum(x³)` vs vector width N; all backends.
  - `bench_fft` — batch × N sweep across backends; set
    `RLX_BENCH_DISPATCH_ONLY=1` on wgpu to skip readback and isolate
    dispatch time.
  - `bench_fft_matrix` — full variant × precision × size × backend FFT/IFFT
    matrix with per-backend CPU-parity checks.
  - `bench_gguf_dequant` — per-scheme dequant + matmul sweep (CPU / Metal / WGPU /
    CUDA when enabled); use `just throttle` before timing runs.
  - `bench_mlx_wgpu` — matmul: `Device::Cpu` vs `Device::Mlx` (set
    `RLX_MLX_DEVICE=cpu` on Linux for MLX CPU path).
  - `bench_mlx_devices` — MLX device legs (Metal / Linux CPU / Linux CUDA);
    see [`docs/benchmarks/mlx-linux.md`](../docs/benchmarks/mlx-linux.md).
  - `bench_fk_fusion` — FKL resize prologue + batch region vs primitives;
    `FK_BENCH_OPS=1` prints fused op counts; `RLX_FK_BATCH_SINGLE_KERNEL=1`
    (CUDA/ROCm/Metal/wgpu) uses one batch-region launch; TPU uses per-slice HLO.
    See [`docs/fk-fusion.md`](../docs/fk-fusion.md).

Cross-platform results: [`docs/benchmarks/higher-order-ad.md`](../docs/benchmarks/higher-order-ad.md).

Linux MLX (compile, `RLX_MLX_DEVICE`, vs `rlx-cpu`): [`docs/benchmarks/mlx-linux.md`](../docs/benchmarks/mlx-linux.md).

## Install

```toml
[dependencies]
rlx-bench = "0.2"
```

## FFT benchmark

```sh
just throttle
cargo run -p rlx-bench --release --example bench_fft --features metal,gpu
RLX_BENCH_DISPATCH_ONLY=1 cargo run -p rlx-bench --release --example bench_fft --features gpu
./scripts/bench_fft_rig.sh   # remote CUDA rig (Windows + WSL)
```

`bench_fft_matrix` is the comprehensive harness: every variant (fft / ifft /
rfft) × precision (f32 / f64 / c64) × size × backend, with a per-backend
CPU-parity check per cell (use `--features metal,gpu,mlx,coreml,native-gpu-fft`
locally, or `cuda,gpu,native-gpu-fft` on a CUDA host). Focused perf A/Bs:
`bench_fft_cpu_parallel` (rayon batch scaling), `bench_fft_cpu_radix4`
(radix-2 vs radix-4), `bench_fft_stft_batch` (per-frame vs batched STFT),
`bench_fft_wgpu_multirow`, `bench_fft_backends` (cross-backend matrix).

## Run

```sh
cargo run -p rlx-bench --release --example bench_all
cargo run -p rlx-bench --release --example bench_nth_order --features metal,mlx,gpu,cuda
RLX_MLX_DEVICE=cpu cargo run -p rlx-bench --release --example bench_mlx_wgpu --features mlx
./rig.sh bench-nth-order both
./rig.sh bench-mlx-devices wsl
```

## License

GPL-3.0-only.