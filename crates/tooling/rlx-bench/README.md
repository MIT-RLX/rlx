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
- **GPU thermal watchdog + `rlx-gpu` monitor/control** — every timed run
  records `gpu_peak` (peak temp/power) on `BenchResult`, and the `rlx-gpu`
  bin reads live NVIDIA/AMD telemetry and, as root, sets power caps /
  locked clocks / fan. See [GPU monitor & control](#gpu-monitor--control-rlx-gpu).
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

## GPU monitor & control (`rlx-gpu`)

`rlx-gpu` reads live GPU temperature / power / clock / fan telemetry and,
as root, sets the thermal knobs — a hardware tool built on
`rlx_runtime::device_thermal` and the per-vendor `rlx_cuda::nvml` (NVML) /
`rlx_rocm::rsmi` (ROCm-SMI) `libloading` shims. Build with the matching
backend feature to reach hardware; without one it prints an empty inventory.

```sh
# Monitor (read-only, unprivileged)
cargo run -p rlx-bench --bin rlx-gpu --features rocm -- --watch
cargo run -p rlx-bench --bin rlx-gpu --features cuda -- --device cuda --json

# Control (needs root): power cap (both vendors), locked clocks (NVIDIA),
# fan %; each has a --reset-* counterpart. Values are validated against the
# device's reported range.
sudo rlx-gpu --device rocm --index 0 --power-cap 200
sudo rlx-gpu --device cuda --index 0 --lock-clocks 1500
```

Every reading is best-effort: a sensor the board doesn't expose stays blank
rather than reporting a fake value (laptop GPUs omit the junction sensor and
a settable cap; APUs report socket power only). Control returns a typed
`ThermalError` — `PermissionDenied` without root, `Unsupported` where the
board rejects a knob (e.g. ROCm clock-lock, which is not wired — use the
power cap on discrete AMD parts), `OutOfRange` outside the valid envelope.

The timing harness also samples the backing GPU around each timed loop and
attaches the peak as `gpu_peak` to `BenchResult`, so a run whose wall-clock
was silently inflated by thermal throttling is visible.

## License

MIT OR Apache-2.0.