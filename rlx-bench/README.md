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

## Install

```toml
[dependencies]
rlx-bench = "0.1"
```

## Run

```sh
cargo run -p rlx-bench --release --example bench_all
```

## License

GPL-3.0-only.