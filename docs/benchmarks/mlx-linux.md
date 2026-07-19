# MLX on Linux — build, devices, and benchmarks

RLX ships vendored MLX via `rlx-mlx-sys` on **Linux and Windows** (CPU path)
and **macOS** (Metal). This page covers Linux/WSL: how to compile, pick the
MLX device at runtime, and how `Device::Mlx` compares to `Device::Cpu`.

Primary reference implementation: **WSL2 Ubuntu** on the CUDA rig
(`./rig.sh --wsl`), NVIDIA GPU host. Numbers below are from that environment
unless noted.

## Compile

### CPU (default)

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
sudo apt-get install build-essential cmake liblapacke-dev   # optional: ccache
cargo build -p rlx-mlx
```

| | First build | Incremental |
|--|-------------|-------------|
| **CPU MLX** | ~5–10 min | ~1 min |

`build.rs` sets `MLX_BUILD_CUDA=OFF` even when CUDA/cuDNN are installed.

### CUDA (opt-in)

Compiles upstream MLX's nvcc kernel tree (~400 MB `libmlx.a`). **Not** enabled
by default.

```sh
RLX_MLX_JOBS=4 RLX_MLX_CUDA=1 cargo build -p rlx-mlx --features cuda
# or
cargo build -p rlx-mlx --features mlx-cuda   # via rlx-runtime / rlx crate
```

| | First build |
|--|-------------|
| **CUDA MLX** | ~45–90 min |

Requirements: `cuda-compiler-12-*`, `libcudnn9-dev-cuda-12`, `liblapacke-dev`.

Tips: install `ccache`; use `cargo build` (debug) on Linux so CMake uses
`Debug` for nvcc; pin arch with `RLX_MLX_CUDA_ARCH=89` (Ada/4090).

See also [`rlx-mlx-sys/README.md`](../../rlx-mlx-sys/README.md).

### Rig (WSL)

```sh
./rig.sh --wsl build-mlx cpu      # fast path
./rig.sh --wsl build-mlx cuda     # long nvcc compile
./rig.sh --wsl test-mlx cpu       # 54 basic parity tests
./rig.sh --wsl test-mlx cuda      # needs CUDA build + RLX_MLX_DEVICE=gpu
./rig.sh bench-mlx-devices wsl    # matmul sweep (cpu + cuda legs)
```

## Runtime device selection

RLX always uses `Device::Mlx` in the session API. **Inside** MLX, CPU vs GPU
is selected at process init:

| Env | MLX backend | Requires compile |
|-----|-------------|----------------|
| `RLX_MLX_DEVICE=cpu` (or unset on CPU build) | OpenBLAS CPU | default |
| `RLX_MLX_DEVICE=gpu` | CUDA | `RLX_MLX_CUDA=1` or `--features cuda` |

On macOS, default is **Metal** (no env var needed).

For most NVIDIA GPU workloads on Linux, prefer **`Device::Cuda`** (`rlx-cuda`)
rather than MLX CUDA — it avoids the hour-long MLX nvcc build.

## Benchmark: `rlx-mlx` CPU vs `rlx-cpu`

Matmul L1 pattern, **release** profile, WSL Ubuntu, `RLX_MLX_DEVICE=cpu`:

```sh
./rig.sh --wsl run -- bash -lc '
  cd ~/rlx-workspace-mirror/rlx
  export RLX_MLX_DEVICE=cpu
  cargo run -p rlx-bench --release --example bench_mlx_wgpu --features mlx
'
```

### Median latency (µs)

| Shape | `Device::Cpu` | `Device::Mlx` (CPU) | Faster |
|-------|---------------|---------------------|--------|
| 8×64×64 | **2.7** | 72.0 | rlx-cpu (~27×) |
| 256×256×256 | 7,994 | **3,893** | rlx-mlx (~2×) |
| 512×512×512 | **8,877** | 13,107 | rlx-cpu (~1.5×) |
| 1024×1024×1024 | **16,747** | 21,840 | rlx-cpu (~1.3×) |

### Effective matmul throughput

Computed as `2×M×K×N / median_time` (GFLOP/s):

| Shape | `rlx-cpu` | `rlx-mlx` (CPU) |
|-------|-----------|-----------------|
| 256³ | ~4.2 | ~8.6 |
| 512³ | ~30 | ~21 |
| 1024³ | ~**128** | ~98 |

**Takeaways**

- **Tiny shapes:** MLX loses — graph build + lazy `eval` overhead dominates.
- **256³:** MLX CPU (OpenBLAS inside MLX) wins.
- **512³+:** Native `rlx-cpu` BLAS/NEON path wins on large GEMMs.

## Benchmark: MLX CPU vs MLX CUDA (same Linux build)

Same rig, **debug** profile, explicit `RLX_MLX_DEVICE` (CUDA build required):

| Shape | MLX CPU (µs) | MLX CUDA (µs) | Faster |
|-------|--------------|---------------|--------|
| 8×64×64 | 1,151 | **180** | CUDA |
| 256×256×256 | 6,834 | **1,288** | CUDA |
| 512×512×512 | 15,414 | **9,961** | CUDA |
| 1024×1024×1024 | **23,853** | 58,658 | CPU |

At 1024³ in this debug run, MLX CUDA was slower than MLX CPU (likely debug
kernels + sync overhead). Re-run with `--release` before drawing production
conclusions.

```sh
cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx,mlx-cuda
```

## Benchmark: `rlx-cuda` vs `rlx-mlx` (CUDA compile)

Matmul L1 via `bench_mlx_wgpu`, **release** profile, WSL Ubuntu (NVIDIA GPU).
MLX leg uses a CUDA-enabled `libmlx.a` (`RLX_MLX_CUDA=1` or prior
`./rig.sh --wsl build-mlx cuda`) and **`RLX_MLX_DEVICE=gpu`** at runtime.

```sh
./rig.sh --wsl run -- bash -lc '
  cd ~/rlx-workspace-mirror/rlx
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  export RLX_MLX_CUDA_ARCH=89
  export RLX_MLX_DEVICE=gpu
  RLX_MLX_CUDA=1 cargo build -p rlx-bench --release --example bench_mlx_wgpu --features mlx,cuda
  ./target/release/examples/bench_mlx_wgpu
'
```

### Median latency (µs) — `cuda` vs `mlx` only

| Shape | **`Device::Cuda`** | **`Device::Mlx` (GPU)** | Faster |
|-------|--------------------|-------------------------|--------|
| 8×64×64 | 188 | 176 | ~tie (overhead dominates) |
| 256×256×256 | **495** | 1,308 | rlx-cuda (~2.6×) |
| 512×512×512 | **1,505** | 9,822 | rlx-cuda (~6.5×) |
| 1024×1024×1024 | **4,543** | 53,452 | rlx-cuda (~11.8×) |

### Effective matmul throughput (1024³)

`2×M×K×N / median_time`:

| Backend | GFLOP/s (approx.) |
|---------|-------------------|
| **rlx-cuda** | ~473 |
| rlx-mlx (GPU) | ~40 |
| rlx-cpu (same run) | ~238 |

**Takeaways**

- **Large GEMMs:** `rlx-cuda` is much faster than MLX’s vendored CUDA path at
  512³ and 1024³ on this rig.
- **Tiny shapes:** Both GPU backends lose to graph/eval overhead; numbers are
  not meaningful for picking a backend.
- **Build cost:** MLX CUDA still requires a long nvcc compile (~1h first time);
  `rlx-cuda` links the shared `rlx-gpu-kernels` tree and is the default
  recommendation for NVIDIA Linux workloads (see runtime note above).

## Reference: Apple Silicon (MLX Metal)

Same matmul pattern, **debug** profile, M4 Pro (local host, not WSL):

| Shape | MLX Metal (µs) |
|-------|------------------|
| 256³ | **71** |
| 1024³ | **1,138** |

Metal remains the primary MLX target; Linux CPU/CUDA are secondary paths.

## Related examples

| Example | Purpose |
|---------|---------|
| `bench_mlx_wgpu` | `Device::Cpu` vs `Device::Mlx` (+ optional wgpu/cuda) |
| `bench_mlx_devices` | Per-leg MLX device sweep (`RLX_MLX_DEVICE`, spawn subprocesses on Linux) |
| `bench_all` | Full pattern × backend matrix |

```sh
cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx,mlx-cuda
```

## Verification checklist

```sh
# 1. CPU compile + tests (WSL)
./rig.sh --wsl build-mlx cpu && ./rig.sh --wsl test-mlx cpu

# 2. Confirm CUDA not built by default
grep MLX_BUILD_CUDA ~/rlx-workspace-mirror/rlx/target/debug/build/rlx-mlx-sys-*/out/build/CMakeCache.txt

# 3. Matmul vs rlx-cpu
./rig.sh --wsl run -- bash -lc 'cd ~/rlx-workspace-mirror/rlx && RLX_MLX_DEVICE=cpu cargo run -p rlx-bench --release --example bench_mlx_wgpu --features mlx'
```
## License

GPL-3.0-only.
