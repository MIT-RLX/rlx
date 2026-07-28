# rlx-mlx-sys

Vendored [MLX](https://github.com/ml-explore/mlx) C++ (git submodule at
`vendor/mlx`) plus the `rlx_mlx_shim` C ABI built via CMake + `cc` in
`build.rs`. Consumed by [`rlx-mlx`](../rlx-mlx); not meant for direct use
outside RLX unless you accept the shim API stability policy (none yet).

After clone:

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
```

The submodule is marked `shallow = true` in `.gitmodules` (depth-1 clone of
the pinned MLX commit) so local `.git` stays small. A full MLX history is not
required to build.

## Linux compile times

| Backend | First build (approx.) | Opt-in |
|---------|----------------------|--------|
| **CPU only** (default) | ~5–10 min | always |
| **CUDA** (nvcc kernels) | ~45–90 min | `RLX_MLX_CUDA=1` or `--features cuda` |

**CPU-only is the default** even when CUDA/cuDNN are installed. Auto-building
the CUDA backend added an hour+ to every clean `cargo build` on WSL rigs.

### Faster Linux builds

1. Install system LAPACK headers (avoids bootstrapping OpenBLAS into `OUT_DIR`):

   ```sh
   sudo apt-get install libblas-dev liblapack-dev liblapacke-dev
   ```

2. Use **debug** cargo profile for iteration — `build.rs` maps `cargo build`
   → CMake `Debug` (much faster nvcc when CUDA is enabled).

3. Install **ccache** (MLX enables it automatically when found):

   ```sh
   sudo apt-get install ccache
   ```

4. Pin parallel nvcc jobs if WSL OOMs or thrashes:

   ```sh
   RLX_MLX_JOBS=4 cargo build -p rlx-mlx-sys --features cuda
   ```

5. Pin GPU arch when cross-compiling or CI has no GPU:

   ```sh
   RLX_MLX_CUDA_ARCH=89 RLX_MLX_CUDA=1 cargo build -p rlx-mlx --features cuda
   ```

6. Runtime device selection (after build): `RLX_MLX_DEVICE=cpu|gpu`.

Full guide: [`docs/benchmarks/mlx-linux.md`](../docs/benchmarks/mlx-linux.md).

### macOS / Windows

Requires macOS + Xcode (`xcrun metal`) for Metal kernels. CPU MLX builds on
Linux and Windows; on Linux, `lapacke.h` must be present (`liblapacke-dev`) or
`build.rs` bootstraps OpenBLAS into `OUT_DIR`.

## License

MIT OR Apache-2.0.
