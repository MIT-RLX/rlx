# rlx-mlx-sys

Vendored [MLX](https://github.com/ml-explore/mlx) C++ (git submodule at
`vendor/mlx`) plus the `rlx_mlx_shim` C ABI built via CMake + `cc` in
`build.rs`. Consumed by [`rlx-mlx`](../rlx-mlx); not meant for direct use
outside RLX unless you accept the shim API stability policy (none yet).

After clone:

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
```

Requires macOS + Xcode (`xcrun metal`) for Metal kernels.
