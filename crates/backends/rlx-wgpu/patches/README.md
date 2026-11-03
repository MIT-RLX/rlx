# wgpu-hal patches (not applied by the build)

## `wgpu-hal-30-metal-math-mode.patch`

wgpu-hal compiles every WGSL→MSL library with `MTLCompileOptions::new()`:
Apple's fast math, with no way to change it from wgpu or naga. A browser
(Dawn) compiles the same WGSL in safe math, so a raw builtin rounds
differently natively and in Chrome on the same GPU. The patch adds
`WGPU_METAL_MATH_MODE=safe|relaxed|fast`; `safe` also selects Metal's precise
transcendental functions. It is shaped for upstream, as a `BackendOptions`
member next to `gl` / `dx12` would be.

It is not vendored, because rlx-wgpu no longer depends on the math mode for
its own kernels: `/`, `sqrt`, `1/sqrt`, `exp`, `log` and `tanh` go through
rlxsl's helpers (`with_numeric_helpers`), which are exact, or within 1–2 ULP,
in any mode. Measured on an M4 Pro (native wgpu, 2^18 inputs, ULPs against
the f64 value rounded to f32):

| builtin | fast (default) | safe + precise |
|---|---|---|
| `x / y` | 26% off, ≤2 ULP | exact |
| `sqrt` | 29% off, ≤1 ULP | exact |
| `exp` | ≤8 ULP | ≤1 ULP |
| `tanh` | ≤46 ULP | ≤2 ULP |
| `pow` | ≤26 ULP | ≤1 ULP |
| `sin`, `cos` | ≤2 ULP | ≤2 ULP |
| `a * b + c` | contracted to `fma` | still contracted |

Safe math cost 37% on a tanh→sigmoid→GELU chain over 4M elements (16.6 →
22.8 ms) and nothing measurable on a 1024² matmul. What it would still buy
rlx: the builtin `sin` / `cos` / `pow` in FFT twiddles, `axial_rope2d` and the
splat rasterizer.

To try it without touching the workspace, apply the patch to a copy of
wgpu-hal 30.0.1 and point a separate build at it:

```sh
cp -R ~/.cargo/registry/src/index.crates.io-*/wgpu-hal-30.0.1 /tmp/wgpu-hal
patch -d /tmp/wgpu-hal -p1 < crates/backends/rlx-wgpu/patches/wgpu-hal-30-metal-math-mode.patch
CARGO_TARGET_DIR=target-mathmode WGPU_METAL_MATH_MODE=safe \
  cargo test -p rlx-wgpu --config 'patch.crates-io.wgpu-hal.path="/tmp/wgpu-hal"'
```

(`--config patch` rewrites `Cargo.lock`; restore it afterwards.)
## License

MIT OR Apache-2.0.
