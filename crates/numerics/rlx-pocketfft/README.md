# rlx-pocketfft

`scipy.fft` reproduced **bit-for-bit**, in pure Rust.

SciPy's FFTs (and `torch.fft` on CPU) are [pocketfft](https://github.com/scipy/pocketfft)'s
C++ header. A generic FFT factors the length, orders the butterflies and computes
the twiddles differently, so it lands a few ULP away. `engine` is that header
(`scipy/_lib/pocketfft` at `9367142748fcc9696a1c9e5a99b76ed9897c9daa`, the
submodule SciPy 1.17.1 pins) translated operation for operation: the FFTPACK
passes, Bluestein, the plan choice and the two-table twiddles. No C is compiled,
so it builds for every target, wasm32 included.

## Rounding options

Two build choices decide the last bits, and `Options` selects both:

| option | values | what it models |
|---|---|---|
| `Contraction` | `Clang` (default), `None` | `x ± a*b` fused where clang `-ffp-contract=on` fuses (one scalar expression, left product first), or never (an x86-64 build without FMA) |
| `Twiddles` | `Platform` (default), `AppleSinCos`, `Separate` | each `cos(a)`, `sin(a)` pair as one `__sincos_stret` (clang on Darwin merges them), or as separate libm calls |

Measured on macOS arm64:

* `Clang` + `AppleSinCos` (= the default there, `Options::SCIPY_MACOS_ARM64`)
  matches every `scipy.fft` output in `tests/vectors/scipy_fft.txt`; `Separate`
  twiddles miss 286 of them;
* both contractions match the header compiled with `-ffp-contract=on` / `off`
  for every n = 1..=600 (`tests/vectors/cpp_contract_{on,off}.txt`, from
  `tools/contraction_ref.cpp`); the two builds differ at 595 of those lengths.

`AppleSinCos` exists only on macOS. Elsewhere it falls back to `Separate`
(`Twiddles::exact_here` says so), because no other target ships Apple's libm.

| function | SciPy |
|---|---|
| `rfft`, `irfft` (`*_with` for `Options`) | `scipy.fft.rfft`, `irfft` |
| `fft`, `ifft` (`*_with`) | `scipy.fft.fft`, `ifft` |
| `next_fast_len_real`, `next_fast_len_cmplx` | `scipy.fft.next_fast_len(n, real=True / False)` |
| `engine::{r2c, c2r, c2c, C2cPlan}` | pocketfft's 1-D transforms, generic over `f32` / `f64` |

1-D input, including SciPy's zero-pad / truncate to `n`. `Norm::Backward` is
`norm=None`. Complex values are `[re, im]`.

```rust
use rlx_pocketfft::{Norm, Options, irfft, rfft, rfft_with};

let spec = rfft(&signal, None, Norm::Backward);
let back = irfft(&spec, Some(signal.len()), Norm::Backward);
let x86 = rfft_with(&signal, None, Norm::Backward, Options::NO_FMA);
```

In an rlx graph the same engine is `Graph::fft_algo(x, inverse, norm,
FftAlgo::SCIPY)`: a native row kernel on CPU, and on GPUs an elementwise lowering
of the same plan (`C2cPlan::spec`) that reproduces the CPU bits on Metal, MLX and
wgpu.

## License

MIT OR Apache-2.0. Translated from pocketfft, BSD-3-Clause (`LICENSE-POCKETFFT.md`).
