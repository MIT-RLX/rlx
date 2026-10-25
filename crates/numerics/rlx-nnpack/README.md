# rlx-nnpack

torch CPU's NNPACK convolution reproduced **bit-for-bit, in pure Rust**.

torch 2.10's CPU build routes float32 convolutions with no dilation and batch ≥ 16
through NNPACK (`_ConvBackend.NnpackSpatial`). For kernels wider than 8 (and smaller
ones where 8×8 tiles would need over 4× as many tiles) NNPACK picks its 16×16 Fourier
algorithm (`ft16x16`), which rounds nothing like a direct sum, so a direct convolution
can't match torch's output. This crate is a translation of that algorithm,
operation for operation, as torch's macOS arm64 wheel compiles it:

* `Maratyszcza/NNPACK@c07e3a0` (the submodule torch 2.10 vendors) and
  `Maratyszcza/psimd@072586a`: the psimd FFT transforms and the NEON complex tuple
  GEMMs, lane by lane;
* an FMA exactly where clang's default `-ffp-contract=on` fuses a one-expression
  `x ± a*b` (the left product first), and where the NEON kernels call `vfmaq`/`vfmsq`;
* flush-to-zero emulated in software (NNPACK runs every stage with denormals
  disabled), signed zeros included.

No C and no SIMD intrinsics: the same bits on every target, wasm included.

```rust
// NCHW input, OIHW kernel; pad is [top, right, bottom, left]; stride 1, no activation.
match rlx_nnpack::conv_output(batch, in_ch, out_ch, in_h, in_w, [0, 0, 0, 0], k_h, k_w, &input, &kernel, &bias) {
    Ok(y) => { /* NCHW output, as torch computes it */ }
    Err(status) => { /* NNPACK's status (rlx_nnpack::status), incl. UNSUPPORTED_ALGORITHM
                       for the ft8x8 / wt8x8 configurations, which are not ported */ }
}
```

**Parity:** `tests/torch_parity.rs` against torch's own `_nnpack_spatial_convolution`
(`parity/export_torch_nnpack.py`, run in a torch 2.10 environment): 1-D Hydra shapes,
2-D kernels up to 16×16, padding, bias, more than 256 input channels, denormal and
signed-zero inputs. During the port it was also checked against the C sources built as
torch builds them: 322 random configurations, 2.07 M outputs, all bit-identical.

Licences: NNPACK (BSD-2, `LICENSE-NNPACK`), psimd (MIT, `LICENSE-PSIMD`).

## License

MIT OR Apache-2.0.
