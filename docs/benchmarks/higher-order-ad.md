# Higher-order AD benchmarks

See `crates/core/rlx-autodiff/README.md` for the API and [`CHANGELOG.md`](../../CHANGELOG.md) for the higher-order landing notes.

## Related tests

```sh
# GPU parity (x³, relu, tanh, gelu, silu @ order 3)
cargo test -p rlx-runtime --features cpu,apple --test third_order_gpu_parity
./rig.sh test-third-order-gpu   # CPU + CUDA decompose + wgpu on rig
just test-third-order-gpu       # macOS: Apple backends; Linux: wgpu (+ CUDA decompose when built)
```

## Higher-order AD — status

What is in good shape today:

- 1st/2nd/3rd order scalar `nth_order_grad` (F32/F64), CSE between layers,
  piecewise short-circuit (ReLU/Abs → zero 3rd deriv), closed-form
  `activation_deriv` (no broken Compare→Cast paths).
- GPU parity: CPU vs CUDA / Metal / MLX / wgpu on x³, relu, tanh, gelu, silu
  (`third_order_gpu_parity`).
- Directional GPU parity: `<v, Hv>` on `sum(x²)` + scalar x³ (`directional_nth_gpu_parity`).
- 2nd-order decompose parity (`higher_order_decompose_parity`, 25 cases × GPU backends + unit tests):
  norms (Rms/Layer/Group incl. γ/β), RoPE, Attention (rank-3/4, all `MaskKind`),
  Conv2d (x, w, grouped w, dynamic-batch x/w), MaxPool2d, Cumsum, Gather, FakeQuantize (Tanh STE),
  SoftmaxCrossEntropy, Scan (full + Griewank checkpoints + length 130, limit ≤ 256), Gelu/Silu activation backward.
- F16/BF16: native typed graphs promoted to F32 at GPU compile (mirrors CPU); widen-at-boundary
  I/O via `IoDtypeManifest`; CPU + GPU parity (`higher_order_low_precision_parity`).
- Elementwise fusion on stacked grad graphs (default on; opt out with `RLX_HIGHER_ORDER_NO_FUSE=1`);
  iterative `peel_scalar_expands` removes redundant `Expand(constant)` before fusion
  (consumers include activations, binary ops, `Softmax`, `Reduce`, `Reshape`).
- All training `*Backward` ops used by autodiff decompose to primitives for stacked AD:
  `CumsumBackward`, `GatherBackward`, `SoftmaxCrossEntropyBackward` (compare/where one-hot),
  `FakeQuantizeBackward`, `ScanBackward` / `ScanBackwardXs` (trajectory-cached, length ≤ 256,
  Griewank segment recompute via `forward_body` when `0 < num_checkpoints < length`),
  plus norms, conv, pool, attention, RoPE, activations.
- `MaxPool2dBackward` decompose (static NCHW argmax scatter; unit + 2nd-order GPU parity).
- `Conv2dBackwardInput` decomposes to forward `Op::Conv` when batch is dynamic (spatial static).
- `Conv2dBackwardWeight` grouped im2col: per-group static im2col+matmul with M·K tiling
  (chunk size 16384; accumulates partial dw across im2col row tiles). Dynamic-batch NCHW
  decomposes via `Op::Im2Col` + matmul; `infer_bindings_from_inputs` auto-sets
  `sym::ROWS = N · H_out · W_out` when batch is inferred from runtime inputs.
- `Op::Im2Col` on GPU: host fallback on Metal, wgpu, and CUDA (D2H/H2D).
- wgpu readback: pooled staging + copy fused into final compute submit; tiny (≤16 B) scalar
  outputs use persistent staging with `map_buffer_on_submit` fused into the copy submit
  (avoids post-submit `map_async` on MoltenVK).
- Rig: `./rig.sh test-third-order-gpu` runs CPU + CUDA/WGPU parity (mirrors `just test-third-order-gpu`).
- MLIP helpers, `hvp` / `directional_nth_grad`, pyrlx bindings.

See also `rlx-autodiff/README.md` (API) and
`rlx-bench/examples/bench_nth_order.rs`.
## License

MIT OR Apache-2.0.
