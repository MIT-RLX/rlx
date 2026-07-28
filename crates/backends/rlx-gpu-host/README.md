# rlx-gpu-host

Backend-agnostic **host-fallback** kernels for RLX GPU backends.

Some ops have no native GPU kernel on a given backend (RNN/SSM family, im2col,
deformable attention, UMAP kNN, …). The universal fallback is: copy the device
arena down to host memory, run the shared `rlx-cpu` implementation, copy the
result back — `D2H → CPU → H2D`.

That staging wrapper used to be copy-pasted into every GPU backend
(`rlx-cuda`, `rlx-rocm`, `rlx-wgpu`, …), differing only in the device-specific
memcpy calls. This crate holds the wrapper **once**, generic over a
[`DeviceArena`] staging trait. Each backend implements `DeviceArena` for its
own stream/buffer handle (a ~15-line adapter in `host_stage.rs`) and either
calls the shared `run_*` functions directly or uses
[`forward_arena_op!`](crate::forward_arena_op) to generate the thin forwarder.

Thin CUDA/ROCm/wgpu adapters live in each crate's `*_host.rs` / `host_ops.rs`
(module facades keep stable call sites). Shared staging also covers Scan/HostOp,
indexing, RNG fill, SPD manifold ops, and GGUF dequant-matmul CPU fallback.

The compute itself is *not* here — it stays in `rlx-cpu`. This crate is purely
the byte-offset staging glue, so a fix to (say) the LSTM fallback lands in one
place instead of N.

This is by definition the **non-perf** path; native kernels stay per-backend.
## License

MIT OR Apache-2.0.
