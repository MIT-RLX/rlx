# rlx-gpu-kernels

CUDA C++ kernel sources shared by [`rlx-cuda`](../rlx-cuda) (NVRTC) and
[`rlx-rocm`](../rlx-rocm) (hipRTC). Each `.cu` file is embedded as a
`pub const …: &str` for JIT compilation at runtime — no `nvcc` / `hipcc`
at workspace build time.

HIP and CUDA use the same sources for the kernels in `kernels/` (plain
`__global__` / `__syncthreads` syntax). NVIDIA-only WMMA matmul lives in
`matmul_wmma.cu`; AMD MFMA matmul is behind the `rocm` feature in
`kernels/rocm/matmul_mfma.cu`. Multi-stage pow-2 FFT lives in `fft.cu`
(bit-reverse, inner tile, outer radix-4 / radix-2 stages, plus a dtod
copy kernel for row-region staging).

**Consumers:** depend on this crate and `use rlx_gpu_kernels::BINARY_CU` (or
re-export). Do not `include_str!` across workspace crate boundaries.
