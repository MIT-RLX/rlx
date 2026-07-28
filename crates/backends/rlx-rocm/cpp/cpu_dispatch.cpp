// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// rlx-rocm HIP-CPU validation TU.
//
// Compiled only when `cargo build --features hip-cpu-validate`.
//
// Kernel sources live in `rlx-gpu-kernels/kernels/`. Rather than duplicate the 358-line
// `launch_<kernel>` wrapper layer here, we just pull in rlx-cuda's
// `cpu_dispatch.cpp` directly. The wrappers it defines compile into
// `rlx_rocm_cpu_dispatch.a` exactly the same way they compile into
// `rlx_cuda_cpu_dispatch.a` — same HIP-CPU semantics, same kernels,
// same FFI surface. Any improvement to rlx-cuda's harness flows here
// automatically.
#include "../../rlx-cuda/cpp/cpu_dispatch.cpp"
